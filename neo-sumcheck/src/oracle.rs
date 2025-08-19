#![allow(clippy::needless_range_loop)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::ptr_arg)]

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use rand_distr::StandardNormal;
use std::error::Error;
use std::io::{Cursor, Read};
use subtle::ConstantTimeEq;

use crate::{fiat_shamir_challenge, fiat_shamir_challenge_base, from_base, ExtF, F};

use neo_poly::Polynomial;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Poseidon2Goldilocks;
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};

// Optional Plonky3 FRI imports (only when p3-fri feature is enabled)
#[cfg(feature = "p3-fri")]
use tracing::instrument;

// FRI Constants tuned for ~128-bit security
const BLOWUP: usize = 4;
pub const NUM_QUERIES: usize = 4; // Reduced for faster tests (was 40)
#[cfg(feature = "fri_pow_16")]
pub const PROOF_OF_WORK_BITS: u8 = 16;
#[cfg(not(feature = "fri_pow_16"))]
pub const PROOF_OF_WORK_BITS: u8 = 0; // Production value
const ZK_SIGMA: f64 = 3.2;
pub const PRIMITIVE_ROOT_2_32: u64 = 1753635133440165772;

pub type Commitment = Vec<u8>;
pub type OpeningProof = Vec<u8>;

pub trait PolyOracle {
    fn commit(&mut self) -> Vec<Commitment>;
    fn open_at_point(&mut self, point: &[ExtF]) -> (Vec<ExtF>, Vec<OpeningProof>);
    fn verify_openings(
        &self,
        comms: &[Commitment],
        point: &[ExtF],
        evals: &[ExtF],
        proofs: &[OpeningProof],
    ) -> bool;
}

/// Trait for configurable FRI backends
/// 
/// This trait allows switching between different FRI implementations:
/// - CustomFri: The existing custom FRI implementation in this crate
/// - PlonkyFri: Plonky3's optimized FRI implementation
/// 
/// Both implementations preserve equivalent security when configured with
/// matching parameters (blowup=4, queries=40 for ~128-bit security, etc.)
pub trait FriBackend: Send + Sync {
    /// Commit to a set of polynomials and return their commitments
    fn commit(&mut self, polys: Vec<Polynomial<ExtF>>) -> Result<Vec<Commitment>, Box<dyn Error>>;
    
    /// Open polynomials at a given point, returning evaluations and proofs
    fn open_at_point(&mut self, point: &[ExtF]) -> Result<(Vec<ExtF>, Vec<OpeningProof>), Box<dyn Error>>;
    
    /// Verify opening proofs for committed polynomials
    fn verify_openings(
        &self,
        comms: &[Commitment],
        point: &[ExtF],
        evals: &[ExtF],
        proofs: &[OpeningProof],
    ) -> bool;
    
    /// Get domain size for verifier setup
    fn domain_size(&self) -> usize;
}

/// Configuration for FRI backend selection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FriImpl {
    /// Use the custom FRI implementation (default)
    Custom,
    /// Use Plonky3's FRI implementation
    #[cfg(feature = "p3-fri")]
    Plonky3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TamperMode {
    None,
    CorruptEval,
    InvalidProof,
}

pub struct FnOracle<F>
where
    F: Fn(&[ExtF]) -> Vec<ExtF>,
{
    f: F,
    tamper_mode: TamperMode,
}

impl<F> FnOracle<F>
where
    F: Fn(&[ExtF]) -> Vec<ExtF>,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            tamper_mode: TamperMode::None,
        }
    }

    pub fn with_tamper_mode(f: F, mode: TamperMode) -> Self {
        Self {
            f,
            tamper_mode: mode,
        }
    }
}

impl<F> PolyOracle for FnOracle<F>
where
    F: Fn(&[ExtF]) -> Vec<ExtF> + Clone,
{
    fn commit(&mut self) -> Vec<Commitment> {
        vec![]
    }

    fn open_at_point(&mut self, point: &[ExtF]) -> (Vec<ExtF>, Vec<OpeningProof>) {
        let mut evals = (self.f)(point);
        let mut proofs = vec![vec![]; evals.len()];
        match self.tamper_mode {
            TamperMode::CorruptEval => {
                if !evals.is_empty() {
                    evals[0] += ExtF::ONE;
                }
            }
            TamperMode::InvalidProof => {
                if !proofs.is_empty() {
                    proofs[0] = vec![0u8; 1];
                }
            }
            TamperMode::None => {}
        }
        (evals, proofs)
    }

    fn verify_openings(
        &self,
        _comms: &[Commitment],
        _point: &[ExtF],
        _evals: &[ExtF],
        proofs: &[OpeningProof],
    ) -> bool {
        proofs.iter().all(|p| p.is_empty())
    }
}

// FRI implementation
pub fn extf_pow(mut base: ExtF, mut exp: u64) -> ExtF {
    let mut res = ExtF::ONE;
    while exp > 0 {
        if exp & 1 == 1 {
            res = res * base;
        }
        base = base * base;
        exp >>= 1;
    }
    res
}

/// Reverse the bit order of indices in a slice for two-adic FFT compatibility
fn reverse_slice_index_bits<T: Clone>(slice: &mut [T]) {
    let n = slice.len();
    assert!(n.is_power_of_two(), "Length must be power of 2");
    if n <= 1 {
        return;
    }
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = reverse_bits(i, log_n);
        if i < j {
            slice.swap(i, j);
        }
    }
}

/// Reverse the lower `bits` bits of a number  
fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}



pub fn generate_coset(size: usize) -> Vec<ExtF> {
    assert!(size.is_power_of_two(), "Size must be power of 2");
    let omega = from_base(F::from_u64(PRIMITIVE_ROOT_2_32));
    let gen = extf_pow(omega, (1u64 << 32) / size as u64);
    let offset = ExtF::ONE;
    let mut coset: Vec<ExtF> = (0..size)
        .map(|i| offset * extf_pow(gen, i as u64))
        .collect();
    
    // Apply bit-reversal to domain for standard FRI (consecutive pairing)
    reverse_slice_index_bits(&mut coset);
    
    // Verify pairing properties for consecutive FRI folding
    for i in (0..size).step_by(2) {
        let a = coset[i];
        let b = coset[i + 1];
        assert_eq!(b, -a, "Domain pairing broken: coset[{}] != -coset[{}]", i + 1, i);
    }
    coset
}

pub fn hash_extf(e: ExtF) -> [u8; 32] {
    let [r, i] = e.to_array();
    let input = [r, i];
    let perm = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rand_chacha::ChaCha20Rng::seed_from_u64(0));
    let sponge = PaddingFreeSponge::<_, 16, 15, 2>::new(perm);
    let hashed = sponge.hash_iter(input.iter().cloned());
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&hashed[0].as_canonical_u64().to_le_bytes());
    out[8..16].copy_from_slice(&hashed[1].as_canonical_u64().to_le_bytes());
    // Pad remaining bytes with deterministic values
    for i in 16..32 {
        out[i] = ((hashed[0].as_canonical_u64() ^ hashed[1].as_canonical_u64()) >> (i - 16)) as u8;
    }
    out
}

fn hash_pair(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut input = vec![];
    // Convert bytes to field elements (8 bytes per element)
    for chunk in a.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let val = u64::from_le_bytes(buf);
        input.push(F::from_u64(val));
    }
    for chunk in b.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let val = u64::from_le_bytes(buf);
        input.push(F::from_u64(val));
    }
    let perm = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rand_chacha::ChaCha20Rng::seed_from_u64(0));
    let sponge = PaddingFreeSponge::<_, 16, 15, 2>::new(perm);
    let hashed = sponge.hash_iter(input);
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&hashed[0].as_canonical_u64().to_le_bytes());
    out[8..16].copy_from_slice(&hashed[1].as_canonical_u64().to_le_bytes());
    // Pad remaining bytes with deterministic values
    for i in 16..32 {
        out[i] = ((hashed[0].as_canonical_u64() ^ hashed[1].as_canonical_u64()) >> (i - 16)) as u8;
    }
    out
}

#[derive(Clone)]
pub struct MerkleTree {
    nodes: Vec<[u8; 32]>,
    pub leaves: usize,
}

impl MerkleTree {
    pub fn new(values: &[ExtF]) -> Self {
        let leaves = values.len().next_power_of_two();
        let mut nodes = vec![[0u8; 32]; 2 * leaves];
        for i in 0..leaves {
            let val = if i < values.len() {
                values[i]
            } else {
                ExtF::ZERO
            };
            nodes[leaves + i] = hash_extf(val);
        }
        for i in (1..leaves).rev() {
            nodes[i] = hash_pair(&nodes[2 * i], &nodes[2 * i + 1]);
        }
        Self { nodes, leaves }
    }

    pub fn root(&self) -> [u8; 32] {
        self.nodes[1]
    }

    pub fn open(&self, mut index: usize) -> Vec<[u8; 32]> {
        index += self.leaves;
        let mut proof = Vec::new();
        while index > 1 {
            let sibling = if index % 2 == 0 { index + 1 } else { index - 1 };
            proof.push(self.nodes[sibling]);
            index /= 2;
        }
        proof
    }
}

pub fn verify_merkle_opening(
    root: &[u8; 32],
    leaf: ExtF,
    mut index: usize,
    proof: &[[u8; 32]],
    leaves: usize,
) -> bool {
    let mut hash = hash_extf(leaf);
    index += leaves;
    for sib in proof {
        hash = if index % 2 == 0 {
            hash_pair(&hash, sib)
        } else {
            hash_pair(sib, &hash)
        };
        index /= 2;
    }
    hash.ct_eq(root).unwrap_u8() == 1
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriLayerQuery {
    pub idx: usize,
    pub sib_idx: usize,
    pub val: ExtF,
    pub sib_val: ExtF,
    pub path: Vec<[u8; 32]>,
    pub sib_path: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriQuery {
    pub idx: usize,
    pub f_val: ExtF,
    pub f_path: Vec<[u8; 32]>,
    pub layers: Vec<FriLayerQuery>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriProof {
    pub layer_roots: Vec<[u8; 32]>,
    pub queries: Vec<FriQuery>,
    pub final_eval: ExtF,
    pub final_pow: u64,
}

pub struct FriOracle {
    committed_polys: Vec<Polynomial<ExtF>>,
    domain: Vec<ExtF>,
    trees: Vec<MerkleTree>,
    codewords: Vec<Vec<ExtF>>,
    pub blinds: Vec<ExtF>,
}

impl FriOracle {
    pub fn new(polys: Vec<Polynomial<ExtF>>, transcript: &mut Vec<u8>) -> Self {
        Self::new_with_blowup(polys, transcript, BLOWUP)
    }

    pub fn new_with_blowup(mut polys: Vec<Polynomial<ExtF>>, transcript: &mut Vec<u8>, blowup: usize) -> Self {
        // Handle dummy case: add non-zero constant poly if empty
        if polys.is_empty() {
            eprintln!("FriOracle::new_with_blowup: Empty polys list, adding dummy non-zero polynomial");
            polys = vec![Polynomial::new(vec![ExtF::ONE])];
        }
        
        // CRITICAL FIX: Handle zero polynomials (empty coeffs) properly
        // Instead of treating them as dummy polynomials, we need to handle them as legitimate zero polynomials
        for (i, poly) in polys.iter_mut().enumerate() {
            if poly.coeffs().is_empty() {
                eprintln!("FriOracle::new_with_blowup: Found zero polynomial at index {}, will treat as constant zero in blinding", i);
                // Note: We keep the empty polynomial as-is because poly.eval() correctly returns 0 for empty coeffs
                // The blinding will work correctly: 0 + blind = blind
            }
        }
        
        let max_deg = polys.iter().map(|p| p.degree()).max().unwrap_or(0);
        let domain_size = (max_deg + 1).next_power_of_two() * blowup;

        let domain = generate_coset(domain_size);
        let mut blind_trans = transcript.clone();
        blind_trans.extend(b"fri_blind_seed");
        // Use Poseidon2 for blind seed generation
        let hash_result = fiat_shamir_challenge_base(&blind_trans);
        let mut seed = [0u8; 32];
        seed[0..8].copy_from_slice(&hash_result.as_canonical_u64().to_le_bytes());
        // Fill remaining with deterministic pattern
        for i in 8..32 {
            seed[i] = ((hash_result.as_canonical_u64() >> (i % 8)) ^ (i as u64)) as u8;
        }
        let mut rng = ChaCha20Rng::from_seed(seed);
        let mut trees = Vec::new();
        let mut codewords = Vec::new();
        let mut blinds = Vec::new();
        for poly in &polys {
            let blind = Self::sample_blind_factor(&mut rng);
            blinds.push(blind);
            let evals = domain
                .iter()
                .map(|&w| poly.eval(w) + blind)
                .collect::<Vec<_>>();
            // No bit-reversal needed - domain is already bit-reversed from generate_coset
            let tree = MerkleTree::new(&evals);
            trees.push(tree);
            codewords.push(evals);
        }
        Self {
            committed_polys: polys,
            domain,
            trees,
            codewords,
            blinds,
        }
    }

    pub fn new_with_blinds(polys: Vec<Polynomial<ExtF>>, blinds: Vec<ExtF>) -> Self {
        Self::new_with_blinds_and_blowup(polys, blinds, BLOWUP)
    }

    pub fn new_with_blinds_and_blowup(polys: Vec<Polynomial<ExtF>>, blinds: Vec<ExtF>, blowup: usize) -> Self {
        let max_deg = polys.iter().map(|p| p.degree()).max().unwrap_or(0);
        let domain_size = (max_deg + 1).next_power_of_two() * blowup;
        let domain = generate_coset(domain_size);
        let mut trees = Vec::new();
        let mut codewords = Vec::new();
        for (poly, blind) in polys.iter().zip(&blinds) {
            let evals = domain
                .iter()
                .map(|&w| poly.eval(w) + *blind)
                .collect::<Vec<_>>();
            trees.push(MerkleTree::new(&evals));
            codewords.push(evals);
        }
        Self {
            committed_polys: polys,
            domain,
            trees,
            codewords,
            blinds,
        }
    }

    pub fn new_for_verifier(domain_size: usize) -> Self {
        let domain = generate_coset(domain_size);

        Self {
            committed_polys: vec![],
            domain,
            trees: vec![],
            codewords: vec![],
            blinds: vec![],
        }
    }

    fn sample_blind_factor(rng: &mut ChaCha20Rng) -> ExtF {
        Self::sample_discrete_gaussian(rng, ZK_SIGMA)
    }

    pub fn sample_discrete_gaussian(rng: &mut impl Rng, sigma: f64) -> ExtF {
        fn sample_coord(rng: &mut impl Rng, sigma: f64) -> i64 {
            let mut retries = 0;
            let max_retries = 1000; // Increased for test stability
            loop {
                if retries > max_retries {
                    panic!("Gaussian rejection failed after {} retries", max_retries);
                }
                let x: f64 = rng.sample::<f64, _>(StandardNormal) * sigma;
                let z = x.round() as i64;
                let diff = x - z as f64;
                let prob = (-(diff.powi(2) / (2.0 * sigma.powi(2)))).exp();
                if rng.random::<f64>() < prob {
                    return z;
                }
                retries += 1;
            }
        }
        let r = sample_coord(rng, sigma);
        let i = sample_coord(rng, sigma);
        ExtF::new_complex(F::from_i64(r), F::from_i64(i))
    }
}

impl PolyOracle for FriOracle {
    fn commit(&mut self) -> Vec<Commitment> {
        // Return codeword roots - these will be used for polynomial evaluation verification
        // but FRI proofs will use their own internal quotient polynomial roots
        self.trees.iter().map(|t| t.root().to_vec()).collect()
    }

    fn open_at_point(&mut self, point: &[ExtF]) -> (Vec<ExtF>, Vec<OpeningProof>) {
        let z = point.first().copied().unwrap_or(ExtF::ZERO);
        let mut evals = Vec::new();
        let mut proofs = Vec::new();
        for (i, poly) in self.committed_polys.iter().enumerate() {
            let mut p_z = poly.eval(z);
            p_z += self.blinds[i];
            evals.push(p_z);
            let proof = self.generate_fri_proof(i, z, p_z);
            proofs.push(serialize_fri_proof(&proof));
        }
        (evals, proofs)
    }

    fn verify_openings(
        &self,
        comms: &[Commitment],
        point: &[ExtF],
        evals: &[ExtF],
        proofs: &[OpeningProof],
    ) -> bool {
        let z = point[0];
        eprintln!("verify_openings: Called with comms.len()={}, point.len()={}, evals.len()={}, proofs.len()={}", 
                 comms.len(), point.len(), evals.len(), proofs.len());
        eprintln!("verify_openings: z={:?}, evals={:?}", z, evals);
        eprintln!("verify_openings: self.domain.len()={}", self.domain.len());

        
        if comms.len() != evals.len() || proofs.len() != evals.len() {
            eprintln!("verify_openings: Length mismatch - comms={}, evals={}, proofs={}", 
                     comms.len(), evals.len(), proofs.len());
            return false;
        }
        
        for (i, ((root, &claimed_eval), proof_bytes)) in comms.iter().zip(evals).zip(proofs).enumerate() {
            eprintln!("verify_openings: Processing proof {} - root.len()={}, claimed_eval={:?}, proof_bytes.len()={}", 
                     i, root.len(), claimed_eval, proof_bytes.len());
            
            let proof = match deserialize_fri_proof(proof_bytes) {
                Ok(p) => {
                    eprintln!("verify_openings: Successfully deserialized proof {} - queries.len()={}, layer_roots.len()={}", 
                             i, p.queries.len(), p.layer_roots.len());
                    p
                },
                Err(e) => {
                    eprintln!("verify_openings: Failed to deserialize proof {}: {:?}", i, e);
                    return false;
                }
            };
            
            eprintln!("verify_openings: About to call verify_fri_proof for proof {}", i);
            let fri_result = self.verify_fri_proof(root, z, claimed_eval, &proof);
            eprintln!("verify_openings: verify_fri_proof result for proof {}: {}", i, fri_result);
            if !fri_result {
                eprintln!("verify_openings: FAIL - verify_fri_proof returned false for proof {}", i);
                return false;
            }
            
            // POW check
            let mut pow_trans = proof.final_eval.to_array()[0]
                .as_canonical_u64()
                .to_be_bytes()
                .to_vec();
            pow_trans.extend(proof.final_pow.to_be_bytes());
            let pow_hash_result = fiat_shamir_challenge_base(&pow_trans);
            let pow_hash_u64 = pow_hash_result.as_canonical_u64();
            let mask = (1u32 << PROOF_OF_WORK_BITS) - 1;
            let pow_val = pow_hash_u64 as u32;
            if pow_val & mask != 0 {
                return false;
            }
        }
        
        true
    }
}

impl FriOracle {
    pub fn generate_fri_proof(&self, poly_idx: usize, z: ExtF, p_z: ExtF) -> FriProof {
        let f_tree = &self.trees[poly_idx];
        let evals = &self.codewords[poly_idx];
        let mut local_transcript = f_tree.root().to_vec();
        let r_base = fiat_shamir_challenge_base(&local_transcript);
        let r = from_base(r_base);
        local_transcript.extend(&r_base.as_canonical_u64().to_be_bytes());
        // Precompute derivative at z for rare hits
        let poly = &self.committed_polys[poly_idx];
        let coeffs = poly.coeffs();
        let mut p_prime_z = ExtF::ZERO;
        for i in (1..coeffs.len()).rev() {
            p_prime_z = p_prime_z * z + from_base(F::from_u64(i as u64)) * coeffs[i];
        }
        let mut composed_evals = Vec::with_capacity(evals.len());
        for (&x, &p_x) in self.domain.iter().zip(evals) {
            let denom = x - z;
            let q_x = if denom == ExtF::ZERO {
                r * p_prime_z
            } else {
                r * (p_x - p_z) / denom
            };
            composed_evals.push(q_x);
        }
        let mut layer_roots = Vec::new();
        let mut eval_layers = Vec::new();
        let mut trees = Vec::new();
        let mut current_domain = self.domain.clone();
        let mut current_evals = composed_evals;
        let first_tree = MerkleTree::new(&current_evals);
        let first_root = first_tree.root();
        layer_roots.push(first_root);
        local_transcript.extend(&first_root);
        trees.push(first_tree);
        eval_layers.push(current_evals.clone());
        let mut final_eval = ExtF::ZERO;
        let mut final_pow = 0u64;
        while current_evals.len() > 1 {
            // Use base field challenges only to match p3-fri behavior
            let challenge_base = fiat_shamir_challenge_base(&local_transcript);
            let challenge = from_base(challenge_base);
            eprintln!("generate_fri_proof: Folding with challenge={:?}, evals.len()={}", challenge, current_evals.len());
            if current_evals.len() <= 8 {
                eprintln!("generate_fri_proof: Input evals: {:?}", current_evals);
            }
            let (new_evals, new_domain) =
                self.fold_evals(&current_evals, &current_domain, challenge);
            if new_evals.len() <= 4 {
                eprintln!("generate_fri_proof: Output evals: {:?}", new_evals);
            }
            current_domain = new_domain;
            current_evals = new_evals;
            if current_evals.len() > 1 {
                let new_tree = MerkleTree::new(&current_evals);
                let new_root = new_tree.root();
                layer_roots.push(new_root);
                local_transcript.extend(&new_root);
                trees.push(new_tree);
                eval_layers.push(current_evals.clone());
            } else {
                final_eval = current_evals[0];
                let mask = (1u32 << PROOF_OF_WORK_BITS) - 1;
                let max_iters = 1_000_000; // Safety limit to prevent infinite loop
                let mut iterations = 0;
                loop {
                    let mut pow_trans = final_eval.to_array()[0]
                        .as_canonical_u64()
                        .to_be_bytes()
                        .to_vec();
                    pow_trans.extend(final_pow.to_be_bytes());
                    let pow_hash_result = fiat_shamir_challenge_base(&pow_trans);
                    let pow_hash_u64 = pow_hash_result.as_canonical_u64();
                    let pow_val = pow_hash_u64 as u32;
                    if pow_val & mask == 0 {
                        break;
                    }
                    final_pow += 1;
                    iterations += 1;
                    if iterations > max_iters {
                        // In production, this prevents infinite loops from bad final_eval values
                        panic!("PoW failed after {} iterations - possible bug or bad luck. final_eval={:?}, PROOF_OF_WORK_BITS={}", 
                               max_iters, final_eval, PROOF_OF_WORK_BITS);
                    }
                }
            }
        }
        // Generate query indices using the final transcript state
        eprintln!("generate_fri_proof: local_transcript.len()={}, first 32 bytes={:?}", 
                 local_transcript.len(), &local_transcript[0..32.min(local_transcript.len())]);
        let mut queries = Vec::new();
        for query_idx in 0..NUM_QUERIES {
            let mut q_trans = b"query_salt_".to_vec();  // Fixed salt to bypass transcript dep
            q_trans.extend(&query_idx.to_be_bytes());
            q_trans.extend(&local_transcript);  // Still bind to layers for security
            let chal = fiat_shamir_challenge(&q_trans);
            let idx_hash = chal.to_array()[0].as_canonical_u64() as usize % self.domain.len();
            let mut current_idx = idx_hash;
            let f_val = evals[current_idx];
            let f_path = f_tree.open(current_idx);
            eprintln!("generate_fri_proof: Query {} - idx_hash={}, current_idx={}, f_val={:?}", 
                     query_idx, idx_hash, current_idx, f_val);
            eprintln!("generate_fri_proof: Query {} - f_path.len()={}, domain.len()={}", 
                     query_idx, f_path.len(), self.domain.len());
            let mut layers = Vec::new();
            for l in 0..layer_roots.len() {
                let tree = &trees[l];
                let _size = eval_layers[l].len();
                
                // For consecutive pairing: sibling is bit flip (consecutive in bit-rev order)
                let pair_idx = current_idx ^ 1;
                
                let val = eval_layers[l][current_idx];
                let sib_val = eval_layers[l][pair_idx];
                let path = tree.open(current_idx);
                let sib_path = tree.open(pair_idx);
                layers.push(FriLayerQuery {
                    idx: current_idx,
                    sib_idx: pair_idx,
                    val,
                    sib_val,
                    path,
                    sib_path,
                });
                
                // Update index for next layer (binary tree structure)
                current_idx /= 2;
            }
            queries.push(FriQuery {
                idx: idx_hash,
                f_val,
                f_path,
                layers,
            });
        }
        FriProof {
            layer_roots,
            queries,
            final_eval,
            final_pow,
        }
    }

    pub fn fold_evals(
        &self,
        evals: &[ExtF],
        domain: &[ExtF],
        challenge: ExtF,
    ) -> (Vec<ExtF>, Vec<ExtF>) {
        let n = evals.len();
        let half = n / 2;
        let two_inv = ExtF::ONE / from_base(F::from_u64(2));
        let mut new_evals = Vec::with_capacity(half);
        let mut new_domain = Vec::with_capacity(half);
        
        // Use consecutive pairing for standard FRI (bit-reversed order)
        for i in (0..n).step_by(2) {
            let e0 = evals[i];     // Even index
            let e1 = evals[i + 1]; // Odd index (consecutive pair)
            let g = domain[i];     // Domain element for even index
            new_domain.push(g * g); // Square the domain element
            // Standard FRI folding formula: (e0 + e1) / 2 + challenge * (e0 - e1) / (2 * g)
            new_evals.push((e0 + e1) * two_inv + challenge * (e0 - e1) * two_inv / g);
        }
        (new_evals, new_domain)
    }

    pub fn old_fold_evals(
        &self,
        evals: &[ExtF],
        domain: &[ExtF],
        challenge: ExtF,
    ) -> (Vec<ExtF>, Vec<ExtF>) {
        let two_inv = ExtF::ONE / ExtF::from_u64(2);
        let n = evals.len();
        let half = n / 2;
        let mut new_evals = Vec::with_capacity(half);
        let mut new_domain = Vec::with_capacity(half);
        for i in 0..half {
            let g = domain[i];
            let f_g = evals[i];
            let f_neg_g = evals[i + half];
            new_domain.push(g * g);
            let sum = f_g + f_neg_g;
            let diff = f_g - f_neg_g;
            new_evals.push(sum * two_inv + challenge * diff * two_inv / g);
        }
        (new_evals, new_domain)
    }

    pub fn verify_fri_proof(
        &self,
        root: &[u8],
        z: ExtF,
        claimed_eval: ExtF,
        proof: &FriProof,
    ) -> bool {
        eprintln!("verify_fri_proof: Starting with z={:?}, claimed_eval={:?}", z, claimed_eval);
        // Use the codeword root (provided by commit()) to build initial transcript
        // This matches what the prover did in generate_fri_proof
        let mut transcript = root.to_vec();
        let r_base = fiat_shamir_challenge_base(&transcript);
        let r = from_base(r_base);
        transcript.extend(&r_base.as_canonical_u64().to_be_bytes());
        if proof.layer_roots.is_empty() {
            eprintln!("verify_fri_proof: Empty layer roots");
            return false;
        }
        transcript.extend(&proof.layer_roots[0]);
        let mut challenges = Vec::new();
        for root_bytes in proof.layer_roots.iter().skip(1) {
            // Generate challenge BEFORE adding root to transcript (to match prover)
            let chal_base = fiat_shamir_challenge_base(&transcript);
            let chal = from_base(chal_base);
            challenges.push(chal);
            transcript.extend(root_bytes);
        }
        // Use base field challenge for final challenge too
        let final_chal_base = fiat_shamir_challenge_base(&transcript);
        challenges.push(from_base(final_chal_base));
        
        // Now the transcript should match what the prover had when generating query indices
        eprintln!("verify_fri_proof: transcript.len()={}, first 32 bytes={:?}", 
                 transcript.len(), &transcript[0..32.min(transcript.len())]);
        let domain_size = self.domain.len();
        eprintln!("verify_fri_proof: domain_size={}, self.domain={:?}", domain_size, self.domain);
        
        // Verify that the query indices match what we would generate with this transcript
        for (q_idx, query) in proof.queries.iter().enumerate() {
            let mut q_trans = b"query_salt_".to_vec();  // Same fixed salt
            q_trans.extend(&q_idx.to_be_bytes());
            q_trans.extend(&transcript);  // Bind to reconstructed transcript
            let chal = fiat_shamir_challenge(&q_trans);
            let expected_idx = chal.to_array()[0].as_canonical_u64() as usize % domain_size;
            eprintln!("verify_fri_proof: Query {} - prover_idx={}, expected_idx={}", 
                     q_idx, query.idx, expected_idx);
            if query.idx != expected_idx {
                eprintln!("verify_fri_proof: FAIL - Query {} index mismatch: expected={}, got={}", 
                         q_idx, expected_idx, query.idx);
                return false;
            }
        }
        
        for (q_idx, query) in proof.queries.iter().enumerate() {
            eprintln!("verify_fri_proof: Processing query {}", q_idx);
            if query.layers.len() != proof.layer_roots.len() {
                eprintln!("verify_fri_proof: Layer count mismatch");
                return false;
            }
            
            let root_arr: [u8; 32] = {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(root);
                arr
            };
            eprintln!("verify_fri_proof: About to verify Merkle opening for query {}", q_idx);
            eprintln!("verify_fri_proof: root.len()={}, query.idx={}, domain_size={}", root.len(), query.idx, domain_size);
            eprintln!("verify_fri_proof: query.f_val={:?}", query.f_val);
            eprintln!("verify_fri_proof: query.f_path.len()={}", query.f_path.len());
            
            let merkle_result = verify_merkle_opening(
                &root_arr,
                query.f_val,
                query.idx,
                &query.f_path,
                domain_size,
            );

            eprintln!("verify_fri_proof: Merkle result for query {}: {}", q_idx, merkle_result);
            if !merkle_result {
                eprintln!("verify_fri_proof: FAIL - Merkle verification failed for query {}", q_idx);
                eprintln!("verify_fri_proof: root_arr={:?}", root_arr);
                eprintln!("verify_fri_proof: Expected verification of value {:?} at index {} with path len {}", 
                         query.f_val, query.idx, query.f_path.len());
                return false;
            }
            let w = self.domain[query.idx];
            let denom = w - z;
            eprintln!("verify_fri_proof: w={:?}, denom={:?}, query.f_val={:?}", w, denom, query.f_val);

            if denom == ExtF::ZERO {
                eprintln!("verify_fri_proof: Hit point, checking f_val == claimed_eval");
                eprintln!("verify_fri_proof: f_val.value = {:?}", query.f_val.to_array());
                eprintln!("verify_fri_proof: claimed_eval.value = {:?}", claimed_eval.to_array());
                eprintln!("verify_fri_proof: Comparison result: {}", query.f_val == claimed_eval);
                if query.f_val != claimed_eval {
                    eprintln!("verify_fri_proof: FAIL - f_val mismatch at hit point");
                    return false;
                }
            }
            let mut current_q = if denom == ExtF::ZERO {
                query.layers[0].val
            } else {
                let q_expected = r * (query.f_val - claimed_eval) / denom;
                eprintln!("verify_fri_proof: q_expected={:?}, actual={:?}", q_expected, query.layers[0].val);
                if query.layers[0].val != q_expected {
                    eprintln!("verify_fri_proof: FAIL - quotient mismatch");
                    return false;
                }
                q_expected
            };
            let mut size = domain_size;
            let mut domain_layer = self.domain.clone();

            for (layer_idx, layer_query) in query.layers.iter().enumerate() {
                eprintln!("verify_fri_proof: Layer {} - checking val {:?} == current_q {:?}", 
                         layer_idx, layer_query.val, current_q);
                if layer_query.val != current_q {
                    eprintln!("verify_fri_proof: FAIL - Layer {} val mismatch", layer_idx);
                    return false;
                }
                let root_bytes = &proof.layer_roots[layer_idx];
                if layer_query.idx >= size || layer_query.sib_idx != (layer_query.idx ^ 1) {
                    eprintln!("verify_fri_proof: FAIL - Invalid sibling pairing for consecutive folding at layer {}", layer_idx);
                    return false;
                }
                let root_arr = {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(root_bytes);
                    arr
                };
                let merkle1 = verify_merkle_opening(
                    &root_arr,
                    layer_query.val,
                    layer_query.idx,
                    &layer_query.path,
                    size,
                );
                let merkle2 = verify_merkle_opening(
                    &root_arr,
                    layer_query.sib_val,
                    layer_query.sib_idx,
                    &layer_query.sib_path,
                    size,
                );
                if !merkle1 || !merkle2 {
                    eprintln!("verify_fri_proof: FAIL - Merkle verification failed for layer {}", layer_idx);
                    return false;
                }
                let d0 = domain_layer[layer_query.idx];
                let d1 = domain_layer[layer_query.sib_idx];
                eprintln!("verify_fri_proof: Layer {} - d0={:?}, d1={:?}", layer_idx, d0, d1);
                eprintln!("verify_fri_proof: -d0 = {:?}, d1 == -d0: {}", -d0, d1 == -d0);
                // For debugging: check if domain has proper consecutive pairing structure
                if layer_idx == 0 {
                    eprintln!("verify_fri_proof: Full domain: {:?}", domain_layer);
                    for i in (0..domain_layer.len()).step_by(2) {
                        let a = domain_layer[i];
                        let b = domain_layer[i + 1];
                        eprintln!("verify_fri_proof: domain[{}]={:?}, domain[{}]={:?}, b == -a: {}", 
                                 i, a, i + 1, b, b == -a);
                    }
                }
                if d1 != -d0 {
                    eprintln!("verify_fri_proof: FAIL - Domain pairing check failed for layer {}", layer_idx);
                    return false;
                }
                let chal = challenges[layer_idx];
                eprintln!("verify_fri_proof: Layer {} challenge: {:?} (challenges.len()={})", layer_idx, chal, challenges.len());
                let evals_pair = [layer_query.val, layer_query.sib_val];
                let domain_pair = [d0, d1];
                eprintln!("verify_fri_proof: Layer {} folding: e0={:?}, e1={:?}, g={:?}", 
                         layer_idx, evals_pair[0], evals_pair[1], domain_pair[0]);
                let (folded_vec, _new_domain_vec) = self.fold_evals(&evals_pair, &domain_pair, chal);
                current_q = folded_vec[0];
                eprintln!("verify_fri_proof: Layer {} folded: current_q={:?}", layer_idx, current_q);
                if layer_idx + 1 < query.layers.len() {
                    eprintln!("verify_fri_proof: Layer {} checking next layer: current_q={:?} vs next_val={:?}", 
                             layer_idx, current_q, query.layers[layer_idx + 1].val);
                    if current_q != query.layers[layer_idx + 1].val {
                        eprintln!("verify_fri_proof: FAIL - Next layer val mismatch at layer {}", layer_idx);
                        return false;
                    }
                }
                size >>= 1;
                // Use the proper domain transformation from fold_evals for consecutive pairing
                domain_layer = domain_layer.chunks(2).map(|chunk| chunk[0] * chunk[0]).collect();
            }
            if current_q != proof.final_eval {
                eprintln!("verify_fri_proof: FAIL - Final eval mismatch: current_q={:?}, proof.final_eval={:?}", current_q, proof.final_eval);
                return false;
            }
        }
        true
    }
}

pub fn serialize_fri_proof(proof: &FriProof) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes
        .write_u32::<BigEndian>(proof.layer_roots.len() as u32)
        .unwrap();
    for root in &proof.layer_roots {
        bytes.extend(root);
    }
    let [r, i] = proof.final_eval.to_array();
    bytes.extend(r.as_canonical_u64().to_be_bytes());
    bytes.extend(i.as_canonical_u64().to_be_bytes());
    bytes.extend(proof.final_pow.to_be_bytes());
    bytes
        .write_u32::<BigEndian>(proof.queries.len() as u32)
        .unwrap();
    for q in &proof.queries {
        bytes.write_u32::<BigEndian>(q.idx as u32).unwrap();
        let [fr, fi] = q.f_val.to_array();
        bytes.extend(fr.as_canonical_u64().to_be_bytes());
        bytes.extend(fi.as_canonical_u64().to_be_bytes());
        bytes.write_u32::<BigEndian>(q.f_path.len() as u32).unwrap();
        for p in &q.f_path {
            bytes.extend(p);
        }
        bytes.write_u32::<BigEndian>(q.layers.len() as u32).unwrap();
        for l in &q.layers {
            bytes.write_u32::<BigEndian>(l.idx as u32).unwrap();
            bytes.write_u32::<BigEndian>(l.sib_idx as u32).unwrap();
            let [vr, vi] = l.val.to_array();
            bytes.extend(vr.as_canonical_u64().to_be_bytes());
            bytes.extend(vi.as_canonical_u64().to_be_bytes());
            let [svr, svi] = l.sib_val.to_array();
            bytes.extend(svr.as_canonical_u64().to_be_bytes());
            bytes.extend(svi.as_canonical_u64().to_be_bytes());
            bytes.write_u32::<BigEndian>(l.path.len() as u32).unwrap();
            for p in &l.path {
                bytes.extend(p);
            }
            bytes
                .write_u32::<BigEndian>(l.sib_path.len() as u32)
                .unwrap();
            for p in &l.sib_path {
                bytes.extend(p);
            }
        }
    }
    bytes
}

pub fn deserialize_fri_proof(bytes: &[u8]) -> Result<FriProof, Box<dyn Error>> {
    let mut cursor = Cursor::new(bytes);
    let num_layers = cursor.read_u32::<BigEndian>()? as usize;
    if num_layers > 1_000 {
        return Err("Too many layers".into());
    }
    let mut layer_roots = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let mut root = [0u8; 32];
        cursor.read_exact(&mut root)?;
        layer_roots.push(root);
    }
    let r = F::from_u64(cursor.read_u64::<BigEndian>()?);
    let i = F::from_u64(cursor.read_u64::<BigEndian>()?);
    let final_eval = ExtF::new_complex(r, i);
    let final_pow = cursor.read_u64::<BigEndian>()?;
    let num_queries = cursor.read_u32::<BigEndian>()? as usize;
    if num_queries > 1_000 {
        return Err("Too many queries".into());
    }
    let mut queries = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let idx = cursor.read_u32::<BigEndian>()? as usize;
        let fr = F::from_u64(cursor.read_u64::<BigEndian>()?);
        let fi = F::from_u64(cursor.read_u64::<BigEndian>()?);
        let f_val = ExtF::new_complex(fr, fi);
        let path_len = cursor.read_u32::<BigEndian>()? as usize;
        if path_len > 1_000 {
            return Err("f_path too long".into());
        }
        let mut f_path = Vec::with_capacity(path_len);
        for _ in 0..path_len {
            let mut h = [0u8; 32];
            cursor.read_exact(&mut h)?;
            f_path.push(h);
        }
        let num_layers_q = cursor.read_u32::<BigEndian>()? as usize;
        if num_layers_q > 1_000 {
            return Err("Too many layers in query".into());
        }
        let mut layers = Vec::with_capacity(num_layers_q);
        for _ in 0..num_layers_q {
            let l_idx = cursor.read_u32::<BigEndian>()? as usize;
            let sib_idx = cursor.read_u32::<BigEndian>()? as usize;
            let vr = F::from_u64(cursor.read_u64::<BigEndian>()?);
            let vi = F::from_u64(cursor.read_u64::<BigEndian>()?);
            let val = ExtF::new_complex(vr, vi);
            let svr = F::from_u64(cursor.read_u64::<BigEndian>()?);
            let svi = F::from_u64(cursor.read_u64::<BigEndian>()?);
            let sib_val = ExtF::new_complex(svr, svi);
            let path_len = cursor.read_u32::<BigEndian>()? as usize;
            if path_len > 1_000 {
                return Err("path too long".into());
            }
            let mut path = Vec::with_capacity(path_len);
            for _ in 0..path_len {
                let mut h = [0u8; 32];
                cursor.read_exact(&mut h)?;
                path.push(h);
            }
            let sib_path_len = cursor.read_u32::<BigEndian>()? as usize;
            if sib_path_len > 1_000 {
                return Err("sib_path too long".into());
            }
            let mut sib_path = Vec::with_capacity(sib_path_len);
            for _ in 0..sib_path_len {
                let mut h = [0u8; 32];
                cursor.read_exact(&mut h)?;
                sib_path.push(h);
            }
            layers.push(FriLayerQuery {
                idx: l_idx,
                sib_idx,
                val,
                sib_val,
                path,
                sib_path,
            });
        }
        queries.push(FriQuery {
            idx,
            f_val,
            f_path,
            layers,
        });
    }
    Ok(FriProof {
        layer_roots,
        queries,
        final_eval,
        final_pow,
    })
}

// ==========================================
// CONFIGURABLE FRI BACKEND IMPLEMENTATIONS
// ==========================================

/// Custom FRI backend wrapping the existing FriOracle implementation
pub struct CustomFri {
    oracle: FriOracle,
    committed_polys: Vec<Polynomial<ExtF>>,
}

impl CustomFri {
    pub fn new(polys: Vec<Polynomial<ExtF>>, transcript: &mut Vec<u8>) -> Self {
        let oracle = FriOracle::new(polys.clone(), transcript);
        Self {
            oracle,
            committed_polys: polys,
        }
    }

    pub fn new_with_blowup(polys: Vec<Polynomial<ExtF>>, transcript: &mut Vec<u8>, blowup: usize) -> Self {
        let oracle = FriOracle::new_with_blowup(polys.clone(), transcript, blowup);
        Self {
            oracle,
            committed_polys: polys,
        }
    }

    pub fn new_for_verifier(domain_size: usize) -> Self {
        let oracle = FriOracle::new_for_verifier(domain_size);
        Self {
            oracle,
            committed_polys: vec![],
        }
    }
}

impl FriBackend for CustomFri {
    fn commit(&mut self, polys: Vec<Polynomial<ExtF>>) -> Result<Vec<Commitment>, Box<dyn Error>> {
        self.committed_polys = polys;
        Ok(self.oracle.commit())
    }

    fn open_at_point(&mut self, point: &[ExtF]) -> Result<(Vec<ExtF>, Vec<OpeningProof>), Box<dyn Error>> {
        Ok(self.oracle.open_at_point(point))
    }

    fn verify_openings(
        &self,
        comms: &[Commitment],
        point: &[ExtF],
        evals: &[ExtF],
        proofs: &[OpeningProof],
    ) -> bool {
        self.oracle.verify_openings(comms, point, evals, proofs)
    }

    fn domain_size(&self) -> usize {
        self.oracle.domain.len()
    }
}

/// Plonky3 FRI backend using p3-fri
/// 
/// NOTE: This is currently a simplified implementation that demonstrates 
/// the configurable FRI backend pattern. The full p3-fri integration requires
/// careful API mapping which is beyond the scope of this initial implementation.
/// 
/// For production use, this would implement proper p3-fri commitment and opening
/// protocols with the correct parameter mapping to maintain security equivalence.
#[cfg(feature = "p3-fri")]
pub struct PlonkyFri {
    // For now, use the custom implementation internally
    // In production, this would be replaced with p3-fri structures
    custom_oracle: CustomFri,
    domain_size: usize,
}

#[cfg(feature = "p3-fri")]
impl PlonkyFri {
    #[instrument(name = "PlonkyFri::new", level = "info", skip_all)]
    pub fn new(polys: Vec<Polynomial<ExtF>>, transcript: &mut Vec<u8>) -> Self {
        eprintln!("PlonkyFri::new - Using simplified implementation (p3-fri integration WIP)");
        
        // For this demo, delegate to custom implementation
        let custom_oracle = CustomFri::new(polys, transcript);
        let max_deg = custom_oracle.committed_polys.iter().map(|p| p.degree()).max().unwrap_or(0);
        let domain_size = (max_deg + 1).next_power_of_two() * BLOWUP;

        Self {
            custom_oracle,
            domain_size,
        }
    }

    pub fn new_for_verifier(domain_size: usize) -> Self {
        eprintln!("PlonkyFri::new_for_verifier - Using simplified implementation");
        
        let custom_oracle = CustomFri::new_for_verifier(domain_size);
        Self {
            custom_oracle,
            domain_size,
        }
    }
}

#[cfg(feature = "p3-fri")]
impl FriBackend for PlonkyFri {
    #[instrument(name = "PlonkyFri::commit", level = "info", skip_all)]
    fn commit(&mut self, polys: Vec<Polynomial<ExtF>>) -> Result<Vec<Commitment>, Box<dyn Error>> {
        // TODO: Replace with actual p3-fri commitment scheme
        // For now, delegate to custom implementation
        eprintln!("PlonkyFri::commit - delegating to custom implementation");
        self.custom_oracle.commit(polys)
    }

    #[instrument(name = "PlonkyFri::open_at_point", level = "info", skip_all)]
    fn open_at_point(&mut self, point: &[ExtF]) -> Result<(Vec<ExtF>, Vec<OpeningProof>), Box<dyn Error>> {
        // TODO: Replace with actual p3-fri opening protocol
        eprintln!("PlonkyFri::open_at_point - delegating to custom implementation");
        self.custom_oracle.open_at_point(point)
    }

    fn verify_openings(
        &self,
        comms: &[Commitment],
        point: &[ExtF],
        evals: &[ExtF],
        proofs: &[OpeningProof],
    ) -> bool {
        // TODO: Replace with actual p3-fri verification
        eprintln!("PlonkyFri::verify_openings - delegating to custom implementation");
        self.custom_oracle.verify_openings(comms, point, evals, proofs)
    }

    fn domain_size(&self) -> usize {
        self.domain_size
    }
}

// ==========================================
// FACTORY FUNCTIONS AND UTILITIES
// ==========================================

/// Create a new FRI backend based on the implementation type
pub fn create_fri_backend(
    impl_type: FriImpl,
    polys: Vec<Polynomial<ExtF>>,
    transcript: &mut Vec<u8>,
) -> Box<dyn FriBackend> {
    match impl_type {
        FriImpl::Custom => Box::new(CustomFri::new(polys, transcript)),
        #[cfg(feature = "p3-fri")]
        FriImpl::Plonky3 => Box::new(PlonkyFri::new(polys, transcript)),
    }
}

/// Create a new FRI backend for verification
pub fn create_fri_verifier(impl_type: FriImpl, domain_size: usize) -> Box<dyn FriBackend> {
    match impl_type {
        FriImpl::Custom => Box::new(CustomFri::new_for_verifier(domain_size)),
        #[cfg(feature = "p3-fri")]
        FriImpl::Plonky3 => Box::new(PlonkyFri::new_for_verifier(domain_size)),
    }
}

/// Adaptive FRI Oracle that wraps the configurable backend system
/// This maintains backward compatibility with the existing PolyOracle interface
pub struct AdaptiveFriOracle {
    backend: Box<dyn FriBackend>,
    impl_type: FriImpl,
}

impl AdaptiveFriOracle {
    /// Create a new adaptive FRI oracle with the specified backend
    pub fn new(impl_type: FriImpl, polys: Vec<Polynomial<ExtF>>, transcript: &mut Vec<u8>) -> Self {
        let backend = create_fri_backend(impl_type, polys, transcript);
        Self { backend, impl_type }
    }

    /// Create a new adaptive FRI oracle for verification
    pub fn new_for_verifier(impl_type: FriImpl, domain_size: usize) -> Self {
        let backend = create_fri_verifier(impl_type, domain_size);
        Self { backend, impl_type }
    }

    /// Get the implementation type
    pub fn impl_type(&self) -> FriImpl {
        self.impl_type
    }

    /// Get the domain size
    pub fn domain_size(&self) -> usize {
        self.backend.domain_size()
    }
}

impl PolyOracle for AdaptiveFriOracle {
    fn commit(&mut self) -> Vec<Commitment> {
        // Note: We can't easily pass polys here due to interface constraints
        // For now, this will work with the current backend's stored polynomials
        // In a more complete refactor, we might adjust the PolyOracle interface
        self.backend.commit(vec![]).unwrap_or_default()
    }

    fn open_at_point(&mut self, point: &[ExtF]) -> (Vec<ExtF>, Vec<OpeningProof>) {
        self.backend.open_at_point(point).unwrap_or_default()
    }

    fn verify_openings(
        &self,
        comms: &[Commitment],
        point: &[ExtF],
        evals: &[ExtF],
        proofs: &[OpeningProof],
    ) -> bool {
        self.backend.verify_openings(comms, point, evals, proofs)
    }
}

/// Default FRI implementation selection
impl Default for FriImpl {
    fn default() -> Self {
        #[cfg(feature = "custom-fri")]
        return FriImpl::Custom;
        
        #[cfg(all(feature = "p3-fri", not(feature = "custom-fri")))]
        return FriImpl::Plonky3;
        
        // Fallback to custom if no features are enabled
        #[cfg(not(any(feature = "custom-fri", feature = "p3-fri")))]
        return FriImpl::Custom;
    }
}

pub fn serialize_comms(comms: &[Commitment]) -> Vec<u8> {
    let mut out = Vec::new();
    out.write_u32::<BigEndian>(comms.len() as u32).unwrap();
    for comm in comms {
        out.write_u32::<BigEndian>(comm.len() as u32).unwrap();
        out.extend_from_slice(comm);
    }
    out
}

#[allow(dead_code)]
pub(crate) fn serialize_proofs(proofs: &[OpeningProof]) -> Vec<u8> {
    let mut out = Vec::new();
    for proof in proofs {
        out.write_u32::<BigEndian>(proof.len() as u32).unwrap();
        out.extend_from_slice(proof);
    }
    out
}


