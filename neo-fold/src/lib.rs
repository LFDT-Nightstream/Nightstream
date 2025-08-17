use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use neo_ccs::{ccs_sumcheck_verifier, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::{embed_base_to_ext, project_ext_to_base, ExtF, ExtFieldNorm, F};
use neo_modint::ModInt;
use neo_poly::Polynomial;
use neo_ring::RingElement;
use neo_sumcheck::{
    batched_sumcheck_prover, batched_sumcheck_verifier, challenger::NeoChallenger,
    fiat_shamir_challenge, Commitment, FriOracle, OpeningProof, PolyOracle, UnivPoly,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::Matrix;
use rand::{rngs::StdRng, SeedableRng};
use std::io::{Cursor, Read};

pub type FriCommitment = Commitment;
pub type FriProof = OpeningProof;

#[derive(Clone)]
pub struct EvalInstance {
    pub commitment: Vec<RingElement<ModInt>>,
    pub r: Vec<ExtF>,    // Eval point over extension
    pub ys: Vec<ExtF>,   // Evals of Zj at r
    pub u: ExtF,         // Relaxation scalar
    pub e_eval: ExtF,    // Eval of E at r
    pub norm_bound: u64, // b for low-norm
}

#[derive(Clone)]
pub struct Proof {
    pub transcript: Vec<u8>,
}

#[derive(Clone)]
pub struct FriConfig {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
}

pub struct FoldState {
    pub structure: CcsStructure,
    pub eval_instances: Vec<EvalInstance>,
    pub ccs_instance: Option<(CcsInstance, CcsWitness)>,
    pub extension_degree: usize, // 2 for quadratic
    pub fri_config: FriConfig,
    pub transcript: Vec<u8>,
    pub sumcheck_msgs: Vec<Vec<(Polynomial<ExtF>, ExtF)>>,
    pub rhos: Vec<F>,
    pub fri_commit: Option<FriCommitment>,
    pub fri_proof: Option<FriProof>,
    pub max_blind_norm: u64,
}

impl FoldState {
    pub fn new(structure: CcsStructure) -> Self {
        Self {
            structure,
            eval_instances: vec![],
            ccs_instance: None,
            extension_degree: 2,
            fri_config: FriConfig {
                log_blowup: 1,
                num_queries: 50,
                proof_of_work_bits: 8,
            },
            transcript: vec![],
            sumcheck_msgs: vec![],
            rhos: vec![],
            fri_commit: None,
            fri_proof: None,
            max_blind_norm: SECURE_PARAMS.max_blind_norm,
        }
    }

    pub fn verify_state(&self) -> bool {
        // Stub: check if final state has single eval instance
        self.eval_instances.len() == 1
    }

    /// Recursive IVC driver. Folds the current proof into a verifier CCS,
    /// verifies that proof, then recurses for `depth - 1`.
    pub fn recursive_ivc(&mut self, depth: usize, committer: &AjtaiCommitter) -> bool {
        if depth == 0 {
            return self.verify_state();
        }
        // Use the committer's bound for blinding norms at this level
        self.max_blind_norm = committer.params().max_blind_norm;
        // Generate a proof for the current CCS instance (or a default one)
        let (inst, wit) = self
            .ccs_instance
            .clone()
            .unwrap_or_else(|| {
                let inst = CcsInstance {
                    commitment: vec![],
                    public_input: vec![],
                    u: F::ZERO,
                    e: F::ONE,
                };
                let wit = CcsWitness { z: vec![ExtF::ZERO] };
                (inst, wit)
            });
        let current_proof =
            self.generate_proof((inst.clone(), wit.clone()), (inst, wit), committer);
        if !self.verify(&current_proof.transcript, committer) {
            return false;
        }
        // Extract a witness for the verifier CCS using the transcript
        let ver_wit = extractor(&current_proof);
        // For this toy implementation, reuse the existing structure as the verifier circuit
        let verifier_ccs = self.structure.clone();
        let ver_inst = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        // Set up next recursive state using the extracted witness
        self.structure = verifier_ccs;
        self.ccs_instance = Some((ver_inst, ver_wit));
        self.eval_instances.clear();
        // Continue recursion on the verifier instance
        self.recursive_ivc(depth - 1, committer)
    }

    // Poseidon2-based transcript hash to align with FS challenges
    fn hash_transcript(&self, data: &[u8]) -> [u8; 32] {
        // Derive 32 bytes by calling the Poseidon2-based FS challenge twice with domain separation
        let mut t0 = data.to_vec();
        t0.extend_from_slice(b"|hash0");
        let h0 = fiat_shamir_challenge(&t0);
        let mut t1 = data.to_vec();
        t1.extend_from_slice(b"|hash1");
        let h1 = fiat_shamir_challenge(&t1);
        let a0 = h0.to_array();
        let a1 = h1.to_array();
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&a0[0].as_canonical_u64().to_be_bytes());
        out[8..16].copy_from_slice(&a0[1].as_canonical_u64().to_be_bytes());
        out[16..24].copy_from_slice(&a1[0].as_canonical_u64().to_be_bytes());
        out[24..32].copy_from_slice(&a1[1].as_canonical_u64().to_be_bytes());
        out
    }

    pub fn generate_proof(
        &mut self,
        instance1: (CcsInstance, CcsWitness),
        instance2: (CcsInstance, CcsWitness),
        committer: &AjtaiCommitter,
    ) -> Proof {
        let mut transcript = std::mem::take(&mut self.transcript);
        transcript.clear();
        self.sumcheck_msgs.clear();
        self.rhos.clear();
        // Initialize challenger for folding protocol
        let mut challenger = NeoChallenger::new("neo_folding");
        transcript.extend(b"neo_pi_ccs1");
        self.ccs_instance = Some(instance1);
        let msgs1 = pi_ccs(self, committer, &mut transcript);
        self.sumcheck_msgs.push(msgs1);
        transcript.extend(b"neo_pi_dec1");
        pi_dec(self, committer, &mut transcript);
        transcript.extend(b"neo_pi_ccs2");
        self.ccs_instance = Some(instance2);
        let msgs2 = pi_ccs(self, committer, &mut transcript);
        self.sumcheck_msgs.push(msgs2);
        transcript.extend(b"neo_pi_dec2");
        pi_dec(self, committer, &mut transcript);
        transcript.extend(b"neo_pi_rlc");
        // Derive rotation challenge ρ ∈ C using the challenger bound to current transcript
        challenger.observe_bytes("transcript_prefix", &transcript);
        let rho_rot = challenger.challenge_rotation("rlc_rho", committer.params().n);
        // Serialize rotation coefficients for verifier
        serialize_rotation(&rho_rot, &mut transcript);
        pi_rlc(self, rho_rot.clone(), committer, &mut transcript);
        // No legacy base-limb storage; if needed, store a hash of rho_rot instead
        transcript.extend(b"neo_fri");
        let (commit, proof) = self.fri_compress_final().expect("FRI failed");
        transcript.extend(&serialize_fri_commit(&commit));
        transcript.extend(&serialize_fri_proof(&proof));
        self.fri_commit = Some(commit);
        self.fri_proof = Some(proof);
        challenger.observe_bytes("fri_proof", &transcript);
        let hash = self.hash_transcript(&transcript);
        transcript.extend(&hash);
        self.transcript = transcript.clone();
        Proof { transcript }
    }

    pub fn verify(&self, full_transcript: &[u8], committer: &AjtaiCommitter) -> bool {
        if full_transcript.len() < 32 {
            return false;
        }
        let (prefix, hash_bytes) = full_transcript.split_at(full_transcript.len() - 32);
        let mut expected = [0u8; 32];
        expected.copy_from_slice(hash_bytes);
        if self.hash_transcript(prefix) != expected {
            return false;
        }

        let mut cursor = Cursor::new(prefix);
        let mut reconstructed = Vec::new();

        // --- First CCS instance ---
        if read_tag(&mut cursor, b"neo_pi_ccs1").is_err() {
            return false;
        }
        let commit1 = read_commit(&mut cursor, committer.params().n);
        let msgs1 = extract_msgs_ccs(&mut cursor, self.structure.max_deg);
        let mut vt_transcript = cursor.get_ref()[0..cursor.position() as usize].to_vec();
        let domain_size = (self.structure.max_deg + 1).next_power_of_two().max(1) * 4;
        let mut oracle = FriOracle::new_for_verifier(domain_size);
        let comms1 = match read_comms_block(&mut cursor) {
            Some(c) => c,
            None => return false,
        };
        let (r1, final_eval1) = match ccs_sumcheck_verifier(
            &self.structure,
            ExtF::ZERO,
            &msgs1,
            committer.params().norm_bound,
            &comms1,
            &mut oracle,
            &mut vt_transcript,
        ) {
            Some(res) => res,
            None => return false,
        };
        let ys1 = match read_ys(&mut cursor, self.structure.mats.len()) {
            Some(v) => v,
            None => return false,
        };
        let first_eval = EvalInstance {
            commitment: commit1.clone(),
            r: r1.clone(),
            ys: ys1.clone(),
            u: ExtF::ZERO,
            e_eval: final_eval1,
            norm_bound: committer.params().norm_bound,
        };
        let first_instance = CcsInstance {
            commitment: commit1,
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        if !verify_ccs(
            &self.structure,
            &first_instance,
            self.max_blind_norm,
            &msgs1,
            &[first_eval.clone()],
            committer,
        ) {
            return false;
        }
        reconstructed.push(first_eval);

        // --- Decomposition check for first instance ---
        if read_tag(&mut cursor, b"neo_pi_dec1").is_err() {
            return false;
        }
        if read_tag(&mut cursor, b"dec_rand").is_err() {
            return false;
        }
        let dec_commit1 = read_commit(&mut cursor, committer.params().n);
        let prev_eval = reconstructed.last().cloned().unwrap();
        let dec_eval = EvalInstance {
            commitment: dec_commit1.clone(),
            r: prev_eval.r.clone(),
            ys: prev_eval.ys.clone(),
            u: prev_eval.u,
            e_eval: prev_eval.e_eval,
            norm_bound: committer.params().norm_bound,
        };
        if !verify_dec(committer, &prev_eval, &dec_eval) {
            return false;
        }
        reconstructed.push(dec_eval);

        // --- Second CCS instance ---
        if read_tag(&mut cursor, b"neo_pi_ccs2").is_err() {
            return false;
        }
        let commit2 = read_commit(&mut cursor, committer.params().n);
        let msgs2 = extract_msgs_ccs(&mut cursor, self.structure.max_deg);
        let mut vt_transcript2 = cursor.get_ref()[0..cursor.position() as usize].to_vec();
        let mut oracle2 = FriOracle::new_for_verifier(domain_size);
        let comms2 = match read_comms_block(&mut cursor) {
            Some(c) => c,
            None => return false,
        };
        let (r2, final_eval2) = match ccs_sumcheck_verifier(
            &self.structure,
            ExtF::ZERO,
            &msgs2,
            committer.params().norm_bound,
            &comms2,
            &mut oracle2,
            &mut vt_transcript2,
        ) {
            Some(res) => res,
            None => return false,
        };
        let ys2 = match read_ys(&mut cursor, self.structure.mats.len()) {
            Some(v) => v,
            None => return false,
        };
        let second_eval = EvalInstance {
            commitment: commit2.clone(),
            r: r2.clone(),
            ys: ys2.clone(),
            u: ExtF::ZERO,
            e_eval: final_eval2,
            norm_bound: committer.params().norm_bound,
        };
        let second_instance = CcsInstance {
            commitment: commit2,
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        if !verify_ccs(
            &self.structure,
            &second_instance,
            self.max_blind_norm,
            &msgs2,
            &[second_eval.clone()],
            committer,
        ) {
            return false;
        }
        reconstructed.push(second_eval);

        // --- Decomposition check for second instance ---
        if read_tag(&mut cursor, b"neo_pi_dec2").is_err() {
            return false;
        }
        if read_tag(&mut cursor, b"dec_rand").is_err() {
            return false;
        }
        let dec_commit2 = read_commit(&mut cursor, committer.params().n);
        let prev_eval = reconstructed.last().cloned().unwrap();
        let dec_eval2 = EvalInstance {
            commitment: dec_commit2.clone(),
            r: prev_eval.r.clone(),
            ys: prev_eval.ys.clone(),
            u: prev_eval.u,
            e_eval: prev_eval.e_eval,
            norm_bound: committer.params().norm_bound,
        };
        if !verify_dec(committer, &prev_eval, &dec_eval2) {
            return false;
        }
        reconstructed.push(dec_eval2);

        // --- Random linear combination ---
        if read_tag(&mut cursor, b"neo_pi_rlc").is_err() {
            return false;
        }
        // Read rotation challenge ρ
        let rho_rot = match read_rotation(&mut cursor, committer.params().n) {
            Some(r) => r,
            None => return false,
        };
        let combo_commit = read_commit(&mut cursor, committer.params().n);
        if reconstructed.len() < 4 {
            return false;
        }
        let e1 = reconstructed[reconstructed.len() - 3].clone();
        let e2 = reconstructed[reconstructed.len() - 1].clone();
        // Use the same scalar derived from `rho_rot` as the committer
        let rho_scalar = rho_scalar_from_rotation(&rho_rot);
        let combo_ys = e1
            .ys
            .iter()
            .zip(&e2.ys)
            .map(|(&y1, &y2)| y1 + rho_scalar * y2)
            .collect();
        let u_new = e1.u + rho_scalar * e2.u;
        let e_eval_new = e1.e_eval + rho_scalar * e2.e_eval;
        let rho_norm = rho_rot.norm_inf() as f64;
        let new_norm_bound = ((e1.norm_bound as f64).powi(2)
            + (rho_norm.powi(2) * (e2.norm_bound as f64).powi(2)))
        .sqrt()
        .ceil() as u64;
        let combo_eval = EvalInstance {
            commitment: combo_commit.clone(),
            r: e1.r.clone(),
            ys: combo_ys,
            u: u_new,
            e_eval: e_eval_new,
            norm_bound: new_norm_bound,
        };
        if !verify_rlc(&e1, &e2, &rho_rot, &combo_eval, committer) {
            return false;
        }
        reconstructed.push(combo_eval);

        // --- FRI compression ---
        if read_tag(&mut cursor, b"neo_fri").is_err() {
            return false;
        }
        let commit = read_fri_commit(&mut cursor);
        let proof_obj = read_fri_proof(&mut cursor);
        if let Some(last_eval) = reconstructed.last() {
            if !Self::fri_verify_compressed(
                &commit,
                &proof_obj,
                &last_eval.r,
                last_eval.e_eval,
                last_eval.ys.len(),
            ) {
                return false;
            }
        } else {
            return false;
        }

        true
    }
}

pub fn extractor(proof: &Proof) -> CcsWitness {
    fn rewind_transcript(trans: &[u8], fork: u8) -> ExtF {
        let mut t = trans.to_vec();
        t.push(fork);
        fiat_shamir_challenge(&t)
    }
    let rho1 = rewind_transcript(&proof.transcript, 0);
    let rho2 = rewind_transcript(&proof.transcript, 1);
    assert!(rho1 != rho2, "Equal challenges");
    let open1 = open_from_transcript(&proof.transcript, rho1);
    let open2 = open_from_transcript(&proof.transcript, rho2);
    let w_ext = (open1.eval - open2.eval) / (rho1 - rho2);
    CcsWitness { z: vec![w_ext] }
}

fn open_from_transcript(_transcript: &[u8], _rho: ExtF) -> Opening {
    Opening {
        eval: _rho,
        proof: vec![],
    }
}

fn univpoly_to_polynomial(poly: &dyn UnivPoly, degree: usize) -> Polynomial<ExtF> {
    let points: Vec<ExtF> = (0..=degree)
        .map(|i| embed_base_to_ext(F::from_u64(i as u64)))
        .collect();
    let evals: Vec<ExtF> = points.iter().map(|&x| poly.evaluate(&[x])).collect();
    Polynomial::interpolate(&points, &evals)
}

#[allow(dead_code)]
struct Opening {
    eval: ExtF,
    proof: Vec<RingElement<ModInt>>,
}

// Helper: Multilinear extension over base F, padded to power of 2
fn multilinear_extension(z: &[ExtF], l: usize) -> neo_sumcheck::MultilinearEvals {
    let mut padded_z = z.to_vec();
    padded_z.resize(1 << l, ExtF::ZERO);
    neo_sumcheck::MultilinearEvals::new(padded_z)
}

fn eq_poly(point: &[ExtF], b: usize, l: usize) -> ExtF {
    (0..l).fold(ExtF::ONE, |acc, j| {
        let bit = (b >> j) & 1;
        acc * if bit == 1 {
            point[j]
        } else {
            ExtF::ONE - point[j]
        }
    })
}

struct CCSQPoly<'a> {
    structure: &'a CcsStructure,
    mjz_rows: Vec<Vec<ExtF>>, // precomputed (M_j z)_b values
    alpha: ExtF,
    l: usize,
}

impl<'a> UnivPoly for CCSQPoly<'a> {
    fn evaluate(&self, point: &[ExtF]) -> ExtF {
        if point.len() != self.l {
            return ExtF::ZERO;
        }
        let mut sum_q = ExtF::ZERO;
        let mut alpha_pow = ExtF::ONE;
        for b in 0..self.structure.num_constraints {
            let eq = eq_poly(point, b, self.l);
            let mut inputs = vec![ExtF::ZERO; self.structure.mats.len()];
            for j in 0..self.structure.mats.len() {
                inputs[j] = self.mjz_rows[j][b];
            }
            sum_q += eq * alpha_pow * self.structure.f.evaluate(&inputs);
            alpha_pow *= self.alpha;
        }
        sum_q
    }

    fn degree(&self) -> usize {
        self.l
    }

    fn max_individual_degree(&self) -> usize {
        2
    }
}

fn construct_q<'a>(
    structure: &'a CcsStructure,
    instance: &'a CcsInstance,
    witness: &'a CcsWitness,
    transcript: &mut Vec<u8>,
) -> Box<dyn UnivPoly + 'a> {
    let l_constraints = (structure.num_constraints as f64).log2().ceil() as usize;
    let l_witness = (structure.witness_size as f64).log2().ceil() as usize;
    let l = l_constraints.max(l_witness);
    transcript.extend(b"ccs_alpha");
    let alpha = fiat_shamir_challenge(transcript);

    let mut full_z: Vec<ExtF> = instance
        .public_input
        .iter()
        .map(|&x| embed_base_to_ext(x))
        .collect();
    full_z.extend_from_slice(&witness.z);
    assert_eq!(full_z.len(), structure.witness_size);
    let s = structure.mats.len();
    let mut mjz_rows = vec![vec![ExtF::ZERO; structure.num_constraints]; s];
    for j in 0..s {
        for b in 0..structure.num_constraints {
            let mut sum = ExtF::ZERO;
            for k in 0..structure.witness_size {
                let m = structure.mats[j].get(b, k).unwrap_or(ExtF::ZERO);
                let z = *full_z.get(k).unwrap_or(&ExtF::ZERO);
                sum += m * z;
            }
            mjz_rows[j][b] = sum;
        }
    }

    Box::new(CCSQPoly {
        structure,
        mjz_rows,
        alpha,
        l,
    })
}

struct NormCheckPoly {
    values: Vec<ExtF>,
    degree: usize,
}

impl UnivPoly for NormCheckPoly {
    fn evaluate(&self, point: &[ExtF]) -> ExtF {
        if self.degree == 0 {
            return self.values.get(0).copied().unwrap_or(ExtF::ZERO);
        }
        assert_eq!(point.len(), self.degree);
        let mut result = ExtF::ZERO;

        for (idx, &val) in self.values.iter().enumerate() {
            let mut basis = ExtF::ONE;
            for (j, &x) in point.iter().enumerate() {
                let bit = (idx >> j) & 1;
                basis *= if bit == 1 { x } else { ExtF::ONE - x };
            }

            result += basis * val;
        }

        result
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn max_individual_degree(&self) -> usize {
        1
    }
}

fn batch_norm_checks(
    witness: &CcsWitness,
    b: u64,
    degree: usize,
    transcript: &mut Vec<u8>,
) -> Box<dyn UnivPoly> {
    // Derive random alpha for batching the witness entries
    transcript.extend(b"norm_alpha");
    let alpha = fiat_shamir_challenge(transcript);

    let padded = 1usize << degree;
    let mut values = vec![ExtF::ZERO; padded];

    let mut alpha_pow = ExtF::ONE;
    for (i, value) in values.iter_mut().enumerate() {
        let w = if i < witness.z.len() {
            witness.z[i]
        } else {
            ExtF::ZERO
        };
        // Polynomial that vanishes for all integers in [-(b-1)/2, (b-1)/2]
        let mut prod = w; // include root at 0
        let bound = (b - 1) / 2;
        for k in 1..=bound {
            let kf = embed_base_to_ext(F::from_u64(k));
            prod *= w * w - kf * kf;
        }
        *value = alpha_pow * prod;
        alpha_pow *= alpha;
    }

    Box::new(NormCheckPoly { values, degree })
}

struct ScaledPoly<'a> {
    poly: &'a dyn UnivPoly,
    scalar: ExtF,
}

impl<'a> UnivPoly for ScaledPoly<'a> {
    fn evaluate(&self, point: &[ExtF]) -> ExtF {
        self.scalar * self.poly.evaluate(point)
    }

    fn degree(&self) -> usize {
        self.poly.degree()
    }

    fn max_individual_degree(&self) -> usize {
        self.poly.max_individual_degree()
    }
}

pub fn pi_ccs(
    fold_state: &mut FoldState,
    committer: &AjtaiCommitter,
    transcript: &mut Vec<u8>,
) -> Vec<(Polynomial<ExtF>, ExtF)> {
    if let Some((ccs_instance, ccs_witness)) = fold_state.ccs_instance.take() {
        let params = committer.params();
        let q_poly = construct_q(
            &fold_state.structure,
            &ccs_instance,
            &ccs_witness,
            transcript,
        );
        let l_constraints = (fold_state.structure.num_constraints as f32).log2().ceil() as usize;
        let l_witness = (ccs_witness.z.len() as f32).log2().ceil() as usize;
        let l = l_constraints.max(l_witness);
        let norm_poly = batch_norm_checks(&ccs_witness, params.norm_bound, l, transcript);
        transcript.extend(b"ccs_rho");
        let rho = fiat_shamir_challenge(transcript);
        let scaled_norm = ScaledPoly {
            poly: &*norm_poly,
            scalar: rho,
        };
        // Convert univariate polynomials to dense form for FRI commitments
        let q_dense = univpoly_to_polynomial(&*q_poly, l);
        let norm_dense = univpoly_to_polynomial(&*norm_poly, l);
        let scaled_norm_dense = norm_dense.clone() * Polynomial::new(vec![rho]);
        let mut prover_oracle = FriOracle::new(vec![q_dense.clone(), scaled_norm_dense.clone()], transcript);
        let (sumcheck_msgs, comms) = match batched_sumcheck_prover(
            &[ExtF::ZERO, ExtF::ZERO],
            &[&*q_poly, &scaled_norm],
            &mut prover_oracle,
            transcript,
        ) {
            Ok(v) => v,
            Err(_) => return vec![],
        };
        let mut vt_transcript = transcript.clone();
        let q_poly_vt = construct_q(
            &fold_state.structure,
            &ccs_instance,
            &ccs_witness,
            &mut vt_transcript,
        );
        let norm_poly_vt =
            batch_norm_checks(&ccs_witness, params.norm_bound, l, &mut vt_transcript);
        let q_vt_dense = univpoly_to_polynomial(&*q_poly_vt, l);
        let norm_vt_dense = univpoly_to_polynomial(&*norm_poly_vt, l);
        let vt_scaled = norm_vt_dense.clone() * Polynomial::new(vec![rho]);
        let mut vt_oracle = FriOracle::new(vec![q_vt_dense, vt_scaled], &mut vt_transcript);
        let (challenges, evals) = match batched_sumcheck_verifier(
            &[ExtF::ZERO, ExtF::ZERO],
            &sumcheck_msgs,
            &comms,
            &mut vt_oracle,
            &mut vt_transcript,
        ) {
            Some(res) => res,
            None => (vec![ExtF::ZERO; l], vec![ExtF::ZERO; 2]),
        };

        let mut ys = Vec::new();
        for mat in &fold_state.structure.mats {
            let mut mz = vec![ExtF::ZERO; fold_state.structure.num_constraints];
            for row in 0..fold_state.structure.num_constraints {
                let mut sum = ExtF::ZERO;
                for col in 0..fold_state.structure.witness_size {
                    sum += mat.get(row, col).unwrap_or(ExtF::ZERO) * ccs_witness.z[col];
                }
                mz[row] = sum;
            }
            let mz_mle = multilinear_extension(&mz, challenges.len());
            ys.push(mz_mle.evaluate(&challenges));
        }
        let eval = EvalInstance {
            commitment: ccs_instance.commitment.clone(),
            r: challenges.clone(),
            ys: ys.clone(),
            u: ExtF::ZERO,
            e_eval: *evals.get(1).unwrap_or(&ExtF::ZERO),
            norm_bound: params.norm_bound,
        };
        fold_state.eval_instances.push(eval);
        transcript.extend(serialize_commit(&ccs_instance.commitment));
        serialize_sumcheck_msgs(transcript, &sumcheck_msgs);
        // Append oracle commitments for sumcheck opening verification (must precede ys to match verifier)
        serialize_comms_block(
            transcript,
            &comms.iter().map(|c| c.clone()).collect::<Vec<_>>(),
        );
        serialize_ys(transcript, &ys);

        sumcheck_msgs
    } else {
        vec![]
    }
}

pub fn pi_rlc(
    fold_state: &mut FoldState,
    rho_rot: RingElement<ModInt>,
    committer: &AjtaiCommitter,
    transcript: &mut Vec<u8>,
) {
    if fold_state.eval_instances.len() >= 4 {
        let e1 = fold_state.eval_instances[fold_state.eval_instances.len() - 3].clone();
        let e2 = fold_state.eval_instances[fold_state.eval_instances.len() - 1].clone();
        let combo_commit =
            committer.random_linear_combo_rotation(&e1.commitment, &e2.commitment, &rho_rot);
        transcript.extend(serialize_commit(&combo_commit));
        // Derive scalar ρ deterministically from rotation for scalar actions on ys/u/e_eval
        let rho_scalar = rho_scalar_from_rotation(&rho_rot);
        let combo_ys = e1
            .ys
            .iter()
            .zip(&e2.ys)
            .map(|(&y1, &y2)| y1 + rho_scalar * y2)
            .collect();
        let u_new = e1.u + rho_scalar * e2.u;
        let e_eval_new = e1.e_eval + rho_scalar * e2.e_eval;
        let rho_norm = rho_rot.norm_inf() as f64;
        let new_norm_bound = ((e1.norm_bound as f64).powi(2)
            + (rho_norm.powi(2) * (e2.norm_bound as f64).powi(2)))
        .sqrt()
        .ceil() as u64;
        fold_state.eval_instances.push(EvalInstance {
            commitment: combo_commit,
            r: e1.r,
            ys: combo_ys,
            u: u_new,
            e_eval: e_eval_new,
            norm_bound: new_norm_bound,
        });
        fold_state.max_blind_norm = ((fold_state.max_blind_norm.pow(2) * 2) as f64)
            .sqrt()
            .ceil() as u64;
    }
}

pub fn pi_dec(fold_state: &mut FoldState, committer: &AjtaiCommitter, transcript: &mut Vec<u8>) {
    if let Some(eval) = fold_state.eval_instances.last().cloned() {
        let params = committer.params();
        let ys_base = eval.ys.iter().map(|&y| y.to_array()[0]).collect::<Vec<F>>();
        let decomp_mat = decomp_b(&ys_base, params.b, params.d);
        let w = AjtaiCommitter::pack_decomp(&decomp_mat, &params);
        let mut challenger = NeoChallenger::new("neo_pi_dec");
        challenger.observe_bytes("transcript", transcript);
        transcript.extend(b"dec_rand");
        challenger.observe_bytes("dec_rand", b"dec_rand");
        let seed = challenger.challenge_base("dec_seed").as_canonical_u64();
        let mut rng = StdRng::seed_from_u64(seed);
        if let Ok((new_commit, _, _, _)) = committer.commit_with_rng(&w, &mut rng) {
            transcript.extend(serialize_commit(&new_commit));

            fold_state.eval_instances.push(EvalInstance {
                commitment: new_commit,
                r: eval.r.clone(),
                ys: eval.ys.clone(),
                u: eval.u,
                e_eval: eval.e_eval,
                norm_bound: params.norm_bound,
            });
        }
    }
}

pub fn verify_ccs(
    structure: &CcsStructure,
    _instance: &CcsInstance,
    max_blind_norm: u64,
    sumcheck_msgs: &[(Polynomial<ExtF>, ExtF)],
    new_evals: &[EvalInstance],
    committer: &AjtaiCommitter,
) -> bool {
    if new_evals.len() != 1 {
        return false;
    }
    let eval = &new_evals[0];
    if eval.ys.len() != structure.mats.len() {
        return false;
    }
    if eval.norm_bound > max_blind_norm {
        return false;
    }
    for &y in &eval.ys {
        if project_ext_to_base(y).is_none() || y.abs_norm() > eval.norm_bound {
            return false;
        }
    }
    if project_ext_to_base(eval.e_eval).is_none()
        || eval.e_eval.abs_norm() > max_blind_norm
        || eval.e_eval.abs_norm() > eval.norm_bound
    {
        return false;
    }
    if project_ext_to_base(eval.u).is_none() {
        return false;
    }
    let f_val = structure.f.evaluate(&eval.ys);
    let relaxed = eval.u * eval.e_eval;
    let f_val_base = project_ext_to_base(f_val);
    let relaxed_base = project_ext_to_base(relaxed);
    if f_val_base.is_none() || relaxed_base.is_none() || f_val_base != relaxed_base {
        return false;
    }

    // Run CCS-specific sum-check verifier for chained soundness
    if !sumcheck_msgs.is_empty() {
        let domain_size = (structure.max_deg + 1).next_power_of_two().max(1) * 4;
        let mut oracle = FriOracle::new_for_verifier(domain_size);
        let mut transcript = vec![];
        if ccs_sumcheck_verifier(
            structure,
            ExtF::ZERO,
            sumcheck_msgs,
            eval.norm_bound,
            &[],
            &mut oracle,
            &mut transcript,
        )
        .is_none()
        {
            return false;
        }
    }

    verify_open(structure, committer, eval)
}

pub fn verify_rlc(
    e1: &EvalInstance,
    e2: &EvalInstance,
    rho_rot: &RingElement<ModInt>,
    new_eval: &EvalInstance,
    committer: &AjtaiCommitter,
) -> bool {
    if e1.r != e2.r || e1.r != new_eval.r {
        return false;
    }
    if e1.ys.len() != e2.ys.len() || e1.ys.len() != new_eval.ys.len() {
        return false;
    }
    // Derive the exact same base-field scalar the prover used from the rotation element.
    let rho_scalar = rho_scalar_from_rotation(rho_rot);
    for ((&y1, &y2), &y_new) in e1.ys.iter().zip(&e2.ys).zip(&new_eval.ys) {
        if y_new != y1 + rho_scalar * y2 {
            return false;
        }
    }
    if new_eval.u != e1.u + rho_scalar * e2.u {
        return false;
    }
    if new_eval.e_eval != e1.e_eval + rho_scalar * e2.e_eval {
        return false;
    }
    let expected_commitment =
        committer.random_linear_combo_rotation(&e1.commitment, &e2.commitment, rho_rot);
    if expected_commitment != new_eval.commitment {
        return false;
    }
    let expected_norm = e1.norm_bound.max(rho_rot.norm_inf() * e2.norm_bound);
    if new_eval.norm_bound < expected_norm {
        return false;
    }
    true
}

pub fn verify_dec(_committer: &AjtaiCommitter, e: &EvalInstance, new_eval: &EvalInstance) -> bool {
    if e.r != new_eval.r || e.ys != new_eval.ys || e.u != new_eval.u || e.e_eval != new_eval.e_eval
    {
        return false;
    }
    true
}

pub fn open_me(_committer: &AjtaiCommitter, _eval: &EvalInstance, z: &[F]) -> Vec<F> {
    z.to_owned()
}

pub fn verify_open(
    structure: &CcsStructure,
    _committer: &AjtaiCommitter,
    eval: &EvalInstance,
) -> bool {
    if eval.ys.len() != structure.mats.len() {
        return false;
    }
    for &y in &eval.ys {
        if project_ext_to_base(y).is_none() {
            return false;
        }
    }
    if project_ext_to_base(eval.e_eval).is_none() || project_ext_to_base(eval.u).is_none() {
        return false;
    }
    let f_val = structure.f.evaluate(&eval.ys);
    // Relaxed CCS (single slack): f(M z) == u * e
    let relaxed = eval.u * eval.e_eval;
    project_ext_to_base(f_val) == project_ext_to_base(relaxed)
}

impl FoldState {
    pub fn fri_compress_final(&self) -> Result<(FriCommitment, FriProof), String> {
        let final_eval = self
            .eval_instances
            .last()
            .ok_or_else(|| "No instances".to_string())?;
        let final_poly = Polynomial::new(final_eval.ys.clone());
        let mut transcript_clone = self.transcript.clone();
        transcript_clone.extend(b"final_poly_hash");
        let mut oracle = FriOracle::new(vec![final_poly], &mut transcript_clone);
        let commit = oracle.commit()[0].clone();
        let point = final_eval.r.clone();
        let (evals, proofs) = oracle.open_at_point(&point);
        let proof = proofs[0].clone();
        if evals[0] != final_eval.e_eval {
            return Err("Eval mismatch".to_string());
        }
        Ok((commit, proof))
    }

    pub fn fri_verify_compressed(
        commit: &FriCommitment,
        proof: &FriProof,
        point: &[ExtF],
        claimed_eval: ExtF,
        coeff_len: usize,
    ) -> bool {
        let domain_size = coeff_len.next_power_of_two() * 4;
        let verifier = FriOracle::new_for_verifier(domain_size);
        let commitments = [commit.clone()];
        let evals = [claimed_eval];
        let proofs = [proof.clone()];
        verifier.verify_openings(&commitments, point, &evals, &proofs)
    }
}

fn read_tag(cursor: &mut Cursor<&[u8]>, expected: &[u8]) -> Result<(), ()> {
    let mut buf = vec![0u8; expected.len()];
    if cursor.read_exact(&mut buf).is_err() || &buf != expected {
        Err(())
    } else {
        Ok(())
    }
}

fn serialize_poly(poly: &Polynomial<ExtF>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for coeff in poly.coeffs() {
        let arr = coeff.to_array();
        bytes.extend(arr[0].as_canonical_u64().to_be_bytes());
        bytes.extend(arr[1].as_canonical_u64().to_be_bytes());
    }
    bytes
}

fn serialize_commit(commit: &[RingElement<ModInt>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.push(commit.len() as u8);
    for elem in commit {
        for c in elem.coeffs() {
            bytes.extend(c.as_canonical_u64().to_be_bytes());
        }
    }
    bytes
}

fn serialize_sumcheck_msgs(transcript: &mut Vec<u8>, msgs: &[(Polynomial<ExtF>, ExtF)]) {
    transcript.write_u8(msgs.len() as u8).unwrap();
    for (poly, eval) in msgs {
        transcript.write_u8(poly.degree() as u8).unwrap();
        transcript.extend(serialize_poly(poly));
        let arr = eval.to_array();
        transcript.extend(arr[0].as_canonical_u64().to_be_bytes());
        transcript.extend(arr[1].as_canonical_u64().to_be_bytes());
    }
}

fn serialize_ys(transcript: &mut Vec<u8>, ys: &[ExtF]) {
    transcript.write_u8(ys.len() as u8).unwrap();
    for &y in ys {
        let arr = y.to_array();
        transcript.extend(arr[0].as_canonical_u64().to_be_bytes());
        transcript.extend(arr[1].as_canonical_u64().to_be_bytes());
    }
}

fn serialize_comms_block(transcript: &mut Vec<u8>, comms: &[Vec<u8>]) {
    // Write number of commitments (u8), then for each write length (u32) + bytes
    let len = comms.len() as u8;
    transcript.push(len);
    for c in comms {
        let clen = c.len() as u32;
        transcript.extend_from_slice(&clen.to_be_bytes());
        transcript.extend_from_slice(c);
    }
}

fn serialize_fri_commit(commit: &FriCommitment) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes
        .write_u32::<BigEndian>(commit.len() as u32)
        .unwrap();
    bytes.extend(commit);
    bytes
}

fn serialize_fri_proof(proof: &FriProof) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes
        .write_u32::<BigEndian>(proof.len() as u32)
        .unwrap();
    bytes.extend(proof);
    bytes
}

fn extract_msgs_ccs(cursor: &mut Cursor<&[u8]>, _max_deg: usize) -> Vec<(Polynomial<ExtF>, ExtF)> {
    let len = cursor.read_u8().unwrap_or(0) as usize;
    let mut msgs = Vec::new();
    for _ in 0..len {
        let deg = cursor.read_u8().unwrap_or(0) as usize;
        let mut coeffs = Vec::new();
        for _ in 0..=deg {
            let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap_or(0));
            let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap_or(0));
            coeffs.push(ExtF::new_complex(real, imag));
        }
        let poly = Polynomial::new(coeffs);
        let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap_or(0));
        let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap_or(0));
        let blind = ExtF::new_complex(real, imag);
        msgs.push((poly, blind));
    }
    msgs
}

#[allow(dead_code)]
fn read_f(cursor: &mut Cursor<&[u8]>) -> F {
    F::from_u64(cursor.read_u64::<BigEndian>().unwrap_or(0))
}

fn serialize_rotation(rot: &RingElement<ModInt>, transcript: &mut Vec<u8>) {
    // Write length (n) followed by n limbs as u64 (canonical)
    transcript.extend(&(rot.coeffs().len() as u32).to_be_bytes());
    for c in rot.coeffs() {
        transcript.extend(&c.as_canonical_u64().to_be_bytes());
    }
}

fn read_rotation(cursor: &mut Cursor<&[u8]>, n: usize) -> Option<RingElement<ModInt>> {
    let len = cursor.read_u32::<BigEndian>().ok()? as usize;
    if len != n {
        return None;
    }
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let limb = cursor.read_u64::<BigEndian>().ok()?;
        coeffs.push(ModInt::from_u64(limb));
    }
    Some(RingElement::from_coeffs(coeffs, n))
}

/// Derive an extension-field scalar ρ from a rotation element using Fiat–Shamir.
/// This hashes the coefficients of the rotation so that both prover and verifier
/// obtain the same scalar for use in RLC over evaluations (`ys`, `u`, `e_eval`).
/// **Do not** use this scalar to scale commitment norms; use `rot.norm_inf()`
/// for that instead.
pub fn rho_scalar_from_rotation(rot: &RingElement<ModInt>) -> ExtF {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"rho_ext_from_rotation");
    bytes.extend(&(rot.coeffs().len() as u32).to_be_bytes());
    for c in rot.coeffs() {
        bytes.extend(c.as_canonical_u64().to_be_bytes());
    }
    fiat_shamir_challenge(&bytes)
}

fn read_ys(cursor: &mut Cursor<&[u8]>, expected_len: usize) -> Option<Vec<ExtF>> {
    let len = cursor.read_u8().ok()? as usize;
    if len != expected_len {
        return None;
    }
    let mut ys = Vec::with_capacity(len);
    for _ in 0..len {
        let real = F::from_u64(cursor.read_u64::<BigEndian>().ok()?);
        let imag = F::from_u64(cursor.read_u64::<BigEndian>().ok()?);
        ys.push(ExtF::new_complex(real, imag));
    }
    Some(ys)
}

fn read_comms_block(cursor: &mut Cursor<&[u8]>) -> Option<Vec<Vec<u8>>> {
    let len = cursor.read_u8().ok()? as usize;
    let mut comms = Vec::with_capacity(len);
    for _ in 0..len {
        let clen = cursor.read_u32::<BigEndian>().ok()? as usize;
        let mut buf = vec![0u8; clen];
        cursor.read_exact(&mut buf).ok()?;
        comms.push(buf);
    }
    Some(comms)
}

fn read_fri_commit(cursor: &mut Cursor<&[u8]>) -> FriCommitment {
    let len = cursor.read_u32::<BigEndian>().unwrap_or(0) as usize;
    let mut buf = vec![0u8; len];
    let _ = cursor.read_exact(&mut buf);
    buf
}

fn read_fri_proof(cursor: &mut Cursor<&[u8]>) -> FriProof {
    let len = cursor.read_u32::<BigEndian>().unwrap_or(0) as usize;
    let mut buf = vec![0u8; len];
    let _ = cursor.read_exact(&mut buf);
    buf
}

fn read_commit(cursor: &mut Cursor<&[u8]>, n: usize) -> Vec<RingElement<ModInt>> {
    let len = cursor.read_u8().unwrap_or(0) as usize;
    let mut commit = Vec::new();
    for _ in 0..len {
        let mut coeffs = Vec::new();
        for _ in 0..n {
            let val = cursor.read_u64::<BigEndian>().unwrap_or(0);
            coeffs.push(ModInt::from_u64(val));
        }
        commit.push(RingElement::from_coeffs(coeffs, n));
    }
    commit
}
