//! Output Sumcheck for binding program outputs to proofs.
//!
//! Proves: Σ_k eq(r_addr, k) · io_mask(k) · (Val_final(k) − Val_io(k)) = 0
//!
//! Security: Verifier enforces claim == 0, transcript binds outputs before challenges.

use crate::mle::build_chi_table;
use neo_math::{from_complex, KExtensions, K};
use neo_reductions::sumcheck::RoundOracle;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{Field, PrimeCharacteristicRing};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ============================================================================
// Constants & Transcript Helpers
// ============================================================================

const MAX_NUM_BITS: usize = 30;

fn sample_k_challenge(tr: &mut Poseidon2Transcript) -> K {
    from_complex(
        tr.challenge_field(b"output_check/chal/re"),
        tr.challenge_field(b"output_check/chal/im"),
    )
}

fn absorb_k(tr: &mut Poseidon2Transcript, label: &'static [u8], k: K) {
    tr.append_fields(label, &k.as_coeffs());
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputCheckError {
    AddressOutOfDomain {
        addr: u64,
        max_addr: u64,
    },
    DuplicateAddress {
        addr: u64,
    },
    NonZeroClaim,
    RoundCheckFailed {
        round: usize,
        message: String,
    },
    External(String),
    FinalClaimMismatch {
        got: K,
        expected: K,
    },
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    NumBitsTooLarge {
        num_bits: usize,
        max: usize,
    },
    WrongDegree {
        round: usize,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for OutputCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AddressOutOfDomain { addr, max_addr } => write!(f, "Address {} >= {}", addr, max_addr),
            Self::DuplicateAddress { addr } => write!(f, "Duplicate address {}", addr),
            Self::NonZeroClaim => write!(f, "Non-zero claim"),
            Self::RoundCheckFailed { round, message } => write!(f, "Round {}: {}", round, message),
            Self::External(msg) => write!(f, "External: {}", msg),
            Self::FinalClaimMismatch { got, expected } => write!(f, "Final: {:?} != {:?}", got, expected),
            Self::DimensionMismatch { expected, got } => write!(f, "Dim: {} != {}", expected, got),
            Self::NumBitsTooLarge { num_bits, max } => write!(f, "num_bits {} > {}", num_bits, max),
            Self::WrongDegree { round, expected, got } => write!(f, "Round {} degree {} != {}", round, got, expected),
        }
    }
}

impl std::error::Error for OutputCheckError {}

// ============================================================================
// Program I/O
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgramIO<F> {
    claims: BTreeMap<u64, F>,
}

impl<F: Field> Default for ProgramIO<F> {
    fn default() -> Self {
        Self {
            claims: BTreeMap::new(),
        }
    }
}

impl<F: Field> ProgramIO<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_claim(mut self, addr: u64, value: F) -> Self {
        self.claims.insert(addr, value);
        self
    }

    pub fn with_output(self, addr: u64, value: F) -> Self {
        self.with_claim(addr, value)
    }
    pub fn with_input(self, addr: u64, value: F) -> Self {
        self.with_claim(addr, value)
    }

    pub fn try_with_claim(mut self, addr: u64, value: F) -> Result<Self, OutputCheckError> {
        if self.claims.contains_key(&addr) {
            return Err(OutputCheckError::DuplicateAddress { addr });
        }
        self.claims.insert(addr, value);
        Ok(self)
    }

    pub fn claimed_addresses(&self) -> impl Iterator<Item = u64> + '_ {
        self.claims.keys().copied()
    }
    pub fn get_claim(&self, addr: u64) -> Option<F> {
        self.claims.get(&addr).copied()
    }
    pub fn num_claims(&self) -> usize {
        self.claims.len()
    }
    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }
    pub fn claims(&self) -> impl Iterator<Item = (u64, F)> + '_ {
        self.claims.iter().map(|(&a, &v)| (a, v))
    }

    pub fn validate(&self, num_bits: usize) -> Result<(), OutputCheckError> {
        if num_bits > MAX_NUM_BITS {
            return Err(OutputCheckError::NumBitsTooLarge {
                num_bits,
                max: MAX_NUM_BITS,
            });
        }
        let max_addr = 1u64 << num_bits;
        for &addr in self.claims.keys() {
            if addr >= max_addr {
                return Err(OutputCheckError::AddressOutOfDomain { addr, max_addr });
            }
        }
        Ok(())
    }

    pub fn absorb_into_transcript(&self, tr: &mut Poseidon2Transcript)
    where
        F: Into<neo_math::F> + Copy,
    {
        tr.append_message(b"output_check/num_claims", &(self.claims.len() as u64).to_le_bytes());
        for (&addr, &value) in &self.claims {
            tr.append_message(b"output_check/addr", &addr.to_le_bytes());
            tr.append_fields(b"output_check/value", &[value.into()]);
        }
    }
}

// ============================================================================
// Core Polynomial Helpers (inline, no structs)
// ============================================================================

/// χ_r(k) = Π_i (r[i] if k_i else 1-r[i])
fn chi_at_point(r: &[K], k: u64, num_bits: usize) -> K {
    (0..num_bits).fold(K::ONE, |acc, i| {
        let bit = ((k >> i) & 1) == 1;
        acc * if bit { r[i] } else { K::ONE - r[i] }
    })
}

/// eq(a, b) = Π_i (a_i·b_i + (1-a_i)(1-b_i))
fn eq_points(a: &[K], b: &[K]) -> K {
    a.iter().zip(b).fold(K::ONE, |acc, (ai, bi)| {
        acc * (*ai * *bi + (K::ONE - *ai) * (K::ONE - *bi))
    })
}

/// Evaluate polynomial from coefficients using Horner's method
fn eval_poly(coeffs: &[K], x: K) -> K {
    coeffs.iter().rev().fold(K::ZERO, |acc, &c| acc * x + c)
}

/// Lagrange interpolation → coefficients
fn interpolate(xs: &[K], ys: &[K]) -> Vec<K> {
    let n = xs.len();
    let mut coeffs = vec![K::ZERO; n];
    for i in 0..n {
        let mut numer = vec![K::ZERO; n];
        numer[0] = K::ONE;
        let mut cur_deg = 0;
        let mut denom = K::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut next = vec![K::ZERO; n];
            for d in 0..=cur_deg {
                next[d + 1] += numer[d];
                next[d] -= xs[j] * numer[d];
            }
            numer = next;
            cur_deg += 1;
            denom *= xs[i] - xs[j];
        }
        let scale = ys[i] * denom.inv();
        for d in 0..n {
            coeffs[d] += scale * numer[d];
        }
    }
    coeffs
}

/// Sparse evaluation of I/O mask: Σ_{addr in claimed} eq(r, addr)
fn eval_io_mask<F: Field>(program_io: &ProgramIO<F>, r: &[K], num_bits: usize) -> K {
    program_io
        .claimed_addresses()
        .map(|addr| chi_at_point(r, addr, num_bits))
        .sum()
}

/// Sparse evaluation of claimed values: Σ_k claimed[k] · χ_r(k)
fn eval_claimed_io<F: Field + Into<K>>(program_io: &ProgramIO<F>, r: &[K], num_bits: usize) -> K {
    program_io
        .claims()
        .map(|(addr, val)| {
            let val_k: K = val.into();
            val_k * chi_at_point(r, addr, num_bits)
        })
        .sum()
}

// ============================================================================
// Sumcheck Parameters
// ============================================================================

#[derive(Clone, Debug)]
pub struct OutputSumcheckParams<F> {
    pub k: usize,
    pub num_bits: usize,
    pub r_addr: Vec<K>,
    pub program_io: ProgramIO<F>,
}

impl<F: Field + Into<neo_math::F> + Copy> OutputSumcheckParams<F> {
    pub fn sample_from_transcript(
        tr: &mut Poseidon2Transcript,
        num_bits: usize,
        program_io: ProgramIO<F>,
    ) -> Result<Self, OutputCheckError> {
        program_io.validate(num_bits)?;
        program_io.absorb_into_transcript(tr);
        let r_addr = (0..num_bits)
            .map(|i| {
                tr.append_message(b"output_check/r_addr/idx", &(i as u64).to_le_bytes());
                sample_k_challenge(tr)
            })
            .collect();
        Ok(Self {
            k: 1 << num_bits,
            num_bits,
            r_addr,
            program_io,
        })
    }

    /// For testing only - use sample_from_transcript in production
    pub fn new_for_testing(
        num_bits: usize,
        r_addr: Vec<K>,
        program_io: ProgramIO<F>,
    ) -> Result<Self, OutputCheckError> {
        program_io.validate(num_bits)?;
        Ok(Self {
            k: 1 << num_bits,
            num_bits,
            r_addr,
            program_io,
        })
    }

    pub fn num_rounds(&self) -> usize {
        self.num_bits
    }
    pub fn degree_bound(&self) -> usize {
        3
    }
}

// ============================================================================
// Output Sumcheck Prover
// ============================================================================

pub struct OutputSumcheckProver<F> {
    eq_table: Vec<K>,
    io_mask_table: Vec<K>,
    val_final_table: Vec<K>,
    val_io_table: Vec<K>,
    rounds_remaining: usize,
    challenges: Vec<K>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field + Into<K> + Clone> OutputSumcheckProver<F> {
    pub fn new(params: OutputSumcheckParams<F>, final_memory_state: &[F]) -> Result<Self, OutputCheckError> {
        let (num_bits, k) = (params.num_bits, params.k);
        if final_memory_state.len() != k {
            return Err(OutputCheckError::DimensionMismatch {
                expected: k,
                got: final_memory_state.len(),
            });
        }

        let eq_table = build_chi_table(&params.r_addr);

        // Build sparse mask table
        let mut io_mask_table = vec![K::ZERO; k];
        for addr in params.program_io.claimed_addresses() {
            if (addr as usize) < k {
                io_mask_table[addr as usize] = K::ONE;
            }
        }

        let val_final_table: Vec<K> = final_memory_state.iter().map(|v| (*v).into()).collect();

        // Build sparse val_io table
        let mut val_io_table = vec![K::ZERO; k];
        for (addr, val) in params.program_io.claims() {
            if (addr as usize) < k {
                val_io_table[addr as usize] = val.into();
            }
        }

        Ok(Self {
            eq_table,
            io_mask_table,
            val_final_table,
            val_io_table,
            rounds_remaining: num_bits,
            challenges: Vec::with_capacity(num_bits),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn compute_claim(&self) -> K {
        (0..self.eq_table.len())
            .map(|i| self.eq_table[i] * self.io_mask_table[i] * (self.val_final_table[i] - self.val_io_table[i]))
            .sum()
    }

    pub fn challenges(&self) -> &[K] {
        &self.challenges
    }
}

#[derive(Clone, Debug)]
pub struct OutputSumcheckVerifier<F> {
    params: OutputSumcheckParams<F>,
}

impl<F: Field + Into<K> + Into<neo_math::F> + Copy> OutputSumcheckVerifier<F> {
    pub fn new(params: OutputSumcheckParams<F>) -> Self {
        Self { params }
    }

    pub fn verify(
        &self,
        proof: &OutputSumcheckProof,
        val_final_at_r_prime: K,
        challenges: &[K],
    ) -> Result<(), OutputCheckError> {
        if proof.round_polys.len() != challenges.len() {
            return Err(OutputCheckError::DimensionMismatch {
                expected: proof.round_polys.len(),
                got: challenges.len(),
            });
        }
        if challenges.len() != self.params.num_rounds() {
            return Err(OutputCheckError::DimensionMismatch {
                expected: self.params.num_rounds(),
                got: challenges.len(),
            });
        }

        let final_claim = verify_sumcheck_rounds_no_transcript(
            &proof.round_polys,
            K::ZERO,
            self.params.degree_bound() + 1,
            challenges,
        )?;

        let eq_eval = eq_points(&self.params.r_addr, challenges);
        let io_mask_eval = eval_io_mask(&self.params.program_io, challenges, self.params.num_bits);
        let val_io_eval = eval_claimed_io(&self.params.program_io, challenges, self.params.num_bits);
        let expected = eq_eval * io_mask_eval * (val_final_at_r_prime - val_io_eval);

        if final_claim != expected {
            return Err(OutputCheckError::FinalClaimMismatch {
                got: final_claim,
                expected,
            });
        }

        Ok(())
    }
}

impl<F: Field + Into<K> + Clone> RoundOracle for OutputSumcheckProver<F> {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let n = self.eq_table.len();
        if n == 1 {
            let term = self.eq_table[0] * self.io_mask_table[0] * (self.val_final_table[0] - self.val_io_table[0]);
            return vec![term; points.len()];
        }
        let half = n / 2;

        // LSB-first binding: pair consecutive entries (2*i, 2*i+1).
        points
            .iter()
            .map(|&x| {
                (0..half)
                    .map(|i| {
                        let idx0 = 2 * i;
                        let idx1 = idx0 + 1;

                        let eq0 = self.eq_table[idx0];
                        let eq1 = self.eq_table[idx1];
                        let mask0 = self.io_mask_table[idx0];
                        let mask1 = self.io_mask_table[idx1];
                        let vf0 = self.val_final_table[idx0];
                        let vf1 = self.val_final_table[idx1];
                        let vio0 = self.val_io_table[idx0];
                        let vio1 = self.val_io_table[idx1];

                        let eq_x = eq0 + (eq1 - eq0) * x;
                        let mask_x = mask0 + (mask1 - mask0) * x;
                        let vf_x = vf0 + (vf1 - vf0) * x;
                        let vio_x = vio0 + (vio1 - vio0) * x;

                        eq_x * mask_x * (vf_x - vio_x)
                    })
                    .sum()
            })
            .collect()
    }

    fn num_rounds(&self) -> usize {
        self.rounds_remaining
    }
    fn degree_bound(&self) -> usize {
        3
    }

    fn fold(&mut self, r: K) {
        let n = self.eq_table.len();
        if n <= 1 {
            return;
        }

        // LSB-first fold: pair consecutive entries (2*i, 2*i+1).
        let half = n / 2;
        for i in 0..half {
            let idx0 = 2 * i;
            let idx1 = idx0 + 1;

            let eq0 = self.eq_table[idx0];
            let eq1 = self.eq_table[idx1];
            let mask0 = self.io_mask_table[idx0];
            let mask1 = self.io_mask_table[idx1];
            let vf0 = self.val_final_table[idx0];
            let vf1 = self.val_final_table[idx1];
            let vio0 = self.val_io_table[idx0];
            let vio1 = self.val_io_table[idx1];

            self.eq_table[i] = eq0 + (eq1 - eq0) * r;
            self.io_mask_table[i] = mask0 + (mask1 - mask0) * r;
            self.val_final_table[i] = vf0 + (vf1 - vf0) * r;
            self.val_io_table[i] = vio0 + (vio1 - vio0) * r;
        }
        self.eq_table.truncate(half);
        self.io_mask_table.truncate(half);
        self.val_final_table.truncate(half);
        self.val_io_table.truncate(half);
        self.challenges.push(r);
        self.rounds_remaining = self.rounds_remaining.saturating_sub(1);
    }
}

// ============================================================================
// Proof Structures
// ============================================================================

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputSumcheckProof {
    pub round_polys: Vec<Vec<K>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputBindingProof {
    pub output_sc: OutputSumcheckProof,
}

// ============================================================================
// Core Verification Logic (shared by all verifiers)
// ============================================================================

/// Verify sumcheck rounds, returning (final_claim, challenges).
/// initial_claim is enforced (not trusted from proof).
fn verify_sumcheck_rounds(
    tr: &mut Poseidon2Transcript,
    round_polys: &[Vec<K>],
    initial_claim: K,
    expected_num_coeffs: usize,
    coeff_label: &'static [u8],
) -> Result<(K, Vec<K>), OutputCheckError> {
    let mut current_claim = initial_claim;
    let mut challenges = Vec::with_capacity(round_polys.len());

    for (round, coeffs) in round_polys.iter().enumerate() {
        if coeffs.len() != expected_num_coeffs {
            return Err(OutputCheckError::WrongDegree {
                round,
                expected: expected_num_coeffs,
                got: coeffs.len(),
            });
        }
        let (p_0, p_1) = (eval_poly(coeffs, K::ZERO), eval_poly(coeffs, K::ONE));
        if p_0 + p_1 != current_claim {
            return Err(OutputCheckError::RoundCheckFailed {
                round,
                message: format!("p(0)+p(1)={:?}, expected {:?}", p_0 + p_1, current_claim),
            });
        }
        for &c in coeffs {
            absorb_k(tr, coeff_label, c);
        }
        let r = sample_k_challenge(tr);
        challenges.push(r);
        current_claim = eval_poly(coeffs, r);
    }

    Ok((current_claim, challenges))
}

fn verify_sumcheck_rounds_no_transcript(
    round_polys: &[Vec<K>],
    initial_claim: K,
    expected_num_coeffs: usize,
    challenges: &[K],
) -> Result<K, OutputCheckError> {
    if round_polys.len() != challenges.len() {
        return Err(OutputCheckError::DimensionMismatch {
            expected: round_polys.len(),
            got: challenges.len(),
        });
    }

    let mut current_claim = initial_claim;
    for (round, (coeffs, &r)) in round_polys.iter().zip(challenges.iter()).enumerate() {
        if coeffs.len() != expected_num_coeffs {
            return Err(OutputCheckError::WrongDegree {
                round,
                expected: expected_num_coeffs,
                got: coeffs.len(),
            });
        }

        let (p_0, p_1) = (eval_poly(coeffs, K::ZERO), eval_poly(coeffs, K::ONE));
        if p_0 + p_1 != current_claim {
            return Err(OutputCheckError::RoundCheckFailed {
                round,
                message: format!("p(0)+p(1)={:?}, expected {:?}", p_0 + p_1, current_claim),
            });
        }

        current_claim = eval_poly(coeffs, r);
    }

    Ok(current_claim)
}

// ============================================================================
// Proof Generation
// ============================================================================

fn run_sumcheck_prover<O: RoundOracle>(
    tr: &mut Poseidon2Transcript,
    oracle: &mut O,
    coeff_label: &'static [u8],
) -> Vec<Vec<K>> {
    let num_rounds = oracle.num_rounds();
    let eval_points: Vec<K> = (0..=oracle.degree_bound())
        .map(|i| K::from_u64(i as u64))
        .collect();
    let mut round_polys = Vec::with_capacity(num_rounds);

    for _ in 0..num_rounds {
        let coeffs = interpolate(&eval_points, &oracle.evals_at(&eval_points));
        for &c in &coeffs {
            absorb_k(tr, coeff_label, c);
        }
        round_polys.push(coeffs);
        oracle.fold(sample_k_challenge(tr));
    }
    round_polys
}

/// Generate an output sumcheck proof and return the sampled `r'` challenges.
pub fn generate_output_sumcheck_proof_and_challenges<F: Field + Into<K> + Into<neo_math::F> + Clone + Copy>(
    tr: &mut Poseidon2Transcript,
    num_bits: usize,
    program_io: ProgramIO<F>,
    final_memory_state: &[F],
) -> Result<(OutputSumcheckProof, Vec<K>), OutputCheckError> {
    let params = OutputSumcheckParams::sample_from_transcript(tr, num_bits, program_io)?;
    let mut prover = OutputSumcheckProver::new(params, final_memory_state)?;
    let round_polys = run_sumcheck_prover(tr, &mut prover, b"output_check/round_coeff");
    let r_prime = prover.challenges().to_vec();
    Ok((OutputSumcheckProof { round_polys }, r_prime))
}

// ============================================================================
// Proof Verification
// ============================================================================

#[derive(Clone, Debug)]
pub struct OutputSumcheckState {
    pub r_prime: Vec<K>,
    pub output_final: K,
    pub eq_eval: K,
    pub io_mask_eval: K,
    pub val_io_eval: K,
}

/// Verify output sumcheck rounds and return state needed for an external final check.
pub fn verify_output_sumcheck_rounds_get_state<F: Field + Into<K> + Into<neo_math::F> + Copy>(
    tr: &mut Poseidon2Transcript,
    num_bits: usize,
    program_io: ProgramIO<F>,
    proof: &OutputSumcheckProof,
) -> Result<OutputSumcheckState, OutputCheckError> {
    let params = OutputSumcheckParams::sample_from_transcript(tr, num_bits, program_io)?;
    let (output_final, r_prime) = verify_sumcheck_rounds(
        tr,
        &proof.round_polys,
        K::ZERO,
        params.degree_bound() + 1,
        b"output_check/round_coeff",
    )?;

    let eq_eval = eq_points(&params.r_addr, &r_prime);
    let io_mask_eval = eval_io_mask(&params.program_io, &r_prime, params.num_bits);
    let val_io_eval = eval_claimed_io(&params.program_io, &r_prime, params.num_bits);

    Ok(OutputSumcheckState {
        r_prime,
        output_final,
        eq_eval,
        io_mask_eval,
        val_io_eval,
    })
}

/// Verify standalone output sumcheck proof (without Twist binding).
/// Use this when Val_final opening is provided externally.
pub fn verify_output_sumcheck<F: Field + Into<K> + Into<neo_math::F> + Clone + Copy>(
    tr: &mut Poseidon2Transcript,
    num_bits: usize,
    program_io: ProgramIO<F>,
    val_final_at_r_prime: K,
    proof: &OutputSumcheckProof,
) -> Result<(), OutputCheckError> {
    let params = OutputSumcheckParams::sample_from_transcript(tr, num_bits, program_io)?;

    let (final_claim, r_prime) = verify_sumcheck_rounds(
        tr,
        &proof.round_polys,
        K::ZERO,
        params.degree_bound() + 1,
        b"output_check/round_coeff",
    )?;

    let eq_eval = eq_points(&params.r_addr, &r_prime);
    let io_mask_eval = eval_io_mask(&params.program_io, &r_prime, num_bits);
    let val_io_eval = eval_claimed_io(&params.program_io, &r_prime, num_bits);
    let expected = eq_eval * io_mask_eval * (val_final_at_r_prime - val_io_eval);

    if final_claim != expected {
        return Err(OutputCheckError::FinalClaimMismatch {
            got: final_claim,
            expected,
        });
    }

    Ok(())
}

/// Generate standalone output sumcheck proof (for testing or external Val_final).
pub fn generate_output_sumcheck_proof<F: Field + Into<K> + Into<neo_math::F> + Clone + Copy>(
    tr: &mut Poseidon2Transcript,
    num_bits: usize,
    program_io: ProgramIO<F>,
    final_memory_state: &[F],
) -> Result<OutputSumcheckProof, OutputCheckError> {
    let params = OutputSumcheckParams::sample_from_transcript(tr, num_bits, program_io)?;
    let mut prover = OutputSumcheckProver::new(params, final_memory_state)?;
    let round_polys = run_sumcheck_prover(tr, &mut prover, b"output_check/round_coeff");
    Ok(OutputSumcheckProof { round_polys })
}
