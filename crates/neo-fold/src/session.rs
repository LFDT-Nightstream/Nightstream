//! Thin, ergonomic session layer for Π-CCS folding.
//!
//! This module provides:
//!   1) A small adapter trait (`NeoStep`) and `prove_step` for Nova/Sonobe-style step synthesis.
//!   2) A direct IO path via `ProveInput` + `prove_step_from_io` (callers hand us (x, w)).
//!   3) A session driver (`FoldingSession`) that **hides** commitment mixers.
//!
//! Concepts (paper-aligned):
//! - Accumulator: the k ME(b, L) claims you carry between steps (the inputs to the next Π_CCS).
//! - k is never defaulted: if you supply Accumulator with k>0, we multi-fold; if not, k=1 (simple case).
//!
//! Notes:
//! - We default to Ajtai mixers internally (no frontend mixers required).
//! - If you want k>1, pass an explicit Accumulator via `with_initial_accumulator(...)`.
//!
//! This file contains only ergonomics. The formal pipeline Π_CCS → Π_RLC → Π_DEC
//! lives in `fold_many_prove/verify` (engine-agnostic).

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ajtai::{Commitment as Cmt, s_lincomb, s_mul, decomp_b, DecompStyle};
use neo_params::NeoParams;
use neo_math::{F, D, K};
use neo_math::ring::Rq as RqEl;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::folding::{self, CommitMixers, FoldRun};
use crate::pi_ccs::FoldingMode;
use crate::PiCcsError;
use neo_reductions::engines::utils;

/// Optional application-level "output claim".
/// (Not consumed by Π-CCS core yet; kept for API parity / future use.)
#[derive(Clone, Debug)]
pub struct OutputClaim<Ff> {
    pub tag: &'static [u8],
    pub expected: Ff,
}

/// Direct inputs for a single step when you don't want to implement `NeoStep`.
/// We'll compute the commitment and split (x | w) for you.
#[derive(Clone, Debug)]
pub struct ProveInput<'a> {
    pub ccs: &'a CcsStructure<F>,   // the circuit (must match witness length)
    pub public_input: &'a [F],      // x
    pub witness: &'a [F],           // w
    pub output_claims: &'a [OutputClaim<F>], // optional; recorded but not enforced here
}

/// Where special coordinates live inside the step witness `z`.
#[derive(Clone, Debug)]
pub struct StepSpec {
    /// Number of public state elements carried between steps (y).
    pub y_len: usize,
    /// Index of constant 1 inside z (optional informational).
    pub const1_index: usize,
    /// Indices in z carried out as next-step public state (y).
    pub y_step_indices: Vec<usize>,
    /// Optional: indices in z corresponding to app-level public inputs.
    pub app_input_indices: Option<Vec<usize>>,
    /// Number of public inputs `m_in` (prefix length in z).
    pub m_in: usize,
}

/// What a step must return for the session to run Π-CCS.
#[derive(Clone, Debug)]
pub struct StepArtifacts {
    pub ccs: CcsStructure<F>,
    /// Concrete witness vector for this step (length = m).
    pub witness: Vec<F>,
    /// App inputs you want logged/exposed (optional, informational).
    pub public_app_inputs: Vec<F>,
    pub spec: StepSpec,
}

/// User implements this for their program.
pub trait NeoStep {
    type ExternalInputs: Clone;

    /// How many public state elements are threaded between steps?
    fn state_len(&self) -> usize;

    /// Static metadata about where special elements live inside `z`.
    fn step_spec(&self) -> StepSpec;

    /// Produce the CCS structure and a concrete witness for this step.
    fn synthesize_step(
        &mut self,
        step_idx: usize,
        y_prev: &[F],
        inputs: &Self::ExternalInputs,
    ) -> StepArtifacts;
}

/// Decompose z ∈ F^m into base-b digits Z ∈ F^{D×m} (balanced, for correct modular recomposition).
///
/// Uses balanced decomposition to ensure z ≡ Σ Z[ρ,·]·b^ρ (mod p), which is required for
/// F (CCS constraints) to hold when the engine recomposes z from Z.
fn decompose_z_to_Z(params: &NeoParams, z: &[F]) -> Mat<F> {
    let d = D;
    let m = z.len();
    
    // Column-major digits of length d for each column, balanced so recomposition equals z mod p
    let digits_col_major = decomp_b(z, params.b, d, DecompStyle::Balanced);
    
    // Convert to row-major Mat<F> of shape d×m
    let mut row_major = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = digits_col_major[c * d + r];
        }
    }
    Mat::from_row_major(d, m, row_major)
}

/// Convert a rotation matrix rot(a) to the ring element a for S-action.
/// 
/// The first column of rot(a) contains cf(a) (coefficient form of the ring element a).
/// We extract those coefficients and use cf_inv to recover the ring element.
fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;
    
    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);
    
    // Extract the first column which contains cf(a)
    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    
    // Convert coefficient array to ring element
    cf_inv(coeffs)
}

/// Default Ajtai mixers (hidden internally).
fn default_mixers() -> CommitMixers<
    fn(&[Mat<F>], &[Cmt]) -> Cmt,
    fn(&[Cmt], u32) -> Cmt
> {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }
    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for i in 1..cs.len() {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, &cs[i]);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }
    CommitMixers { mix_rhos_commits, combine_b_pows }
}

/// An *Accumulator* is the k ME(b, L) claims carried between steps.
/// This is exactly the paper's ME(b, L)^k input vector for the next Π_CCS.
#[derive(Clone, Debug)]
pub struct Accumulator {
    pub me: Vec<MeInstance<Cmt, F, K>>,
    pub witnesses: Vec<Mat<F>>, // Z_i for each me[i]
}

impl Accumulator {
    /// Sanity checks: dimensions, common r, consistent m_in, witness shape, and y-vector padding.
    pub fn check(&self, params: &NeoParams, s: &CcsStructure<F>) -> Result<(), PiCcsError> {
        if self.me.len() != self.witnesses.len() {
            return Err(PiCcsError::InvalidInput("Accumulator: me.len() != witnesses.len()".into()));
        }
        if self.me.is_empty() {
            return Ok(());
        }
        // Dims for r length
        let dims = utils::build_dims_and_policy(params, s)?;
        let ell_n = dims.ell_n;
        let want_pad = 1usize << dims.ell_d;

        // Common r and m_in
        let r0 = &self.me[0].r;
        if r0.len() != ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "Accumulator: r length mismatch (expected ell_n={}, got {})", ell_n, r0.len()
            )));
        }
        let m_in0 = self.me[0].m_in;
        for (i, (m, Z)) in self.me.iter().zip(self.witnesses.iter()).enumerate() {
            if m.r.len() != ell_n {
                return Err(PiCcsError::InvalidInput(format!(
                    "Accumulator[{}]: r length mismatch (expected ell_n={}, got {})", i, ell_n, m.r.len()
                )));
            }
            if m.r != *r0 {
                return Err(PiCcsError::InvalidInput("Accumulator: all ME inputs must share the same r".into()));
            }
            if m.m_in != m_in0 {
                return Err(PiCcsError::InvalidInput("Accumulator: all ME inputs must share the same m_in".into()));
            }
            if Z.rows() != D || Z.cols() != s.m {
                return Err(PiCcsError::InvalidInput(format!(
                    "Accumulator[{}]: Z has shape {}x{}, expected {}x{}", i, Z.rows(), Z.cols(), D, s.m
                )));
            }
            if m.X.rows() != D || m.X.cols() != m.m_in {
                return Err(PiCcsError::InvalidInput("Accumulator: X dimension mismatch with m_in".into()));
            }
            // Validate y-vector shape: t rows, each padded to 2^{ell_d}
            if m.y.len() != s.t() || !m.y.iter().all(|row| row.len() == want_pad) {
                return Err(PiCcsError::InvalidInput(format!(
                    "Accumulator[{}]: y shape invalid; expected t={} rows padded to 2^{{ell_d}}={}",
                    i, s.t(), want_pad
                )));
            }
        }
        Ok(())
    }
}

/// Return the column indices of Z that must populate X, in order,
/// as dictated by the step spec.
fn indices_from_spec(spec: &StepSpec) -> Vec<usize> {
    let mut idx = Vec::with_capacity(spec.m_in);
    idx.push(spec.const1_index);
    idx.extend(&spec.y_step_indices);
    if let Some(app) = &spec.app_input_indices {
        idx.extend(app);
    }
    idx
}

/// Ergonomic helper: build an ME(b, L) instance from a raw witness z with **balanced** digits.
/// This is handy for constructing an explicit Accumulator.
///
/// - `z` is the full vector (x || w), length must equal `s.m`.
/// - `r` must have length `ell_n` (from dims).
/// - `m_in` is how many columns of Z to project into X (first m_in).
///
/// This function computes y-vectors from Z and r, and pads them to 2^{ell_d} to ensure
/// consistency with the protocol engine (which expects padded y-vectors).
pub fn me_from_z_balanced<Lm: SModuleHomomorphism<F, Cmt>>(
    params: &NeoParams,
    s: &CcsStructure<F>,  // should be identity-first
    l: &Lm,
    z: &[F],
    r: &[K],
    m_in: usize,
) -> Result<(MeInstance<Cmt, F, K>, Mat<F>), PiCcsError> {
    if z.len() != s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced: z length {} != CCS.m {}", z.len(), s.m
        )));
    }
    if m_in > s.m {
        return Err(PiCcsError::InvalidInput("me_from_z_balanced: m_in exceeds s.m".into()));
    }

    let dims = utils::build_dims_and_policy(params, s)?;
    if r.len() != dims.ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced: r length {} != ell_n {}", r.len(), dims.ell_n
        )));
    }
    let d_pad = 1usize << dims.ell_d;

    // Balanced Ajtai decomposition (row-major D×m), matching existing tests/tools.
    let d = D;
    let m = z.len();
    let z_digits = decomp_b(z, params.b, d, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m {
        for row in 0..d {
            row_major[row * m + col] = z_digits[col * d + row];
        }
    }
    let Z = Mat::from_row_major(d, m, row_major);
    let c = l.commit(&Z);

    // X := first m_in columns of Z
    let mut X = Mat::zero(d, m_in, F::ZERO);
    for rr in 0..d {
        for cc in 0..m_in {
            X[(rr, cc)] = Z[(rr, cc)];
        }
    }

    // Build χ_r(row) and v_j := M_j^T χ_r (K^m), then y_{(j)} := Z · v_j (pad to d_pad)
    let n_sz = 1usize << r.len();
    let mut chi_r = vec![K::ZERO; n_sz];
    for row in 0..n_sz {
        let mut w = K::ONE;
        for bit in 0..r.len() {
            let rb = r[bit];
            let is_one = ((row >> bit) & 1) == 1;
            w *= if is_one { rb } else { K::ONE - rb };
        }
        chi_r[row] = w;
    }

    let t = s.t();
    let mut vjs: Vec<Vec<K>> = Vec::with_capacity(t);
    for j in 0..t {
        let mut vj = vec![K::ZERO; s.m];
        // Only rows < s.n contribute (others are zero rows)
        for row in 0..s.n {
            let wr = chi_r[row];
            if wr == K::ZERO { continue; }
            for c in 0..s.m {
                vj[c] += K::from(s.matrices[j][(row, c)]) * wr;
            }
        }
        vjs.push(vj);
    }

    // y rows padded to 2^{ell_d}; y_scalars = base-b recomposition of first D digits
    let bF = F::from_u64(params.b as u64);
    let mut pow_b_f = vec![F::ONE; d];
    for t in 1..d { pow_b_f[t] = pow_b_f[t - 1] * bF; }
    let pow_b_k: Vec<K> = pow_b_f.iter().copied().map(K::from).collect();

    let mut y: Vec<Vec<K>> = Vec::with_capacity(t);
    let mut y_scalars: Vec<K> = Vec::with_capacity(t);
    for j in 0..t {
        let mut yj = vec![K::ZERO; d_pad];
        // first D digits
        for rho in 0..d {
            let mut acc = K::ZERO;
            for c in 0..s.m {
                acc += K::from(Z[(rho, c)]) * vjs[j][c];
            }
            yj[rho] = acc;
        }
        // higher positions remain zero

        let mut scalar = K::ZERO;
        for rho in 0..d { scalar += yj[rho] * pow_b_k[rho]; }
        y.push(yj);
        y_scalars.push(scalar);
    }

    let me = MeInstance::<Cmt, F, K> {
        c_step_coords: vec![], u_offset: 0, u_len: 0,
        c,
        X,
        r: r.to_vec(),
        y,
        y_scalars,
        m_in,
        fold_digest: [0u8; 32],
    };

    Ok((me, Z))
}

/// Same as `me_from_z_balanced`, but X is formed by selecting the given Z-column indices,
/// in that exact order (required for NC constraints to hold).
///
/// This variant is used when public inputs are not contiguous at the front of z,
/// as specified by StepSpec indices.
pub fn me_from_z_balanced_select<Lm: SModuleHomomorphism<F, Cmt>>(
    params: &NeoParams,
    s: &CcsStructure<F>,  // should be identity-first
    l: &Lm,
    z: &[F],
    r: &[K],
    x_col_indices: &[usize], // which columns of Z form X, in order
) -> Result<(MeInstance<Cmt, F, K>, Mat<F>), PiCcsError> {
    if z.len() != s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced_select: z length {} != CCS.m {}", z.len(), s.m
        )));
    }

    let dims = utils::build_dims_and_policy(params, s)?;
    if r.len() != dims.ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced_select: r length {} != ell_n {}", r.len(), dims.ell_n
        )));
    }
    let d_pad = 1usize << dims.ell_d;

    // Balanced Ajtai decomposition (row-major D×m), matching existing tests/tools.
    let d = D;
    let m = z.len();
    let z_digits = decomp_b(z, params.b, d, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m {
        for row in 0..d {
            row_major[row * m + col] = z_digits[col * d + row];
        }
    }
    let Z = Mat::from_row_major(d, m, row_major);
    let c = l.commit(&Z);

    // X := selected columns of Z (not the first m_in)
    let m_in = x_col_indices.len();
    let mut X = Mat::zero(d, m_in, F::ZERO);
    for (j, &col) in x_col_indices.iter().enumerate() {
        if col >= Z.cols() {
            return Err(PiCcsError::InvalidInput(format!(
                "X column index {} out of range (Z has {} cols)", col, Z.cols()
            )));
        }
        for rho in 0..d {
            X[(rho, j)] = Z[(rho, col)];
        }
    }

    // Debug assertion that X equals the projection of Z
    #[cfg(feature = "debug-logs")]
    {
        for (j, &col) in x_col_indices.iter().enumerate() {
            for rho in 0..d {
                debug_assert_eq!(X[(rho, j)], Z[(rho, col)], "X != Z[:, col]");
            }
        }
    }

    // Build χ_r(row) and v_j := M_j^T χ_r (K^m), then y_{(j)} := Z · v_j (pad to d_pad)
    let n_sz = 1usize << r.len();
    let mut chi_r = vec![K::ZERO; n_sz];
    for row in 0..n_sz {
        let mut w = K::ONE;
        for bit in 0..r.len() {
            let rb = r[bit];
            let is_one = ((row >> bit) & 1) == 1;
            w *= if is_one { rb } else { K::ONE - rb };
        }
        chi_r[row] = w;
    }

    let t = s.t();
    let mut vjs: Vec<Vec<K>> = Vec::with_capacity(t);
    for j in 0..t {
        let mut vj = vec![K::ZERO; s.m];
        // Only rows < s.n contribute (others are zero rows)
        for row in 0..s.n {
            let wr = chi_r[row];
            if wr == K::ZERO { continue; }
            for c in 0..s.m {
                vj[c] += K::from(s.matrices[j][(row, c)]) * wr;
            }
        }
        vjs.push(vj);
    }

    // y rows padded to 2^{ell_d}; y_scalars = base-b recomposition of first D digits
    let bF = F::from_u64(params.b as u64);
    let mut pow_b_f = vec![F::ONE; d];
    for t in 1..d { pow_b_f[t] = pow_b_f[t - 1] * bF; }
    let pow_b_k: Vec<K> = pow_b_f.iter().copied().map(K::from).collect();

    let mut y: Vec<Vec<K>> = Vec::with_capacity(t);
    let mut y_scalars: Vec<K> = Vec::with_capacity(t);
    for j in 0..t {
        let mut yj = vec![K::ZERO; d_pad];
        // first D digits
        for rho in 0..d {
            let mut acc = K::ZERO;
            for c in 0..s.m {
                acc += K::from(Z[(rho, c)]) * vjs[j][c];
            }
            yj[rho] = acc;
        }
        // higher positions remain zero

        let mut scalar = K::ZERO;
        for rho in 0..d { scalar += yj[rho] * pow_b_k[rho]; }
        y.push(yj);
        y_scalars.push(scalar);
    }

    let me = MeInstance::<Cmt, F, K> {
        c_step_coords: vec![], u_offset: 0, u_len: 0,
        c,
        X,
        r: r.to_vec(),
        y,
        y_scalars,
        m_in,
        fold_digest: [0u8; 32],
    };

    Ok((me, Z))
}

/// Minimal session that provides an ergonomic per-step API.
/// Mixers are hidden and default to Ajtai; the initial Accumulator is optional.
/// If absent, we run the *simple* k=1 case (no ME inputs).
pub struct FoldingSession<L>
where
    L: SModuleHomomorphism<F, Cmt> + Clone,
{
    mode: FoldingMode,
    params: NeoParams,
    l: L,
    mixers: CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>,

    // Collected MCS steps (instance + witness), in order
    mcss: Vec<(McsInstance<Cmt, F>, McsWitness<F>)>,

    // Optional initial accumulated ME(b, L)^k inputs (k = me.len()).
    acc0: Option<Accumulator>,

    // Optional: app-level claims recorded per step (not enforced here yet)
    step_claims: Vec<Vec<OutputClaim<F>>>,

    /// Current threaded state y (if any). Length is determined by `NeoStep::state_len()`.
    curr_state: Option<Vec<F>>,
}

impl<L> FoldingSession<L>
where
    L: SModuleHomomorphism<F, Cmt> + Clone,
{
    /// Create a new session with default Ajtai mixers and no initial accumulator (k=1 simple flow).
    pub fn new(
        mode: FoldingMode,
        params: NeoParams,
        l: L,
    ) -> Self {
        Self {
            mode,
            params,
            l,
            mixers: default_mixers(),
            mcss: vec![],
            acc0: None,
            step_claims: vec![],
            curr_state: None,
        }
    }

    /// Set an explicit initial state y₀ for the IVC (optional).
    /// If not set, y₀ defaults to all zeros of length `state_len()`.
    pub fn set_initial_state(&mut self, y0: Vec<F>) {
        self.curr_state = Some(y0);
    }

    /// Inject an explicit initial Accumulator (k = acc.me.len()). This enables k>1 multi-folding.
    pub fn with_initial_accumulator(mut self, acc: Accumulator, s: &CcsStructure<F>) -> Result<Self, PiCcsError> {
        let s_norm = s
            .ensure_identity_first()
            .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;

        // STRICT validation: Ajtai/NC requires M₀ = I_n (fail fast instead of mysterious sumcheck errors)
        s_norm
            .assert_m0_is_identity_for_nc()
            .map_err(|e| PiCcsError::InvalidInput(e.to_string()))?;

        acc.check(&self.params, &s_norm)?;
        self.acc0 = Some(acc);
        Ok(self)
    }

    /// Access the accumulated public MCS instances (for verification APIs).
    pub fn mcss_public(&self) -> Vec<McsInstance<Cmt, F>> {
        self.mcss.iter().map(|(i, _)| i.clone()).collect()
    }

    /// Add one step using the `NeoStep` synthesis adapter.
    /// This accumulates the step instance and witness without performing any folding.
    pub fn add_step<S: NeoStep>(
        &mut self,
        stepper: &mut S,
        inputs: &S::ExternalInputs,
    ) -> Result<(), PiCcsError> {
        // 1) Decide previous state y_prev
        let state_len = stepper.state_len();
        let y_prev = self
            .curr_state
            .clone()
            .unwrap_or_else(|| vec![F::ZERO; state_len]);

        // 2) Let the app synthesize CCS + witness given y_prev
        let StepArtifacts { ccs, witness: z, spec, public_app_inputs: _ } =
            stepper.synthesize_step(self.mcss.len(), &y_prev, inputs);

        // Safety: require state_len to match StepSpec
        if spec.y_len != state_len {
            return Err(PiCcsError::InvalidInput(format!(
                "StepSpec.y_len ({}) must equal stepper.state_len() ({})",
                spec.y_len, state_len
            )));
        }

        // Identity-first normalization
        let s_norm = ccs
            .ensure_identity_first()
            .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;

        // STRICT validation: Ajtai/NC requires M₀ = I_n (fail fast instead of mysterious sumcheck errors)
        s_norm
            .assert_m0_is_identity_for_nc()
            .map_err(|e| PiCcsError::InvalidInput(e.to_string()))?;

        // z must match CCS dimension
        if z.len() != s_norm.m {
            return Err(PiCcsError::InvalidInput(format!(
                "step witness length {} != CCS.m {}",
                z.len(), s_norm.m
            )));
        }
        if spec.m_in > z.len() {
            return Err(PiCcsError::InvalidInput("m_in exceeds witness length".into()));
        }

        // 3) Build MCS instance + witness as before
        let x_indices = indices_from_spec(&spec);
        
        if x_indices.len() != spec.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "StepSpec produced {} public-input indices, expected m_in={}",
                x_indices.len(),
                spec.m_in
            )));
        }
        
        // Validate uniqueness
        {
            use std::collections::BTreeSet;
            if x_indices.iter().copied().collect::<BTreeSet<_>>().len() != x_indices.len() {
                return Err(PiCcsError::InvalidInput(
                    "StepSpec indices contain duplicates".into(),
                ));
            }
        }
        
        // Validate range
        if let Some(&idx) = x_indices.iter().find(|&&i| i >= z.len()) {
            return Err(PiCcsError::InvalidInput(format!(
                "StepSpec index {} out of bounds (witness length {})",
                idx,
                z.len()
            )));
        }
        
        let x: Vec<F> = x_indices.iter().map(|&i| z[i]).collect();
        
        let Z = decompose_z_to_Z(&self.params, &z);
        let c = self.l.commit(&Z);
        let m_in = spec.m_in;
        
        // w is the private witness (suffix)
        let w = z[m_in..].to_vec();

        let mcs_inst = McsInstance { c, x, m_in };
        let mcs_wit = McsWitness { w, Z };

        self.mcss.push((mcs_inst, mcs_wit));
        self.step_claims.push(vec![]);

        // 4) Update current state y from the witness coordinates
        if spec.y_len > 0 {
            if spec.y_len > spec.y_step_indices.len() {
                return Err(PiCcsError::InvalidInput(
                    "StepSpec: y_len exceeds y_step_indices.len()".into(),
                ));
            }
            let mut new_state = Vec::with_capacity(spec.y_len);
            for &idx in spec.y_step_indices.iter().take(spec.y_len) {
                if idx >= z.len() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "StepSpec y_step_index {} out of bounds for witness of length {}",
                        idx, z.len()
                    )));
                }
                new_state.push(z[idx]);
            }
            self.curr_state = Some(new_state);
        }

        Ok(())
    }

    /// Add one step directly from (x, w) without implementing `NeoStep`.
    /// We compute the commitment and split (x | w) for you.
    /// This accumulates the step instance and witness without performing any folding.
    pub fn add_step_from_io(
        &mut self,
        input: &ProveInput<'_>,
    ) -> Result<(), PiCcsError> {
        // Normalize CCS to identity-first
        let s_norm = input
            .ccs
            .ensure_identity_first()
            .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;

        // STRICT validation: Ajtai/NC requires M₀ = I_n (fail fast instead of mysterious sumcheck errors)
        s_norm
            .assert_m0_is_identity_for_nc()
            .map_err(|e| PiCcsError::InvalidInput(e.to_string()))?;

        let m_in = input.public_input.len();
        if m_in + input.witness.len() != s_norm.m {
            return Err(PiCcsError::InvalidInput(format!(
                "len(x) + len(w) = {} but CCS.m = {}",
                m_in + input.witness.len(),
                s_norm.m
            )));
        }

        // Build z = [x | w], compute Z and commitment c
        let mut z = Vec::with_capacity(s_norm.m);
        z.extend_from_slice(input.public_input);
        z.extend_from_slice(input.witness);

        let Z = decompose_z_to_Z(&self.params, &z);
        let c = self.l.commit(&Z);

        // Produce MCS instance + witness
        let mcs_inst = McsInstance { c, x: input.public_input.to_vec(), m_in };
        let mcs_wit = McsWitness { w: input.witness.to_vec(), Z };

        self.mcss.push((mcs_inst, mcs_wit));
        self.step_claims.push(input.output_claims.to_vec());

        Ok(())
    }

    /// Fold and prove: run folding over all collected steps and return a `FoldRun`.
    /// This is where the actual cryptographic work happens (Π_CCS → RLC → DEC for each step).
    /// This method manages the transcript internally for ease of use.
    pub fn fold_and_prove(
        &mut self,
        s: &CcsStructure<F>,
    ) -> Result<FoldRun, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
        self.fold_and_prove_with_transcript(&mut tr, s)
    }

    /// Fold and prove with a caller-provided transcript (advanced users).
    /// This is where the actual cryptographic work happens (Π_CCS → RLC → DEC for each step).
    pub fn fold_and_prove_with_transcript(
        &mut self,
        tr: &mut Poseidon2Transcript,
        s: &CcsStructure<F>,
    ) -> Result<FoldRun, PiCcsError> {
        
        // Optimization: Avoid cloning if already normalized (M0 = I)
        let is_normalized = if s.n == s.m && !s.matrices.is_empty() {
            if !s.sparse_matrices.is_empty() {
                let m0 = &s.sparse_matrices[0];
                m0.values.len() == s.n &&
                m0.row_ptrs.len() == s.n + 1 &&
                m0.row_ptrs.iter().enumerate().all(|(i, &v)| i == v) &&
                m0.col_indices.iter().enumerate().all(|(i, &v)| i == v) &&
                m0.values.iter().all(|&v| v == F::ONE)
            } else {
                s.matrices[0].is_identity()
            }
        } else {
            false
        };

        let s_cow = if is_normalized && !s.sparse_matrices.is_empty() {
            std::borrow::Cow::Borrowed(s)
        } else {
            std::borrow::Cow::Owned(s.ensure_identity_first()
                .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?)
        };
        let s_norm = &*s_cow;

        // STRICT validation: Ajtai/NC requires M₀ = I_n (fail fast instead of mysterious sumcheck errors)
        s_norm
            .assert_m0_is_identity_for_nc()
            .map_err(|e| PiCcsError::InvalidInput(e.to_string()))?;

        // Determine canonical m_in from steps and ensure they all match (needed for RLC).
        let m_in_steps = self
            .mcss
            .first()
            .map(|(inst, _)| inst.m_in)
            .unwrap_or(0);
        if !self.mcss.iter().all(|(inst, _)| inst.m_in == m_in_steps) {
            return Err(PiCcsError::InvalidInput(
                "all steps must share the same m_in".into(),
            ));
        }

        // Validate or default the accumulator: None → k=1 simple case (no ME inputs).
        let (seed_me, seed_me_wit) = match &self.acc0 {
            Some(acc) => {
                acc.check(&self.params, &s_norm)?;
                // Also ensure accumulator m_in matches steps' m_in to avoid X-mixing shape issues.
                let acc_m_in = acc.me.first().map(|m| m.m_in).unwrap_or(m_in_steps);
                if acc_m_in != m_in_steps {
                    return Err(PiCcsError::InvalidInput(
                        "initial Accumulator.m_in must match steps' m_in".into(),
                    ));
                }
                (acc.me.clone(), acc.witnesses.clone())
            }
            None => (vec![], vec![]), // k=1
        };

        folding::fold_many_prove(
            self.mode.clone(),
            tr,
            &self.params,
            &s_norm,
            &self.mcss,
            &seed_me,
            &seed_me_wit,
            &self.l,
            self.mixers,
        )
    }

    /// Verify a finished run against the public MCS list.
    /// This method manages the transcript internally for ease of use.
    pub fn verify(
        &self,
        s: &CcsStructure<F>,
        mcss_public: &[neo_ccs::McsInstance<Cmt, F>],
        run: &FoldRun,
    ) -> Result<bool, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
        self.verify_with_transcript(&mut tr, s, mcss_public, run)
    }

    /// Verify with a caller-provided transcript (advanced users).
    pub fn verify_with_transcript(
        &self,
        tr: &mut Poseidon2Transcript,
        s: &CcsStructure<F>,
        mcss_public: &[neo_ccs::McsInstance<Cmt, F>],
        run: &FoldRun,
    ) -> Result<bool, PiCcsError> {
        // Normalize CCS
        let s_norm = s
            .ensure_identity_first()
            .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;

        // m_in consistency across public MCS
        let m_in_steps = mcss_public.first().map(|inst| inst.m_in).unwrap_or(0);
        if !mcss_public.iter().all(|inst| inst.m_in == m_in_steps) {
            return Err(PiCcsError::InvalidInput(
                "all steps must share the same m_in".into(),
            ));
        }

        // Validate (or empty) initial accumulator to mirror finalize()
        let (seed_me, _seed_me_wit) = match &self.acc0 {
            Some(acc) => {
                acc.check(&self.params, &s_norm)?;
                let acc_m_in = acc.me.first().map(|m| m.m_in).unwrap_or(m_in_steps);
                if acc_m_in != m_in_steps {
                    return Err(PiCcsError::InvalidInput(
                        "initial Accumulator.m_in must match steps' m_in".into(),
                    ));
                }
                // ME inputs are already well-formed (checked by acc.check())
                (acc.me.clone(), acc.witnesses.clone())
            }
            None => (vec![], vec![]), // k=1
        };

        folding::fold_many_verify(
            self.mode.clone(),
            tr,
            &self.params,
            &s_norm,
            mcss_public,
            &seed_me,
            run,
            self.mixers,
        )
    }
}
