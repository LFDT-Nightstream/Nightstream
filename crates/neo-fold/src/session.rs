//! Thin, ergonomic session layer for Π-CCS folding.
//!
//! This module provides:
//!   1) A small adapter trait (`NeoStep`) and `FoldingSession::add_step` for Nova/Sonobe-style step synthesis.
//!   2) A direct IO path via `ProveInput` + `FoldingSession::add_step_from_io` (callers hand us (x, w)).
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
//! lives in `shard::fold_shard_prove/verify` (Route A integration).

#![allow(non_snake_case)]
#![doc = include_str!("session/README.md")]

mod resources;
pub use resources::*;
mod layout;
pub use layout::*;
mod ccs_builder;
pub use ccs_builder::*;
mod circuit;
pub use circuit::*;
pub use crate::witness_layout;

use neo_ajtai::{has_seed_for_dims, s_lincomb, s_mul, unload_global_pp_for_dims, Commitment as Cmt};
use neo_ajtai::AjtaiSModule;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F, K};
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::builder::{build_shard_witness_shared_cpu_bus_with_aux, CpuArithmetization, ShardWitnessAux};
use neo_memory::plain::{LutTable, PlainMemLayout};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::witness::LutTableSpec;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::{SeedableRng, TryRngCore};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::pi_ccs::FoldingMode;
use crate::shard::{self, CommitMixers, ShardProof as FoldRun, ShardProverContext, StepLinkingConfig};
use crate::PiCcsError;
use neo_reductions::engines::utils;
use neo_reductions::engines::optimized_engine::oracle::SparseCache;
use neo_ccs::sparse::{CcsMatrix, CscMat};

#[inline]
fn mode_uses_sparse_cache(mode: &FoldingMode) -> bool {
    match mode {
        FoldingMode::Optimized => true,
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(_) => true,
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => false,
    }
}

#[inline]
fn canonical_ccs_square_dim(s: &CcsStructure<F>) -> usize {
    s.n.max(s.m)
}

fn pad_ccs_to_square(s: &CcsStructure<F>, dim: usize) -> Result<CcsStructure<F>, PiCcsError> {
    if dim < s.n || dim < s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "pad_ccs_to_square requires dim >= max(n, m), got dim={} for CCS {}x{}",
            dim, s.n, s.m
        )));
    }
    if s.n == dim && s.m == dim {
        return Ok(s.clone());
    }

    let mut matrices: Vec<CcsMatrix<F>> = Vec::with_capacity(s.matrices.len());
    for mj in &s.matrices {
        match mj {
            CcsMatrix::Identity { .. } => {
                return Err(PiCcsError::InvalidInput(
                    "cannot pad CCS containing Identity matrix while non-square".into(),
                ));
            }
            CcsMatrix::Csc(m) => {
                let mut m_new: CscMat<F> = m.clone();
                // Increase effective shape by adding trailing all-zero rows/cols.
                m_new.nrows = dim;
                if m_new.ncols < dim {
                    let last = *m_new
                        .col_ptr
                        .last()
                        .ok_or_else(|| PiCcsError::InvalidInput("CSC col_ptr is empty".into()))?;
                    m_new.col_ptr.resize(dim + 1, last);
                }
                m_new.ncols = dim;
                matrices.push(CcsMatrix::Csc(m_new));
            }
        }
    }

    Ok(CcsStructure {
        matrices,
        f: s.f.clone(),
        n: dim,
        m: dim,
    })
}

/// Normalize a CCS for the Ajtai/NC pipeline used by the session layer.
///
/// This:
/// - pads to a **square** CCS with `n=m=max(n,m)` by adding trailing all-zero rows/cols, then
/// - ensures identity-first form (`M₀ = I`), which is required by the NC constraints.
fn normalize_ccs_for_session(s: &CcsStructure<F>) -> Result<Cow<'_, CcsStructure<F>>, PiCcsError> {
    let dim = canonical_ccs_square_dim(s);
    let is_id0 = s.n == s.m
        && s
            .matrices
            .first()
            .map(|m0| m0.is_identity())
            .unwrap_or(false);
    if s.n == dim && s.m == dim && is_id0 {
        return Ok(Cow::Borrowed(s));
    }

    let mut owned = if s.n == dim && s.m == dim {
        s.clone()
    } else {
        pad_ccs_to_square(s, dim)?
    };

    owned = owned
        .ensure_identity_first_owned()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    owned
        .assert_m0_is_identity_for_nc()
        .map_err(|e| PiCcsError::InvalidInput(e.to_string()))?;

    Ok(Cow::Owned(owned))
}

/// Optional application-level "output claim".
/// (Not consumed by Π-CCS core yet; kept for API parity / future use.)
#[derive(Clone, Debug)]
pub struct OutputClaim<Ff> {
    pub tag: &'static [u8],
    pub expected: Ff,
}

// Twist/Shout linkage is supported only via the shared CPU-bus path. Callers must provide:
// - metadata-only Twist/Shout instances (no independent mem/lut commitments), and
// - a CPU witness/CCS that binds those bus fields (binding + padding constraints).

/// Direct inputs for a single step when you don't want to implement `NeoStep`.
/// We'll compute the commitment and split (x | w) for you.
#[derive(Clone, Debug)]
pub struct ProveInput<'a> {
    pub ccs: &'a CcsStructure<F>,            // the circuit (must match witness length)
    pub public_input: &'a [F],               // x
    pub witness: &'a [F],                    // w
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

impl StepSpec {
    /// Derive a verifier-side step-linking configuration under the common IVC convention:
    ///
    /// - public `x` is ordered as `[const1] ++ y_step ++ app_inputs`
    ///   (this is exactly how `FoldingSession::add_step` constructs `x`), and
    /// - the *previous-step state* `y_prev` is stored as the first `y_len` elements of `app_inputs`.
    ///
    /// This yields constraints:
    /// `steps[i].x[1 + j] == steps[i+1].x[1 + y_step_len + j]` for `j = 0..y_len`.
    pub fn ivc_step_linking_pairs(&self) -> Result<Vec<(usize, usize)>, PiCcsError> {
        if self.y_len == 0 {
            return Ok(Vec::new());
        }
        let Some(app) = &self.app_input_indices else {
            return Err(PiCcsError::InvalidInput(
                "StepSpec: ivc_step_linking_pairs requires app_input_indices (y_prev as prefix of app_inputs)".into(),
            ));
        };
        if self.y_len > self.y_step_indices.len() {
            return Err(PiCcsError::InvalidInput(
                "StepSpec: ivc_step_linking_pairs requires y_step_indices to include y_step".into(),
            ));
        }
        if self.y_len > app.len() {
            return Err(PiCcsError::InvalidInput(
                "StepSpec: ivc_step_linking_pairs requires app_input_indices to include y_prev as a prefix".into(),
            ));
        }

        // Derive x-layout from the StepSpec to avoid hard-coding offsets.
        let x_indices = indices_from_spec(self);
        if x_indices.len() != self.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "StepSpec: ivc_step_linking_pairs requires m_in == 1 + y_step_indices.len() + app_input_indices.len(); got m_in={}, computed {}",
                self.m_in,
                x_indices.len()
            )));
        }

        // Reject duplicates to avoid ambiguous x-position mapping.
        let mut pos_by_z: HashMap<usize, usize> = HashMap::with_capacity(x_indices.len());
        for (pos, &z_idx) in x_indices.iter().enumerate() {
            if pos_by_z.insert(z_idx, pos).is_some() {
                return Err(PiCcsError::InvalidInput(
                    "StepSpec: ivc_step_linking_pairs requires unique x indices (duplicates found)".into(),
                ));
            }
        }

        // Under the IVC convention, y_prev is the prefix of app_inputs and y_step is the prefix of y_step_indices.
        let mut pairs = Vec::with_capacity(self.y_len);
        for j in 0..self.y_len {
            let y_step_z = self.y_step_indices[j];
            let y_prev_z = app[j];
            let step_pos = *pos_by_z.get(&y_step_z).ok_or_else(|| {
                PiCcsError::InvalidInput("StepSpec: y_step_indices must be included in x".into())
            })?;
            let prev_pos = *pos_by_z.get(&y_prev_z).ok_or_else(|| {
                PiCcsError::InvalidInput("StepSpec: app_input_indices must be included in x".into())
            })?;
            pairs.push((step_pos, prev_pos));
        }

        Ok(pairs)
    }
}

/// What a step must return for the session to run Π-CCS.
#[derive(Clone, Debug)]
pub struct StepArtifacts {
    pub ccs: Arc<CcsStructure<F>>,
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
    fn synthesize_step(&mut self, step_idx: usize, y_prev: &[F], inputs: &Self::ExternalInputs) -> StepArtifacts;
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
fn default_mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        if cs.len() == 1 {
            return cs[0].clone();
        }
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
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
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
            return Err(PiCcsError::InvalidInput(
                "Accumulator: me.len() != witnesses.len()".into(),
            ));
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
                "Accumulator: r length mismatch (expected ell_n={}, got {})",
                ell_n,
                r0.len()
            )));
        }
        let m_in0 = self.me[0].m_in;
        for (i, (m, Z)) in self.me.iter().zip(self.witnesses.iter()).enumerate() {
            if m.r.len() != ell_n {
                return Err(PiCcsError::InvalidInput(format!(
                    "Accumulator[{}]: r length mismatch (expected ell_n={}, got {})",
                    i,
                    ell_n,
                    m.r.len()
                )));
            }
            if m.r != *r0 {
                return Err(PiCcsError::InvalidInput(
                    "Accumulator: all ME inputs must share the same r".into(),
                ));
            }
            if m.m_in != m_in0 {
                return Err(PiCcsError::InvalidInput(
                    "Accumulator: all ME inputs must share the same m_in".into(),
                ));
            }
            if Z.rows() != D || Z.cols() != s.m {
                return Err(PiCcsError::InvalidInput(format!(
                    "Accumulator[{}]: Z has shape {}x{}, expected {}x{}",
                    i,
                    Z.rows(),
                    Z.cols(),
                    D,
                    s.m
                )));
            }
            if m.X.rows() != D || m.X.cols() != m.m_in {
                return Err(PiCcsError::InvalidInput(
                    "Accumulator: X dimension mismatch with m_in".into(),
                ));
            }
            // Validate y-vector shape: t rows, each padded to 2^{ell_d}
            if m.y.len() != s.t() || !m.y.iter().all(|row| row.len() == want_pad) {
                return Err(PiCcsError::InvalidInput(format!(
                    "Accumulator[{}]: y shape invalid; expected t={} rows padded to 2^{{ell_d}}={}",
                    i,
                    s.t(),
                    want_pad
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
    s: &CcsStructure<F>, // should be identity-first
    l: &Lm,
    z: &[F],
    r: &[K],
    m_in: usize,
) -> Result<(MeInstance<Cmt, F, K>, Mat<F>), PiCcsError> {
    if z.len() != s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced: z length {} != CCS.m {}",
            z.len(),
            s.m
        )));
    }
    if m_in > s.m {
        return Err(PiCcsError::InvalidInput("me_from_z_balanced: m_in exceeds s.m".into()));
    }

    let dims = utils::build_dims_and_policy(params, s)?;
    if r.len() != dims.ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced: r length {} != ell_n {}",
            r.len(),
            dims.ell_n
        )));
    }
    let d_pad = 1usize << dims.ell_d;

    let Z = encode_vector_balanced_to_mat(params, z);
    let d = Z.rows();
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
        s.matrices[j].add_mul_transpose_into(&chi_r, &mut vj, s.n);
        vjs.push(vj);
    }

    // y rows padded to 2^{ell_d}; y_scalars = base-b recomposition of first D digits
    let bF = F::from_u64(params.b as u64);
    let mut pow_b_f = vec![F::ONE; d];
    for t in 1..d {
        pow_b_f[t] = pow_b_f[t - 1] * bF;
    }
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
        for rho in 0..d {
            scalar += yj[rho] * pow_b_k[rho];
        }
        y.push(yj);
        y_scalars.push(scalar);
    }

    let me = MeInstance::<Cmt, F, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
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
    s: &CcsStructure<F>, // should be identity-first
    l: &Lm,
    z: &[F],
    r: &[K],
    x_col_indices: &[usize], // which columns of Z form X, in order
) -> Result<(MeInstance<Cmt, F, K>, Mat<F>), PiCcsError> {
    if z.len() != s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced_select: z length {} != CCS.m {}",
            z.len(),
            s.m
        )));
    }

    let dims = utils::build_dims_and_policy(params, s)?;
    if r.len() != dims.ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced_select: r length {} != ell_n {}",
            r.len(),
            dims.ell_n
        )));
    }
    let d_pad = 1usize << dims.ell_d;

    let Z = encode_vector_balanced_to_mat(params, z);
    let d = Z.rows();
    let c = l.commit(&Z);

    // X := selected columns of Z (not the first m_in)
    let m_in = x_col_indices.len();
    let mut X = Mat::zero(d, m_in, F::ZERO);
    for (j, &col) in x_col_indices.iter().enumerate() {
        if col >= Z.cols() {
            return Err(PiCcsError::InvalidInput(format!(
                "X column index {} out of range (Z has {} cols)",
                col,
                Z.cols()
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
        s.matrices[j].add_mul_transpose_into(&chi_r, &mut vj, s.n);
        vjs.push(vj);
    }

    // y rows padded to 2^{ell_d}; y_scalars = base-b recomposition of first D digits
    let bF = F::from_u64(params.b as u64);
    let mut pow_b_f = vec![F::ONE; d];
    for t in 1..d {
        pow_b_f[t] = pow_b_f[t - 1] * bF;
    }
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
        for rho in 0..d {
            scalar += yj[rho] * pow_b_k[rho];
        }
        y.push(yj);
        y_scalars.push(scalar);
    }

    let me = MeInstance::<Cmt, F, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
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
    L: SModuleHomomorphism<F, Cmt> + Clone + Sync,
{
    mode: FoldingMode,
    params: NeoParams,
    l: L,
    commit_m: Option<usize>,
    mixers: CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>,

    // Cached CCS preprocessing for proving (best-effort reuse).
    prover_ctx: Option<SessionCcsCache>,
    // Cached CCS preprocessing for verification (must be explicitly preloaded).
    verifier_ctx: Option<SessionCcsCache>,

    // Collected per-step bundles (CCS-only steps have empty LUT/MEM vectors)
    steps: Vec<StepWitnessBundle<Cmt, F, K>>,
    /// Auxiliary data for the most recent shared-CPU-bus witness build (if any).
    ///
    /// This is used to support ergonomic APIs such as “output binding without manually providing
    /// final_memory_state”.
    shared_bus_aux: Option<ShardWitnessAux>,
    /// Optional declarative resource configuration used by `execute_shard_shared_cpu_bus_configured`.
    shared_bus_resources: Option<SharedBusResources>,

    /// Optional verifier-side step-to-step linking constraints.
    ///
    /// If more than one step is collected, verification requires either:
    /// - a non-empty `step_linking` config, or
    /// - `allow_unlinked_steps=true` (explicit and unsafe).
    step_linking: Option<StepLinkingConfig>,
    /// Explicit escape hatch: allow verifying multi-step runs without step linking.
    ///
    /// This is unsafe for any workflow where step-to-step chaining is part of the statement.
    allow_unlinked_steps: bool,
    /// Best-effort diagnostic for why auto step-linking did not engage (if attempted).
    auto_step_linking_error: Option<String>,

    // Optional initial accumulated ME(b, L)^k inputs (k = me.len()).
    acc0: Option<Accumulator>,

    // Optional: app-level claims recorded per step (not enforced here yet)
    step_claims: Vec<Vec<OutputClaim<F>>>,

    /// Current threaded state y (if any). Length is determined by `NeoStep::state_len()`.
    curr_state: Option<Vec<F>>,
}

#[derive(Clone)]
struct SessionCcsCache {
    /// Address of the caller-supplied `CcsStructure` used to build this cache.
    src_ptr: usize,
    /// Optional identity-first CCS, stored only if normalization requires insertion.
    s_norm: Option<CcsStructure<F>>,
    /// Precomputed circuit artifacts (digest + optional sparse cache).
    ctx: ShardProverContext,
}

impl<L> FoldingSession<L>
where
    L: SModuleHomomorphism<F, Cmt> + Clone + Sync,
{
    /// Create a new session with default Ajtai mixers and no initial accumulator (k=1 simple flow).
    pub fn new(mode: FoldingMode, params: NeoParams, l: L) -> Self {
        Self {
            mode,
            params,
            l,
            commit_m: None,
            mixers: default_mixers(),
            prover_ctx: None,
            verifier_ctx: None,
            steps: vec![],
            shared_bus_aux: None,
            shared_bus_resources: None,
            step_linking: None,
            allow_unlinked_steps: false,
            auto_step_linking_error: None,
            acc0: None,
            step_claims: vec![],
            curr_state: None,
        }
    }

    /// Access the selected Neo parameters for this session.
    pub fn params(&self) -> &NeoParams {
        &self.params
    }

    /// Access the underlying committer used by this session.
    pub fn committer(&self) -> &L {
        &self.l
    }

    fn ensure_committer_m_matches(&self, m: usize) -> Result<(), PiCcsError> {
        let Some(commit_m) = self.commit_m else {
            return Ok(());
        };
        if commit_m != m {
            return Err(PiCcsError::InvalidInput(format!(
                "session committer configured for CCS.m={}, but this step expects CCS.m={}; construct the session from the same (padded) CCS shape",
                commit_m, m
            )));
        }
        Ok(())
    }

    /// Replace the stored shared-bus resource configuration (Twist layouts/init + Shout tables/specs).
    pub fn set_shared_bus_resources(&mut self, resources: SharedBusResources) {
        self.shared_bus_resources = Some(resources);
    }

    /// Mutably access (and lazily initialize) the shared-bus resource configuration.
    pub fn shared_bus_resources_mut(&mut self) -> &mut SharedBusResources {
        self.shared_bus_resources.get_or_insert_with(SharedBusResources::new)
    }

    /// Execute a VM shard in shared-CPU-bus mode using the resources stored on this session.
    ///
    /// This is equivalent to calling `execute_shard_shared_cpu_bus(...)` with the maps from
    /// `SharedBusResources`, but avoids re-threading those arguments on every call.
    pub fn execute_shard_shared_cpu_bus_configured<V, A, Tw, Sh>(
        &mut self,
        vm: V,
        twist: Tw,
        shout: Sh,
        max_steps: usize,
        chunk_size: usize,
        cpu_arith: &A,
    ) -> Result<(), PiCcsError>
    where
        V: neo_vm_trace::VmCpu<u64, u64>,
        Tw: neo_vm_trace::Twist<u64, u64>,
        Sh: neo_vm_trace::Shout<u64>,
        A: CpuArithmetization<F, Cmt>,
    {
        let (bundles, aux) = {
            let resources = self.shared_bus_resources.as_ref().ok_or_else(|| {
                PiCcsError::InvalidInput(
                    "missing shared-bus resources; call set_shared_bus_resources(...) or shared_bus_resources_mut() first".into(),
                )
            })?;

            build_shard_witness_shared_cpu_bus_with_aux(
                vm,
                twist,
                shout,
                max_steps,
                chunk_size,
                &resources.mem_layouts,
                &resources.lut_tables,
                &resources.lut_table_specs,
                &resources.initial_mem,
                cpu_arith,
            )
            .map_err(|e| PiCcsError::InvalidInput(format!("shared-bus witness build failed: {e:?}")))?
        };

        self.add_step_bundles(bundles);
        self.shared_bus_aux = Some(aux);
        Ok(())
    }

    /// Enable verifier-side step-to-step linking for multi-step runs.
    pub fn set_step_linking(&mut self, cfg: StepLinkingConfig) {
        self.step_linking = Some(cfg);
        self.auto_step_linking_error = None;
    }

    /// Enable verifier-side step-to-step linking using the common IVC convention from `StepSpec`.
    ///
    /// See `StepSpec::ivc_step_linking_pairs` for the exact assumptions.
    pub fn enable_step_linking_from_step_spec(&mut self, spec: &StepSpec) -> Result<(), PiCcsError> {
        let pairs = spec.ivc_step_linking_pairs()?;
        self.set_step_linking(StepLinkingConfig::new(pairs));
        Ok(())
    }

    /// Explicitly allow multi-step verification without step linking (unsafe).
    pub fn unsafe_allow_unlinked_steps(&mut self) {
        self.allow_unlinked_steps = true;
        self.auto_step_linking_error = None;
    }

    /// Set an explicit initial state y₀ for the IVC (optional).
    /// If not set, y₀ defaults to all zeros of length `state_len()`.
    pub fn set_initial_state(&mut self, y0: Vec<F>) {
        self.curr_state = Some(y0);
    }

    /// Inject an explicit initial Accumulator (k = acc.me.len()). This enables k>1 multi-folding.
    pub fn with_initial_accumulator(mut self, acc: Accumulator, s: &CcsStructure<F>) -> Result<Self, PiCcsError> {
        let s_norm_owned: Option<CcsStructure<F>> = match normalize_ccs_for_session(s)? {
            Cow::Borrowed(_) => None,
            Cow::Owned(s_norm) => Some(s_norm),
        };
        let s_norm = s_norm_owned.as_ref().unwrap_or(s);

        acc.check(&self.params, s_norm)?;
        self.acc0 = Some(acc);
        Ok(self)
    }

    /// Access the accumulated public MCS instances (for verification APIs).
    pub fn mcss_public(&self) -> Vec<McsInstance<Cmt, F>> {
        self.steps.iter().map(|step| step.mcs.0.clone()).collect()
    }

    /// Access the collected *public* per-step bundles (MCS + optional Twist/Shout instances).
    ///
    /// This is useful for specialized verifiers (e.g. RV32 B1 statement checks) that need access
    /// to memory/lookup instances, not just the MCS list.
    pub fn steps_public(&self) -> Vec<StepInstanceBundle<Cmt, F, K>> {
        self.steps.iter().map(StepInstanceBundle::from).collect()
    }

    /// Add one step using the `NeoStep` synthesis adapter.
    /// This accumulates the step instance and witness without performing any folding.
    ///
    /// If the returned `StepSpec` matches the common IVC layout (see
    /// `StepSpec::ivc_step_linking_pairs`), this automatically enables verifier-side step linking
    /// so multi-step verification works by default.
    pub fn add_step<S: NeoStep>(&mut self, stepper: &mut S, inputs: &S::ExternalInputs) -> Result<(), PiCcsError> {
        let step_idx = self.steps.len();
        // 1) Decide previous state y_prev
        let state_len = stepper.state_len();
        let y_prev = self
            .curr_state
            .clone()
            .unwrap_or_else(|| vec![F::ZERO; state_len]);

        // 2) Let the app synthesize CCS + witness given y_prev
        let StepArtifacts {
            ccs,
            witness: mut z,
            spec,
            public_app_inputs: _,
        } = stepper.synthesize_step(step_idx, &y_prev, inputs);

        // Safety: require state_len to match StepSpec
        if spec.y_len != state_len {
            return Err(PiCcsError::InvalidInput(format!(
                "StepSpec.y_len ({}) must equal stepper.state_len() ({})",
                spec.y_len, state_len
            )));
        }

        if self.step_linking.is_none() && !self.allow_unlinked_steps {
            match spec.ivc_step_linking_pairs() {
                Ok(pairs) => {
                    if !pairs.is_empty() {
                        self.set_step_linking(StepLinkingConfig::new(pairs));
                    }
                }
                Err(e) => {
                    self.auto_step_linking_error = Some(format!("step {step_idx}: {e}"));
                }
            }
        }

        // Canonicalize witness length to the padded-square CCS dimension.
        let m_expected = ccs.m;
        let m_target = canonical_ccs_square_dim(ccs.as_ref());
        self.ensure_committer_m_matches(m_target)?;
        if z.len() == m_expected && m_expected < m_target {
            z.resize(m_target, F::ZERO);
        }
        if z.len() != m_target {
            return Err(PiCcsError::InvalidInput(format!(
                "step witness length {} must equal CCS.m={} (or CCS.m padded to {})",
                z.len(),
                m_expected,
                m_target
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
                return Err(PiCcsError::InvalidInput("StepSpec indices contain duplicates".into()));
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

        let Z = encode_vector_balanced_to_mat(&self.params, &z);
        let c = self.l.commit(&Z);
        let m_in = spec.m_in;

        // w is the private witness (suffix)
        let w = z[m_in..].to_vec();

        let mcs_inst = McsInstance { c, x, m_in };
        let mcs_wit = McsWitness { w, Z };

        self.shared_bus_aux = None;
        self.steps.push((mcs_inst, mcs_wit).into());
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
                        idx,
                        z.len()
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
    pub fn add_step_from_io(&mut self, input: &ProveInput<'_>) -> Result<(), PiCcsError> {
        let m_in = input.public_input.len();
        let m_expected = input.ccs.m;
        let m_target = canonical_ccs_square_dim(input.ccs);
        self.ensure_committer_m_matches(m_target)?;
        let total = m_in
            .checked_add(input.witness.len())
            .ok_or_else(|| PiCcsError::InvalidInput("len(x) + len(w) overflow".into()))?;
        if total != m_expected && total != m_target {
            return Err(PiCcsError::InvalidInput(format!(
                "len(x) + len(w) = {} but CCS.m = {} (or padded to {})",
                total, m_expected, m_target
            )));
        }

        // Build z = [x | w], compute Z and commitment c
        let mut z = Vec::with_capacity(m_target);
        z.extend_from_slice(input.public_input);
        z.extend_from_slice(input.witness);
        if total == m_expected && m_expected < m_target {
            z.resize(m_target, F::ZERO);
        }
        debug_assert_eq!(z.len(), m_target);

        let Z = encode_vector_balanced_to_mat(&self.params, &z);
        let c = self.l.commit(&Z);

        // Produce MCS instance + witness
        let mcs_inst = McsInstance {
            c,
            x: input.public_input.to_vec(),
            m_in,
        };
        let mcs_wit = McsWitness {
            w: z[m_in..].to_vec(),
            Z,
        };

        self.shared_bus_aux = None;
        self.steps.push((mcs_inst, mcs_wit).into());
        self.step_claims.push(input.output_claims.to_vec());

        Ok(())
    }

    /// Convenience: add one step directly from `(x, w)` without constructing `ProveInput`.
    pub fn add_step_io(
        &mut self,
        ccs: &CcsStructure<F>,
        public_input: &[F],
        witness: &[F],
    ) -> Result<(), PiCcsError> {
        let input = ProveInput {
            ccs,
            public_input,
            witness,
            output_claims: &[],
        };
        self.add_step_from_io(&input)
    }

    /// Convenience: add `n_steps` steps by repeatedly calling `add_step`.
    pub fn add_steps<S: NeoStep>(
        &mut self,
        stepper: &mut S,
        inputs: &S::ExternalInputs,
        n_steps: usize,
    ) -> Result<(), PiCcsError> {
        for _ in 0..n_steps {
            self.add_step(stepper, inputs)?;
        }
        Ok(())
    }

    /// Add a pre-built step bundle directly.
    ///
    /// This is the low-level API for when you have already constructed a `StepWitnessBundle`
    /// with memory (Twist) and/or lookup (Shout) instances.
    ///
    /// Use this method when your proof requires Twist/Shout arguments in addition to CCS.
    pub fn add_step_bundle(&mut self, bundle: StepWitnessBundle<Cmt, F, K>) {
        self.shared_bus_aux = None;
        self.steps.push(bundle);
        self.step_claims.push(vec![]);
    }

    /// Add multiple pre-built step bundles at once.
    pub fn add_step_bundles(&mut self, bundles: impl IntoIterator<Item = StepWitnessBundle<Cmt, F, K>>) {
        for bundle in bundles {
            self.add_step_bundle(bundle);
        }
    }

    /// Execute a VM for one shard and add shared-CPU-bus step bundles to this session.
    ///
    /// This is an ergonomic wrapper around `neo_memory::builder::build_shard_witness_shared_cpu_bus_with_aux`.
    /// It also stores auxiliary outputs (including the terminal Twist memory state) so the session
    /// can later prove output binding without the caller manually providing `final_memory_state`.
    pub fn execute_shard_shared_cpu_bus<V, A, Tw, Sh>(
        &mut self,
        vm: V,
        twist: Tw,
        shout: Sh,
        max_steps: usize,
        chunk_size: usize,
        mem_layouts: &HashMap<u32, PlainMemLayout>,
        lut_tables: &HashMap<u32, LutTable<F>>,
        lut_table_specs: &HashMap<u32, LutTableSpec>,
        initial_mem: &HashMap<(u32, u64), F>,
        cpu_arith: &A,
    ) -> Result<(), PiCcsError>
    where
        V: neo_vm_trace::VmCpu<u64, u64>,
        Tw: neo_vm_trace::Twist<u64, u64>,
        Sh: neo_vm_trace::Shout<u64>,
        A: CpuArithmetization<F, Cmt>,
    {
        let (bundles, aux) = build_shard_witness_shared_cpu_bus_with_aux(
            vm,
            twist,
            shout,
            max_steps,
            chunk_size,
            mem_layouts,
            lut_tables,
            lut_table_specs,
            initial_mem,
            cpu_arith,
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("shared-bus witness build failed: {e:?}")))?;

        self.add_step_bundles(bundles);
        self.shared_bus_aux = Some(aux);
        Ok(())
    }

    /// Check if any steps have Twist (memory) instances.
    pub fn has_twist_instances(&self) -> bool {
        self.steps.iter().any(|s| !s.mem_instances.is_empty())
    }

    /// Check if any steps have Shout (lookup) instances.
    pub fn has_shout_instances(&self) -> bool {
        self.steps.iter().any(|s| !s.lut_instances.is_empty())
    }

    fn ensure_accumulator_matches_ccs(&mut self, s: &CcsStructure<F>) -> Result<(), PiCcsError> {
        let Some(acc) = self.acc0.as_mut() else {
            return Ok(());
        };

        if acc.me.is_empty() {
            return Ok(());
        }
        if acc
            .me
            .iter()
            .all(|me| me.y.len() == s.t() && me.y_scalars.len() == s.t())
        {
            return Ok(());
        }

        let dims = utils::build_dims_and_policy(&self.params, s)?;
        let d_pad = 1usize << dims.ell_d;

        for (me, z_mat) in acc.me.iter_mut().zip(acc.witnesses.iter()) {
            let (y_vecs_d, y_scalars) = neo_memory::mle::compute_me_y_for_ccs(s, z_mat, &me.r, self.params.b as u64);

            let d = z_mat.rows();
            let mut y_padded: Vec<Vec<K>> = Vec::with_capacity(y_vecs_d.len());
            for y_d in y_vecs_d {
                let mut yj = vec![K::ZERO; d_pad];
                for rho in 0..d {
                    yj[rho] = y_d[rho];
                }
                y_padded.push(yj);
            }

            me.y = y_padded;
            me.y_scalars = y_scalars;
        }

        Ok(())
    }

    fn prepared_ccs_for_accumulator<'s>(&self, s: &'s CcsStructure<F>) -> Result<&'s CcsStructure<F>, PiCcsError> {
        if !(self.has_twist_instances() || self.has_shout_instances()) {
            return Ok(s);
        }
        if self.steps.is_empty() {
            return Ok(s);
        }

        let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> =
            self.steps.iter().map(StepInstanceBundle::from).collect();
        let (s_prepared, _cpu_bus) =
            crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(s, &steps_public)?;
        Ok(s_prepared)
    }

    fn build_ccs_cache(
        &self,
        s: &CcsStructure<F>,
        ccs_sparse_cache: Option<Arc<SparseCache<F>>>,
    ) -> Result<SessionCcsCache, PiCcsError> {
        let src_ptr = (s as *const CcsStructure<F>) as usize;

        let mut s_norm_opt: Option<CcsStructure<F>> = None;
        let s_norm_ref: &CcsStructure<F> = match normalize_ccs_for_session(s)? {
            Cow::Borrowed(s) => s,
            Cow::Owned(s) => {
                s_norm_opt = Some(s);
                s_norm_opt.as_ref().expect("just set")
            }
        };

        if let Some(ref cache) = ccs_sparse_cache {
            if cache.len() != s_norm_ref.t() {
                return Err(PiCcsError::InvalidInput(format!(
                    "SparseCache matrix count mismatch: cache has {}, CCS has {}",
                    cache.len(),
                    s_norm_ref.t()
                )));
            }
        }

        let ccs_sparse_cache = if let Some(cache) = ccs_sparse_cache {
            Some(cache)
        } else if mode_uses_sparse_cache(&self.mode) {
            Some(Arc::new(SparseCache::build(s_norm_ref)))
        } else {
            None
        };

        let ccs_mat_digest =
            utils::digest_ccs_matrices_with_sparse_cache(s_norm_ref, ccs_sparse_cache.as_deref());
        let ctx = ShardProverContext {
            ccs_mat_digest,
            ccs_sparse_cache,
        };

        Ok(SessionCcsCache {
            src_ptr,
            s_norm: s_norm_opt,
            ctx,
        })
    }

    fn ensure_prover_ctx_for_ccs(&mut self, s: &CcsStructure<F>) -> Result<(), PiCcsError> {
        let src_ptr = (s as *const CcsStructure<F>) as usize;
        if let Some(cache) = &self.prover_ctx {
            if cache.src_ptr == src_ptr {
                return Ok(());
            }
        }
        self.prover_ctx = Some(self.build_ccs_cache(s, None)?);
        Ok(())
    }

    /// Preload prover-side CCS preprocessing (sparse-cache + matrix-digest) to avoid scanning dense matrices.
    ///
    /// This is intended for callers who already have a `SparseCache` (e.g. built from sparse R1CS
    /// constraints) and want to skip the expensive `SparseCache::build` pass over dense matrices.
    ///
    /// Note: verification uses a separate cache; call `preload_verifier_ccs_sparse_cache(...)` if desired.
    pub fn preload_ccs_sparse_cache(
        &mut self,
        s: &CcsStructure<F>,
        ccs_sparse_cache: Arc<SparseCache<F>>,
    ) -> Result<(), PiCcsError> {
        self.prover_ctx = Some(self.build_ccs_cache(s, Some(ccs_sparse_cache))?);
        Ok(())
    }

    /// Preload verifier-side CCS preprocessing (sparse-cache + matrix-digest).
    ///
    /// This does **not** affect proving. It exists so benchmarks can model a verifier that has
    /// preprocessed the public circuit independently of the prover.
    pub fn preload_verifier_ccs_sparse_cache(
        &mut self,
        s: &CcsStructure<F>,
        ccs_sparse_cache: Arc<SparseCache<F>>,
    ) -> Result<(), PiCcsError> {
        self.verifier_ctx = Some(self.build_ccs_cache(s, Some(ccs_sparse_cache))?);
        Ok(())
    }

    /// Fold and prove: run folding over all collected steps and return a `FoldRun`.
    /// This is where the actual cryptographic work happens (Π_CCS → RLC → DEC for each step).
    /// This method manages the transcript internally for ease of use.
    pub fn fold_and_prove(&mut self, s: &CcsStructure<F>) -> Result<FoldRun, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
        self.fold_and_prove_with_transcript(&mut tr, s)
    }

    /// Convenience: fold, prove, and verify using the internally collected steps.
    ///
    /// This returns the proof run if verification succeeds.
    pub fn prove_and_verify_collected(&mut self, s: &CcsStructure<F>) -> Result<FoldRun, PiCcsError> {
        let run = self.fold_and_prove(s)?;
        let ok = self.verify_collected(s, &run)?;
        if !ok {
            return Err(PiCcsError::ProtocolError("verification failed".into()));
        }
        Ok(run)
    }

    /// Fold and prove with a caller-provided transcript (advanced users).
    /// This is where the actual cryptographic work happens (Π_CCS → RLC → DEC for each step).
    pub fn fold_and_prove_with_transcript(
        &mut self,
        tr: &mut Poseidon2Transcript,
        s: &CcsStructure<F>,
    ) -> Result<FoldRun, PiCcsError> {
        self.ensure_prover_ctx_for_ccs(s)?;

        // Temporarily take the prover ctx to avoid borrow conflicts while we may mutate `self`.
        let cache = self.prover_ctx.take().expect("prover ctx must be set");
        let s_norm = cache.s_norm.as_ref().unwrap_or(s);
        let ctx = cache.ctx.clone();

        let result = (|| {
            // Shared CPU bus: compute the prepared CCS shape (copy-outs) for accumulator validation.
            // IMPORTANT: Use the normalized CCS so M₀=I holds even if the caller passed a non-identity-first CCS.
            let s_prepared = self.prepared_ccs_for_accumulator(s_norm)?;
            s_prepared
                .assert_m0_is_identity_for_nc()
                .map_err(|e| PiCcsError::InvalidInput(e.to_string()))?;
            self.ensure_accumulator_matches_ccs(s_prepared)?;

            // Determine canonical m_in from steps and ensure they all match (needed for RLC).
            let m_in_steps = self.steps.first().map(|step| step.mcs.0.m_in).unwrap_or(0);
            if !self.steps.iter().all(|step| step.mcs.0.m_in == m_in_steps) {
                return Err(PiCcsError::InvalidInput("all steps must share the same m_in".into()));
            }

            // Validate or default the accumulator: None → k=1 simple case (no ME inputs).
            let (seed_me, seed_me_wit): (&[MeInstance<Cmt, F, K>], &[Mat<F>]) = match &self.acc0 {
                Some(acc) => {
                    acc.check(&self.params, s_prepared)?;
                    // Also ensure accumulator m_in matches steps' m_in to avoid X-mixing shape issues.
                    let acc_m_in = acc.me.first().map(|m| m.m_in).unwrap_or(m_in_steps);
                    if acc_m_in != m_in_steps {
                        return Err(PiCcsError::InvalidInput(
                            "initial Accumulator.m_in must match steps' m_in".into(),
                        ));
                    }
                    (&acc.me, &acc.witnesses)
                }
                None => (&[], &[]), // k=1
            };

            // If PP is reloadable (seeded), unload it before memory-heavy oracle/sumcheck work.
            // This keeps peak RSS low on constrained runtimes (e.g. WASM).
            if has_seed_for_dims(D, s_norm.m) {
                let _ = unload_global_pp_for_dims(D, s_norm.m);
            }

            shard::fold_shard_prove_with_context(
                self.mode.clone(),
                tr,
                &self.params,
                s_norm,
                &self.steps,
                seed_me,
                seed_me_wit,
                &self.l,
                self.mixers,
                &ctx,
            )
        })();

        self.prover_ctx = Some(cache);
        result
    }

    pub fn fold_and_prove_with_output_binding(
        &mut self,
        tr: &mut Poseidon2Transcript,
        s: &CcsStructure<F>,
        ob_cfg: &crate::output_binding::OutputBindingConfig,
        final_memory_state: &[F],
    ) -> Result<FoldRun, PiCcsError> {
        self.ensure_prover_ctx_for_ccs(s)?;

        let cache = self.prover_ctx.take().expect("prover ctx must be set");
        let ctx = cache.ctx.clone();
        let s_norm = cache.s_norm.as_ref().unwrap_or(s);

        let result = (|| {
            let s_prepared = self.prepared_ccs_for_accumulator(s_norm)?;
            s_prepared
                .assert_m0_is_identity_for_nc()
                .map_err(|e| PiCcsError::InvalidInput(e.to_string()))?;
            self.ensure_accumulator_matches_ccs(s_prepared)?;

            let m_in_steps = self.steps.first().map(|step| step.mcs.0.m_in).unwrap_or(0);
            if !self.steps.iter().all(|step| step.mcs.0.m_in == m_in_steps) {
                return Err(PiCcsError::InvalidInput("all steps must share the same m_in".into()));
            }

            let (seed_me, seed_me_wit): (&[MeInstance<Cmt, F, K>], &[Mat<F>]) = match &self.acc0 {
                Some(acc) => {
                    acc.check(&self.params, s_prepared)?;
                    let acc_m_in = acc.me.first().map(|m| m.m_in).unwrap_or(m_in_steps);
                    if acc_m_in != m_in_steps {
                        return Err(PiCcsError::InvalidInput(
                            "initial Accumulator.m_in must match steps' m_in".into(),
                        ));
                    }
                    (&acc.me, &acc.witnesses)
                }
                None => (&[], &[]),
            };

            shard::fold_shard_prove_with_output_binding_with_context(
                self.mode.clone(),
                tr,
                &self.params,
                s_norm,
                &self.steps,
                seed_me,
                seed_me_wit,
                &self.l,
                self.mixers,
                ob_cfg,
                final_memory_state,
                &ctx,
            )
        })();

        self.prover_ctx = Some(cache);
        result
    }

    /// Fold and prove with output binding, managing the transcript internally.
    pub fn fold_and_prove_with_output_binding_simple(
        &mut self,
        s: &CcsStructure<F>,
        ob_cfg: &crate::output_binding::OutputBindingConfig,
        final_memory_state: &[F],
    ) -> Result<FoldRun, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
        self.fold_and_prove_with_output_binding(&mut tr, s, ob_cfg, final_memory_state)
    }

    fn final_memory_state_for_output_binding(&self, ob_cfg: &crate::output_binding::OutputBindingConfig) -> Result<Vec<F>, PiCcsError> {
        let aux = self.shared_bus_aux.as_ref().ok_or_else(|| {
            PiCcsError::InvalidInput(
                "output binding auto mode requires shared-bus aux; call execute_shard_shared_cpu_bus(...) first".into(),
            )
        })?;
        let last_step = self
            .steps
            .last()
            .ok_or_else(|| PiCcsError::InvalidInput("output binding requires >= 1 step".into()))?;

        if ob_cfg.mem_idx >= last_step.mem_instances.len() {
            return Err(PiCcsError::InvalidInput("output binding mem_idx out of range".into()));
        }
        if ob_cfg.mem_idx >= aux.mem_ids.len() {
            return Err(PiCcsError::InvalidInput(
                "output binding mem_idx out of range for shared-bus aux".into(),
            ));
        }

        let expected_k = 1usize
            .checked_shl(ob_cfg.num_bits as u32)
            .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
        let mem_inst = &last_step.mem_instances[ob_cfg.mem_idx].0;
        if mem_inst.k != expected_k {
            return Err(PiCcsError::InvalidInput(format!(
                "output binding: cfg.num_bits implies k={}, but mem_inst.k={}",
                expected_k, mem_inst.k
            )));
        }

        let mem_id = aux.mem_ids[ob_cfg.mem_idx];
        let mut final_memory_state = vec![F::ZERO; expected_k];
        if let Some(st) = aux.final_mem_states.get(&mem_id) {
            for (&addr, &val) in st {
                let Ok(addr_usize) = usize::try_from(addr) else {
                    continue;
                };
                if addr_usize < expected_k {
                    final_memory_state[addr_usize] = val;
                }
            }
        }
        Ok(final_memory_state)
    }

    /// Fold and prove with output binding, deriving `final_memory_state` from the most recent
    /// shared-CPU-bus witness build (see `execute_shard_shared_cpu_bus`).
    pub fn fold_and_prove_with_output_binding_auto(
        &mut self,
        tr: &mut Poseidon2Transcript,
        s: &CcsStructure<F>,
        ob_cfg: &crate::output_binding::OutputBindingConfig,
    ) -> Result<FoldRun, PiCcsError> {
        let final_memory_state = self.final_memory_state_for_output_binding(ob_cfg)?;
        self.fold_and_prove_with_output_binding(tr, s, ob_cfg, &final_memory_state)
    }

    /// Fold and prove with output binding (auto final memory state), managing the transcript internally.
    pub fn fold_and_prove_with_output_binding_auto_simple(
        &mut self,
        s: &CcsStructure<F>,
        ob_cfg: &crate::output_binding::OutputBindingConfig,
    ) -> Result<FoldRun, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
        self.fold_and_prove_with_output_binding_auto(&mut tr, s, ob_cfg)
    }

    /// Convenience: fold+prove with output binding (auto final memory) and verify (collected steps).
    ///
    /// This returns the proof run if verification succeeds.
    pub fn prove_and_verify_with_output_binding_collected_auto_simple(
        &mut self,
        s: &CcsStructure<F>,
        ob_cfg: &crate::output_binding::OutputBindingConfig,
    ) -> Result<FoldRun, PiCcsError> {
        let run = self.fold_and_prove_with_output_binding_auto_simple(s, ob_cfg)?;
        let ok = self.verify_with_output_binding_collected_simple(s, &run, ob_cfg)?;
        if !ok {
            return Err(PiCcsError::ProtocolError("verification failed".into()));
        }
        Ok(run)
    }

    /// Verify a finished run against the public MCS list.
    /// This method manages the transcript internally for ease of use.
    ///
    /// Note: this does not reuse prover-side preprocessing caches. To model a verifier that
    /// preprocesses the public circuit, call `preload_verifier_ccs_sparse_cache(...)` once.
    pub fn verify(
        &self,
        s: &CcsStructure<F>,
        mcss_public: &[neo_ccs::McsInstance<Cmt, F>],
        run: &FoldRun,
    ) -> Result<bool, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
        self.verify_with_transcript(&mut tr, s, mcss_public, run)
    }

    /// Verify a finished run using the internally collected steps.
    /// Convenient when you don't want to manually extract the public MCS list.
    pub fn verify_collected(&self, s: &CcsStructure<F>, run: &FoldRun) -> Result<bool, PiCcsError> {
        let mcss_public = self.mcss_public();
        self.verify(s, &mcss_public, run)
    }

    /// Verify with a caller-provided transcript (advanced users).
    pub fn verify_with_transcript(
        &self,
        tr: &mut Poseidon2Transcript,
        s: &CcsStructure<F>,
        mcss_public: &[neo_ccs::McsInstance<Cmt, F>],
        run: &FoldRun,
    ) -> Result<bool, PiCcsError> {
        let src_ptr = (s as *const CcsStructure<F>) as usize;
        let verifier_cache = self
            .verifier_ctx
            .as_ref()
            .filter(|cache| cache.src_ptr == src_ptr);
        let verifier_ctx = verifier_cache.map(|cache| &cache.ctx);

        // Normalize CCS (avoid cloning when already normalized or preloaded).
        let s_norm_owned: Option<CcsStructure<F>> = match verifier_cache {
            Some(_) => None,
            None => match normalize_ccs_for_session(s)? {
                Cow::Borrowed(_) => None,
                Cow::Owned(s_norm) => Some(s_norm),
            },
        };
        let s_norm: &CcsStructure<F> = match verifier_cache {
            Some(cache) => cache.s_norm.as_ref().unwrap_or(s),
            None => s_norm_owned.as_ref().unwrap_or(s),
        };

        // m_in consistency across public MCS
        let m_in_steps = mcss_public.first().map(|inst| inst.m_in).unwrap_or(0);
        if !mcss_public.iter().all(|inst| inst.m_in == m_in_steps) {
            return Err(PiCcsError::InvalidInput("all steps must share the same m_in".into()));
        }

        // Build steps_public from the internal bundles to include mem/lut instances.
        let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = self.steps.iter().map(|bundle| bundle.into()).collect();
        let s_prepared = self.prepared_ccs_for_accumulator(s_norm)?;

        // Validate (or empty) initial accumulator to mirror finalize()
        let seed_me: &[MeInstance<Cmt, F, K>] = match &self.acc0 {
            Some(acc) => {
                acc.check(&self.params, s_prepared)?;
                let acc_m_in = acc.me.first().map(|m| m.m_in).unwrap_or(m_in_steps);
                if acc_m_in != m_in_steps {
                    return Err(PiCcsError::InvalidInput(
                        "initial Accumulator.m_in must match steps' m_in".into(),
                    ));
                }
                // ME inputs are already well-formed (checked by acc.check())
                &acc.me
            }
            None => &[], // k=1
        };

        let step_linking = self
            .step_linking
            .as_ref()
            .filter(|cfg| !cfg.prev_next_equalities.is_empty());

        let outputs = if steps_public.len() > 1 {
            match step_linking {
                Some(cfg) => match verifier_ctx {
                    Some(ctx) => shard::fold_shard_verify_with_step_linking_with_context(
                        self.mode.clone(),
                        tr,
                        &self.params,
                        s_norm,
                        &steps_public,
                        seed_me,
                        run,
                        self.mixers,
                        cfg,
                        ctx,
                    )?,
                    None => shard::fold_shard_verify_with_step_linking(
                        self.mode.clone(),
                        tr,
                        &self.params,
                        s_norm,
                        &steps_public,
                        seed_me,
                        run,
                        self.mixers,
                        cfg,
                    )?,
                },
                None if self.allow_unlinked_steps => match verifier_ctx {
                    Some(ctx) => shard::fold_shard_verify_with_context(
                        self.mode.clone(),
                        tr,
                        &self.params,
                        s_norm,
                        &steps_public,
                        seed_me,
                        run,
                        self.mixers,
                        ctx,
                    )?,
                    None => shard::fold_shard_verify(
                        self.mode.clone(),
                        tr,
                        &self.params,
                        s_norm,
                        &steps_public,
                        seed_me,
                        run,
                        self.mixers,
                    )?,
                },
                None => {
                    let mut msg = "multi-step verification requires step linking; call FoldingSession::set_step_linking(...) or FoldingSession::unsafe_allow_unlinked_steps()".to_string();
                    if let Some(diag) = &self.auto_step_linking_error {
                        msg.push_str(&format!(" (auto step-linking from StepSpec failed: {diag})"));
                    }
                    return Err(PiCcsError::InvalidInput(
                        msg,
                    ));
                }
            }
        } else {
            match verifier_ctx {
                Some(ctx) => shard::fold_shard_verify_with_context(
                    self.mode.clone(),
                    tr,
                    &self.params,
                    s_norm,
                    &steps_public,
                    seed_me,
                    run,
                    self.mixers,
                    ctx,
                )?,
                None => shard::fold_shard_verify(
                    self.mode.clone(),
                    tr,
                    &self.params,
                    s_norm,
                    &steps_public,
                    seed_me,
                    run,
                    self.mixers,
                )?,
            }
        };

        // For CCS-only sessions (no Twist/Shout), val-lane obligations should be empty
        // For Twist+Shout sessions, val-lane obligations are expected and valid
        let has_twist_or_shout = self.has_twist_instances() || self.has_shout_instances();
        if !has_twist_or_shout && !outputs.obligations.val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "CCS-only session verification produced unexpected val-lane obligations".into(),
            ));
        }
        Ok(true)
    }

    /// Verify with output binding, managing the transcript internally.
    pub fn verify_with_output_binding_simple(
        &self,
        s: &CcsStructure<F>,
        mcss_public: &[neo_ccs::McsInstance<Cmt, F>],
        run: &FoldRun,
        ob_cfg: &crate::output_binding::OutputBindingConfig,
    ) -> Result<bool, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
        self.verify_with_output_binding(&mut tr, s, mcss_public, run, ob_cfg)
    }

    /// Verify with output binding using the internally collected steps (and public MCS list).
    pub fn verify_with_output_binding_collected_simple(
        &self,
        s: &CcsStructure<F>,
        run: &FoldRun,
        ob_cfg: &crate::output_binding::OutputBindingConfig,
    ) -> Result<bool, PiCcsError> {
        let mcss_public = self.mcss_public();
        self.verify_with_output_binding_simple(s, &mcss_public, run, ob_cfg)
    }

    pub fn verify_with_output_binding(
        &self,
        tr: &mut Poseidon2Transcript,
        s: &CcsStructure<F>,
        mcss_public: &[neo_ccs::McsInstance<Cmt, F>],
        run: &FoldRun,
        ob_cfg: &crate::output_binding::OutputBindingConfig,
    ) -> Result<bool, PiCcsError> {
        let src_ptr = (s as *const CcsStructure<F>) as usize;
        let verifier_cache = self
            .verifier_ctx
            .as_ref()
            .filter(|cache| cache.src_ptr == src_ptr);
        let verifier_ctx = verifier_cache.map(|cache| &cache.ctx);

        // Normalize CCS (avoid cloning when already normalized or preloaded).
        let s_norm_owned: Option<CcsStructure<F>> = match verifier_cache {
            Some(_) => None,
            None => match normalize_ccs_for_session(s)? {
                Cow::Borrowed(_) => None,
                Cow::Owned(s_norm) => Some(s_norm),
            },
        };
        let s_norm: &CcsStructure<F> = match verifier_cache {
            Some(cache) => cache.s_norm.as_ref().unwrap_or(s),
            None => s_norm_owned.as_ref().unwrap_or(s),
        };

        let m_in_steps = mcss_public.first().map(|inst| inst.m_in).unwrap_or(0);
        if !mcss_public.iter().all(|inst| inst.m_in == m_in_steps) {
            return Err(PiCcsError::InvalidInput("all steps must share the same m_in".into()));
        }

        let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = self.steps.iter().map(|bundle| bundle.into()).collect();
        let s_prepared = self.prepared_ccs_for_accumulator(s_norm)?;

        let seed_me: &[MeInstance<Cmt, F, K>] = match &self.acc0 {
            Some(acc) => {
                acc.check(&self.params, s_prepared)?;
                let acc_m_in = acc.me.first().map(|m| m.m_in).unwrap_or(m_in_steps);
                if acc_m_in != m_in_steps {
                    return Err(PiCcsError::InvalidInput(
                        "initial Accumulator.m_in must match steps' m_in".into(),
                    ));
                }
                &acc.me
            }
            None => &[],
        };

        let step_linking = self
            .step_linking
            .as_ref()
            .filter(|cfg| !cfg.prev_next_equalities.is_empty());

        let outputs = if steps_public.len() > 1 {
            match step_linking {
                Some(cfg) => match verifier_ctx {
                    Some(ctx) => shard::fold_shard_verify_with_output_binding_and_step_linking_with_context(
                        self.mode.clone(),
                        tr,
                        &self.params,
                        s_norm,
                        &steps_public,
                        seed_me,
                        run,
                        self.mixers,
                        ob_cfg,
                        cfg,
                        ctx,
                    )?,
                    None => shard::fold_shard_verify_with_output_binding_and_step_linking(
                        self.mode.clone(),
                        tr,
                        &self.params,
                        s_norm,
                        &steps_public,
                        seed_me,
                        run,
                        self.mixers,
                        ob_cfg,
                        cfg,
                    )?,
                },
                None if self.allow_unlinked_steps => match verifier_ctx {
                    Some(ctx) => shard::fold_shard_verify_with_output_binding_with_context(
                        self.mode.clone(),
                        tr,
                        &self.params,
                        s_norm,
                        &steps_public,
                        seed_me,
                        run,
                        self.mixers,
                        ob_cfg,
                        ctx,
                    )?,
                    None => shard::fold_shard_verify_with_output_binding(
                        self.mode.clone(),
                        tr,
                        &self.params,
                        s_norm,
                        &steps_public,
                        seed_me,
                        run,
                        self.mixers,
                        ob_cfg,
                    )?,
                },
                None => {
                    let mut msg = "multi-step verification requires step linking; call FoldingSession::set_step_linking(...) or FoldingSession::unsafe_allow_unlinked_steps()".to_string();
                    if let Some(diag) = &self.auto_step_linking_error {
                        msg.push_str(&format!(" (auto step-linking from StepSpec failed: {diag})"));
                    }
                    return Err(PiCcsError::InvalidInput(
                        msg,
                    ));
                }
            }
        } else {
            match verifier_ctx {
                Some(ctx) => shard::fold_shard_verify_with_output_binding_with_context(
                    self.mode.clone(),
                    tr,
                    &self.params,
                    s_norm,
                    &steps_public,
                    seed_me,
                    run,
                    self.mixers,
                    ob_cfg,
                    ctx,
                )?,
                None => shard::fold_shard_verify_with_output_binding(
                    self.mode.clone(),
                    tr,
                    &self.params,
                    s_norm,
                    &steps_public,
                    seed_me,
                    run,
                    self.mixers,
                    ob_cfg,
                )?,
            }
        };

        let has_twist_or_shout = self.has_twist_instances() || self.has_shout_instances();
        if !has_twist_or_shout && !outputs.obligations.val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "CCS-only session verification produced unexpected val-lane obligations".into(),
            ));
        }

        Ok(true)
    }
}

impl FoldingSession<AjtaiSModule> {
    /// Build a session with an Ajtai committer and auto-picked Goldilocks parameters.
    ///
    /// This is intended as a “few lines” frontend: it hides Ajtai PP generation and `NeoParams`
    /// selection. The committer dimension uses the session's canonical square padding rule
    /// (`m = max(ccs.n, ccs.m)`) to match internal CCS normalization.
    ///
    /// For deterministic parameters (e.g. in tests), use `new_ajtai_seeded`.
    pub fn new_ajtai(mode: FoldingMode, ccs: &CcsStructure<F>) -> Result<Self, PiCcsError> {
        let m_commit = canonical_ccs_square_dim(ccs);
        let params = NeoParams::goldilocks_auto_r1cs_ccs(m_commit)
            .map_err(|e| PiCcsError::InvalidInput(format!("NeoParams::goldilocks_auto_r1cs_ccs failed: {e}")))?;

        let mut seed = [0u8; 32];
        rand_chacha::rand_core::OsRng
            .try_fill_bytes(&mut seed)
            .map_err(|e| PiCcsError::InvalidInput(format!("OsRng failed: {e}")))?;
        let mut rng = rand_chacha::ChaCha8Rng::from_seed(seed);
        let pp = neo_ajtai::setup_par(&mut rng, D, params.kappa as usize, m_commit)
            .map_err(|e| PiCcsError::InvalidInput(format!("Ajtai setup failed: {e}")))?;
        let committer = AjtaiSModule::new(Arc::new(pp));
        let mut session = FoldingSession::new(mode, params, committer);
        session.commit_m = Some(m_commit);
        Ok(session)
    }

    /// Same as `new_ajtai`, but with a deterministic ChaCha8 seed for reproducible tests/benchmarks.
    ///
    /// This uses the sequential Ajtai setup to avoid any parallelism-related determinism concerns.
    pub fn new_ajtai_seeded(
        mode: FoldingMode,
        ccs: &CcsStructure<F>,
        seed: [u8; 32],
    ) -> Result<Self, PiCcsError> {
        let m_commit = canonical_ccs_square_dim(ccs);
        let params = NeoParams::goldilocks_auto_r1cs_ccs(m_commit)
            .map_err(|e| PiCcsError::InvalidInput(format!("NeoParams::goldilocks_auto_r1cs_ccs failed: {e}")))?;

        let mut rng = rand_chacha::ChaCha8Rng::from_seed(seed);
        let pp = neo_ajtai::setup(&mut rng, D, params.kappa as usize, m_commit)
            .map_err(|e| PiCcsError::InvalidInput(format!("Ajtai setup failed: {e}")))?;
        let committer = AjtaiSModule::new(Arc::new(pp));
        let mut session = FoldingSession::new(mode, params, committer);
        session.commit_m = Some(m_commit);
        Ok(session)
    }
}
