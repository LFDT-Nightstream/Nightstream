//! Π_RLC.V — in-circuit verifier (paper §7.4 steps 1–2).
//!
//! Reduction:  CE(b, ℒ)^{K+k}  →  CE(B, ℒ)    where B = b^k.
//!
//! The verifier samples `ρ_1, …, ρ_{K+k} ∈ 𝒞` from the transcript and
//! recomputes the combined CE claim:
//!
//! ```text
//!   c       = Σ_i ρ_i · c_i      (ring action on commitments)
//!   X       = Σ_i ρ_i · X_i      (ring action on the public-input matrix)
//!   y_j     = Σ_i ρ_i · y_{i,j}  (scalar-K linear combination)
//!   r       = inputs' shared r   (deterministic equality wiring)
//! ```
//!
//! ## What this gadget owns
//!
//! - [`enforce_rlc_commitment_combination`] — `combined.c = Σ_i ρ_i · c_i`
//!   via [`crate::engine::r1cs_circuit::ring_action::enforce_ring_mul_toom3`]
//!   per (lane, pair). Most expensive of the three; for Goldilocks
//!   Appendix B.2 (κ = 18, d = 54, K + k = 15) this is ~450k R1CS rows.
//! - [`enforce_rlc_x_combination`] — `combined.X = Σ_i ρ_i · X_i` per
//!   column.
//! - [`enforce_rlc_y_row_combination`] — K-element row mixing via
//!   `(ρ_i ⊗ Id_𝕂)` on each `(c0, c1)` track.
//! - `with_rhos` allocation variants of all three that accept
//!   pre-allocated ρ wires (from
//!   [`crate::engine::r1cs_circuit::alphabet_sampling::enforce_pi_rlc_rhos_from_transcript`]),
//!   so a single transcript-derived ρ feeds the full c / X / y chain.
//!
//! ## Not in this file
//!
//! - The transcript-derived ρ-sampling itself: see
//!   [`crate::engine::r1cs_circuit::alphabet_sampling`].
//! - Composition (binding to Π_CCS.V outputs and Π_DEC.V inputs): see
//!   [`crate::paper::nifs::circuit::enforce_nifs_v_circuit_with_transcript`].
//! - `r` equality wiring lives in Π_DEC.V's strict mode
//!   ([`crate::paper::reductions::pi_dec_circuit::enforce_r_consistency`]).

use neo_ajtai::Commitment;
use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::ring_action::{
    enforce_ring_action_projection_batch, enforce_ring_mul_toom3, projection_quotient, PROJECTION_QUOTIENT_LEN,
};
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};

/// Wires for one commitment + the matching ρ polynomial coefficients.
///
/// `c_data` is laid out column-major over κ columns of `d` rows each, matching
/// [`neo_ajtai::Commitment::data`]: `data[lane * d + row]` is the `row`-th
/// coefficient of the `lane`-th ring element.
///
/// `rho_coeffs` is `d` consecutive F-columns, in the same order as
/// `neo_math::ring::cf(rq)` — i.e., the first column of the native rotation
/// matrix `RotRho`.
#[derive(Clone, Debug)]
pub struct RlcPairWires {
    pub rho_coeffs: [Var; D],
    pub c_data: Vec<Var>,
    pub kappa: usize,
}

/// Allocate witness for `(K+k)` (ρ, commitment) input pairs plus one expected
/// combined commitment, ready to be constrained.
#[derive(Clone, Debug)]
pub struct RlcCommitmentWires {
    pub inputs: Vec<RlcPairWires>,
    pub combined_c_data: Vec<Var>,
    pub kappa: usize,
}

/// Allocate witness variables for `(K+k)` pairs and the expected combined
/// commitment. No constraints are emitted yet — callers must call
/// [`enforce_rlc_commitment_combination`].
pub fn alloc_rlc_commitment_inputs(
    builder: &mut R1csBuilder,
    rhos_first_col: &[[F; D]],
    inputs: &[Commitment],
    combined: &Commitment,
) -> Result<RlcCommitmentWires, Error> {
    if inputs.is_empty() {
        return Err(Error::Empty);
    }
    if rhos_first_col.len() != inputs.len() {
        return Err(Error::PairCountMismatch {
            rhos: rhos_first_col.len(),
            inputs: inputs.len(),
        });
    }
    let kappa = inputs[0].kappa;
    if combined.kappa != kappa || combined.d != D {
        return Err(Error::ShapeMismatch {
            what: "combined commitment shape",
            expected: format!("(d={D}, kappa={kappa})"),
            got: format!("(d={}, kappa={})", combined.d, combined.kappa),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs.len());
    for (idx, (rho_col, c)) in rhos_first_col.iter().zip(inputs.iter()).enumerate() {
        if c.kappa != kappa || c.d != D {
            return Err(Error::ShapeMismatch {
                what: "input commitment shape",
                expected: format!("(d={D}, kappa={kappa})"),
                got: format!("(d={}, kappa={}) at idx {idx}", c.d, c.kappa),
            });
        }
        let mut rho_coeffs = [Var::ONE; D];
        for (slot, &v) in rho_coeffs.iter_mut().zip(rho_col.iter()) {
            *slot = builder.alloc(v);
        }
        let c_data = builder.alloc_vec(&c.data);
        input_wires.push(RlcPairWires {
            rho_coeffs,
            c_data,
            kappa,
        });
    }
    let combined_c_data = builder.alloc_vec(&combined.data);
    Ok(RlcCommitmentWires {
        inputs: input_wires,
        combined_c_data,
        kappa,
    })
}

/// Allocate commitment-combination wires, reusing **pre-allocated** ρ
/// coefficient wires (e.g., as derived from
/// [`crate::engine::r1cs_circuit::alphabet_sampling::enforce_pi_rlc_rhos_from_transcript`]).
///
/// The caller passes in `rho_wires` ALREADY allocated inside `builder`; we
/// only allocate the `c_data` columns. Use this in the transcript-driven
/// path so the same ρ wires feed commitment, X, and y_row combinations.
pub fn alloc_rlc_commitment_inputs_with_rhos(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    inputs: &[Commitment],
    combined: &Commitment,
) -> Result<RlcCommitmentWires, Error> {
    if inputs.is_empty() {
        return Err(Error::Empty);
    }
    if rho_wires.len() != inputs.len() {
        return Err(Error::PairCountMismatch {
            rhos: rho_wires.len(),
            inputs: inputs.len(),
        });
    }
    let kappa = inputs[0].kappa;
    if combined.kappa != kappa || combined.d != D {
        return Err(Error::ShapeMismatch {
            what: "combined commitment shape",
            expected: format!("(d={D}, kappa={kappa})"),
            got: format!("(d={}, kappa={})", combined.d, combined.kappa),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs.len());
    for (idx, (rho, c)) in rho_wires.iter().zip(inputs.iter()).enumerate() {
        if c.kappa != kappa || c.d != D {
            return Err(Error::ShapeMismatch {
                what: "input commitment shape",
                expected: format!("(d={D}, kappa={kappa})"),
                got: format!("(d={}, kappa={}) at idx {idx}", c.d, c.kappa),
            });
        }
        input_wires.push(RlcPairWires {
            rho_coeffs: *rho,
            c_data: builder.alloc_vec(&c.data),
            kappa,
        });
    }
    let combined_c_data = builder.alloc_vec(&combined.data);
    Ok(RlcCommitmentWires {
        inputs: input_wires,
        combined_c_data,
        kappa,
    })
}

/// Enforce `combined.c = Σ_i ρ_i · c_i` lane-by-lane via the ring-action gadget.
pub fn enforce_rlc_commitment_combination(builder: &mut R1csBuilder, wires: &RlcCommitmentWires) {
    let kappa = wires.kappa;
    for lane in 0..kappa {
        // For this lane, compute Σ_i (ρ_i · c_i_lane) coefficient-wise.
        // Each ring_mul produces D output variables; we then sum them.
        let mut per_pair_out: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
        for pair in &wires.inputs {
            let mut c_lane = [Var::ONE; D];
            for (slot, src) in c_lane
                .iter_mut()
                .zip(pair.c_data[lane * D..(lane + 1) * D].iter())
            {
                *slot = *src;
            }
            let out = enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &c_lane);
            per_pair_out.push(out);
        }

        // For each output coefficient m: combined.c.data[lane * D + m] == Σ_i per_pair_out[i][m].
        for m in 0..D {
            let mut combo = Lc::zero();
            for pair_out in &per_pair_out {
                combo.add_term(pair_out[m], F::ONE);
            }
            let target = wires.combined_c_data[lane * D + m];
            builder.enforce_eq(&Lc::from_var(target), &combo);
        }
    }
}

// ── Projection-checked commitment combination (Road A, candidate E) ──────

/// Native prover companion for
/// [`enforce_rlc_commitment_combination_projection`]: the per-lane
/// division quotients `q_lane` with
/// `Σ_i ρ_i(X)·c_{i,lane}(X) = q_lane(X)·Φ(X) + combined_lane(X)`.
///
/// **Lemma 5 schedule**: the caller absorbs `combined` and every
/// quotient returned here into the transcript **before** squeezing β.
/// Compute-then-absorb-then-squeeze is the soundness; the circuit
/// function below only enforces the algebra.
pub fn rlc_projection_quotients(
    rhos_first_col: &[[F; D]],
    inputs: &[Commitment],
) -> Result<Vec<[F; PROJECTION_QUOTIENT_LEN]>, Error> {
    if inputs.is_empty() {
        return Err(Error::Empty);
    }
    if rhos_first_col.len() != inputs.len() {
        return Err(Error::PairCountMismatch {
            rhos: rhos_first_col.len(),
            inputs: inputs.len(),
        });
    }
    let kappa = inputs[0].kappa;
    let mut per_lane = Vec::with_capacity(kappa);
    for lane in 0..kappa {
        let pairs: Vec<([F; D], [F; D])> = rhos_first_col
            .iter()
            .zip(inputs.iter())
            .map(|(rho, c)| {
                let mut lane_coeffs = [F::ZERO; D];
                lane_coeffs.copy_from_slice(&c.data[lane * D..(lane + 1) * D]);
                (*rho, lane_coeffs)
            })
            .collect();
        let (_, q) = projection_quotient(&pairs);
        per_lane.push(q);
    }
    Ok(per_lane)
}

/// Projection-checked variant of
/// [`enforce_rlc_commitment_combination`] — Road A of the enc(F')
/// decision (encoding.md candidate E; soundness case: security-note
/// Lemma 5). Per κ-lane, **one** polynomial identity at β replaces the
/// `(K+k)` Toom-3 ring products: the inputs batch inside the identity
/// because the consumer is the aggregate mix (Lemma 5's batching rule;
/// J = κ identities from this client).
///
/// `powers` is the shared [`enforce_beta_ladder`] output for a β that
/// the caller squeezed AFTER absorbing the inputs, `combined`, and the
/// `quotients` (which must be the values
/// [`rlc_projection_quotients`] computed — the returned wires exist so
/// the integration can bind them to the absorbed values, Lemma 5
/// adoption audit item 1).
pub fn enforce_rlc_commitment_combination_projection(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    wires: &RlcCommitmentWires,
    quotients: &[[F; PROJECTION_QUOTIENT_LEN]],
) -> Result<Vec<[Var; PROJECTION_QUOTIENT_LEN]>, Error> {
    let kappa = wires.kappa;
    if quotients.len() != kappa {
        return Err(Error::ShapeMismatch {
            what: "projection quotient count",
            expected: format!("kappa = {kappa}"),
            got: format!("{}", quotients.len()),
        });
    }
    let mut quotient_wires = Vec::with_capacity(kappa);
    for lane in 0..kappa {
        // Owned per-pair lane arrays first (the batch API borrows).
        let pair_arrays: Vec<([Var; D], [Var; D])> = wires
            .inputs
            .iter()
            .map(|pair| {
                let mut c_lane = [Var::ONE; D];
                for (slot, src) in c_lane
                    .iter_mut()
                    .zip(pair.c_data[lane * D..(lane + 1) * D].iter())
                {
                    *slot = *src;
                }
                (pair.rho_coeffs, c_lane)
            })
            .collect();
        let pair_refs: Vec<(&[Var; D], &[Var; D])> = pair_arrays.iter().map(|(rho, c)| (rho, c)).collect();

        let mut out_lane = [Var::ONE; D];
        for (slot, src) in out_lane
            .iter_mut()
            .zip(wires.combined_c_data[lane * D..(lane + 1) * D].iter())
        {
            *slot = *src;
        }

        let q_wires: [Var; PROJECTION_QUOTIENT_LEN] = quotients[lane].map(|value| builder.alloc(value));
        enforce_ring_action_projection_batch(builder, powers, &pair_refs, &out_lane, &q_wires);
        quotient_wires.push(q_wires);
    }
    Ok(quotient_wires)
}

// ── X-combination: `combined.X = Σ ρ_i · X_i` ─────────────────────────────

/// Wires for one input's X matrix + the matching ρ polynomial coefficients.
///
/// `x_flat` is row-major over a `D × m_in` matrix: `x_flat[rr * m_in + col]`.
#[derive(Clone, Debug)]
pub struct RlcXPairWires {
    pub rho_coeffs: [Var; D],
    pub x_flat: Vec<Var>,
    pub m_in: usize,
}

#[derive(Clone, Debug)]
pub struct RlcXWires {
    pub inputs: Vec<RlcXPairWires>,
    pub combined_x_flat: Vec<Var>,
    pub m_in: usize,
}

/// Allocate witness for `(K+k)` `(ρ, X_i)` pairs plus the expected combined X.
pub fn alloc_rlc_x_inputs(
    builder: &mut R1csBuilder,
    rhos_first_col: &[[F; D]],
    inputs_x: &[neo_ccs::Mat<F>],
    combined_x: &neo_ccs::Mat<F>,
) -> Result<RlcXWires, Error> {
    if inputs_x.is_empty() {
        return Err(Error::Empty);
    }
    if rhos_first_col.len() != inputs_x.len() {
        return Err(Error::PairCountMismatch {
            rhos: rhos_first_col.len(),
            inputs: inputs_x.len(),
        });
    }
    let m_in = inputs_x[0].cols();
    if combined_x.rows() != D || combined_x.cols() != m_in {
        return Err(Error::ShapeMismatch {
            what: "combined X shape",
            expected: format!("(rows=D, cols={m_in})"),
            got: format!("(rows={}, cols={})", combined_x.rows(), combined_x.cols()),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs_x.len());
    for (idx, (rho_col, x_i)) in rhos_first_col.iter().zip(inputs_x.iter()).enumerate() {
        if x_i.rows() != D || x_i.cols() != m_in {
            return Err(Error::ShapeMismatch {
                what: "input X shape",
                expected: format!("(rows=D, cols={m_in})"),
                got: format!("(rows={}, cols={}) at idx {idx}", x_i.rows(), x_i.cols()),
            });
        }
        let mut rho_coeffs = [Var::ONE; D];
        for (slot, &v) in rho_coeffs.iter_mut().zip(rho_col.iter()) {
            *slot = builder.alloc(v);
        }
        let x_flat = builder.alloc_vec(x_i.as_slice());
        input_wires.push(RlcXPairWires {
            rho_coeffs,
            x_flat,
            m_in,
        });
    }
    let combined_x_flat = builder.alloc_vec(combined_x.as_slice());
    Ok(RlcXWires {
        inputs: input_wires,
        combined_x_flat,
        m_in,
    })
}

/// Variant of [`alloc_rlc_x_inputs`] that reuses pre-allocated ρ wires.
pub fn alloc_rlc_x_inputs_with_rhos(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    inputs_x: &[neo_ccs::Mat<F>],
    combined_x: &neo_ccs::Mat<F>,
) -> Result<RlcXWires, Error> {
    if inputs_x.is_empty() {
        return Err(Error::Empty);
    }
    if rho_wires.len() != inputs_x.len() {
        return Err(Error::PairCountMismatch {
            rhos: rho_wires.len(),
            inputs: inputs_x.len(),
        });
    }
    let m_in = inputs_x[0].cols();
    if combined_x.rows() != D || combined_x.cols() != m_in {
        return Err(Error::ShapeMismatch {
            what: "combined X shape",
            expected: format!("(rows=D, cols={m_in})"),
            got: format!("(rows={}, cols={})", combined_x.rows(), combined_x.cols()),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs_x.len());
    for (idx, (rho, x_i)) in rho_wires.iter().zip(inputs_x.iter()).enumerate() {
        if x_i.rows() != D || x_i.cols() != m_in {
            return Err(Error::ShapeMismatch {
                what: "input X shape",
                expected: format!("(rows=D, cols={m_in})"),
                got: format!("(rows={}, cols={}) at idx {idx}", x_i.rows(), x_i.cols()),
            });
        }
        input_wires.push(RlcXPairWires {
            rho_coeffs: *rho,
            x_flat: builder.alloc_vec(x_i.as_slice()),
            m_in,
        });
    }
    let combined_x_flat = builder.alloc_vec(combined_x.as_slice());
    Ok(RlcXWires {
        inputs: input_wires,
        combined_x_flat,
        m_in,
    })
}

/// Enforce `combined.X = Σ_i ρ_i · X_i` column-by-column.
///
/// `X` has logical shape `D × m_in`, but `project_x_from_witness_mat` only
/// populates `active_cols = ceil(m_in / D)` ring columns; the rest are
/// structural zeros. We enforce that invariant in-circuit on both the
/// inputs and the combined output, then ring-fold only the active columns
/// — a `(m_in / active_cols) × ` speedup for the X fold, which is the
/// second-largest contributor to F'-recursive's row count.
pub fn enforce_rlc_x_combination(builder: &mut R1csBuilder, wires: &RlcXWires) {
    let m_in = wires.m_in;
    let active_cols = crate::paper::relations::superneo_public_x_cols(m_in);

    // Inactive input columns must be zero. Native `project_x_from_witness_mat`
    // produces these zeros by construction; pinning them in-circuit closes
    // the gap an adversary could otherwise smuggle the X fold around.
    for pair in &wires.inputs {
        for rr in 0..D {
            for col in active_cols..m_in {
                builder.enforce_eq(&Lc::from_var(pair.x_flat[rr * m_in + col]), &Lc::zero());
            }
        }
    }

    // Ring-fold only the active columns.
    let mut per_pair_per_col: Vec<Vec<[Var; D]>> = Vec::with_capacity(wires.inputs.len());
    for pair in &wires.inputs {
        let mut per_col = Vec::with_capacity(active_cols);
        for col in 0..active_cols {
            let mut x_col = [Var::ONE; D];
            for (rr, slot) in x_col.iter_mut().enumerate() {
                *slot = pair.x_flat[rr * m_in + col];
            }
            let out = enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &x_col);
            per_col.push(out);
        }
        per_pair_per_col.push(per_col);
    }

    // combined.X[(rr, col)] == Σ_i per_pair_per_col[i][col][rr] for active cols.
    for rr in 0..D {
        for col in 0..active_cols {
            let mut combo = Lc::zero();
            for per_col in &per_pair_per_col {
                combo.add_term(per_col[col][rr], F::ONE);
            }
            let target = wires.combined_x_flat[rr * m_in + col];
            builder.enforce_eq(&Lc::from_var(target), &combo);
        }
    }

    // Inactive combined columns must be zero (the prover supplies these
    // wires too; without this, they could be anything).
    for rr in 0..D {
        for col in active_cols..m_in {
            builder.enforce_eq(&Lc::from_var(wires.combined_x_flat[rr * m_in + col]), &Lc::zero());
        }
    }
}

// ── y_ring-combination: `combined.y_ring[j] = Σ ρ_i · y_{i,j}` over R_K ────

/// Wires for one input's y_ring[j] row + the matching ρ polynomial.
///
/// `y_c0[kk]`, `y_c1[kk]` are the base-field limbs of `y_ring[j][kk] ∈ K`.
/// Convention: K-element `(c0, c1)` with `c0 + c1·X` and `X² = W`.
#[derive(Clone, Debug)]
pub struct RlcYRowPairWires {
    pub rho_coeffs: [Var; D],
    pub y_c0: [Var; D],
    pub y_c1: [Var; D],
}

#[derive(Clone, Debug)]
pub struct RlcYRowWires {
    pub inputs: Vec<RlcYRowPairWires>,
    pub combined_c0: [Var; D],
    pub combined_c1: [Var; D],
}

/// Allocate witness for one j-row of y_ring: `(K+k)` pairs `(ρ, y_{i,j})` plus
/// the expected combined `y_j`. Each `y_{i,j}` is a length-D `Vec<K>`.
pub fn alloc_rlc_y_row_inputs(
    builder: &mut R1csBuilder,
    rhos_first_col: &[[F; D]],
    inputs_y: &[Vec<K>],
    combined_y: &[K],
) -> Result<RlcYRowWires, Error> {
    if inputs_y.is_empty() {
        return Err(Error::Empty);
    }
    if rhos_first_col.len() != inputs_y.len() {
        return Err(Error::PairCountMismatch {
            rhos: rhos_first_col.len(),
            inputs: inputs_y.len(),
        });
    }
    for (idx, y) in inputs_y.iter().enumerate() {
        if y.len() != D {
            return Err(Error::ShapeMismatch {
                what: "y_ring row length",
                expected: format!("{D}"),
                got: format!("{} at idx {idx}", y.len()),
            });
        }
    }
    if combined_y.len() != D {
        return Err(Error::ShapeMismatch {
            what: "combined y_ring row length",
            expected: format!("{D}"),
            got: format!("{}", combined_y.len()),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs_y.len());
    for (rho_col, y_i) in rhos_first_col.iter().zip(inputs_y.iter()) {
        let mut rho_coeffs = [Var::ONE; D];
        for (slot, &v) in rho_coeffs.iter_mut().zip(rho_col.iter()) {
            *slot = builder.alloc(v);
        }
        let mut y_c0 = [Var::ONE; D];
        let mut y_c1 = [Var::ONE; D];
        for (kk, val) in y_i.iter().enumerate() {
            let [c0, c1] = val.as_coeffs();
            y_c0[kk] = builder.alloc(c0);
            y_c1[kk] = builder.alloc(c1);
        }
        input_wires.push(RlcYRowPairWires { rho_coeffs, y_c0, y_c1 });
    }
    let mut combined_c0 = [Var::ONE; D];
    let mut combined_c1 = [Var::ONE; D];
    for (rr, val) in combined_y.iter().enumerate() {
        let [c0, c1] = val.as_coeffs();
        combined_c0[rr] = builder.alloc(c0);
        combined_c1[rr] = builder.alloc(c1);
    }
    Ok(RlcYRowWires {
        inputs: input_wires,
        combined_c0,
        combined_c1,
    })
}

/// Variant of [`alloc_rlc_y_row_inputs`] that reuses pre-allocated ρ wires.
pub fn alloc_rlc_y_row_inputs_with_rhos(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    inputs_y: &[Vec<K>],
    combined_y: &[K],
) -> Result<RlcYRowWires, Error> {
    if inputs_y.is_empty() {
        return Err(Error::Empty);
    }
    if rho_wires.len() != inputs_y.len() {
        return Err(Error::PairCountMismatch {
            rhos: rho_wires.len(),
            inputs: inputs_y.len(),
        });
    }
    for (idx, y) in inputs_y.iter().enumerate() {
        if y.len() != D {
            return Err(Error::ShapeMismatch {
                what: "y_ring row length",
                expected: format!("{D}"),
                got: format!("{} at idx {idx}", y.len()),
            });
        }
    }
    if combined_y.len() != D {
        return Err(Error::ShapeMismatch {
            what: "combined y_ring row length",
            expected: format!("{D}"),
            got: format!("{}", combined_y.len()),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs_y.len());
    for (rho, y_i) in rho_wires.iter().zip(inputs_y.iter()) {
        let mut y_c0 = [Var::ONE; D];
        let mut y_c1 = [Var::ONE; D];
        for (kk, val) in y_i.iter().enumerate() {
            let [c0, c1] = val.as_coeffs();
            y_c0[kk] = builder.alloc(c0);
            y_c1[kk] = builder.alloc(c1);
        }
        input_wires.push(RlcYRowPairWires {
            rho_coeffs: *rho,
            y_c0,
            y_c1,
        });
    }
    let mut combined_c0 = [Var::ONE; D];
    let mut combined_c1 = [Var::ONE; D];
    for (rr, val) in combined_y.iter().enumerate() {
        let [c0, c1] = val.as_coeffs();
        combined_c0[rr] = builder.alloc(c0);
        combined_c1[rr] = builder.alloc(c1);
    }
    Ok(RlcYRowWires {
        inputs: input_wires,
        combined_c0,
        combined_c1,
    })
}

/// Enforce `combined.y[rr] = Σ_i (M_{ρ_i} · y_{i})[rr]` for one j-row.
///
/// `M_{ρ_i}` acts on a `K`-valued vector by acting separately on each base-field
/// limb. Reuses [`enforce_ring_mul_toom3`] on each `(c0, c1)` track.
pub fn enforce_rlc_y_row_combination(builder: &mut R1csBuilder, wires: &RlcYRowWires) {
    let mut per_pair_c0: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
    let mut per_pair_c1: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
    for pair in &wires.inputs {
        per_pair_c0.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &pair.y_c0));
        per_pair_c1.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &pair.y_c1));
    }
    for rr in 0..D {
        let mut combo0 = Lc::zero();
        let mut combo1 = Lc::zero();
        for (p0, p1) in per_pair_c0.iter().zip(per_pair_c1.iter()) {
            combo0.add_term(p0[rr], F::ONE);
            combo1.add_term(p1[rr], F::ONE);
        }
        builder.enforce_eq(&Lc::from_var(wires.combined_c0[rr]), &combo0);
        builder.enforce_eq(&Lc::from_var(wires.combined_c1[rr]), &combo1);
    }
}

// ── SplitNc NC-channel support (sub-step RLCDEC) ──────────────────────────

/// Wires for a length-`d_pad` K-vector input under RLC combination, where
/// `d_pad = D.next_power_of_two()` is the Ajtai-padded ring dimension.
///
/// Native SplitNc emits `y_ring[j]` rows and `y_zcol` columns as canonical
/// length-`d_pad` `Vec<K>` values: real data in `0..D`, zero padding in
/// `[D, d_pad)`. The rotation matrix `ρ_i` acts only on the first `D` lanes
/// (it's a `D × D` matrix); the tail `[D, d_pad)` of the combined output is
/// identically zero in native:
///
/// ```text
/// combined[0..D]      = Σ_i (ρ_i · input_i[0..D])
/// combined[D..d_pad]  = 0
/// ```
///
/// We mirror that in-circuit: rotation on the first `D` lanes (via
/// [`enforce_ring_mul_toom3`]) plus lane-wise zero pins on every input tail
/// and the combined tail. The input-tail pins are redundant when this helper
/// is called from NIFS.V after SplitNc output canonicalization, but they make
/// the Π_RLC helper's own contract self-contained and prevent future callers
/// from treating padded lanes as unconstrained sidecar data.
#[derive(Clone, Debug)]
pub struct RlcPaddedKVectorPairWires {
    pub rho_coeffs: [Var; D],
    /// Length `d_pad` per K-limb. Lanes `[0, D)` are folded; lanes
    /// `[D, d_pad)` exist for shape parity with native but are not consumed
    /// by the combination.
    pub y_c0: Vec<Var>,
    pub y_c1: Vec<Var>,
}

#[derive(Clone, Debug)]
pub struct RlcPaddedKVectorWires {
    pub inputs: Vec<RlcPaddedKVectorPairWires>,
    pub combined_c0: Vec<Var>,
    pub combined_c1: Vec<Var>,
    pub d_pad: usize,
}

fn alloc_padded_inputs_inner(
    builder: &mut R1csBuilder,
    inputs_y: &[Vec<K>],
    combined_y: &[K],
    d_pad: usize,
    rho_provider: impl Fn(&mut R1csBuilder, usize) -> [Var; D],
) -> Result<RlcPaddedKVectorWires, Error> {
    if inputs_y.is_empty() {
        return Err(Error::Empty);
    }
    if d_pad < D {
        return Err(Error::ShapeMismatch {
            what: "d_pad < D",
            expected: format!(">= {D}"),
            got: format!("{d_pad}"),
        });
    }
    for (idx, y) in inputs_y.iter().enumerate() {
        if y.len() != d_pad {
            return Err(Error::ShapeMismatch {
                what: "padded y length",
                expected: format!("{d_pad}"),
                got: format!("{} at idx {idx}", y.len()),
            });
        }
    }
    if combined_y.len() != d_pad {
        return Err(Error::ShapeMismatch {
            what: "combined padded y length",
            expected: format!("{d_pad}"),
            got: format!("{}", combined_y.len()),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs_y.len());
    for (idx, y_i) in inputs_y.iter().enumerate() {
        let rho_coeffs = rho_provider(builder, idx);
        let mut y_c0 = Vec::with_capacity(d_pad);
        let mut y_c1 = Vec::with_capacity(d_pad);
        for val in y_i.iter() {
            let [c0, c1] = val.as_coeffs();
            y_c0.push(builder.alloc(c0));
            y_c1.push(builder.alloc(c1));
        }
        input_wires.push(RlcPaddedKVectorPairWires { rho_coeffs, y_c0, y_c1 });
    }
    let mut combined_c0 = Vec::with_capacity(d_pad);
    let mut combined_c1 = Vec::with_capacity(d_pad);
    for val in combined_y.iter() {
        let [c0, c1] = val.as_coeffs();
        combined_c0.push(builder.alloc(c0));
        combined_c1.push(builder.alloc(c1));
    }
    Ok(RlcPaddedKVectorWires {
        inputs: input_wires,
        combined_c0,
        combined_c1,
        d_pad,
    })
}

/// Allocate witness wires for a padded K-vector RLC: `(K+k)` pairs `(ρ_i,
/// input_i.padded_y)` plus the expected `combined.padded_y`. Each input is
/// a length-`d_pad` `Vec<K>`.
///
/// Replaces the broken D-only alias previously named `alloc_rlc_y_zcol_inputs`.
pub fn alloc_rlc_padded_k_vector_inputs(
    builder: &mut R1csBuilder,
    rhos_first_col: &[[F; D]],
    inputs_y: &[Vec<K>],
    combined_y: &[K],
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    if rhos_first_col.len() != inputs_y.len() {
        return Err(Error::PairCountMismatch {
            rhos: rhos_first_col.len(),
            inputs: inputs_y.len(),
        });
    }
    alloc_padded_inputs_inner(builder, inputs_y, combined_y, d_pad, |b, idx| {
        let mut rho = [Var::ONE; D];
        for (slot, &v) in rho.iter_mut().zip(rhos_first_col[idx].iter()) {
            *slot = b.alloc(v);
        }
        rho
    })
}

/// Variant of [`alloc_rlc_padded_k_vector_inputs`] that reuses
/// pre-allocated ρ wires from the same transcript-derived ρ-sampler.
pub fn alloc_rlc_padded_k_vector_inputs_with_rhos(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    inputs_y: &[Vec<K>],
    combined_y: &[K],
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    if rho_wires.len() != inputs_y.len() {
        return Err(Error::PairCountMismatch {
            rhos: rho_wires.len(),
            inputs: inputs_y.len(),
        });
    }
    alloc_padded_inputs_inner(builder, inputs_y, combined_y, d_pad, |_b, idx| rho_wires[idx])
}

/// Enforce `combined.padded_y = Σ_i ρ_i · input_i.padded_y`:
/// - First `D` lanes via rotation (mirrors native `M_{ρ_i}` action).
/// - Input and combined tail lanes `[D, d_pad)` constrained to zero.
///
/// SplitNc requires this for both `y_zcol` (NC channel) and per-`j`
/// `y_ring` rows when consumed at production shape (`D = 54`, `d_pad = 64`).
pub fn enforce_rlc_padded_k_vector_combination(builder: &mut R1csBuilder, wires: &RlcPaddedKVectorWires) {
    // 1. Apply rotation matrix ρ_i to the first D lanes of each input.
    let mut per_pair_c0: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
    let mut per_pair_c1: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
    for pair in &wires.inputs {
        // ring_mul expects [Var; D] inputs — take the first D lanes only.
        let mut y0_d = [Var::ONE; D];
        let mut y1_d = [Var::ONE; D];
        for i in 0..D {
            y0_d[i] = pair.y_c0[i];
            y1_d[i] = pair.y_c1[i];
        }
        per_pair_c0.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &y0_d));
        per_pair_c1.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &y1_d));
    }

    // 2. Sum the rotated outputs into combined[0..D] lane-wise.
    for rr in 0..D {
        let mut combo0 = Lc::zero();
        let mut combo1 = Lc::zero();
        for (p0, p1) in per_pair_c0.iter().zip(per_pair_c1.iter()) {
            combo0.add_term(p0[rr], F::ONE);
            combo1.add_term(p1[rr], F::ONE);
        }
        builder.enforce_eq(&Lc::from_var(wires.combined_c0[rr]), &combo0);
        builder.enforce_eq(&Lc::from_var(wires.combined_c1[rr]), &combo1);
    }

    // 3. Tail lanes [D, d_pad) must equal zero. Native inputs are
    // canonicalized to zero padding and native outputs leave the tail zero.
    for pair in &wires.inputs {
        for rr in D..wires.d_pad {
            builder.enforce_eq(&Lc::from_var(pair.y_c0[rr]), &Lc::zero());
            builder.enforce_eq(&Lc::from_var(pair.y_c1[rr]), &Lc::zero());
        }
    }
    for rr in D..wires.d_pad {
        builder.enforce_eq(&Lc::from_var(wires.combined_c0[rr]), &Lc::zero());
        builder.enforce_eq(&Lc::from_var(wires.combined_c1[rr]), &Lc::zero());
    }
}

// ── Backward-compatible y_zcol aliases that delegate to the padded helper ──
//
// Synthetic tests in `tests/reductions/pi_rlc.rs` use `d_pad = D` shape
// (no tail to zero). Real SplitNc callers pass `d_pad = D.next_power_of_two()`.

/// Allocate y_zcol RLC wires. `inputs_y_zcol[i].len()` and
/// `combined_y_zcol.len()` must equal `d_pad`.
pub fn alloc_rlc_y_zcol_inputs(
    builder: &mut R1csBuilder,
    rhos_first_col: &[[F; D]],
    inputs_y_zcol: &[Vec<K>],
    combined_y_zcol: &[K],
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    alloc_rlc_padded_k_vector_inputs(builder, rhos_first_col, inputs_y_zcol, combined_y_zcol, d_pad)
}

/// Variant of [`alloc_rlc_y_zcol_inputs`] reusing pre-allocated ρ wires.
pub fn alloc_rlc_y_zcol_inputs_with_rhos(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    inputs_y_zcol: &[Vec<K>],
    combined_y_zcol: &[K],
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    alloc_rlc_padded_k_vector_inputs_with_rhos(builder, rho_wires, inputs_y_zcol, combined_y_zcol, d_pad)
}

/// Enforce `combined.y_zcol = Σ_i ρ_i · input_i.y_zcol` with native
/// rotation-on-first-D + zero-tail semantics.
pub fn enforce_rlc_y_zcol_combination(builder: &mut R1csBuilder, wires: &RlcPaddedKVectorWires) {
    enforce_rlc_padded_k_vector_combination(builder, wires);
}

/// Enforce `combined.s_col == input_i.s_col` for every `i`. The NC
/// column-domain point is shared by every input CE claim and carried
/// through to the combined parent — Π_RLC doesn't mix `s_col`, it just
/// asserts agreement.
///
/// Mirrors native `pi_rlc::prove`'s implicit s_col propagation: the
/// engine's combined CE inherits `inputs[0].s_col`, and every other input
/// must already equal it (the native sumcheck flow assumes this).
pub fn enforce_rlc_s_col_consistency(
    builder: &mut R1csBuilder,
    input_s_cols: &[Vec<KVar>],
    combined_s_col: &[KVar],
) -> Result<(), Error> {
    if input_s_cols.is_empty() {
        return Err(Error::Empty);
    }
    let len = combined_s_col.len();
    for (idx, s) in input_s_cols.iter().enumerate() {
        if s.len() != len {
            return Err(Error::ShapeMismatch {
                what: "s_col length",
                expected: format!("{len}"),
                got: format!("{} at idx {idx}", s.len()),
            });
        }
        for (a, b) in s.iter().zip(combined_s_col.iter()) {
            builder.enforce_eq(&Lc::from_var(a.c0), &Lc::from_var(b.c0));
            builder.enforce_eq(&Lc::from_var(a.c1), &Lc::from_var(b.c1));
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("\u{03A0}_RLC.V: empty input set")]
    Empty,
    #[error("\u{03A0}_RLC.V: |rhos| ({rhos}) \u{2260} |inputs| ({inputs})")]
    PairCountMismatch { rhos: usize, inputs: usize },
    #[error("\u{03A0}_RLC.V: shape mismatch — {what}: expected {expected}, got {got}")]
    ShapeMismatch {
        what: &'static str,
        expected: String,
        got: String,
    },
}
