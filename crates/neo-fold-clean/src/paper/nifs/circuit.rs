//! NIFS.V composition in-circuit — Π_CCS.V → Π_RLC.V → Π_DEC.V.
//!
//! Production NIFS.V verifier: accepts proofs produced by the native
//! [`crate::paper::nifs::prove`] path bit-for-bit (transcript-state parity)
//! and identity-for-identity (algebraic-check parity). F' R1CS embeds this
//! gadget verbatim in its recursive step.
//!
//! ## Composition
//!
//! ```text
//! 1. enforce_split_nc_pi_ccs_v(transcript, fresh, running, proof.pi_ccs)
//!      → derived.{r_prime, s_col_prime, outputs, fresh_x, running_c_data}
//!    (Internally: header_digest catch-up squeeze, so transcript now
//!     matches what native engine::optimized::verify_pi_ccs leaves.)
//!
//! 2. ρ ← enforce_pi_rlc_rhos_from_transcript(transcript, k_total)
//!    (Π_RLC samples ρ AFTER catch-up, matching native pi_rlc::verify.)
//!
//! 3. Allocate parent + children CE wires (msg.combined, msg.children).
//!    enforce_split_nc_d_pad_shape verifies y_ring[j]/y_zcol lane counts.
//!
//! 4. Π_RLC.V folds:
//!      - commitment:  parent.c    = Σ_i ρ_i · output_i.c
//!      - X:           parent.X    = Σ_i ρ_i · output_i.X
//!      - per j:       parent.y_ring[j] = Σ_i ρ_i · output_i.y_ring[j]
//!                     (padded helper: rotation on first D, zero on tail)
//!      - y_zcol:      parent.y_zcol = Σ_i ρ_i · output_i.y_zcol
//!                     (same padded shape)
//!      - s_col:       output_i.s_col == parent.s_col (consistency)
//!
//! 5. Π_DEC.V strict: b-ary recomposition for c/X/y_ring plus r/s_col
//!    consistency parent↔child. No y_zcol recomposition and no unsigned
//!    X bitness check; this mirrors native `verify_dec_public`.
//!
//! 6. Point binding: dec_wires.parent.r == ccs.r_prime.
//! ```
//!
//! All wires consumed downstream by F' R1CS (parent commitment, fresh X,
//! running commitment data) are surfaced via [`NifsVOutputs`].

use neo_math::ring::D;
use neo_math::{F, K};

use crate::engine::r1cs_circuit::alphabet_sampling::enforce_pi_rlc_rhos_from_transcript;
use crate::engine::r1cs_circuit::builder::{Lc, Var};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_split_nc_pi_ccs_v, SplitNcPiCcsOutputWires, SplitNcPiCcsVConfig, SplitNcPiCcsVMessages,
};
use crate::paper::reductions::pi_dec_circuit::{
    self, alloc_dec_inputs, enforce_dec_v_strict, enforce_split_nc_d_pad_shape, DecInputWires,
};
use crate::paper::reductions::pi_rlc_circuit::{
    enforce_rlc_commitment_combination, enforce_rlc_padded_k_vector_combination, enforce_rlc_s_col_consistency,
    enforce_rlc_x_combination, RlcCommitmentWires, RlcPaddedKVectorPairWires, RlcPaddedKVectorWires, RlcPairWires,
    RlcXPairWires, RlcXWires,
};
use crate::paper::reductions::{pi_ccs, pi_ccs_split_nc_circuit};
use crate::paper::relations::{CcsClaim, CeClaim};

/// Configuration for one NIFS.V step.
pub struct NifsVCircuitConfig<'a> {
    pub pi_ccs: SplitNcPiCcsVConfig<'a>,
}

/// Witness/protocol messages from a real native `nifs::prove` proof.
///
/// `pi_ccs` carries the Π_CCS sumcheck transcript + outputs; `combined`
/// is the Π_RLC parent CE claim; `children` is the Π_DEC k-fold output
/// (the new running accumulator).
pub struct NifsVCircuitMessages<'a> {
    pub fresh: &'a [CcsClaim],
    pub running: &'a [CeClaim],
    pub running_parent_authority: Option<&'a CeClaim>,
    pub pi_ccs: &'a pi_ccs::Proof,
    pub combined: &'a CeClaim,
    pub children: &'a [CeClaim],
}

/// Output wires from one NIFS.V step that F' R1CS composition consumes.
pub struct NifsVOutputs {
    /// `Σ b^i · child_i.c.data` — the Π_DEC parent commitment's data wires.
    /// F' R1CS hashes this to form the new `acc_digest`.
    pub parent_c_data: Vec<Var>,
    /// Fresh CCS instances' public-input wires `[fresh_idx][x_lane]`.
    /// F' R1CS uses these to enforce the HyperNova recursive link.
    pub fresh_x: Vec<Vec<Var>>,
    /// Per-running-claim commitment data wires `[running_idx][lane]`.
    /// Kept for callers that need the per-claim view; F' R1CS binds the
    /// running accumulator via the digest below, not via re-hashing
    /// these wires.
    pub running_c_data: Vec<Vec<Var>>,
    /// Four-lane Poseidon2 digest of the running accumulator's b-ary-
    /// recomposed commitment data. Already computed inside the SplitNc
    /// Π_CCS verifier as the ME-input accumulator handle. F' R1CS
    /// reuses this for its `acc_digest_in` binding, avoiding a duplicate
    /// Poseidon2 chain.
    pub running_acc_digest: [Var; 4],
    /// Per-running-claim CE-claim wire bundles — the running input to
    /// this NIFS.V step. The decider's CE-continuity gate compares the
    /// *previous* step's [`Self::children`] against this step's
    /// `running` to enforce step-to-step accumulator equality beyond
    /// commitment-only binding.
    pub running: Vec<SplitNcPiCcsOutputWires>,
    /// Π_RLC parent authority for [`Self::running`], if running is non-empty.
    pub running_parent_authority: Option<SplitNcPiCcsOutputWires>,
    /// Current step's Π_RLC parent, checked by Π_DEC against
    /// [`Self::children`] and carried as the next step's parent authority.
    pub parent: pi_dec_circuit::CeClaimWires,
    /// Per-child CE-claim wires emitted by Π_DEC — the claims that
    /// become the *next* step's `running`. The decider's
    /// CE-continuity gate pins `prev.children == next.running` element
    /// by element.
    pub children: Vec<pi_dec_circuit::CeClaimWires>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("NIFS.V circuit: {0}")]
    Inner(String),
    #[error(transparent)]
    PiCcs(#[from] pi_ccs_split_nc_circuit::Error),
    #[error(transparent)]
    PiDec(#[from] crate::paper::reductions::pi_dec_circuit::Error),
    #[error(transparent)]
    PiRlc(#[from] crate::paper::reductions::pi_rlc_circuit::Error),
}

/// Enforce NIFS.V on top of `transcript`. See module docstring for the
/// composition order.
pub fn enforce_nifs_v_circuit_with_transcript(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &NifsVCircuitConfig<'_>,
    transcript: &mut TranscriptGadget,
    msg: &NifsVCircuitMessages<'_>,
) -> Result<NifsVOutputs, Error> {
    // ── 1. Π_CCS.V SplitNc verifier ───────────────────────────────────────
    let ccs = enforce_split_nc_pi_ccs_v(
        builder,
        transcript,
        &cfg.pi_ccs,
        &SplitNcPiCcsVMessages {
            fresh: msg.fresh,
            running: msg.running,
            running_parent_authority: msg.running_parent_authority,
            outputs: &msg.pi_ccs.outputs,
            sumcheck_rounds_fe: &msg.pi_ccs.sumcheck.sumcheck_rounds,
            sumcheck_rounds_nc: &msg.pi_ccs.sumcheck.sumcheck_rounds_nc,
            header_digest: &msg.pi_ccs.sumcheck.header_digest,
        },
    )?;
    let d_pad = 1usize << cfg.pi_ccs.ell_d;
    let k_total = ccs.outputs.len();
    if k_total == 0 {
        return Err(Error::Inner("Π_CCS.V emitted zero outputs".into()));
    }
    let kappa = ccs.outputs[0].c_kappa;
    let m_in = ccs.outputs[0].x_cols;

    // ── 2. Π_RLC ρ sampling (AFTER catch-up squeeze inside CCS.V) ─────────
    let rho_wires = enforce_pi_rlc_rhos_from_transcript(builder, transcript, k_total);

    // ── 3. Parent + children DEC wires + SplitNc shape check ──────────────
    let dec_wires = alloc_dec_inputs(builder, msg.combined, msg.children);
    enforce_split_nc_d_pad_shape(&dec_wires, d_pad)?;

    // ── 4. Π_RLC.V folds: c, X, per-j y_ring, y_zcol, s_col ──────────────
    enforce_rlc_commitment_fold(builder, &rho_wires, &ccs.outputs, &dec_wires, kappa)?;
    enforce_rlc_x_fold(builder, &rho_wires, &ccs.outputs, &dec_wires, m_in)?;
    let t = cfg.pi_ccs.structure.t();
    for j in 0..t {
        enforce_rlc_y_ring_row_fold(builder, &rho_wires, &ccs.outputs, &dec_wires, j, d_pad)?;
    }
    enforce_rlc_y_zcol_fold(builder, &rho_wires, &ccs.outputs, &dec_wires, d_pad)?;
    let input_s_cols: Vec<Vec<KVar>> = ccs.outputs.iter().map(|o| o.s_col.clone()).collect();
    enforce_rlc_s_col_consistency(builder, &input_s_cols, &dec_wires.parent.s_col)?;

    // ── 5. Π_DEC.V strict ────────────────────────────────────────────────
    enforce_dec_v_strict(builder, pp, &dec_wires)?;

    // ── 6. Point binding: dec parent.r == ccs r_prime ────────────────────
    enforce_kvar_vec_eq(builder, &dec_wires.parent.r, &ccs.r_prime)?;

    let running = ccs.running.clone();
    let running_parent_authority = ccs.running_parent_authority.clone();
    let parent = dec_wires.parent.clone();
    let children = dec_wires.children.clone();
    Ok(NifsVOutputs {
        parent_c_data: dec_wires.parent.c_data.clone(),
        fresh_x: ccs.fresh_x,
        running_c_data: ccs.running_c_data,
        running_acc_digest: ccs.running_acc_digest,
        running,
        running_parent_authority,
        parent,
        children,
    })
}

// ── Private RLC fold helpers ──────────────────────────────────────────────

fn enforce_rlc_commitment_fold(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
    kappa: usize,
) -> Result<(), Error> {
    if rho_wires.len() != outputs.len() {
        return Err(Error::Inner(format!(
            "ρ count {} != outputs count {}",
            rho_wires.len(),
            outputs.len()
        )));
    }
    let inputs: Vec<RlcPairWires> = rho_wires
        .iter()
        .zip(outputs.iter())
        .map(|(rho, o)| RlcPairWires {
            rho_coeffs: *rho,
            c_data: o.c_data.clone(),
            kappa: o.c_kappa,
        })
        .collect();
    let wires = RlcCommitmentWires {
        inputs,
        combined_c_data: dec_wires.parent.c_data.clone(),
        kappa,
    };
    enforce_rlc_commitment_combination(builder, &wires);
    Ok(())
}

fn enforce_rlc_x_fold(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
    m_in: usize,
) -> Result<(), Error> {
    let inputs: Vec<RlcXPairWires> = rho_wires
        .iter()
        .zip(outputs.iter())
        .map(|(rho, o)| RlcXPairWires {
            rho_coeffs: *rho,
            x_flat: o.x.clone(),
            m_in: o.x_cols,
        })
        .collect();
    let wires = RlcXWires {
        inputs,
        combined_x_flat: dec_wires.parent.x.clone(),
        m_in,
    };
    enforce_rlc_x_combination(builder, &wires);
    Ok(())
}

fn enforce_rlc_y_ring_row_fold(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
    j: usize,
    d_pad: usize,
) -> Result<(), Error> {
    let inputs: Vec<Vec<KVar>> = outputs.iter().map(|o| o.y_ring[j].clone()).collect();
    let combined = kvars_from_flat_dec(&dec_wires.parent.y_ring[j])?;
    let wires = padded_k_vector_wires_from_existing(rho_wires, &inputs, &combined, d_pad)?;
    enforce_rlc_padded_k_vector_combination(builder, &wires);
    Ok(())
}

fn enforce_rlc_y_zcol_fold(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
    d_pad: usize,
) -> Result<(), Error> {
    let inputs: Vec<Vec<KVar>> = outputs.iter().map(|o| o.y_zcol.clone()).collect();
    let combined = kvars_from_flat_dec(&dec_wires.parent.y_zcol)?;
    let wires = padded_k_vector_wires_from_existing(rho_wires, &inputs, &combined, d_pad)?;
    enforce_rlc_padded_k_vector_combination(builder, &wires);
    Ok(())
}

/// Convert DEC's flat-K representation `[k0.c0, k0.c1, k1.c0, k1.c1, …]`
/// back into a `Vec<KVar>` view, no fresh wire allocation. DEC stores
/// `y_ring`/`y_zcol` as `Vec<Var>` of length `d_pad * K_LIMBS` where
/// `K_LIMBS = 2`; SplitNc consumers want `Vec<KVar>` of length `d_pad`.
fn kvars_from_flat_dec(flat: &[Var]) -> Result<Vec<KVar>, Error> {
    if flat.len() % 2 != 0 {
        return Err(Error::Inner(format!(
            "DEC flat K-vector has odd limb count {}",
            flat.len()
        )));
    }
    Ok(flat
        .chunks_exact(2)
        .map(|c| KVar { c0: c[0], c1: c[1] })
        .collect())
}

/// Construct a [`RlcPaddedKVectorWires`] from already-allocated wires
/// (no fresh allocation). Used by both `y_ring[j]` and `y_zcol` folds.
fn padded_k_vector_wires_from_existing(
    rhos: &[[Var; D]],
    inputs: &[Vec<KVar>],
    combined: &[KVar],
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    if combined.len() != d_pad {
        return Err(Error::Inner(format!(
            "padded K-vector: combined.len ({}) != d_pad ({})",
            combined.len(),
            d_pad
        )));
    }
    if rhos.len() != inputs.len() {
        return Err(Error::Inner(format!(
            "padded K-vector: ρ count {} != input count {}",
            rhos.len(),
            inputs.len()
        )));
    }
    let mut pair_wires = Vec::with_capacity(inputs.len());
    for (idx, (rho, y)) in rhos.iter().zip(inputs.iter()).enumerate() {
        if y.len() != d_pad {
            return Err(Error::Inner(format!(
                "padded K-vector: inputs[{idx}].len ({}) != d_pad ({})",
                y.len(),
                d_pad
            )));
        }
        let y_c0: Vec<Var> = y.iter().map(|v| v.c0).collect();
        let y_c1: Vec<Var> = y.iter().map(|v| v.c1).collect();
        pair_wires.push(RlcPaddedKVectorPairWires {
            rho_coeffs: *rho,
            y_c0,
            y_c1,
        });
    }
    let combined_c0: Vec<Var> = combined.iter().map(|v| v.c0).collect();
    let combined_c1: Vec<Var> = combined.iter().map(|v| v.c1).collect();
    Ok(RlcPaddedKVectorWires {
        inputs: pair_wires,
        combined_c0,
        combined_c1,
        d_pad,
    })
}

fn enforce_kvar_vec_eq(builder: &mut R1csBuilder, a: &[KVar], b: &[KVar]) -> Result<(), Error> {
    if a.len() != b.len() {
        return Err(Error::Inner(format!(
            "KVar vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    for (x, y) in a.iter().zip(b.iter()) {
        builder.enforce_eq(&Lc::from_var(x.c0), &Lc::from_var(y.c0));
        builder.enforce_eq(&Lc::from_var(x.c1), &Lc::from_var(y.c1));
    }
    Ok(())
}

// Suppress unused-import warnings if F/K become unused in future edits.
#[allow(dead_code)]
fn _silence_unused_f_k(_f: F, _k: K) {}
