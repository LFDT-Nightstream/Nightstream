//! `PaddedRowIdentity` recursive PiCCS verifier.
//!
//! Owns: one joint SumCheck, the paper terminal equation, input authority,
//! and the constrained output wire surface. The relation digest and claim
//! digests come from verifier-owned data.
//!
//! Does not own: native proving, matrix evaluation, PiRLC, or PiDEC.
//!
//! Emits constraints: claim canonicality, transcript replay, one joint
//! SumCheck, the terminal equation, and the output claim.
//!
//! | Phase | Constraint family |
//! | --- | --- |
//! | input | shape, canonicality, and digest binding |
//! | proof | transcript, SumCheck, and terminal equality |
//! | output | ring evaluations and output digest |

use neo_ccs::{CcsStructure, SparsePoly};
use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

use super::stage;
use super::{
    alloc_constant_var, enforce_accumulator_ce_claim_digest, enforce_ccs_claim_digest,
    enforce_pi_ccs_instance_digest_parent_authority, enforce_pi_ccs_outputs_digest,
    enforce_strict_binary_accumulator_family_digest, AccumulatorCeClaimDigestInputs, Error,
    PiCcsOutputMessageDigestInputs, PiCcsOutputsPreimage,
};
use crate::engine::r1cs_circuit::builder::{Lc, Var};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::digest::AccumulatorHandle;
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_output_message::Profile as PiCcsOutputMessageProfile;
use crate::paper::relations::product_commitment_circuit::{alloc_adv, AdvCommitmentWires};
use crate::paper::relations::{validate_adv_shape, CcsClaim, CeClaim};

#[path = "verifier_claims.rs"]
mod claims;
use claims::*;

#[path = "verifier/padded_row.rs"]
mod padded_row;

/// Matrix-independent CCS header consumed by the in-circuit verifier.
///
/// The verifier evaluates the relation polynomial over claimed matrix
/// evaluations; it never reads a matrix coefficient. Owning only this header
/// makes that boundary explicit and lets preprocessing discover recursive
/// dimensions without allocating candidate matrices.
#[derive(Clone)]
pub struct PiCcsVerifierRelation {
    n: usize,
    m: usize,
    polynomial: SparsePoly<F>,
}

impl PiCcsVerifierRelation {
    pub fn from_structure(structure: &CcsStructure<F>) -> Self {
        Self::from_parts(structure.n, structure.m, structure.f.clone())
    }

    pub(crate) fn from_parts(n: usize, m: usize, polynomial: SparsePoly<F>) -> Self {
        assert!(n > 0, "PiCCS verifier relation requires at least one row");
        assert!(m > 0, "PiCCS verifier relation requires at least one column");
        assert!(
            polynomial.arity() > 0,
            "PiCCS verifier relation requires a nonempty polynomial"
        );
        Self { n, m, polynomial }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn t(&self) -> usize {
        self.polynomial.arity()
    }

    pub fn max_degree(&self) -> u32 {
        self.polynomial.max_degree()
    }

    pub fn polynomial(&self) -> &SparsePoly<F> {
        &self.polynomial
    }
}

impl From<&CcsStructure<F>> for PiCcsVerifierRelation {
    fn from(structure: &CcsStructure<F>) -> Self {
        Self::from_structure(structure)
    }
}

/// Verifier-owned PiCCS constants.
pub struct PiCcsVerifierConfig<'a> {
    pub params: &'a Params,
    pub structure: PiCcsVerifierRelation,
    pub matrix_digest: [F; 4],
}

/// Witness/protocol messages from a real native `pi_ccs::prove` proof.
///
/// - `fresh` / `running`: the public claim arrays both sides already hold.
/// - `outputs`: the K+k output CE claims the prover sends on the wire.
/// - `sumcheck_rounds`: the one joint SumCheck message stream.
pub struct PiCcsVerifierMessages<'a> {
    pub fresh: &'a [CcsClaim],
    pub running: &'a [CeClaim],
    pub running_parent_authority: Option<&'a CeClaim>,
    pub outputs: &'a [CeClaim],
    /// Redundant wire-format digest checked by native `pi_ccs::verify`.
    /// The circuit recomputes it from `outputs`; this field is never treated
    /// as authority.
    pub outputs_digest: [F; 4],
    pub sumcheck_rounds: &'a [Vec<K>],
}

/// Per-output wire bundle that downstream Π_RLC.V (and any consumer that
/// needs to fold output claims) consumes after a successful PiCCS verify.
///
/// Each output is one CE claim, and Π_RLC combines them into a parent CE.
/// Every field here is a witness wire (or row of wires) already constrained
/// by the PiCCS verifier's identity and transcript checks.
#[derive(Clone)]
pub struct PiCcsOutputWires {
    pub c_d: usize,
    pub c_d_var: Var,
    pub c_kappa: usize,
    pub c_kappa_var: Var,
    pub c_data: Vec<Var>,
    pub adv: Option<AdvCommitmentWires>,
    pub x: Vec<Var>,
    pub x_rows: usize,
    pub x_rows_var: Var,
    pub x_cols: usize,
    pub x_cols_var: Var,
    pub m_in: usize,
    pub m_in_var: Var,
    pub r: Vec<KVar>,
    pub y_ring: Vec<Vec<KVar>>,
    /// SuperNeo scalar/constant-term view of `y_ring`.
    ///
    /// The selected verifier constrains this as `ct[j] == y_ring[j][0]`;
    /// downstream continuity gates then carry the denormalized field
    /// wire-to-wire without treating it as independent authority.
    pub ct: Vec<KVar>,
    /// `fold_digest` field of the CE claim, projected to four base-field
    /// lanes (matches `digest32_as_fields`). Carried by the bundle so
    /// downstream consumers (decider CE-continuity gate, audit tooling)
    /// can re-bind without re-walking the original claim's witness.
    pub fold_digest_fields: [Var; 4],
}

/// Derived wires the PiCCS verifier exposes to the downstream
/// NIFS.V / Π_RLC composition once verification passes.
///
/// `fresh_x` and `running_c_data` are surfaced separately because F' R1CS
/// (above NIFS.V) needs them to enforce the HyperNova recursive link
/// (`u_i.public == enc_inst(hash(prior_state))`) and the accumulator
/// digest binding (`acc_digest_in == digest(running)`). They're already
/// constrained by the Π_CCS.V verifier; this just exposes the wires
/// without forcing F' to re-walk the witness layout.
pub struct PiCcsVerifierResult {
    pub r_prime: Vec<KVar>,
    pub outputs: Vec<PiCcsOutputWires>,
    /// Π_CCS output digest recomputed from the constrained output wires and
    /// checked against the redundant proof field. NIFS.V appends these same
    /// wires before sampling Π_RLC challenges.
    pub output_claims_digest: [Var; 4],
    /// Exact pre-SIS field-to-column ownership used to compute
    /// [`Self::output_claims_digest`].
    pub output_message_preimage: PiCcsOutputsPreimage,
    /// `fresh_x[i]` = the `m_in` public-input `F`-wires of `fresh[i]`.
    pub fresh_x: Vec<Vec<Var>>,
    /// Product-commitment coordinates of those same fresh claims. Each
    /// entry shares allocation and transcript binding with `fresh_x[i]`.
    pub fresh_adv: Vec<Option<AdvCommitmentWires>>,
    /// `running_c_data[i]` = the `D * kappa` commitment-data wires of `running[i]`.
    pub running_c_data: Vec<Vec<Var>>,
    /// Per-running-claim CE core wires used by the exact ordered accumulator
    /// binding.
    pub running: Vec<PiCcsOutputWires>,
    /// Π_RLC parent whose Π_DEC children form `running`. It is a separately
    /// checked transcript input, not the Construction-2 accumulator handle.
    pub running_parent_authority: Option<PiCcsOutputWires>,
    /// Four-lane Poseidon2 handle of the exact ordered running child vector.
    /// The checked parent is only a recomposition cache.
    pub running_acc_digest: [Var; 4],
}

// ── Wire-allocation structs ───────────────────────────────────────────────

/// Wires for one fresh CCS claim, allocated once and reused across the
/// digest gadget and any wire-to-wire output binding.
struct CcsClaimWires {
    c_d: usize,
    c_d_var: Var,
    c_kappa: usize,
    c_kappa_var: Var,
    c_data: Vec<Var>,
    adv: Option<AdvCommitmentWires>,
    x: Vec<Var>,
    m_in: usize,
    m_in_var: Var,
}

/// Wires for one CE claim (running input or output), allocated once.
///
/// `x` is the X matrix in row-major order: `x[r * x_cols + c] = X[(r, c)]`.
/// The SuperNeo-packed view used by the ME-projection digest is built on
/// demand via [`Self::x_packed`] without reallocating any wires.
struct CeClaimWires {
    c_d: usize,
    c_d_var: Var,
    c_kappa: usize,
    c_kappa_var: Var,
    c_data: Vec<Var>,
    adv: Option<AdvCommitmentWires>,
    x: Vec<Var>,
    x_rows: usize,
    x_rows_var: Var,
    x_cols: usize,
    x_cols_var: Var,
    r: Vec<KVar>,
    y_ring: Vec<Vec<KVar>>,
    ct: Vec<KVar>,
    m_in: usize,
    m_in_var: Var,
    fold_digest_fields: [Var; 4],
}

fn accumulator_digest_inputs(claim: &CeClaimWires) -> AccumulatorCeClaimDigestInputs<'_> {
    AccumulatorCeClaimDigestInputs {
        c_d: claim.c_d,
        c_kappa: claim.c_kappa,
        c_data: &claim.c_data,
        x_rows: claim.x_rows,
        x_cols: claim.x_cols,
        x_flat_row_major: &claim.x,
        r: &claim.r,
        y_ring: &claim.y_ring,
        ct: &claim.ct,
        m_in: claim.m_in,
        fold_digest_fields: claim.fold_digest_fields,
        adv: claim.adv.as_ref(),
    }
}

// ── Public entry ──────────────────────────────────────────────────────────

/// Enforce the selected PiCCS verifier on top of `transcript`.
pub fn enforce_pi_ccs(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    cfg: &PiCcsVerifierConfig<'_>,
    msg: &PiCcsVerifierMessages<'_>,
) -> Result<PiCcsVerifierResult, Error> {
    enforce_pi_ccs_inner(builder, transcript, cfg, msg, None)
}

/// Folded-F' entrypoint. The header is verifier-key advice, so its values
/// do not become constants in a relation that ultimately verifies itself.
pub fn enforce_pi_ccs_with_matrix_digest_wires(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    cfg: &PiCcsVerifierConfig<'_>,
    msg: &PiCcsVerifierMessages<'_>,
    matrix_digest: [Var; 4],
) -> Result<PiCcsVerifierResult, Error> {
    enforce_pi_ccs_inner(builder, transcript, cfg, msg, Some(matrix_digest))
}

fn enforce_pi_ccs_inner(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    cfg: &PiCcsVerifierConfig<'_>,
    msg: &PiCcsVerifierMessages<'_>,
    matrix_digest: Option<[Var; 4]>,
) -> Result<PiCcsVerifierResult, Error> {
    padded_row::enforce(builder, transcript, cfg, msg, matrix_digest)
}

// ── Private helpers ───────────────────────────────────────────────────────

fn enforce_kvar_eq(builder: &mut R1csBuilder, a: KVar, b: KVar) {
    builder.enforce_eq(&Lc::from_var(a.c0), &Lc::from_var(b.c0));
    builder.enforce_eq(&Lc::from_var(a.c1), &Lc::from_var(b.c1));
}

fn enforce_var_eq(builder: &mut R1csBuilder, a: Var, b: Var) {
    builder.enforce_eq(&Lc::from_var(a), &Lc::from_var(b));
}

fn alloc_usize(builder: &mut R1csBuilder, value: usize) -> Var {
    let wire = builder.alloc(F::from_u64(value as u64));
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(F::from_u64(value as u64)));
    wire
}

fn enforce_output_fold_digest_matches_header(
    builder: &mut R1csBuilder,
    output_wires: &[CeClaimWires],
    header_wires: [Var; 4],
) {
    for output in output_wires {
        for (output, expected) in output.fold_digest_fields.iter().zip(header_wires) {
            builder.enforce_eq(&Lc::from_var(*output), &Lc::from_var(expected));
        }
    }
}

fn enforce_ct_from_y_ring(builder: &mut R1csBuilder, label: &str, claim: &CeClaimWires) -> Result<(), Error> {
    if claim.ct.len() != claim.y_ring.len() {
        return Err(Error::Shape(format!(
            "{label}.ct.len ({}) != y_ring.len ({})",
            claim.ct.len(),
            claim.y_ring.len()
        )));
    }
    for (j, (ct, row)) in claim.ct.iter().zip(claim.y_ring.iter()).enumerate() {
        let y0 = row.first().copied().ok_or_else(|| {
            Error::Shape(format!(
                "{label}.y_ring[{j}] has no lane-0 constant term for ct binding"
            ))
        })?;
        enforce_kvar_eq(builder, *ct, y0);
    }
    Ok(())
}

fn enforce_y_ring_padding_zero(builder: &mut R1csBuilder, claim: &CeClaimWires) {
    for row in &claim.y_ring {
        for lane in row.iter().skip(D) {
            builder.enforce_eq(&Lc::from_var(lane.c0), &Lc::zero());
            builder.enforce_eq(&Lc::from_var(lane.c1), &Lc::zero());
        }
    }
}
