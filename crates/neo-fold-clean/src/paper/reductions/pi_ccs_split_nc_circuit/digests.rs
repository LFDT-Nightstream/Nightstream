//! SplitNcV1 — per-claim digest gadgets.
//!
//! Owns: exact Poseidon2/SIS preimages for Pi_CCS claim and message bindings.
//!
//! Does not own: claim validation, transcript placement, or digest authority.
//!
//! Emits constraints: yes.
//!
//! Authority boundary: callers must supply already-constrained claim wires;
//! carried prover digests are comparison values, never authority.
//!
//! | Constraint family | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | fresh CCS digest | Bind commitment, advice, public input, and shape | yes | `enforce_ccs_claim_digest` | concrete bridge open |
//! | CE digest | Bind the complete CE message | yes | `enforce_ce_claim_digest` | concrete bridge open |
//! | accumulator child digest | Bind one paper-level ordered child claim; `y_zcol` source binding remains open | yes | `enforce_accumulator_ce_claim_digest` | exact-child bridge open |
//! | accumulator handle | Bind the ordered paper-level `CE(b)^k` child vector | yes | `enforce_accumulator_claims_digest` | exact-child bridge open |
//! | instance digest | Bind fresh claims and checked running parent | yes | `enforce_pi_ccs_instance_digest_parent_authority` | authority bridge open |
//! | output digest | Bind the profile-pinned Pi_CCS message consumed by Pi_RLC | yes | `enforce_pi_ccs_outputs_digest` | source layout model-level; physical SIS bridge open |
//!
//! Mirrors:
//! - `crate::paper::digest::ccs_claim_digest` (per-fresh CCS claim).
//! - `crate::paper::digest::ce_claim_digest` (per-running CE claim).
//! - `crate::paper::digest::pi_ccs_instance_digest` (the authoritative
//!   public-instance digest the SplitNcV1 verifier must absorb in place of
//!   any prover-supplied value).
//!
//! Project soundness rule: the SplitNcV1 composition must recompute
//! these digests in-circuit from authoritative claim wires — never
//! accept a prover-supplied digest as authority.
//!
//! The ME-input handle binds the exact ordered paper-level running children.
//! The checked Π_RLC parent remains a recomposition cache: strict Π_DEC
//! consistency does not uniquely determine its child vector.
//! The per-child serializer is intentionally conservative until the Lean
//! minimal-family payload is refined to Rust's repaired 270-coordinate carrier.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::output_message::{encode_pi_ccs_outputs_preimage, PiCcsOutputMessageDigestInputs, PiCcsOutputsPreimage};
use super::{alloc_constant_var, extend_f_slice_wires, extend_packed_bytes_as_fields_wires, stage, Error};
use crate::engine::r1cs_circuit::builder::{Lc, Var};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::poseidon2::enforce_poseidon2_hash;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::digest::{
    NEBULA_ADV_PRESENT_MARKER, NEBULA_LEAF_MEM_TAG, NEBULA_LEAF_OPS_TAG, PENDING_ACCUMULATOR_FAMILY_CHILDREN,
    PENDING_ACCUMULATOR_FAMILY_COLUMN_POINT, PENDING_ACCUMULATOR_FAMILY_DOMAIN, PENDING_ACCUMULATOR_FAMILY_MATRICES,
    PENDING_ACCUMULATOR_FAMILY_M_IN, PENDING_ACCUMULATOR_FAMILY_ROW_POINT,
};
use crate::paper::f_prime::nebula_lane_circuit::enforce_nebula_leaf_digest_circuit;
use crate::paper::reductions::accumulator_sis_circuit::{
    enforce_accumulator_digest as enforce_sis_accumulator_digest, SisAccumulatorCircuitLayout,
    ACCUMULATOR_CE_CLAIM_SIS_CONFIG, CCS_CLAIM_SIS_CONFIG, CE_CLAIM_SIS_CONFIG, PENDING_ACCUMULATOR_FAMILY_SIS_CONFIG,
    PI_CCS_OUTPUTS_SIS_CONFIG,
};
use crate::paper::reductions::pi_ccs_output_message::Profile;
use crate::paper::relations::product_commitment_circuit::AdvCommitmentWires;

/// Domain bytes for the paper-layer per-claim digests (mirrors
/// `crate::paper::digest::*`). Kept byte-identical to the native strings.
const CCS_CLAIM_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/ccs_claim_digest/v1";
const CE_CLAIM_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/ce_claim_digest/v2";
const ACCUMULATOR_CE_CLAIM_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/accumulator_ce_claim_digest/v2";
const ACCUMULATOR_CLAIMS_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/accumulator/children/v3";
const PI_CCS_INSTANCE_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/pi_ccs_instance_digest/v1";
const PI_CCS_PARENT_AUTHORITY_INSTANCE_DIGEST_DOMAIN: &[u8] =
    b"neo.fold.clean/pi_ccs_instance_digest/parent_authority/v1";

// ── Paper-layer per-claim digests ─────────────────────────────────────────

/// Mirror of `crate::paper::digest::ccs_claim_digest`. Hashes the public
/// fields of one fresh CCS claim into a 4-lane SIS-compressed digest, preserving
/// the native preimage layout exactly:
///
/// ```text
/// pack_bytes_as_fields(b"neo.fold.clean/ccs_claim_digest/v1")
/// ‖ c.d ‖ c.kappa ‖ c.data.len() ‖ c.data
/// ‖ x.len() ‖ x ‖ m_in
/// ```
///
/// `c_d` and `c_kappa` are structural shape and known at gadget-emit time;
/// `c_data` and `x` are witness `F`-wires. This gadget is what the SplitNcV1
/// composition uses to build per-fresh digests for
/// [`enforce_pi_ccs_instance_digest`] — *not* an arbitrary digest the prover
/// can supply, mirroring native authority rules.
pub fn enforce_ccs_claim_digest(
    builder: &mut R1csBuilder,
    c_d: usize,
    c_kappa: usize,
    c_data: &[Var],
    x: &[Var],
    m_in: usize,
    adv: Option<&AdvCommitmentWires>,
) -> [Var; 4] {
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, CCS_CLAIM_DIGEST_DOMAIN);

    preimage.push(alloc_constant_var(builder, F::from_u64(c_d as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(c_kappa as u64)));
    extend_f_slice_wires(builder, &mut preimage, c_data);

    preimage.push(alloc_constant_var(builder, F::from_u64(x.len() as u64)));
    preimage.extend_from_slice(x);

    preimage.push(alloc_constant_var(builder, F::from_u64(m_in as u64)));
    append_adv_leaves_circuit(builder, &mut preimage, adv);

    enforce_sis_accumulator_digest(builder, CCS_CLAIM_SIS_CONFIG, &preimage)
        .expect("fixed nonempty CCS-claim SIS preimage")
        .digest
}

/// Witness wires for one running CE claim, in the exact shape that
/// `paper::digest::ce_claim_digest` consumes natively.
///
/// `x_flat_row_major` is the X matrix flattened in row-major order to match
/// the native `for r in 0..rows: for c in 0..cols: push(X[(r,c)])` loop.
///
/// `fold_digest_fields` is the four-lane decoding of `claim.fold_digest`
/// (the native 32-byte digest), produced by the same `digest32_as_fields`
/// formula the paper layer uses.
pub struct CeClaimDigestInputs<'a> {
    pub c_d: usize,
    pub c_kappa: usize,
    pub c_data: &'a [Var],
    pub x_rows: usize,
    pub x_cols: usize,
    pub x_flat_row_major: &'a [Var],
    pub r: &'a [KVar],
    pub y_ring: &'a [Vec<KVar>],
    pub m_in: usize,
    pub fold_digest_fields: [Var; 4],
    pub adv: Option<&'a AdvCommitmentWires>,
}

/// Witness wires for one CE claim in the exact shape consumed by
/// `paper::digest::accumulator_ce_claim_digest`.
pub struct AccumulatorCeClaimDigestInputs<'a> {
    pub c_d: usize,
    pub c_kappa: usize,
    pub c_data: &'a [Var],
    pub x_rows: usize,
    pub x_cols: usize,
    pub x_flat_row_major: &'a [Var],
    pub r: &'a [KVar],
    pub s_col: &'a [KVar],
    pub y_ring: &'a [Vec<KVar>],
    pub ct: &'a [KVar],
    pub m_in: usize,
    pub fold_digest_fields: [Var; 4],
    pub adv: Option<&'a AdvCommitmentWires>,
}

/// Digest output plus the exact field-to-column ledger consumed by SIS.
pub struct PiCcsOutputsDigestWires {
    pub digest: [Var; 4],
    pub preimage: PiCcsOutputsPreimage,
    pub sis_layout: SisAccumulatorCircuitLayout,
}

/// Exact active child payload consumed by the fixed pending-family codec.
///
/// `x_active_column_major` is already projected into logical public-carrier
/// order. `y_ring_active` contains exactly the 54 non-padding lanes for each
/// of the 13 matrices. The production caller must derive these projections
/// from its fully constrained claim wires; this serializer does not duplicate
/// inactive/padding-zero constraints.
pub struct PendingAccumulatorFamilyChildInputs<'a> {
    pub c_data: &'a [Var],
    pub x_active_column_major: &'a [Var],
    pub y_ring_active: &'a [Vec<KVar>],
}

pub struct PendingAccumulatorFamilyStateInputs<'a> {
    pub old_block: &'a [KVar],
    pub parent_y_zcol: &'a [KVar],
}

pub struct PendingAccumulatorFamilyDigestInputs<'a> {
    pub verifier_rows: usize,
    pub row_point: &'a [KVar],
    pub column_point: &'a [KVar],
    pub m_in: usize,
    pub fold_digest_fields: [Var; 4],
    pub children: &'a [PendingAccumulatorFamilyChildInputs<'a>],
    pub pending: Option<PendingAccumulatorFamilyStateInputs<'a>>,
}

pub struct PendingAccumulatorFamilyDigestWires {
    pub digest: [Var; 4],
    pub preimage: Vec<Var>,
    pub sis_layout: SisAccumulatorCircuitLayout,
}

fn append_kvar_values_without_length(preimage: &mut Vec<Var>, values: &[KVar]) {
    for value in values {
        preimage.push(value.c0);
        preimage.push(value.c1);
    }
}

/// Exact circuit mirror of `pending_accumulator_family_preimage`.
///
/// This owns field order only. Commitment shape, active-X projection,
/// evaluation padding, scalar-view equality, shared-point provenance, and
/// delayed-state authority remain explicit obligations at the production
/// call site.
pub fn encode_pending_accumulator_family_preimage(
    builder: &mut R1csBuilder,
    input: &PendingAccumulatorFamilyDigestInputs<'_>,
) -> Result<Vec<Var>, Error> {
    if input.verifier_rows == 0 {
        return Err(Error::Shape(
            "pending accumulator family verifier kappa must be nonzero".into(),
        ));
    }
    if input.children.len() != PENDING_ACCUMULATOR_FAMILY_CHILDREN {
        return Err(Error::Shape(format!(
            "pending accumulator family needs {} children, got {}",
            PENDING_ACCUMULATOR_FAMILY_CHILDREN,
            input.children.len()
        )));
    }
    if input.row_point.len() != PENDING_ACCUMULATOR_FAMILY_ROW_POINT
        || input.column_point.len() != PENDING_ACCUMULATOR_FAMILY_COLUMN_POINT
        || input.m_in != PENDING_ACCUMULATOR_FAMILY_M_IN
    {
        return Err(Error::Shape(
            "pending accumulator family shared point or m_in shape mismatch".into(),
        ));
    }
    let expected_commitment = neo_math::D * input.verifier_rows;
    let expected_public = PENDING_ACCUMULATOR_FAMILY_M_IN;
    for (child_index, child) in input.children.iter().enumerate() {
        if child.c_data.len() != expected_commitment {
            return Err(Error::Shape(format!(
                "pending accumulator family child {child_index} commitment has {} fields, expected {expected_commitment}",
                child.c_data.len()
            )));
        }
        if child.x_active_column_major.len() != expected_public {
            return Err(Error::Shape(format!(
                "pending accumulator family child {child_index} public payload has {} fields, expected {expected_public}",
                child.x_active_column_major.len()
            )));
        }
        if child.y_ring_active.len() != PENDING_ACCUMULATOR_FAMILY_MATRICES
            || child
                .y_ring_active
                .iter()
                .any(|row| row.len() != neo_math::D)
        {
            return Err(Error::Shape(format!(
                "pending accumulator family child {child_index} evaluations must be 13 by 54"
            )));
        }
    }
    if let Some(pending) = &input.pending {
        if pending.old_block.len() != crate::paper::digest::PENDING_ACCUMULATOR_OLD_BLOCK
            || pending.parent_y_zcol.len() != neo_math::D
        {
            return Err(Error::Shape(
                "pending accumulator family delayed state must be 19 plus 54 extension elements".into(),
            ));
        }
    }

    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, PENDING_ACCUMULATOR_FAMILY_DOMAIN);
    preimage.push(alloc_constant_var(
        builder,
        F::from_u64(PENDING_ACCUMULATOR_FAMILY_CHILDREN as u64),
    ));
    preimage.push(alloc_constant_var(
        builder,
        F::from_u64(PENDING_ACCUMULATOR_FAMILY_ROW_POINT as u64),
    ));
    preimage.push(alloc_constant_var(
        builder,
        F::from_u64(PENDING_ACCUMULATOR_FAMILY_COLUMN_POINT as u64),
    ));
    append_kvar_values_without_length(&mut preimage, input.row_point);
    append_kvar_values_without_length(&mut preimage, input.column_point);
    preimage.push(alloc_constant_var(builder, F::from_u64(input.m_in as u64)));
    preimage.extend_from_slice(&input.fold_digest_fields);
    for child in input.children {
        preimage.extend_from_slice(child.c_data);
        preimage.extend_from_slice(child.x_active_column_major);
        for row in child.y_ring_active {
            append_kvar_values_without_length(&mut preimage, row);
        }
    }
    match &input.pending {
        None => {
            preimage.push(alloc_constant_var(builder, F::ZERO));
            for _ in 0..2 * (crate::paper::digest::PENDING_ACCUMULATOR_OLD_BLOCK + neo_math::D) {
                preimage.push(alloc_constant_var(builder, F::ZERO));
            }
        }
        Some(pending) => {
            preimage.push(alloc_constant_var(builder, F::ONE));
            append_kvar_values_without_length(&mut preimage, pending.old_block);
            append_kvar_values_without_length(&mut preimage, pending.parent_y_zcol);
        }
    }
    Ok(preimage)
}

pub fn enforce_pending_accumulator_family_digest(
    builder: &mut R1csBuilder,
    input: &PendingAccumulatorFamilyDigestInputs<'_>,
) -> Result<PendingAccumulatorFamilyDigestWires, Error> {
    let preimage = encode_pending_accumulator_family_preimage(builder, input)?;
    let sis = enforce_sis_accumulator_digest(builder, PENDING_ACCUMULATOR_FAMILY_SIS_CONFIG, &preimage)
        .expect("fixed nonempty pending-family SIS preimage");
    Ok(PendingAccumulatorFamilyDigestWires {
        digest: sis.digest,
        preimage,
        sis_layout: sis.layout,
    })
}

/// Mirror of `crate::paper::digest::ce_claim_digest`. Preimage layout
/// (must match native byte-for-byte):
///
/// ```text
/// pack_bytes_as_fields(b"neo.fold.clean/ce_claim_digest/v2")
/// ‖ c.d ‖ c.kappa ‖ c.data.len() ‖ c.data
/// ‖ X.rows ‖ X.cols ‖ active_x_cols ‖ X_active(row-major, active cols only)
/// ‖ r.len ‖ r_limbs(flat: c0,c1,c0,c1,…)
/// ‖ y_ring.len ‖ for each row: row.len ‖ row_limbs(flat)
/// ‖ m_in ‖ fold_digest_fields
/// ```
///
/// Inactive X columns (`c >= ceil(m_in / D)`) are required to be zero,
/// enforced in-circuit before the digest absorb and on the native side
/// by `superneo_inactive_x_zero` shape checks in Π_CCS validation.
///
/// Note: native does NOT push a `coeff_width` header per K-slice here
/// (unlike `me_input_projection_digest`). The limbs are pushed directly.
pub fn enforce_ce_claim_digest(builder: &mut R1csBuilder, input: &CeClaimDigestInputs) -> Result<[Var; 4], Error> {
    if input.x_flat_row_major.len() != input.x_rows * input.x_cols {
        return Err(Error::Shape(format!(
            "enforce_ce_claim_digest: x_flat_row_major.len ({}) must equal x_rows*x_cols ({}*{}={})",
            input.x_flat_row_major.len(),
            input.x_rows,
            input.x_cols,
            input.x_rows * input.x_cols
        )));
    }

    // Active X columns: `ceil(m_in / D)`. Cols `[active_cols, x_cols)` are
    // structurally zero by `project_x_from_witness_mat`; enforce it
    // in-circuit so the digest can skip them without losing soundness.
    let active_x_cols = crate::paper::relations::superneo_public_x_cols(input.m_in);
    if active_x_cols > input.x_cols {
        return Err(Error::Shape(format!(
            "enforce_ce_claim_digest: active_x_cols ({active_x_cols}) > x_cols ({})",
            input.x_cols
        )));
    }
    enforce_unique_inactive_x_zero(
        builder,
        input.x_flat_row_major,
        input.x_rows,
        input.x_cols,
        active_x_cols,
    );

    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, CE_CLAIM_DIGEST_DOMAIN);

    // Commitment
    preimage.push(alloc_constant_var(builder, F::from_u64(input.c_d as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(input.c_kappa as u64)));
    extend_f_slice_wires(builder, &mut preimage, input.c_data);

    // X: shape (full rows × full cols) + entries of *active* columns only.
    // See `paper::digest::ce_claim_digest` for the matching native absorb.
    preimage.push(alloc_constant_var(builder, F::from_u64(input.x_rows as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(input.x_cols as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(active_x_cols as u64)));
    for r in 0..input.x_rows {
        for c in 0..active_x_cols {
            preimage.push(input.x_flat_row_major[r * input.x_cols + c]);
        }
    }

    // r (length + flat limbs, no per-K coeff width header).
    preimage.push(alloc_constant_var(builder, F::from_u64(input.r.len() as u64)));
    for k in input.r {
        preimage.push(k.c0);
        preimage.push(k.c1);
    }

    // y_ring (outer length + per-row { row length + flat limbs }).
    preimage.push(alloc_constant_var(builder, F::from_u64(input.y_ring.len() as u64)));
    for row in input.y_ring {
        preimage.push(alloc_constant_var(builder, F::from_u64(row.len() as u64)));
        for v in row {
            preimage.push(v.c0);
            preimage.push(v.c1);
        }
    }

    preimage.push(alloc_constant_var(builder, F::from_u64(input.m_in as u64)));
    preimage.extend_from_slice(&input.fold_digest_fields);
    append_adv_leaves_circuit(builder, &mut preimage, input.adv);

    Ok(enforce_sis_accumulator_digest(builder, CE_CLAIM_SIS_CONFIG, &preimage)
        .expect("fixed nonempty CE-claim SIS preimage")
        .digest)
}

/// Mirror of `crate::paper::digest::accumulator_ce_claim_digest`.
///
/// The current clean pipeline requires `aux_openings`, `c_step_coords`,
/// `u_offset`, and `u_len` to be empty/zero before a claim reaches this
/// gadget; their zero encodings are still included in the hash so native and
/// circuit layouts stay identical.
pub fn enforce_accumulator_ce_claim_digest(
    builder: &mut R1csBuilder,
    input: &AccumulatorCeClaimDigestInputs,
) -> Result<[Var; 4], Error> {
    if input.x_flat_row_major.len() != input.x_rows * input.x_cols {
        return Err(Error::Shape(format!(
            "enforce_accumulator_ce_claim_digest: x_flat_row_major.len ({}) must equal x_rows*x_cols ({}*{}={})",
            input.x_flat_row_major.len(),
            input.x_rows,
            input.x_cols,
            input.x_rows * input.x_cols
        )));
    }
    let active_x_cols = crate::paper::relations::superneo_public_x_cols(input.m_in);
    if active_x_cols > input.x_cols {
        return Err(Error::Shape(format!(
            "enforce_accumulator_ce_claim_digest: active_x_cols ({active_x_cols}) > x_cols ({})",
            input.x_cols
        )));
    }
    enforce_unique_inactive_x_zero(
        builder,
        input.x_flat_row_major,
        input.x_rows,
        input.x_cols,
        active_x_cols,
    );

    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, ACCUMULATOR_CE_CLAIM_DIGEST_DOMAIN);

    preimage.push(alloc_constant_var(builder, F::from_u64(input.c_d as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(input.c_kappa as u64)));
    extend_f_slice_wires(builder, &mut preimage, input.c_data);

    preimage.push(alloc_constant_var(builder, F::from_u64(input.x_rows as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(input.x_cols as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(active_x_cols as u64)));
    for r in 0..input.x_rows {
        for c in 0..active_x_cols {
            preimage.push(input.x_flat_row_major[r * input.x_cols + c]);
        }
    }

    extend_kvar_slice(builder, &mut preimage, input.r);
    extend_kvar_slice(builder, &mut preimage, input.s_col);
    extend_kvar_rows(builder, &mut preimage, input.y_ring);
    extend_kvar_slice(builder, &mut preimage, input.ct);
    // aux_openings: empty in the current clean pipeline.
    preimage.push(alloc_constant_var(builder, F::ZERO));
    preimage.push(alloc_constant_var(builder, F::from_u64(input.m_in as u64)));
    preimage.extend_from_slice(&input.fold_digest_fields);
    // c_step_coords.len, u_offset, u_len: all zero in this pipeline.
    preimage.push(alloc_constant_var(builder, F::ZERO));
    preimage.push(alloc_constant_var(builder, F::ZERO));
    preimage.push(alloc_constant_var(builder, F::ZERO));
    append_adv_leaves_circuit(builder, &mut preimage, input.adv);

    Ok(
        enforce_sis_accumulator_digest(builder, ACCUMULATOR_CE_CLAIM_SIS_CONFIG, &preimage)
            .expect("fixed nonempty accumulator CE-claim SIS preimage")
            .digest,
    )
}

/// Mirror of `paper::digest::accumulator_claims_digest`.
///
/// Child digests are absorbed in index order, so neither permutation nor a
/// different strict-PiDEC decomposition of the same parent can alias before
/// the Poseidon2 binding assumption.
pub fn enforce_accumulator_claims_digest(builder: &mut R1csBuilder, child_digests: &[[Var; 4]]) -> [Var; 4] {
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, ACCUMULATOR_CLAIMS_DIGEST_DOMAIN);
    preimage.push(alloc_constant_var(builder, F::from_u64(child_digests.len() as u64)));
    for digest in child_digests {
        preimage.extend_from_slice(digest);
    }
    enforce_poseidon2_hash(builder, &preimage)
}

/// Mirror of `crate::paper::digest::pi_ccs_outputs_digest`.
///
/// This binds the new Π_CCS output evaluation messages as Fiat-Shamir input
/// before Π_RLC derives `rho`, matching SuperNeo §7.3-§7.4. Commitment, X,
/// product-commitment, shape, and challenge fields are omitted deliberately:
/// the Π_CCS verifier has already bound them to transcript-authenticated inputs
/// or derived challenges by wire equality. Rehashing them here is redundant.
/// `profile` must come from verifier-owned relation shape; this function never
/// infers protocol dimensions from the message.
pub fn enforce_pi_ccs_outputs_digest(
    builder: &mut R1csBuilder,
    profile: Profile,
    inputs: &[PiCcsOutputMessageDigestInputs<'_>],
) -> Result<PiCcsOutputsDigestWires, Error> {
    let preimage = encode_pi_ccs_outputs_preimage(builder, profile, inputs)?;
    let wires = preimage.wires();
    builder.begin_encoding_stage(stage::OUTPUT_MESSAGE_SIS);
    let sis = enforce_sis_accumulator_digest(builder, PI_CCS_OUTPUTS_SIS_CONFIG, &wires)
        .expect("fixed nonempty PiCCS-output SIS preimage");

    Ok(PiCcsOutputsDigestWires {
        digest: sis.digest,
        preimage,
        sis_layout: sis.layout,
    })
}

fn enforce_unique_inactive_x_zero(builder: &mut R1csBuilder, x: &[Var], rows: usize, cols: usize, active_cols: usize) {
    let mut constrained = std::collections::HashSet::new();
    for r in 0..rows {
        for c in active_cols..cols {
            let wire = x[r * cols + c];
            if constrained.insert(wire.col()) {
                builder.enforce_eq(&Lc::from_var(wire), &Lc::zero());
            }
        }
    }
}

// The fields deliberately absent above are all pinned before this digest is
// used: commitment/X by `bind_outputs_to_inputs`, r/s_col by verifier
// challenges, ct by `enforce_ct_from_y_ring`, fold_digest by the header
// catch-up, and structural or padding fields by shape/canonicality checks.
// Keeping that reconstruction at the call site is the soundness condition for
// this projection.

fn extend_kvar_slice(builder: &mut R1csBuilder, preimage: &mut Vec<Var>, values: &[KVar]) {
    preimage.push(alloc_constant_var(builder, F::from_u64(values.len() as u64)));
    for value in values {
        preimage.push(value.c0);
        preimage.push(value.c1);
    }
}

fn extend_kvar_rows(builder: &mut R1csBuilder, preimage: &mut Vec<Var>, rows: &[Vec<KVar>]) {
    preimage.push(alloc_constant_var(builder, F::from_u64(rows.len() as u64)));
    for row in rows {
        extend_kvar_slice(builder, preimage, row);
    }
}

fn append_adv_leaves_circuit(builder: &mut R1csBuilder, preimage: &mut Vec<Var>, adv: Option<&AdvCommitmentWires>) {
    let Some(adv) = adv else {
        return;
    };
    preimage.push(alloc_constant_var(builder, F::from_u64(NEBULA_ADV_PRESENT_MARKER)));
    for digest in [
        enforce_nebula_leaf_digest_circuit(builder, NEBULA_LEAF_OPS_TAG, adv.ops.d, adv.ops.kappa, &adv.ops.data),
        enforce_nebula_leaf_digest_circuit(builder, NEBULA_LEAF_MEM_TAG, adv.is.d, adv.is.kappa, &adv.is.data),
        enforce_nebula_leaf_digest_circuit(builder, NEBULA_LEAF_MEM_TAG, adv.fs.d, adv.fs.kappa, &adv.fs.data),
    ] {
        preimage.extend_from_slice(&digest);
    }
}

/// Mirror of `crate::paper::digest::pi_ccs_instance_digest`. Hashes the
/// pre-computed per-claim digests so the SplitNcV1 verifier can recompute
/// the public-instance digest from authoritative claim wires rather than
/// trust a prover-supplied value.
///
/// Preimage:
/// ```text
/// pack_bytes_as_fields(b"neo.fold.clean/pi_ccs_instance_digest/v1")
/// ‖ fresh.len ‖ for each fresh: ccs_claim_digest[0..4]
/// ‖ running.len ‖ for each running: ce_claim_digest[0..4]
/// ```
pub fn enforce_pi_ccs_instance_digest(
    builder: &mut R1csBuilder,
    fresh_digests: &[[Var; 4]],
    running_digests: &[[Var; 4]],
) -> [Var; 4] {
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, PI_CCS_INSTANCE_DIGEST_DOMAIN);

    preimage.push(alloc_constant_var(builder, F::from_u64(fresh_digests.len() as u64)));
    for d in fresh_digests {
        preimage.extend_from_slice(d);
    }

    preimage.push(alloc_constant_var(builder, F::from_u64(running_digests.len() as u64)));
    for d in running_digests {
        preimage.extend_from_slice(d);
    }

    enforce_poseidon2_hash(builder, &preimage)
}

/// Mirror of `crate::paper::digest::pi_ccs_instance_digest_parent_authority`.
pub fn enforce_pi_ccs_instance_digest_parent_authority(
    builder: &mut R1csBuilder,
    fresh_digests: &[[Var; 4]],
    running_count: usize,
    running_parent_digest: Option<[Var; 4]>,
) -> [Var; 4] {
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, PI_CCS_PARENT_AUTHORITY_INSTANCE_DIGEST_DOMAIN);

    preimage.push(alloc_constant_var(builder, F::from_u64(fresh_digests.len() as u64)));
    for d in fresh_digests {
        preimage.extend_from_slice(d);
    }

    preimage.push(alloc_constant_var(builder, F::from_u64(running_count as u64)));
    match (running_count, running_parent_digest) {
        (0, None) => preimage.push(alloc_constant_var(builder, F::ZERO)),
        (_, Some(digest)) => {
            preimage.push(alloc_constant_var(builder, F::ONE));
            preimage.extend_from_slice(&digest);
        }
        (_, None) => preimage.push(alloc_constant_var(builder, F::from_u64(u64::MAX))),
    }

    enforce_poseidon2_hash(builder, &preimage)
}
