//! Claim and message digest gadgets for the selected PiCCS circuit.
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
//! | accumulator child digest | Bind one ordered child claim | yes | `enforce_accumulator_ce_claim_digest` | exact-child bridge open |
//! | accumulator handle | Bind the ordered paper-level `CE(b)^k` child vector | yes | `enforce_accumulator_claims_digest` | exact-child bridge open |
//! | instance digest | Bind fresh claims and checked running parent | yes | `enforce_pi_ccs_instance_digest_parent_authority` | authority bridge open |
//! | output digest | Bind the profile-pinned Pi_CCS message consumed by Pi_RLC | yes | `enforce_pi_ccs_outputs_digest` | source layout model-level; physical SIS bridge open |
//!
//! Mirrors:
//! - `crate::paper::digest::ccs_claim_digest` (per-fresh CCS claim).
//! - `crate::paper::digest::pi_ccs_instance_digest` (the authoritative
//!   public-instance digest the selected verifier must absorb in place of
//!   any prover-supplied value).
//!
//! Project soundness rule: the PiCCS composition must recompute
//! these digests in-circuit from authoritative claim wires — never
//! accept a prover-supplied digest as authority.
//!
//! The ME-input handle binds the exact ordered paper-level running children.
//! The checked Π_RLC parent remains a recomposition cache: strict Π_DEC
//! consistency does not uniquely determine its child vector.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::output_message::{encode_pi_ccs_outputs_preimage, PiCcsOutputMessageDigestInputs, PiCcsOutputsPreimage};
use super::{alloc_constant_var, extend_f_slice_wires, extend_packed_bytes_as_fields_wires, stage, Error};
use crate::engine::r1cs_circuit::builder::Var;
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::poseidon2::enforce_poseidon2_hash;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::digest::{
    ACCUMULATOR_FAMILY_AGGREGATE_TAG, ACCUMULATOR_FAMILY_DIGEST_TAG, NEBULA_ADV_PRESENT_MARKER, NEBULA_LEAF_MEM_TAG,
    NEBULA_LEAF_OPS_TAG,
};
use crate::paper::f_prime::nebula_lane_circuit::enforce_nebula_leaf_digest_circuit;
use crate::paper::reductions::accumulator_sis_circuit::{
    enforce_accumulator_digest as enforce_sis_accumulator_digest, SisAccumulatorCircuitLayout,
    ACCUMULATOR_CE_CLAIM_SIS_CONFIG, CCS_CLAIM_SIS_CONFIG, PI_CCS_OUTPUTS_SIS_CONFIG, PROTOCOL_BINDING_MAX_FIELDS,
};
use crate::paper::reductions::pi_ccs_output_message::Profile;
use crate::paper::relations::product_commitment_circuit::AdvCommitmentWires;

/// Domain bytes for the paper-layer per-claim digests (mirrors
/// `crate::paper::digest::*`). Kept byte-identical to the native strings.
const CCS_CLAIM_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/ccs_claim_digest/v1";
const ACCUMULATOR_CE_CLAIM_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/accumulator_ce_claim_digest/v3";
const ACCUMULATOR_CLAIMS_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/accumulator/children/v4";
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
/// `c_data` and `x` are witness `F`-wires. This gadget is what the one-joint
/// composition uses to build per-fresh digests for
/// [`enforce_pi_ccs_instance_digest_parent_authority`] — not an arbitrary digest the prover
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

/// Mirror of `crate::paper::digest::accumulator_ce_claim_digest`.
///
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
    if input.x_cols != active_x_cols {
        return Err(Error::Shape(format!(
            "enforce_accumulator_ce_claim_digest: x_cols ({}) must equal compact coefficient width ({active_x_cols})",
            input.x_cols
        )));
    }

    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, ACCUMULATOR_CE_CLAIM_DIGEST_DOMAIN);

    preimage.push(alloc_constant_var(builder, F::from_u64(input.c_d as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(input.c_kappa as u64)));
    extend_f_slice_wires(builder, &mut preimage, input.c_data);

    preimage.push(alloc_constant_var(builder, F::from_u64(input.x_rows as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(input.x_cols as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(active_x_cols as u64)));
    for r in 0..input.x_rows {
        for c in 0..input.x_cols {
            preimage.push(input.x_flat_row_major[r * input.x_cols + c]);
        }
    }

    extend_kvar_slice(builder, &mut preimage, input.r);
    extend_kvar_rows(builder, &mut preimage, input.y_ring);
    extend_kvar_slice(builder, &mut preimage, input.ct);
    preimage.push(alloc_constant_var(builder, F::from_u64(input.m_in as u64)));
    preimage.extend_from_slice(&input.fold_digest_fields);
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

/// Circuit mirror of `strict_binary_accumulator_family_digest`.
///
/// The caller must also enforce strict PiDEC for `parent` and `children`.
/// That relation proves that omitted child `X`, `r`, `ct`, fold-digest, and
/// padding fields are uniquely derived from the fields serialized here.
pub fn enforce_strict_binary_accumulator_family_digest(
    builder: &mut R1csBuilder,
    parent: &AccumulatorCeClaimDigestInputs<'_>,
    children: &[AccumulatorCeClaimDigestInputs<'_>],
) -> Result<[Var; 4], Error> {
    enforce_strict_binary_accumulator_family_digest_inner(builder, parent, children, None)
}

pub(crate) fn enforce_strict_binary_accumulator_family_digest_with_aggregate_stage(
    builder: &mut R1csBuilder,
    parent: &AccumulatorCeClaimDigestInputs<'_>,
    children: &[AccumulatorCeClaimDigestInputs<'_>],
    aggregate_stage: &'static str,
) -> Result<[Var; 4], Error> {
    enforce_strict_binary_accumulator_family_digest_inner(builder, parent, children, Some(aggregate_stage))
}

fn enforce_strict_binary_accumulator_family_digest_inner(
    builder: &mut R1csBuilder,
    parent: &AccumulatorCeClaimDigestInputs<'_>,
    children: &[AccumulatorCeClaimDigestInputs<'_>],
    aggregate_stage: Option<&'static str>,
) -> Result<[Var; 4], Error> {
    let first = children
        .first()
        .ok_or_else(|| Error::Shape("strict-binary accumulator family is empty".into()))?;
    validate_family_member_shape("parent", parent, first)?;
    for (index, child) in children.iter().enumerate() {
        validate_family_member_shape(&format!("child[{index}]"), child, first)?;
    }

    let active_x_cols = crate::paper::relations::superneo_public_x_cols(first.m_in);
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, ACCUMULATOR_FAMILY_DIGEST_TAG);
    preimage.push(alloc_constant_var(builder, F::from_u64(children.len() as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(first.c_d as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(first.c_kappa as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(first.c_data.len() as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(first.x_rows as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(first.x_cols as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(active_x_cols as u64)));
    for row in 0..parent.x_rows {
        for column in 0..active_x_cols {
            preimage.push(parent.x_flat_row_major[row * parent.x_cols + column]);
        }
    }
    extend_kvar_slice(builder, &mut preimage, first.r);
    preimage.push(alloc_constant_var(builder, F::from_u64(first.y_ring.len() as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(neo_math::D as u64)));
    preimage.push(alloc_constant_var(builder, F::from_u64(first.m_in as u64)));
    preimage.extend_from_slice(&first.fold_digest_fields);
    preimage.push(alloc_constant_var(
        builder,
        if first.adv.is_some() { F::ONE } else { F::ZERO },
    ));

    for (index, child) in children.iter().enumerate() {
        preimage.push(alloc_constant_var(builder, F::from_u64(index as u64)));
        preimage.extend_from_slice(child.c_data);
        for row in child.y_ring {
            for value in row.iter().take(neo_math::D) {
                preimage.push(value.c0);
                preimage.push(value.c1);
            }
        }
        if let Some(adv) = child.adv {
            preimage.extend_from_slice(&adv.ops.data);
            preimage.extend_from_slice(&adv.is.data);
            preimage.extend_from_slice(&adv.fs.data);
        }
    }

    let chunk_digests = preimage
        .chunks(PROTOCOL_BINDING_MAX_FIELDS)
        .map(|chunk| {
            enforce_sis_accumulator_digest(builder, ACCUMULATOR_CE_CLAIM_SIS_CONFIG, chunk)
                .expect("bounded nonempty accumulator-family SIS chunk")
                .digest
        })
        .collect::<Vec<_>>();
    if let Some(aggregate_stage) = aggregate_stage {
        builder.begin_encoding_stage(aggregate_stage);
    }
    let mut aggregate = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut aggregate, ACCUMULATOR_FAMILY_AGGREGATE_TAG);
    aggregate.push(alloc_constant_var(builder, F::from_u64(preimage.len() as u64)));
    aggregate.push(alloc_constant_var(builder, F::from_u64(chunk_digests.len() as u64)));
    for digest in chunk_digests {
        aggregate.extend_from_slice(&digest);
    }
    Ok(enforce_poseidon2_hash(builder, &aggregate))
}

fn validate_family_member_shape(
    label: &str,
    member: &AccumulatorCeClaimDigestInputs<'_>,
    first: &AccumulatorCeClaimDigestInputs<'_>,
) -> Result<(), Error> {
    let active_x_cols = crate::paper::relations::superneo_public_x_cols(first.m_in);
    let same_shape = first.m_in % neo_math::D == 0
        && first.c_d == neo_math::D
        && first.c_kappa > 0
        && first.c_data.len() == first.c_d * first.c_kappa
        && member.c_d == first.c_d
        && member.c_kappa == first.c_kappa
        && member.c_data.len() == first.c_data.len()
        && member.c_data.len() == member.c_d * member.c_kappa
        && member.x_rows == first.x_rows
        && member.x_cols == first.x_cols
        && member.x_flat_row_major.len() == member.x_rows * member.x_cols
        && member.x_rows == neo_math::D
        && member.x_cols == active_x_cols
        && member.r.len() == first.r.len()
        && member.y_ring.len() == first.y_ring.len()
        && member
            .y_ring
            .iter()
            .zip(first.y_ring.iter())
            .all(|(row, first_row)| row.len() == first_row.len() && row.len() == neo_math::D.next_power_of_two())
        && member.ct.len() == member.y_ring.len()
        && member.m_in == first.m_in
        && member.adv.is_some() == first.adv.is_some()
        && adv_wires_have_shape(member.adv, first.c_d, first.c_kappa);
    if !same_shape {
        return Err(Error::Shape(format!(
            "{label} does not have the strict-binary accumulator-family shape"
        )));
    }
    Ok(())
}

fn adv_wires_have_shape(adv: Option<&AdvCommitmentWires>, d: usize, kappa: usize) -> bool {
    let Some(adv) = adv else {
        return true;
    };
    [&adv.ops, &adv.is, &adv.fs]
        .into_iter()
        .all(|commitment| commitment.d == d && commitment.kappa == kappa && commitment.data.len() == d * kappa)
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

// The fields deliberately absent above are all pinned before this digest is
// used: commitment/X by `bind_outputs_to_inputs`, r by verifier
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
