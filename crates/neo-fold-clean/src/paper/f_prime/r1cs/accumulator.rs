//! Outgoing Construction-2 accumulator binding inside F′.
//!
//! Owns: the current conservative Rust CE-core serialization, one
//! SIS-compressed digest per ordered child, and the outer Poseidon2
//! arity-plus-child-digests hash.
//!
//! Does not own: PiDEC validation, the checked parent cache, `state_x_out`, or
//! the unresolved `y_zcol` source relation.
//!
//! Emits constraints: yes.
//!
//! Authority boundary: the parent is deliberately absent because strict
//! PiDEC recomposition is not child-vector injective. This serializer is not
//! yet the Lean-minimal Phi81 family payload; that reduction waits on the
//! concrete 270-coordinate Rust/Lean bridge.
//!
//! | Stage path | Mathematical obligation | Current payload | Lean owner |
//! |---|---|---|---|
//! | `fprime.recursive.step.accumulator.output_authority.child_digests` | `d_i = SIS_claim(enc_core(child_i))` in index order | conservative CE core; excludes `y_zcol` | `FPrime.AccumulatorBinding.claim_eq_or_failure` |
//! | `fprime.recursive.step.accumulator.output_authority.aggregate` | `H_acc(k || d_0 || ... || d_(k-1))` | arity plus ordered child digests | `FPrime.AccumulatorBinding.digest_eq_or_failure` |

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{R1csBuilder, Var};
use crate::paper::f_prime::stage;
use crate::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_accumulator_ce_claim_digest, enforce_accumulator_claims_digest, enforce_pending_accumulator_family_digest,
    AccumulatorCeClaimDigestInputs, PendingAccumulatorFamilyChildInputs, PendingAccumulatorFamilyDigestInputs,
    PendingAccumulatorFamilyStateInputs, PendingProjectionWires,
};
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;

use super::Error;

pub(super) fn enforce_nifs_output_acc_digest(
    builder: &mut R1csBuilder,
    children: &[CeClaimWires],
    outgoing_pending: Option<&PendingProjectionWires>,
) -> Result<[Var; 4], Error> {
    enforce_output_acc_digest(builder, children, outgoing_pending, true)
}

/// Terminal-fold entrypoint for the same profile-aware codec. The terminal
/// caller owns its row-family boundary, so this variant preserves the legacy
/// terminal stage layout instead of adding recursive-F' stage markers.
pub(crate) fn enforce_terminal_output_acc_digest(
    builder: &mut R1csBuilder,
    children: &[CeClaimWires],
    outgoing_pending: Option<&PendingProjectionWires>,
) -> Result<[Var; 4], Error> {
    enforce_output_acc_digest(builder, children, outgoing_pending, false)
}

fn enforce_output_acc_digest(
    builder: &mut R1csBuilder,
    children: &[CeClaimWires],
    outgoing_pending: Option<&PendingProjectionWires>,
    record_recursive_stages: bool,
) -> Result<[Var; 4], Error> {
    if let Some(pending) = outgoing_pending {
        return enforce_pending_output_digest(builder, children, pending, record_recursive_stages);
    }
    if record_recursive_stages {
        builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR_OUTPUT_CHILD_DIGESTS);
    }
    let child_digests = children
        .iter()
        .map(|child| enforce_child_digest(builder, child))
        .collect::<Result<Vec<_>, _>>()?;
    if record_recursive_stages {
        builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR_OUTPUT_AGGREGATE);
    }
    Ok(enforce_accumulator_claims_digest(builder, &child_digests))
}

fn enforce_pending_output_digest(
    builder: &mut R1csBuilder,
    children: &[CeClaimWires],
    pending: &PendingProjectionWires,
    record_recursive_stage: bool,
) -> Result<[Var; 4], Error> {
    if record_recursive_stage {
        builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR_OUTPUT_PENDING_FAMILY);
    }
    let Some(first) = children.first() else {
        return Err(Error::Inner("pending-family output accumulator has no children".into()));
    };
    let x_active: Vec<Vec<Var>> = children
        .iter()
        .map(|child| {
            let active_columns = child.m_in.div_ceil(neo_math::D);
            (0..active_columns)
                .flat_map(|column| (0..neo_math::D).map(move |lane| child.x[lane * child.x_cols + column]))
                .collect()
        })
        .collect();
    let y_ring_active: Vec<Vec<Vec<KVar>>> = children
        .iter()
        .map(|child| {
            y_ring_kvars(child).map(|rows| {
                rows.into_iter()
                    .map(|row| row[..neo_math::D].to_vec())
                    .collect()
            })
        })
        .collect::<Result<_, _>>()?;
    let child_inputs: Vec<PendingAccumulatorFamilyChildInputs<'_>> = children
        .iter()
        .enumerate()
        .map(|(index, child)| PendingAccumulatorFamilyChildInputs {
            c_data: &child.c_data,
            x_active_column_major: &x_active[index],
            y_ring_active: &y_ring_active[index],
        })
        .collect();
    Ok(enforce_pending_accumulator_family_digest(
        builder,
        &PendingAccumulatorFamilyDigestInputs {
            verifier_rows: first.c_kappa,
            row_point: &first.r,
            column_point: &first.s_col,
            m_in: first.m_in,
            fold_digest_fields: first.fold_digest_fields,
            children: &child_inputs,
            pending: Some(PendingAccumulatorFamilyStateInputs {
                old_block: &pending.old_block,
                parent_y_zcol: &pending.parent_y_zcol,
            }),
        },
    )
    .map_err(|error| Error::Inner(format!("pending-family output digest: {error}")))?
    .digest)
}

fn enforce_child_digest(builder: &mut R1csBuilder, claim: &CeClaimWires) -> Result<[Var; 4], Error> {
    let y_ring = y_ring_kvars(claim)?;
    enforce_accumulator_ce_claim_digest(
        builder,
        &AccumulatorCeClaimDigestInputs {
            c_d: claim.c_d,
            c_kappa: claim.c_kappa,
            c_data: &claim.c_data,
            x_rows: claim.x_rows,
            x_cols: claim.x_cols,
            x_flat_row_major: &claim.x,
            r: &claim.r,
            s_col: &claim.s_col,
            y_ring: &y_ring,
            ct: &claim.ct,
            m_in: claim.m_in,
            fold_digest_fields: claim.fold_digest_fields,
            adv: claim.adv.as_ref(),
        },
    )
    .map_err(|error| Error::Inner(format!("output accumulator CE digest: {error}")))
}

fn y_ring_kvars(claim: &CeClaimWires) -> Result<Vec<Vec<KVar>>, Error> {
    claim
        .y_ring
        .iter()
        .enumerate()
        .map(|(matrix, row)| flat_kvars(row, claim.y_ring_lanes, &format!("y_ring[{matrix}]")))
        .collect()
}

fn flat_kvars(flat: &[Var], lanes: usize, what: &str) -> Result<Vec<KVar>, Error> {
    let expected = lanes * 2;
    if flat.len() != expected {
        return Err(Error::Inner(format!(
            "{what} has {} base-field limbs, expected {expected} for {lanes} K-lanes",
            flat.len()
        )));
    }
    Ok((0..lanes)
        .map(|lane| KVar {
            c0: flat[2 * lane],
            c1: flat[2 * lane + 1],
        })
        .collect())
}
