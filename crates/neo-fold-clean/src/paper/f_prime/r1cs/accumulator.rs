//! Outgoing Construction-2 accumulator binding inside F′.
//!
//! Owns: the strict-binary family serialization, bounded SIS chunks, and the
//! outer Poseidon2 aggregate.
//!
//! Does not own: PiDEC validation, the checked parent cache, `state_x_out`, or
//! the exact selected CE claim relation.
//!
//! Emits constraints: yes.
//!
//! Authority boundary: strict PiDEC makes child X the unique split of parent
//! X and pins all omitted shared or derived fields. Every non-unique child
//! commitment, active evaluation, and Nebula coordinate remains serialized.
//!
//! | Stage path | Mathematical obligation | Current payload | Lean owner |
//! |---|---|---|---|
//! | `fprime.recursive.step.accumulator.output_authority.claimed_digest` | allocate the claimed outgoing digest as an authoritative private input | four base-field elements | open |
//! | `fprime.recursive.step.accumulator.output_authority.child_digests` | serialize the canonical family in child order and SIS-bind bounded chunks | parent X plus exact non-unique child fields | `FPrime.AccumulatorBinding.claim_eq_or_failure` |
//! | `fprime.recursive.step.accumulator.output_authority.aggregate` | bind ordered chunk digests and total length | ordered chunks and the claimed digest | `FPrime.AccumulatorBinding.digest_eq_or_failure` |

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{R1csBuilder, Var};
use crate::paper::f_prime::stage;
use crate::paper::reductions::pi_ccs_circuit::{
    enforce_strict_binary_accumulator_family_digest, AccumulatorCeClaimDigestInputs,
};
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;

use super::Error;

pub(super) fn enforce_nifs_output_acc_digest(
    builder: &mut R1csBuilder,
    parent: &CeClaimWires,
    children: &[CeClaimWires],
) -> Result<[Var; 4], Error> {
    enforce_output_acc_digest(builder, parent, children, true)
}

/// Terminal-fold entrypoint for the same profile-aware codec. The terminal
/// caller owns its row-family boundary, so this variant preserves the legacy
/// terminal stage layout instead of adding recursive-F' stage markers.
pub(crate) fn enforce_terminal_output_acc_digest(
    builder: &mut R1csBuilder,
    parent: &CeClaimWires,
    children: &[CeClaimWires],
) -> Result<[Var; 4], Error> {
    enforce_output_acc_digest(builder, parent, children, false)
}

fn enforce_output_acc_digest(
    builder: &mut R1csBuilder,
    parent: &CeClaimWires,
    children: &[CeClaimWires],
    record_recursive_stages: bool,
) -> Result<[Var; 4], Error> {
    if record_recursive_stages {
        builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR_OUTPUT_CHILD_DIGESTS);
    }
    let parent_y_ring = y_ring_kvars(parent)?;
    let child_y_rings = children
        .iter()
        .map(y_ring_kvars)
        .collect::<Result<Vec<_>, _>>()?;
    let parent_inputs = accumulator_inputs(parent, &parent_y_ring);
    let child_inputs = children
        .iter()
        .zip(&child_y_rings)
        .map(|(child, y_ring)| accumulator_inputs(child, y_ring))
        .collect::<Vec<_>>();
    if record_recursive_stages {
        builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR_OUTPUT_AGGREGATE);
    }
    enforce_strict_binary_accumulator_family_digest(builder, &parent_inputs, &child_inputs)
        .map_err(|error| Error::Inner(format!("output accumulator family digest: {error}")))
}

fn accumulator_inputs<'a>(claim: &'a CeClaimWires, y_ring: &'a [Vec<KVar>]) -> AccumulatorCeClaimDigestInputs<'a> {
    AccumulatorCeClaimDigestInputs {
        c_d: claim.c_d,
        c_kappa: claim.c_kappa,
        c_data: &claim.c_data,
        x_rows: claim.x_rows,
        x_cols: claim.x_cols,
        x_flat_row_major: &claim.x,
        r: &claim.r,
        y_ring,
        ct: &claim.ct,
        m_in: claim.m_in,
        fold_digest_fields: claim.fold_digest_fields,
        adv: claim.adv.as_ref(),
    }
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
