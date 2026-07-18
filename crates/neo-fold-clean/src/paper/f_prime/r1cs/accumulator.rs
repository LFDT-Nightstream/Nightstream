//! Outgoing Construction-2 accumulator binding inside F′.
//!
//! Owns: the current conservative Rust CE-core serialization, one Poseidon2
//! digest per ordered child, and the outer arity-plus-child-digests hash.
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
//! | `fprime.recursive.step.accumulator.output_authority.child_digests` | `d_i = H_claim(enc_core(child_i))` in index order | conservative CE core; excludes `y_zcol` | `FPrime.AccumulatorBinding.claim_eq_or_failure` |
//! | `fprime.recursive.step.accumulator.output_authority.aggregate` | `H_acc(k || d_0 || ... || d_(k-1))` | arity plus ordered child digests | `FPrime.AccumulatorBinding.digest_eq_or_failure` |

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{R1csBuilder, Var};
use crate::paper::f_prime::stage;
use crate::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_accumulator_ce_claim_digest, enforce_accumulator_claims_digest, AccumulatorCeClaimDigestInputs,
};
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;

use super::Error;

pub(super) fn enforce_nifs_output_acc_digest(
    builder: &mut R1csBuilder,
    children: &[CeClaimWires],
) -> Result<[Var; 4], Error> {
    builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR_OUTPUT_CHILD_DIGESTS);
    let child_digests = children
        .iter()
        .map(|child| enforce_child_digest(builder, child))
        .collect::<Result<Vec<_>, _>>()?;
    builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR_OUTPUT_AGGREGATE);
    Ok(enforce_accumulator_claims_digest(builder, &child_digests))
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
