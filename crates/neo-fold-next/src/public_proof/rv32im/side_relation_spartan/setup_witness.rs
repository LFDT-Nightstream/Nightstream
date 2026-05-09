//! Owns deterministic and proof-time witnesses for side-binding setup.

use std::sync::Arc;

use neo_ajtai::Commitment;
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::rv32im::kernel::{
    phase0_full_width_for_schema, FamilyEvalClaimWitness, OpenedAjtaiObjectWitness, PackedColumnOracleRef,
    SimpleKernelError,
};

use super::*;

pub(super) fn expected_payload_coeffs(claim: &crate::rv32im::kernel::FamilyEvalClaim) -> Vec<Vec<K>> {
    claim
        .payload
        .column_evals
        .iter()
        .map(|column| column.coeffs.to_vec())
        .collect()
}

pub(super) fn build_dummy_opened_object_witnesses(
    public: &Rv32imSideOpeningPublic,
) -> Result<Vec<Arc<OpenedAjtaiObjectWitness>>, SimpleKernelError> {
    public
        .evals
        .iter()
        .map(|eval| build_dummy_opened_object_witness(&eval.claim))
        .collect()
}

fn build_dummy_opened_object_witness(
    claim: &crate::rv32im::kernel::FamilyEvalClaim,
) -> Result<Arc<OpenedAjtaiObjectWitness>, SimpleKernelError> {
    let params =
        NeoParams::goldilocks_auto_r1cs_ccs(phase0_full_width_for_schema(claim.payload.schema)).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM side binding could not derive Phase 0 dummy parameters for {:?}: {err}",
                claim.payload.schema
            ))
        })?;
    let row_len = 1usize << (claim.opened_object.row_domain_log_size as usize);
    let packed_column_count = claim.payload.column_evals.len();
    let packed_columns = (0..packed_column_count)
        .map(|column_index| PackedColumnOracleRef {
            column_index: column_index as u32,
            rows: vec![[F::ZERO; D]; row_len],
        })
        .collect::<Vec<_>>();
    let commitment_vector = (0..packed_column_count)
        .map(|_| Commitment::zeros(D, params.kappa as usize))
        .collect::<Vec<_>>();
    Ok(Arc::new(OpenedAjtaiObjectWitness {
        opened_object: claim.opened_object.clone(),
        commitment_context: claim.commitment_context.clone(),
        row_domain_log_size: claim.opened_object.row_domain_log_size,
        packed_column_count: packed_column_count as u32,
        packed_columns,
        commitment_vector,
    }))
}

pub(super) fn build_opened_object_witnesses_from_claim_witnesses(
    public: &Rv32imSideOpeningPublic,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Vec<Arc<OpenedAjtaiObjectWitness>>, SimpleKernelError> {
    if claim_witnesses.len() != public.evals.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM side binding prove path claim-witness count does not match the carried public eval set".into(),
        ));
    }
    public
        .evals
        .iter()
        .zip(claim_witnesses.iter())
        .map(|(eval, claim_witness)| {
            if eval.claim != claim_witness.claim {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM side binding prove path {:?}/{} claim does not match the carried public eval",
                    eval.claim.payload.schema, eval.claim.id.slot
                )));
            }
            Ok(claim_witness.witness.clone())
        })
        .collect()
}

pub(super) fn rv32im_side_binding_shape_digest(public: &Rv32imSideOpeningPublic) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_binding_shape");
    tr.append_u64s(
        b"neo.fold.next/nightstream/rv32im/side_binding_shape/counts",
        &[public.opened_objects.len() as u64, public.evals.len() as u64],
    );
    for eval in &public.evals {
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/side_binding_shape/eval",
            &[
                eval.claim.payload.schema.tag(),
                eval.claim.point.len() as u64,
                eval.claim.payload.column_evals.len() as u64,
            ],
        );
    }
    tr.digest32()
}
