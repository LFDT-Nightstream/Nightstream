//! Owns deterministic setup witnesses for side-opening Spartan shape generation.

use neo_math::F;
use neo_transcript::Transcript;

use crate::proof::PublicStep;
use crate::public_proof::rv32im::side_opening_relation::{
    build_rv32im_kernel_opening_claim_from_statement, validate_rv32im_side_opening_relation_statement,
};
use crate::rv32im::kernel::{
    build_claim_packaged_public_step, build_kernel_binding_opening_public_step,
    build_kernel_prepared_step_opening_public_step, SimpleKernelError,
};
use crate::rv32im::stage1::Stage1RowBinding;
use crate::rv32im::stage2::{
    RamAccessKind, RamEvent, RegisterReadEvent, RegisterReadRole, RegisterWriteEvent, TwistLinkEvent,
};
use crate::rv32im::stage3::ContinuityEvent;
use crate::rv32im::tables::Rv32FamilyTag;

use super::*;

pub(super) fn validate_side_opening_statement(
    statement: &Rv32imSideOpeningRelationStatement,
) -> Result<(), SimpleKernelError> {
    validate_rv32im_side_opening_relation_statement(statement)
}

pub(super) fn rv32im_side_opening_shape_digest(
    statement: &Rv32imSideOpeningRelationStatement,
    _: &Rv32imSideOpeningRelationWitness,
) -> [u8; 32] {
    let mut tr = neo_transcript::Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_opening_shape");
    tr.append_message(
        b"neo.fold.next/nightstream/rv32im/side_opening_shape/statement_digest",
        &statement.expected_digest(),
    );
    tr.digest32()
}

pub(super) fn dummy_rv32im_side_opening_witness(
    statement: &Rv32imSideOpeningRelationStatement,
) -> Result<Rv32imSideOpeningRelationWitness, SimpleKernelError> {
    let kernel_opening_claim = build_rv32im_kernel_opening_claim_from_statement(statement)?;
    Ok(Rv32imSideOpeningRelationWitness {
        stage1_selected_rows: Rv32imStage1SelectedRowsWitness {
            first: dummy_stage1_row_binding(),
            effect_position: 0,
            effect: dummy_stage1_row_binding(),
            commit_position: 0,
            commit: dummy_stage1_row_binding(),
            last: dummy_stage1_row_binding(),
        },
        stage2_selected_events: Rv32imStage2SelectedEventsWitness {
            first_read: statement
                .stage2
                .claim
                .points
                .first_read
                .as_ref()
                .map(|_| dummy_register_read_event()),
            last_read: statement
                .stage2
                .claim
                .points
                .last_read
                .as_ref()
                .map(|_| dummy_register_read_event()),
            first_write: statement
                .stage2
                .claim
                .points
                .first_write
                .as_ref()
                .map(|_| dummy_register_write_event()),
            last_write: statement
                .stage2
                .claim
                .points
                .last_write
                .as_ref()
                .map(|_| dummy_register_write_event()),
            first_ram: statement
                .stage2
                .claim
                .points
                .first_ram
                .as_ref()
                .map(|_| dummy_ram_event()),
            last_ram: statement
                .stage2
                .claim
                .points
                .last_ram
                .as_ref()
                .map(|_| dummy_ram_event()),
            first_twist: statement
                .stage2
                .claim
                .points
                .first_twist
                .as_ref()
                .map(|_| dummy_twist_link_event()),
            last_twist: statement
                .stage2
                .claim
                .points
                .last_twist
                .as_ref()
                .map(|_| dummy_twist_link_event()),
        },
        stage3_selected_continuity: Rv32imStage3SelectedContinuityWitness {
            first_continuity: statement
                .stage3
                .claim
                .points
                .first_continuity
                .as_ref()
                .map(|_| dummy_continuity_event()),
            last_continuity: statement
                .stage3
                .claim
                .points
                .last_continuity
                .as_ref()
                .map(|_| dummy_continuity_event()),
        },
        stage1_packaged: dummy_single_step_packaged_witness(&build_claim_packaged_public_step(
            "rv32im/stage1",
            &statement.stage1.claim.claim_words(),
        )?),
        stage2_packaged: dummy_single_step_packaged_witness(&build_claim_packaged_public_step(
            "rv32im/stage2",
            &statement.stage2.claim.claim_words(),
        )?),
        stage3_packaged: dummy_single_step_packaged_witness(&build_claim_packaged_public_step(
            "rv32im/stage3",
            &statement.stage3.claim.claim_words(),
        )?),
        bindings_packaged: dummy_single_step_packaged_witness(&build_kernel_binding_opening_public_step(
            &kernel_opening_claim.bindings,
        )?),
        prepared_steps_packaged: dummy_single_step_packaged_witness(&build_kernel_prepared_step_opening_public_step(
            &kernel_opening_claim.prepared_steps,
        )?),
    })
}

pub(super) fn setup_rv32im_side_opening_witness(
    statement: &Rv32imSideOpeningRelationStatement,
    _: &Rv32imSideOpeningRelationWitness,
) -> Result<Rv32imSideOpeningRelationWitness, SimpleKernelError> {
    dummy_rv32im_side_opening_witness(statement)
}

pub(super) fn setup_rv32im_side_opening_witness_without_packaged_final_main_claims(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<Rv32imSideOpeningRelationWitness, SimpleKernelError> {
    let mut setup_witness = setup_rv32im_side_opening_witness(statement, witness)?;
    for packaged in [
        &mut setup_witness.stage1_packaged,
        &mut setup_witness.stage2_packaged,
        &mut setup_witness.stage3_packaged,
        &mut setup_witness.bindings_packaged,
        &mut setup_witness.prepared_steps_packaged,
    ] {
        packaged.final_main_claim_digests = vec![[F::ZERO; 4]; RV32IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT];
    }
    Ok(setup_witness)
}

fn dummy_single_step_packaged_witness(
    step: &PublicStep,
) -> super::super::side_claim_relation::Rv32imSingleStepPackagedProofWitness {
    super::super::side_claim_relation::Rv32imSingleStepPackagedProofWitness {
        step: step.clone(),
        final_main_claim_digests: vec![[F::ZERO; 4]; RV32IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT],
        proof_digest: [0; 32],
    }
}

fn dummy_stage1_row_binding() -> Stage1RowBinding {
    Stage1RowBinding {
        trace_index: 0,
        step_index: 0,
        sequence_index: 0,
        fetch_pc: 0,
        fetched_word: 0,
        opcode: crate::rv32im::isa::Rv32Opcode::Addi,
        trace_opcode: None,
        trace_virtual_opcode: None,
        family: Rv32FamilyTag::NativeAlu,
        next_pc: 0,
        alu_result: 0,
        effective_addr: None,
        writes_rd: false,
        rd: 0,
        rd_after: 0,
        is_first_in_sequence: false,
        virtual_sequence_remaining: None,
        is_effect_row: false,
        is_commit_row: false,
        is_real: false,
        preserves_x0: true,
    }
}

fn dummy_register_read_event() -> RegisterReadEvent {
    RegisterReadEvent {
        trace_index: 0,
        step_index: 0,
        role: RegisterReadRole::Rs1,
        reg: 0,
        value: 0,
    }
}

fn dummy_register_write_event() -> RegisterWriteEvent {
    RegisterWriteEvent {
        trace_index: 0,
        step_index: 0,
        reg: 0,
        previous: 0,
        next: 0,
    }
}

fn dummy_ram_event() -> RamEvent {
    RamEvent {
        trace_index: 0,
        step_index: 0,
        kind: RamAccessKind::Read,
        addr: 0,
        previous: 0,
        next: 0,
    }
}

fn dummy_twist_link_event() -> TwistLinkEvent {
    TwistLinkEvent {
        trace_index: 0,
        step_index: 0,
        family: Rv32FamilyTag::NativeAlu,
        routed_write_value: None,
        routed_memory_before: None,
        routed_memory_after: None,
    }
}

fn dummy_continuity_event() -> ContinuityEvent {
    ContinuityEvent {
        step_index: 0,
        pc: 0,
        next_pc: 0,
        successor_pc: None,
        final_step: false,
        continuity_holds: false,
    }
}
