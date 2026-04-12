//! Owns exact selected-opening and packaged-step digest gadgets for the side relation.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_math::F;
use p3_field::PrimeField64;
use spartan2::provider::goldi::F as SpartanF;

use crate::finalize::public_chunk_digest as native_public_chunk_digest;
use crate::proof::PublicChunk;
use crate::proof::{FoldSchedule, PublicStep};
use crate::rv64im::kernel::{
    KernelBindingOpeningClaim, KernelPreparedStepOpeningClaim, Stage1SelectedOpeningClaim, Stage2SelectedOpeningClaim,
    Stage3SelectedOpeningClaim,
};
use crate::rv64im::main_relation_circuit::public_chunk::{
    alloc_public_step, public_chunk_instance_digest, PublicStepVar,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::stage1::{stage1_row_words, Stage1RowBinding};
use crate::rv64im::stage2::{
    ram_timeline_words, register_read_timeline_words, register_write_timeline_words, twist_link_timeline_words,
    RamEvent, RegisterReadEvent, RegisterWriteEvent, TwistLinkEvent,
};
use crate::rv64im::stage3::{continuity_event_words, ContinuityEvent};

use super::exact_package::exact_vector_packaged_step_var_from_native_words_with_step_label;
use super::word::alloc_u64;

pub fn stage1_row_digest<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    row: &Stage1RowBinding,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    digest_u64_words(
        cs.namespace(|| format!("{label}_stage1")),
        b"neo.fold.next/rv64im/stage1_selected_row",
        b"stage1/row",
        &stage1_row_words(row),
        &format!("{label}_stage1"),
    )
}

pub fn register_read_event_digest<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    event: &RegisterReadEvent,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    digest_u64_words(
        cs.namespace(|| format!("{label}_read")),
        b"neo.fold.next/rv64im/stage2_selected_register_read",
        b"stage2/read",
        &register_read_timeline_words(event),
        &format!("{label}_read"),
    )
}

pub fn register_write_event_digest<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    event: &RegisterWriteEvent,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    digest_u64_words(
        cs.namespace(|| format!("{label}_write")),
        b"neo.fold.next/rv64im/stage2_selected_register_write",
        b"stage2/write",
        &register_write_timeline_words(event),
        &format!("{label}_write"),
    )
}

pub fn ram_event_digest<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    event: &RamEvent,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    digest_u64_words(
        cs.namespace(|| format!("{label}_ram")),
        b"neo.fold.next/rv64im/stage2_selected_ram_event",
        b"stage2/ram",
        &ram_timeline_words(event),
        &format!("{label}_ram"),
    )
}

pub fn twist_link_event_digest<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    event: &TwistLinkEvent,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    digest_u64_words(
        cs.namespace(|| format!("{label}_twist")),
        b"neo.fold.next/rv64im/stage2_selected_twist_link",
        b"stage2/twist",
        &twist_link_timeline_words(event),
        &format!("{label}_twist"),
    )
}

pub fn continuity_event_digest<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    event: &ContinuityEvent,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let words = std::iter::once(event.final_step as u64)
        .chain(std::iter::once(1u64))
        .chain(continuity_event_words(event))
        .collect::<Vec<_>>();
    digest_u64_words(
        cs.namespace(|| format!("{label}_continuity")),
        b"neo.fold.next/rv64im/stage3_selected_continuity",
        b"stage3/continuity",
        &words,
        &format!("{label}_continuity"),
    )
}

pub fn alloc_public_step_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    step: &PublicStep,
    label: &str,
) -> Result<PublicStepVar, SynthesisError> {
    alloc_public_step(cs, step, label)
}

pub fn single_step_packaged_statement_digest<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    step: &PublicStep,
    final_main_claim_digests: &[[F; 4]],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let step_var = alloc_public_step_var(
        &mut cs.namespace(|| format!("{label}_step")),
        step,
        &format!("{label}_step"),
    )?;
    single_step_packaged_statement_digest_from_step_var(&mut cs, step_var, step, final_main_claim_digests, label)
}

pub fn single_step_packaged_statement_digest_from_exact_words<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    words: &[u64],
    final_main_claim_digests: &[[F; 4]],
    namespace_label: &str,
    step_label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let step_var = exact_vector_packaged_step_var_from_native_words_with_step_label(
        cs.namespace(|| format!("{namespace_label}_exact_step")),
        &format!("{namespace_label}_exact_step"),
        step_label,
        words,
    )?;
    let step = crate::rv64im::kernel::build_claim_packaged_public_step(step_label, words)
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    single_step_packaged_statement_digest_from_step_var(
        &mut cs,
        step_var,
        &step,
        final_main_claim_digests,
        namespace_label,
    )
}

pub fn stage1_opening_packaged_statement_digest<CS: ConstraintSystem<SpartanF>>(
    cs: CS,
    claim: &Stage1SelectedOpeningClaim,
    final_main_claim_digests: &[[F; 4]],
    _: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    single_step_packaged_statement_digest_from_exact_words(
        cs,
        &claim.claim_words(),
        final_main_claim_digests,
        "rv64im_stage1",
        "rv64im/stage1",
    )
}

pub fn stage2_opening_packaged_statement_digest<CS: ConstraintSystem<SpartanF>>(
    cs: CS,
    claim: &Stage2SelectedOpeningClaim,
    final_main_claim_digests: &[[F; 4]],
    _: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    single_step_packaged_statement_digest_from_exact_words(
        cs,
        &claim.claim_words(),
        final_main_claim_digests,
        "rv64im_stage2",
        "rv64im/stage2",
    )
}

pub fn stage3_opening_packaged_statement_digest<CS: ConstraintSystem<SpartanF>>(
    cs: CS,
    claim: &Stage3SelectedOpeningClaim,
    final_main_claim_digests: &[[F; 4]],
    _: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    single_step_packaged_statement_digest_from_exact_words(
        cs,
        &claim.claim_words(),
        final_main_claim_digests,
        "rv64im_stage3",
        "rv64im/stage3",
    )
}

pub fn kernel_binding_opening_packaged_statement_digest<CS: ConstraintSystem<SpartanF>>(
    cs: CS,
    claim: &KernelBindingOpeningClaim,
    final_main_claim_digests: &[[F; 4]],
    _: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    single_step_packaged_statement_digest_from_exact_words(
        cs,
        &claim.claim_words(),
        final_main_claim_digests,
        "rv64im_kernel_opening_bundle_bindings",
        "rv64im/kernel_opening_bundle/bindings",
    )
}

pub fn kernel_prepared_step_opening_packaged_statement_digest<CS: ConstraintSystem<SpartanF>>(
    cs: CS,
    claim: &KernelPreparedStepOpeningClaim,
    final_main_claim_digests: &[[F; 4]],
    _: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    single_step_packaged_statement_digest_from_exact_words(
        cs,
        &claim.claim_words(),
        final_main_claim_digests,
        "rv64im_kernel_opening_bundle_prepared_steps",
        "rv64im/kernel_opening_bundle/prepared_steps",
    )
}

fn single_step_packaged_statement_digest_from_step_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    step_var: PublicStepVar,
    step: &PublicStep,
    final_main_claim_digests: &[[F; 4]],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let start_index = alloc_constant(
        cs.namespace(|| format!("{label}_start_index")),
        SpartanF::ZERO,
        &format!("{label}_start_index"),
    )?;
    let public_step_count = alloc_constant(
        cs.namespace(|| format!("{label}_public_step_count")),
        SpartanF::ONE,
        &format!("{label}_public_step_count"),
    )?;
    let chunk_digest = public_chunk_instance_digest(
        &mut cs.namespace(|| format!("{label}_chunk_digest")),
        &start_index,
        &public_step_count,
        &[step_var],
        &format!("{label}_chunk_digest"),
    )?;
    let native_chunk_digest = native_public_chunk_digest(&PublicChunk {
        start_index: 0,
        steps: vec![step.clone()],
    });
    let native_chunk_digest_values = native_chunk_digest
        .iter()
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
        .collect::<Vec<_>>();

    let flat_final_main_claim_digests = final_main_claim_digests
        .iter()
        .flat_map(|digest| digest.iter().copied())
        .collect::<Vec<_>>();
    let flat_vars = flat_final_main_claim_digests
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(
                cs.namespace(|| format!("{label}_final_main_claim_digest_{idx}")),
                || Ok(SpartanF::from_canonical_u64(value.as_canonical_u64())),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let flat_values = flat_final_main_claim_digests
        .iter()
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
        .collect::<Vec<_>>();

    let mut tr = Poseidon2TranscriptCircuit::new(
        cs.namespace(|| format!("{label}_statement_tr")),
        b"neo.fold.next/final_statement",
    )?;
    tr.append_message(
        cs.namespace(|| format!("{label}_version")),
        b"neo.fold.next/final_statement/version",
        b"v2",
    )?;
    tr.append_u64s(
        cs.namespace(|| format!("{label}_schedule")),
        b"neo.fold.next/final_statement/fold_schedule",
        &FoldSchedule::RowsPerChunk(1).meta_words(),
    )?;
    tr.append_u64s(
        cs.namespace(|| format!("{label}_header")),
        b"neo.fold.next/final_statement/header",
        &[1, final_main_claim_digests.len() as u64],
    )?;
    tr.append_fields(
        cs.namespace(|| format!("{label}_chunk")),
        b"neo.fold.next/final_statement/chunk_digest",
        &chunk_digest,
        &native_chunk_digest_values,
    )?;
    tr.append_fields(
        cs.namespace(|| format!("{label}_final_claims")),
        b"neo.fold.next/final_statement/final_main_claim_digest",
        &flat_vars,
        &flat_values,
    )?;
    tr.digest32(cs.namespace(|| format!("{label}_digest")))
}

pub fn digest_u64_words<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    app_domain: &'static [u8],
    label_bytes: &'static [u8],
    words: &[u64],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let word_vars = words
        .iter()
        .enumerate()
        .map(|(idx, word)| {
            alloc_u64(
                cs.namespace(|| format!("{label}_word_{idx}")),
                *word,
                &format!("{label}_word_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut halves = Vec::with_capacity(word_vars.len() * 2);
    let mut half_values = Vec::with_capacity(word_vars.len() * 2);
    for word in &word_vars {
        let [lo_var, hi_var] = word.half_vars();
        let [lo_value, hi_value] = word.half_values();
        halves.push(lo_var);
        halves.push(hi_var);
        half_values.push(lo_value);
        half_values.push(hi_value);
    }
    let mut tr = Poseidon2TranscriptCircuit::new(cs.namespace(|| format!("{label}_tr")), app_domain)?;
    tr.append_u64_halves(
        cs.namespace(|| format!("{label}_append")),
        label_bytes,
        &halves,
        &half_values,
        word_vars.len(),
    )?;
    tr.digest32(cs.namespace(|| format!("{label}_digest")))
}

fn alloc_constant<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))?;
    cs.enforce(
        || format!("{label}_constant"),
        |lc| lc + out.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (value, CS::one()),
    );
    Ok(out)
}
