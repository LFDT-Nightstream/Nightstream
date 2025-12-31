//! Convenience wrappers for verifying RISC-V shard proofs safely.
//!
//! These helpers are intentionally small: they standardize the step-linking configuration
//! for RV32 B1 chunked execution so callers don't accidentally verify a "bag of chunks".

#![allow(non_snake_case)]

use std::collections::HashMap;

use crate::pi_ccs::FoldingMode;
use crate::shard::{
    fold_shard_verify_with_output_binding_and_step_linking, fold_shard_verify_with_step_linking, CommitMixers,
    ShardFoldOutputs, ShardProof, StepLinkingConfig,
};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{F, K};
use neo_memory::mem_init_from_initial_mem;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{rv32_b1_step_linking_pairs, Rv32B1Layout};
use neo_memory::witness::StepInstanceBundle;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

pub fn rv32_b1_step_linking_config(layout: &Rv32B1Layout) -> StepLinkingConfig {
    StepLinkingConfig::new(rv32_b1_step_linking_pairs(layout))
}

/// Enforce that the *public statement* initial memory matches chunk 0's `MemInstance.init`.
///
/// This lets later chunk `init` snapshots remain proof-internal rollover data (Twist needs them),
/// while keeping the user-facing statement independent of `chunk_size`.
pub fn rv32_b1_enforce_chunk0_mem_init_matches_statement<Cmt2, K2>(
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    statement_initial_mem: &HashMap<(u32, u64), F>,
    steps: &[StepInstanceBundle<Cmt2, F, K2>],
) -> Result<(), PiCcsError> {
    let chunk0 = steps
        .first()
        .ok_or_else(|| PiCcsError::InvalidInput("no steps provided".into()))?;

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    if chunk0.mem_insts.len() != mem_ids.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "mem instance count mismatch: chunk0 has {}, but mem_layouts has {}",
            chunk0.mem_insts.len(),
            mem_ids.len()
        )));
    }

    for (idx, mem_id) in mem_ids.into_iter().enumerate() {
        let layout = mem_layouts.get(&mem_id).ok_or_else(|| {
            PiCcsError::InvalidInput(format!("missing PlainMemLayout for mem_id={mem_id}"))
        })?;
        let expected = mem_init_from_initial_mem(mem_id, layout.k, statement_initial_mem)?;
        let got = &chunk0.mem_insts[idx].init;
        if *got != expected {
            return Err(PiCcsError::InvalidInput(format!(
                "chunk0 MemInstance.init mismatch for mem_id={mem_id}"
            )));
        }
    }

    Ok(())
}

pub fn fold_shard_verify_rv32_b1<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    layout: &Rv32B1Layout,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let step_linking = rv32_b1_step_linking_config(layout);
    fold_shard_verify_with_step_linking(mode, tr, params, s_me, steps, acc_init, proof, mixers, &step_linking)
}

pub fn fold_shard_verify_rv32_b1_with_statement_mem_init<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    statement_initial_mem: &HashMap<(u32, u64), F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    layout: &Rv32B1Layout,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    rv32_b1_enforce_chunk0_mem_init_matches_statement(mem_layouts, statement_initial_mem, steps)?;
    fold_shard_verify_rv32_b1(mode, tr, params, s_me, steps, acc_init, proof, mixers, layout)
}

pub fn fold_shard_verify_rv32_b1_with_output_binding<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    layout: &Rv32B1Layout,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let step_linking = rv32_b1_step_linking_config(layout);
    fold_shard_verify_with_output_binding_and_step_linking(
        mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg, &step_linking,
    )
}
