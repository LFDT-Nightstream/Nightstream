//! Owns the small published-statement helper shared by the main proof surface.

use crate::rv32im::chunk_fold_step::Rv32imChunkFoldCarry;
use crate::rv32im::final_relation::{rv32im_chunk_fold_carry_recursive_accumulator_digest, Rv32imRecursiveAccumulator};
use crate::rv32im::main_recursion::{
    build_rv32im_main_recursion_backend_statement_from_parts_with_vk_fs, Rv32imEncodedPublicInput, Rv32imVerifierKeyFs,
};
use crate::rv32im::SimpleKernelError;

pub(crate) fn build_rv32im_main_recursion_x_last_from_accumulator_with_vk_fs(
    vk_fs: &Rv32imVerifierKeyFs,
    chunk_count: u64,
    accumulator_final: &Rv32imRecursiveAccumulator,
) -> Result<Rv32imEncodedPublicInput, SimpleKernelError> {
    let folded_accumulator_digest =
        rv32im_chunk_fold_carry_recursive_accumulator_digest(&Rv32imChunkFoldCarry::from_main(
            crate::proof::Carry {
                claims: accumulator_final.final_main_claims.clone(),
                witnesses: Vec::new(),
            },
            accumulator_final.terminal_handle,
        ));
    Ok(build_rv32im_main_recursion_backend_statement_from_parts_with_vk_fs(
        vk_fs,
        chunk_count,
        folded_accumulator_digest,
        accumulator_final.terminal_handle.0,
    )
    .x_out)
}
