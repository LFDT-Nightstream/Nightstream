//! Owns the small published-statement helper shared by the main proof surface.

use crate::rv64im::chunk_fold_step::Rv64imChunkFoldCarry;
use crate::rv64im::final_relation::{rv64im_chunk_fold_carry_recursive_accumulator_digest, Rv64imRecursiveAccumulator};
use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs, Rv64imEncodedPublicInput, Rv64imVerifierKeyFs,
};
use crate::rv64im::SimpleKernelError;

pub(crate) fn build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(
    vk_fs: &Rv64imVerifierKeyFs,
    chunk_count: u64,
    accumulator_final: &Rv64imRecursiveAccumulator,
) -> Result<Rv64imEncodedPublicInput, SimpleKernelError> {
    let folded_accumulator_digest =
        rv64im_chunk_fold_carry_recursive_accumulator_digest(&Rv64imChunkFoldCarry::from_main(
            crate::proof::Carry {
                claims: accumulator_final.final_main_claims.clone(),
                witnesses: Vec::new(),
            },
            accumulator_final.terminal_handle,
        ));
    Ok(build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
        vk_fs,
        chunk_count,
        folded_accumulator_digest,
        accumulator_final.terminal_handle.0,
    )
    .x_out)
}
