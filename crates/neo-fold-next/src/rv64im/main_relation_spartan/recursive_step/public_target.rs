//! Owns the public IO layout for the RV64IM main-recursion Spartan surfaces.

use std::ops::Range;

use p3_field::PrimeField64;
use serde::{Deserialize, Serialize};

use crate::rv64im::construction2::Rv64imMainRecursionConstruction2PublicBoundary;
use crate::rv64im::ivc_snark::SpartanF;
use crate::rv64im::main_recursion::Rv64imEncodedPublicInput;
use crate::rv64im::main_relation_spartan::{digest32_as_spartan_fields, Rv64imMainRecursionStepSpartanStatement};

use super::u64_halves_as_spartan_fields;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepSpartanPublishedTarget {
    pub vk_fs_digest: [u8; 32],
    pub chunk_count: u64,
    pub z_0: [u8; 32],
    pub z_i: [u8; 32],
    pub pc: u64,
    pub x_out: Rv64imEncodedPublicInput,
    pub construction2_u_i: Rv64imMainRecursionConstruction2PublicBoundary,
    pub folded_accumulator_out_digest: [u8; 32],
    pub bridge_handoff_digest: [u8; 32],
    pub terminal_verified_step_statement_digest: [u8; 32],
}

impl Rv64imMainRecursionStepSpartanPublishedTarget {
    pub fn terminal_r2_public_value_range(&self) -> Range<usize> {
        Self::terminal_r2_public_value_range_static()
    }

    pub fn terminal_r2_public_value_range_static() -> Range<usize> {
        let start = 4 + 2 + 4 + 4 + 2 + 4;
        start..start + 256
    }

    pub fn terminal_f_prime_r2_public_values(&self) -> Vec<SpartanF> {
        terminal_f_prime_r2_public_values_from_parts(
            self.vk_fs_digest,
            self.chunk_count,
            self.z_0,
            self.z_i,
            self.pc,
            &self.x_out,
            self.folded_accumulator_out_digest,
            self.bridge_handoff_digest,
            self.terminal_verified_step_statement_digest,
        )
    }

    pub fn terminal_f_prime_r2_public_value_range(&self) -> Range<usize> {
        self.terminal_r2_public_value_range()
    }

    pub fn output_statement(&self) -> Rv64imMainRecursionStepSpartanStatement {
        Rv64imMainRecursionStepSpartanStatement {
            x_out: self.x_out.clone(),
            folded_accumulator_digest: self.folded_accumulator_out_digest,
        }
    }
}

pub(crate) fn terminal_f_prime_r2_public_values_from_parts(
    vk_fs_digest: [u8; 32],
    chunk_count: u64,
    z_0: [u8; 32],
    z_i: [u8; 32],
    pc: u64,
    x_out: &Rv64imEncodedPublicInput,
    folded_accumulator_out_digest: [u8; 32],
    bridge_handoff_digest: [u8; 32],
    terminal_verified_step_statement_digest: [u8; 32],
) -> Vec<SpartanF> {
    let mut values = Vec::with_capacity(32 + 256);
    values.extend(digest32_as_spartan_fields(vk_fs_digest));
    values.extend(u64_halves_as_spartan_fields(chunk_count));
    values.extend(digest32_as_spartan_fields(z_0));
    values.extend(digest32_as_spartan_fields(z_i));
    values.extend(u64_halves_as_spartan_fields(pc));
    values.extend(digest32_as_spartan_fields(x_out.bytes()));
    values.extend(
        x_out
            .field_image()
            .into_iter()
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    );
    values.extend(digest32_as_spartan_fields(folded_accumulator_out_digest));
    values.extend(digest32_as_spartan_fields(bridge_handoff_digest));
    values.extend(digest32_as_spartan_fields(terminal_verified_step_statement_digest));
    values
}
