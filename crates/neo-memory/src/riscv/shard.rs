use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

use crate::riscv::ccs::{rv32_b1_step_linking_pairs, Rv32B1Layout};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32BoundaryState {
    pub pc0: F,
    pub regs0: [F; 32],
    pub pc_final: F,
    pub regs_final: [F; 32],
    pub halted_in: F,
    pub halted_out: F,
}

pub fn extract_boundary_state(layout: &Rv32B1Layout, x: &[F]) -> Result<Rv32BoundaryState, String> {
    let required = [
        layout.pc0,
        layout.regs0_start + 31,
        layout.pc_final,
        layout.regs_final_start + 31,
        layout.halted_in,
        layout.halted_out,
    ];
    let max = required.into_iter().max().unwrap_or(0);
    if max >= x.len() {
        return Err(format!(
            "public x too short for RV32 boundary extraction: need idx {max} but x.len()={}",
            x.len()
        ));
    }

    let mut regs0 = [F::ZERO; 32];
    let mut regs_final = [F::ZERO; 32];
    for r in 0..32 {
        regs0[r] = x[layout.regs0_start + r];
        regs_final[r] = x[layout.regs_final_start + r];
    }

    Ok(Rv32BoundaryState {
        pc0: x[layout.pc0],
        regs0,
        pc_final: x[layout.pc_final],
        regs_final,
        halted_in: x[layout.halted_in],
        halted_out: x[layout.halted_out],
    })
}

pub fn check_rv32_b1_chunk_chaining(layout: &Rv32B1Layout, chunk_publics: &[&[F]]) -> Result<(), String> {
    if chunk_publics.len() <= 1 {
        return Ok(());
    }

    let pairs = rv32_b1_step_linking_pairs(layout);
    for (i, (a, b)) in chunk_publics.iter().zip(chunk_publics.iter().skip(1)).enumerate() {
        for &(out_idx, in_idx) in &pairs {
            let out = a.get(out_idx).copied().ok_or_else(|| {
                format!(
                    "chunk {i} public x too short for linking: need idx {out_idx} but x.len()={}",
                    a.len()
                )
            })?;
            let inn = b.get(in_idx).copied().ok_or_else(|| {
                format!(
                    "chunk {} public x too short for linking: need idx {in_idx} but x.len()={}",
                    i + 1,
                    b.len()
                )
            })?;
            if out != inn {
                return Err(format!(
                    "RV32 chunk linking mismatch at boundary {i}: x_i[{out_idx}] != x_{}[{in_idx}]",
                    i + 1
                ));
            }
        }
    }
    Ok(())
}
