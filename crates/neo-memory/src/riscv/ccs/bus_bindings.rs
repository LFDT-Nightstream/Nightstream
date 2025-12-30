use std::collections::HashMap;

use p3_goldilocks::Goldilocks as F;

use crate::cpu::constraints::{CpuConstraintBuilder, ShoutCpuBinding, TwistCpuBinding};
use crate::cpu::r1cs_adapter::SharedCpuBusConfig;
use crate::plain::PlainMemLayout;
use crate::riscv::lookups::{RAM_ID, PROG_ID};

use super::config::{derive_mem_ids_and_ell_addrs, derive_shout_ids_and_ell_addrs};
use super::constants::{
    ADD_TABLE_ID, AND_TABLE_ID, DIV_TABLE_ID, DIVU_TABLE_ID, EQ_TABLE_ID, MULH_TABLE_ID, MULHSU_TABLE_ID,
    MULHU_TABLE_ID, MUL_TABLE_ID, NEQ_TABLE_ID, OR_TABLE_ID, REMU_TABLE_ID, REM_TABLE_ID, SLL_TABLE_ID, SLT_TABLE_ID,
    SLTU_TABLE_ID, SRA_TABLE_ID, SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
};
use super::Rv32B1Layout;

fn shout_cpu_binding(layout: &Rv32B1Layout, table_id: u32) -> ShoutCpuBinding {
    match table_id {
        AND_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.and_has_lookup,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        XOR_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.xor_has_lookup,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        OR_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.or_has_lookup,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        ADD_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.add_has_lookup,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        SUB_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_sub,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        SLT_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.slt_has_lookup,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        SLTU_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.sltu_has_lookup,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        SLL_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.sll_has_lookup,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        SRL_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.srl_has_lookup,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        SRA_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.sra_has_lookup,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        EQ_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_beq,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        NEQ_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_bne,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        MUL_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_mul,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        MULH_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_mulh,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        MULHU_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_mulhu,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        MULHSU_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_mulhsu,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        DIV_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_div,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        DIVU_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_divu,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        REM_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_rem,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        REMU_TABLE_ID => ShoutCpuBinding {
            has_lookup: layout.is_remu,
            addr: layout.lookup_key,
            val: layout.alu_out,
        },
        _ => {
            // Bind unused tables to fixed-zero CPU columns so they are provably inactive.
            let zero = layout.reg_in(0, 0);
            ShoutCpuBinding {
                has_lookup: zero,
                addr: zero,
                val: zero,
            }
        }
    }
}

fn twist_cpu_binding(layout: &Rv32B1Layout, mem_id: u32) -> TwistCpuBinding {
    if mem_id == RAM_ID.0 {
        TwistCpuBinding {
            has_read: layout.ram_has_read,
            has_write: layout.ram_has_write,
            read_addr: layout.eff_addr,
            write_addr: layout.eff_addr,
            rv: layout.mem_rv,
            wv: layout.ram_wv,
            inc: None,
        }
    } else if mem_id == PROG_ID.0 {
        let zero = layout.reg_in(0, 0);
        TwistCpuBinding {
            has_read: layout.is_active,
            has_write: zero,
            read_addr: layout.pc_in,
            write_addr: zero,
            rv: layout.instr_word,
            wv: zero,
            inc: None,
        }
    } else {
        // Disable any additional Twist instances by binding to fixed-zero CPU columns.
        let zero = layout.reg_in(0, 0);
        TwistCpuBinding {
            has_read: zero,
            has_write: zero,
            read_addr: zero,
            write_addr: zero,
            rv: zero,
            wv: zero,
            inc: None,
        }
    }
}

pub(super) fn injected_bus_constraints_len(layout: &Rv32B1Layout, table_ids: &[u32], mem_ids: &[u32]) -> usize {
    let shout_cpu: Vec<ShoutCpuBinding> = table_ids.iter().map(|&id| shout_cpu_binding(layout, id)).collect();
    let twist_cpu: Vec<TwistCpuBinding> = mem_ids.iter().map(|&id| twist_cpu_binding(layout, id)).collect();

    let mut builder = CpuConstraintBuilder::<F>::new(layout.m, layout.m, layout.const_one);
    for (i, cpu) in shout_cpu.iter().enumerate() {
        builder.add_shout_instance_bound(&layout.bus, &layout.bus.shout_cols[i], cpu);
    }
    for (i, cpu) in twist_cpu.iter().enumerate() {
        builder.add_twist_instance_bound(&layout.bus, &layout.bus.twist_cols[i], cpu);
    }
    builder.constraints().len()
}

/// Shared CPU-bus bindings for the RV32 B1 step circuit.
///
/// This config:
/// - binds `PROG_ID` reads to `pc_in` / `instr_word`, forces no ROM writes,
/// - binds `RAM_ID` reads/writes to `eff_addr` / `mem_rv` / `ram_wv` (with selectors derived from instruction flags),
/// - binds RV32IM Shout opcode tables (ids 0..=19) to `lookup_key` / `alu_out`.
pub fn rv32_b1_shared_cpu_bus_config(
    layout: &Rv32B1Layout,
    shout_table_ids: &[u32],
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
) -> Result<SharedCpuBusConfig<F>, String> {
    let (table_ids, _ell_addrs) = derive_shout_ids_and_ell_addrs(shout_table_ids)?;

    let mut shout_cpu = HashMap::new();
    for table_id in table_ids {
        shout_cpu.insert(table_id, shout_cpu_binding(layout, table_id));
    }

    let (mem_ids, _ell_addrs) = derive_mem_ids_and_ell_addrs(&mem_layouts)?;
    let mut twist_cpu = HashMap::new();
    for mem_id in mem_ids {
        twist_cpu.insert(mem_id, twist_cpu_binding(layout, mem_id));
    }

    Ok(SharedCpuBusConfig {
        mem_layouts,
        initial_mem,
        const_one_col: layout.const_one,
        shout_cpu,
        twist_cpu,
    })
}
