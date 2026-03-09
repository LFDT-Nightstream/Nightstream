#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::relations::{CcsClaim, CcsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::F;
use neo_memory::cpu::build_bus_layout_for_instances_with_shout_and_twist_lanes;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout};
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, uninterleave_bits, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode,
    RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::trace::extract_shout_lanes_over_time;
use neo_memory::witness::{LutInstance, LutTableSpec, LutWitness, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use neo_vm_trace::trace_program;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::suite::{default_mixers, setup_ajtai_committer, widen_ccs_cols_for_test};

fn divu(lhs: u32, rhs: u32) -> u32 {
    if rhs == 0 {
        u32::MAX
    } else {
        lhs / rhs
    }
}

fn remu(lhs: u32, rhs: u32) -> u32 {
    if rhs == 0 {
        lhs
    } else {
        lhs % rhs
    }
}

fn build_shout_only_bus_z_packed_divu(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lane_data: &neo_memory::riscv::trace::ShoutLaneOverTime,
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if ell_addr != 38 {
        return Err(format!(
            "build_shout_only_bus_z_packed_divu: expected ell_addr=38 (got ell_addr={ell_addr})"
        ));
    }
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_shout_only_bus_z_packed_divu: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.has_lookup.len() != t {
        return Err("build_shout_only_bus_z_packed_divu: lane length mismatch".into());
    }

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m,
        m_in,
        t,
        core::iter::once((ell_addr, /*lanes=*/ 1usize)),
        core::iter::empty::<(usize, usize)>(),
    )?;
    if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
        return Err("build_shout_only_bus_z_packed_divu: expected 1 shout instance and 0 twist instances".into());
    }

    let mut z = vec![F::ZERO; m];
    z[..m_in].copy_from_slice(x_prefix);

    let cols = &bus.shout_cols[0].lanes[0];
    for j in 0..t {
        let has = lane_data.has_lookup[j];
        z[bus.bus_cell(cols.has_lookup, j)] = if has { F::ONE } else { F::ZERO };
        z[bus.bus_cell(cols.primary_val(), j)] = if has { F::from_u64(lane_data.value[j]) } else { F::ZERO };

        let mut packed = [F::ZERO; 38];
        if has {
            let (lhs_u64, rhs_u64) = uninterleave_bits(lane_data.key[j] as u128);
            let lhs = lhs_u64 as u32;
            let rhs = rhs_u64 as u32;
            let quot = lane_data.value[j] as u32;
            let expected_quot = divu(lhs, rhs);
            if quot != expected_quot {
                return Err(format!(
                    "build_shout_only_bus_z_packed_divu: lane.value mismatch at j={j} (got {quot:#x}, expected {expected_quot:#x})"
                ));
            }

            let rhs_f = F::from_u64(rhs as u64);
            let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
            let rhs_is_zero = if rhs == 0 { 1u32 } else { 0u32 };

            let rem = if rhs == 0 {
                0u32
            } else {
                ((lhs as u64) % (rhs as u64)) as u32
            };
            let diff = rem.wrapping_sub(rhs);

            packed[0] = F::from_u64(lhs as u64);
            packed[1] = F::from_u64(rhs as u64);
            packed[2] = F::from_u64(rem as u64);
            packed[3] = rhs_inv;
            packed[4] = if rhs_is_zero == 1 { F::ONE } else { F::ZERO };
            packed[5] = F::from_u64(diff as u64);
            for bit in 0..32usize {
                packed[6 + bit] = if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
            }
        }

        for (idx, col_id) in cols.addr_bits.clone().enumerate() {
            z[bus.bus_cell(col_id, j)] = packed[idx];
        }
    }

    Ok(z)
}

fn build_shout_only_bus_z_packed_remu(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lane_data: &neo_memory::riscv::trace::ShoutLaneOverTime,
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if ell_addr != 38 {
        return Err(format!(
            "build_shout_only_bus_z_packed_remu: expected ell_addr=38 (got ell_addr={ell_addr})"
        ));
    }
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_shout_only_bus_z_packed_remu: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.has_lookup.len() != t {
        return Err("build_shout_only_bus_z_packed_remu: lane length mismatch".into());
    }

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m,
        m_in,
        t,
        core::iter::once((ell_addr, /*lanes=*/ 1usize)),
        core::iter::empty::<(usize, usize)>(),
    )?;
    if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
        return Err("build_shout_only_bus_z_packed_remu: expected 1 shout instance and 0 twist instances".into());
    }

    let mut z = vec![F::ZERO; m];
    z[..m_in].copy_from_slice(x_prefix);

    let cols = &bus.shout_cols[0].lanes[0];
    for j in 0..t {
        let has = lane_data.has_lookup[j];
        z[bus.bus_cell(cols.has_lookup, j)] = if has { F::ONE } else { F::ZERO };
        z[bus.bus_cell(cols.primary_val(), j)] = if has { F::from_u64(lane_data.value[j]) } else { F::ZERO };

        let mut packed = [F::ZERO; 38];
        if has {
            let (lhs_u64, rhs_u64) = uninterleave_bits(lane_data.key[j] as u128);
            let lhs = lhs_u64 as u32;
            let rhs = rhs_u64 as u32;
            let rem = lane_data.value[j] as u32;
            let expected_rem = remu(lhs, rhs);
            if rem != expected_rem {
                return Err(format!(
                    "build_shout_only_bus_z_packed_remu: lane.value mismatch at j={j} (got {rem:#x}, expected {expected_rem:#x})"
                ));
            }

            let rhs_f = F::from_u64(rhs as u64);
            let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
            let rhs_is_zero = if rhs == 0 { 1u32 } else { 0u32 };

            let quot = if rhs == 0 {
                0u32
            } else {
                (lhs as u64 / rhs as u64) as u32
            };
            let diff = rem.wrapping_sub(rhs);

            packed[0] = F::from_u64(lhs as u64);
            packed[1] = F::from_u64(rhs as u64);
            packed[2] = F::from_u64(quot as u64);
            packed[3] = rhs_inv;
            packed[4] = if rhs_is_zero == 1 { F::ONE } else { F::ZERO };
            packed[5] = F::from_u64(diff as u64);
            for bit in 0..32usize {
                packed[6 + bit] = if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
            }
        }

        for (idx, col_id) in cols.addr_bits.clone().enumerate() {
            z[bus.bus_cell(col_id, j)] = packed[idx];
        }
    }

    Ok(z)
}

#[test]
fn riscv_trace_wiring_ccs_shared_cpu_bus_shout_divu_remu_semantics_redteam() {
    // Same program as the e2e test; tamper DIVU diff bit 0.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 91,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 6,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");

    let mut exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 16).expect("from_trace_padded_pow2");
    let tables = RiscvShoutTables::new(32);
    let divu_id = tables.opcode_to_id(RiscvOpcode::Divu);
    let remu_id = tables.opcode_to_id(RiscvOpcode::Remu);
    for row in exec.rows.iter_mut() {
        if row.active {
            row.shout_events
                .retain(|ev| ev.shout_id == divu_id || ev.shout_id == remu_id);
        }
    }
    let divu_rows = exec
        .rows
        .iter()
        .filter(|row| {
            row.active
                && matches!(
                    row.decoded,
                    Some(RiscvInstruction::RAlu {
                        op: RiscvOpcode::Divu,
                        ..
                    })
                )
        })
        .count();
    let divu_shout_rows = exec
        .rows
        .iter()
        .filter(|row| {
            row.active
                && matches!(
                    row.decoded,
                    Some(RiscvInstruction::RAlu {
                        op: RiscvOpcode::Divu,
                        ..
                    })
                )
                && row.shout_events.iter().any(|ev| ev.shout_id == divu_id)
        })
        .count();
    let remu_rows = exec
        .rows
        .iter()
        .filter(|row| {
            row.active
                && matches!(
                    row.decoded,
                    Some(RiscvInstruction::RAlu {
                        op: RiscvOpcode::Remu,
                        ..
                    })
                )
        })
        .count();
    let remu_shout_rows = exec
        .rows
        .iter()
        .filter(|row| {
            row.active
                && matches!(
                    row.decoded,
                    Some(RiscvInstruction::RAlu {
                        op: RiscvOpcode::Remu,
                        ..
                    })
                )
                && row.shout_events.iter().any(|ev| ev.shout_id == remu_id)
        })
        .count();
    assert!(divu_rows > 0, "expected at least one DIVU row");
    assert!(remu_rows > 0, "expected at least one REMU row");
    assert!(
        divu_shout_rows > 0 && divu_shout_rows <= divu_rows,
        "native DIVU shout coverage mismatch (divu_rows={divu_rows}, divu_shout_rows={divu_shout_rows})"
    );
    assert!(
        remu_shout_rows > 0 && remu_shout_rows <= remu_rows,
        "native REMU shout coverage mismatch (remu_rows={remu_rows}, remu_shout_rows={remu_shout_rows})"
    );
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let mut ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");
    let min_m = layout
        .m_in
        .checked_add((/*bus_cols=*/ 38usize + 2usize).checked_mul(exec.rows.len()).expect("bus cols * steps"))
        .expect("m_in + bus region");
    widen_ccs_cols_for_test(&mut ccs, min_m);
    w.resize(ccs.m - layout.m_in, F::ZERO);

    // Params + committer.
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n.max(ccs.m)).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    // Main CPU trace witness commitment (honest).
    let z_cpu: Vec<F> = x.iter().copied().chain(w.iter().copied()).collect();
    let Z_cpu = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z_cpu);
    let c_cpu = l.commit(&Z_cpu);
    let mcs = (
        CcsClaim {
            c: c_cpu,
            x: x.clone(),
            m_in: layout.m_in,
        },
        CcsWitness { w, Z: Z_cpu },
    );

    // Shout instances: DIVU and REMU packed, 1 lane each (tamper DIVU diff_bit0).
    let t = exec.rows.len();
    let shout_table_ids = vec![divu_id.0, remu_id.0];
    let shout_lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids, /*xlen=*/ 32).expect("extract shout lanes");
    assert_eq!(shout_lanes.len(), 2);

    let divu_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: Vec::new(),
        k: 0,
        d: 38,
        n_side: 2,
        steps: t,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcodePacked {
            opcode: RiscvOpcode::Divu,
            xlen: 32,
        }),
        table: Vec::new(),
    addr_group: None,
    selector_group: None,
    };
    let remu_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: Vec::new(),
        k: 0,
        d: 38,
        n_side: 2,
        steps: t,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcodePacked {
            opcode: RiscvOpcode::Remu,
            xlen: 32,
        }),
        table: Vec::new(),
    addr_group: None,
    selector_group: None,
    };

    let mut divu_z =
        build_shout_only_bus_z_packed_divu(ccs.m, layout.m_in, t, divu_inst.d * divu_inst.ell, &shout_lanes[0], &x)
            .expect("DIVU packed z");
    let j = shout_lanes[0]
        .has_lookup
        .iter()
        .position(|&b| b)
        .expect("expected at least one DIVU lookup");
    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        ccs.m,
        layout.m_in,
        t,
        core::iter::once((/*ell_addr=*/ 38usize, /*lanes=*/ 1usize)),
        core::iter::empty::<(usize, usize)>(),
    )
    .expect("bus layout");
    let cols = &bus.shout_cols[0].lanes[0];
    let diff_bit0_col_id = cols
        .addr_bits
        .clone()
        .nth(6)
        .expect("expected addr_bits[6] for diff bit 0");
    let cell = bus.bus_cell(diff_bit0_col_id, j);
    divu_z[cell] = if divu_z[cell] == F::ONE { F::ZERO } else { F::ONE };

    let divu_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &divu_z);
    let divu_c = l.commit(&divu_Z);
    let divu_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: vec![divu_c],
        ..divu_inst
    };
    let divu_wit = LutWitness { mats: vec![divu_Z] };

    let remu_z =
        build_shout_only_bus_z_packed_remu(ccs.m, layout.m_in, t, remu_inst.d * remu_inst.ell, &shout_lanes[1], &x)
            .expect("REMU packed z");
    let remu_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &remu_z);
    let remu_c = l.commit(&remu_Z);
    let remu_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: vec![remu_c],
        ..remu_inst
    };
    let remu_wit = LutWitness { mats: vec![remu_Z] };

    let steps_witness = vec![crate::common_setup::canonicalize_step_time_columns(StepWitnessBundle {
        mcs,
        lut_instances: vec![(divu_inst, divu_wit), (remu_inst, remu_wit)],
        mem_instances: Vec::new(),
        time_columns: crate::common_setup::empty_time_columns(),
        _phantom: PhantomData,
    })];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-divu-remu-semantics-redteam");
    if let Ok(proof) = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    ) {
        let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-divu-remu-semantics-redteam");
        fold_shard_verify(
            FoldingMode::Optimized,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_instance,
            &[],
            &proof,
            mixers,
        )
        .expect_err("tampered packed DIVU diff bit must be caught by Route-A time constraints");
    }
}
