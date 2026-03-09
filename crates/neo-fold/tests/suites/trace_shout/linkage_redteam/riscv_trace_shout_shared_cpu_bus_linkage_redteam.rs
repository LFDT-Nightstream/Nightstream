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
use p3_field::PrimeCharacteristicRing;

use crate::suite::{default_mixers, setup_ajtai_committer};

fn build_shout_only_bus_z(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lanes: usize,
    lane_data: &[neo_memory::riscv::trace::ShoutLaneOverTime],
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if ell_addr != 3 {
        return Err(format!(
            "build_shout_only_bus_z: expected ell_addr=3 for packed ADD (got ell_addr={ell_addr})"
        ));
    }
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_shout_only_bus_z: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.len() != lanes {
        return Err(format!(
            "build_shout_only_bus_z: lane_data.len()={} != lanes={}",
            lane_data.len(),
            lanes
        ));
    }

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m,
        m_in,
        t,
        core::iter::once((ell_addr, lanes)),
        core::iter::empty::<(usize, usize)>(),
    )?;
    if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
        return Err("build_shout_only_bus_z: expected 1 shout instance and 0 twist instances".into());
    }

    let mut z = vec![F::ZERO; m];
    z[..m_in].copy_from_slice(x_prefix);

    let shout = &bus.shout_cols[0];
    for (lane_idx, cols) in shout.lanes.iter().enumerate() {
        let lane = &lane_data[lane_idx];
        if lane.has_lookup.len() != t {
            return Err("build_shout_only_bus_z: lane length mismatch".into());
        }
        for j in 0..t {
            let has = lane.has_lookup[j];
            z[bus.bus_cell(cols.has_lookup, j)] = if has { F::ONE } else { F::ZERO };
            z[bus.bus_cell(cols.primary_val(), j)] = if has { F::from_u64(lane.value[j]) } else { F::ZERO };

            // Packed-key layout: [lhs_u32, rhs_u32, carry_bit]
            let mut packed = [F::ZERO; 3];
            if has {
                let (lhs_u64, rhs_u64) = uninterleave_bits(lane.key[j] as u128);
                let lhs = (lhs_u64 as u32) as u64;
                let rhs = (rhs_u64 as u32) as u64;
                let carry = (lhs.wrapping_add(rhs) >> 32) & 1;
                packed[0] = F::from_u64(lhs);
                packed[1] = F::from_u64(rhs);
                packed[2] = F::from_u64(carry);
            }
            for (idx, col_id) in cols.addr_bits.clone().enumerate() {
                z[bus.bus_cell(col_id, j)] = packed[idx];
            }
        }
    }

    Ok(z)
}

fn run_shared_cpu_bus_shout_linkage_redteam<FSelect>(select_tamper_col: FSelect)
where
    FSelect: Fn(&Rv32TraceCcsLayout) -> usize,
{
    // Program:
    // - ADDI x1, x0, 1
    // - ADDI x2, x1, 2
    // - HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Find an active row with a lookup, then tamper with `shout_val` in the trace witness.
    let mut tamper_row: Option<usize> = None;
    for row in 0..layout.t {
        let idx = layout.cell(layout.trace.shout_has_lookup, row);
        if w[idx - layout.m_in] == F::ONE {
            tamper_row = Some(row);
            break;
        }
    }
    let row = tamper_row.expect("expected at least one Shout lookup in the trace");
    let tamper_col = select_tamper_col(&layout);
    let tamper_idx = layout.cell(tamper_col, row);
    w[tamper_idx - layout.m_in] += F::ONE;

    // Params + committer.
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n.max(ccs.m)).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    // Main CPU trace witness commitment (committed to the tampered witness).
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

    // Shout instance: ADD table, 1 lane (honest sidecar witness).
    let t = exec.rows.len();
    let shout_table_ids = vec![RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add).0];
    let shout_lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids, /*xlen=*/ 32).expect("extract shout lanes");

    let add_lut_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: Vec::new(),
        k: 0,
        d: 3,
        n_side: 2,
        steps: t,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcodePacked {
            opcode: RiscvOpcode::Add,
            xlen: 32,
        }),
        table: Vec::new(),
    addr_group: None,
    selector_group: None,
    };
    let add_z = build_shout_only_bus_z(
        ccs.m,
        layout.m_in,
        t,
        /*ell_addr=*/ add_lut_inst.d * add_lut_inst.ell,
        /*lanes=*/ 1,
        &shout_lanes,
        &x,
    )
    .expect("ADD Shout z");
    let add_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &add_z);
    let add_c = l.commit(&add_Z);
    let add_lut_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: vec![add_c],
        ..add_lut_inst
    };
    let add_lut_wit = LutWitness { mats: vec![add_Z] };

    let steps_witness = vec![crate::common_setup::canonicalize_step_time_columns(StepWitnessBundle {
        mcs,
        lut_instances: vec![(add_lut_inst, add_lut_wit)],
        mem_instances: Vec::new(),
        time_columns: crate::common_setup::empty_time_columns(),
        _phantom: PhantomData,
    })];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    // In no-shared mode, shout linkage is validated during Route-A verification.
    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-redteam");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("tampered trace shout_val should still admit a proof object before verify checks");

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-redteam");
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
    .expect_err("tampered trace shout linkage must fail during no-shared verification");
}

#[test]
fn riscv_trace_shared_cpu_bus_shout_val_linkage_redteam() {
    run_shared_cpu_bus_shout_linkage_redteam(|layout| layout.trace.shout_val);
}

#[test]
fn riscv_trace_shared_cpu_bus_shout_lhs_linkage_redteam() {
    run_shared_cpu_bus_shout_linkage_redteam(|layout| layout.trace.shout_lhs);
}
