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
    decode_program, encode_program, uninterleave_bits, BranchCondition, RiscvCpu, RiscvInstruction, RiscvMemory,
    RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::trace::extract_shout_lanes_over_time;
use neo_memory::witness::{LutInstance, LutTableSpec, LutWitness, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;

use crate::suite::{default_mixers, setup_ajtai_committer, widen_ccs_cols_for_test};

fn build_shout_only_bus_z(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lanes: usize,
    lane_data: &[neo_memory::riscv::trace::ShoutLaneOverTime],
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if ell_addr != 35 {
        return Err(format!(
            "build_shout_only_bus_z: expected ell_addr=35 for packed EQ (got ell_addr={ell_addr})"
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

            // Packed-key layout (ell_addr=35):
            // [lhs_u32, rhs_u32, borrow_bit, diff_bits[0..32]].
            let mut packed = [F::ZERO; 35];
            if has {
                let (lhs_u64, rhs_u64) = uninterleave_bits(lane.key[j] as u128);
                let lhs = lhs_u64 as u32;
                let rhs = rhs_u64 as u32;
                let borrow = if lhs < rhs { 1u32 } else { 0u32 };
                let diff = lhs.wrapping_sub(rhs);

                packed[0] = F::from_u64(lhs as u64);
                packed[1] = F::from_u64(rhs as u64);
                packed[2] = if borrow == 1 { F::ONE } else { F::ZERO };
                for bit in 0..32usize {
                    packed[3 + bit] = if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
                }
            }
            for (idx, col_id) in cols.addr_bits.clone().enumerate() {
                z[bus.bus_cell(col_id, j)] = packed[idx];
            }
        }
    }

    Ok(z)
}

#[test]
fn riscv_trace_shared_cpu_bus_shout_eq_semantics_redteam() {
    // Program (use BEQ to generate an `Eq` Shout event; there is no dedicated `Eq` ALU instruction encoding):
    // - LUI x1, 0        (x1 = 0)
    // - LUI x2, 1        (x2 = 4096)
    // - BEQ x1, x2, +8   (not taken; EQ=0)  <-- diff != 0, inv is constrained
    // - HALT
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 0 },
        RiscvInstruction::Lui { rd: 2, imm: 1 },
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 1,
            rs2: 2,
            imm: 8,
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
    let mut ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");
    let min_m = layout
        .m_in
        .checked_add((/*bus_cols=*/ 35usize + 2usize).checked_mul(exec.rows.len()).expect("bus cols * steps"))
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

    // Shout instance: EQ table, 1 lane (tamper the packed borrow bit witness).
    let t = exec.rows.len();
    let shout_table_ids = vec![RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Eq).0];
    let shout_lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids, /*xlen=*/ 32).expect("extract shout lanes");

    let eq_lut_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: Vec::new(),
        k: 0,
        d: 35,
        n_side: 2,
        steps: t,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcodePacked {
            opcode: RiscvOpcode::Eq,
            xlen: 32,
        }),
        table: Vec::new(),
    addr_group: None,
    selector_group: None,
    };

    let mut eq_z = build_shout_only_bus_z(
        ccs.m,
        layout.m_in,
        t,
        /*ell_addr=*/ eq_lut_inst.d * eq_lut_inst.ell,
        /*lanes=*/ 1,
        &shout_lanes,
        &x,
    )
    .expect("EQ Shout z");

    // Find a lookup row with lhs != rhs, then flip the borrow bit (addr_bits[2]).
    let lane0 = &shout_lanes[0];
    let mut tamper_j: Option<usize> = None;
    for j in 0..t {
        if !lane0.has_lookup[j] {
            continue;
        }
        let (lhs_u64, rhs_u64) = uninterleave_bits(lane0.key[j] as u128);
        if (lhs_u64 as u32) != (rhs_u64 as u32) {
            tamper_j = Some(j);
            break;
        }
    }
    let j = tamper_j.expect("expected at least one EQ lookup with lhs != rhs");

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        ccs.m,
        layout.m_in,
        t,
        core::iter::once((/*ell_addr=*/ 35usize, /*lanes=*/ 1usize)),
        core::iter::empty::<(usize, usize)>(),
    )
    .expect("bus layout");
    let cols = &bus.shout_cols[0].lanes[0];
    let borrow_col_id = cols.addr_bits.clone().nth(2).expect("borrow col id");
    let borrow_cell = bus.bus_cell(borrow_col_id, j);
    eq_z[borrow_cell] = if eq_z[borrow_cell] == F::ZERO { F::ONE } else { F::ZERO };

    let eq_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &eq_z);
    let eq_c = l.commit(&eq_Z);
    let eq_lut_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: vec![eq_c],
        ..eq_lut_inst
    };
    let eq_lut_wit = LutWitness { mats: vec![eq_Z] };

    let steps_witness = vec![crate::common_setup::canonicalize_step_time_columns(StepWitnessBundle {
        mcs,
        lut_instances: vec![(eq_lut_inst, eq_lut_wit)],
        mem_instances: Vec::new(),
        time_columns: crate::common_setup::empty_time_columns(),
        _phantom: PhantomData,
    })];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    // The prover may either:
    // - reject because the witness no longer satisfies the protocol invariants, or
    // - emit a proof that fails verification.
    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-eq-semantics-redteam");
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
        let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-eq-semantics-redteam");
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
        .expect_err("tampered packed EQ borrow witness must be caught by Route-A time constraints");
    }
}
