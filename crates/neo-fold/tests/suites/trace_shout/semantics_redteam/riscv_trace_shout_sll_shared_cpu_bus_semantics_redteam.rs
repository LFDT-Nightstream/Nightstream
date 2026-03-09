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
    if ell_addr != 38 {
        return Err(format!(
            "build_shout_only_bus_z: expected ell_addr=38 for packed SLL (got ell_addr={ell_addr})"
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
        let lane = lane_data
            .get(lane_idx)
            .ok_or_else(|| format!("missing lane_data[{lane_idx}]"))?;
        for j in 0..t {
            let has_lookup = lane.has_lookup[j];
            z[bus.bus_cell(cols.has_lookup, j)] = if has_lookup { F::ONE } else { F::ZERO };

            if has_lookup {
                z[bus.bus_cell(cols.primary_val(), j)] = F::from_u64(lane.value[j]);
            }

            if has_lookup {
                let (lhs_u64, rhs_u64) = uninterleave_bits(lane.key[j] as u128);
                let lhs = lhs_u64 as u32;
                let shamt = (rhs_u64 as u32) & 0x1F;
                let wide = (lhs as u64) << shamt;
                let carry = (wide >> 32) as u32;

                // Packed-key layout (ell_addr=38):
                // [lhs_u32, shamt_bits[0..5], carry_bits[0..32]].
                let mut packed = vec![F::ZERO; ell_addr];
                packed[0] = F::from_u64(lhs as u64);
                for bit in 0..5 {
                    packed[1 + bit] = if ((shamt >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
                }
                for bit in 0..32 {
                    packed[6 + bit] = if ((carry >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
                }

                for (idx, col_id) in cols.addr_bits.clone().enumerate() {
                    z[bus.bus_cell(col_id, j)] = packed[idx];
                }
            }
        }
    }

    Ok(z)
}

#[test]
fn riscv_trace_wiring_ccs_shared_cpu_bus_shout_sll_semantics_redteam() {
    // Program:
    // - LUI x1, 1        (x1 = 4096)
    // - SLLI x2, x1, 3   (x2 = 32768)
    // - HALT
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 1 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 2,
            rs1: 1,
            imm: 3,
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

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
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

    // Shout instance: SLL table, 1 lane (tamper one carry bit).
    let t = exec.rows.len();
    let shout_table_ids = vec![RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Sll).0];
    let shout_lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids, /*xlen=*/ 32).expect("extract shout lanes");

    let sll_lut_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: Vec::new(),
        k: 0,
        d: 38,
        n_side: 2,
        steps: t,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcodePacked {
            opcode: RiscvOpcode::Sll,
            xlen: 32,
        }),
        table: Vec::new(),
    addr_group: None,
    selector_group: None,
    };

    let mut sll_z = build_shout_only_bus_z(
        ccs.m,
        layout.m_in,
        t,
        /*ell_addr=*/ sll_lut_inst.d * sll_lut_inst.ell,
        /*lanes=*/ 1,
        &shout_lanes,
        &x,
    )
    .expect("SLL Shout z");

    // Flip carry bit 0 on any active lookup row.
    let lane0 = &shout_lanes[0];
    let j = lane0
        .has_lookup
        .iter()
        .position(|&b| b)
        .expect("expected at least one SLL lookup");

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        ccs.m,
        layout.m_in,
        t,
        core::iter::once((/*ell_addr=*/ 38usize, /*lanes=*/ 1usize)),
        core::iter::empty::<(usize, usize)>(),
    )
    .expect("bus layout");
    let cols = &bus.shout_cols[0].lanes[0];
    let carry_bit0_col_id = cols
        .addr_bits
        .clone()
        .nth(6)
        .expect("expected addr_bits[6] for carry bit 0");
    let cell = bus.bus_cell(carry_bit0_col_id, j);
    sll_z[cell] = if sll_z[cell] == F::ONE { F::ZERO } else { F::ONE };

    let sll_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &sll_z);
    let sll_c = l.commit(&sll_Z);
    let sll_lut_inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: vec![sll_c],
        ..sll_lut_inst
    };
    let sll_lut_wit = LutWitness { mats: vec![sll_Z] };

    let steps_witness = vec![crate::common_setup::canonicalize_step_time_columns(StepWitnessBundle {
        mcs,
        lut_instances: vec![(sll_lut_inst, sll_lut_wit)],
        mem_instances: Vec::new(),
        time_columns: crate::common_setup::empty_time_columns(),
        _phantom: PhantomData,
    })];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    // The prover may either:
    // - reject because the tampered witness no longer satisfies the protocol invariants, or
    // - emit a proof that fails verification.
    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-sll-semantics-redteam");
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
        let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-sll-semantics-redteam");
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
        .expect_err("tampered packed SLL carry bit must be caught by Route-A time constraints");
    }
}
