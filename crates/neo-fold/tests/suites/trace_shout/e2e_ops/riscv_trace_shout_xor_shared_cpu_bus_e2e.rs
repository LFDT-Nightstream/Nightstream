#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::relations::{CcsClaim, CcsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::F;
use neo_memory::cpu::build_bus_layout_for_instances_with_shout_and_twist_lanes;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout};
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::trace::extract_shout_lanes_over_time;
use neo_memory::witness::{LutInstance, LutTableSpec, LutWitness, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;

use crate::suite::{default_mixers, setup_ajtai_committer};

fn plan_paged_shout_addr(m: usize, m_in: usize, t: usize, ell_addr: usize, lanes: usize) -> Result<Vec<usize>, String> {
    if t == 0 {
        return Err("plan_paged_shout_addr: t must be >= 1".into());
    }
    if m_in > m {
        return Err(format!("plan_paged_shout_addr: m_in={m_in} > m={m}"));
    }
    if lanes == 0 {
        return Err("plan_paged_shout_addr: lanes must be >= 1".into());
    }
    if ell_addr == 0 {
        return Err("plan_paged_shout_addr: ell_addr must be >= 1".into());
    }

    // Match `neo_fold::memory_sidecar::shout_paging::plan_shout_addr_pages`.
    let avail = m - m_in;
    let max_bus_cols_total = avail / t;
    let per_lane_capacity = max_bus_cols_total / lanes;
    if per_lane_capacity < 3 {
        return Err(format!(
            "plan_paged_shout_addr: insufficient per-lane capacity (need >=3 cols per lane, have {per_lane_capacity})"
        ));
    }
    let max_addr_cols_per_page = per_lane_capacity - 2;

    let mut out = Vec::new();
    let mut remaining = ell_addr;
    while remaining > 0 {
        let take = remaining.min(max_addr_cols_per_page);
        out.push(take);
        remaining -= take;
    }
    Ok(out)
}

fn build_paged_shout_only_bus_zs(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lanes: usize,
    lane_data: &[neo_memory::riscv::trace::ShoutLaneOverTime],
    x_prefix: &[F],
) -> Result<Vec<Vec<F>>, String> {
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_paged_shout_only_bus_zs: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.len() != lanes {
        return Err(format!(
            "build_paged_shout_only_bus_zs: lane_data.len()={} != lanes={}",
            lane_data.len(),
            lanes
        ));
    }

    let page_ell_addrs = plan_paged_shout_addr(m, m_in, t, ell_addr, lanes)?;
    let mut out: Vec<Vec<F>> = Vec::with_capacity(page_ell_addrs.len());

    let mut bit_base: usize = 0;
    for (page_idx, &page_ell_addr) in page_ell_addrs.iter().enumerate() {
        let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
            m,
            m_in,
            t,
            core::iter::once((page_ell_addr, lanes)),
            core::iter::empty::<(usize, usize)>(),
        )?;
        if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
            return Err("build_paged_shout_only_bus_zs: expected 1 shout instance and 0 twist instances".into());
        }

        let mut z = vec![F::ZERO; m];
        z[..m_in].copy_from_slice(x_prefix);

        let shout = &bus.shout_cols[0];
        for (lane_idx, cols) in shout.lanes.iter().enumerate() {
            let lane = &lane_data[lane_idx];
            if lane.has_lookup.len() != t {
                return Err("build_paged_shout_only_bus_zs: lane length mismatch".into());
            }
            for j in 0..t {
                let has = lane.has_lookup[j];
                z[bus.bus_cell(cols.has_lookup, j)] = if has { F::ONE } else { F::ZERO };
                z[bus.bus_cell(cols.primary_val(), j)] = if has { F::from_u64(lane.value[j]) } else { F::ZERO };

                for (local_idx, col_id) in cols.addr_bits.clone().enumerate() {
                    let bit_idx = bit_base
                        .checked_add(local_idx)
                        .ok_or_else(|| "build_paged_shout_only_bus_zs: bit index overflow".to_string())?;
                    if bit_idx >= 64 {
                        return Err(format!(
                            "build_paged_shout_only_bus_zs: bit_idx={bit_idx} out of range for u64 key (page_idx={page_idx})"
                        ));
                    }
                    let b = if has { (lane.key[j] >> bit_idx) & 1 } else { 0 };
                    z[bus.bus_cell(col_id, j)] = if b == 1 { F::ONE } else { F::ZERO };
                }
            }
        }

        out.push(z);
        bit_base = bit_base
            .checked_add(page_ell_addr)
            .ok_or_else(|| "build_paged_shout_only_bus_zs: bit_base overflow".to_string())?;
    }
    if bit_base != ell_addr {
        return Err(format!(
            "build_paged_shout_only_bus_zs: paging mismatch (got bit_base={bit_base}, expected ell_addr={ell_addr})"
        ));
    }

    Ok(out)
}

#[test]
fn riscv_trace_wiring_ccs_shared_cpu_bus_shout_xor_paged_prove_verify() {
    // Program:
    // - LUI  x1, 0x80000        (x1 = 0x80000000)
    // - XORI x2, x0, 1          (x2 = 1)
    // - XOR  x3, x1, x2         (x3 = 0x80000001)
    // - HALT
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 0x80000 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Xor,
            rd: 3,
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
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Params + committer.
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n.max(ccs.m)).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    // Main CPU trace witness commitment.
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

    // Shout instance: XOR table, 1 lane, bit-addressed (ell_addr=64) → paged across multiple mats.
    let t = exec.rows.len();
    let shout_table_ids = vec![RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Xor).0];
    let shout_lanes = extract_shout_lanes_over_time(&exec, &shout_table_ids, /*xlen=*/ 32).expect("extract shout lanes");

    let ell_addr = 64usize;
    let lanes = 1usize;
    let page_ell_addrs = plan_paged_shout_addr(ccs.m, layout.m_in, t, ell_addr, lanes).expect("paging plan");

    let paged_zs = build_paged_shout_only_bus_zs(ccs.m, layout.m_in, t, ell_addr, lanes, &shout_lanes, &x)
        .expect("XOR Shout paged z");
    assert_eq!(paged_zs.len(), page_ell_addrs.len(), "z/page drift");

    let mut mats: Vec<Mat<F>> = Vec::with_capacity(paged_zs.len());
    let mut comms: Vec<Cmt> = Vec::with_capacity(paged_zs.len());
    for z in paged_zs {
        let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
        comms.push(l.commit(&Z));
        mats.push(Z);
    }

    let xor_lut_inst = LutInstance::<Cmt, F> {
        table_id: shout_table_ids[0],
        comms,
        k: 0,
        d: ell_addr,
        n_side: 2,
        steps: t,
        lanes,
        ell: 1,
        table_spec: Some(LutTableSpec::RiscvOpcode {
            opcode: RiscvOpcode::Xor,
            xlen: 32,
        }),
        table: Vec::new(),
    addr_group: None,
    selector_group: None,
    };
    let xor_lut_wit = LutWitness { mats };

    let steps_witness = vec![crate::common_setup::canonicalize_step_time_columns(StepWitnessBundle {
        mcs,
        lut_instances: vec![(xor_lut_inst, xor_lut_wit)],
        mem_instances: Vec::new(),
        time_columns: crate::common_setup::empty_time_columns(),
        _phantom: PhantomData,
    })];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, neo_math::K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_prove = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-xor-paged");
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
    .expect("prove");

    assert_eq!(
        proof.steps[0].mem.shout_me_claims_time.len(),
        page_ell_addrs.len(),
        "expected 1 Shout ME(time) claim per paging mat"
    );
    assert_eq!(
        proof.steps[0].shout_time_fold.len(),
        page_ell_addrs.len(),
        "expected 1 shout_time_fold proof per paging mat"
    );

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-trace-no-shared-bus-shout-xor-paged");
    let _ = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof,
        mixers,
    )
    .expect("verify");
}
