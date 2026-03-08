#![allow(non_snake_case)]

#[path = "../../common/fib_twist_shout_vm.rs"]
mod fib_twist_shout_vm;
#[path = "../../common/shared_bus_port_constraints.rs"]
mod shared_bus_port_constraints;
#[path = "../../common/twist_low_level_fixtures.rs"]
mod twist_low_level_fixtures;

use fib_twist_shout_vm::{add_mod_q, fib_mod_q_u64};
use neo_ajtai::Commitment as Cmt;
use neo_fold::output_binding::simple_output_config;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::CcsBuilder;
use neo_fold::shard::{fold_shard_prove_with_output_binding, fold_shard_verify_with_output_binding};
use neo_math::{F, K};
use neo_memory::cpu::{ShoutCpuBinding, ShoutInstanceShape, TwistCpuBinding};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::StepInstanceBundle;
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

const DEFAULT_STEPS: usize = 64;
const M_IN: usize = 1;
const CPU_F_CURR_BEFORE: usize = 0;
const CPU_F_NEXT_BEFORE: usize = 1;
const CPU_F_CURR_AFTER: usize = 2;
const CPU_F_NEXT_AFTER: usize = 3;
const CPU_SHOUT_HAS_LOOKUP: usize = 4;
const CPU_SHOUT_ADDR: usize = 5;
const CPU_SHOUT_VAL: usize = 6;
const CPU_TWIST_RA: usize = 7;
const CPU_TWIST_WA: usize = 8;
const CPU_TWIST_HAS_READ: usize = 9;
const CPU_TWIST_HAS_WRITE: usize = 10;
const CPU_TWIST_INC: usize = 11;
const CPU_COLS: usize = 12;
const BUS_SHOUT_VAL: usize = CPU_COLS + 2;
const BUS_TWIST_WV: usize = CPU_COLS + 7;
const BUS_TWIST_RV: usize = CPU_COLS + 8;
const TOTAL_COLS: usize = CPU_COLS + 10;

fn read_usize_env(key: &str) -> Option<usize> {
    std::env::var(key).ok().map(|v| {
        v.parse::<usize>()
            .unwrap_or_else(|_| panic!("invalid {key}={v:?} (expected usize)"))
    })
}

fn flat_col(base: usize, j: usize, t: usize) -> usize {
    M_IN + base * t + j
}

fn build_fib_ccs(t: usize) -> neo_ccs::relations::CcsStructure<F> {
    let mut cs = CcsBuilder::<F>::new(M_IN, /*const_one_col=*/ 0).expect("CcsBuilder");

    for j in 0..t {
        let f_curr_before = flat_col(CPU_F_CURR_BEFORE, j, t);
        let f_next_before = flat_col(CPU_F_NEXT_BEFORE, j, t);
        let f_curr_after = flat_col(CPU_F_CURR_AFTER, j, t);
        let f_next_after = flat_col(CPU_F_NEXT_AFTER, j, t);
        let shout_val = flat_col(BUS_SHOUT_VAL, j, t);
        let twist_wv = flat_col(BUS_TWIST_WV, j, t);
        let twist_rv = flat_col(BUS_TWIST_RV, j, t);

        cs.r1cs_terms(
            [(f_curr_before, F::ONE), (f_next_before, F::ONE)],
            [(shout_val, F::ONE)],
            [(f_next_after, F::ONE)],
        );
        cs.eq(f_curr_after, f_next_before);
        cs.eq(f_next_before, twist_rv);
        cs.eq(f_next_after, twist_wv);

        if j + 1 < t {
            cs.eq(flat_col(0, j + 1, t), f_curr_after);
            cs.eq(flat_col(1, j + 1, t), f_next_after);
        }
    }

    shared_bus_port_constraints::append_unshared_bus_binding_constraints(
        &mut cs,
        0,
        M_IN + CPU_COLS * t,
        t,
        &[ShoutInstanceShape {
            ell_addr: 1,
            lanes: 1,
            n_vals: 1,
            addr_group: None,
            selector_group: None,
        }],
        &[vec![ShoutCpuBinding {
            has_lookup: flat_col(CPU_SHOUT_HAS_LOOKUP, 0, t),
            addr: Some(flat_col(CPU_SHOUT_ADDR, 0, t)),
            val: flat_col(CPU_SHOUT_VAL, 0, t),
        }]],
        &[(1, 1)],
        &[vec![TwistCpuBinding {
            has_read: flat_col(CPU_TWIST_HAS_READ, 0, t),
            has_write: flat_col(CPU_TWIST_HAS_WRITE, 0, t),
            read_addr: flat_col(CPU_TWIST_RA, 0, t),
            write_addr: flat_col(CPU_TWIST_WA, 0, t),
            rv: flat_col(CPU_F_NEXT_BEFORE, 0, t),
            wv: flat_col(CPU_F_NEXT_AFTER, 0, t),
            inc: Some(flat_col(CPU_TWIST_INC, 0, t)),
        }]],
    )
    .expect("append canonical unshared bus bindings");

    cs.build_rect(M_IN + TOTAL_COLS * t, 0).expect("build_rect")
}

#[test]
fn twist_shout_fibonacci_cycle_trace() {
    let n_steps = read_usize_env("NEO_FIB_STEPS")
        .or_else(|| read_usize_env("NEO_FIB_CHUNKS").map(|chunks| chunks * 32))
        .unwrap_or(DEFAULT_STEPS);
    assert!(n_steps > 0, "NEO_FIB_STEPS must be > 0");

    let ccs = build_fib_ccs(n_steps);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;
    let l = twist_low_level_fixtures::setup_ajtai_committer(ccs.m, params.kappa as usize);
    let mixers = crate::common_setup::default_mixers();

    let q = params.q;
    let sanity_vm = fib_twist_shout_vm::FibTwistShoutVm::new(n_steps as u64, q);
    assert_eq!(sanity_vm.f_curr, 0);
    assert_eq!(sanity_vm.f_next, 1);
    let mut f_curr = 0u64;
    let mut f_next = 1u64;
    let mut mem_curr = 1u64;

    let mut f_curr_before = Vec::with_capacity(n_steps);
    let mut f_next_before = Vec::with_capacity(n_steps);
    let mut f_curr_after = Vec::with_capacity(n_steps);
    let mut f_next_after = Vec::with_capacity(n_steps);
    let shout_has_lookup = vec![F::ONE; n_steps];
    let shout_addr = vec![F::ONE; n_steps];
    let shout_val_shadow = vec![F::ONE; n_steps];
    let twist_ra = vec![F::ZERO; n_steps];
    let twist_wa = vec![F::ZERO; n_steps];
    let twist_has_read = vec![F::ONE; n_steps];
    let twist_has_write = vec![F::ONE; n_steps];
    let mut twist_inc_shadow = Vec::with_capacity(n_steps);

    let mut shout_trace = PlainLutTrace {
        has_lookup: vec![F::ONE; n_steps],
        addr: vec![1; n_steps],
        val: vec![F::ONE; n_steps],
    };
    let mut mem_trace = PlainMemTrace {
        steps: n_steps,
        has_read: vec![F::ONE; n_steps],
        has_write: vec![F::ONE; n_steps],
        read_addr: vec![0; n_steps],
        write_addr: vec![0; n_steps],
        read_val: vec![F::ZERO; n_steps],
        write_val: vec![F::ZERO; n_steps],
        inc_at_write_addr: vec![F::ZERO; n_steps],
    };

    for j in 0..n_steps {
        assert_eq!(mem_curr, f_next, "memory/state mismatch before step {j}");
        let f_new = add_mod_q(f_curr, f_next, q);

        f_curr_before.push(F::from_u64(f_curr));
        f_next_before.push(F::from_u64(f_next));
        f_curr_after.push(F::from_u64(f_next));
        f_next_after.push(F::from_u64(f_new));

        mem_trace.read_val[j] = F::from_u64(mem_curr);
        mem_trace.write_val[j] = F::from_u64(f_new);
        mem_trace.inc_at_write_addr[j] = F::from_u64(f_new) - F::from_u64(mem_curr);
        twist_inc_shadow.push(mem_trace.inc_at_write_addr[j]);

        f_curr = f_next;
        f_next = f_new;
        mem_curr = f_new;
    }
    shout_trace.val.fill(F::ONE);

    let expected_next = fib_mod_q_u64(n_steps + 1, q);
    assert_eq!(expected_next, f_next, "fibonacci simulation mismatch");

    let mem_layout = PlainMemLayout {
        k: 2,
        d: 1,
        n_side: 2,
        lanes: 1,
    };
    let (mem_inst, mem_wit) =
        twist_low_level_fixtures::make_twist_instance(0, &mem_layout, MemInit::Sparse(vec![(0, F::ONE)]), n_steps);
    let shout_table = LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::ZERO, F::ONE],
    };
    let (lut_inst, lut_wit) = twist_low_level_fixtures::make_shout_instance(0, shout_table, n_steps, 1);

    let step = twist_low_level_fixtures::create_step_with_bus(
        &params,
        &ccs,
        &l,
        M_IN,
        vec![F::ONE],
        vec![
            f_curr_before,
            f_next_before,
            f_curr_after,
            f_next_after,
            shout_has_lookup,
            shout_addr,
            shout_val_shadow,
            twist_ra,
            twist_wa,
            twist_has_read,
            twist_has_write,
            twist_inc_shadow,
        ],
        vec![(lut_inst, lut_wit, vec![shout_trace])],
        vec![(mem_inst, mem_wit, vec![mem_trace])],
    );
    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let ob_cfg = simple_output_config(1, 0, F::from_u64(expected_next));
    let final_memory_state = vec![F::from_u64(expected_next), F::ZERO];

    let mut tr_prove = Poseidon2Transcript::new(b"twist-shout/fibonacci");
    let proof = fold_shard_prove_with_output_binding(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
        &ob_cfg,
        &final_memory_state,
    )
    .expect("prove should succeed");
    assert!(proof.output_proof.is_some(), "output binding proof must be attached");

    let mut tr_verify = Poseidon2Transcript::new(b"twist-shout/fibonacci");
    let outputs = fold_shard_verify_with_output_binding(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof,
        mixers,
        &ob_cfg,
    )
    .expect("verify should succeed");

    assert!(
        !outputs.obligations.val.is_empty(),
        "twist proving should emit val-lane obligations"
    );
}
