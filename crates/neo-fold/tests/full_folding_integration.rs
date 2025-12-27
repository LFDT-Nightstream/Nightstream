#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{
    matrix::Mat,
    relations::{CcsStructure, McsInstance, McsWitness, MeInstance},
};
use neo_fold::finalize::{FinalizeReport, ObligationFinalizer};
use neo_fold::shard::{fold_shard_prove, fold_shard_verify, fold_shard_verify_and_finalize, ShardObligations};
use neo_fold::shard::CommitMixers;
use neo_fold::PiCcsError;
use neo_math::{D, K};
use neo_memory::plain::{PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::utils;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

const TEST_M: usize = 64;
// Shared-bus padding validation requires a public constant-one column.
const M_IN: usize = 1;

/// Dummy commit that produces zero commitments.
#[derive(Clone, Copy, Default)]
struct DummyCommit;

impl SModuleHomomorphism<F, Cmt> for DummyCommit {
    fn commit(&self, z: &Mat<F>) -> Cmt {
        Cmt::zeros(z.rows(), 1)
    }

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

fn build_add_ccs(m: usize, chunk_size: usize, bus_base: usize, shout_ell_addr: usize, twist_ell_addr: usize) -> CcsStructure<F> {
    // A tiny R1CS CCS:
    //
    // 1) "Program output" constraint on row 0:
    //      (const_one + lhs0 + lhs1 - out) * const_one = 0
    //
    // 2) Shared-bus binding "touch" constraints, one per bus column:
    //      bus_col * const_one = bus_col
    //
    // 3) Shared-bus padding constraints:
    //      (1 - has_*) * field = 0
    //
    // This keeps the test CPU semantics minimal but satisfies the shared-bus guardrails.
    let n = m;
    let mut A = Mat::zero(n, m, F::ZERO);
    let mut B = Mat::zero(n, m, F::ZERO);
    let mut C = Mat::zero(n, m, F::ZERO);

    // Row 0: (z0 + z1 + z2 - z3) * z0 = 0
    A[(0, 0)] = F::ONE;
    A[(0, 1)] = F::ONE;
    A[(0, 2)] = F::ONE;
    A[(0, 3)] = F::ZERO - F::ONE;
    B[(0, 0)] = F::ONE;

    let bus_cols = (shout_ell_addr + 2) + (2 * twist_ell_addr + 5);
    let bus_region_len = bus_cols * chunk_size;
    assert!(
        bus_base + bus_region_len <= m,
        "bus region out of bounds (bus_base={}, bus_cols={}, chunk_size={}, m={})",
        bus_base,
        bus_cols,
        chunk_size,
        m
    );

    let per_step_padding_constraints = (shout_ell_addr + 1) + (2 * twist_ell_addr + 3);
    let include_binding_constraints = chunk_size == 1;
    let per_step_binding_constraints = if include_binding_constraints { bus_cols } else { 0 };
    let total_constraints =
        1 + chunk_size * (per_step_padding_constraints + per_step_binding_constraints);
    assert!(
        total_constraints <= n,
        "not enough rows for shared-bus guardrail constraints: need {}, have n={}",
        total_constraints,
        n
    );

    let mut row = 1usize;
    for j in 0..chunk_size {
        let mut col_id = 0usize;

        // Shout padding: (1 - has_lookup) * {addr_bits[b], val} = 0
        let shout_has_lookup = bus_base + (col_id + shout_ell_addr) * chunk_size + j;
        let shout_val = bus_base + (col_id + shout_ell_addr + 1) * chunk_size + j;
        // (1 - has_lookup) * val = 0
        A[(row, 0)] = F::ONE;
        A[(row, shout_has_lookup)] = F::ZERO - F::ONE;
        B[(row, shout_val)] = F::ONE;
        row += 1;
        for b in 0..shout_ell_addr {
            let bit = bus_base + (col_id + b) * chunk_size + j;
            A[(row, 0)] = F::ONE;
            A[(row, shout_has_lookup)] = F::ZERO - F::ONE;
            B[(row, bit)] = F::ONE;
            row += 1;
        }
        col_id += shout_ell_addr + 2;

        // Twist padding.
        let twist_has_read = bus_base + (col_id + 2 * twist_ell_addr + 0) * chunk_size + j;
        let twist_has_write = bus_base + (col_id + 2 * twist_ell_addr + 1) * chunk_size + j;
        let twist_wv = bus_base + (col_id + 2 * twist_ell_addr + 2) * chunk_size + j;
        let twist_rv = bus_base + (col_id + 2 * twist_ell_addr + 3) * chunk_size + j;
        let twist_inc = bus_base + (col_id + 2 * twist_ell_addr + 4) * chunk_size + j;

        // (1 - has_read) * rv = 0
        A[(row, 0)] = F::ONE;
        A[(row, twist_has_read)] = F::ZERO - F::ONE;
        B[(row, twist_rv)] = F::ONE;
        row += 1;
        // (1 - has_write) * wv = 0
        A[(row, 0)] = F::ONE;
        A[(row, twist_has_write)] = F::ZERO - F::ONE;
        B[(row, twist_wv)] = F::ONE;
        row += 1;
        // (1 - has_write) * inc = 0
        A[(row, 0)] = F::ONE;
        A[(row, twist_has_write)] = F::ZERO - F::ONE;
        B[(row, twist_inc)] = F::ONE;
        row += 1;

        for b in 0..twist_ell_addr {
            // (1 - has_read) * ra_bits[b] = 0
            let ra_bit = bus_base + (col_id + b) * chunk_size + j;
            A[(row, 0)] = F::ONE;
            A[(row, twist_has_read)] = F::ZERO - F::ONE;
            B[(row, ra_bit)] = F::ONE;
            row += 1;
        }
        for b in 0..twist_ell_addr {
            // (1 - has_write) * wa_bits[b] = 0
            let wa_bit = bus_base + (col_id + twist_ell_addr + b) * chunk_size + j;
            A[(row, 0)] = F::ONE;
            A[(row, twist_has_write)] = F::ZERO - F::ONE;
            B[(row, wa_bit)] = F::ONE;
            row += 1;
        }
    }

    // Non-padding "touch" constraints for every bus column (only for chunk_size==1 fixtures).
    //
    //   bus_col * 1 = bus_col
    //
    // These are tautologies, but they prevent a common linkage footgun where the CPU CCS only
    // references bus columns inside the padding rows.
    if include_binding_constraints {
        for j in 0..chunk_size {
            for col_id in 0..bus_cols {
                let bus_cell = bus_base + col_id * chunk_size + j;
                A[(row, bus_cell)] = F::ONE;
                B[(row, 0)] = F::ONE;
                C[(row, bus_cell)] = F::ONE;
                row += 1;
            }
        }
    }

    // Our constraints are R1CS-style: A(z)*B(z) = C(z).
    neo_ccs::r1cs_to_ccs(A, B, C)
}

fn build_mcs_from_z(
    params: &NeoParams,
    l: &DummyCommit,
    m_in: usize,
    z: Vec<F>,
) -> (McsInstance<Cmt, F>, McsWitness<F>) {
    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&Z);
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    (McsInstance { c, x, m_in }, McsWitness { w, Z })
}

fn write_bus_for_chunk(
    z: &mut [F],
    bus_base: usize,
    chunk_size: usize,
    lut_inst: &neo_memory::witness::LutInstance<Cmt, F>,
    lut_trace: &PlainLutTrace<F>,
    mem_inst: &neo_memory::witness::MemInstance<Cmt, F>,
    mem_trace: &PlainMemTrace<F>,
) {
    assert_eq!(lut_inst.steps, chunk_size);
    assert_eq!(mem_inst.steps, chunk_size);
    assert_eq!(lut_trace.has_lookup.len(), chunk_size);
    assert_eq!(lut_trace.addr.len(), chunk_size);
    assert_eq!(lut_trace.val.len(), chunk_size);
    assert_eq!(mem_trace.steps, chunk_size);

    // Clear bus region.
    let bus_cols = (lut_inst.d * lut_inst.ell + 2) + (2 * mem_inst.d * mem_inst.ell + 5);
    let bus_region_len = bus_cols * chunk_size;
    z[bus_base..bus_base + bus_region_len].fill(F::ZERO);

    // Fill each step j independently, with canonical column order:
    // Shout(addr_bits, has_lookup, val) then Twist(ra_bits, wa_bits, flags/vals/inc).
    for j in 0..chunk_size {
        let mut col_id = 0usize;

        // Shout addr bits (masked by has_lookup).
        let has_lookup = lut_trace.has_lookup[j];
        if has_lookup == F::ONE {
            let mut tmp = lut_trace.addr[j];
            for _dim in 0..lut_inst.d {
                let comp = (tmp % (lut_inst.n_side as u64)) as u64;
                tmp /= lut_inst.n_side as u64;
                for bit in 0..lut_inst.ell {
                    let b = if (comp >> bit) & 1 == 1 { F::ONE } else { F::ZERO };
                    z[bus_base + col_id * chunk_size + j] = b;
                    col_id += 1;
                }
            }
        } else {
            col_id += lut_inst.d * lut_inst.ell;
        }
        z[bus_base + col_id * chunk_size + j] = has_lookup;
        col_id += 1;
        z[bus_base + col_id * chunk_size + j] = if has_lookup == F::ONE { lut_trace.val[j] } else { F::ZERO };
        col_id += 1;

        // Twist read bits (masked by has_read).
        let has_read = mem_trace.has_read[j];
        if has_read == F::ONE {
            let mut tmp = mem_trace.read_addr[j];
            for _dim in 0..mem_inst.d {
                let comp = (tmp % (mem_inst.n_side as u64)) as u64;
                tmp /= mem_inst.n_side as u64;
                for bit in 0..mem_inst.ell {
                    let b = if (comp >> bit) & 1 == 1 { F::ONE } else { F::ZERO };
                    z[bus_base + col_id * chunk_size + j] = b;
                    col_id += 1;
                }
            }
        } else {
            col_id += mem_inst.d * mem_inst.ell;
        }

        // Twist write bits (masked by has_write).
        let has_write = mem_trace.has_write[j];
        if has_write == F::ONE {
            let mut tmp = mem_trace.write_addr[j];
            for _dim in 0..mem_inst.d {
                let comp = (tmp % (mem_inst.n_side as u64)) as u64;
                tmp /= mem_inst.n_side as u64;
                for bit in 0..mem_inst.ell {
                    let b = if (comp >> bit) & 1 == 1 { F::ONE } else { F::ZERO };
                    z[bus_base + col_id * chunk_size + j] = b;
                    col_id += 1;
                }
            }
        } else {
            col_id += mem_inst.d * mem_inst.ell;
        }

        z[bus_base + col_id * chunk_size + j] = has_read;
        col_id += 1;
        z[bus_base + col_id * chunk_size + j] = has_write;
        col_id += 1;
        z[bus_base + col_id * chunk_size + j] = if has_write == F::ONE { mem_trace.write_val[j] } else { F::ZERO };
        col_id += 1;
        z[bus_base + col_id * chunk_size + j] = if has_read == F::ONE { mem_trace.read_val[j] } else { F::ZERO };
        col_id += 1;
        z[bus_base + col_id * chunk_size + j] =
            if has_write == F::ONE { mem_trace.inc_at_write_addr[j] } else { F::ZERO };
        col_id += 1;

        debug_assert_eq!(col_id, bus_cols);
    }
}

fn default_mixers() -> Mixers {
    fn mix_rhos_commits(_rhos: &[Mat<F>], _cs: &[Cmt]) -> Cmt {
        Cmt::zeros(D, 1)
    }
    fn combine_b_pows(_cs: &[Cmt], _b: u32) -> Cmt {
        Cmt::zeros(D, 1)
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn build_single_chunk_inputs() -> (
    NeoParams,
    CcsStructure<F>,
    StepWitnessBundle<Cmt, F, K>,
    Vec<MeInstance<Cmt, F, K>>,
    Vec<Mat<F>>,
    DummyCommit,
    Mixers,
    F,
) {
    let m = TEST_M;
    let base_params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let params = NeoParams::new(
        base_params.q,
        base_params.eta,
        base_params.d,
        base_params.kappa,
        base_params.m,
        base_params.b,
        16, // bump k_rho to satisfy Π_RLC norm bound comfortably for this test
        base_params.T,
        base_params.s,
        base_params.lambda,
    )
    .expect("params");
    let l = DummyCommit::default();
    let mixers = default_mixers();

    // Program values: lookup_val + write_val = out
    let const_one = F::ONE;
    let write_val = F::from_u64(1);
    let lookup_val = F::from_u64(1);
    let out_val = const_one + write_val + lookup_val;

    // Build CCS (single chunk) enforcing out = write_val + lookup_val (bus vars are extra).
    let acc_init: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    // Plain memory trace for one step
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let plain_mem = PlainMemTrace {
        steps: 1,
        // One write to addr 0 with value 1 (no reads).
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::from_u64(1)],
        inc_at_write_addr: vec![F::from_u64(1)],
    };
    let mem_init = MemInit::Zero;

    // Plain lookup trace for one step
    let plain_lut = PlainLutTrace {
        // One lookup: key 0 -> value 1 (must match table content).
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![F::from_u64(1)],
    };
    let lut_table = neo_memory::plain::LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::from_u64(1), F::from_u64(2)],
    };

    // Shared-bus mode: instances are metadata-only; access rows live in the CPU witness.
    let mem_inst = neo_memory::witness::MemInstance::<Cmt, F> {
        comms: Vec::new(),
        k: mem_layout.k,
        d: mem_layout.d,
        n_side: mem_layout.n_side,
        steps: plain_mem.steps,
        ell: mem_layout.n_side.trailing_zeros() as usize,
        init: mem_init.clone(),
        _phantom: PhantomData,
    };
    let mem_wit = neo_memory::witness::MemWitness { mats: Vec::new() };
    let lut_inst = neo_memory::witness::LutInstance::<Cmt, F> {
        comms: Vec::new(),
        k: lut_table.k,
        d: lut_table.d,
        n_side: lut_table.n_side,
        steps: plain_mem.steps,
        ell: lut_table.n_side.trailing_zeros() as usize,
        table_spec: None,
        table: lut_table.content.clone(),
        _phantom: PhantomData,
    };
    let lut_wit = neo_memory::witness::LutWitness { mats: Vec::new() };

    // CPU witness z: core coords in [0..4), bus in the tail segment.
    let bus_cols = (lut_inst.d * lut_inst.ell + 2) + (2 * mem_inst.d * mem_inst.ell + 5);
    let chunk_size = plain_mem.steps;
    let bus_base = m - bus_cols * chunk_size;

    // Build CCS (single chunk) enforcing out = const_one + lookup_val + write_val,
    // and also referencing the shared-bus columns (without changing CCS semantics).
    let ccs = build_add_ccs(
        m,
        chunk_size,
        bus_base,
        /*shout_ell_addr=*/ lut_inst.d * lut_inst.ell,
        /*twist_ell_addr=*/ mem_inst.d * mem_inst.ell,
    );
    let _dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");

    let mut z = vec![F::ZERO; m];
    z[0] = const_one;
    z[1] = lookup_val;
    z[2] = write_val;
    z[3] = out_val;
    write_bus_for_chunk(&mut z, bus_base, chunk_size, &lut_inst, &plain_lut, &mem_inst, &plain_mem);
    let (mcs_inst, mcs_wit) = build_mcs_from_z(&params, &l, M_IN, z);

    let step_bundle = StepWitnessBundle {
        mcs: (mcs_inst.clone(), mcs_wit.clone()),
        lut_instances: vec![(lut_inst.clone(), lut_wit)],
        mem_instances: vec![(mem_inst.clone(), mem_wit)],
        _phantom: PhantomData::<K>,
    };

    (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers, out_val)
}

#[test]
fn full_folding_integration_single_chunk() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers, out_val) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let _outputs = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    )
    .expect("verify should succeed");

    // Print a short summary so it's clear what was enforced.
    let step0 = &proof.steps[0];
    let mem_me_val = step0.mem.cpu_me_claims_val.len();
    let ccs_me = step0.fold.ccs_out.len();
    let total_me = ccs_me;
    let children = step0.fold.dec_children.len();
    println!("Full folding step:");
    println!("  CCS ME count: {}", ccs_me);
    println!("  Twist+Shout ME count (r_time lane): 0 (shared bus)");
    println!("  Twist ME count (r_val lane): {}", mem_me_val);
    println!("  Total ME into RLC (r_time lane): {}", total_me);
    println!("  Children after DEC: {}", children);
    println!("  Lookup enforced: key 0 -> val 1 from table [1, 2]");
    println!("  Memory enforced: write addr 0 := 1 (inc +1)");

    // Program output comes from CCS: out = const_one + lookup + write = 3.
    println!("  Program output (CCS out) = {}", out_val.as_canonical_u64());

    // Show a small, deterministic slice of the folded output.
    let final_children = proof.compute_final_main_children(&acc_init);
    if let Some(first) = final_children.first() {
        let r_len = first.r.len();
        let y0_prefix: Vec<K> = first
            .y
            .get(0)
            .map(|row| row.iter().take(2).cloned().collect())
            .unwrap_or_default();
        let y_scalars_prefix: Vec<K> = first.y_scalars.iter().take(2).cloned().collect();
        println!(
            "  First child: r_len={}, y[0][..2]={:?}, y_scalars[..2]={:?}",
            r_len, y0_prefix, y_scalars_prefix
        );
    }
}

#[test]
fn full_folding_integration_multi_step_chunk() {
    let (params, _ccs_single, _step_bundle_1, acc_init, acc_wit_init, l, mixers, _out_val) = build_single_chunk_inputs();
    let m = TEST_M;

    // 4-step RW memory trace (k=2) with alternating write/read.
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };

    let plain_mem = PlainMemTrace {
        steps: 4,
        has_read: vec![F::ZERO, F::ONE, F::ZERO, F::ONE],
        has_write: vec![F::ONE, F::ZERO, F::ONE, F::ZERO],
        read_addr: vec![0, 0, 0, 1],
        write_addr: vec![0, 0, 1, 0],
        read_val: vec![F::ZERO, F::ONE, F::ZERO, F::from_u64(2)],
        write_val: vec![F::ONE, F::ZERO, F::from_u64(2), F::ZERO],
        inc_at_write_addr: vec![F::ONE, F::ZERO, F::from_u64(2), F::ZERO],
    };
    let mem_init = MemInit::Zero;

    // 4-step RO lookup trace (k=2) with lookups at steps 0 and 2.
    let lut_table = neo_memory::plain::LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::ONE, F::from_u64(2)],
    };
    let plain_lut = PlainLutTrace {
        has_lookup: vec![F::ONE, F::ZERO, F::ONE, F::ZERO],
        addr: vec![0, 0, 1, 0],
        val: vec![F::ONE, F::ZERO, F::from_u64(2), F::ZERO],
    };

    let mem_inst = neo_memory::witness::MemInstance::<Cmt, F> {
        comms: Vec::new(),
        k: mem_layout.k,
        d: mem_layout.d,
        n_side: mem_layout.n_side,
        steps: plain_mem.steps,
        ell: mem_layout.n_side.trailing_zeros() as usize,
        init: mem_init.clone(),
        _phantom: PhantomData,
    };
    let mem_wit = neo_memory::witness::MemWitness { mats: Vec::new() };
    let lut_inst = neo_memory::witness::LutInstance::<Cmt, F> {
        comms: Vec::new(),
        k: lut_table.k,
        d: lut_table.d,
        n_side: lut_table.n_side,
        steps: plain_mem.steps,
        ell: lut_table.n_side.trailing_zeros() as usize,
        table_spec: None,
        table: lut_table.content.clone(),
        _phantom: PhantomData,
    };
    let lut_wit = neo_memory::witness::LutWitness { mats: Vec::new() };

    let bus_cols = (lut_inst.d * lut_inst.ell + 2) + (2 * mem_inst.d * mem_inst.ell + 5);
    let chunk_size = plain_mem.steps;
    let bus_base = m - bus_cols * chunk_size;

    // CCS must be built for this chunk_size/bus_base so it references the correct bus indices.
    let ccs = build_add_ccs(
        m,
        chunk_size,
        bus_base,
        /*shout_ell_addr=*/ lut_inst.d * lut_inst.ell,
        /*twist_ell_addr=*/ mem_inst.d * mem_inst.ell,
    );

    let mut z = vec![F::ZERO; m];
    // Satisfy the add CCS constraint on row 0 with a trivial assignment.
    z[0] = F::ONE;
    z[1] = F::ZERO;
    z[2] = F::ZERO;
    z[3] = F::ONE;
    write_bus_for_chunk(&mut z, bus_base, chunk_size, &lut_inst, &plain_lut, &mem_inst, &plain_mem);
    let (mcs_inst, mcs_wit) = build_mcs_from_z(&params, &l, M_IN, z);

    let step_bundle = StepWitnessBundle {
        mcs: (mcs_inst, mcs_wit),
        lut_instances: vec![(lut_inst, lut_wit)],
        mem_instances: vec![(mem_inst, mem_wit)],
        _phantom: PhantomData,
    };

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-multi-step-chunk");
    let err = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect_err("multi-step chunks are intentionally unsupported in shared-bus-only mode");
    match err {
        PiCcsError::InvalidInput(msg) => assert!(
            msg.contains("chunk_size==1"),
            "unexpected InvalidInput for chunk_size>1: {msg}"
        ),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn tamper_batched_claimed_sum_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers, _out_val) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-tamper-claim");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    // Claim 0 is ccs/time; claim 1 is the first Shout time claim in this fixture.
    proof.steps[0].batched_time.claimed_sums[1] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-tamper-claim");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(result.is_err(), "tampered claimed sum must fail verification");
}

#[test]
fn tamper_me_opening_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers, _out_val) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-tamper-me");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    // Mutate a CPU ME opening used by shared-bus memory checks (a bus opening at r_time).
    let step0 = &mut proof.steps[0];
    let ccs_out0 = &mut step0.fold.ccs_out[0];
    let bus_cols = 10usize; // 1 Shout (3) + 1 Twist (7) in this fixture
    let bus_y_base = ccs_out0.y_scalars.len() - bus_cols;
    ccs_out0.y_scalars[bus_y_base] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-tamper-me");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(result.is_err(), "tampered ME opening must fail verification");
}

#[test]
fn tamper_shout_addr_pre_round_poly_fails() {
    use neo_fold::shard::MemOrLutProof;

    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers, _out_val) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-tamper-shout-addr-pre");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mem0 = proof.steps.get_mut(0).expect("one step");
    let shout0 = mem0.mem.proofs.get_mut(0).expect("one Shout proof");
    let shout_proof = match shout0 {
        MemOrLutProof::Shout(p) => p,
        _ => panic!("expected Shout proof"),
    };

    shout_proof.addr_pre.round_polys[0][0][0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-tamper-shout-addr-pre");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(
        result.is_err(),
        "tampered Shout addr-pre round poly must fail verification"
    );
}

#[test]
fn tamper_twist_val_eval_round_poly_fails() {
    use neo_fold::shard::MemOrLutProof;

    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers, _out_val) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-tamper-twist-val-eval-rounds");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mem0 = proof.steps.get_mut(0).expect("one step");
    let twist0 = mem0.mem.proofs.get_mut(1).expect("one Twist proof");
    let twist_proof = match twist0 {
        MemOrLutProof::Twist(p) => p,
        _ => panic!("expected Twist proof"),
    };
    let val_eval = twist_proof.val_eval.as_mut().expect("val_eval present");

    val_eval.rounds_lt[0][0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-tamper-twist-val-eval-rounds");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(
        result.is_err(),
        "tampered Twist val-eval round poly must fail verification"
    );
}

#[test]
fn missing_val_fold_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers, _out_val) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-missing-val-fold");
    let mut proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    assert!(
        proof.steps[0].val_fold.is_some(),
        "fixture should produce val_fold when Twist is present"
    );
    proof.steps[0].val_fold = None;

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-missing-val-fold");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let result = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(result.is_err(), "missing val_fold must fail verification");
}

#[test]
fn verify_and_finalize_receives_val_lane() {
    struct RequireValLane;

    impl ObligationFinalizer<Cmt, F, K> for RequireValLane {
        type Error = PiCcsError;

        fn finalize(&mut self, obligations: &ShardObligations<Cmt, F, K>) -> Result<FinalizeReport, Self::Error> {
            if obligations.val.is_empty() {
                return Err(PiCcsError::ProtocolError(
                    "expected non-empty val-lane obligations for Twist".into(),
                ));
            }
            Ok(FinalizeReport {
                did_finalize_main: !obligations.main.is_empty(),
                did_finalize_val: !obligations.val.is_empty(),
            })
        }
    }

    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers, _out_val) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-finalizer");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-finalizer");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let mut fin = RequireValLane;
    fold_shard_verify_and_finalize(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &mut fin,
    )
    .expect("verify_and_finalize should succeed");
}

#[test]
fn main_only_finalizer_is_rejected_when_val_lane_present() {
    struct MainOnly;

    impl ObligationFinalizer<Cmt, F, K> for MainOnly {
        type Error = PiCcsError;

        fn finalize(&mut self, obligations: &ShardObligations<Cmt, F, K>) -> Result<FinalizeReport, Self::Error> {
            Ok(FinalizeReport {
                did_finalize_main: !obligations.main.is_empty(),
                did_finalize_val: false,
            })
        }
    }

    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers, _out_val) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-finalizer-main-only");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-finalizer-main-only");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let mut fin = MainOnly;
    let res = fold_shard_verify_and_finalize(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
        &mut fin,
    );

    assert!(
        res.is_err(),
        "finalizer that does not finalize val-lane obligations must be rejected"
    );
}

#[test]
fn wrong_shout_lookup_value_witness_fails() {
    // In shared-bus mode, Shout consumes its access rows from the CPU witness.
    // So a wrong Shout value must make verification fail.
    let (params, ccs, mut step_bundle, acc_init, acc_wit_init, l, mixers, _out_val) = build_single_chunk_inputs();

    // Flip the Shout `val` entry inside the CPU bus tail (step j=0, last column of the Shout slot).
    // Layout for 1 Shout instance (d=1, ell=1, chunk_size=1):
    //   [addr_bit0, has_lookup, val] => Shout val is column id 2.
    // There is also 1 Twist instance after it, but we don't touch it.
    let m = step_bundle.mcs.1.Z.cols();
    let bus_cols = 10usize;
    let bus_base = m - bus_cols;
    let shout_val_col_id = 2usize;
    let mut z = neo_memory::ajtai::decode_vector(&params, &step_bundle.mcs.1.Z);
    z[bus_base + shout_val_col_id] += F::ONE;
    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
    let c = l.commit(&Z);
    step_bundle.mcs.0.c = c;
    step_bundle.mcs.1.Z = Z;
    step_bundle.mcs.1.w = z[M_IN..].to_vec();

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold-wrong-shout-bus");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed (even with invalid Shout bus)");

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold-wrong-shout-bus");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let res = fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    );

    assert!(res.is_err(), "wrong Shout bus value must fail verification");
}
