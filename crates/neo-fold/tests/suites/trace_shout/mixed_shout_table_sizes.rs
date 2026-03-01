//! Mixed-size Shout addr-pre regression test.
//!
//! Route A used to require a single `ell_addr = d*ell` across all Shout instances in a step
//! because addr-pre was batched into one shared-challenge sumcheck. That forced small explicit
//! tables to be padded up to the largest table's address space (or rejected outright).
//!
//! This test exercises the fixed behavior: we run addr-pre batched **per `ell_addr` group**,
//! so a 16-entry table (`ell_addr = 4`) and a 256-entry table (`ell_addr = 8`) can coexist.

#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsClaim, CcsStructure, CcsWitness, CeClaim};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::CommitMixers;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{D, F, K};
use neo_memory::plain::{LutTable, PlainLutTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

const TEST_N: usize = 128;
const M_IN: usize = 0;

/// Setup real Ajtai public parameters for tests.
fn setup_ajtai_pp(m: usize, seed: u64) -> AjtaiSModule {
    let d = D;
    let m_commit = neo_memory::ajtai::commit_cols_for_ccs_m(m);
    if neo_ajtai::has_global_pp_for_dims(d, m_commit) {
        return AjtaiSModule::from_global_for_dims(d, m_commit).expect("from_global_for_dims");
    }
    let kappa = neo_params::NeoParams::goldilocks_auto_r1cs_ccs(m_commit)
        .expect("params")
        .kappa as usize;

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let pp = ajtai_setup(&mut rng, d, kappa, m_commit).expect("Ajtai setup should succeed");
    if let Err(e) = set_global_pp(pp) {
        if !neo_ajtai::has_global_pp_for_dims(d, m_commit) {
            panic!("set_global_pp: {e}");
        }
    }
    AjtaiSModule::from_global_for_dims(d, m_commit).expect("from_global_for_dims")
}

fn default_mixers() -> Mixers {
    crate::common_setup::default_mixers()
}

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn create_mcs_from_z(
    params: &NeoParams,
    l: &AjtaiSModule,
    m_in: usize,
    z: Vec<F>,
) -> (CcsClaim<Cmt, F>, CcsWitness<F>) {
    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&Z);

    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    (CcsClaim { c, x, m_in }, CcsWitness { w, Z })
}

fn make_shout_instance(
    table: &LutTable<F>,
    steps: usize,
) -> (
    neo_memory::witness::LutInstance<Cmt, F>,
    neo_memory::witness::LutWitness<F>,
) {
    let ell = table.n_side.trailing_zeros() as usize;
    (
        neo_memory::witness::LutInstance {
            table_id: table.table_id,
            comms: Vec::new(),
            k: table.k,
            d: table.d,
            n_side: table.n_side,
            steps,
            lanes: 1,
            ell,
            table_spec: None,
            table: table.content.clone(),
            table_digest: None,
            addr_group: None,
            selector_group: None,
        },
        neo_memory::witness::LutWitness { mats: Vec::new() },
    )
}

fn write_shout_bus_step(
    z: &mut [F],
    bus_base: usize,
    chunk_size: usize,
    j: usize,
    inst: &neo_memory::witness::LutInstance<Cmt, F>,
    trace: &PlainLutTrace<F>,
    col_id: &mut usize,
) {
    debug_assert_eq!(chunk_size, 1);
    debug_assert_eq!(j, 0);

    let has_lookup = trace.has_lookup[j];
    if has_lookup == F::ONE {
        let mut tmp = trace.addr[j];
        for _dim in 0..inst.d {
            let comp = (tmp % (inst.n_side as u64)) as u64;
            tmp /= inst.n_side as u64;
            for bit in 0..inst.ell {
                z[bus_base + *col_id * chunk_size + j] = if (comp >> bit) & 1 == 1 { F::ONE } else { F::ZERO };
                *col_id += 1;
            }
        }
    } else {
        *col_id += inst.d * inst.ell;
    }
    z[bus_base + *col_id * chunk_size + j] = has_lookup;
    *col_id += 1;
    z[bus_base + *col_id * chunk_size + j] = if has_lookup == F::ONE { trace.val[j] } else { F::ZERO };
    *col_id += 1;
}

fn create_step_with_shout_bus(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    l: &AjtaiSModule,
    tag: u64,
    luts: Vec<(&LutTable<F>, PlainLutTrace<F>)>,
) -> StepWitnessBundle<Cmt, F, K> {
    let chunk_size = 1usize;
    let mut bus_cols_total = 0usize;
    let mut lut_instances = Vec::with_capacity(luts.len());
    let mut traces = Vec::with_capacity(luts.len());

    for (table, trace) in luts {
        let steps = trace.has_lookup.len();
        assert_eq!(steps, chunk_size);
        assert_eq!(trace.addr.len(), chunk_size);
        assert_eq!(trace.val.len(), chunk_size);
        let (inst, wit) = make_shout_instance(table, steps);
        bus_cols_total += inst.d * inst.ell + 2;
        lut_instances.push((inst, wit));
        traces.push(trace);
    }

    let bus_base = ccs.m - bus_cols_total * chunk_size;
    let mut z = vec![F::ZERO; ccs.m];
    z[0] = F::from_u64(tag);

    let mut col_id = 0usize;
    for ((inst, _wit), trace) in lut_instances.iter().zip(traces.iter()) {
        write_shout_bus_step(&mut z, bus_base, chunk_size, 0, inst, trace, &mut col_id);
    }
    debug_assert_eq!(col_id, bus_cols_total);

    let (mcs, mcs_wit) = create_mcs_from_z(params, l, M_IN, z);
    crate::common_setup::canonicalize_step_time_columns(StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances,
        mem_instances: vec![],
        time_columns: crate::common_setup::empty_time_columns(),
        _phantom: PhantomData::<K>,
    })
}

#[test]
fn mixed_shout_tables_16_and_256_entries_same_step() {
    let ccs = create_identity_ccs(TEST_N);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x5001);
    let mixers = default_mixers();

    // Table 0: 16-entry identity table (4 address bits).
    let table_16 = LutTable {
        table_id: 0,
        k: 16,
        d: 1,
        n_side: 16,
        content: (0u64..16).map(F::from_u64).collect(),
    };
    // Table 1: 256-entry identity table (8 address bits).
    let table_256 = LutTable {
        table_id: 1,
        k: 256,
        d: 1,
        n_side: 256,
        content: (0u64..256).map(F::from_u64).collect(),
    };

    let trace_16 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![7],
        val: vec![table_16.content[7]],
    };
    let trace_256 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![200],
        val: vec![table_256.content[200]],
    };

    let step_bundle = create_step_with_shout_bus(
        &params,
        &ccs,
        &l,
        123,
        vec![(&table_16, trace_16), (&table_256, trace_256)],
    );

    let acc_init: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"mixed-shout-sizes");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed with mixed ell_addr tables");

    // Addr-pre should be split into two `ell_addr` groups: 4 bits for table_16, 8 bits for table_256.
    let pre = &proof.steps[0].mem.shout_addr_pre;
    assert_eq!(pre.groups.len(), 2);
    assert_eq!(pre.groups[0].ell_addr, 4);
    assert_eq!(pre.groups[1].ell_addr, 8);
    assert_eq!(pre.groups[0].active_lanes, vec![0]);
    assert_eq!(pre.groups[1].active_lanes, vec![1]);
    assert_eq!(pre.groups[0].round_polys.len(), 1);
    assert_eq!(pre.groups[1].round_polys.len(), 1);
    assert_eq!(pre.groups[0].r_addr.len(), 4);
    assert_eq!(pre.groups[1].r_addr.len(), 8);

    let mut tr_verify = Poseidon2Transcript::new(b"mixed-shout-sizes");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let _ = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    )
    .expect("verify should succeed with mixed ell_addr tables");
}
