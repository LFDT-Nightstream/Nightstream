//! Route A demo: using Shout as a 32-bit range check via an implicit identity table.
//!
//! This test demonstrates:
//! - No dense `2^32` table materialization: `table_spec = IdentityU32`, `table = []`, `k = 0`.
//! - Verifier computes `tablẽ(r_addr)` in O(32) from the public spec.
//! - The relation enforces `val == addr` (packed from `addr_bits`) when `has_lookup = 1`.
//! - Multi-lane lookups may reuse the same address (multiset semantics).
//!
//! Run:
//! - `cargo test -p neo-fold --test shout_identity_u32_range_check --release`

use std::marker::PhantomData;

use neo_ajtai::{AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsClaim, CcsStructure, CcsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::FoldingSession;
use neo_math::{F, K};
use neo_memory::witness::{LutInstance, LutTableSpec, LutWitness, StepWitnessBundle};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

const TEST_M: usize = 128;
const M_IN: usize = 0;

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    // Constant polynomial: no active constraints (keeps shared-bus padding guardrails off in tests).
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn create_mcs_from_z(
    params: &NeoParams,
    l: &impl SModuleHomomorphism<F, Cmt>,
    m_in: usize,
    z: Vec<F>,
) -> (CcsClaim<Cmt, F>, CcsWitness<F>) {
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    let z_mat = neo_memory::ajtai::encode_vector_balanced_to_mat(params, &z);
    let c = l.commit(&z_mat);

    (CcsClaim { c, x, m_in }, CcsWitness { w, Z: z_mat })
}

/// Write one Shout lane row into the shared CPU bus tail at `bus_base`.
fn write_shout_lane_row(
    z: &mut [F],
    bus_base: usize,
    chunk_size: usize,
    j: usize,
    inst: &LutInstance<Cmt, F>,
    addr: u64,
    val: F,
    has_lookup: F,
) {
    debug_assert_eq!(chunk_size, 1);
    debug_assert_eq!(j, 0);

    // Layout: [addr_bits(d*ell), has_lookup, val]
    let mut col_id = 0usize;
    if has_lookup == F::ONE {
        let mut tmp = addr;
        for _dim in 0..inst.d {
            let comp = (tmp % (inst.n_side as u64)) as u64;
            tmp /= inst.n_side as u64;
            for bit in 0..inst.ell {
                z[bus_base + col_id * chunk_size + j] = if (comp >> bit) & 1 == 1 { F::ONE } else { F::ZERO };
                col_id += 1;
            }
        }
    } else {
        col_id += inst.d * inst.ell;
    }

    z[bus_base + col_id * chunk_size + j] = has_lookup;
    col_id += 1;
    z[bus_base + col_id * chunk_size + j] = if has_lookup == F::ONE { val } else { F::ZERO };
    col_id += 1;

    debug_assert_eq!(col_id, inst.d * inst.ell + 2);
}

#[test]
fn route_a_shout_identity_u32_range_check_two_lanes_same_value_verifies() {
    let ccs = create_identity_ccs(TEST_M);
    let mut session = FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, [7u8; 32])
        .expect("new_ajtai_seeded");
    let params = session.params().clone();

    // "Range check a u32": represent it as 32 address bits, and enforce val == addr.
    let x: u64 = 0x1234_5678;

    let inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: Vec::new(),
        k: 0,
        d: 32,
        n_side: 2,
        steps: 1,
        lanes: 2,
        ell: 1,
        table_spec: Some(LutTableSpec::IdentityU32),
        table: vec![],
        table_digest: None,
        addr_group: None,
        selector_group: None,
    };
    let wit = LutWitness { mats: Vec::new() };

    let lane_len = inst.d * inst.ell + 2;
    let bus_cols_total = inst.lanes * lane_len;
    let bus_base = ccs.m - bus_cols_total;

    let mut z = vec![F::ZERO; ccs.m];
    for lane in 0..inst.lanes {
        let lane_base = bus_base + lane * lane_len;
        write_shout_lane_row(&mut z, lane_base, 1, 0, &inst, x, F::from_u64(x), F::ONE);
    }

    let (mcs, mcs_wit) = create_mcs_from_z(&params, session.committer(), M_IN, z);
    let step_bundle = crate::common_setup::canonicalize_step_time_columns(StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![(inst, wit)],
        mem_instances: vec![],
        time_columns: crate::common_setup::empty_time_columns(),
        _phantom: PhantomData::<K>,
    });

    session.add_step_bundle(step_bundle);
    let _run = session
        .prove_and_verify_collected(&ccs)
        .expect("verify should succeed");
}

#[test]
fn route_a_shout_identity_u32_range_check_rejects_wrong_val() {
    let ccs = create_identity_ccs(TEST_M);
    let mut session = FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, [8u8; 32])
        .expect("new_ajtai_seeded");
    let params = session.params().clone();

    let x: u64 = 0x1234_5678;
    let bad: u64 = x.wrapping_add(5);

    let inst = LutInstance::<Cmt, F> {
        table_id: 0,
        comms: Vec::new(),
        k: 0,
        d: 32,
        n_side: 2,
        steps: 1,
        lanes: 1,
        ell: 1,
        table_spec: Some(LutTableSpec::IdentityU32),
        table: vec![],
        table_digest: None,
        addr_group: None,
        selector_group: None,
    };
    let wit = LutWitness { mats: Vec::new() };

    let lane_len = inst.d * inst.ell + 2;
    let bus_base = ccs.m - lane_len;

    let mut z = vec![F::ZERO; ccs.m];
    // Keep the same addr_bits, but provide an incorrect value.
    write_shout_lane_row(&mut z, bus_base, 1, 0, &inst, x, F::from_u64(bad), F::ONE);

    let (mcs, mcs_wit) = create_mcs_from_z(&params, session.committer(), M_IN, z);
    let step_bundle = crate::common_setup::canonicalize_step_time_columns(StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: vec![(inst, wit)],
        mem_instances: vec![],
        time_columns: crate::common_setup::empty_time_columns(),
        _phantom: PhantomData::<K>,
    });

    session.add_step_bundle(step_bundle);
    let _ = session
        .prove_and_verify_collected(&ccs)
        .expect_err("verification must reject if val != addr under IdentityU32");
}
