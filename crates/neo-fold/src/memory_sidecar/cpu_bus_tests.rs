#![allow(non_snake_case)]

use super::cpu_bus::append_bus_openings_to_me_instance;
use super::cpu_bus::decode_cpu_z_to_k;
use super::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps;
use neo_ajtai::Commitment;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::relations::CcsStructure;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_memory::ajtai::encode_vector_for_ccs_m;
use neo_memory::cpu::build_bus_layout_for_instances;
use neo_memory::cpu::constraints::{
    CpuConstraint, CpuConstraintBuilder, CpuConstraintLabel, ShoutCpuBinding, TwistCpuBinding,
};
use neo_memory::mem_init::MemInit;
use neo_memory::witness::{LutInstance, MemInstance, StepInstanceBundle};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn chi_for_row_index(r: &[K], idx: usize) -> K {
    let mut acc = K::ONE;
    for (bit, &ri) in r.iter().enumerate() {
        let is_one = ((idx >> bit) & 1) == 1;
        acc *= if is_one { ri } else { K::ONE - ri };
    }
    acc
}

#[test]
fn append_bus_openings_matches_manual_for_chunk_size_2() {
    let params = NeoParams::goldilocks_127();

    // Minimal shared-bus layout: one Shout instance with ell_addr=2.
    let m_in = 2usize;
    let chunk_size = 2usize;
    let m = D;
    let bus = build_bus_layout_for_instances(m, m_in, chunk_size, [2usize], []).expect("bus layout should build");
    assert!(bus.bus_cols > 0);

    // SuperNeo packed witness decomposition Z ∈ F^{D×ceil(m/D)} over logical width bus.m.
    let z_logical_seed: Vec<F> = (0..bus.m).map(|i| F::from_u64((i as u64) + 1)).collect();
    let Z = encode_vector_for_ccs_m(&params, bus.m, &z_logical_seed).expect("encode witness");

    // Start with a core ME instance of length core_t=1.
    let y_pad = (params.d as usize).next_power_of_two();
    let core_t = 1usize;
    let mut me = neo_ccs::CeClaim::<Commitment, F, K> {
        c: Commitment::zeros(params.d as usize, 1),
        X: Mat::zero(D, m_in, F::ZERO),
        r: vec![
            K::from(F::from_u64(3)),
            K::from(F::from_u64(5)),
            K::from(F::from_u64(7)),
            K::from(F::from_u64(11)),
        ],
        s_col: vec![],
        y_ring: vec![vec![K::ZERO; y_pad]],
        ct: vec![K::ZERO],
        aux_openings: Vec::new(),
        y_zcol: vec![],
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    append_bus_openings_to_me_instance(&params, &bus, core_t, &Z, &mut me).expect("append");

    assert_eq!(me.y_ring.len(), core_t);
    assert_eq!(me.ct.len(), core_t);
    assert_eq!(me.aux_openings.len(), bus.bus_cols);

    // Manual check of each appended bus opening.
    let time_weights: Vec<K> = (0..bus.chunk_size)
        .map(|j| chi_for_row_index(&me.r, bus.time_index(j)))
        .collect();
    let z_logical = decode_cpu_z_to_k(&params, &Z, bus.m).expect("decode");

    for col_id in 0..bus.bus_cols {
        let mut expect_scalar = K::ZERO;
        for j in 0..bus.chunk_size {
            let w = time_weights[j];
            let z_idx = bus.bus_cell(col_id, j);
            expect_scalar += w * z_logical[z_idx];
        }
        assert_eq!(me.aux_openings[col_id], expect_scalar, "col_id={col_id}");
    }
}

fn build_identity_first_r1cs_ccs_from_cpu_constraints(
    n: usize,
    m: usize,
    const_one_col: usize,
    constraints: &[CpuConstraint<F>],
) -> CcsStructure<F> {
    assert!(constraints.len() <= n, "too many constraints for n");

    let mut a_data = vec![F::ZERO; n * m];
    let mut b_data = vec![F::ZERO; n * m];
    let c_data = vec![F::ZERO; n * m];

    for (row, constraint) in constraints.iter().enumerate() {
        if constraint.negate_condition {
            a_data[row * m + const_one_col] = F::ONE;
            a_data[row * m + constraint.condition_col] = -F::ONE;
            for &col in &constraint.additional_condition_cols {
                a_data[row * m + col] = -F::ONE;
            }
        } else {
            a_data[row * m + constraint.condition_col] = F::ONE;
            for &col in &constraint.additional_condition_cols {
                a_data[row * m + col] = F::ONE;
            }
        }

        for &(col, coeff) in &constraint.b_terms {
            b_data[row * m + col] += coeff;
        }
    }

    let i_n = Mat::identity(n);
    let a = Mat::from_row_major(n, m, a_data);
    let b = Mat::from_row_major(n, m, b_data);
    let c = Mat::from_row_major(n, m, c_data);

    // Identity-first R1CS embedding: f(x0, x1, x2, x3) = x1*x2 - x3.
    let f = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 0, 1],
            },
        ],
    );

    CcsStructure::new(vec![i_n, a, b, c], f).expect("build CCS from constraints")
}

fn minimal_bus_steps(
    m_in: usize,
    chunk_size: usize,
    shout_d: usize,
    shout_ell: usize,
    twist_d: usize,
    twist_ell: usize,
) -> Vec<StepInstanceBundle<Commitment, F, K>> {
    let mut x = vec![F::ONE; m_in];
    if m_in == 0 {
        x = Vec::new();
    }
    let mcs_inst = neo_ccs::CcsClaim::<Commitment, F> {
        c: Commitment::zeros(1, 1),
        x,
        m_in,
    };

    let lut = LutInstance::<Commitment, F> {
        table_id: 0,
        comms: Vec::new(),
        k: 1usize << shout_d,
        d: shout_d,
        n_side: 2,
        steps: chunk_size,
        lanes: 1,
        ell: shout_ell,
        table_spec: None,
        table: Vec::new(),
        table_digest: None,
        addr_group: None,
        selector_group: None,
    };

    let mem = MemInstance::<Commitment, F> {
        mem_id: 0,
        comms: Vec::new(),
        k: 1usize << twist_d,
        d: twist_d,
        n_side: 2,
        steps: chunk_size,
        lanes: 1,
        ell: twist_ell,
        init: MemInit::Zero,
        init_digest: None,
        guest_addr_remap: None,
    };

    let mut step: StepInstanceBundle<Commitment, F, K> = StepInstanceBundle::from(mcs_inst);
    step.lut_insts = vec![lut];
    step.mem_insts = vec![mem];
    vec![step]
}

#[test]
fn shared_cpu_bus_padding_validator_accepts_implied_addr_bit_padding() {
    let m_in = 1usize;
    let chunk_size = 1usize;
    let shout_d = 2usize;
    let shout_ell = 1usize;
    let twist_d = 2usize;
    let twist_ell = 1usize;

    // Minimal CPU+bus witness shape:
    // - m_in=1 (const-one public input)
    // - 8 CPU columns for bindings
    // - bus tail for 1 shout (ell_addr=2) + 1 twist (ell_addr=2)
    let cpu_cols = 8usize;
    let bus_cols = (shout_d * shout_ell + 2) + (2 * (twist_d * twist_ell) + 5);
    let m = m_in + cpu_cols + bus_cols;
    let n = m;
    let const_one_col = 0usize;

    let bus = build_bus_layout_for_instances(m, m_in, chunk_size, [shout_d * shout_ell], [twist_d * twist_ell])
        .expect("bus layout");
    assert!(bus.bus_cols > 0);

    // Bindings live in the CPU region immediately before the bus tail.
    let cpu_has_read = m_in;
    let cpu_has_write = m_in + 1;
    let cpu_read_addr = m_in + 2;
    let cpu_write_addr = m_in + 3;
    let cpu_rv = m_in + 4;
    let cpu_wv = m_in + 5;
    let cpu_has_lookup = m_in + 6;
    let cpu_lookup_val = m_in + 7;

    let shout = &bus.shout_cols[0].lanes[0];
    let twist = &bus.twist_cols[0].lanes[0];

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, const_one_col);
    builder.add_shout_instance_bound(
        &bus,
        shout,
        &ShoutCpuBinding {
            has_lookup: cpu_has_lookup,
            addr: None,
            val: cpu_lookup_val,
        },
    );
    builder.add_twist_instance_bound(
        &bus,
        twist,
        &TwistCpuBinding {
            has_read: cpu_has_read,
            has_write: cpu_has_write,
            read_addr: cpu_read_addr,
            write_addr: cpu_write_addr,
            rv: cpu_rv,
            wv: cpu_wv,
            inc: None,
        },
    );

    // Use the builder's constraints but rebuild CCS locally so we can easily mutate the list in negative tests.
    let constraints = builder.constraints().to_vec();
    let ccs = build_identity_first_r1cs_ccs_from_cpu_constraints(n, m, const_one_col, &constraints);

    let steps = minimal_bus_steps(m_in, chunk_size, shout_d, shout_ell, twist_d, twist_ell);
    prepare_ccs_for_shared_cpu_bus_steps(&ccs, &steps).expect("padding validator should accept");
}

#[test]
fn shared_cpu_bus_padding_validator_requires_flag_boolean_for_implied_padding() {
    let m_in = 1usize;
    let chunk_size = 1usize;
    let shout_d = 2usize;
    let shout_ell = 1usize;
    let twist_d = 2usize;
    let twist_ell = 1usize;

    let cpu_cols = 8usize;
    let bus_cols = (shout_d * shout_ell + 2) + (2 * (twist_d * twist_ell) + 5);
    let m = m_in + cpu_cols + bus_cols;
    let n = m;
    let const_one_col = 0usize;

    let bus = build_bus_layout_for_instances(m, m_in, chunk_size, [shout_d * shout_ell], [twist_d * twist_ell])
        .expect("bus layout");

    let cpu_has_read = m_in;
    let cpu_has_write = m_in + 1;
    let cpu_read_addr = m_in + 2;
    let cpu_write_addr = m_in + 3;
    let cpu_rv = m_in + 4;
    let cpu_wv = m_in + 5;
    let cpu_has_lookup = m_in + 6;
    let cpu_lookup_val = m_in + 7;

    let shout = &bus.shout_cols[0].lanes[0];
    let twist = &bus.twist_cols[0].lanes[0];

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, const_one_col);
    builder.add_shout_instance_bound(
        &bus,
        shout,
        &ShoutCpuBinding {
            has_lookup: cpu_has_lookup,
            addr: None,
            val: cpu_lookup_val,
        },
    );
    builder.add_twist_instance_bound(
        &bus,
        twist,
        &TwistCpuBinding {
            has_read: cpu_has_read,
            has_write: cpu_has_write,
            read_addr: cpu_read_addr,
            write_addr: cpu_write_addr,
            rv: cpu_rv,
            wv: cpu_wv,
            inc: None,
        },
    );

    let constraints: Vec<CpuConstraint<F>> = builder
        .constraints()
        .iter()
        .cloned()
        .filter(|c| c.label != CpuConstraintLabel::ShoutHasLookupBoolean)
        .collect();
    // Sanity: we actually removed something.
    assert!(constraints.len() < builder.constraints().len());
    let ccs = build_identity_first_r1cs_ccs_from_cpu_constraints(n, m, const_one_col, &constraints);

    let steps = minimal_bus_steps(m_in, chunk_size, shout_d, shout_ell, twist_d, twist_ell);
    assert!(
        prepare_ccs_for_shared_cpu_bus_steps(&ccs, &steps).is_err(),
        "validator unexpectedly accepted implied padding without a boolean constraint on has_lookup"
    );
}

#[test]
fn shared_cpu_bus_padding_validator_requires_explicit_padding_for_nonbit_fields() {
    let m_in = 1usize;
    let chunk_size = 1usize;
    let shout_d = 2usize;
    let shout_ell = 1usize;
    let twist_d = 2usize;
    let twist_ell = 1usize;

    let cpu_cols = 8usize;
    let bus_cols = (shout_d * shout_ell + 2) + (2 * (twist_d * twist_ell) + 5);
    let m = m_in + cpu_cols + bus_cols;
    let n = m;
    let const_one_col = 0usize;

    let bus = build_bus_layout_for_instances(m, m_in, chunk_size, [shout_d * shout_ell], [twist_d * twist_ell])
        .expect("bus layout");

    let cpu_has_read = m_in;
    let cpu_has_write = m_in + 1;
    let cpu_read_addr = m_in + 2;
    let cpu_write_addr = m_in + 3;
    let cpu_rv = m_in + 4;
    let cpu_wv = m_in + 5;
    let cpu_has_lookup = m_in + 6;
    let cpu_lookup_val = m_in + 7;

    let shout = &bus.shout_cols[0].lanes[0];
    let twist = &bus.twist_cols[0].lanes[0];

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, const_one_col);
    builder.add_shout_instance_bound(
        &bus,
        shout,
        &ShoutCpuBinding {
            has_lookup: cpu_has_lookup,
            addr: None,
            val: cpu_lookup_val,
        },
    );
    builder.add_twist_instance_bound(
        &bus,
        twist,
        &TwistCpuBinding {
            has_read: cpu_has_read,
            has_write: cpu_has_write,
            read_addr: cpu_read_addr,
            write_addr: cpu_write_addr,
            rv: cpu_rv,
            wv: cpu_wv,
            inc: None,
        },
    );

    let constraints: Vec<CpuConstraint<F>> = builder
        .constraints()
        .iter()
        .cloned()
        .filter(|c| c.label != CpuConstraintLabel::ReadValueZeroPadding)
        .collect();
    assert!(constraints.len() < builder.constraints().len());
    let ccs = build_identity_first_r1cs_ccs_from_cpu_constraints(n, m, const_one_col, &constraints);

    let steps = minimal_bus_steps(m_in, chunk_size, shout_d, shout_ell, twist_d, twist_ell);
    assert!(
        prepare_ccs_for_shared_cpu_bus_steps(&ccs, &steps).is_err(),
        "validator unexpectedly accepted missing explicit padding for rv"
    );
}
