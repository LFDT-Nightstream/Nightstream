use super::*;
use neo_ccs::{CcsStructure, SparsePoly};
use neo_reductions::{engines::pi_ccs_joint::build_joint_dims, superneo_eval::SuperneoEvalCacheBuilder, Challenges};

fn empty_matrix(session: &MetalSession) -> (Arc<SuperneoEvalCache>, MetalJointMatrixPlan) {
    let mut cache = SuperneoEvalCacheBuilder::new(1, D, 1).unwrap();
    cache.push_row(0, 0, Vec::new()).unwrap();
    let cache = Arc::new(cache.finish().unwrap());
    let plan = session
        .prepare_joint_matrix_plan(Arc::clone(&cache))
        .unwrap();
    (cache, plan)
}

#[test]
fn empty_rows_do_not_read_dummy_offsets() {
    let session = MetalSession::new().unwrap();
    let (cache, plan) = empty_matrix(&session);
    assert_eq!(plan.matrices[0].row_offset_width, 0);
    // Empty-table buffers carry no entries. Poison them with a plausible
    // nonzero row so an accidental read produces the wrong application value.
    session
        .write_shared(&plan.matrices[0].row_offsets, &[0u32, 1u32])
        .unwrap();
    session
        .write_shared(&plan.matrices[0].row_blocks, &[0u32])
        .unwrap();
    let masks = session
        .prepare_witness_digit_masks(&[1, 0], 1, 1, 1, D)
        .unwrap();
    let table = session
        .build_joint_application_tables(&plan, &cache, &masks, 1, 1, 1, false)
        .unwrap();
    assert_eq!(session.read_buffer::<u64>(&table, 1), vec![0]);
}

#[test]
fn zero_carried_tables_fit_their_advertised_length() {
    let session = MetalSession::new().unwrap();
    let (cache, plan) = empty_matrix(&session);
    let structure = CcsStructure::new_verifier_artifact_header(1, D, 1, SparsePoly::new(1, vec![])).unwrap();
    let mut params = neo_params::NeoParams::nightstream_goldilocks_k16();
    let security = params
        .padded_row_security_summary_for_shape(
            structure.n,
            structure.m,
            structure.t(),
            structure.max_degree(),
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )
        .unwrap();
    params.lambda = params.lambda.min(security.security_bits);
    let dims = build_joint_dims(&params, &structure, 1, params.k_rho as usize).unwrap();
    let point = vec![K::ZERO; dims.variables];
    let fresh = [neo_ccs::CcsWitness {
        w: vec![],
        Z: Mat::zero(D, 1, F::ZERO),
    }];
    let running = vec![Mat::zero(D, 1, F::ZERO); params.k_rho as usize];
    let input = PaperJointOracleInput {
        structure: &structure,
        params: &params,
        fresh_witnesses: &fresh,
        running_witnesses: &running,
        challenges: Challenges::new(point.clone(), K::ONE),
        prior_point: Some(&point),
        dims,
        cache,
    };
    let count = fresh.len() + running.len();
    let masks = session
        .prepare_witness_digit_masks(&vec![0; 2 * count], count, 1, 1, D)
        .unwrap();
    let (tables, length) = session
        .build_joint_common_tables(&plan, &input, &masks, false)
        .unwrap();
    assert!(length * 2 * size_of::<u64>() <= tables.length());
    assert_eq!(session.read_buffer::<u64>(&tables, length * 2), vec![0; length * 2]);
}

#[test]
fn application_storage_does_not_reserve_future_rounds() {
    let session = MetalSession::new().unwrap();
    let mut bytes = Vec::new();
    // Both relations fit in the same assignment domain. Only the actual
    // application row count changes, so future rounds need no new storage yet.
    for rows in [3, 5] {
        let mut cache = SuperneoEvalCacheBuilder::new(rows, D, 1).unwrap();
        for row in 0..rows {
            cache.push_row(0, row, Vec::new()).unwrap();
        }
        let cache = Arc::new(cache.finish().unwrap());
        let plan = session
            .prepare_joint_matrix_plan(Arc::clone(&cache))
            .unwrap();
        let polynomial = SparsePoly::new(
            1,
            vec![
                neo_ccs::poly::Term {
                    coeff: F::ONE,
                    exps: vec![2],
                },
                neo_ccs::poly::Term {
                    coeff: -F::ONE,
                    exps: vec![1],
                },
            ],
        );
        let structure = CcsStructure::new_verifier_artifact_header(rows, D, 1, polynomial).unwrap();
        let mut params = neo_params::NeoParams::nightstream_goldilocks_k16();
        let security = params
            .padded_row_security_summary_for_shape(
                rows,
                D,
                1,
                structure.max_degree(),
                neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
            )
            .unwrap();
        params.lambda = params.lambda.min(security.security_bits);
        let dims = build_joint_dims(&params, &structure, 1, params.k_rho as usize).unwrap();
        let point = vec![K::ZERO; dims.variables];
        let fresh = [neo_ccs::CcsWitness {
            w: vec![],
            Z: Mat::virtual_constant(D, 1, F::ZERO),
        }];
        let running = vec![Mat::virtual_constant(D, 1, F::ZERO); params.k_rho as usize];
        let input = PaperJointOracleInput {
            structure: &structure,
            params: &params,
            fresh_witnesses: &fresh,
            running_witnesses: &running,
            challenges: Challenges::new(point.clone(), K::ONE),
            prior_point: Some(&point),
            dims,
            cache,
        };
        let before = session.activity().allocated_bytes;
        let _oracle = MetalPaperJointOracle::new(&session, &plan, input).unwrap();
        bytes.push(session.activity().allocated_bytes - before);
    }
    assert_eq!(bytes[1] - bytes[0], (2 * size_of::<F>()) as u64);
}

#[test]
fn application_row_scratch_does_not_expand_the_carrier() {
    let session = MetalSession::new().unwrap();
    let mut bytes = Vec::new();
    for blocks in [1, 2] {
        let columns = blocks * D;
        let mut cache = SuperneoEvalCacheBuilder::new(1, columns, 1).unwrap();
        cache
            .push_row(0, 0, [(columns - 1, F::from_u64(7))])
            .unwrap();
        let cache = Arc::new(cache.finish().unwrap());
        let plan = session
            .prepare_joint_matrix_plan(Arc::clone(&cache))
            .unwrap();
        let mut words = vec![0u64; 2 * blocks];
        words[2 * blocks - 1] = 1 << (D - 1);
        let masks = session
            .prepare_witness_digit_masks(&words, 1, blocks, 1, columns)
            .unwrap();
        let before = session.activity().allocated_bytes;
        let table = session
            .build_joint_application_tables(&plan, &cache, &masks, 1, 1, 1, false)
            .unwrap();
        bytes.push(session.activity().allocated_bytes - before);
        assert_eq!(
            session.read_buffer::<u64>(&table, 1),
            vec![(-F::from_u64(7)).as_canonical_u64()]
        );
    }
    assert_eq!(
        bytes[0], bytes[1],
        "row scratch must not grow with unused carrier columns"
    );
}

#[test]
fn zero_witness_suffix_does_not_allocate_device_masks() {
    let session = MetalSession::new().unwrap();
    let count = neo_params::NeoParams::nightstream_goldilocks_k16().k_rho as usize + 1;
    let blocks = 2;
    let mut words = vec![0u64; count * 2 * blocks];
    words[0] = 1;
    let before = session.activity().allocated_bytes;
    let masks = session
        .prepare_witness_digit_masks(&words, count, blocks, 1, D + 1)
        .unwrap();
    assert!(masks.matches_joint(count, blocks));
    assert_eq!(masks.active_witnesses(), &[0]);
    assert_eq!(
        session.read_buffer::<u64>(masks.words(), 2 * blocks),
        words[..2 * blocks]
    );
    assert_eq!(
        session.activity().allocated_bytes - before,
        (2 * blocks * size_of::<u64>()) as u64
    );

    words.fill(0);
    let before = session.activity().allocated_bytes;
    let masks = session
        .prepare_witness_digit_masks(&words, count, blocks, 1, D + 1)
        .unwrap();
    assert!(masks.active_witnesses().is_empty());
    assert_eq!(session.activity().allocated_bytes - before, size_of::<u64>() as u64);
}
