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
    assert!(length * 2 * size_of::<u64>() <= tables[0].length());
    assert!(length.div_ceil(2) * 2 * size_of::<u64>() <= tables[1].length());
    assert_eq!(session.read_buffer::<u64>(&tables[0], length * 2), vec![0; length * 2]);
}
