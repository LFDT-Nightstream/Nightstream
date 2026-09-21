use super::*;
use neo_ccs::{poly::Term, GeometricRowRun, SparsePoly};
use neo_math::{KExtensions, K};
use neo_reductions::superneo_eval::{
    check_ccs_relation_zero_cached_with_blocks, SuperneoCachedRelationError, SuperneoEvalCacheBuilder,
};
use p3_field::PrimeCharacteristicRing;
use std::sync::Arc;

#[test]
fn device_relation_matches_cpu_for_valid_invalid_and_constant_polynomials() {
    // Cross the session's 256-thread dispatch boundary and a partial ring block.
    let rows = 257;
    let columns = 2 * D + 1;
    let polynomial = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![7, 0, 0, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1, 0],
            },
            Term {
                coeff: F::from_u64(7),
                exps: vec![0, 0, 0, 0],
            },
            Term {
                coeff: F::from_u64(3),
                exps: vec![0, 0, 0, 2],
            },
        ],
    );
    let mut structure = CcsStructure::new_verifier_artifact_header(rows, columns, 4, polynomial).unwrap();
    let mut compact = SuperneoEvalCacheBuilder::new(rows, columns, 4).unwrap();
    let mut expanded = SuperneoEvalCacheBuilder::new(rows, columns, 4).unwrap();
    for row in 0..rows {
        let run = GeometricRowRun::new(
            row,
            1 + row % (columns - 41),
            41,
            F::from_usize(row + 1),
            F::from_u64(3),
        );
        let mut entries = Vec::new();
        run.for_each_term(|_, column, value| entries.push((column, value)));
        let value: F = entries.iter().map(|(_, value)| *value).sum();
        let output = value.exp_u64(7) + value + F::from_u64(7);
        compact.push_row_with_runs(0, row, [], [run]).unwrap();
        expanded.push_row(0, row, entries).unwrap();
        for builder in [&mut compact, &mut expanded] {
            builder.push_row(1, row, [(0, F::ONE)]).unwrap();
            builder
                .push_row(2, row, (output != F::ZERO).then_some((0, output)))
                .unwrap();
            builder.push_row(3, row, []).unwrap();
        }
    }
    let cache = Arc::new(compact.finish().unwrap());
    let expanded = expanded.finish().unwrap();
    let session = MetalSession::new().unwrap();
    let plan = session
        .prepare_joint_matrix_plan(Arc::clone(&cache))
        .unwrap();
    let mut witness = Mat::zero(D, columns.div_ceil(D), F::ZERO);
    for column in 0..columns {
        witness[(column % D, column / D)] = F::ONE;
    }
    let compare = |structure: &CcsStructure<F>, witness: &Mat<F>| {
        let blocks = SuperneoZBlocks::from_witness_mat(witness, columns).unwrap();
        let expected = match check_ccs_relation_zero_cached_with_blocks(&expanded, &structure.f, &blocks) {
            Ok(()) => None,
            Err(SuperneoCachedRelationError::UnsatisfiedRow { row }) => Some(row),
            Err(error) => panic!("unexpected CPU relation error: {error}"),
        };
        let before = session.activity().dispatches;
        let actual = session
            .first_unsatisfied_row(&plan, structure, witness)
            .unwrap();
        assert_eq!(actual, expected);
        assert!(session.activity().dispatches > before);
        assert!(
            plan.opening.get().is_none(),
            "row checking must not build an opening transpose"
        );
        actual
    };
    assert_eq!(compare(&structure, &witness), None);
    witness[(0, 1)] = -F::ONE;
    assert!(compare(&structure, &witness).is_some_and(|row| row > 0));
    witness[(0, 1)] = F::ONE;
    // Matrix rows do not use the completion tail. The terminal caller checks
    // fresh-tail zero separately; the row evaluator must preserve that split.
    witness[(D - 1, columns / D)] = -F::ONE;
    assert_eq!(compare(&structure, &witness), None);
    structure.f = SparsePoly::new(4, vec![]);
    assert_eq!(compare(&structure, &witness), None);
    structure.f = SparsePoly::new(
        4,
        vec![Term {
            coeff: F::ONE,
            exps: vec![0; 4],
        }],
    );
    assert_eq!(compare(&structure, &witness), Some(0));
    let before = session.activity().dispatches;
    witness[(0, 0)] = F::from_u64(2);
    assert!(session
        .first_unsatisfied_row(&plan, &structure, &witness)
        .is_err());
    structure.f = SparsePoly::new(3, vec![]);
    assert!(session
        .first_unsatisfied_row(&plan, &structure, &witness)
        .is_err());
    assert_eq!(session.activity().dispatches, before);

    let zero = Mat::virtual_constant(D, columns.div_ceil(D), F::ZERO);
    let point = vec![K::from_coeffs([F::from_u64(3), F::ONE]); rows.next_power_of_two().ilog2() as usize];
    let before = session.activity();
    let openings = session
        .eval_joint_dec_openings(&plan, &[zero], &point, columns)
        .unwrap()
        .unwrap();
    assert_eq!(openings[0].eval_k, vec![K::ZERO; D]);
    assert!(openings[0]
        .eval_a
        .iter()
        .all(|values| values == &vec![K::ZERO; D]));
    assert_eq!(session.activity().allocated_bytes, before.allocated_bytes);
    assert_eq!(session.activity().dispatches, before.dispatches);
    assert!(plan.opening.get().is_none());
}
