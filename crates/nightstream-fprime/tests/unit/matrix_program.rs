use super::*;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};
use std::ops::ControlFlow;

#[path = "../support/poseidon_affine_fixture.rs"]
mod affine_fixture;

const FIRST_POSEIDON_CONSTANT: u64 = 15_504_881_536_434_223_753;

#[test]
fn affine_poseidon_inputs_preserve_values_indices_and_tag_selection() {
    let program = poseidon_input::Program::decode(&affine_fixture::program()).expect("affine input program");
    let first = program.state(32, 31, 0).expect("first affine input");
    let second = program.state(32, 31, 1).expect("second affine input");
    for (form, expected) in [
        (&first[0], vec![(4, 1), (5, 3), (31, 7)]),
        (&first[1], vec![(5, 4), (31, 11)]),
        (&second[0], vec![(31, 13)]),
        (&second[1], vec![(4, 5), (31, 17)]),
    ] {
        assert_eq!(
            form.entries()
                .iter()
                .map(|entry| (entry.column, entry.coefficient.as_canonical_u64()))
                .collect::<Vec<_>>(),
            expected,
        );
    }
    assert!(first[2..].iter().all(|form| form.entries().is_empty()));
    assert!(second[2..].iter().all(|form| form.entries().is_empty()));
    assert!(program
        .state(32, 31, 2)
        .expect("inactive tag")
        .iter()
        .all(|form| form.entries().is_empty()));
    assert!(program.state(32, 32, 0).is_err(), "constant column must be in range");
}

#[test]
fn affine_poseidon_inputs_reject_malformed_words_and_sources() {
    for (label, input) in affine_fixture::malformed() {
        assert!(
            poseidon_input::Program::decode(&input)
                .and_then(|program| program.state(32, 31, 0))
                .is_err(),
            "accepted {label}",
        );
    }
}

fn affine_constant(coefficient: u64) -> Value {
    json!([[[0, 1, 0, 1, 0, 1], [1, coefficient]]])
}

fn encoded_program() -> Value {
    let one_column = 5_999;
    let ordinary = json!([0, [[0, [[0, 1]]], one_column, [[[0, 3, [0, 3, 0], 0]], []], [0]]]);
    let pin = json!([1, [one_column, [[[90, 7]]]]]);
    let multiplication = json!([
        4,
        [
            [1, 1, 1],
            one_column,
            affine_constant(5),
            affine_constant(6),
            affine_constant(30)
        ]
    ]);
    let poseidon = json!([2, [1, one_column, [2, 86, 100], []]]);
    let phi81 = json!([
        3,
        [
            [[1, 1, 1]],
            one_column,
            [0, 54, 4_000],
            0,
            54,
            [[[0, 54, [0, 54, 4_054], 0]], []],
            [0, 54, 4_108],
            [0, 54, 4_162]
        ]
    ]);
    json!([ordinary, pin, multiplication, poseidon, phi81])
}

fn source_row(index: usize) -> Result<SourceRow, PackageError> {
    if index != 0 {
        return Err(PackageError::Invalid("test source row"));
    }
    Ok(SourceRow {
        a: SourceCombination {
            constant: Goldilocks::from_u64(2),
            terms: vec![Entry {
                column: 0,
                coefficient: Goldilocks::from_u64(3),
            }],
        },
        b: SourceCombination {
            constant: Goldilocks::ZERO,
            terms: vec![Entry {
                column: 1,
                coefficient: Goldilocks::from_u64(4),
            }],
        },
        c: SourceCombination {
            constant: Goldilocks::from_u64(5),
            terms: vec![Entry {
                column: 2,
                coefficient: Goldilocks::from_u64(6),
            }],
        },
    })
}

fn entries(program: &MatrixProgram, row: usize, matrix: usize) -> Vec<(usize, u64)> {
    program.row(6_000, row, &source_row).expect("matrix row")[matrix]
        .entries()
        .iter()
        .map(|entry| (entry.column, entry.coefficient.as_canonical_u64()))
        .collect()
}

fn source_row_at(column: usize) -> Result<SourceRow, PackageError> {
    Ok(SourceRow {
        a: SourceCombination {
            constant: Goldilocks::ZERO,
            terms: vec![Entry {
                column,
                coefficient: Goldilocks::from_u64(7),
            }],
        },
        b: SourceCombination {
            constant: Goldilocks::ZERO,
            terms: vec![],
        },
        c: SourceCombination {
            constant: Goldilocks::ZERO,
            terms: vec![],
        },
    })
}

fn projection_program(projection: Value) -> MatrixProgram {
    MatrixProgram::decode(&json!([[
        0,
        [[0, [[0, 1]]], 5_999, [[[0, 3, [0, 3, 20], 0]], []], projection]
    ]]))
    .expect("wire-valid projection program")
}

#[test]
fn every_lean_matrix_opcode_decodes_exact_rows() {
    let program = MatrixProgram::decode(&encoded_program()).expect("matrix program");
    program.validate(1).expect("source schedule");
    assert_eq!(program.row_count().expect("row count"), 197);

    assert_eq!(entries(&program, 0, 1), vec![(5_999, 1)]);
    assert_eq!(entries(&program, 0, 2), vec![(0, 3), (5_999, 2)]);
    assert_eq!(entries(&program, 0, 3), vec![(1, 4)]);
    assert_eq!(entries(&program, 0, 4), vec![(2, 6), (5_999, 5)]);

    assert_eq!(entries(&program, 1, 1), vec![(5_999, 1)]);
    assert_eq!(entries(&program, 1, 4), vec![(90, 7)]);

    assert_eq!(entries(&program, 2, 2), vec![(5_999, 5)]);
    assert_eq!(entries(&program, 2, 3), vec![(5_999, 6)]);
    assert_eq!(entries(&program, 2, 4), vec![(5_999, 30)]);

    let poseidon_row = 3;
    assert_eq!(entries(&program, poseidon_row, 1), vec![(5_999, 1)]);
    assert_eq!(
        entries(&program, poseidon_row, 5),
        vec![(5_999, FIRST_POSEIDON_CONSTANT)]
    );
    let first_output = entries(&program, poseidon_row, 4);
    assert_eq!(first_output.len(), 41);
    assert_eq!(first_output[0], (100, 1));
    assert_eq!(first_output[1], (101, 3));
    assert_eq!(entries(&program, poseidon_row + 32, 4)[0].0, 100 + 32 * 41);
    assert_eq!(entries(&program, poseidon_row + 54, 4)[0].0, 100 + 54 * 41);

    assert_eq!(entries(&program, poseidon_row + 85, 4)[0].0, 100 + 85 * 41);

    let phi_row = poseidon_row + 86;
    assert_eq!(
        entries(&program, phi_row, 0),
        vec![(4_000, 1), (5_999, GOLDILOCKS_MODULUS - 2)]
    );
    assert_eq!(entries(&program, phi_row, 2), vec![(4_054, 1)]);
    assert_eq!(entries(&program, phi_row, 4), vec![(4_108, 1), (4_162, 1)]);
    assert_eq!(entries(&program, phi_row, 7), vec![(5_999, 1)]);

    let phi_at_one = entries(&program, phi_row + 1, 4);
    assert_eq!(phi_at_one.len(), 108);
    assert_eq!(phi_at_one[0], (4_108, 1));
    assert_eq!(phi_at_one[53], (4_108 + 53, 1));
    assert_eq!(phi_at_one[54], (4_162, 3));
    assert_eq!(phi_at_one[107], (4_162 + 53, 3));

    assert_eq!(MEANINGFUL_PORTS, 13);
}

#[test]
fn linear_poseidon_visitor_matches_every_random_access_row() {
    let program = MatrixProgram::decode(&encoded_program()).expect("matrix program");
    let block = match &program.blocks[3] {
        Block::Poseidon(block) => block,
        _ => panic!("fixture Poseidon2 block"),
    };
    let row_count = block.row_count().expect("Poseidon2 row count");
    let expected = (0..row_count)
        .map(|row| block.row(6_000, row).expect("random-access row"))
        .collect::<Vec<_>>();
    let mut visited = Vec::new();
    let flow = block
        .visit_rows_until(6_000, 0, row_count, |row| {
            visited.push(owned_row(row));
            Ok(ControlFlow::Continue(()))
        })
        .expect("linear Poseidon2 rows");
    assert_eq!(flow, ControlFlow::Continue(()));
    assert_eq!(visited, expected);
}

#[test]
fn malformed_matrix_programs_fail_closed() {
    assert!(matches!(
        MatrixProgram::decode(&json!([[9, []]])),
        Err(PackageError::Invalid("production matrix block tag"))
    ));
    assert!(matches!(
        MatrixProgram::decode(&json!([[1, [0, [[[0, GOLDILOCKS_MODULUS]]]]]])),
        Err(PackageError::NonCanonicalField { .. })
    ));

    let wrong_kind =
        MatrixProgram::decode(&json!([[2, [1, 5_999, [0, 86, 100], []]]])).expect("wire-valid Poseidon block");
    assert!(matches!(
        wrong_kind.row(6_000, 0, &source_row),
        Err(PackageError::Invalid("Poseidon2 retained kind"))
    ));

    let missing_source =
        MatrixProgram::decode(&json!([[0, [[0, [[0, 1]]], 5_999, [[[0, 1, [0, 1, 0], 0]], []], [0]]]]))
            .expect("wire-valid ordinary block");
    assert!(matches!(
        missing_source.row(6_000, 0, &|_| Ok(SourceRow {
            a: SourceCombination {
                constant: Goldilocks::ZERO,
                terms: vec![Entry {
                    column: 2,
                    coefficient: Goldilocks::ONE,
                }],
            },
            b: SourceCombination {
                constant: Goldilocks::ZERO,
                terms: vec![],
            },
            c: SourceCombination {
                constant: Goldilocks::ZERO,
                terms: vec![],
            },
        })),
        Err(PackageError::Invalid("missing matrix source substitution"))
    ));
}

#[test]
fn mapped_source_projection_recovers_the_lean_source_column() {
    let program = projection_program(json!([1, [[100, 0, 3]]]));
    let row = program
        .row(6_000, 0, &|_| source_row_at(101))
        .expect("projected ordinary row");
    assert_eq!(
        row[2]
            .entries()
            .iter()
            .map(|entry| (entry.column, entry.coefficient.as_canonical_u64()))
            .collect::<Vec<_>>(),
        vec![(21, 7)]
    );
}

#[test]
fn missing_source_projection_range_fails_closed() {
    let program = projection_program(json!([1, [[100, 0, 1]]]));
    assert!(matches!(
        program.row(6_000, 0, &|_| source_row_at(101)),
        Err(PackageError::Invalid("missing or overlapping matrix source projection"))
    ));
}

#[test]
fn overlapping_source_projection_ranges_fail_closed() {
    let program = projection_program(json!([1, [[100, 0, 2], [101, 1, 1]]]));
    assert!(matches!(
        program.row(6_000, 0, &|_| source_row_at(101)),
        Err(PackageError::Invalid("missing or overlapping matrix source projection"))
    ));
}

#[test]
fn embedded_ordinary_rows_use_their_own_table_and_checked_projection() {
    let block = json!([[0, [[0, 1]]], 9, [[[10, 2, [0, 2, 2], 0]], []], [1, [[0, 10, 2]]]]);
    let rows = json!([[2, [[0, 3], [0, GOLDILOCKS_MODULUS - 1]]], [0, [[1, 5]]], [7, [[0, 1]]]]);
    let wire = json!([[6, block, rows]]);
    let program = MatrixProgram::decode(&wire).expect("embedded ordinary program");
    program.validate(0).expect("no external source rows needed");
    assert_eq!(program.row_count().unwrap(), 1);
    let row = program
        .row(10, 0, &|_| panic!("embedded row read the external source"))
        .expect("embedded row");
    for (port, expected) in [
        (1, vec![(9, 1)]),
        (2, vec![(2, 2), (9, 2)]),
        (3, vec![(3, 5)]),
        (4, vec![(2, 1), (9, 7)]),
    ] {
        assert_eq!(
            row[port]
                .entries()
                .iter()
                .map(|entry| (entry.column, entry.coefficient.as_canonical_u64()))
                .collect::<Vec<_>>(),
            expected
        );
    }

    let mut removed = wire.clone();
    removed[0][2][0][1]
        .as_array_mut()
        .unwrap()
        .push(json!([2, 0]));
    assert!(
        MatrixProgram::decode(&removed)
            .unwrap()
            .row(10, 0, &source_row)
            .is_err(),
        "a zero term must not hide an unmapped helper read"
    );
    let mut missing = wire.clone();
    missing[0][2].as_array_mut().unwrap().pop();
    let missing = MatrixProgram::decode(&missing).unwrap();
    assert!(
        missing.validate(usize::MAX).is_err(),
        "external row count cannot repair a short template"
    );
    assert!(missing.row(10, 0, &source_row).is_err());
    let mut noncanonical = wire;
    noncanonical[0][2][0][0] = json!(GOLDILOCKS_MODULUS);
    assert!(MatrixProgram::decode(&noncanonical).is_err());
}

#[test]
fn sparse_poseidon_inputs_check_stored_columns_before_normalizing() {
    let wire = json!([[
        [0, 2, 0, 2],
        [
            6,
            [[[1, 2], [1, GOLDILOCKS_MODULUS - 2]], [[2, 3]], [[3, 4]], [[4, 5]]],
            2
        ]
    ]]);
    let program = poseidon_input::Program::decode(&wire).expect("sparse Poseidon input");
    let first = program.state(8, 0, 0).unwrap();
    let second = program.state(8, 0, 1).unwrap();
    assert!(first[0].entries().is_empty());
    for (form, column, coefficient) in [(&first[1], 2, 3), (&second[0], 3, 4), (&second[1], 4, 5)] {
        assert_eq!(
            form.entries(),
            vec![Entry {
                column,
                coefficient: Goldilocks::from_u64(coefficient)
            }]
        );
    }
    assert!(first[2..].iter().all(|form| form.entries().is_empty()));

    let mut outside = wire.clone();
    outside[0][1][1][0]
        .as_array_mut()
        .unwrap()
        .push(json!([8, 0]));
    assert!(
        poseidon_input::Program::decode(&outside)
            .unwrap()
            .state(8, 0, 0)
            .is_err(),
        "an out-of-range zero coefficient must fail before normalization"
    );
    let mut short = wire.clone();
    short[0][1][1].as_array_mut().unwrap().pop();
    assert!(poseidon_input::Program::decode(&short)
        .unwrap()
        .state(8, 0, 1)
        .is_err());
    let mut noncanonical = wire;
    noncanonical[0][1][1][0][0][1] = json!(GOLDILOCKS_MODULUS);
    assert!(poseidon_input::Program::decode(&noncanonical).is_err());
}

#[test]
fn direct_phi81_challenges_preserve_centering_and_source_order() {
    let one = 5_999;
    let families = json!([[2, 1, 1]]);
    let input = json!([[[0, 108, [0, 108, 4_216], 0]], []]);
    let output = json!([0, 108, 4_450]);
    let quotient = json!([0, 108, 4_600]);
    let retained = json!([[3, [families, one, [0, 108, 4_000], 0, 54, input, output, quotient]]]);
    let forms = (0..108)
        .map(|index| json!([[4_000 + index, 1], [one, GOLDILOCKS_MODULUS - 2]]))
        .collect::<Vec<_>>();
    let direct = json!([[3, [families, one, forms, 54, input, output, quotient]]]);
    let before = MatrixProgram::decode(&retained).unwrap();
    let after = MatrixProgram::decode(&direct).unwrap();
    assert_eq!(before.row_count().unwrap(), 216);
    assert_eq!(after.row_count().unwrap(), 216);
    for row in 0..216 {
        let before = before.row(6_000, row, &source_row).unwrap();
        let after = after.row(6_000, row, &source_row).unwrap();
        for port in 0..MEANINGFUL_PORTS {
            assert_eq!(before[port].entries(), after[port].entries(), "row {row}, port {port}");
        }
    }
    let mut outside = direct.clone();
    outside[0][1][2][0]
        .as_array_mut()
        .unwrap()
        .push(json!([6_000, 0]));
    assert!(MatrixProgram::decode(&outside)
        .unwrap()
        .row(6_000, 0, &source_row)
        .is_err());
    let mut short = direct.clone();
    short[0][1][2].as_array_mut().unwrap().pop();
    assert!(MatrixProgram::decode(&short)
        .unwrap()
        .row(6_000, 108, &source_row)
        .is_err());
    let mut noncanonical = direct;
    noncanonical[0][1][2][0][0][1] = json!(GOLDILOCKS_MODULUS);
    assert!(MatrixProgram::decode(&noncanonical).is_err());
}

fn map_program(program: &Value, width: usize, projection: &Value) -> Value {
    Value::Array(
        program
            .as_array()
            .unwrap()
            .iter()
            .map(|block| json!([5, width, projection, block]))
            .collect(),
    )
}

#[test]
fn mapped_blocks_preserve_all_ports_and_external_source_indices() {
    let original = encoded_program();
    let mapped = map_program(&original, 6_000, &json!([1, [[0, 10, 6_000]]]));
    let nested = map_program(&mapped, 6_010, &json!([1, [[10, 3, 6_000]]]));
    let before = MatrixProgram::decode(&original).unwrap();
    let after = MatrixProgram::decode(&mapped).unwrap();
    let nested = MatrixProgram::decode(&nested).unwrap();
    after.validate(1).unwrap();
    nested.validate(1).unwrap();
    let count = before.row_count().unwrap();
    assert_eq!(after.row_count().unwrap(), count);
    for row in 0..count {
        let source = before.row(6_000, row, &source_row).unwrap();
        for (program, width, shift) in [(&after, 6_010, 10), (&nested, 6_003, 3)] {
            let actual = program.row(width, row, &source_row).unwrap();
            for port in 0..MEANINGFUL_PORTS {
                let expected = source[port]
                    .entries()
                    .into_iter()
                    .map(|entry| Entry {
                        column: entry.column + shift,
                        ..entry
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual[port].entries(),
                    expected,
                    "row {row}, port {port}, shift {shift}"
                );
            }
        }
    }
    assert!(
        after.row(6_009, 0, &source_row).is_err(),
        "target width excludes the moved one column"
    );
    let invalid_source = map_program(&original, 5_999, &json!([0]));
    assert!(MatrixProgram::decode(&invalid_source).is_err());
}

#[test]
fn mapped_retained_blocks_reject_interior_gaps_overlaps_and_noncontiguous_images() {
    let original = json!([[2, [1, 3_999, [2, 86, 100], []]]]);
    // The first and last coordinate of the retained operand are live in all
    // three cases. An endpoint-only test would miss the invalid interior.
    for ranges in [
        json!([[0, 0, 200], [201, 201, 3_799]]),
        json!([[0, 0, 4_000], [200, 200, 1]]),
        json!([[0, 0, 200], [200, 201, 3_800]]),
    ] {
        assert!(MatrixProgram::decode(&map_program(&original, 4_000, &json!([1, ranges]))).is_err());
    }
    let adjacent = json!([1, [[0, 10, 200], [200, 210, 3_800]]]);
    assert!(MatrixProgram::decode(&map_program(&original, 4_000, &adjacent)).is_ok());
}

#[test]
fn mapped_pins_reject_removed_reads_before_cancellation() {
    let projection = json!([1, [[0, 0, 5], [6, 6, 2]]]);
    for entries in [json!([[5, 0]]), json!([[5, 1], [5, GOLDILOCKS_MODULUS - 1]])] {
        let program = json!([[1, [0, [entries]]]]);
        assert!(MatrixProgram::decode(&map_program(&program, 8, &projection)).is_err());
    }
}

#[test]
#[ignore = "Run tools/recursive-constraint-minimizer/experiments/check_wide_matrix_reader.sh with Lean-emitted operands"]
fn wide_candidate_matrix_operands_decode() {
    use std::io::Read;
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let operands: Value = serde_json::from_str(&input).expect("Lean-emitted matrix operands");
    let width = operands[3].as_u64().unwrap() as usize;
    let rows = operands[4].as_u64().unwrap() as usize;
    let mut program = MatrixProgram::decode(&operands[1]).expect("complete candidate matrix program");
    assert_eq!(program.row_count().unwrap(), rows);
    let target = ColumnProjection::new(width, SourceProjection::Identity);
    for block in &mut program.blocks {
        target
            .apply(block)
            .expect("candidate operand fits its committed width");
    }
    println!(
        "wide matrix operands: {} blocks, {rows} rows, {width} logical coordinates",
        program.blocks.len()
    );
}
