use super::*;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};

const FIRST_POSEIDON_CONSTANT: u64 = 15_504_881_536_434_223_753;

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
            [0, 54, 4_054],
            [0, 54, 4_108],
            [0, 54 * 33, 4_162]
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
    assert_eq!(program.row_count().expect("row count"), 1_933);

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

    assert!(entries(&program, poseidon_row + 86, 4).is_empty());

    let phi_row = poseidon_row + 94;
    assert_eq!(
        entries(&program, phi_row, 0),
        vec![(4_000, 1), (5_999, GOLDILOCKS_MODULUS - 2)]
    );
    assert_eq!(entries(&program, phi_row, 2), vec![(4_054, 1)]);
    assert_eq!(entries(&program, phi_row, 4), vec![(4_162, 1)]);
    assert_eq!(entries(&program, phi_row, 7), vec![(5_999, 1)]);

    let phi_final = entries(&program, phi_row + 33, 4);
    assert_eq!(phi_final.len(), 34);
    assert_eq!(phi_final[0], (4_108, 1));
    assert_eq!(phi_final[1], (4_162, GOLDILOCKS_MODULUS - 1));
    assert_eq!(phi_final[33], (4_162 + 32, GOLDILOCKS_MODULUS - 1));

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
    block
        .visit_rows(6_000, 0, row_count, |row| {
            visited.push(row);
            Ok(())
        })
        .expect("linear Poseidon2 rows");
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
