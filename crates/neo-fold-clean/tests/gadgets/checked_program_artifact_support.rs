//! Certifying normalizer for mixed definition/assertion R1CS artifacts.
//!
//! This test-only helper preserves exact row order. It classifies a row as an
//! SSA definition only when one fresh output is mechanically isolated from
//! already-known columns. Every other row remains an assertion and every
//! previously unseen column in that assertion becomes an explicit program
//! input. Assertions are never solved for prover-controlled values.

use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_math::F;
use p3_field::PrimeField64;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rhs {
    Linear(Vec<(usize, u64)>),
    Product(Vec<(usize, u64)>, Vec<(usize, u64)>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Definition {
    pub output: usize,
    pub rhs: Rhs,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub a: Vec<(usize, u64)>,
    pub b: Vec<(usize, u64)>,
    pub c: Vec<(usize, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instruction {
    Define(Definition),
    Check(Row),
}

#[derive(Clone, Debug)]
pub struct NormalizedProgram {
    pub instructions: Vec<Instruction>,
    pub input_columns: Vec<usize>,
    pub definition_count: usize,
    pub check_count: usize,
}

#[derive(Clone, Debug)]
pub struct CanonicalizedProgram {
    pub instructions: Vec<Instruction>,
    /// Dense local-column to original-column map. Local column zero is always
    /// the R1CS constant-one column.
    pub column_map: Vec<usize>,
}

/// Canonicalize a normalized program's column names by first occurrence.
/// Programs with the same sparse instruction shape become byte-equal even
/// when embedded at different global column offsets.
pub fn canonicalize_program(program: &NormalizedProgram) -> CanonicalizedProgram {
    fn canonical_column(
        global: usize,
        global_to_local: &mut HashMap<usize, usize>,
        column_map: &mut Vec<usize>,
    ) -> usize {
        if let Some(&local) = global_to_local.get(&global) {
            local
        } else {
            let local = column_map.len();
            global_to_local.insert(global, local);
            column_map.push(global);
            local
        }
    }
    fn canonical_terms(
        source: &[(usize, u64)],
        global_to_local: &mut HashMap<usize, usize>,
        column_map: &mut Vec<usize>,
    ) -> Vec<(usize, u64)> {
        source
            .iter()
            .map(|&(global, coefficient)| (canonical_column(global, global_to_local, column_map), coefficient))
            .collect()
    }
    let mut global_to_local = HashMap::new();
    global_to_local.insert(0usize, 0usize);
    let mut column_map = vec![0usize];
    let instructions = program
        .instructions
        .iter()
        .map(|instruction| match instruction {
            Instruction::Define(definition) => {
                let rhs = match &definition.rhs {
                    Rhs::Linear(source) => Rhs::Linear(canonical_terms(source, &mut global_to_local, &mut column_map)),
                    Rhs::Product(left, right) => Rhs::Product(
                        canonical_terms(left, &mut global_to_local, &mut column_map),
                        canonical_terms(right, &mut global_to_local, &mut column_map),
                    ),
                };
                Instruction::Define(Definition {
                    output: canonical_column(definition.output, &mut global_to_local, &mut column_map),
                    rhs,
                })
            }
            Instruction::Check(row) => Instruction::Check(Row {
                a: canonical_terms(&row.a, &mut global_to_local, &mut column_map),
                b: canonical_terms(&row.b, &mut global_to_local, &mut column_map),
                c: canonical_terms(&row.c, &mut global_to_local, &mut column_map),
            }),
        })
        .collect();
    CanonicalizedProgram {
        instructions,
        column_map,
    }
}

pub fn relabel_instructions(instructions: &[Instruction], column_map: &[usize]) -> Vec<Instruction> {
    let column = |local: usize| {
        *column_map
            .get(local)
            .unwrap_or_else(|| panic!("missing relabel entry for local column {local}"))
    };
    let terms = |source: &[(usize, u64)]| {
        source
            .iter()
            .map(|&(local, coefficient)| (column(local), coefficient))
            .collect::<Vec<_>>()
    };
    instructions
        .iter()
        .map(|instruction| match instruction {
            Instruction::Define(definition) => Instruction::Define(Definition {
                output: column(definition.output),
                rhs: match &definition.rhs {
                    Rhs::Linear(source) => Rhs::Linear(terms(source)),
                    Rhs::Product(left, right) => Rhs::Product(terms(left), terms(right)),
                },
            }),
            Instruction::Check(row) => Instruction::Check(Row {
                a: terms(&row.a),
                b: terms(&row.b),
                c: terms(&row.c),
            }),
        })
        .collect()
}

fn indexed_rows(trips: &[(usize, usize, F)], row_start: usize, row_end: usize) -> Vec<Vec<(usize, u64)>> {
    let mut rows = vec![Vec::new(); row_end - row_start];
    for &(row, column, coefficient) in trips {
        if row_start <= row && row < row_end {
            rows[row - row_start].push((column, coefficient.as_canonical_u64()));
        }
    }
    rows
}

/// Materialize the complete exact row range, including compact seeded Phi81
/// contributions to `A`. Omitting these blocks changes `A * z = C` into the
/// unrelated assertion `0 = C` in exported checked programs.
fn exact_rows(builder: &R1csBuilder, row_start: usize, row_end: usize) -> Vec<Row> {
    let (a, b, c) = builder.sparse_triplets();
    let mut a_rows = indexed_rows(a, row_start, row_end);
    let b_rows = indexed_rows(b, row_start, row_end);
    let c_rows = indexed_rows(c, row_start, row_end);
    for block in builder.seeded_phi81_a_blocks() {
        if block.row_end() <= row_start || row_end <= block.row_start() {
            continue;
        }
        block.for_each_term::<F, _>(|row, column, coefficient| {
            if row_start <= row && row < row_end {
                a_rows[row - row_start].push((column, coefficient.as_canonical_u64()));
            }
        });
    }
    let rows = a_rows
        .into_iter()
        .zip(b_rows)
        .zip(c_rows)
        .map(|((a, b), c)| Row { a, b, c })
        .collect();
    rows
}

fn canonical_neg(coefficient: u64) -> u64 {
    if coefficient == 0 {
        0
    } else {
        F::ORDER_U64 - coefficient
    }
}

fn refs(row: &Row) -> impl Iterator<Item = usize> + '_ {
    row.a.iter().chain(&row.b).chain(&row.c).map(|term| term.0)
}

fn definition_refs(definition: &Definition) -> Box<dyn Iterator<Item = usize> + '_> {
    match &definition.rhs {
        Rhs::Linear(terms) => Box::new(terms.iter().map(|term| term.0)),
        Rhs::Product(left, right) => Box::new(left.iter().chain(right).map(|term| term.0)),
    }
}

fn row_of_definition(definition: &Definition) -> Row {
    match &definition.rhs {
        Rhs::Linear(terms) => Row {
            a: std::iter::once((definition.output, 1))
                .chain(
                    terms
                        .iter()
                        .map(|&(column, coefficient)| (column, canonical_neg(coefficient))),
                )
                .collect(),
            b: vec![(0, 1)],
            c: Vec::new(),
        },
        Rhs::Product(left, right) => Row {
            a: left.clone(),
            b: right.clone(),
            c: vec![(definition.output, 1)],
        },
    }
}

fn product_definition(row: &Row, known: &[bool]) -> Option<Definition> {
    let [(output, coefficient)] = row.c.as_slice() else {
        return None;
    };
    if *coefficient != 1 || known[*output] {
        return None;
    }
    if row.a.iter().chain(&row.b).all(|term| known[term.0]) {
        Some(Definition {
            output: *output,
            rhs: Rhs::Product(row.a.clone(), row.b.clone()),
        })
    } else {
        None
    }
}

fn linear_definition(row: &Row, known: &[bool]) -> Option<Definition> {
    if row.b != [(0, 1)] || !row.c.is_empty() {
        return None;
    }
    let (&(output, output_coefficient), negated_rhs) = row.a.split_first()?;
    if output_coefficient != 1 || known[output] || negated_rhs.iter().any(|term| !known[term.0]) {
        return None;
    }
    Some(Definition {
        output,
        rhs: Rhs::Linear(
            negated_rhs
                .iter()
                .map(|&(column, coefficient)| (column, canonical_neg(coefficient)))
                .collect(),
        ),
    })
}

pub fn normalize_with_inputs(builder: &R1csBuilder, declared_inputs: &[usize]) -> NormalizedProgram {
    let rows = exact_rows(builder, 0, builder.rows());
    let mut known = vec![false; builder.cols()];
    let mut is_input = vec![false; builder.cols()];
    for &column in declared_inputs {
        assert!(
            column < builder.cols(),
            "declared input column {column} is out of range"
        );
        known[column] = true;
        is_input[column] = true;
    }
    assert!(known[0], "the constant-one column must be a declared input");
    let mut instructions = Vec::with_capacity(builder.rows());
    let mut definition_count = 0;
    let mut check_count = 0;

    for (row_index, row) in rows.into_iter().enumerate() {
        let candidate = product_definition(&row, &known).or_else(|| linear_definition(&row, &known));
        match candidate {
            Some(definition) => {
                assert_eq!(
                    row_of_definition(&definition),
                    row,
                    "row {row_index} definition normalization changed exact sparse order"
                );
                assert!(
                    definition_refs(&definition).all(|column| known[column]),
                    "row {row_index} definition reads an unknown column"
                );
                known[definition.output] = true;
                definition_count += 1;
                instructions.push(Instruction::Define(definition));
            }
            None => {
                for column in refs(&row) {
                    if !known[column] {
                        known[column] = true;
                        is_input[column] = true;
                    }
                }
                check_count += 1;
                instructions.push(Instruction::Check(row));
            }
        }
    }

    let input_columns = is_input
        .iter()
        .enumerate()
        .filter_map(|(column, input)| input.then_some(column))
        .collect();
    NormalizedProgram {
        instructions,
        input_columns,
        definition_count,
        check_count,
    }
}

pub fn normalize(builder: &R1csBuilder) -> NormalizedProgram {
    normalize_with_inputs(builder, &[0])
}

/// Normalize an exact row prefix without inspecting later owners in a
/// composed builder. The prefix must begin at row zero, so its SSA/input
/// classification is identical to normalizing that owner in isolation.
pub fn normalize_prefix(builder: &R1csBuilder, row_end: usize) -> NormalizedProgram {
    assert!(row_end <= builder.rows(), "program prefix exceeds builder rows");
    let rows = exact_rows(builder, 0, row_end);
    let mut known = vec![false; builder.cols()];
    let mut is_input = vec![false; builder.cols()];
    known[0] = true;
    is_input[0] = true;
    let mut instructions = Vec::with_capacity(row_end);
    let mut definition_count = 0;
    let mut check_count = 0;

    for (row_index, row) in rows.into_iter().enumerate() {
        let candidate = product_definition(&row, &known).or_else(|| linear_definition(&row, &known));
        match candidate {
            Some(definition) => {
                assert_eq!(
                    row_of_definition(&definition),
                    row,
                    "row {row_index} definition normalization changed exact sparse order"
                );
                assert!(
                    definition_refs(&definition).all(|column| known[column]),
                    "row {row_index} definition reads an unknown column"
                );
                known[definition.output] = true;
                definition_count += 1;
                instructions.push(Instruction::Define(definition));
            }
            None => {
                for column in refs(&row) {
                    if !known[column] {
                        known[column] = true;
                        is_input[column] = true;
                    }
                }
                check_count += 1;
                instructions.push(Instruction::Check(row));
            }
        }
    }

    let input_columns = is_input
        .iter()
        .enumerate()
        .filter_map(|(column, input)| input.then_some(column))
        .collect();
    NormalizedProgram {
        instructions,
        input_columns,
        definition_count,
        check_count,
    }
}

/// Normalize one exact contiguous owner in a composed builder. Columns below
/// `first_allocated_column` are pre-existing inputs; only the ones referenced
/// by this owner are retained in `input_columns`. Columns first encountered in
/// a non-definitional assertion are adversarial inputs, exactly as in
/// [`normalize_with_inputs`].
pub fn normalize_range(
    builder: &R1csBuilder,
    row_start: usize,
    row_end: usize,
    first_allocated_column: usize,
) -> NormalizedProgram {
    assert!(row_start <= row_end, "program range is reversed");
    assert!(row_end <= builder.rows(), "program range exceeds builder rows");
    assert!(
        first_allocated_column <= builder.cols(),
        "first allocated column exceeds builder columns"
    );
    let row_count = row_end - row_start;
    let rows = exact_rows(builder, row_start, row_end);
    let mut known = vec![false; builder.cols()];
    let mut is_input = vec![false; builder.cols()];
    known[..first_allocated_column].fill(true);
    known[0] = true;
    let mut instructions = Vec::with_capacity(row_count);
    let mut definition_count = 0;
    let mut check_count = 0;

    for (local_row, row) in rows.into_iter().enumerate() {
        for column in refs(&row) {
            if column < first_allocated_column {
                is_input[column] = true;
            }
        }
        let candidate = product_definition(&row, &known).or_else(|| linear_definition(&row, &known));
        match candidate {
            Some(definition) => {
                assert_eq!(
                    row_of_definition(&definition),
                    row,
                    "row {} definition normalization changed exact sparse order",
                    row_start + local_row
                );
                assert!(
                    definition_refs(&definition).all(|column| known[column]),
                    "row {} definition reads an unknown column",
                    row_start + local_row
                );
                known[definition.output] = true;
                definition_count += 1;
                instructions.push(Instruction::Define(definition));
            }
            None => {
                for column in refs(&row) {
                    if !known[column] {
                        known[column] = true;
                        is_input[column] = true;
                    }
                }
                check_count += 1;
                instructions.push(Instruction::Check(row));
            }
        }
    }

    let input_columns = is_input
        .iter()
        .enumerate()
        .filter_map(|(column, input)| input.then_some(column))
        .collect();
    NormalizedProgram {
        instructions,
        input_columns,
        definition_count,
        check_count,
    }
}

fn lean_terms(terms: &[(usize, u64)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|&(column, coefficient)| format!("({column}, {coefficient})"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub fn lean_instruction(instruction: &Instruction) -> String {
    match instruction {
        Instruction::Define(definition) => match &definition.rhs {
            Rhs::Linear(terms) => format!(".define ⟨{}, .linear {}⟩", definition.output, lean_terms(terms)),
            Rhs::Product(left, right) => format!(
                ".define ⟨{}, .product {} {}⟩",
                definition.output,
                lean_terms(left),
                lean_terms(right)
            ),
        },
        Instruction::Check(row) => format!(
            ".check ⟨{}, {}, {}⟩",
            lean_terms(&row.a),
            lean_terms(&row.b),
            lean_terms(&row.c)
        ),
    }
}

pub fn lean_instructions(instructions: &[Instruction]) -> String {
    instructions
        .iter()
        .map(lean_instruction)
        .collect::<Vec<_>>()
        .join(",\n   ")
}
