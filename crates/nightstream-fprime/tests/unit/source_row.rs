use super::*;
use crate::package::{TemplateTerm, GOLDILOCKS_MODULUS};
use crate::sparse::SparseTerm;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn combination(constant: u64, terms: Vec<(ColumnRef, u64)>) -> TemplateCombination {
    TemplateCombination {
        constant: Goldilocks::from_u64(constant),
        terms: terms
            .into_iter()
            .map(|(column, coefficient)| TemplateTerm {
                column,
                coefficient: Goldilocks::from_u64(coefficient),
            })
            .collect(),
    }
}

fn sparse(constant: u64, terms: &[(usize, u64)]) -> SparseCombination {
    SparseCombination {
        constant: Goldilocks::from_u64(constant),
        terms: terms
            .iter()
            .map(|&(column, coefficient)| SparseTerm {
                column,
                coefficient: Goldilocks::from_u64(coefficient),
            })
            .collect(),
    }
}

fn permutation() -> PermutationTemplate {
    PermutationTemplate {
        input_count: 2,
        local_column_count: 3,
        output_local_start: 1,
        rows: vec![],
    }
}

fn template_row() -> TemplateRow {
    TemplateRow {
        output_local: 0,
        a: combination(
            10,
            vec![
                (ColumnRef::Input(0), 2),
                (ColumnRef::Input(1), 3),
                (ColumnRef::Local(0), 4),
            ],
        ),
        b: combination(0, vec![]),
        c: combination(0, vec![(ColumnRef::Local(0), 1)]),
    }
}

fn hash_chain() -> HashChain {
    HashChain {
        phase: 1,
        row_start: 0,
        row_count: 3,
        input_start: 20,
        input_length: 8,
        witness_start: 100,
        witness_length: 9,
        absorb_count: 2,
        digest_length: 4,
        digest_start: 200,
    }
}

fn term_words(combination: &SourceCombination) -> Vec<(usize, u64)> {
    combination
        .terms
        .iter()
        .map(|term| (term.column, term.coefficient.as_canonical_u64()))
        .collect()
}

#[test]
fn hash_template_rows_use_exact_absorb_and_padding_inputs() {
    let permutation = permutation();
    let chain = hash_chain();
    let first = source_template_row(
        &permutation,
        &template_row(),
        ScheduledInvocation::Hash {
            chain,
            ordinal: 0,
            row_start: 0,
            witness_start: 100,
        },
    )
    .expect("first hash row");
    assert_eq!(first.a.constant.as_canonical_u64(), 10);
    assert_eq!(term_words(&first.a), vec![(20, 2), (21, 3), (100, 4)]);

    let middle = invocation_input(
        &permutation,
        ScheduledInvocation::Hash {
            chain,
            ordinal: 1,
            row_start: 1,
            witness_start: 103,
        },
        0,
    )
    .expect("middle hash input");
    assert_eq!(middle.constant, Goldilocks::ZERO);
    assert_eq!(term_words(&middle), vec![(101, 1), (24, 1)]);

    let padded = invocation_input(
        &permutation,
        ScheduledInvocation::Hash {
            chain,
            ordinal: 2,
            row_start: 2,
            witness_start: 106,
        },
        0,
    )
    .expect("padding hash input");
    assert_eq!(padded.constant, Goldilocks::ONE);
    assert_eq!(term_words(&padded), vec![(104, 1)]);
    assert!(matches!(
        invocation_input(
            &permutation,
            ScheduledInvocation::Hash {
                chain,
                ordinal: 0,
                row_start: 0,
                witness_start: 100,
            },
            2,
        ),
        Err(PackageError::Invalid("matrix source hash input"))
    ));
}

#[test]
fn explicit_template_rows_preserve_input_and_local_term_order() {
    let permutation = permutation();
    let explicit = PermutationInvocation {
        phase: 2,
        row_start: 7,
        witness_start: 110,
        inputs: vec![sparse(5, &[(7, 6)]), sparse(1, &[(8, 9)])],
    };
    let row = source_template_row(&permutation, &template_row(), ScheduledInvocation::Explicit(&explicit))
        .expect("explicit permutation row");
    assert_eq!(row.a.constant.as_canonical_u64(), 23);
    assert_eq!(term_words(&row.a), vec![(7, 12), (8, 27), (110, 4)]);
    assert_eq!(term_words(&row.c), vec![(110, 1)]);
}

#[test]
fn assertion_and_generic_witness_rows_are_independent_of_matrix_expansion() {
    let assertion = SparseRow {
        row_index: 3,
        a: sparse(2, &[(4, 5)]),
        b: sparse(6, &[(7, 8)]),
        c: sparse(9, &[(10, 11)]),
    };
    let decoded_assertion = source_assertion(&assertion);
    assert_eq!(decoded_assertion.a.constant.as_canonical_u64(), 2);
    assert_eq!(term_words(&decoded_assertion.a), vec![(4, 5)]);
    assert_eq!(term_words(&decoded_assertion.c), vec![(10, 11)]);

    let witness = WitnessInstruction {
        row_index: 4,
        target: 12,
        a: sparse(13, &[(14, 15)]),
        b: sparse(16, &[(17, 18)]),
    };
    let decoded_witness = source_witness(&witness);
    assert_eq!(decoded_witness.c.constant, Goldilocks::ZERO);
    assert_eq!(term_words(&decoded_witness.c), vec![(12, 1)]);

    let mut selected = Some(decoded_assertion);
    assert!(matches!(
        merge_row(&mut selected, decoded_witness),
        Err(PackageError::Invalid("duplicate matrix source row"))
    ));
    assert!(GOLDILOCKS_MODULUS > 18);
}
