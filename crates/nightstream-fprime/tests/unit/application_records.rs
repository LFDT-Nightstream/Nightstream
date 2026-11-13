use super::*;
use std::sync::Arc;

fn header() -> ApplicationRowHeader {
    ApplicationRowHeader {
        constants: [5, 1, 5],
        term_counts: [3, 0, 0],
    }
}

fn terms() -> [ApplicationTerm; 3] {
    [3, Fq::ORDER_U64 - 3, 0].map(|coefficient| ApplicationTerm {
        form: ApplicationForm::A,
        variable: 1,
        coefficient,
    })
}

fn nodes() -> [ApplicationRecipeNode; 5] {
    use ApplicationRecipeNode::*;
    [Add, Constant(5), Multiply, Constant(0), Variable(1)]
}

#[test]
fn ordered_terms_recipes_and_framing_counts_survive_sealing() {
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    assert_eq!(
        writer
            .append_row(header(), terms().into_iter().map(Ok))
            .unwrap(),
        0
    );
    assert_eq!(
        writer
            .append_recipe(0, nodes().into_iter().map(Ok))
            .unwrap(),
        0
    );
    let records = Arc::new(writer.finish().unwrap());
    assert_eq!(records.row_header(0).unwrap(), header());
    assert_eq!(records.row_count(), 1);
    assert_eq!(records.recipe_count(), 1);
    assert_eq!(records.recipe_row(0).unwrap(), 0);
    assert_eq!(records.row_preimage_word_count(), 44 + 12 * terms().len());
    assert_eq!(records.recipe_preimage_word_count(), 3 * 12 + 2 * 8);
    let mut actual = Vec::new();
    assert!(records
        .visit_terms(0, |term| {
            actual.push(term);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap()
        .is_continue());
    assert_eq!(actual, terms());
    let mut actual = Vec::new();
    assert!(records
        .visit_recipe_nodes(0, |node| {
            actual.push(node);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap()
        .is_continue());
    assert_eq!(actual, nodes());
    assert_eq!(
        records
            .evaluate_form::<PackageError>(0, ApplicationForm::A, |_| Ok(7))
            .unwrap(),
        5
    );
    assert_eq!(records.evaluate_recipe(0, |_| Ok(7)).unwrap(), 5);
    let clone = Arc::clone(&records);
    assert!(Arc::ptr_eq(&records, &clone));
    std::thread::scope(|scope| {
        let read = scope.spawn(|| records.evaluate_recipe(0, |_| Ok(11)).unwrap());
        assert_eq!(
            clone
                .evaluate_form::<PackageError>(0, ApplicationForm::A, |_| Ok(9))
                .unwrap(),
            5
        );
        assert_eq!(read.join().unwrap(), 5);
    });
    let mut calls = 0;
    assert!(records
        .visit_terms(0, |_| {
            calls += 1;
            Ok(ControlFlow::Break(()))
        })
        .unwrap()
        .is_break());
    assert_eq!(calls, 1);
    assert!(records.data.rejects_writes());
    assert!(records.rows.rejects_writes());
    assert!(records.recipes.rejects_writes());
    assert!(records.row_header(1).is_err());
    assert!(records.recipe_row(1).is_err());
}

#[test]
fn reused_recipe_evaluator_clears_pending_frames_after_lookup_error() {
    use ApplicationRecipeNode::*;

    let recipes = [
        vec![
            Add,
            Multiply,
            Variable(0),
            Add,
            Constant(2),
            Variable(1),
            Multiply,
            Add,
            Variable(0),
            Constant(1),
            Variable(2),
        ],
        vec![Constant(11)],
        vec![Variable(1)],
    ];
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    for nodes in recipes {
        let row = writer
            .append_row(
                ApplicationRowHeader {
                    constants: [0; 3],
                    term_counts: [0; 3],
                },
                [],
            )
            .unwrap();
        writer
            .append_recipe(row, nodes.into_iter().map(Ok))
            .unwrap();
    }
    let records = writer.finish().unwrap();
    let values = [3, 5, 7];
    let mut evaluator = RecipeEvaluator::new(&records).unwrap();
    for (recipe, expected) in [(0, 49), (1, 11), (2, 5), (0, 49)] {
        assert_eq!(
            evaluator
                .evaluate(recipe, |variable| Ok(values[variable]))
                .unwrap(),
            expected
        );
        assert_eq!(
            records
                .evaluate_recipe(recipe, |variable| Ok(values[variable]))
                .unwrap(),
            expected
        );
    }
    let mut looked_up = Vec::new();
    assert!(matches!(
        evaluator.evaluate(0, |variable| {
            looked_up.push(variable);
            if variable == 1 {
                Err(PackageError::Input("injected recipe lookup failure"))
            } else {
                Ok(values[variable])
            }
        }),
        Err(PackageError::Input("injected recipe lookup failure"))
    ));
    assert_eq!(looked_up, [0, 1]);
    assert_eq!(
        evaluator
            .evaluate(1, |_| panic!("constant has no lookup"))
            .unwrap(),
        11
    );
    assert_eq!(
        evaluator
            .evaluate(2, |variable| Ok(values[variable]))
            .unwrap(),
        5
    );
    assert_eq!(
        evaluator
            .evaluate(0, |variable| Ok(values[variable]))
            .unwrap(),
        49
    );
}

#[test]
fn walks_read_each_row_across_windows_in_any_order() {
    let rows = (storage::WINDOW_BYTES / size_of::<u64>() as u64) as usize + 1;
    let term = |form, variable, coefficient| {
        Ok(ApplicationTerm {
            form,
            variable,
            coefficient,
        })
    };
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    for row in 0..rows {
        writer
            .append_row(
                ApplicationRowHeader {
                    constants: [row as u64 % 7, 1, 0],
                    term_counts: [2, 1, 1],
                },
                [
                    term(ApplicationForm::A, row, 2),
                    term(ApplicationForm::A, row + 1, 3),
                    term(ApplicationForm::B, 0, 1),
                    term(ApplicationForm::C, row, 5),
                ],
            )
            .unwrap();
        writer
            .append_recipe(row, [Ok(ApplicationRecipeNode::Variable(row))])
            .unwrap();
    }
    let records = writer.finish().unwrap();
    let sides = |row: usize| {
        let row = row as u64;
        [row % 7 + 2 * (row + 1) + 3 * (row + 2), 2, 5 * (row + 1)]
    };
    let value = |variable: usize| Ok::<_, PackageError>(variable as u64 + 1);
    let mut walk = records.walk();
    for recipe in 0..rows {
        let row = walk.recipe_row(recipe).unwrap();
        assert_eq!(row, recipe);
        assert_eq!(walk.evaluate_row(row, value).unwrap(), sides(row));
    }
    let mut walk = records.walk();
    let stride = rows / 64;
    let backward = (0..rows).rev().step_by(stride);
    for row in backward.chain((0..rows).step_by(stride)) {
        assert_eq!(walk.recipe_row(row).unwrap(), row);
        assert_eq!(walk.evaluate_row(row, value).unwrap(), sides(row));
    }
    assert!(walk.recipe_row(rows).is_err());
    assert!(walk.evaluate_row(rows, value).is_err());
}

#[test]
fn first_unsatisfied_row_and_first_error_follow_row_order() {
    let rows = 50_000;
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    for row in 0..rows {
        let term = |form, variable| {
            Ok(ApplicationTerm {
                form,
                variable,
                coefficient: 1,
            })
        };
        writer
            .append_row(
                ApplicationRowHeader {
                    constants: [0, 1, 0],
                    term_counts: [1, 0, 1],
                },
                [term(ApplicationForm::A, row), term(ApplicationForm::C, rows + row)],
            )
            .unwrap();
    }
    let records = writer.finish().unwrap();
    let broken = [30_000, 40_000];
    let value = |missing: usize| {
        move |variable: usize| match variable.checked_sub(rows) {
            _ if variable == missing => Err(PackageError::Input("missing variable")),
            Some(row) if broken.contains(&row) => Ok(0),
            Some(row) => Ok(row as u64),
            None => Ok(variable as u64),
        }
    };
    let none = usize::MAX;
    assert_eq!(
        records
            .first_unsatisfied_row(|variable| Ok::<_, PackageError>((variable % rows) as u64))
            .unwrap(),
        None
    );
    assert_eq!(records.first_unsatisfied_row(value(none)).unwrap(), Some(30_000));
    assert_eq!(records.first_unsatisfied_row(value(45_000)).unwrap(), Some(30_000));
    assert!(matches!(
        records.first_unsatisfied_row(value(rows + 20_000)),
        Err(PackageError::Input("missing variable"))
    ));
}

#[test]
fn writer_buffers_do_not_grow_with_distinct_rows_or_terms() {
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    let bytes = writer.buffer_bytes();
    let rows = bytes / ((6 + 2) * size_of::<u64>()) + 1;
    for row in 0..rows {
        writer
            .append_row(
                ApplicationRowHeader {
                    constants: [row as u64, 1, 0],
                    term_counts: [1, 0, 0],
                },
                [Ok(ApplicationTerm {
                    form: ApplicationForm::A,
                    variable: row,
                    coefficient: 1,
                })],
            )
            .unwrap();
        assert_eq!(writer.buffer_bytes(), bytes);
    }
    let records = writer.finish().unwrap();
    assert_eq!(records.row_count(), rows);
    for row in [0, rows / 2, rows - 1] {
        assert_eq!(
            records
                .evaluate_form::<PackageError>(row, ApplicationForm::A, |variable| Ok(variable as u64))
                .unwrap(),
            2 * row as u64
        );
    }
}

#[test]
fn partial_append_poison_prevents_a_usable_snapshot() {
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    let mut stream = terms().into_iter().map(Ok).collect::<Vec<_>>();
    stream[1] = Err(PackageError::Io(std::io::Error::other("injected term read failure")));
    assert!(matches!(writer.append_row(header(), stream), Err(PackageError::Io(_))));
    assert_eq!(writer.row_count(), 0);
    assert!(writer
        .append_row(header(), terms().into_iter().map(Ok))
        .is_err());
    assert!(writer.finish().is_err());

    let mut writer = ApplicationRecordsWriter::new().unwrap();
    writer
        .append_row(header(), terms().into_iter().map(Ok))
        .unwrap();
    assert!(writer
        .append_recipe(
            0,
            [
                Ok(ApplicationRecipeNode::Add),
                Err(PackageError::Io(std::io::Error::other("injected recipe read failure")))
            ]
        )
        .is_err());
    assert_eq!(writer.row_count(), 1);
    assert_eq!(writer.recipe_count(), 0);
    assert!(writer.finish().is_err());
}

#[test]
fn sealing_rejects_a_truncated_record_file() {
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    writer
        .append_row(header(), terms().into_iter().map(Ok))
        .unwrap();
    writer.data.truncate_for_test().unwrap();
    assert!(writer.finish().is_err());
}

#[test]
fn ingestion_rejects_bad_counts_field_words_and_recipe_arity() {
    for invalid_terms in [
        Vec::new(),
        vec![ApplicationTerm {
            form: ApplicationForm::B,
            variable: 0,
            coefficient: 1,
        }],
    ] {
        let mut writer = ApplicationRecordsWriter::new().unwrap();
        assert!(writer
            .append_row(header(), invalid_terms.into_iter().map(Ok))
            .is_err());
        assert!(writer.finish().is_err());
    }
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    assert!(writer
        .append_row(
            ApplicationRowHeader {
                constants: [u64::MAX, 0, 0],
                term_counts: [0; 3]
            },
            []
        )
        .is_err());
    assert!(writer.finish().is_err());
    for nodes in [
        vec![],
        vec![ApplicationRecipeNode::Add, ApplicationRecipeNode::Constant(1)],
        vec![ApplicationRecipeNode::Constant(1), ApplicationRecipeNode::Constant(2)],
    ] {
        let mut writer = ApplicationRecordsWriter::new().unwrap();
        writer
            .append_row(header(), terms().into_iter().map(Ok))
            .unwrap();
        assert!(writer.append_recipe(0, nodes.into_iter().map(Ok)).is_err());
        assert!(writer.finish().is_err());
    }
}

fn serialized_records() -> (ApplicationRecords, Vec<u8>) {
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    writer
        .append_row(header(), terms().into_iter().map(Ok))
        .unwrap();
    writer
        .append_recipe(0, nodes().into_iter().map(Ok))
        .unwrap();
    let records = writer.finish().unwrap();
    let mut bytes = Vec::new();
    records.write_to(&mut bytes).unwrap();
    (records, bytes)
}

#[test]
fn streamed_records_preserve_exact_words_and_rebuild_sealed_indices() {
    let (original, bytes) = serialized_records();
    let row_and_recipe_counts = [1, 1];
    let interleaved_constants_and_term_counts = [5, 3, 1, 0, 5, 0];
    let ordered_terms = [1, 3, 1, Fq::ORDER_U64 - 3, 1, 0];
    let linked_row_and_node_count = [0, 5];
    let add_constant_times_variable = [2, 0, 1, 5, 3, 0, 1, 0, 0, 1];
    let words = [
        row_and_recipe_counts.as_slice(),
        &interleaved_constants_and_term_counts,
        &ordered_terms,
        &linked_row_and_node_count,
        &add_constant_times_variable,
    ]
    .concat();
    assert_eq!(
        bytes,
        words
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>()
    );
    let mut input = std::io::Cursor::new(&bytes);
    let restored = ApplicationRecords::read_from(&mut input, 1, 1).unwrap();
    assert_eq!(input.position(), bytes.len() as u64);
    assert_eq!(restored.row_header(0).unwrap(), header());
    assert_eq!(restored.recipe_row(0).unwrap(), 0);
    assert_eq!(restored.row_preimage_word_count(), original.row_preimage_word_count());
    assert_eq!(
        restored.recipe_preimage_word_count(),
        original.recipe_preimage_word_count()
    );
    assert_eq!(
        restored
            .evaluate_form::<PackageError>(0, ApplicationForm::A, |_| Ok(9))
            .unwrap(),
        5
    );
    assert_eq!(restored.evaluate_recipe(0, |_| Ok(9)).unwrap(), 5);
    assert!(restored.data.rejects_writes());
    assert!(restored.rows.rejects_writes());
    assert!(restored.recipes.rejects_writes());
    let mut encoded_again = Vec::new();
    restored.write_to(&mut encoded_again).unwrap();
    assert_eq!(encoded_again, bytes);
}

#[test]
fn streamed_records_rebuild_indices_after_interleaved_appends() {
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    for row in 0..2 {
        writer
            .append_row(
                ApplicationRowHeader {
                    constants: [row as u64, 1, 0],
                    term_counts: [1, 0, 0],
                },
                [Ok(ApplicationTerm {
                    form: ApplicationForm::A,
                    variable: row,
                    coefficient: 2,
                })],
            )
            .unwrap();
        writer
            .append_recipe(
                row,
                [
                    ApplicationRecipeNode::Add,
                    ApplicationRecipeNode::Variable(row),
                    ApplicationRecipeNode::Constant(row as u64),
                ]
                .into_iter()
                .map(Ok),
            )
            .unwrap();
    }
    let original = writer.finish().unwrap();
    let mut expected = Vec::new();
    original.write_to(&mut expected).unwrap();
    let restored = ApplicationRecords::read_from(&mut expected.as_slice(), 2, 2).unwrap();
    for row in 0..2 {
        assert_eq!(restored.row_header(row).unwrap(), original.row_header(row).unwrap());
        assert_eq!(restored.recipe_row(row).unwrap(), row);
        assert_eq!(
            restored
                .evaluate_form::<PackageError>(row, ApplicationForm::A, |variable| Ok(3 + variable as u64))
                .unwrap(),
            6 + 3 * row as u64,
        );
        assert_eq!(
            restored
                .evaluate_recipe(row, |variable| Ok(3 + variable as u64))
                .unwrap(),
            3 + 2 * row as u64
        );
    }
    let mut actual = Vec::new();
    restored.write_to(&mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn streamed_records_reject_bad_fields_counts_grammar_and_every_truncation() {
    let (_, bytes) = serialized_records();
    let modulus = Fq::ORDER_U64;
    for (case, word, value) in [
        ("row count differs from outer metadata", 0, 2),
        ("recipe count differs from outer metadata", 1, 2),
        ("noncanonical row constant", 2, modulus),
        ("term-count arithmetic overflows", 3, u64::MAX),
        ("noncanonical term coefficient", 9, modulus),
        ("linked row does not exist", 14, 1),
        ("missing right operand", 15, 4),
        ("unknown recipe tag", 16, 4),
        ("root leaf followed by extra nodes", 16, 1),
        ("binary node with nonzero payload", 17, 1),
        ("noncanonical recipe constant", 19, modulus),
    ] {
        let mut invalid = bytes.clone();
        invalid[word * 8..(word + 1) * 8].copy_from_slice(&value.to_le_bytes());
        assert!(
            ApplicationRecords::read_from(&mut invalid.as_slice(), 1, 1).is_err(),
            "{case}: word {word}"
        );
    }
    for end in 0..bytes.len() {
        assert!(
            matches!(
                ApplicationRecords::read_from(&mut &bytes[..end], 1, 1),
                Err(PackageError::Io(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof
            ),
            "truncated at byte {end}",
        );
    }
    let empty = ApplicationRecordsWriter::new().unwrap().finish().unwrap();
    let mut bytes = Vec::new();
    empty.write_to(&mut bytes).unwrap();
    let empty = ApplicationRecords::read_from(&mut bytes.as_slice(), 0, 0).unwrap();
    assert_eq!((empty.row_count(), empty.recipe_count()), (0, 0));
}

#[test]
fn records_propagate_write_failures() {
    struct FailedOutput;
    impl std::io::Write for FailedOutput {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("injected output failure"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let (records, _) = serialized_records();
    assert!(matches!(records.write_to(&mut FailedOutput), Err(PackageError::Io(_))));
}
