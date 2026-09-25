use super::*;
use std::sync::Arc;

fn header() -> ApplicationRowHeader {
    ApplicationRowHeader {
        constants: [5, 1, 5],
        term_counts: [3, 0, 0],
    }
}

fn terms() -> [ApplicationTerm; 3] {
    [3, crate::package::GOLDILOCKS_MODULUS - 3, 0].map(|coefficient| ApplicationTerm {
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
            .evaluate_form(0, ApplicationForm::A, |_| Ok(7))
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
                .evaluate_form(0, ApplicationForm::A, |_| Ok(9))
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

    // x * (2 + y) + (x + 1) * z exercises pending frames on both
    // branches. The following constant and variable recipes have no frames.
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
                Err(PackageError::Invalid("injected recipe lookup failure"))
            } else {
                Ok(values[variable])
            }
        }),
        Err(PackageError::Invalid("injected recipe lookup failure"))
    ));
    assert_eq!(looked_up, [0, 1]);
    // The failed lookup leaves Add/Multiply frames and stored left operands.
    // A leaf evaluated next must not consume any of those old frames.
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
fn writer_buffers_do_not_grow_with_distinct_rows_or_terms() {
    let mut writer = ApplicationRecordsWriter::new().unwrap();
    let bytes = writer.buffer_bytes();
    // Cross the actual standard-library buffer capacity; this is not a policy cap.
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
                .evaluate_form(row, ApplicationForm::A, |variable| Ok(variable as u64))
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
    let words = [
        1,
        1, // Row and recipe counts.
        5,
        3,
        1,
        0,
        5,
        0, // Interleaved affine constants and term counts.
        1,
        3,
        1,
        crate::package::GOLDILOCKS_MODULUS - 3,
        1,
        0, // Ordered terms retain cancellation and zero coefficients.
        0,
        5, // Linked row and recipe node count.
        2,
        0,
        1,
        5,
        3,
        0,
        1,
        0,
        0,
        1, // Add(Constant(5), Multiply(Constant(0), Variable(1))).
    ];
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
            .evaluate_form(0, ApplicationForm::A, |_| Ok(9))
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
    let snapshot = PrivateSnapshot::create(|output| original.write_to(output)).unwrap();
    let restored = snapshot
        .with_reader(|input| ApplicationRecords::read_from(input, 2, 2))
        .unwrap();
    for row in 0..2 {
        assert_eq!(restored.row_header(row).unwrap(), original.row_header(row).unwrap());
        assert_eq!(restored.recipe_row(row).unwrap(), row);
        assert_eq!(
            restored
                .evaluate_form(row, ApplicationForm::A, |variable| Ok(3 + variable as u64))
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
    let mut expected = Vec::new();
    snapshot.copy_to(&mut expected).unwrap();
    let mut actual = Vec::new();
    restored.write_to(&mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn streamed_records_reject_bad_fields_counts_grammar_and_every_truncation() {
    let (_, bytes) = serialized_records();
    let modulus = crate::package::GOLDILOCKS_MODULUS;
    for (word, value) in [
        (0, 2),        // Row count differs from outer metadata.
        (1, 2),        // Recipe count differs from outer metadata.
        (2, modulus),  // Noncanonical row constant.
        (3, u64::MAX), // Term-count arithmetic cannot fit.
        (9, modulus),  // Noncanonical term coefficient.
        (14, 1),       // Linked row does not exist.
        (15, 4),       // Missing right operand in the declared node stream.
        (16, 4),       // Unknown recipe tag.
        (16, 1),       // Root becomes a leaf followed by extra nodes.
        (17, 1),       // Binary nodes require zero payload.
        (19, modulus), // Noncanonical recipe constant.
    ] {
        let mut invalid = bytes.clone();
        invalid[word * 8..(word + 1) * 8].copy_from_slice(&value.to_le_bytes());
        assert!(
            ApplicationRecords::read_from(&mut invalid.as_slice(), 1, 1).is_err(),
            "word {word}"
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
fn private_snapshot_streams_bytes_and_propagates_write_failures() {
    struct FailedOutput;
    impl std::io::Write for FailedOutput {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("injected output failure"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let value = serde_json::json!([1, [2, 3], []]);
    let snapshot = PrivateSnapshot::create(|output| {
        serde_json::to_writer(output, &value)?;
        Ok(())
    })
    .unwrap();
    let expected = serde_json::to_vec(&value).unwrap();
    assert_eq!(snapshot.len(), expected.len() as u64);
    let mut actual = Vec::new();
    snapshot.copy_to(&mut actual).unwrap();
    assert_eq!(actual, expected);
    let decoded: serde_json::Value = snapshot
        .with_reader(|input| Ok(serde_json::from_reader(input)?))
        .unwrap();
    assert_eq!(decoded, value);
    assert!(matches!(snapshot.copy_to(&mut FailedOutput), Err(PackageError::Io(_))));
    let (records, _) = serialized_records();
    assert!(matches!(records.write_to(&mut FailedOutput), Err(PackageError::Io(_))));
    assert!(matches!(
        PrivateSnapshot::create(|output| {
            output.write_all(b"partial snapshot")?;
            Err(PackageError::Io(std::io::Error::other("injected snapshot failure")))
        }),
        Err(PackageError::Io(_))
    ));
}
