//! A relation that the public builders of `nightstream::relation` make,
//! proved through `nebula::R1csApplication` as the Starstream interleaving
//! relation is (SPEC.md section 7.6). It is a counter: each row reads
//! `table[count]` from a ROM, writes it to a RAM at `count`, and adds one to
//! `count`. A step holds one row or two. A second form of the counter has a
//! flag and two columns that each group of two rows shares.

use std::num::NonZeroUsize;

use nightstream::nebula::{
    batched_slots, pack_regions, Application, CatalogBindingError, Circuit, Geometry, R1csApplication,
    R1csApplicationError, R1csStepError, RowRelation, Run, SegmentError, Shared, Size, VerifyError,
};
use nightstream::relation::{
    ColumnFamilySpec, ColumnRegistry, ColumnWidth, ConstraintTag, ContinuityCatalog, ContinuityGroup, ContinuityLink,
    MemoryCatalog, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemoryPreload, MemorySpec,
    R1csBuilder, R1csCheckError, R1csRelation,
};
use nightstream::{Goldilocks as F, PrimeCharacteristicRing};

const ONE: usize = 0;
const BEFORE: usize = 1;
const AFTER: usize = 2;
const VALUE: usize = 3;
const COLUMNS: usize = 4;
const TABLE: [u32; 4] = [5, 6, 7, 8];

fn registry() -> ColumnRegistry {
    ColumnRegistry::new([ColumnFamilySpec {
        region: "counter",
        start: 0,
        len: COLUMNS,
        name: "words".into(),
        role: "test",
        width: ColumnWidth::U32,
    }])
    .unwrap()
}

/// `count_after = count_before + 1`.
fn relation() -> R1csRelation<&'static str> {
    let mut builder = R1csBuilder::new(1, ONE).unwrap();
    builder.with_tag(ConstraintTag::new("count", "counter"), |rows| {
        rows.push_linear_zero([(AFTER, F::ONE), (BEFORE, -F::ONE), (ONE, -F::ONE)]);
    });
    builder.build(&registry()).unwrap()
}

/// Memory 0 is the table ROM, memory 1 the RAM. Both ports are always active.
fn catalog() -> MemoryCatalog<u32> {
    let port = |kind| MemoryPortSpec {
        address_columns: vec![BEFORE],
        value_column: VALUE,
        kind,
        activation: MemoryPortActivation::Always,
    };
    MemoryCatalog::new(
        [
            MemorySpec {
                id: 0,
                kind: MemoryKind::Rom,
                ports: vec![port(MemoryPortKind::Read)],
            },
            MemorySpec {
                id: 1,
                kind: MemoryKind::Ram,
                ports: vec![port(MemoryPortKind::Write {
                    value_before_column: None,
                })],
            },
        ],
        &registry(),
    )
    .unwrap()
}

/// ROM and RAM of four cells, two slots for each row, one segment of
/// `4 / rows` steps.
fn geometry(rows: usize) -> Geometry {
    Geometry::new(2, 2, 2 * rows, 4 / rows, 1).unwrap()
}

/// The count of a row is the count before the next row.
fn continuity() -> ContinuityCatalog {
    ContinuityCatalog::new(
        [ContinuityGroup {
            name: "count",
            role: "test",
            links: vec![ContinuityLink {
                previous_step_column: AFTER,
                next_step_column: BEFORE,
            }],
        }],
        &registry(),
    )
    .unwrap()
}

fn application(rows: usize) -> R1csApplication<u32> {
    application_of(RowRelation::of(&relation()).unwrap(), rows).unwrap()
}

/// The application of `relation` with the counter's continuity and catalog.
fn application_of(relation: RowRelation, rows: usize) -> Result<R1csApplication<u32>, R1csApplicationError> {
    let rows = NonZeroUsize::new(rows).unwrap();
    let catalog = catalog();
    let (regions, used) = pack_regions(
        catalog
            .entries()
            .iter()
            .map(|memory| (memory.id, memory.kind, vec![2])),
    )
    .unwrap();
    assert_eq!(used, [4, 4], "four ROM cells and four RAM cells");
    let size = Size {
        geometry: geometry(rows.get()),
        rows,
        regions,
        slots: batched_slots(&catalog, &[vec![0], vec![1]], rows).unwrap(),
    };
    R1csApplication::new(relation, &continuity(), &catalog, size)
}

fn preload(table: [u32; 4]) -> MemoryPreload<u32> {
    let mut preload = MemoryPreload::default();
    for (address, value) in (0..).zip(table) {
        preload.insert(0, vec![address], value);
    }
    preload
}

/// The honest rows: count `c` reads and writes `TABLE[c]`.
fn rows() -> Vec<Vec<F>> {
    (0..4u64)
        .map(|count| {
            vec![
                F::ONE,
                F::from_u64(count),
                F::from_u64(count + 1),
                F::from_u32(TABLE[count as usize]),
            ]
        })
        .collect()
}

/// The step inputs of `rows` rows each.
fn inputs(rows: usize) -> Vec<Vec<F>> {
    self::rows().chunks(rows).map(<[Vec<F>]>::concat).collect()
}

#[test]
fn check_rows_names_the_row_that_fails() {
    let relation = relation();
    for z in rows() {
        assert_eq!(relation.check_rows(&z), Ok(()));
    }
    let mut z = rows()[0].clone();
    z[AFTER] += F::ONE;
    assert_eq!(relation.check_rows(&z), Err(R1csCheckError::Row { row: 0 }));
    assert_eq!(
        relation.check_rows(&z[1..]),
        Err(R1csCheckError::Width {
            expected: COLUMNS,
            got: COLUMNS - 1
        })
    );
}

/// `Geometry::new` derives the timestamp width and the scan slots, and refuses
/// a geometry that breaks SPEC.md section 7.1.
#[test]
fn a_geometry_derives_its_widths_and_keeps_the_rules() {
    assert_eq!(
        geometry(2),
        Geometry {
            r: 2,
            mu: 2,
            w_ts: 4,
            b_ops: 4,
            b_scan: 4,
            n: 2,
            s_max: 1,
        }
    );
    assert_eq!(
        Geometry::new(3, 2, 2, 4, 1),
        Some(Geometry {
            r: 3,
            mu: 2,
            w_ts: 4,
            b_ops: 2,
            b_scan: 3,
            n: 4,
            s_max: 1,
        }),
        "rule 4: the RAM can be smaller than the ROM"
    );
    assert_eq!(Geometry::new(2, 2, 2, 3, 1), None, "rule 1: n divides the cells");
    assert_eq!(Geometry::new(2, 2, 2, 16, 1), None, "rule 1: each space has n cells");
    assert_eq!(
        Geometry::new(16, 16, 168, 64, 1),
        None,
        "rule 6: 131,072 cells and 64 steps of 168 slots exceed 139,509"
    );
    assert_eq!(Geometry::new(2, 2, 2, 0, 1), None, "a segment has steps");
    assert_eq!(Geometry::new(64, 64, 2, 4, 1), None, "the spaces fit in a word");
}

/// A relation with a verifier-supplied column other than the constant-one
/// column is refused, because no memory verifier supplies that column, so
/// the column would be a free word of each row.
#[test]
fn a_relation_with_another_public_input_is_refused() {
    let mut builder = R1csBuilder::new(2, ONE).unwrap();
    builder.with_tag(ConstraintTag::new("count", "counter"), |rows| {
        rows.push_linear_zero([(AFTER, F::ONE), (BEFORE, -F::ONE), (ONE, -F::ONE)]);
    });
    let relation = builder.build(&registry()).unwrap();
    assert!(matches!(
        RowRelation::of(&relation),
        Err(R1csApplicationError::PublicInput)
    ));
    assert!(
        RowRelation::of(&self::relation()).is_ok(),
        "the constant-one column alone is public"
    );
}

/// A preload cell of a memory that the catalog does not have is refused, so
/// two preloads never give one program.
#[test]
fn a_preload_of_an_unknown_memory_is_refused() {
    let mut unknown = preload(TABLE);
    unknown.insert(7, vec![0], 1);
    let app = application(1);
    assert!(matches!(app.program(&unknown), Err(CatalogBindingError::UnknownMemory)));
    assert!(app.program(&preload(TABLE)).is_ok());
}

/// A size that gives the RAM no region is refused, because the circuit would
/// not check the RAM.
#[test]
fn a_size_must_give_each_memory_a_region() {
    let rows = NonZeroUsize::MIN;
    let catalog = catalog();
    let (mut regions, _) = pack_regions(
        catalog
            .entries()
            .iter()
            .map(|memory| (memory.id, memory.kind, vec![2])),
    )
    .unwrap();
    regions.retain(|(id, _)| *id == 0);
    let size = Size {
        geometry: geometry(1),
        rows,
        regions,
        slots: batched_slots(&catalog, &[vec![0], vec![]], rows).unwrap(),
    };
    assert!(matches!(
        R1csApplication::new(RowRelation::of(&relation()).unwrap(), &continuity(), &catalog, size),
        Err(R1csApplicationError::Unchecked)
    ));
}

/// The honest rows run natively in steps of one row and of two rows: the
/// count reaches 4 after one access to each memory for each row. A row that
/// reads another value than the ROM holds has no step.
#[test]
fn a_relation_of_the_builders_runs_through_nebula() {
    for rows in [1, 2] {
        let app = application(rows);
        let program = app.program(&preload(TABLE)).unwrap();
        let mut run = Run::new(app.clone(), &program, vec![0]);
        run.segment(&inputs(rows)).unwrap();
        assert_eq!(run.state(), &vec![4]);
        assert_eq!(run.timestamp(), 8, "four reads and four writes");

        let mut wrong = inputs(rows);
        wrong[1][VALUE] = F::from_u64(9);
        let mut run = Run::new(app, &program, vec![0]);
        assert!(matches!(run.segment(&wrong), Err(SegmentError::Access)));
    }
}

#[test]
fn construction_rejects_invalid_geometry_before_loading_a_file() {
    let invalid = [
        Geometry { r: 64, ..geometry(1) },
        Geometry { mu: 64, ..geometry(1) },
        Geometry {
            n: usize::MAX,
            ..geometry(1)
        },
        Geometry {
            r: 16,
            mu: 16,
            w_ts: 18,
            b_ops: 1,
            b_scan: 1,
            n: 1 << 17,
            s_max: 1,
        },
    ];
    for geometry in invalid {
        let result = nightstream::nebula::Prover::load("", geometry, application(1), nightstream::Engine::Optimized, 1);
        assert!(matches!(result, Err(nightstream::Error::Circuit(_))));
    }
    let valid = nightstream::nebula::Prover::load("", geometry(1), application(1), nightstream::Engine::Optimized, 1);
    assert!(matches!(valid, Err(nightstream::Error::Io(_))));
}

#[test]
fn construction_rejects_a_rank_in_an_empty_catalog_without_panicking() {
    let empty = MemoryCatalog::<u32>::new([], &registry()).unwrap();
    assert_eq!(
        batched_slots(&empty, &[vec![0]], NonZeroUsize::MIN),
        Err(CatalogBindingError::Plan)
    );
    assert_eq!(batched_slots(&empty, &[], NonZeroUsize::MIN), Ok(Vec::new()));
    assert_eq!(
        batched_slots(&catalog(), &[vec![2]], NonZeroUsize::MIN),
        Err(CatalogBindingError::Plan)
    );
    assert_eq!(
        batched_slots(&catalog(), &[vec![0], vec![1]], NonZeroUsize::new(usize::MAX).unwrap()),
        Err(CatalogBindingError::Plan)
    );
    assert_eq!(
        batched_slots(&catalog(), &[vec![0], vec![1]], NonZeroUsize::new(2).unwrap()),
        Ok(vec![vec![0], vec![2], vec![1], vec![3]])
    );
}

#[test]
fn construction_does_not_wrap_region_sizes() {
    for widths in [vec![64], vec![usize::MAX, 1]] {
        assert_eq!(
            pack_regions([(0u32, MemoryKind::Rom, widths)]),
            Err(CatalogBindingError::Region)
        );
    }
    assert_eq!(
        pack_regions([(0u32, MemoryKind::Rom, vec![63]), (1, MemoryKind::Rom, vec![63])]),
        Err(CatalogBindingError::Region)
    );
    let (_, used) = pack_regions([(0u32, MemoryKind::Rom, vec![63])]).unwrap();
    assert_eq!(used, [1u64 << 63, 0]);
}

#[test]
fn construction_rejects_columns_outside_the_row() {
    let catalog = catalog();
    let (regions, _) = pack_regions(
        catalog
            .entries()
            .iter()
            .map(|memory| (memory.id, memory.kind, vec![2])),
    )
    .unwrap();
    let size = Size {
        geometry: geometry(1),
        rows: NonZeroUsize::MIN,
        regions,
        slots: batched_slots(&catalog, &[vec![0], vec![1]], NonZeroUsize::MIN).unwrap(),
    };
    let original = RowRelation::of(&relation()).unwrap();
    let mut term = original.clone();
    term.rows[0][0].push((COLUMNS, F::ONE));
    let mut one = original.clone();
    one.one = COLUMNS;
    let mut port = original.clone();
    port.width = COLUMNS - 1;
    for invalid in [term, one, port] {
        assert!(matches!(
            R1csApplication::new(invalid, &continuity(), &catalog, size.clone()),
            Err(R1csApplicationError::Column)
        ));
    }
    let wider = ColumnRegistry::new([ColumnFamilySpec {
        region: "counter",
        start: 0,
        len: COLUMNS + 1,
        name: "words".into(),
        role: "test",
        width: ColumnWidth::U32,
    }])
    .unwrap();
    let link = ContinuityCatalog::new(
        [ContinuityGroup {
            name: "count",
            role: "test",
            links: vec![ContinuityLink {
                previous_step_column: COLUMNS,
                next_step_column: BEFORE,
            }],
        }],
        &wider,
    )
    .unwrap();
    assert!(matches!(
        R1csApplication::new(original.clone(), &link, &catalog, size.clone()),
        Err(R1csApplicationError::Column)
    ));
    let mut overflow = original.clone();
    overflow.width = usize::MAX;
    let mut large = size.clone();
    large.rows = NonZeroUsize::new(2).unwrap();
    assert!(matches!(
        R1csApplication::new(overflow, &continuity(), &catalog, large),
        Err(R1csApplicationError::Dimension)
    ));
    assert!(R1csApplication::new(original, &continuity(), &catalog, size).is_ok());
}

/// The circuit of two rows in each step proves the honest run. The verifier
/// accepts it and rejects another last count and another table.
#[test]
#[ignore = "Full production-profile memory proof; run separately under the 300-second cap."]
fn a_relation_of_the_builders_proves_and_verifies_on_cpu() {
    let engine = nightstream::Engine::Optimized;
    let app = application(2);
    let circuit = Circuit::compile(geometry(2), app.clone()).unwrap();
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let program = app.program(&preload(TABLE)).unwrap();

    let mut run = prover.start(&program, vec![0]).unwrap();
    run.segment(&inputs(2)).unwrap();
    let last = run.state().clone();
    let proof = run.finish().unwrap();
    let verified = verifier.verify(&program, &vec![0], &last, &proof).unwrap();
    assert_eq!(verified.steps, 2);
    assert!(matches!(
        verifier.verify(&program, &vec![0], &vec![5], &proof),
        Err(VerifyError::Proof(_))
    ));
    let other = app.program(&preload([5, 6, 7, 9])).unwrap();
    assert!(matches!(
        verifier.verify(&other, &vec![0], &last, &proof),
        Err(VerifyError::Proof(_))
    ));
}

/// The flag and the two shared columns of the second counter.
const FLAG: usize = 4;
const S0: usize = 5;
const S1: usize = 6;
const SHARED_WIDTH: usize = 7;

/// The counter with `S1 = S0 + 1`, which reads only shared columns, and
/// `FLAG · (S0 − VALUE) = 0` for each row. Each group of two rows shares
/// `S0` and `S1`.
fn shared_relation() -> RowRelation {
    let registry = ColumnRegistry::new([ColumnFamilySpec {
        region: "counter",
        start: 0,
        len: SHARED_WIDTH,
        name: "words".into(),
        role: "test",
        width: ColumnWidth::U32,
    }])
    .unwrap();
    let mut builder = R1csBuilder::new(1, ONE).unwrap();
    builder.with_tag(ConstraintTag::new("count", "counter"), |rows| {
        rows.push_linear_zero([(AFTER, F::ONE), (BEFORE, -F::ONE), (ONE, -F::ONE)]);
        rows.push_linear_zero([(S1, F::ONE), (S0, -F::ONE), (ONE, -F::ONE)]);
        rows.push_row([(FLAG, F::ONE)], [(S0, F::ONE), (VALUE, -F::ONE)], []);
    });
    let mut relation = RowRelation::of(&builder.build(&registry).unwrap()).unwrap();
    relation.shared = Shared {
        columns: 2,
        rows: NonZeroUsize::new(2).unwrap(),
    };
    relation
}

/// The honest one-row witnesses of the second counter: the odd rows are
/// flagged, and each row's own shared columns hold its own value.
fn shared_rows() -> Vec<Vec<F>> {
    (0..4u64)
        .map(|count| {
            let value = F::from_u32(TABLE[count as usize]);
            vec![
                F::ONE,
                F::from_u64(count),
                F::from_u64(count + 1),
                value,
                F::from_u64(count % 2),
                value,
                value + F::ONE,
            ]
        })
        .collect()
}

/// The step witnesses of `rows` one-row witnesses each, every group taking
/// the shared columns of its row `source`.
fn shared_inputs(rows: &[Vec<F>], per_step: usize, source: impl Fn(&[&[F]]) -> usize + Copy) -> Vec<Vec<F>> {
    let shared = shared_relation().shared;
    rows.chunks(per_step)
        .map(|step| {
            let step: Vec<&[F]> = step.iter().map(Vec::as_slice).collect();
            shared.step_witness(SHARED_WIDTH, &step, source)
        })
        .collect()
}

fn flagged(group: &[&[F]]) -> usize {
    group.iter().position(|z| z[FLAG] == F::ONE).unwrap_or(0)
}

/// A step holds each row without its shared columns, then one block for
/// each group, and keeps the row that reads only shared columns once for each
/// group. The honest run holds every row and runs through Nebula. A changed
/// shared word, or the block of the group's row that is not flagged, breaks a
/// row. Two flagged rows of one group with different values have no block:
/// their gated rows require `S0 = 5` and `S0 = 6`.
#[test]
fn a_group_of_rows_shares_one_block_of_shared_columns() {
    let block = SHARED_WIDTH - 2;
    for per_step in [2, 4] {
        let groups = per_step / 2;
        let app = application_of(shared_relation(), per_step).unwrap();
        assert_eq!(app.aux(), per_step * block + groups * 2);
        assert_eq!(app.relation_rows(), per_step * 2 + groups + (per_step - 1));
        let inputs = shared_inputs(&shared_rows(), per_step, flagged);
        for z in &inputs {
            assert!(app.check(z).is_ok(), "{per_step} rows per step");
        }
        let program = app.program(&preload(TABLE)).unwrap();
        let mut run = Run::new(app.clone(), &program, vec![0]);
        run.segment(&inputs).unwrap();
        assert_eq!(run.state(), &vec![4]);

        let mut wrong = inputs[0].clone();
        wrong[per_step * block] += F::ONE;
        assert!(matches!(app.check(&wrong), Err(R1csStepError::Relation { .. })));
        assert!(matches!(
            app.check(&wrong[1..]),
            Err(R1csStepError::Width { words, expected }) if words + 1 == expected
        ));
        for z in shared_inputs(&shared_rows(), per_step, |_| 0) {
            assert!(matches!(app.check(&z), Err(R1csStepError::Relation { .. })));
        }
    }
    let app = application_of(shared_relation(), 2).unwrap();
    let mut both = shared_rows();
    both[0][FLAG] = F::ONE;
    for source in [0, 1] {
        let z = &shared_inputs(&both, 2, move |_| source)[0];
        assert!(matches!(app.check(z), Err(R1csStepError::Relation { .. })));
    }
}

/// A step must hold whole groups, and the constant-one, continuity and
/// memory columns must not be shared.
#[test]
fn shared_columns_are_whole_groups_without_an_interface_column() {
    assert!(matches!(
        application_of(shared_relation(), 1),
        Err(R1csApplicationError::Group)
    ));
    for columns in [4, 5, SHARED_WIDTH, SHARED_WIDTH + 1] {
        let mut relation = shared_relation();
        relation.shared.columns = columns;
        assert!(
            matches!(application_of(relation, 2), Err(R1csApplicationError::Column)),
            "{columns} shared columns"
        );
    }
    assert!(application_of(shared_relation(), 2).is_ok());
}
