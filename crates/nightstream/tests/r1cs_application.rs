//! A relation that the public builders of `nightstream::relation` make,
//! proved through `nebula::R1csApplication` as the Starstream interleaving
//! relation is (SPEC.md section 7.6). It is a counter: each row reads
//! `table[count]` from a ROM, writes it to a RAM at `count`, and adds one to
//! `count`. A step holds one row or two.

use std::num::NonZeroUsize;

use nightstream::nebula::{
    batched_slots, Circuit, Geometry, R1csApplication, Region, RowRelation, Run, SegmentError, Size, VerifyError,
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
        name: "words",
        role: "test",
        width: ColumnWidth::U32,
    }])
    .unwrap()
}

/// `count_after = count_before + 1`.
fn relation() -> R1csRelation<&'static str> {
    let mut builder = R1csBuilder::new(COLUMNS, 1, ONE).unwrap();
    builder.with_tag(ConstraintTag::new("count", "counter"), |rows| {
        rows.push_linear_zero([(AFTER, F::ONE), (BEFORE, -F::ONE), (ONE, -F::ONE)]);
    });
    builder.build().unwrap()
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
    Geometry {
        r: 2,
        mu: 2,
        w_ts: 4,
        b_ops: 2 * rows,
        b_scan: 2 * rows,
        n: 4 / rows,
        s_max: 1,
    }
}

fn application(rows: usize) -> R1csApplication<u32> {
    let rows = NonZeroUsize::new(rows).unwrap();
    let continuity = ContinuityCatalog::new(
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
    .unwrap();
    let catalog = catalog();
    let size = Size {
        geometry: geometry(rows.get()),
        rows,
        regions: vec![
            (
                0,
                Region {
                    base: 0,
                    widths: vec![2],
                },
            ),
            (
                1,
                Region {
                    base: 0,
                    widths: vec![2],
                },
            ),
        ],
        slots: batched_slots(&catalog, &[vec![0], vec![1]], rows),
    };
    R1csApplication::new(RowRelation::of(&relation()).unwrap(), &continuity, &catalog, size).unwrap()
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

/// The circuit of two rows in each step proves the honest run. The verifier
/// accepts it and rejects another last count and another table.
#[test]
#[ignore = "Full production-profile memory proof; run separately under the 300-second cap."]
fn a_relation_of_the_builders_proves_and_verifies_on_cpu() {
    let engine = nightstream::Engine::Optimized;
    let reference = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts")
            .join("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .unwrap();
    let app = application(2);
    let circuit = Circuit::compile(&reference, geometry(2), app.clone()).unwrap();
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = circuit.verifier(engine, 114).unwrap();
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
