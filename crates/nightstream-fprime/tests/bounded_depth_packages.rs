use std::fs;
use std::path::PathBuf;

use neo_ccs::crypto::poseidon2::permute_state;
use neo_math::Fq;
use nightstream_fprime::{
    bounded_depth_identity, load_bounded_depth_packages, parse_atoms, BoundedDepthPackages, PackageError, StepPackage,
    BOUNDED_DEPTH_PACKAGES_IDENTITY, PUBLIC_WORDS,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::Value;

fn artifact(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join(name),
    )
    .unwrap_or_else(|error| panic!("read {name}: {error}"))
}

fn parity() -> Vec<Value> {
    match parse_atoms(&artifact("nightstream-fprime-bounded-depth-parity-v1.json")).expect("parity file") {
        Value::Array(fields) if fields.len() == 5 => fields,
        _ => panic!("the parity file has five fields"),
    }
}

fn packages() -> BoundedDepthPackages {
    load_bounded_depth_packages(
        &artifact("nightstream-fprime-bounded-depth-light-v1.json"),
        &artifact("nightstream-fprime-bounded-depth-refresh-v1.json"),
        BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )
    .expect("bounded-depth packages")
}

fn words(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|word| word.as_u64().expect("word"))
        .collect()
}

/// Lean `Parity.sharedWord`.
fn shared_word(column: usize) -> Fq {
    if column == 0 {
        return Fq::ONE;
    }
    let q = u128::from(Fq::ORDER_U64);
    Fq::from_u64(((column as u128 * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) % q) as u64)
}

fn shared_columns(summary: &Value) -> Vec<Fq> {
    (0..words(summary)[3] as usize).map(shared_word).collect()
}

#[test]
fn identity_code_matches_lean_on_the_probe_packages() {
    let fields = parity();
    let expected: [u64; 4] = words(&fields[4]).try_into().expect("four words");
    assert_eq!(bounded_depth_identity(&fields[3]).expect("probe identity"), expected);
}

#[test]
fn relation_rows_match_the_lean_checksum() {
    let fields = parity();
    let packages = packages();
    for (package, summary) in [(packages.light(), &fields[1]), (packages.refresh(), &fields[2])] {
        let [ordinary, sbox, checksum, _, _, _, _, _] = words(summary)[..] else {
            panic!("a summary has eight words");
        };
        let expanded = package.clone().expand().expect("expanded package");
        let rows = expanded.rows();
        let circuit = package.circuit();
        assert_eq!(
            rows.ordinary_row_count() as u64,
            ordinary,
            "relation rows of circuit {circuit}"
        );
        assert_eq!(
            rows.row_count() as u64,
            ordinary + sbox,
            "census rows of circuit {circuit}"
        );
        assert_eq!(
            rows.ordinary_checksum(),
            checksum,
            "relation-row checksum of circuit {circuit}"
        );
    }
}

/// Lean `RowsParity.mix`.
fn mix(unit: u64, column: u64) -> Fq {
    let point = Fq::from_u64(unit * 16_777_259 + column + 1);
    point * point * point + Fq::from_u64(unit)
}

/// The dense rows that `Relation::new` gives `FactoredRows` (the package's
/// plan rows and the spans of its layout), as Lean's rows summary
/// (`Parity.rowsSummary`): the relation rows, the S-box rows, the logical
/// width and the checksum `RowsParity.denseSum`. A source column's weight is
/// the weights `mix(2, x)` of its span's cells, recomposed with the ratio
/// (Lean `RowsParity.columnWeight`), and port `4 · row + matrix` has the
/// weight `mix(4 · row + matrix, 1)`.
fn dense_rows(package: &StepPackage) -> [u64; 4] {
    let expanded = package.clone().expand().expect("expanded package");
    let (rows, layout) = (expanded.rows(), expanded.layout());
    let spans = layout.spans();
    let column_weight: Vec<Fq> = spans
        .starts
        .iter()
        .zip(&spans.widths)
        .map(|(&start, &count)| {
            (0..u64::from(count)).rev().fold(Fq::ZERO, |sum, digit| {
                sum * spans.ratio + mix(2, u64::from(start) + digit)
            })
        })
        .collect();
    let mut total = Fq::ZERO;
    for unit in 0..4 * rows.row_count() {
        let port = rows
            .port(unit / 4, unit % 4)
            .fold(Fq::ZERO, |sum, (column, coefficient)| {
                let weight = column_weight
                    .get(column)
                    .expect("each entry reads a source column with a span");
                sum + coefficient * *weight
            });
        total += mix(unit as u64, 1) * port;
    }
    let ordinary = rows.ordinary_row_count();
    [
        ordinary as u64,
        (rows.row_count() - ordinary) as u64,
        layout.width() as u64,
        total.as_canonical_u64(),
    ]
}

/// The packages of a program, loaded under the identity of their own pair: the
/// Nebula programs have no pinned identity.
fn program_packages(stem: &str) -> BoundedDepthPackages {
    let light = artifact(&format!("nightstream-fprime-bounded-depth-{stem}light-v1.json"));
    let refresh = artifact(&format!("nightstream-fprime-bounded-depth-{stem}refresh-v1.json"));
    let pair = Value::Array(vec![
        parse_atoms(&light).expect("light atoms"),
        parse_atoms(&refresh).expect("refresh atoms"),
    ]);
    let identity = bounded_depth_identity(&pair).expect("identity");
    load_bounded_depth_packages(&light, &refresh, identity).expect("packages")
}

/// The factored rows of each shipped package have the dense matrices of
/// Lean's model `FinalRows.packageRows`: Lean's parity file of the reference
/// application and its rows file of each Nebula program give the same words.
#[test]
fn factored_rows_match_the_lean_dense_checksum() {
    let fields = parity();
    let packages = packages();
    for (package, summary) in [(packages.light(), &fields[1]), (packages.refresh(), &fields[2])] {
        let words = words(summary);
        let expected = [words[0], words[1], words[6], words[7]];
        assert_eq!(dense_rows(package), expected, "reference circuit {}", package.circuit());
    }
    for program in ["nebula-v1", "nebula-n1"] {
        let file = parse_atoms(&artifact(&format!(
            "nightstream-fprime-bounded-depth-{program}-rows-v1.json"
        )))
        .expect("rows file");
        let fields = file.as_array().expect("array");
        assert_eq!(fields.len(), 3, "rows file of {program}");
        assert_eq!(fields[0].as_u64(), Some(1), "rows file version of {program}");
        let packages = program_packages(&format!("{program}-"));
        for (index, package) in [packages.light(), packages.refresh()]
            .into_iter()
            .enumerate()
        {
            let expected: [u64; 4] = words(&fields[index + 1]).try_into().expect("four words");
            assert_eq!(dense_rows(package), expected, "{program} circuit {}", package.circuit());
        }
    }
}

/// Each shipped package has the capacity that Lean's package facts state
/// (`EmittedPackages.Facts`, `SPEC.md` section 4.4): its S-box columns end
/// within the source width `2^24`, its rows fit the cube of `2^28` (Lean
/// `cubeVariables`), and its logical width fits the key width `2^26` (Lean
/// `StepRelation.keyWidth`). The row counts and the logical widths are Lean's
/// (`factored_rows_match_the_lean_dense_checksum`).
#[test]
fn step_packages_fit_their_capacity() {
    for stem in ["", "nebula-v1-", "nebula-n1-"] {
        let packages = program_packages(stem);
        for package in [packages.light(), packages.refresh()] {
            let name = format!("{stem}circuit {}", package.circuit());
            assert_eq!(package.width(), 1 << 24, "source width of {name}");
            assert!(
                package.sbox_start() + 150 * package.block_count() <= package.width(),
                "S-box columns of {name}"
            );
            let expanded = package.clone().expand().expect("expanded package");
            assert!(expanded.rows().row_count() <= 1 << 28, "rows of {name}");
            assert!(expanded.layout().width() <= 1 << 26, "logical width of {name}");
        }
    }
}

#[test]
fn loader_rejects_another_identity() {
    let mut other = BOUNDED_DEPTH_PACKAGES_IDENTITY;
    other[0] ^= 1;
    let loaded = load_bounded_depth_packages(
        &artifact("nightstream-fprime-bounded-depth-light-v1.json"),
        &artifact("nightstream-fprime-bounded-depth-refresh-v1.json"),
        other,
    );
    assert!(matches!(loaded, Err(PackageError::IdentityMismatch { .. })));
}

#[test]
fn block_rows_compute_the_native_permutation() {
    let packages = packages();
    let light = packages.light();
    let expanded = light.clone().expand().expect("expanded package");
    let rows = expanded.rows();
    let width = light.sbox_start() + 150 * light.block_count();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut assignment: Vec<Fq> = (0..width)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            Fq::from_u64(state >> 1)
        })
        .collect();
    assignment[0] = Fq::ONE;
    for block in light.blocks() {
        let lanes: [Fq; 16] = block.lane_values(&assignment).try_into().expect("16 lanes");
        assignment[block.first_output()..block.first_output() + 16].copy_from_slice(&permute_state(lanes));
    }
    for row in rows.ordinary_row_count()..rows.row_count() {
        let [(column, coefficient)] = rows.port(row, 2).collect::<Vec<_>>()[..] else {
            panic!("an S-box row's C port is one S-box column");
        };
        assert_eq!(coefficient, Fq::ONE);
        assignment[column] = rows.evaluate(row, 3, &assignment).exp_u64(7);
    }
    for row in rows.sbox_rows() {
        assert_eq!(rows.residual(row, &assignment).as_canonical_u64(), 0, "row {row}");
    }
    let source = light.source_rows().expect("source rows");
    assert_eq!(source.row_count(), rows.row_count());
    let mut with_form = 0;
    for row in 0..rows.row_count() {
        for port in 0..4 {
            assert_eq!(
                rows.evaluate(row, port, &assignment),
                source.evaluate(row, port, &assignment),
                "row {row}, port {port}"
            );
            with_form += usize::from(!rows.port(row, port).eq(source.port(row, port)));
        }
    }
    assert!(with_form > 0, "some port reads an output form");
}

#[test]
fn circuit_columns_match_the_lean_witness() {
    let fields = parity();
    let packages = packages();
    for (package, summary) in [(packages.light(), &fields[1]), (packages.refresh(), &fields[2])] {
        let columns = package
            .circuit_columns(&shared_columns(summary))
            .expect("circuit columns");
        assert_eq!(
            package.witness_checksum(&columns),
            words(summary)[4],
            "witness of circuit {}",
            package.circuit()
        );
    }
}

#[test]
fn fresh_columns_satisfy_the_rows_that_define_them() {
    let fields = parity();
    let packages = packages();
    for (package, summary) in [(packages.light(), &fields[1]), (packages.refresh(), &fields[2])] {
        let expanded = package.clone().expand().expect("expanded package");
        let rows = expanded.rows();
        let columns = package
            .circuit_columns(&shared_columns(summary))
            .expect("circuit columns");
        let columns = package
            .relation_columns(columns, rows)
            .expect("relation columns");
        let direct = expanded
            .assignment_columns(&shared_columns(summary), None)
            .unwrap();
        assert_eq!(columns, direct, "all fresh columns of circuit {}", package.circuit());
        let defining = rows.ring_rows().iter().cloned().chain([rows.sbox_rows()]);
        for row in defining.flatten() {
            assert_eq!(
                rows.residual(row, &columns).as_canonical_u64(),
                0,
                "circuit {} row {row}",
                package.circuit()
            );
        }
    }
}

#[test]
fn logical_assignment_matches_the_lean_layout() {
    let fields = parity();
    let packages = packages();
    for (package, summary) in [(packages.light(), &fields[1]), (packages.refresh(), &fields[2])] {
        let expanded = package.clone().expand().expect("expanded package");
        let (rows, layout) = (expanded.rows(), expanded.layout());
        let columns = package
            .circuit_columns(&shared_columns(summary))
            .expect("circuit columns");
        let columns = package
            .relation_columns(columns, rows)
            .expect("relation columns");
        let logical = layout.assignment(&columns).expect("logical assignment");
        assert_eq!(
            layout.checksum(&logical, package.first_fresh()),
            words(summary)[5],
            "logical assignment of circuit {}",
            package.circuit()
        );
        let unit = |value: &Fq| matches!(value.as_canonical_u64(), 0 | 1) || *value == -Fq::ONE;
        assert!(logical[PUBLIC_WORDS..].iter().all(unit), "a digit outside {{-1, 0, 1}}");
        for (column, value) in columns.iter().enumerate().skip(PUBLIC_WORDS) {
            if let Some((offset, kind)) = layout.slot(column) {
                let recomposed = (0..kind.width())
                    .rev()
                    .fold(Fq::ZERO, |sum, digit| sum * Fq::from_u64(3) + logical[offset + digit]);
                assert_eq!(recomposed, *value, "column {column} recomposes");
            }
        }
        println!("circuit {}: {} logical columns", package.circuit(), layout.width());
    }
}
