//! Reviewed Lean artifact for two production packed centered-domain rows.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::SELECTIVE_ROW_ARTIFACT_SCHEMA_VERSION;
use neo_fold_clean::paper::relations::Structure;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRadixFourCenteredDomainRows.lean";
const GENERAL_SELECTOR_PORT: usize = 1;
const RIGHT_COORDINATE_PORT: usize = 2;
const LEFT_COORDINATE_PORT: usize = 6;
const EVALUATION_SELECTOR_PORT: usize = 7;

struct MaterializedRow {
    rows: usize,
    columns: usize,
    emitted_row: usize,
    run_index: usize,
    ports: [Vec<(usize, F)>; 13],
}

fn materialize(structure: &Structure, emitted_row: usize, run_index: usize) -> MaterializedRow {
    MaterializedRow {
        rows: structure.n,
        columns: structure.m,
        emitted_row,
        run_index,
        ports: std::array::from_fn(|port| {
            structure.matrices[port]
                .materialize_row(emitted_row)
                .expect("selected row is inside the production structure")
        }),
    }
}

fn has_shape(row: &MaterializedRow, selector_column: usize, pair: bool) -> bool {
    for port in 0..13 {
        let terms = &row.ports[port];
        let valid = match port {
            GENERAL_SELECTOR_PORT | EVALUATION_SELECTOR_PORT => terms == &[(selector_column, F::ONE)],
            LEFT_COORDINATE_PORT => terms.len() == 1 && terms[0].1 == F::ONE,
            RIGHT_COORDINATE_PORT if pair => terms.len() == 1 && terms[0].1 == F::ONE,
            _ => terms.is_empty(),
        };
        if !valid {
            return false;
        }
    }
    true
}

fn write_raw_row(rendered: &mut String, name: &str, row: &MaterializedRow) -> std::fmt::Result {
    writeln!(
        rendered,
        "\ndef {name} : RawRow where\n  schemaVersion := {}\n  rows := {}\n  columns := {}\n  emittedRow := {}\n  runIndex := {}\n  family := .armDomain\n  arm := some 1\n  ports := [",
        SELECTIVE_ROW_ARTIFACT_SCHEMA_VERSION,
        row.rows,
        row.columns,
        row.emitted_row,
        row.run_index,
    )?;
    for (port_index, port) in row.ports.iter().enumerate() {
        let separator = if port_index == 0 { "    " } else { "  , " };
        let terms = port
            .iter()
            .map(|(column, coefficient)| {
                format!(
                    "{{ column := {column}, coefficient := {} }}",
                    coefficient.as_canonical_u64()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(rendered, "{separator}{{ terms := [{terms}] }}")?;
    }
    writeln!(rendered, "]")
}

fn render(pair_row: &MaterializedRow, tail_row: &MaterializedRow) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema\n\n\
/-! Generated file: representative production radix-four centered-domain rows.\n\n\
Owns: one exact two-coordinate row and the exact fixed-zero tail row from the\n\
final recursive-arm matrices of the production WASM census profile.\n\n\
Does not own: source-coordinate meaning, all centered rows, selector dispatch,\n\
constraint necessity, security reduction, or permission to remove rows.\n\n\
Emits constraints: no. Rust materializes both final rows before export.\n\n\
| Artifact row | Final nonempty ports | Assurance use |\n\
|---|---|---|\n\
| pair | G, E, U, A | exact production coefficient binding |\n\
| tail | G, E, U | exact production fixed-zero binding |\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourCenteredDomainRows\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire"
    )
    .expect("render centered-domain artifact header");
    write_raw_row(&mut rendered, "rawPairRow", pair_row).expect("render centered pair row");
    write_raw_row(&mut rendered, "rawTailRow", tail_row).expect("render centered tail row");
    writeln!(
        rendered,
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourCenteredDomainRows"
    )
    .expect("render centered-domain artifact footer");
    rendered
}

pub fn assert_artifact_matches_committed(
    structure: &Structure,
    run_index: usize,
    pair_row: usize,
    tail_row: usize,
    selector_column: usize,
) {
    let pair_row = materialize(structure, pair_row, run_index);
    let tail_row = materialize(structure, tail_row, run_index);
    assert!(has_shape(&pair_row, selector_column, true), "invalid pair row");
    assert!(has_shape(&tail_row, selector_column, false), "invalid tail row");
    let rendered = render(&pair_row, &tail_row);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write reviewed centered-domain artifact");
        panic!("centered-domain Lean artifact drifted; wrote {expected}. Inspect and promote it explicitly");
    }
}
