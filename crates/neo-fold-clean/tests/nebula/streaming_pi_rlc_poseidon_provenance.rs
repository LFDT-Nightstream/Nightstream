//! Focused production PiRLC Poseidon2 source-to-final provenance check.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use neo_fold_clean::frontends::nebula::f_prime::{
    production_pi_rlc_family_body_compiler_audit, production_pi_rlc_family_body_projected_rows_with_source_provenance,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveProjectedGeometricRun, SelectiveProjectedPort, SelectiveProjectedPoseidon2SboxStep,
    SelectiveProjectedRowArtifact, SelectiveProjectedSourceImage, SelectiveProjectedSourceLinearCombination,
    SelectiveProjectedTerm, SelectiveRewriteKind,
};
use p3_field::PrimeField64;

const STEPS_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafSteps.lean";
const ROWS_0_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafRows0.lean";
const ROWS_1_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafRows1.lean";
const CHAINED_ROWS_0_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafRows0.lean";
const CHAINED_ROWS_1_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafRows1.lean";
const CHAINED_IMAGES_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafImages.lean";
const PARTIAL_ROWS_0_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafRows0.lean";
const PARTIAL_ROWS_1_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafRows1.lean";
const PARTIAL_STEPS_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafSteps.lean";
const EXTERNAL_A_SOURCE_START: usize = 1_559;
const EXTERNAL_B_SOURCE_START: usize = 166_308;
const PARTIAL_CARRIED_SOURCE_START: usize = 166_304;
const SOURCE_START: usize = 165_446;
const SOURCE_END: usize = 166_046;
const SOURCE_COLUMN_START: usize = 166_320;
const SOURCE_COLUMN_END: usize = 166_920;
const EMITTED_START: usize = 74_375;
const EMITTED_END: usize = 74_461;
const ODD_EMITTED_START: usize = 309_886;
const EXTERNAL_A_SLOT_START: usize = 38_340;
const PARTIAL_CARRIED_SLOT_START: usize = 2_217_769;
const EXTERNAL_B_SLOT_START: usize = 2_217_933;
const FINAL_SLOT_START: usize = 2_218_425;
const FINAL_SLOT_END: usize = 2_221_951;
const SELECTOR_COLUMN: usize = 648;
const PARTIAL_SELECTOR_COLUMN: usize = 649;
const SLOT_WIDTH: usize = 41;
const LOCAL_SLOT_COUNT: usize = 86;

#[derive(Clone, Copy)]
enum LeafClass {
    Direct,
    Chained,
    Partial,
}

impl LeafClass {
    const fn external_a_source_start(self) -> usize {
        match self {
            Self::Direct => EXTERNAL_A_SOURCE_START,
            Self::Chained => EXTERNAL_A_SOURCE_START + 4,
            Self::Partial => PARTIAL_CARRIED_SOURCE_START,
        }
    }

    const fn external_b_source_start(self) -> usize {
        match self {
            Self::Direct => EXTERNAL_B_SOURCE_START,
            Self::Chained => SOURCE_COLUMN_END - 4,
            Self::Partial => EXTERNAL_B_SOURCE_START,
        }
    }

    const fn source_column_start(self) -> usize {
        match self {
            Self::Direct => SOURCE_COLUMN_START,
            Self::Chained => SOURCE_COLUMN_END,
            Self::Partial => SOURCE_COLUMN_START,
        }
    }

    const fn external_a_slot_start(self) -> usize {
        match self {
            Self::Direct => EXTERNAL_A_SLOT_START,
            Self::Chained => EXTERNAL_A_SLOT_START + 4 * SLOT_WIDTH,
            Self::Partial => PARTIAL_CARRIED_SLOT_START,
        }
    }

    const fn final_slot_start(self) -> usize {
        match self {
            Self::Direct => FINAL_SLOT_START,
            Self::Chained => FINAL_SLOT_END,
            Self::Partial => FINAL_SLOT_START,
        }
    }
}

fn lean_source_column(class: LeafClass, column: usize) -> String {
    if matches!(class, LeafClass::Partial) {
        if (PARTIAL_CARRIED_SOURCE_START..PARTIAL_CARRIED_SOURCE_START + 2).contains(&column) {
            return format!(".externalA {}", column - PARTIAL_CARRIED_SOURCE_START);
        }
        if (EXTERNAL_A_SOURCE_START..EXTERNAL_A_SOURCE_START + 2).contains(&column) {
            return format!(".externalA {}", 2 + column - EXTERNAL_A_SOURCE_START);
        }
        if (EXTERNAL_B_SOURCE_START..EXTERNAL_B_SOURCE_START + 4).contains(&column) {
            return format!(".externalB {}", column - EXTERNAL_B_SOURCE_START);
        }
        if (SOURCE_COLUMN_START..SOURCE_COLUMN_END).contains(&column) {
            return format!(".local {}", column - SOURCE_COLUMN_START);
        }
        panic!("unclassified partial-start Poseidon2 source column {column}");
    }
    let external_a = class.external_a_source_start();
    let external_b = class.external_b_source_start();
    let local = class.source_column_start();
    if (external_a..external_a + 4).contains(&column) {
        format!(".externalA {}", column - external_a)
    } else if (external_b..external_b + 4).contains(&column) {
        format!(".externalB {}", column - external_b)
    } else if (local..local + 600).contains(&column) {
        format!(".local {}", column - local)
    } else {
        panic!("unclassified Poseidon2 source column {column}")
    }
}

fn lean_source_lc(class: LeafClass, value: &SelectiveProjectedSourceLinearCombination) -> String {
    let terms = value
        .terms()
        .iter()
        .map(|term| {
            format!(
                "{{ column := {}, coefficient := {} }}",
                lean_source_column(class, term.column()),
                term.coefficient().as_canonical_u64()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{ constant := {}, terms := [{terms}] }}",
        value.constant().as_canonical_u64()
    )
}

fn lean_source_lc_comparison(class: LeafClass, value: &SelectiveProjectedSourceLinearCombination) -> String {
    let mut terms = value
        .terms()
        .iter()
        .map(|term| {
            (
                lean_source_column(class, term.column()),
                term.coefficient().as_canonical_u64(),
            )
        })
        .collect::<Vec<_>>();
    terms.sort();
    format!("{}:{terms:?}", value.constant().as_canonical_u64())
}

fn lean_explicit_column(class: LeafClass, term: SelectiveProjectedTerm) -> &'static str {
    match term.column() {
        0 => ".one",
        column
            if column
                == match class {
                    LeafClass::Partial => PARTIAL_SELECTOR_COLUMN,
                    LeafClass::Direct | LeafClass::Chained => SELECTOR_COLUMN,
                } =>
        {
            ".selector"
        }
        column => panic!("unclassified explicit Poseidon2 final column {column}"),
    }
}

fn lean_slot(class: LeafClass, run: SelectiveProjectedGeometricRun) -> String {
    assert_eq!(run.length(), SLOT_WIDTH);
    let start = run.column_start();
    if matches!(class, LeafClass::Partial) {
        if (PARTIAL_CARRIED_SLOT_START..PARTIAL_CARRIED_SLOT_START + 2 * SLOT_WIDTH).contains(&start) {
            assert_eq!((start - PARTIAL_CARRIED_SLOT_START) % SLOT_WIDTH, 0);
            return format!(".externalA {}", (start - PARTIAL_CARRIED_SLOT_START) / SLOT_WIDTH);
        }
        if (EXTERNAL_A_SLOT_START..EXTERNAL_A_SLOT_START + 2 * SLOT_WIDTH).contains(&start) {
            assert_eq!((start - EXTERNAL_A_SLOT_START) % SLOT_WIDTH, 0);
            return format!(".externalA {}", 2 + (start - EXTERNAL_A_SLOT_START) / SLOT_WIDTH);
        }
        if (EXTERNAL_B_SLOT_START..EXTERNAL_B_SLOT_START + 4 * SLOT_WIDTH).contains(&start) {
            assert_eq!((start - EXTERNAL_B_SLOT_START) % SLOT_WIDTH, 0);
            return format!(".externalB {}", (start - EXTERNAL_B_SLOT_START) / SLOT_WIDTH);
        }
        if (FINAL_SLOT_START..FINAL_SLOT_START + LOCAL_SLOT_COUNT * SLOT_WIDTH).contains(&start) {
            assert_eq!((start - FINAL_SLOT_START) % SLOT_WIDTH, 0);
            return format!(".local {}", (start - FINAL_SLOT_START) / SLOT_WIDTH);
        }
        panic!("unclassified partial-start geometric Poseidon2 final column {start}");
    }
    let external_a = class.external_a_slot_start();
    let local = class.final_slot_start();
    if (external_a..external_a + 4 * SLOT_WIDTH).contains(&start) {
        assert_eq!((start - external_a) % SLOT_WIDTH, 0);
        format!(".externalA {}", (start - external_a) / SLOT_WIDTH)
    } else if matches!(class, LeafClass::Direct)
        && (EXTERNAL_B_SLOT_START..EXTERNAL_B_SLOT_START + 4 * SLOT_WIDTH).contains(&start)
    {
        assert_eq!((start - EXTERNAL_B_SLOT_START) % SLOT_WIDTH, 0);
        format!(".externalB {}", (start - EXTERNAL_B_SLOT_START) / SLOT_WIDTH)
    } else if matches!(class, LeafClass::Chained) && (FINAL_SLOT_START..FINAL_SLOT_END).contains(&start) {
        assert_eq!((start - FINAL_SLOT_START) % SLOT_WIDTH, 0);
        format!(".previousLocal {}", (start - FINAL_SLOT_START) / SLOT_WIDTH)
    } else if (local..local + LOCAL_SLOT_COUNT * SLOT_WIDTH).contains(&start) {
        assert_eq!((start - local) % SLOT_WIDTH, 0);
        format!(".local {}", (start - local) / SLOT_WIDTH)
    } else {
        panic!("unclassified geometric Poseidon2 final column {start}")
    }
}

fn write_raw_step(
    rendered: &mut String,
    class: LeafClass,
    index: usize,
    step: &SelectiveProjectedPoseidon2SboxStep,
) -> std::fmt::Result {
    writeln!(
        rendered,
        "\ndef rawStep{index:02} : RawStep where\n  rowOffset := {index}\n  input := {}\n  output := {}",
        lean_source_lc(class, step.input()),
        lean_source_lc(class, step.output()),
    )
}

fn lean_port(class: LeafClass, port: &SelectiveProjectedPort) -> String {
    assert!(port.seeded_blocks().is_empty());
    let explicit = port
        .explicit()
        .iter()
        .map(|term| {
            format!(
                "{{ column := {}, coefficient := {} }}",
                lean_explicit_column(class, *term),
                term.coefficient().as_canonical_u64()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let geometric = port
        .geometric_runs()
        .iter()
        .map(|run| {
            format!(
                "{{ slot := {}, initial := {}, ratio := {} }}",
                lean_slot(class, *run),
                run.initial().as_canonical_u64(),
                run.ratio().as_canonical_u64()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ explicit := [{explicit}], geometric := [{geometric}] }}")
}

fn lean_port_comparison(class: LeafClass, port: &SelectiveProjectedPort) -> String {
    let mut explicit = port
        .explicit()
        .iter()
        .map(|term| {
            (
                lean_explicit_column(class, *term),
                term.coefficient().as_canonical_u64(),
            )
        })
        .collect::<Vec<_>>();
    explicit.sort();
    let mut geometric = port
        .geometric_runs()
        .iter()
        .map(|run| {
            (
                lean_slot(class, *run),
                run.initial().as_canonical_u64(),
                run.ratio().as_canonical_u64(),
            )
        })
        .collect::<Vec<_>>();
    geometric.sort();
    format!("{explicit:?}:{geometric:?}")
}

fn write_raw_row(
    rendered: &mut String,
    class: LeafClass,
    index: usize,
    row: &SelectiveProjectedRowArtifact,
) -> std::fmt::Result {
    writeln!(
        rendered,
        "\ndef rawRow{index:02} : RawRow where\n  rowOffset := {index}\n  ports := ["
    )?;
    for (port_index, port) in row.ports().iter().enumerate() {
        let separator = if port_index == 0 { "    " } else { "  , " };
        writeln!(rendered, "{separator}{}", lean_port(class, port))?;
    }
    writeln!(rendered, "  ]")
}

fn render_steps_artifact(steps: &[SelectiveProjectedPoseidon2SboxStep]) -> String {
    assert_eq!(steps.len(), LOCAL_SLOT_COUNT);
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
/-! Generated file: source S-box expressions for one relative production\n\
PiRLC Poseidon2 leaf.\n\n\
Owns: all 86 exact Rust-projected source S-box expressions.\n\n\
Does not own: final rows, field semantics, replay-batch coverage, decoder\n\
soundness, recursive orchestration, or permission to remove constraints.\n\n\
Emits constraints: no.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
def schemaVersion : Nat := 1\n\
def sourceWidth : Nat := 600\n\
def slotWidth : Nat := 41\n\
def externalLaneCount : Nat := 4\n\
def rowCount : Nat := 86"
    )
    .expect("render Poseidon2 leaf header");
    for (index, step) in steps.iter().enumerate() {
        write_raw_step(&mut rendered, LeafClass::Direct, index, step).expect("render Poseidon2 leaf step");
    }
    writeln!(rendered, "\ndef rawSteps : List RawStep := [").expect("render Poseidon2 step list");
    for index in 0..steps.len() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}rawStep{index:02}").expect("render Poseidon2 step item");
    }
    writeln!(rendered, "]").expect("render Poseidon2 step list footer");
    writeln!(
        rendered,
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf"
    )
    .expect("render Poseidon2 step footer");
    rendered
}

fn render_rows_artifact(rows: &[SelectiveProjectedRowArtifact], start: usize, stop: usize, shard: usize) -> String {
    assert_eq!(rows.len(), LOCAL_SLOT_COUNT);
    assert_eq!(stop - start, LOCAL_SLOT_COUNT / 2);
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
/-! Generated file: final selective ports for production PiRLC Poseidon2 leaf\n\
row shard {shard}.\n\n\
Owns: exact Rust-projected final ports for relative rows {start} through {}.\n\n\
Does not own: source S-box semantics, replay-batch coverage, decoder soundness,\n\
recursive orchestration, or permission to remove constraints.\n\n\
Emits constraints: no.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema",
        stop - 1,
    )
    .expect("render Poseidon2 row-shard header");
    for index in start..stop {
        write_raw_row(&mut rendered, LeafClass::Direct, index, &rows[index]).expect("render Poseidon2 leaf row");
    }
    writeln!(rendered, "\ndef rawRows{shard} : List RawRow := [").expect("render Poseidon2 row-shard list");
    for index in start..stop {
        let separator = if index == start { "  " } else { ", " };
        writeln!(rendered, "{separator}rawRow{index:02}").expect("render Poseidon2 row item");
    }
    writeln!(
        rendered,
        "]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf"
    )
    .expect("render Poseidon2 leaf footer");
    rendered
}

fn render_chained_rows_artifact(
    rows: &[SelectiveProjectedRowArtifact],
    start: usize,
    stop: usize,
    shard: usize,
) -> String {
    assert_eq!(rows.len(), LOCAL_SLOT_COUNT);
    assert_eq!(stop - start, LOCAL_SLOT_COUNT / 2);
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
/-! Generated file: final selective ports for one chained production PiRLC\n\
Poseidon2 leaf row shard {shard}.\n\n\
Owns: exact Rust-projected final ports for relative rows {start} through {}.\n\n\
Does not own: source S-box semantics, replay-batch coverage, decoder soundness,\n\
recursive orchestration, or permission to remove constraints.\n\n\
Emits constraints: no.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeaf\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema",
        stop - 1,
    )
    .expect("render chained Poseidon2 row-shard header");
    for index in start..stop {
        write_raw_row(&mut rendered, LeafClass::Chained, index, &rows[index])
            .expect("render chained Poseidon2 leaf row");
    }
    writeln!(rendered, "\ndef rawRows{shard} : List RawRow := [").expect("render chained Poseidon2 row-shard list");
    for index in start..stop {
        let separator = if index == start { "  " } else { ", " };
        writeln!(rendered, "{separator}rawRow{index:02}").expect("render chained Poseidon2 row item");
    }
    writeln!(
        rendered,
        "]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeaf"
    )
    .expect("render chained Poseidon2 leaf footer");
    rendered
}

fn render_partial_rows_artifact(
    rows: &[SelectiveProjectedRowArtifact],
    start: usize,
    stop: usize,
    shard: usize,
) -> String {
    assert_eq!(rows.len(), LOCAL_SLOT_COUNT);
    assert_eq!(stop - start, LOCAL_SLOT_COUNT / 2);
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
/-! Generated file: final selective ports for one partial-start production\n\
PiRLC Poseidon2 leaf row shard {shard}.\n\n\
Owns: exact Rust-projected final ports for relative rows {start} through {}\n\
under the direct-leaf role normalization.\n\n\
Does not own: source S-box semantics, replay-batch coverage, decoder soundness,\n\
recursive orchestration, or permission to remove constraints.\n\n\
Emits constraints: no.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema",
        stop - 1,
    )
    .expect("render partial-start Poseidon2 row-shard header");
    for index in start..stop {
        write_raw_row(&mut rendered, LeafClass::Partial, index, &rows[index])
            .expect("render partial-start Poseidon2 leaf row");
    }
    writeln!(rendered, "\ndef rawRows{shard} : List RawRow := [")
        .expect("render partial-start Poseidon2 row-shard list");
    for index in start..stop {
        let separator = if index == start { "  " } else { ", " };
        writeln!(rendered, "{separator}rawRow{index:02}").expect("render partial-start Poseidon2 row item");
    }
    writeln!(
        rendered,
        "]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf"
    )
    .expect("render partial-start Poseidon2 leaf footer");
    rendered
}

fn render_partial_steps_artifact(steps: &[SelectiveProjectedPoseidon2SboxStep]) -> String {
    assert_eq!(steps.len(), LOCAL_SLOT_COUNT);
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
/-! Generated file: source S-box expressions for one partial-start production\n\
PiRLC Poseidon2 leaf under the direct-leaf role normalization.\n\n\
Owns: all 86 exact Rust-projected source S-box expressions.\n\n\
Does not own: final rows, field semantics, operand-order equivalence, decoder\n\
soundness, recursive orchestration, or permission to remove constraints.\n\n\
Emits constraints: no.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
def schemaVersion : Nat := 1\n\
def sourceWidth : Nat := 600\n\
def slotWidth : Nat := 41\n\
def externalLaneCount : Nat := 4\n\
def rowCount : Nat := 86"
    )
    .expect("render partial-start Poseidon2 leaf header");
    for (index, step) in steps.iter().enumerate() {
        write_raw_step(&mut rendered, LeafClass::Partial, index, step)
            .expect("render partial-start Poseidon2 leaf step");
    }
    writeln!(rendered, "\ndef rawSteps : List RawStep := [").expect("render partial-start Poseidon2 step list");
    for index in 0..steps.len() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}rawStep{index:02}").expect("render partial-start Poseidon2 step item");
    }
    writeln!(rendered, "]").expect("render partial-start Poseidon2 step list footer");
    writeln!(
        rendered,
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf"
    )
    .expect("render partial-start Poseidon2 step footer");
    rendered
}

fn render_chained_images_artifact(images: &[SelectiveProjectedSourceImage]) -> String {
    assert_eq!(images.len(), 4);
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
/-! Generated file: exact final low-norm images of the four prior-output\n\
lanes consumed by one chained production PiRLC Poseidon2 leaf.\n\n\
Does not own: source authority, row satisfaction, replay-batch coverage,\n\
recursive orchestration, or permission to remove constraints.\n\n\
Emits constraints: no.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeaf\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema"
    )
    .expect("render chained Poseidon2 image header");
    for (lane, image) in images.iter().enumerate() {
        assert_eq!(image.column(), SOURCE_COLUMN_END - 4 + lane);
        writeln!(
            rendered,
            "\ndef rawImage{lane} : RawSourceImage where\n  lane := {lane}\n  port := {}",
            lean_port(LeafClass::Chained, image.port()),
        )
        .expect("render chained Poseidon2 source image");
    }
    writeln!(
        rendered,
        "\ndef rawImages : List RawSourceImage := [rawImage0, rawImage1, rawImage2, rawImage3]\n\n\
end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeaf"
    )
    .expect("render chained Poseidon2 image footer");
    rendered
}

fn assert_leaf_artifacts_match_committed(
    steps: &[SelectiveProjectedPoseidon2SboxStep],
    rows: &[SelectiveProjectedRowArtifact],
    chained_rows: &[SelectiveProjectedRowArtifact],
    chained_images: &[SelectiveProjectedSourceImage],
    partial_steps: &[SelectiveProjectedPoseidon2SboxStep],
    partial_rows: &[SelectiveProjectedRowArtifact],
) {
    let rendered = [
        (STEPS_ARTIFACT_PATH, render_steps_artifact(steps)),
        (ROWS_0_ARTIFACT_PATH, render_rows_artifact(rows, 0, 43, 0)),
        (ROWS_1_ARTIFACT_PATH, render_rows_artifact(rows, 43, 86, 1)),
        (
            CHAINED_ROWS_0_ARTIFACT_PATH,
            render_chained_rows_artifact(chained_rows, 0, 43, 0),
        ),
        (
            CHAINED_ROWS_1_ARTIFACT_PATH,
            render_chained_rows_artifact(chained_rows, 43, 86, 1),
        ),
        (
            CHAINED_IMAGES_ARTIFACT_PATH,
            render_chained_images_artifact(chained_images),
        ),
        (
            PARTIAL_ROWS_0_ARTIFACT_PATH,
            render_partial_rows_artifact(partial_rows, 0, 43, 0),
        ),
        (
            PARTIAL_ROWS_1_ARTIFACT_PATH,
            render_partial_rows_artifact(partial_rows, 43, 86, 1),
        ),
        (
            PARTIAL_STEPS_ARTIFACT_PATH,
            render_partial_steps_artifact(partial_steps),
        ),
    ];
    let mut drifted = Vec::new();
    for (relative_path, artifact) in rendered {
        let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), relative_path);
        if artifact != std::fs::read_to_string(&path).unwrap_or_default() {
            let expected = format!("{path}.expected");
            std::fs::write(&expected, artifact).expect("write reviewed Poseidon2 leaf artifact shard");
            drifted.push(expected);
        }
    }
    if !drifted.is_empty() {
        panic!(
            "production PiRLC Poseidon2 leaf artifacts drifted; wrote {}",
            drifted.join(", ")
        );
    }
}

#[test]
fn production_even_replay_poseidon2_exports_exact_sbox_provenance() {
    let compiler = production_pi_rlc_family_body_compiler_audit().expect("production PiRLC compiler audit");
    let rewrite = compiler
        .rows()
        .rewrites()
        .iter()
        .find(|rewrite| {
            rewrite.arm() == 0
                && rewrite.kind() == SelectiveRewriteKind::Poseidon2
                && rewrite.emitted_rows() == (EMITTED_START..EMITTED_END)
        })
        .expect("first even replay Poseidon2 rewrite");
    assert_eq!(rewrite.source_rows(), &[SOURCE_START..SOURCE_END]);
    let chained_rewrite = compiler
        .rows()
        .rewrites()
        .iter()
        .find(|rewrite| {
            rewrite.arm() == 0
                && rewrite.kind() == SelectiveRewriteKind::Poseidon2
                && rewrite.emitted_rows() == (EMITTED_END..EMITTED_END + LOCAL_SLOT_COUNT)
        })
        .expect("second even replay Poseidon2 rewrite");
    assert_eq!(chained_rewrite.source_rows(), &[SOURCE_END..SOURCE_END + 600]);

    let selected_rows = (EMITTED_START..EMITTED_END + LOCAL_SLOT_COUNT).collect::<Vec<_>>();
    let chained_input_columns = (SOURCE_COLUMN_END - 4..SOURCE_COLUMN_END).collect::<Vec<_>>();
    let projected = production_pi_rlc_family_body_projected_rows_with_source_provenance(
        &selected_rows,
        0,
        &chained_input_columns,
        &[],
    )
    .expect("exact production Poseidon2 provenance");
    let source = projected
        .source_provenance()
        .expect("complete production source provenance");
    let steps = source.poseidon2_sbox_steps();
    let (steps, chained_steps) = steps.split_at(LOCAL_SLOT_COUNT);
    let (rows, chained_rows) = projected.row_artifacts().split_at(LOCAL_SLOT_COUNT);
    let chained_images = source.requested_source_images();

    assert_eq!(steps.len(), LOCAL_SLOT_COUNT);
    assert_eq!(chained_steps.len(), LOCAL_SLOT_COUNT);
    assert_eq!(rows.len(), LOCAL_SLOT_COUNT);
    assert_eq!(chained_rows.len(), LOCAL_SLOT_COUNT);
    for (direct, chained) in steps.iter().zip(chained_steps) {
        assert_eq!(
            lean_source_lc(LeafClass::Direct, direct.input()),
            lean_source_lc(LeafClass::Chained, chained.input()),
        );
        assert_eq!(
            lean_source_lc(LeafClass::Direct, direct.output()),
            lean_source_lc(LeafClass::Chained, chained.output()),
        );
    }
    let external_source_columns = steps
        .iter()
        .flat_map(|step| step.input().terms().iter().chain(step.output().terms()))
        .map(|term| term.column())
        .filter(|column| !(SOURCE_COLUMN_START..SOURCE_COLUMN_END).contains(column))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        external_source_columns,
        [1_559, 1_560, 1_561, 1_562, 166_308, 166_309, 166_310, 166_311]
            .into_iter()
            .collect()
    );
    for (offset, step) in steps.iter().enumerate() {
        assert_eq!(step.emitted_row(), EMITTED_START + offset);
        assert_eq!(step.rewrite_id(), rewrite.id().index());
        assert_eq!(step.source_rows(), &[(SOURCE_START, SOURCE_END)]);
        assert_eq!(step.output().terms().len(), 1);
    }
    let geometric_intervals = rows
        .iter()
        .flat_map(|row| row.ports())
        .flat_map(|port| port.geometric_runs())
        .map(|run| (run.column_start(), run.length()))
        .collect::<BTreeSet<_>>();
    let expected_geometric_intervals = (0..4)
        .map(|index| (EXTERNAL_A_SLOT_START + index * SLOT_WIDTH, SLOT_WIDTH))
        .chain((0..4).map(|index| (EXTERNAL_B_SLOT_START + index * SLOT_WIDTH, SLOT_WIDTH)))
        .chain((0..86).map(|index| (FINAL_SLOT_START + index * SLOT_WIDTH, SLOT_WIDTH)))
        .collect::<BTreeSet<_>>();
    assert_eq!(geometric_intervals, expected_geometric_intervals);
    for row in rows {
        for port in row.ports() {
            assert!(port
                .geometric_runs()
                .iter()
                .all(|run| expected_geometric_intervals.contains(&(run.column_start(), run.length()))));
            assert!(port.seeded_blocks().is_empty());
            assert!(port.explicit().iter().all(|term| {
                term.column() == 0
                    || term.column() == SELECTOR_COLUMN
                    || (FINAL_SLOT_START..FINAL_SLOT_END).contains(&term.column())
            }));
        }
    }
    for row in chained_rows {
        for port in row.ports() {
            for run in port.geometric_runs() {
                let _ = lean_slot(LeafClass::Chained, *run);
            }
        }
    }
    let partial_selected_rows = (ODD_EMITTED_START..ODD_EMITTED_START + LOCAL_SLOT_COUNT).collect::<Vec<_>>();
    let partial_projected =
        production_pi_rlc_family_body_projected_rows_with_source_provenance(&partial_selected_rows, 1, &[], &[])
            .expect("exact odd partial-start Poseidon2 provenance");
    let partial_source = partial_projected
        .source_provenance()
        .expect("complete odd partial-start source provenance");
    let partial_steps = partial_source.poseidon2_sbox_steps();
    let partial_rows = partial_projected.row_artifacts();
    assert_eq!(
        partial_projected.selector_columns(),
        [SELECTOR_COLUMN, PARTIAL_SELECTOR_COLUMN]
    );
    assert_eq!(partial_steps.len(), LOCAL_SLOT_COUNT);
    assert_eq!(partial_rows.len(), LOCAL_SLOT_COUNT);
    for (direct, partial) in steps.iter().zip(partial_steps) {
        assert_eq!(
            lean_source_lc_comparison(LeafClass::Direct, direct.input()),
            lean_source_lc_comparison(LeafClass::Partial, partial.input()),
        );
        assert_eq!(
            lean_source_lc_comparison(LeafClass::Direct, direct.output()),
            lean_source_lc_comparison(LeafClass::Partial, partial.output()),
        );
    }
    for (direct, partial) in rows.iter().zip(partial_rows) {
        assert_eq!(direct.ports().len(), partial.ports().len());
        for (direct_port, partial_port) in direct.ports().iter().zip(partial.ports()) {
            assert_eq!(
                lean_port_comparison(LeafClass::Direct, direct_port),
                lean_port_comparison(LeafClass::Partial, partial_port),
            );
        }
    }

    assert_leaf_artifacts_match_committed(steps, rows, chained_rows, chained_images, partial_steps, partial_rows);
}
