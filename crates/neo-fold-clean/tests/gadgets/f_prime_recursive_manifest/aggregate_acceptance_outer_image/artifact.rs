//! Challenge-sharded Lean rendering for the recursive acceptance outer image.
//!
//! Owns: lossless normalization of the audited 960-chunk direct-decoder image,
//! exact row-set reconciliation, and deterministic generated Lean shards.
//!
//! Does not own: production extraction, semantic proofs, artifact promotion,
//! or permission to remove constraints.
//!
//! Emits constraints: no.
//!
//! | Generated branch | Records per shard | Exact content |
//! |---|---:|---|
//! | shape | 1 | dimensions, gate arity, and fixed census |
//! | challenge | 64 chunks | source/encoded placement, 16 decoders, Boolean owner rows |
//! | facade | 15 challenge shards | canonical ordered flattening |

use std::collections::BTreeSet;
use std::fmt::Write as _;

use neo_fold_clean::frontends::f_prime::gadget_native::{
    AggregateAcceptanceBooleanRowOwner, AggregateAcceptanceDecodedImage, AggregateAcceptanceOuterImageAudit,
    GadgetNativeBooleanFamily,
};

const CHALLENGES: usize = 15;
const CHUNKS_PER_CHALLENGE: usize = 64;
const INPUTS_PER_CHUNK: usize = 16;
const ACTIVE_ROWS_PER_CHUNK: usize = 9;
const OUTPUTS_PER_CHUNK: usize = 14;
const GENERATED_MODULE: &str =
    "Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.Generated.AggregateAcceptanceOuterImage";
pub(super) const GENERATED_ROOT: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/PiRlcChallenge/Generated/AggregateAcceptanceOuterImage";
pub(super) const GENERATED_DATA_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/PiRlcChallenge/Generated/AggregateAcceptanceOuterImageData.lean";

pub(super) struct RenderedFile {
    pub relative_path: String,
    pub contents: String,
}

fn validate_row_sets(audit: &AggregateAcceptanceOuterImageAudit) {
    let mut expected_source = BTreeSet::new();
    let mut expected_physical = BTreeSet::new();
    for chunk in &audit.chunks {
        expected_source.extend(chunk.source_rows.clone());
        expected_physical.extend(chunk.active_rows.clone());
        for bit in &chunk.bits {
            expected_source.insert(bit.source_boolean_row);
            expected_physical.insert(bit.boolean_owner.encoded_row());
        }
    }
    assert_eq!(
        expected_source,
        audit.source_rows.iter().map(|row| row.row).collect(),
        "every exported source row must have exactly one outer-image reason",
    );
    assert_eq!(
        expected_physical,
        audit.physical_rows.iter().map(|row| row.row).collect(),
        "every exported physical row must have exactly one outer-image reason",
    );
}

fn validate_profile(audit: &AggregateAcceptanceOuterImageAudit) {
    assert_eq!(audit.source_row_count, 7_169_252);
    assert_eq!(audit.source_columns, 7_100_181);
    assert_eq!(audit.encoded_rows, 7_253_817);
    assert_eq!(audit.encoded_columns, 9_820_662);
    assert_eq!(audit.matrix_arity, 56);
    assert_eq!(audit.chunks.len(), CHALLENGES * CHUNKS_PER_CHALLENGE);
    assert!(audit.linear_definitions.is_empty());
    assert_eq!(audit.source_rows.len(), 19_200);
    assert_eq!(audit.physical_rows.len(), 16_320);
    validate_row_sets(audit);

    let mut singleton = 0usize;
    let mut left = 0usize;
    let mut right = 0usize;
    for chunk in &audit.chunks {
        assert_eq!(chunk.bits.len(), INPUTS_PER_CHUNK);
        assert_eq!(chunk.source_rows.len(), 4);
        assert_eq!(chunk.encoded_outputs.len(), OUTPUTS_PER_CHUNK);
        assert_eq!(chunk.active_rows.len(), ACTIVE_ROWS_PER_CHUNK);
        for bit in &chunk.bits {
            match &bit.decoded {
                AggregateAcceptanceDecodedImage::Singleton { .. } => {
                    assert!(bit.linear_definition_columns.is_empty());
                    singleton += 1;
                }
                AggregateAcceptanceDecodedImage::SparseLinear { .. } => {
                    panic!("fixed recursive profile must use direct singleton decoders")
                }
            }
            match bit.boolean_owner {
                AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft { family, .. } => {
                    assert_eq!(family, GadgetNativeBooleanFamily::Common);
                    left += 1;
                }
                AggregateAcceptanceBooleanRowOwner::CoordinatePairRight { family, .. } => {
                    assert_eq!(family, GadgetNativeBooleanFamily::Common);
                    right += 1;
                }
                AggregateAcceptanceBooleanRowOwner::CoordinateTail { .. } => {
                    panic!("fixed recursive acceptance has no Boolean tail owner")
                }
                AggregateAcceptanceBooleanRowOwner::TranslatedSource { .. } => {
                    panic!("fixed recursive profile has no translated Boolean owner")
                }
            }
        }
    }
    assert_eq!(singleton, 15_360);
    assert_eq!(left, 7_680);
    assert_eq!(right, 7_680);
}

fn header(summary: &str, table: &str) -> String {
    format!(
        "/-! Generated by `gadgets_f_prime_recursive_manifest`; do not hand-edit.\n\n\
{summary}\n\n\
Emits constraints: no. Generated values are non-authoritative production\n\
evidence consumed by handwritten refinement proofs.\n\n\
{table}\n-/\n\n"
    )
}

fn render_shape(audit: &AggregateAcceptanceOuterImageAudit) -> String {
    let mut rendered = String::from(
        "import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptanceOuterImageSchema\n\n",
    );
    rendered.push_str(&header(
        "Owns: exact fixed-profile dimensions and direct-decoder/row census.",
        "| Data branch | Exact value |\n|---|---:|\n| challenges/chunks | 15 / 960 |\n| source rows/columns | 7,169,252 / 7,100,181 |\n| encoded rows/columns | 7,253,817 / 9,820,662 |\n| direct decoders | 15,360 |",
    ));
    writeln!(
        rendered,
        "namespace {GENERATED_MODULE}.Shape\n\nopen Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageArtifact\n"
    )
    .expect("render shape namespace");
    let constants = [
        ("schemaVersion", 2usize),
        ("challengeCount", CHALLENGES),
        ("chunksPerChallenge", CHUNKS_PER_CHALLENGE),
        ("chunkCount", audit.chunks.len()),
        ("inputsPerChunk", INPUTS_PER_CHUNK),
        ("outputsPerChunk", OUTPUTS_PER_CHUNK),
        ("activeRowsPerChunk", ACTIVE_ROWS_PER_CHUNK),
        ("sourceRowCount", audit.source_row_count),
        ("sourceColumnCount", audit.source_columns),
        ("encodedRowCount", audit.encoded_rows),
        ("encodedColumnCount", audit.encoded_columns),
        ("matrixArity", audit.matrix_arity),
        ("selectedSourceRowCount", audit.source_rows.len()),
        ("selectedPhysicalRowCount", audit.physical_rows.len()),
        ("directDecoderCount", 15_360),
        ("pairLeftOwnerCount", 7_680),
        ("pairRightOwnerCount", 7_680),
    ];
    for (name, value) in constants {
        writeln!(rendered, "def {name} : Nat := {value}").expect("render shape constant");
    }
    writeln!(rendered, "\nend {GENERATED_MODULE}.Shape").expect("close shape");
    rendered
}

fn render_owner(owner: AggregateAcceptanceBooleanRowOwner) -> String {
    match owner {
        AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft {
            encoded_row,
            paired_column,
            ..
        } => format!("(.pairLeft {encoded_row} {paired_column})"),
        AggregateAcceptanceBooleanRowOwner::CoordinatePairRight {
            encoded_row,
            paired_column,
            ..
        } => format!("(.pairRight {encoded_row} {paired_column})"),
        AggregateAcceptanceBooleanRowOwner::CoordinateTail { .. } => {
            unreachable!("validated fixed profile excludes tails")
        }
        AggregateAcceptanceBooleanRowOwner::TranslatedSource { .. } => {
            unreachable!("validated fixed profile excludes translated rows")
        }
    }
}

fn render_bit(bit: &neo_fold_clean::frontends::f_prime::gadget_native::AggregateAcceptanceBitOuterImage) -> String {
    let encoded_column = match bit.decoded {
        AggregateAcceptanceDecodedImage::Singleton { encoded_column } => encoded_column,
        AggregateAcceptanceDecodedImage::SparseLinear { .. } => {
            unreachable!("validated fixed profile excludes sparse decoders")
        }
    };
    format!(
        "{{ sourceColumn := {}, sourceBooleanRow := {}, encodedColumn := {}, owner := {} }}",
        bit.source_column,
        bit.source_boolean_row,
        encoded_column,
        render_owner(bit.boolean_owner),
    )
}

fn render_challenge(audit: &AggregateAcceptanceOuterImageAudit, challenge: usize) -> String {
    let module = format!("Challenge{challenge:02}");
    let mut rendered = format!("import {GENERATED_MODULE}.Shape\n\n");
    rendered.push_str(&header(
        &format!(
            "Owns: exact source/encoded placement and Boolean-row ownership for\nPi_RLC challenge {challenge}, sampler chunks 0 through 63."
        ),
        "| Data branch | Records | Exact leaf shape |\n|---|---:|---:|\n| chunk outer images | 64 | 16 inputs / 14 outputs / 9 active rows |",
    ));
    writeln!(
        rendered,
        "set_option maxRecDepth 10000\n\nnamespace {GENERATED_MODULE}.{module}\n\nopen Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageArtifact"
    )
    .expect("render challenge import");
    let start = challenge * CHUNKS_PER_CHALLENGE;
    for (local, chunk) in audit.chunks[start..start + CHUNKS_PER_CHALLENGE]
        .iter()
        .enumerate()
    {
        writeln!(
            rendered,
            "\ndef chunk{local:02} : ChunkOuterImage :=\n  {{ sourceRowStart := {}",
            chunk.source_rows.start
        )
        .expect("render chunk source row");
        writeln!(
            rendered,
            "    sourceAcceptColumn := {}\n    sourceInverseColumn := {}\n    bits := [",
            chunk.source_accept_column, chunk.source_inverse_column,
        )
        .expect("render chunk source columns");
        for (line, bits) in chunk.bits.chunks(4).enumerate() {
            let bit_separator = if line == 0 { "      " } else { "    , " };
            writeln!(
                rendered,
                "{bit_separator}{}",
                bits.iter().map(render_bit).collect::<Vec<_>>().join(", "),
            )
            .expect("render chunk bits");
        }
        writeln!(
            rendered,
            "    ]\n    encodedAccept := {}\n    encodedOutputStart := {}\n    activeRowStart := {} }}",
            chunk.encoded_accept, chunk.encoded_outputs.start, chunk.active_rows.start,
        )
        .expect("render chunk placement");
    }
    writeln!(rendered, "\ndef chunks : List ChunkOuterImage := [").expect("open chunk index");
    for local in 0..CHUNKS_PER_CHALLENGE {
        let separator = if local == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}chunk{local:02}").expect("render chunk reference");
    }
    writeln!(rendered, "]\n\nend {GENERATED_MODULE}.{module}").expect("close challenge");
    rendered
}

fn render_data_facade() -> String {
    let mut rendered = String::new();
    for challenge in 0..CHALLENGES {
        writeln!(rendered, "import {GENERATED_MODULE}.Challenge{challenge:02}").expect("render challenge import");
    }
    rendered.push('\n');
    rendered.push_str(&header(
        "Owns: the canonical challenge order and flattened 960-chunk generated\nouter-image payload.",
        "| Child | Count |\n|---|---:|\n| challenge shards | 15 |\n| flattened chunks | 960 |",
    ));
    writeln!(
        rendered,
        "namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageData\n\nopen Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageArtifact\n\ndef challenges : List (List ChunkOuterImage) := ["
    )
    .expect("render facade namespace");
    for challenge in 0..CHALLENGES {
        let separator = if challenge == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}{GENERATED_MODULE}.Challenge{challenge:02}.chunks")
            .expect("render challenge reference");
    }
    rendered.push_str(
        "]\n\ndef chunks : List ChunkOuterImage := challenges.flatten\n\nend Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageData\n",
    );
    rendered
}

pub(super) fn render(audit: &AggregateAcceptanceOuterImageAudit) -> Vec<RenderedFile> {
    validate_profile(audit);
    let mut files = vec![RenderedFile {
        relative_path: format!("{GENERATED_ROOT}/Shape.lean"),
        contents: render_shape(audit),
    }];
    for challenge in 0..CHALLENGES {
        files.push(RenderedFile {
            relative_path: format!("{GENERATED_ROOT}/Challenge{challenge:02}.lean"),
            contents: render_challenge(audit, challenge),
        });
    }
    files.push(RenderedFile {
        relative_path: GENERATED_DATA_PATH.to_owned(),
        contents: render_data_facade(),
    });
    files
}
