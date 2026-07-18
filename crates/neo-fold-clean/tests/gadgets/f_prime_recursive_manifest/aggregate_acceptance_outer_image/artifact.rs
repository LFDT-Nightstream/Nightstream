//! Challenge-sharded Lean rendering for the recursive acceptance outer image.
//!
//! Owns: lossless normalization of the audited 960-chunk image, exact
//! row-set reconciliation, one shared 391-coefficient decoder vector, and
//! deterministic generated Lean shards.
//!
//! Does not own: production extraction, semantic proofs, artifact promotion,
//! or permission to remove constraints.
//!
//! Emits constraints: no.
//!
//! | Generated branch | Records per shard | Exact content |
//! |---|---:|---|
//! | shape | 1 | dimensions, gate arity, fixed census, shared decoder coefficients |
//! | definition shard | 48 × 15 | removed source definition rows and source LCs by challenge |
//! | challenge | 64 chunks | source/encoded placement, 16 decoders, Boolean owner rows |
//! | facade | 15 challenge shards | canonical ordered flattening |

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use neo_fold_clean::frontends::f_prime::gadget_native::{
    AggregateAcceptanceBooleanRowOwner, AggregateAcceptanceDecodedImage, AggregateAcceptanceLinearDefinitionAudit,
    AggregateAcceptanceOuterImageAudit, GadgetNativeBooleanFamily,
};
use neo_math::F;
use p3_field::PrimeField64;

const CHALLENGES: usize = 15;
const CHUNKS_PER_CHALLENGE: usize = 64;
const INPUTS_PER_CHUNK: usize = 16;
const ACTIVE_ROWS_PER_CHUNK: usize = 9;
const OUTPUTS_PER_CHUNK: usize = 14;
const LINEAR_WIDTH: usize = 391;
const LINEAR_DEFINITIONS: usize = 720;
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

fn signed_canonical(coefficient: u64) -> i128 {
    let canonical = coefficient as i128;
    let modulus = F::ORDER_U64 as i128;
    if canonical > modulus / 2 {
        canonical - modulus
    } else {
        canonical
    }
}

fn signed(coefficient: F) -> i128 {
    signed_canonical(coefficient.as_canonical_u64())
}

fn sparse_signature(terms: &[(usize, F)]) -> Vec<(usize, u64)> {
    let base = terms[0].0;
    terms
        .iter()
        .map(|&(column, coefficient)| (column - base, coefficient.as_canonical_u64()))
        .collect()
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
    for definition in &audit.linear_definitions {
        expected_source.insert(definition.source_row);
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

fn validate_profile(audit: &AggregateAcceptanceOuterImageAudit) -> Vec<Vec<(usize, u64)>> {
    assert_eq!(audit.source_row_count, 2_584_075);
    assert_eq!(audit.source_columns, 2_396_858);
    assert_eq!(audit.encoded_rows, 4_934_494);
    assert_eq!(audit.encoded_columns, 8_120_309);
    assert_eq!(audit.matrix_arity, 56);
    assert_eq!(audit.chunks.len(), CHALLENGES * CHUNKS_PER_CHALLENGE);
    assert_eq!(audit.linear_definitions.len(), LINEAR_DEFINITIONS);
    assert_eq!(audit.source_rows.len(), 19_920);
    assert_eq!(audit.physical_rows.len(), 16_560);
    let mut definition_widths = BTreeMap::new();
    for definition in &audit.linear_definitions {
        *definition_widths
            .entry(definition.terms.len())
            .or_insert(0usize) += 1;
    }
    assert_eq!(
        definition_widths,
        BTreeMap::from([(1usize, 240usize), (8, 240), (64, 240)]),
        "linear-definition width families",
    );
    validate_row_sets(audit);

    let mut sparse_patterns = BTreeMap::<Vec<(usize, u64)>, usize>::new();
    let mut singleton = 0usize;
    let mut linear = 0usize;
    let mut left = 0usize;
    let mut right = 0usize;
    let mut translated = 0usize;
    for (chunk_index, chunk) in audit.chunks.iter().enumerate() {
        assert_eq!(chunk.bits.len(), INPUTS_PER_CHUNK);
        assert_eq!(chunk.source_rows.len(), 4);
        assert_eq!(chunk.encoded_outputs.len(), OUTPUTS_PER_CHUNK);
        assert_eq!(chunk.active_rows.len(), ACTIVE_ROWS_PER_CHUNK);
        for (bit_index, bit) in chunk.bits.iter().enumerate() {
            let expected_linear = chunk_index % 4 == 3 && bit_index == 15;
            match &bit.decoded {
                AggregateAcceptanceDecodedImage::Singleton { .. } => {
                    assert!(!expected_linear, "recursive sparse-bit position drift");
                    assert!(bit.linear_definition_columns.is_empty());
                    singleton += 1;
                }
                AggregateAcceptanceDecodedImage::SparseLinear { terms } => {
                    assert!(expected_linear, "recursive sparse-bit position drift");
                    assert_eq!(terms.len(), LINEAR_WIDTH);
                    assert_eq!(bit.linear_definition_columns.len(), 3);
                    let signature = sparse_signature(terms);
                    *sparse_patterns.entry(signature).or_default() += 1;
                    linear += 1;
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
                AggregateAcceptanceBooleanRowOwner::TranslatedSource { source_row, .. } => {
                    assert_eq!(source_row, bit.source_boolean_row);
                    assert!(expected_linear);
                    translated += 1;
                }
            }
        }
    }
    assert_eq!(singleton, 15_120);
    assert_eq!(linear, 240);
    assert_eq!(left, 7_680);
    assert_eq!(right, 7_440);
    assert_eq!(translated, 240);
    assert_eq!(sparse_patterns.len(), 4);
    assert!(sparse_patterns.values().all(|&count| count == 60));
    let mut patterns = sparse_patterns.into_keys().collect::<Vec<_>>();
    patterns.sort_by_key(|pattern| pattern.last().map_or(0, |term| term.0));
    assert_eq!(
        patterns
            .iter()
            .map(|pattern| pattern.last().map_or(0, |term| term.0))
            .collect::<Vec<_>>(),
        [390, 551, 712, 873],
    );
    patterns
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

fn render_nat_list(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_sparse_patterns(patterns: &[Vec<(usize, u64)>]) -> String {
    let mut rendered = String::from("[\n");
    for (pattern_index, pattern) in patterns.iter().enumerate() {
        let separator = if pattern_index == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}[").expect("render pattern start");
        for (line, terms) in pattern.chunks(12).enumerate() {
            let term_separator = if line == 0 { "    " } else { "  , " };
            writeln!(
                rendered,
                "{term_separator}{}",
                terms
                    .iter()
                    .map(|&(offset, coefficient)| { format!("⟨{offset}, {}⟩", signed_canonical(coefficient)) })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .expect("render pattern terms");
        }
        writeln!(rendered, "  ]").expect("render pattern end");
    }
    rendered.push(']');
    rendered
}

fn render_shape(audit: &AggregateAcceptanceOuterImageAudit, patterns: &[Vec<(usize, u64)>]) -> String {
    let mut rendered = String::from(
        "import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptanceOuterImageSchema\n\n",
    );
    rendered.push_str(&header(
        "Owns: exact fixed-profile dimensions, row/decoder census, and four\nshared 391-term sparse decoder patterns.",
        "| Data branch | Exact value |\n|---|---:|\n| challenges/chunks | 15 / 960 |\n| source rows/columns | 2,584,075 / 2,396,858 |\n| encoded rows/columns | 4,934,494 / 8,120,309 |\n| sparse decoders | 4 patterns; 240 images × 391 terms |",
    ));
    writeln!(
        rendered,
        "namespace {GENERATED_MODULE}.Shape\n\nopen Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageArtifact\n"
    )
    .expect("render shape namespace");
    let constants = [
        ("schemaVersion", 1usize),
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
        ("linearDefinitionCount", audit.linear_definitions.len()),
        ("selectedSourceRowCount", audit.source_rows.len()),
        ("selectedPhysicalRowCount", audit.physical_rows.len()),
        ("singletonDecoderCount", 15_120),
        ("sparseDecoderCount", 240),
        ("pairLeftOwnerCount", 7_680),
        ("pairRightOwnerCount", 7_440),
        ("translatedOwnerCount", 240),
    ];
    for (name, value) in constants {
        writeln!(rendered, "def {name} : Nat := {value}").expect("render shape constant");
    }
    writeln!(
        rendered,
        "\ndef sparseLinearPatterns : List (List SourceLinearTerm) :=\n{}\n\nend {GENERATED_MODULE}.Shape",
        render_sparse_patterns(patterns),
    )
    .expect("render sparse patterns");
    rendered
}

fn render_definition(definition: &AggregateAcceptanceLinearDefinitionAudit) -> String {
    let terms = definition
        .terms
        .iter()
        .map(|&(column, coefficient)| format!("⟨{column}, {}⟩", signed(coefficient)))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{ sourceColumn := {}, sourceRow := {}, terms := [{terms}] }}",
        definition.source_column, definition.source_row,
    )
}

fn challenge_definition_columns(audit: &AggregateAcceptanceOuterImageAudit, challenge: usize) -> BTreeSet<usize> {
    let start = challenge * CHUNKS_PER_CHALLENGE;
    audit.chunks[start..start + CHUNKS_PER_CHALLENGE]
        .iter()
        .flat_map(|chunk| &chunk.bits)
        .flat_map(|bit| bit.linear_definition_columns.iter().copied())
        .collect()
}

fn render_definition_shard(audit: &AggregateAcceptanceOuterImageAudit, challenge: usize) -> String {
    let module = format!("Definitions{challenge:02}");
    let columns = challenge_definition_columns(audit, challenge);
    let definitions = audit
        .linear_definitions
        .iter()
        .filter(|definition| columns.contains(&definition.source_column))
        .collect::<Vec<_>>();
    assert_eq!(columns.len(), 48, "definition-column census per challenge");
    assert_eq!(definitions.len(), 48, "definition census per challenge");
    let mut rendered = format!("import {GENERATED_MODULE}.Shape\n\n");
    rendered.push_str(&header(
        &format!(
            "Owns: the 48 removed generic-linear source definitions reachable from\nPi_RLC challenge {challenge}, including their exact source rows."
        ),
        "| Data branch | Records | Terms/record |\n|---|---:|---:|\n| generic-linear provenance | 16 / 16 / 16 | 1 / 8 / 64 |",
    ));
    writeln!(
        rendered,
        "set_option maxRecDepth 10000\n\nnamespace {GENERATED_MODULE}.{module}\n\nopen Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageArtifact\n\ndef linearDefinitions : List LinearDefinition := ["
    )
    .expect("render definition import");
    for (index, definition) in definitions.iter().enumerate() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}{}", render_definition(definition)).expect("render definition");
    }
    writeln!(rendered, "]\n\nend {GENERATED_MODULE}.{module}").expect("close definitions");
    rendered
}

fn render_decoded(decoded: &AggregateAcceptanceDecodedImage, patterns: &[Vec<(usize, u64)>]) -> String {
    match decoded {
        AggregateAcceptanceDecodedImage::Singleton { encoded_column } => {
            format!("(.singleton {encoded_column})")
        }
        AggregateAcceptanceDecodedImage::SparseLinear { terms } => {
            let signature = sparse_signature(terms);
            let pattern = patterns
                .iter()
                .position(|candidate| candidate == &signature)
                .expect("validated sparse pattern");
            format!("(.sparseLinear {pattern} {})", terms[0].0)
        }
    }
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
        AggregateAcceptanceBooleanRowOwner::TranslatedSource {
            source_row,
            encoded_row,
        } => format!("(.translatedSource {source_row} {encoded_row})"),
    }
}

fn render_bit(
    bit: &neo_fold_clean::frontends::f_prime::gadget_native::AggregateAcceptanceBitOuterImage,
    patterns: &[Vec<(usize, u64)>],
) -> String {
    format!(
        "{{ sourceColumn := {}, sourceBooleanRow := {}, decoded := {}, definitionColumns := {}, owner := {} }}",
        bit.source_column,
        bit.source_boolean_row,
        render_decoded(&bit.decoded, patterns),
        render_nat_list(&bit.linear_definition_columns),
        render_owner(bit.boolean_owner),
    )
}

fn render_challenge(
    audit: &AggregateAcceptanceOuterImageAudit,
    patterns: &[Vec<(usize, u64)>],
    challenge: usize,
) -> String {
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
                bits.iter()
                    .map(|bit| render_bit(bit, patterns))
                    .collect::<Vec<_>>()
                    .join(", "),
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
        writeln!(rendered, "import {GENERATED_MODULE}.Definitions{challenge:02}").expect("render definition import");
    }
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
        "namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageData\n\nopen Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageArtifact\n\nabbrev sparseLinearPatterns := {GENERATED_MODULE}.Shape.sparseLinearPatterns\n\ndef definitionShards : List (List LinearDefinition) := ["
    )
    .expect("render facade namespace");
    for challenge in 0..CHALLENGES {
        let separator = if challenge == 0 { "  " } else { ", " };
        writeln!(
            rendered,
            "{separator}{GENERATED_MODULE}.Definitions{challenge:02}.linearDefinitions"
        )
        .expect("render definition reference");
    }
    rendered.push_str(
        "]\n\ndef linearDefinitions : List LinearDefinition := definitionShards.flatten\n\ndef challenges : List (List ChunkOuterImage) := [\n",
    );
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
    let patterns = validate_profile(audit);
    let mut files = vec![RenderedFile {
        relative_path: format!("{GENERATED_ROOT}/Shape.lean"),
        contents: render_shape(audit, &patterns),
    }];
    for challenge in 0..CHALLENGES {
        files.push(RenderedFile {
            relative_path: format!("{GENERATED_ROOT}/Definitions{challenge:02}.lean"),
            contents: render_definition_shard(audit, challenge),
        });
        files.push(RenderedFile {
            relative_path: format!("{GENERATED_ROOT}/Challenge{challenge:02}.lean"),
            contents: render_challenge(audit, &patterns, challenge),
        });
    }
    files.push(RenderedFile {
        relative_path: GENERATED_DATA_PATH.to_owned(),
        contents: render_data_facade(),
    });
    files
}
