use neo_fold_clean::engine::r1cs_circuit::builder::Poseidon2PermutationAudit;
use p3_field::PrimeField64;

use super::*;

#[path = "full_history_nested_owner_aggregate_support.rs"]
mod full_history_nested_owner_aggregate_support;
#[path = "full_history_recursive_accumulator_artifact_support.rs"]
mod full_history_recursive_accumulator_artifact_support;
#[path = "full_history_recursive_prelude_artifact_support.rs"]
mod full_history_recursive_prelude_artifact_support;

const RECURSIVE_TRANSCRIPT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRecursiveTranscriptArtifact.lean";
const TERMINAL_TRANSCRIPT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalTranscriptArtifact.lean";
const TERMINAL_RUNNING_LINK_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalRunningLinkArtifact.lean";
const NESTED_OWNER_PIECES_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistory";
const NESTED_OWNER_ARTIFACT_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistory";
const NESTED_OWNER_AGGREGATE_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryNestedOwners.lean";
const ALPHABET_TEMPLATE_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/AlphabetSampling/Generated/AlphabetSamplingResidualTemplateRows";
const ALPHABET_LANE_TEMPLATE_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/AlphabetSampling/Generated/AlphabetSamplingResidualTemplateLane.lean";
const ALPHABET_TEMPLATE_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/AlphabetSampling/AlphabetSamplingResidualTemplate.lean";
const OWNER_SHARD_ROW_LIMIT: usize = 450;
const ALPHABET_LANE_ROWS: usize = 104;
const ALPHABET_ACCEPTANCE_BOUND_ROWS: usize = 6;
const ALPHABET_SELECTION_INITIALIZE_ROWS: usize = 1;
const ALPHABET_SELECTION_OUTPUTS: usize = 54;
const ALPHABET_SELECTION_ROWS_PER_OUTPUT: usize = 12 + 33 + 3;
const ALPHABET_TAIL_ROWS: usize = ALPHABET_ACCEPTANCE_BOUND_ROWS
    + ALPHABET_SELECTION_INITIALIZE_ROWS
    + ALPHABET_SELECTION_OUTPUTS * ALPHABET_SELECTION_ROWS_PER_OUTPUT;
const ALPHABET_TEMPLATE_SHARD_SIZE: usize = 450;

struct NestedOwnerSpec {
    range_name: &'static str,
    occurrence: usize,
    module_suffix: &'static str,
    description: &'static str,
}

const NESTED_OWNER_SPECS: &[NestedOwnerSpec] = &[
    NestedOwnerSpec {
        range_name: "fprime.recursive.prelude",
        occurrence: 0,
        module_suffix: "RecursivePrelude",
        description: "recursive F-prime prelude",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fresh_digests",
        occurrence: 0,
        module_suffix: "RecursivePiCcsFreshDigests",
        description: "recursive Pi_CCS fresh-claim digest owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.running_authority",
        occurrence: 0,
        module_suffix: "RecursivePiCcsRunningAuthority",
        description: "recursive Pi_CCS running-parent authority owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.transcript",
        occurrence: 0,
        module_suffix: "RecursivePiCcsTranscript",
        description: "recursive Pi_CCS transcript owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fe_initial",
        occurrence: 0,
        module_suffix: "RecursivePiCcsFeInitial",
        description: "recursive Pi_CCS FE-initial owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fe_claim_and_sumcheck.optional_claim",
        occurrence: 0,
        module_suffix: "RecursivePiCcsFeOptionalClaim",
        description: "recursive Pi_CCS FE optional-claim owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fe_sumcheck",
        occurrence: 0,
        module_suffix: "RecursivePiCcsFeSumcheck",
        description: "recursive Pi_CCS FE SumCheck owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.nc_sumcheck",
        occurrence: 0,
        module_suffix: "RecursivePiCcsNcSumcheck",
        description: "recursive Pi_CCS NC SumCheck owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fe_terminal",
        occurrence: 0,
        module_suffix: "RecursivePiCcsFeTerminal",
        description: "recursive Pi_CCS FE-terminal owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.nc_terminal",
        occurrence: 0,
        module_suffix: "RecursivePiCcsNcTerminal",
        description: "recursive Pi_CCS NC-terminal owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.catchup",
        occurrence: 0,
        module_suffix: "RecursivePiCcsCatchup",
        description: "recursive Pi_CCS header catch-up owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.output_message_hashes",
        occurrence: 0,
        module_suffix: "RecursivePiCcsOutputMessageHashes",
        description: "recursive Pi_CCS output-message hash owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_rlc.transcript_rhos",
        occurrence: 0,
        module_suffix: "RecursivePiRlcTranscriptRhos",
        description: "recursive Pi_RLC transcript and rho owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_rlc.projection_binding",
        occurrence: 0,
        module_suffix: "RecursivePiRlcProjectionBinding",
        description: "recursive Pi_RLC projection-preimage binding owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fresh_digests",
        occurrence: 1,
        module_suffix: "TerminalPiCcsFreshDigests",
        description: "terminal Pi_CCS fresh-claim digest owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.running_authority",
        occurrence: 1,
        module_suffix: "TerminalPiCcsRunningAuthority",
        description: "terminal Pi_CCS running-parent authority owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.transcript",
        occurrence: 1,
        module_suffix: "TerminalPiCcsTranscript",
        description: "terminal Pi_CCS transcript owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fe_initial",
        occurrence: 1,
        module_suffix: "TerminalPiCcsFeInitial",
        description: "terminal Pi_CCS FE-initial owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fe_claim_and_sumcheck.optional_claim",
        occurrence: 1,
        module_suffix: "TerminalPiCcsFeOptionalClaim",
        description: "terminal Pi_CCS FE optional-claim owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fe_sumcheck",
        occurrence: 1,
        module_suffix: "TerminalPiCcsFeSumcheck",
        description: "terminal Pi_CCS FE SumCheck owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.nc_sumcheck",
        occurrence: 1,
        module_suffix: "TerminalPiCcsNcSumcheck",
        description: "terminal Pi_CCS NC SumCheck owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.fe_terminal",
        occurrence: 1,
        module_suffix: "TerminalPiCcsFeTerminal",
        description: "terminal Pi_CCS FE-terminal owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.nc_terminal",
        occurrence: 1,
        module_suffix: "TerminalPiCcsNcTerminal",
        description: "terminal Pi_CCS NC-terminal owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.catchup",
        occurrence: 1,
        module_suffix: "TerminalPiCcsCatchup",
        description: "terminal Pi_CCS header catch-up owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_ccs.output_message_hashes",
        occurrence: 1,
        module_suffix: "TerminalPiCcsOutputMessageHashes",
        description: "terminal Pi_CCS output-message hash owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_rlc.transcript_rhos",
        occurrence: 1,
        module_suffix: "TerminalPiRlcTranscriptRhos",
        description: "terminal Pi_RLC transcript and rho owner",
    },
    NestedOwnerSpec {
        range_name: "nifs.pi_rlc.projection_binding",
        occurrence: 1,
        module_suffix: "TerminalPiRlcProjectionBinding",
        description: "terminal Pi_RLC projection-preimage binding owner",
    },
];

enum Piece {
    Pin(usize),
    Call(usize),
}

struct TranscriptOwner {
    pins: Vec<(usize, u64)>,
    calls: Vec<Poseidon2PermutationAudit>,
    pieces: Vec<Piece>,
}

fn owner<'a>(builder: &'a R1csBuilder, name: &str) -> &'a RowFamilyRange {
    let matches = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "one {name} owner");
    matches[0]
}

fn constant_pin_at(builder: &R1csBuilder, row: usize) -> Option<(usize, u64)> {
    let (a, b, c) = builder.sparse_triplets();
    let a = a
        .iter()
        .filter(|(candidate, _, _)| *candidate == row)
        .map(|(_, column, coefficient)| (*column, coefficient.as_canonical_u64()))
        .collect::<Vec<_>>();
    let b = b
        .iter()
        .filter(|(candidate, _, _)| *candidate == row)
        .map(|(_, column, coefficient)| (*column, coefficient.as_canonical_u64()))
        .collect::<Vec<_>>();
    let c = c
        .iter()
        .filter(|(candidate, _, _)| *candidate == row)
        .map(|(_, column, coefficient)| (*column, coefficient.as_canonical_u64()))
        .collect::<Vec<_>>();
    if b != [(0, 1)] || !c.is_empty() {
        return None;
    }
    let output = a
        .iter()
        .filter(|(column, coefficient)| *column != 0 && *coefficient == 1)
        .map(|(column, _)| *column)
        .collect::<Vec<_>>();
    let [output] = output.as_slice() else {
        return None;
    };
    let value = builder.witness()[*output].as_canonical_u64();
    let expected = if value == 0 {
        vec![(*output, 1)]
    } else {
        vec![(*output, 1), (0, F::ORDER_U64 - value)]
    };
    (a == expected).then_some((*output, value))
}

fn transcript_owner(builder: &R1csBuilder, range: &RowFamilyRange) -> TranscriptOwner {
    let mut calls = builder
        .poseidon2_permutation_audits()
        .into_iter()
        .filter(|call| range.row_start <= call.row_start && call.row_end <= range.row_end)
        .collect::<Vec<_>>();
    calls.sort_unstable_by_key(|call| call.row_start);
    for pair in calls.windows(2) {
        assert!(pair[0].row_end <= pair[1].row_start, "transcript calls do not overlap");
    }
    let mut pins = Vec::new();
    let mut pieces = Vec::new();
    let mut cursor = range.row_start;
    let mut call_index = 0;
    while cursor < range.row_end {
        if call_index < calls.len() && calls[call_index].row_start == cursor {
            assert_eq!(
                calls[call_index].row_end - calls[call_index].row_start,
                600,
                "production Poseidon2 call row count"
            );
            pieces.push(Piece::Call(call_index));
            cursor = calls[call_index].row_end;
            call_index += 1;
        } else {
            let pin = constant_pin_at(builder, cursor)
                .unwrap_or_else(|| panic!("nonconstant transcript row {cursor} outside Poseidon2 calls"));
            pieces.push(Piece::Pin(pins.len()));
            pins.push(pin);
            cursor += 1;
        }
    }
    assert_eq!(call_index, calls.len(), "all transcript calls consumed");
    assert_eq!(
        pins.len() + 600 * calls.len(),
        range.row_end - range.row_start,
        "constant pins and Poseidon2 calls partition transcript owner"
    );
    TranscriptOwner { pins, calls, pieces }
}

fn lean_pin(pin: (usize, u64)) -> String {
    format!("({}, {})", pin.0, pin.1)
}

fn lean_call(call: &Poseidon2PermutationAudit, row_start: usize) -> String {
    format!(
        "{{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }}",
        call.row_start - row_start,
        call.row_end - row_start,
        lean_nat_list(call.input_cols),
        call.first_allocated_col,
    )
}

fn render_transcript(
    module_name: &str,
    description: &str,
    range: &RowFamilyRange,
    owner: &TranscriptOwner,
    range_sha256: &str,
    additional_metadata: &str,
) -> String {
    let pins = owner
        .pins
        .iter()
        .copied()
        .map(lean_pin)
        .collect::<Vec<_>>()
        .join(", ");
    let call_definitions = owner
        .calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            format!(
                "def call{index} : Poseidon2Call.Call :=\n  {}",
                lean_call(call, range.row_start),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let call_references = (0..owner.calls.len())
        .map(|index| format!("call{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let schedule = owner
        .pieces
        .iter()
        .map(|piece| match piece {
            Piece::Pin(index) => format!(".pin {index}"),
            Piece::Call(index) => format!(".call {index}"),
        })
        .collect::<Vec<_>>()
        .join(",\n   ");
    format!(
        "import Nightstream.Implementation.R1CS.Core.TranscriptCertificate\n\n\
         /-! Generated exact {description}. Hashes below are drift metadata only. -/\n\n\
         namespace Nightstream.Implementation.R1CS.{module_name}\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         set_option maxRecDepth 1048576\n\n\
         def rangeSha256 : String := \"{range_sha256}\"\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\n\
         {additional_metadata}\n\
         def constantPins : List (Nat × Nat) := [{pins}]\n\n\
         {call_definitions}\n\n\
         def calls : List Poseidon2Call.Call := [{call_references}]\n\n\
         def trace : TranscriptCertificate.Trace := ⟨constantPins, calls⟩\n\n\
         def schedule : List TranscriptCertificate.PieceRef :=\n  [{schedule}]\n\n\
         def rowPieces : List (List Row) :=\n\
         \x20 schedule.map fun piece => piece.rows trace\n\n\
         def ownerRows : List Row := trace.orderedRows schedule\n\n\
         theorem ownerRows_length : ownerRows.length = rowCount := by native_decide\n\n\
         def pinIndicesBoundedCheck : Bool :=\n\
         \x20 schedule.all fun piece =>\n\
         \x20   match piece with\n\
         \x20   | .pin index => decide (index < trace.pins.length)\n\
         \x20   | .call _ => true\n\n\
         def callIndicesBoundedCheck : Bool :=\n\
         \x20 schedule.all fun piece =>\n\
         \x20   match piece with\n\
         \x20   | .pin _ => true\n\
         \x20   | .call index => decide (index < trace.calls.length)\n\n\
         def everyPinScheduledCheck : Bool :=\n\
         \x20 (List.range trace.pins.length).all fun index =>\n\
         \x20   decide (.pin index ∈ schedule)\n\n\
         def everyCallScheduledCheck : Bool :=\n\
         \x20 (List.range trace.calls.length).all fun index =>\n\
         \x20   decide (.call index ∈ schedule)\n\n\
         theorem pinIndicesBounded_checked : pinIndicesBoundedCheck = true := by native_decide\n\
         theorem callIndicesBounded_checked : callIndicesBoundedCheck = true := by native_decide\n\
         theorem everyPinScheduled_checked : everyPinScheduledCheck = true := by native_decide\n\
         theorem everyCallScheduled_checked : everyCallScheduledCheck = true := by native_decide\n\n\
         theorem traceValid : trace.OrderedValid schedule ownerRows where\n\
         \x20 pinIndicesBounded := by\n\
         \x20   intro index member\n\
         \x20   exact of_decide_eq_true\n\
         \x20     ((List.all_eq_true.mp pinIndicesBounded_checked) (.pin index) member)\n\
         \x20 callIndicesBounded := by\n\
         \x20   intro index member\n\
         \x20   exact of_decide_eq_true\n\
         \x20     ((List.all_eq_true.mp callIndicesBounded_checked) (.call index) member)\n\
         \x20 everyPinScheduled := by\n\
         \x20   intro index indexLt\n\
         \x20   exact of_decide_eq_true\n\
         \x20     ((List.all_eq_true.mp everyPinScheduled_checked) index\n\
         \x20       (List.mem_range.mpr indexLt))\n\
         \x20 everyCallScheduled := by\n\
         \x20   intro index indexLt\n\
         \x20   exact of_decide_eq_true\n\
         \x20     ((List.all_eq_true.mp everyCallScheduled_checked) index\n\
         \x20       (List.mem_range.mpr indexLt))\n\
         \x20 pinValuesCanonical := by native_decide\n\
         \x20 exactRows := rfl\n\n\
         end Nightstream.Implementation.R1CS.{module_name}\n",
        range.row_start,
        range.row_end,
        range.row_end - range.row_start,
    )
}

fn compare(path: &Path, rendered: String, drifted: &mut Vec<PathBuf>) {
    if fs::read_to_string(path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        fs::write(&expected, rendered).expect("write transcript artifact");
        drifted.push(expected);
    }
}

fn recursive_context_metadata(
    builder: &R1csBuilder,
    _transcript_range: &RowFamilyRange,
    base: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
    recursive: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> String {
    assert!(base.is_base && !recursive.is_base, "base then recursive wire audits");
    assert_eq!(base.state_in_columns.len(), 31, "plain base state width");
    assert_eq!(recursive.state_in_columns.len(), 31, "plain recursive state width");
    let chunk_start = recursive
        .state_in_columns
        .last()
        .copied()
        .expect("recursive state columns")
        + 1;
    let next_chunk_digest = (chunk_start..chunk_start + 4).collect::<Vec<_>>();
    assert!(
        next_chunk_digest
            .iter()
            .all(|column| *column < builder.cols()),
        "derived next-chunk digest columns exist in the exact builder"
    );
    assert_eq!(
        recursive.prior_fresh_public_columns.len(),
        1,
        "fixed profile singleton latest claim"
    );
    format!(
        "structure ContextColumns where\n\
         \x20 vkFsDigest : List Nat\n\
         \x20 piCcsHeader : List Nat\n\
         \x20 chunkCount : Nat\n\
         \x20 stepCount : Nat\n\
         \x20 z0 : List Nat\n\
         \x20 zi : List Nat\n\
         \x20 pc : Nat\n\
         \x20 semanticState : List Nat\n\
         \x20 accumulatorDigest : List Nat\n\
         \x20 publicTrace : List Nat\n\
         \x20 initialSemanticState : List Nat\n\
         \x20 nextChunkDigest : List Nat\n\
         deriving DecidableEq, Repr\n\n\
         def contextColumns : ContextColumns :=\n\
         \x20 {{ vkFsDigest := {}\n\
         \x20   piCcsHeader := {}\n\
         \x20   chunkCount := {}\n\
         \x20   stepCount := {}\n\
         \x20   z0 := {}\n\
         \x20   zi := {}\n\
         \x20   pc := {}\n\
         \x20   semanticState := {}\n\
         \x20   accumulatorDigest := {}\n\
         \x20   publicTrace := {}\n\
         \x20   initialSemanticState := {}\n\
         \x20   nextChunkDigest := {} }}\n\n\
         def freshPublicColumns : List (List Nat) := {}\n\n",
        lean_nat_list(recursive.state_in_columns[0..4].iter().copied()),
        lean_nat_list(recursive.state_in_columns[4..8].iter().copied()),
        recursive.state_in_columns[8],
        recursive.state_in_columns[9],
        lean_nat_list(recursive.state_in_columns[10..14].iter().copied()),
        lean_nat_list(recursive.state_in_columns[14..18].iter().copied()),
        recursive.state_in_columns[18],
        lean_nat_list(recursive.state_in_columns[19..23].iter().copied()),
        lean_nat_list(recursive.state_in_columns[23..27].iter().copied()),
        lean_nat_list(recursive.state_in_columns[27..31].iter().copied()),
        lean_nat_list(base.state_in_columns[19..23].iter().copied()),
        lean_nat_list(next_chunk_digest),
        format!(
            "[{}]",
            recursive
                .prior_fresh_public_columns
                .iter()
                .map(|columns| lean_nat_list(columns.iter().copied()))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}

fn nth_owner<'a>(builder: &'a R1csBuilder, name: &str, occurrence: usize) -> &'a RowFamilyRange {
    let mut matches = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .collect::<Vec<_>>();
    matches.sort_unstable_by_key(|range| range.row_start);
    *matches
        .get(occurrence)
        .unwrap_or_else(|| panic!("missing {name} occurrence {occurrence}"))
}

fn lean_terms(terms: &[(usize, F)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|(column, coefficient)| format!("({column}, {})", coefficient.as_canonical_u64()))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_sparse_row(a: &[(usize, F)], b: &[(usize, F)], c: &[(usize, F)]) -> String {
    format!("⟨{}, {}, {}⟩", lean_terms(a), lean_terms(b), lean_terms(c))
}

fn triplet_row_slice<'a>(
    triplets: &'a [(usize, usize, F)],
    row_start: usize,
    row_end: usize,
) -> &'a [(usize, usize, F)] {
    let start = triplets.partition_point(|(row, _, _)| *row < row_start);
    let end = triplets.partition_point(|(row, _, _)| *row < row_end);
    &triplets[start..end]
}

fn mapped_ordinary_rows(
    builder: &R1csBuilder,
    row_start: usize,
    row_end: usize,
    column_map: impl Fn(usize) -> usize,
) -> Vec<String> {
    let (a, b, c) = builder.sparse_triplets();
    let a = triplet_row_slice(a, row_start, row_end);
    let b = triplet_row_slice(b, row_start, row_end);
    let c = triplet_row_slice(c, row_start, row_end);
    let mut ai = 0;
    let mut bi = 0;
    let mut ci = 0;
    (row_start..row_end)
        .map(|row| {
            let a_start = ai;
            let b_start = bi;
            let c_start = ci;
            while ai < a.len() && a[ai].0 == row {
                ai += 1;
            }
            while bi < b.len() && b[bi].0 == row {
                bi += 1;
            }
            while ci < c.len() && c[ci].0 == row {
                ci += 1;
            }
            let a_terms = a[a_start..ai]
                .iter()
                .map(|(_, column, coefficient)| (column_map(*column), *coefficient))
                .collect::<Vec<_>>();
            let b_terms = b[b_start..bi]
                .iter()
                .map(|(_, column, coefficient)| (column_map(*column), *coefficient))
                .collect::<Vec<_>>();
            let c_terms = c[c_start..ci]
                .iter()
                .map(|(_, column, coefficient)| (column_map(*column), *coefficient))
                .collect::<Vec<_>>();
            lean_sparse_row(&a_terms, &b_terms, &c_terms)
        })
        .collect()
}

fn ordinary_rows(builder: &R1csBuilder, row_start: usize, row_end: usize) -> Vec<String> {
    mapped_ordinary_rows(builder, row_start, row_end, |column| column)
}

fn shifted_ternary_reference() -> R1csBuilder {
    let mut isolated = R1csBuilder::new();
    let field = isolated.alloc(F::ZERO);
    neo_fold_clean::paper::reductions::accumulator_sis_circuit::enforce_commit_fields(
        &mut isolated,
        neo_fold_clean::paper::reductions::accumulator_sis_circuit::SIS_DIGEST_COMPRESSION_CONFIG,
        &[field],
    )
    .expect("one-field shifted-ternary reference");
    assert_eq!(isolated.rows(), 180, "isolated shifted-ternary row count");
    isolated
}

fn canonical_u64_reference() -> R1csBuilder {
    let mut isolated = R1csBuilder::new();
    let field = isolated.alloc(F::ZERO);
    neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits(&mut isolated, field);
    assert_eq!(isolated.rows(), 69, "isolated canonical-u64 row count");
    isolated
}

fn verify_shifted_ternary_rows(
    builder: &R1csBuilder,
    reference: &R1csBuilder,
    row_start: usize,
    field_column: usize,
    digit_start: usize,
) {
    let expected = mapped_ordinary_rows(reference, 2, 126, |column| {
        if column == 0 {
            0
        } else if column == 1 {
            field_column
        } else if 58 <= column {
            digit_start + column - 58
        } else {
            0
        }
    });
    let actual = ordinary_rows(builder, row_start, row_start + 124);
    assert_eq!(
        actual, expected,
        "shifted-ternary compact compiler changed exact rows at {row_start}"
    );
}

enum CompactOwnerPiece {
    Poseidon(Poseidon2PermutationAudit),
    SeededPhi81(usize, usize, usize),
    ShiftedTernary {
        row_start: usize,
        row_end: usize,
        field_column: usize,
        digit_start: usize,
    },
    CanonicalU64 {
        row_start: usize,
        row_end: usize,
        field_column: usize,
        bit_start: usize,
    },
    AlphabetLane {
        row_start: usize,
        row_end: usize,
        bit_start: usize,
        cum_prev: usize,
    },
    AlphabetTail {
        row_start: usize,
        row_end: usize,
        bit_starts: Vec<usize>,
        first_allocated: usize,
    },
}

impl CompactOwnerPiece {
    fn row_start(&self) -> usize {
        match self {
            Self::Poseidon(call) => call.row_start,
            Self::SeededPhi81(_, row_start, _) => *row_start,
            Self::ShiftedTernary { row_start, .. } => *row_start,
            Self::CanonicalU64 { row_start, .. } => *row_start,
            Self::AlphabetLane { row_start, .. } => *row_start,
            Self::AlphabetTail { row_start, .. } => *row_start,
        }
    }

    fn row_end(&self) -> usize {
        match self {
            Self::Poseidon(call) => call.row_end,
            Self::SeededPhi81(_, _, row_end) => *row_end,
            Self::ShiftedTernary { row_end, .. } => *row_end,
            Self::CanonicalU64 { row_end, .. } => *row_end,
            Self::AlphabetLane { row_end, .. } => *row_end,
            Self::AlphabetTail { row_end, .. } => *row_end,
        }
    }
}

fn canonical_u64_pieces(
    builder: &R1csBuilder,
    range: &RowFamilyRange,
    reference: &R1csBuilder,
) -> Vec<CompactOwnerPiece> {
    let audits = builder.canonical_u64_audits();
    let by_first_bit = audits
        .iter()
        .enumerate()
        .map(|(index, audit)| (audit.bit_cols[0], index))
        .collect::<std::collections::HashMap<_, _>>();
    let (_, b, _) = builder.sparse_triplets();
    let b = triplet_row_slice(b, range.row_start, range.row_end);
    let mut found = std::collections::HashSet::new();
    let mut pieces = Vec::new();
    for &(row, column, coefficient) in b {
        if coefficient != F::ONE || row + 69 > range.row_end {
            continue;
        }
        let Some(&audit_index) = by_first_bit.get(&column) else {
            continue;
        };
        if found.contains(&audit_index) {
            continue;
        }
        let audit = audits[audit_index];
        let bit_start = audit.bit_cols[0];
        assert!(
            audit
                .bit_cols
                .iter()
                .enumerate()
                .all(|(index, bit)| *bit == bit_start + index),
            "canonical-u64 bit columns are contiguous"
        );
        let expected = mapped_ordinary_rows(reference, 0, 69, |local| {
            if local == 0 {
                0
            } else if local == 1 {
                audit.field_col
            } else {
                bit_start + local - 2
            }
        });
        if ordinary_rows(builder, row, row + 69) != expected {
            continue;
        }
        found.insert(audit_index);
        pieces.push(CompactOwnerPiece::CanonicalU64 {
            row_start: row,
            row_end: row + 69,
            field_column: audit.field_col,
            bit_start,
        });
    }
    pieces
}

#[derive(Clone)]
struct AlphabetTemplates {
    lane_rows: Vec<String>,
    tail_rows: Vec<String>,
}

fn alphabet_lane_column_map(bit_start: usize, cum_prev: usize) -> Vec<usize> {
    std::iter::once(0)
        .chain(bit_start..bit_start + 64)
        .chain(std::iter::once(cum_prev))
        .chain(bit_start + 66..bit_start + 158)
        .collect()
}

fn alphabet_tail_inputs(bit_starts: &[usize]) -> Vec<usize> {
    assert_eq!(bit_starts.len(), 16, "one alphabet sample has sixteen digest lanes");
    let chunk_bases = bit_starts
        .iter()
        .flat_map(|bit_start| (0..4).map(move |chunk| bit_start + 66 + 23 * chunk));
    let bases = chunk_bases.collect::<Vec<_>>();
    std::iter::once(0)
        .chain(bases.iter().copied())
        .chain(bases.iter().map(|base| base + 21))
        .chain(bases.iter().map(|base| base + 22))
        .collect()
}

fn alphabet_tail_column_map(bit_starts: &[usize], first_allocated: usize) -> Vec<usize> {
    alphabet_tail_inputs(bit_starts)
        .into_iter()
        .chain(first_allocated..first_allocated + 3_516)
        .collect()
}

fn inverse_column_map(column_map: &[usize]) -> std::collections::HashMap<usize, usize> {
    let mut inverse = std::collections::HashMap::new();
    for (local, global) in column_map.iter().copied().enumerate() {
        assert!(
            inverse.insert(global, local).is_none(),
            "alphabet residual column map must be injective"
        );
    }
    inverse
}

fn alphabet_cum_prev(builder: &R1csBuilder, lane_row_start: usize, bit_start: usize) -> usize {
    let (a, b, c) = builder.sparse_triplets();
    let accept = bit_start + 66;
    let cum_after = bit_start + 88;
    let matches = (lane_row_start..lane_row_start + 26)
        .filter_map(|row| {
            let terms = triplet_row_slice(a, row, row + 1);
            let has_cum_after = terms
                .iter()
                .any(|(_, column, coefficient)| *column == cum_after && *coefficient == F::ONE);
            let has_accept = terms
                .iter()
                .any(|(_, column, coefficient)| *column == accept && *coefficient == -F::ONE);
            if !has_cum_after || !has_accept {
                return None;
            }
            let remaining = terms
                .iter()
                .filter(|(_, column, _)| *column != accept && *column != cum_after && *column != 0)
                .collect::<Vec<_>>();
            let [(_, cum_prev, coefficient)] = remaining.as_slice() else {
                return None;
            };
            (*coefficient == -F::ONE).then_some((row, *cum_prev, terms))
        })
        .collect::<Vec<_>>();
    let [(row, cum_prev, terms)] = matches.as_slice() else {
        panic!(
            "one exact cumulative recurrence in alphabet lane {lane_row_start}; \
             bit_start={bit_start}, accept={accept}, cum_after={cum_after}, matches={matches:?}"
        );
    };
    assert_eq!(
        triplet_row_slice(b, *row, *row + 1),
        [(*row, 0, F::ONE)],
        "alphabet cumulative recurrence multiplies by one"
    );
    assert!(
        triplet_row_slice(c, *row, *row + 1).is_empty(),
        "alphabet cumulative recurrence has zero C"
    );
    assert_eq!(
        terms.len(),
        3,
        "alphabet cumulative recurrence has exactly three A terms"
    );
    *cum_prev
}

fn alphabet_residual_pieces(
    builder: &R1csBuilder,
    range: &RowFamilyRange,
    canonical: &[CompactOwnerPiece],
    templates: &mut Option<AlphabetTemplates>,
) -> Vec<CompactOwnerPiece> {
    let canonical = canonical
        .iter()
        .filter_map(|piece| match piece {
            CompactOwnerPiece::CanonicalU64 {
                row_start,
                row_end,
                bit_start,
                ..
            } => Some((*row_start, *row_end, *bit_start)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if canonical.is_empty() {
        return Vec::new();
    }
    assert_eq!(canonical.len() % 16, 0, "complete alphabet-sampler u64 census");
    let mut pieces = Vec::new();
    for sample in canonical.chunks(16) {
        let bit_starts = sample.iter().map(|entry| entry.2).collect::<Vec<_>>();
        for (lane, &(row_start, row_end, bit_start)) in sample.iter().enumerate() {
            let lane_start = row_end;
            let lane_end = lane_start + ALPHABET_LANE_ROWS;
            assert!(lane_end <= range.row_end, "alphabet lane stays in owner");
            let cum_prev = alphabet_cum_prev(builder, lane_start, bit_start);
            if lane > 0 {
                assert_eq!(
                    cum_prev,
                    sample[lane - 1].2 + 157,
                    "alphabet cumulative-count chain crosses digest lanes"
                );
            }
            let column_map = alphabet_lane_column_map(bit_start, cum_prev);
            let inverse = inverse_column_map(&column_map);
            let local_rows = mapped_ordinary_rows(builder, lane_start, lane_end, |global| {
                *inverse
                    .get(&global)
                    .unwrap_or_else(|| panic!("unmapped alphabet-lane column {global}"))
            });
            match templates {
                Some(template) => assert_eq!(
                    local_rows, template.lane_rows,
                    "alphabet lane residual must reuse one exact compiler template"
                ),
                None => {
                    *templates = Some(AlphabetTemplates {
                        lane_rows: local_rows,
                        tail_rows: Vec::new(),
                    });
                }
            }
            pieces.push(CompactOwnerPiece::AlphabetLane {
                row_start: lane_start,
                row_end: lane_end,
                bit_start,
                cum_prev,
            });
            assert_eq!(row_start + 69, lane_start, "canonical-u64 then alphabet lane");
        }

        let tail_start = sample.last().expect("sample lanes").1 + ALPHABET_LANE_ROWS;
        let tail_end = tail_start + ALPHABET_TAIL_ROWS;
        assert!(
            tail_end <= range.row_end,
            "alphabet tail {tail_start}..{tail_end} stays in owner {}..{}",
            range.row_start,
            range.row_end
        );
        let first_allocated = bit_starts.last().copied().expect("sample bit starts") + 158;
        let column_map = alphabet_tail_column_map(&bit_starts, first_allocated);
        let inverse = inverse_column_map(&column_map);
        let local_rows = mapped_ordinary_rows(builder, tail_start, tail_end, |global| {
            *inverse
                .get(&global)
                .unwrap_or_else(|| {
                    panic!(
                        "unmapped alphabet-tail column {global}; tail={tail_start}..{tail_end}, first={first_allocated}, bits={bit_starts:?}"
                    )
                })
        });
        let template = templates.as_mut().expect("alphabet lane template");
        if template.tail_rows.is_empty() {
            template.tail_rows = local_rows;
        } else {
            assert_eq!(
                local_rows, template.tail_rows,
                "alphabet tail residual must reuse one exact compiler template"
            );
        }
        pieces.push(CompactOwnerPiece::AlphabetTail {
            row_start: tail_start,
            row_end: tail_end,
            bit_starts,
            first_allocated,
        });
    }
    pieces
}

struct RenderedOwnerPiece {
    row_start: usize,
    row_end: usize,
    rendered: String,
    rendered_lines: usize,
}

fn owner_pieces(
    builder: &R1csBuilder,
    range: &RowFamilyRange,
    shifted_reference: &R1csBuilder,
    u64_reference: &R1csBuilder,
    alphabet_templates: &mut Option<AlphabetTemplates>,
) -> Vec<RenderedOwnerPiece> {
    let mut compact = builder
        .poseidon2_permutation_audits()
        .into_iter()
        .filter(|call| range.row_start <= call.row_start && call.row_end <= range.row_end)
        .map(CompactOwnerPiece::Poseidon)
        .collect::<Vec<_>>();
    let canonical = canonical_u64_pieces(builder, range, u64_reference);
    let alphabet = alphabet_residual_pieces(builder, range, &canonical, alphabet_templates);
    compact.extend(canonical);
    compact.extend(alphabet);
    for (block_index, block) in builder.seeded_phi81_a_blocks().iter().enumerate() {
        if range.row_start <= block.row_start() && block.row_end() <= range.row_end {
            compact.push(CompactOwnerPiece::SeededPhi81(
                block_index,
                block.row_start(),
                block.row_end(),
            ));
        } else {
            assert!(
                block.row_end() <= range.row_start || range.row_end <= block.row_start(),
                "owner {} bisects compact seeded-Phi81 block {block_index}",
                range.name
            );
        }
    }
    for map in super::full_history_terminal_accumulator_artifact_support::ternary_maps(builder, range) {
        let columns = map
            .digit_cols
            .iter()
            .chain(&map.negative_cols)
            .chain(&map.borrow_cols)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(columns.len(), 122, "shifted-ternary mapped column width");
        let digit_start = columns[0];
        assert!(
            columns
                .iter()
                .enumerate()
                .all(|(index, column)| *column == digit_start + index),
            "shifted-ternary columns are one fresh contiguous interval"
        );
        verify_shifted_ternary_rows(
            builder,
            shifted_reference,
            range.row_start + map.row_start,
            map.field_col,
            digit_start,
        );
        compact.push(CompactOwnerPiece::ShiftedTernary {
            row_start: range.row_start + map.row_start,
            row_end: range.row_start + map.row_start + 124,
            field_column: map.field_col,
            digit_start,
        });
    }
    compact.sort_unstable_by_key(CompactOwnerPiece::row_start);
    for pair in compact.windows(2) {
        assert!(pair[0].row_end() <= pair[1].row_start(), "compact pieces overlap");
    }

    let mut pieces = Vec::new();
    let mut cursor = range.row_start;
    let push_ordinary = |pieces: &mut Vec<RenderedOwnerPiece>, start: usize, end: usize| {
        for chunk_start in (start..end).step_by(OWNER_SHARD_ROW_LIMIT) {
            let chunk_end = (chunk_start + OWNER_SHARD_ROW_LIMIT).min(end);
            let rows = ordinary_rows(builder, chunk_start, chunk_end);
            pieces.push(RenderedOwnerPiece {
                row_start: chunk_start,
                row_end: chunk_end,
                rendered: format!(
                    "{{ rowStart := {chunk_start}, rowEnd := {chunk_end}, payload := .ordinary [{}] }}",
                    rows.join(",\n      ")
                ),
                rendered_lines: rows.len() + 1,
            });
        }
    };
    for compact_piece in compact {
        if cursor < compact_piece.row_start() {
            push_ordinary(&mut pieces, cursor, compact_piece.row_start());
        }
        let (rendered, rendered_lines) = match &compact_piece {
            CompactOwnerPiece::Poseidon(call) => (
                format!(
                    "{{ rowStart := {}, rowEnd := {}, payload := .poseidon {{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }} }}",
                    call.row_start,
                    call.row_end,
                    call.row_start - range.row_start,
                    call.row_end - range.row_start,
                    lean_nat_list(call.input_cols),
                    call.first_allocated_col,
                ),
                1,
            ),
            CompactOwnerPiece::AlphabetLane {
                row_start,
                row_end,
                bit_start,
                cum_prev,
            } => (
                format!(
                    "{{ rowStart := {row_start}, rowEnd := {row_end}, payload := .ordinary (AlphabetSamplingResidualTemplate.laneRows {bit_start} {cum_prev}) }}"
                ),
                1,
            ),
            CompactOwnerPiece::AlphabetTail {
                row_start,
                row_end,
                bit_starts,
                first_allocated,
            } => (
                format!(
                    "{{ rowStart := {row_start}, rowEnd := {row_end}, payload := .ordinary (AlphabetSamplingResidualTemplate.tailRows {} {first_allocated}) }}",
                    lean_nat_list(bit_starts.iter().copied())
                ),
                1,
            ),
            CompactOwnerPiece::SeededPhi81(block_index, row_start, row_end) => (
                format!(
                    "{{ rowStart := {row_start}, rowEnd := {row_end}, payload := .seededPhi81 FPrimeFullHistorySeededPhi81.block{block_index} }}"
                ),
                1,
            ),
            CompactOwnerPiece::ShiftedTernary {
                row_start,
                row_end,
                field_column,
                digit_start,
            } => (
                format!(
                    "{{ rowStart := {row_start}, rowEnd := {row_end}, payload := .shiftedTernary {field_column} {digit_start} }}"
                ),
                1,
            ),
            CompactOwnerPiece::CanonicalU64 {
                row_start,
                row_end,
                field_column,
                bit_start,
            } => (
                format!(
                    "{{ rowStart := {row_start}, rowEnd := {row_end}, payload := .canonicalU64 {field_column} {bit_start} }}"
                ),
                1,
            ),
        };
        pieces.push(RenderedOwnerPiece {
            row_start: compact_piece.row_start(),
            row_end: compact_piece.row_end(),
            rendered,
            rendered_lines,
        });
        cursor = compact_piece.row_end();
    }
    if cursor < range.row_end {
        push_ordinary(&mut pieces, cursor, range.row_end);
    }
    assert_eq!(
        pieces.first().map(|piece| piece.row_start),
        (range.row_start < range.row_end).then_some(range.row_start)
    );
    assert_eq!(
        pieces.last().map(|piece| piece.row_end),
        (range.row_start < range.row_end).then_some(range.row_end)
    );
    for pair in pieces.windows(2) {
        assert_eq!(pair[0].row_end, pair[1].row_start, "owner pieces are contiguous");
    }
    pieces
}

fn shard_owner_pieces(pieces: Vec<RenderedOwnerPiece>) -> Vec<Vec<RenderedOwnerPiece>> {
    let mut shards = Vec::<Vec<RenderedOwnerPiece>>::new();
    let mut current = Vec::new();
    let mut lines = 0;
    for piece in pieces {
        if !current.is_empty() && lines + piece.rendered_lines > 500 {
            shards.push(std::mem::take(&mut current));
            lines = 0;
        }
        lines += piece.rendered_lines;
        current.push(piece);
    }
    if !current.is_empty() {
        shards.push(current);
    }
    shards
}

fn render_owner_shard(module_suffix: &str, shard: usize, pieces: &[RenderedOwnerPiece]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.Core.OwnerCertificate\n\
         import Nightstream.Implementation.R1CS.Ownership.AlphabetSampling.AlphabetSamplingResidualTemplate\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Artifact\n\n\
         /-! Generated exact ordered owner pieces, shard {shard}. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistory{module_suffix}.Generated\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.OwnerCertificate\n\n\
         set_option maxRecDepth 1048576\n\n\
         def pieces{shard} : List Piece :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistory{module_suffix}.Generated\n",
        pieces
            .iter()
            .map(|piece| piece.rendered.as_str())
            .collect::<Vec<_>>()
            .join(",\n   ")
    )
}

fn render_owner_artifact(
    spec: &NestedOwnerSpec,
    range: &RowFamilyRange,
    shard_count: usize,
    range_sha256: &str,
) -> String {
    let imports = (0..shard_count)
        .map(|shard| {
            format!(
                "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistory{}Pieces{shard}",
                spec.module_suffix
            )
        })
        .collect::<Vec<_>>();
    let imports = if imports.is_empty() {
        "import Nightstream.Implementation.R1CS.Ownership.Core.OwnerCertificate".to_string()
    } else {
        imports.join("\n")
    };
    let pieces = if shard_count == 0 {
        "[]".to_string()
    } else {
        (0..shard_count)
            .map(|shard| format!("Generated.pieces{shard}"))
            .collect::<Vec<_>>()
            .join(" ++\n    ")
    };
    let rows_proof = if shard_count == 0 {
        "by rfl".to_string()
    } else {
        "by\n  simpa [rows, rowCount, rowStart, rowEnd] using Owner.rows_length owner_valid".to_string()
    };
    format!(
        "{imports}\n\n\
         /-! Exact ordered row certificate for the {}. Hash is drift metadata only. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistory{}\n\n\
         open Nightstream.Implementation.R1CS.OwnerCertificate\n\n\
         set_option maxRecDepth 1048576\n\n\
         def rangeSha256 : String := \"{range_sha256}\"\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\n\
         def pieces : List Piece :=\n  {pieces}\n\n\
         def owner : Owner := ⟨rowStart, rowEnd, pieces⟩\n\n\
         theorem owner_valid : owner.Valid := by native_decide\n\n\
         def rows := owner.rows\n\n\
         theorem rows_length : rows.length = rowCount := {rows_proof}\n\n\
         /-- Independent executable semantics for every compact piece. -/\n\
         def Accepted (assignment : Nat → Nat) : Prop := owner.Accepted assignment\n\n\
         def check (assignment : Nat → Nat) : Bool := owner.check assignment\n\n\
         theorem check_eq_true_iff (assignment : Nat → Nat) :\n\
         \x20   check assignment = true ↔ Accepted assignment :=\n\
         \x20 Owner.check_eq_true_iff owner assignment\n\n\
         theorem sound {{assignment : Nat → Nat}}\n\
         \x20   (canonical : ∀ column, assignment column < goldilocksP)\n\
         \x20   (one : assignment 0 = 1)\n\
         \x20   (satisfies : Satisfies rows assignment) :\n\
         \x20   Accepted assignment :=\n\
         \x20 Owner.sound canonical one satisfies\n\n\
         theorem complete {{assignment : Nat → Nat}}\n\
         \x20   (canonical : ∀ column, assignment column < goldilocksP)\n\
         \x20   (one : assignment 0 = 1)\n\
         \x20   (accepted : Accepted assignment) :\n\
         \x20   Satisfies rows assignment :=\n\
         \x20 Owner.complete canonical one accepted\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistory{}\n",
        spec.description,
        spec.module_suffix,
        range.row_start,
        range.row_end,
        range.row_end - range.row_start,
        spec.module_suffix,
    )
}

fn render_alphabet_rows(definition: &str, rows: &[String]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.Semantics\n\n\
         /-! Generated exact residual rows for the production alphabet sampler. -/\n\n\
         namespace Nightstream.Implementation.R1CS.AlphabetSamplingResidualTemplate.Generated\n\n\
         def {definition} : List Row :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.AlphabetSamplingResidualTemplate.Generated\n",
        rows.join(",\n   ")
    )
}

fn render_alphabet_template(tail_shards: usize) -> String {
    let mut imports = vec![
        "import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateLane".to_string(),
        "import Nightstream.Implementation.R1CS.Core.Relabel".to_string(),
    ];
    imports
        .extend((0..tail_shards).map(|shard| {
            format!("import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows{shard}")
        }));
    let tail_rows = (0..tail_shards)
        .map(|shard| format!("Generated.tailRows{shard}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    format!(
        "{}\n\n\
         /-! Reusable exact checked-row templates for one alphabet-sampler lane and tail. -/\n\n\
         namespace Nightstream.Implementation.R1CS.AlphabetSamplingResidualTemplate\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         def laneTemplateRows : List Row := Generated.laneRows\n\n\
         def tailTemplateRows : List Row :=\n  {tail_rows}\n\n\
         def laneColumnMap (bitStart cumPrev : Nat) : List Nat :=\n\
         \x20 [0] ++ (List.range 64).map (fun index => bitStart + index) ++\n\
         \x20 [cumPrev] ++ (List.range 92).map (fun index => bitStart + 66 + index)\n\n\
         def chunkBases (bitStarts : List Nat) : List Nat :=\n\
         \x20 bitStarts.flatMap fun bitStart =>\n\
         \x20   (List.range 4).map fun chunk => bitStart + 66 + 23 * chunk\n\n\
         def tailInputColumns (bitStarts : List Nat) : List Nat :=\n\
         \x20 let bases := chunkBases bitStarts\n\
         \x20 [0] ++ bases ++ bases.map (fun base => base + 21) ++\n\
         \x20   bases.map (fun base => base + 22)\n\n\
         def tailColumnMap (bitStarts : List Nat) (firstAllocated : Nat) : List Nat :=\n\
         \x20 tailInputColumns bitStarts ++\n\
         \x20   (List.range 3516).map (fun index => firstAllocated + index)\n\n\
         def laneRows (bitStart cumPrev : Nat) : List Row :=\n\
         \x20 laneTemplateRows.map (Relabel.row (laneColumnMap bitStart cumPrev))\n\n\
         def tailRows (bitStarts : List Nat) (firstAllocated : Nat) : List Row :=\n\
         \x20 tailTemplateRows.map (Relabel.row (tailColumnMap bitStarts firstAllocated))\n\n\
         theorem laneTemplateRows_length : laneTemplateRows.length = {ALPHABET_LANE_ROWS} := by native_decide\n\n\
         theorem tailTemplateRows_length : tailTemplateRows.length = {ALPHABET_TAIL_ROWS} := by native_decide\n\n\
         theorem laneRows_length (bitStart cumPrev : Nat) :\n\
         \x20 (laneRows bitStart cumPrev).length = {ALPHABET_LANE_ROWS} := by\n\
         \x20 simp [laneRows, laneTemplateRows_length]\n\n\
         theorem tailRows_length (bitStarts : List Nat) (firstAllocated : Nat) :\n\
         \x20 (tailRows bitStarts firstAllocated).length = {ALPHABET_TAIL_ROWS} := by\n\
         \x20 simp [tailRows, tailTemplateRows_length]\n\n\
         end Nightstream.Implementation.R1CS.AlphabetSamplingResidualTemplate\n",
        imports.join("\n"),
    )
}

fn compare_alphabet_templates(templates: &AlphabetTemplates, drifted: &mut Vec<PathBuf>) {
    let root = formal_repo_root();
    compare(
        &root.join(ALPHABET_LANE_TEMPLATE_PATH),
        render_alphabet_rows("laneRows", &templates.lane_rows),
        drifted,
    );
    for (shard, rows) in templates
        .tail_rows
        .chunks(ALPHABET_TEMPLATE_SHARD_SIZE)
        .enumerate()
    {
        compare(
            &root.join(format!("{ALPHABET_TEMPLATE_PREFIX}{shard}.lean")),
            render_alphabet_rows(&format!("tailRows{shard}"), rows),
            drifted,
        );
    }
    compare(
        &root.join(ALPHABET_TEMPLATE_PATH),
        render_alphabet_template(
            templates
                .tail_rows
                .len()
                .div_ceil(ALPHABET_TEMPLATE_SHARD_SIZE),
        ),
        drifted,
    );
}

fn compare_nested_owner_artifacts(builder: &R1csBuilder, drifted: &mut Vec<PathBuf>) {
    let root = formal_repo_root();
    let shifted_reference = shifted_ternary_reference();
    let u64_reference = canonical_u64_reference();
    let mut alphabet_templates = None;
    for spec in NESTED_OWNER_SPECS {
        let range = nth_owner(builder, spec.range_name, spec.occurrence);
        let shards = shard_owner_pieces(owner_pieces(
            builder,
            range,
            &shifted_reference,
            &u64_reference,
            &mut alphabet_templates,
        ));
        for (shard, pieces) in shards.iter().enumerate() {
            compare(
                &root.join(format!(
                    "{NESTED_OWNER_PIECES_PREFIX}{}Pieces{shard}.lean",
                    spec.module_suffix
                )),
                render_owner_shard(spec.module_suffix, shard, pieces),
                drifted,
            );
        }
        compare(
            &root.join(format!(
                "{NESTED_OWNER_ARTIFACT_PREFIX}{}Artifact.lean",
                spec.module_suffix
            )),
            render_owner_artifact(spec, range, shards.len(), &range_hash(builder, range)),
            drifted,
        );
    }
    compare_alphabet_templates(
        alphabet_templates
            .as_ref()
            .expect("Pi_RLC alphabet template"),
        drifted,
    );
    compare(
        &root.join(NESTED_OWNER_AGGREGATE_PATH),
        full_history_nested_owner_aggregate_support::render_nested_owner_aggregate(),
        drifted,
    );
}

pub fn compare_transcript_artifacts(
    builder: &R1csBuilder,
    base: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
    recursive_audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) {
    let root = formal_repo_root();
    let recursive_range = owner(builder, "fprime.recursive.transcript");
    let terminal_range = owner(builder, "terminal.transcript");
    let running_link = owner(builder, "terminal.running_link");
    let recursive = transcript_owner(builder, recursive_range);
    let terminal = transcript_owner(builder, terminal_range);
    let context_metadata = recursive_context_metadata(builder, recursive_range, base, recursive_audit);
    let mut drifted = Vec::new();
    compare_nested_owner_artifacts(builder, &mut drifted);
    full_history_recursive_prelude_artifact_support::compare_recursive_prelude_hash_artifact(
        builder,
        recursive_audit,
        &mut drifted,
    );
    full_history_recursive_accumulator_artifact_support::compare_recursive_accumulator_artifacts(
        builder,
        recursive_audit,
    );
    compare(
        &root.join(RECURSIVE_TRANSCRIPT_PATH),
        render_transcript(
            "FPrimeFullHistoryRecursiveTranscriptArtifact",
            "recursive F-prime transcript owner",
            recursive_range,
            &recursive,
            &range_hash(builder, recursive_range),
            &context_metadata,
        ),
        &mut drifted,
    );
    compare(
        &root.join(TERMINAL_TRANSCRIPT_PATH),
        render_transcript(
            "FPrimeFullHistoryTerminalTranscriptArtifact",
            "terminal transcript-initialization owner",
            terminal_range,
            &terminal,
            &range_hash(builder, terminal_range),
            "",
        ),
        &mut drifted,
    );
    compare(
        &root.join(TERMINAL_RUNNING_LINK_PATH),
        render_equality_artifact(
            builder,
            running_link,
            "FPrimeFullHistoryTerminalRunningLink",
            "terminal running-accumulator digest continuity",
            &range_hash(builder, running_link),
        ),
        &mut drifted,
    );
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "full-history transcript artifacts drifted: {drifted:?}"
    );
}
