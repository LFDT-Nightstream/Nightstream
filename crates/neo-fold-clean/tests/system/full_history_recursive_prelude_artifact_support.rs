use neo_fold_clean::engine::r1cs_circuit::builder::{Poseidon2HashAudit, Poseidon2HashRoundAuditKind};

use super::*;

const PRELUDE_HASHES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryRecursivePreludeHashes.lean";

fn render_hash_trace(builder: &R1csBuilder, origin: usize, hash: &Poseidon2HashAudit) -> String {
    let calls = builder.poseidon2_permutation_audits();
    let rounds = hash
        .rounds
        .iter()
        .map(|round| {
            let call = calls
                .iter()
                .find(|call| {
                    call.input_cols == round.permutation_input_cols && call.output_cols == round.permutation_output_cols
                })
                .expect("recursive prelude sponge permutation");
            let kind = match &round.kind {
                Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => {
                    format!(".absorb {}", lean_nat_list(chunk_cols.iter().copied()))
                }
                Poseidon2HashRoundAuditKind::Pad => ".pad".to_string(),
            };
            format!(
                "{{ kind := {kind}, stateBeforeColumns := {}, permutationInputColumns := {}, \
                 permutationOutputColumns := {}, definingRows := {}, call := {{ rowStart := {}, \
                 rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }} }}",
                lean_nat_list(round.state_before_cols),
                lean_nat_list(round.permutation_input_cols),
                lean_nat_list(round.permutation_output_cols),
                lean_nat_list(round.defining_rows.iter().map(|row| row - origin)),
                call.row_start - origin,
                call.row_end - origin,
                lean_nat_list(call.input_cols),
                call.first_allocated_col,
            )
        })
        .collect::<Vec<_>>()
        .join("\n    , ");
    format!(
        "{{ inputColumns := {}, zeroColumn := {}, zeroRow := {}, rounds := [\n      {}\n    ], \
         outputColumns := {} }}",
        lean_nat_list(hash.input_cols.iter().copied()),
        hash.zero_col,
        hash.zero_row - origin,
        rounds,
        lean_nat_list(hash.output_cols),
    )
}

fn render(
    builder: &R1csBuilder,
    range: &RowFamilyRange,
    recursive: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> String {
    let hashes = builder
        .poseidon2_hash_audits()
        .into_iter()
        .filter(|hash| range.row_start <= hash.row_start && hash.row_end <= range.row_end)
        .collect::<Vec<_>>();
    assert_eq!(hashes.len(), 2, "claim-shape then chunk-shape digest");
    let claim = &hashes[0];
    let chunk = &hashes[1];
    let start_column = recursive.state_in_columns[9];
    let first_chunk_column = recursive
        .state_in_columns
        .last()
        .copied()
        .expect("recursive state inputs")
        + 1;
    let next_chunk_columns = (first_chunk_column..first_chunk_column + 4).collect::<Vec<_>>();
    let pairs = next_chunk_columns
        .iter()
        .copied()
        .zip(chunk.output_cols)
        .collect::<Vec<_>>();
    assert_eq!(
        chunk.row_end + 4 <= range.row_end,
        true,
        "chunk binding rows in prelude"
    );

    let constant_pins = (range.row_start..range.row_end)
        .filter_map(|row| constant_pin_at(builder, row))
        .collect::<Vec<_>>();
    let constant_columns = constant_pins
        .iter()
        .map(|pin| pin.0)
        .collect::<std::collections::HashSet<_>>();
    assert!(
        claim
            .input_cols
            .iter()
            .all(|column| constant_columns.contains(column)),
        "claim-shape digest inputs are verifier-owned constants"
    );
    let claim_outputs = claim
        .output_cols
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    assert!(
        chunk.input_cols.iter().all(|column| {
            *column == start_column || constant_columns.contains(column) || claim_outputs.contains(column)
        }),
        "chunk-shape digest reads only constants, start, and claim-shape digest"
    );
    let pins = constant_pins
        .iter()
        .map(|pin| format!("({}, {})", pin.0, pin.1))
        .collect::<Vec<_>>()
        .join(", ");
    let pairs = pairs
        .iter()
        .map(|pair| format!("({}, {})", pair.0, pair.1))
        .collect::<Vec<_>>()
        .join(", ");
    let claim_trace = render_hash_trace(builder, range.row_start, claim);
    let chunk_trace = render_hash_trace(builder, range.row_start, chunk);
    format!(
        "import Nightstream.Implementation.R1CS.Core.ConstantPins\n\
         import Nightstream.Implementation.R1CS.Core.EqualityPins\n\
         import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursivePreludeArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge\n\n\
         /-! Exact fixed-profile chunk-shape digest certificate for the recursive prelude. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePreludeHashes\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         set_option maxRecDepth 1048576\n\n\
         def startColumn : Nat := {start_column}\n\
         def nextChunkDigestColumns : List Nat := {}\n\
         def constantPins : List (Nat × Nat) := [{pins}]\n\
         def chunkDigestPairs : List (Nat × Nat) := [{pairs}]\n\n\
         def claimTrace : Poseidon2Sponge.Trace :=\n  {claim_trace}\n\n\
         def chunkTrace : Poseidon2Sponge.Trace :=\n  {chunk_trace}\n\n\
         theorem constantPins_canonical : ConstantPins.ValuesCanonical constantPins := by native_decide\n\
         theorem constantRows_included :\n\
         \x20 rowsIncluded (ConstantPins.rows constantPins)\n\
         \x20   FPrimeFullHistoryRecursivePrelude.rows = true := by native_decide\n\n\
         theorem chunkDigestRows_included :\n\
         \x20 rowsIncluded (EqualityPins.rows chunkDigestPairs)\n\
         \x20   FPrimeFullHistoryRecursivePrelude.rows = true := by native_decide\n\n\
         theorem claimTrace_valid :\n\
         \x20 claimTrace.Valid FPrimeFullHistoryRecursivePrelude.rows := by native_decide\n\n\
         theorem chunkTrace_valid :\n\
         \x20 chunkTrace.Valid FPrimeFullHistoryRecursivePrelude.rows := by native_decide\n\n\
         def claimInputValues : List Nat :=\n\
         \x20 claimTrace.inputColumns.map (ConstantPins.lookup constantPins)\n\n\
         def traceOutputPins (trace : Poseidon2Sponge.Trace)\n\
         \x20   (inputValues : List Nat) : List (Nat × Nat) :=\n\
         \x20 (List.range 4).map fun lane =>\n\
         \x20   (trace.outputColumns.getD lane 0,\n\
         \x20    Poseidon2Sponge.runValueRounds trace.rounds inputValues (fun _ => 0) lane)\n\n\
         def traceOutputKeys (trace : Poseidon2Sponge.Trace) : List Nat :=\n\
         \x20 (List.range 4).map fun lane => trace.outputColumns.getD lane 0\n\n\
         theorem traceOutputPins_keys (trace : Poseidon2Sponge.Trace)\n\
         \x20   (inputValues : List Nat) :\n\
         \x20   ConstantPins.keys (traceOutputPins trace inputValues) =\n\
         \x20     traceOutputKeys trace := by\n\
         \x20 simp [ConstantPins.keys, traceOutputPins, traceOutputKeys, List.map_map,\n\
         \x20   Function.comp_def]\n\n\
         def claimOutputPins : List (Nat × Nat) :=\n\
         \x20 traceOutputPins claimTrace claimInputValues\n\n\
         def fixedInputPins : List (Nat × Nat) :=\n\
         \x20 (startColumn, 1) :: constantPins ++ claimOutputPins\n\n\
         def fixedInputKeys : List Nat :=\n\
         \x20 startColumn :: ConstantPins.keys constantPins ++ traceOutputKeys claimTrace\n\n\
         def chunkInputValues : List Nat :=\n\
         \x20 chunkTrace.inputColumns.map (ConstantPins.lookup fixedInputPins)\n\n\
         def chunkDigestValue : List Nat :=\n\
         \x20 (List.range 4).map fun lane =>\n\
         \x20   Poseidon2Sponge.runValueRounds chunkTrace.rounds\n\
         \x20     chunkInputValues (fun _ => 0) lane\n\n\
         theorem claimInputs_covered :\n\
         \x20 ConstantPins.Covers claimTrace.inputColumns constantPins := by native_decide\n\n\
         theorem fixedInputPins_keys :\n\
         \x20   ConstantPins.keys fixedInputPins = fixedInputKeys := by\n\
         \x20 simp only [fixedInputPins, fixedInputKeys, ConstantPins.keys, List.map_cons,\n\
         \x20   List.map_append]\n\
         \x20 rw [show List.map Prod.fst claimOutputPins = traceOutputKeys claimTrace by\n\
         \x20   simpa [ConstantPins.keys, claimOutputPins] using\n\
         \x20     traceOutputPins_keys claimTrace claimInputValues]\n\n\
         theorem chunkInputKeys_covered :\n\
         \x20   ConstantPins.KeysCover chunkTrace.inputColumns fixedInputKeys := by\n\
         \x20 native_decide\n\n\
         theorem chunkInputs_covered :\n\
         \x20   ConstantPins.Covers chunkTrace.inputColumns fixedInputPins := by\n\
         \x20 rw [ConstantPins.covers_iff_keys, fixedInputPins_keys]\n\
         \x20 exact chunkInputKeys_covered\n\n\
         theorem next_chunk_digest_fixed\n\
         \x20   {{assignment : Nat → Nat}}\n\
         \x20   (canonical : ∀ column, assignment column < goldilocksP)\n\
         \x20   (one : assignment 0 = 1)\n\
         \x20   (start : assignment startColumn = 1)\n\
         \x20   (satisfies : Satisfies FPrimeFullHistoryRecursivePrelude.rows assignment) :\n\
         \x20   ∀ lane, lane < 4 →\n\
         \x20     assignment (nextChunkDigestColumns.getD lane 0) =\n\
         \x20       chunkDigestValue.getD lane 0 := by\n\
         \x20 have constants := ConstantPins.sound constantPins_canonical\n\
         \x20   constantRows_included canonical one satisfies\n\
         \x20 have equalities := EqualityPins.sound chunkDigestRows_included\n\
         \x20   canonical one satisfies\n\
         \x20 have claimInputEq :\n\
         \x20     claimTrace.inputColumns.map assignment = claimInputValues :=\n\
         \x20   ConstantPins.map_assignment_eq_lookup constants claimInputs_covered\n\
         \x20 have claimOutputs : ∀ pin ∈ claimOutputPins, assignment pin.1 = pin.2 := by\n\
         \x20   intro pin member\n\
         \x20   rcases List.mem_map.mp member with ⟨lane, laneMember, rfl⟩\n\
         \x20   have laneLt := List.mem_range.mp laneMember\n\
         \x20   simpa [traceOutputPins, claimInputEq] using\n\
         \x20     Poseidon2Sponge.trace_values_sound claimTrace_valid canonical one\n\
         \x20       satisfies lane laneLt\n\
         \x20 have fixedFacts : ∀ pin ∈ fixedInputPins, assignment pin.1 = pin.2 := by\n\
         \x20   intro pin member\n\
         \x20   simp only [fixedInputPins, List.mem_cons, List.mem_append] at member\n\
         \x20   rcases member with (startPin | constantPin) | claimPin\n\
         \x20   · subst pin\n\
         \x20     exact start\n\
         \x20   · exact constants pin constantPin\n\
         \x20   · exact claimOutputs pin claimPin\n\
         \x20 have chunkInputEq :\n\
         \x20     chunkTrace.inputColumns.map assignment = chunkInputValues :=\n\
         \x20   ConstantPins.map_assignment_eq_lookup fixedFacts chunkInputs_covered\n\
         \x20 have chunkOutputs := Poseidon2Sponge.trace_values_sound\n\
         \x20   chunkTrace_valid canonical one satisfies\n\
         \x20 intro lane laneLt\n\
         \x20 have pairMember :\n\
         \x20     (nextChunkDigestColumns.getD lane 0,\n\
         \x20       chunkTrace.outputColumns.getD lane 0) ∈ chunkDigestPairs := by\n\
         \x20   have cases : lane = 0 ∨ lane = 1 ∨ lane = 2 ∨ lane = 3 := by omega\n\
         \x20   rcases cases with rfl | rfl | rfl | rfl <;> native_decide\n\
         \x20 rw [equalities _ pairMember, chunkOutputs lane laneLt, chunkInputEq]\n\
         \x20 simp [chunkDigestValue, laneLt]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePreludeHashes\n",
        lean_nat_list(next_chunk_columns),
    )
}

pub fn compare_recursive_prelude_hash_artifact(
    builder: &R1csBuilder,
    recursive: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
    drifted: &mut Vec<PathBuf>,
) {
    let range = owner(builder, "fprime.recursive.prelude");
    compare(
        &formal_repo_root().join(PRELUDE_HASHES_PATH),
        render(builder, range, recursive),
        drifted,
    );
}
