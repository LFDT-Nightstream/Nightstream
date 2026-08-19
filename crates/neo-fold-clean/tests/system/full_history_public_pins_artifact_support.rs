use neo_fold_clean::engine::r1cs_circuit::builder::Poseidon2HashAudit;

use super::checked_program_artifact_support::Row as NormalizedRow;
use super::*;

const ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryPublicPinsArtifact.lean";
const SHARD_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPublicPinsInstructions";
const HASHES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPublicPinsPoseidonHashes.lean";
const SHARD_SIZE: usize = 1_200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pin {
    Zero(usize),
    Constant(usize, u64),
    Equal(usize, usize),
}

fn public_pins_range(builder: &R1csBuilder) -> &RowFamilyRange {
    let matches = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == "decider.public_pins")
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "one full-history public-pins owner");
    matches[0]
}

fn public_pins_hash(builder: &R1csBuilder, range: &RowFamilyRange) -> Poseidon2HashAudit {
    let matches = builder
        .poseidon2_hash_audits()
        .into_iter()
        .filter(|hash| range.row_start <= hash.row_start && hash.row_end <= range.row_end)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "one terminal state-x_out sponge");
    matches[0].clone()
}

fn check_pin(row: &NormalizedRow) -> Option<Pin> {
    if row.a.is_empty() && row.c.is_empty() {
        return None;
    }
    assert_eq!(row.b, vec![(0, 1)], "public pin check has non-unit B: {row:?}");
    assert!(row.c.is_empty(), "public pin check has nonempty C: {row:?}");
    let minus_one = F::ORDER_U64 - 1;
    match row.a.as_slice() {
        [(column, 1)] => Some(Pin::Zero(*column)),
        [first, second] => {
            let (output, other) = if first.1 == 1 {
                (first.0, *second)
            } else if second.1 == 1 {
                (second.0, *first)
            } else {
                panic!("public pin check has no unit output: {row:?}");
            };
            if other.0 == 0 {
                let value = if other.1 == 0 { 0 } else { F::ORDER_U64 - other.1 };
                if value == 0 {
                    Some(Pin::Zero(output))
                } else {
                    Some(Pin::Constant(output, value))
                }
            } else {
                assert_eq!(other.1, minus_one, "public pin equality coefficient");
                Some(Pin::Equal(output, other.0))
            }
        }
        _ => panic!("public pin check is not affine or trivial: {row:?}"),
    }
}

fn checks(program: &NormalizedProgram) -> (Vec<Pin>, Vec<NormalizedRow>) {
    let mut pins = Vec::new();
    let mut trivial = Vec::new();
    for instruction in &program.instructions {
        if let Instruction::Check(row) = instruction {
            match check_pin(row) {
                Some(pin) => pins.push(pin),
                None => trivial.push(row.clone()),
            }
        }
    }
    assert_eq!(
        pins.len() + trivial.len(),
        program.check_count,
        "every retained public-pin assertion is classified"
    );
    (pins, trivial)
}

fn lean_terms(terms: &[(usize, u64)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|&(column, coefficient)| format!("({column}, {coefficient})"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_rows(rows: &[NormalizedRow]) -> String {
    rows.iter()
        .map(|row| {
            format!(
                "⟨{}, {}, {}⟩",
                lean_terms(&row.a),
                lean_terms(&row.b),
                lean_terms(&row.c)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n   ")
}

fn lean_pins(pins: &[Pin]) -> String {
    pins.iter()
        .map(|pin| match pin {
            Pin::Zero(column) => format!(".zero {column}"),
            Pin::Constant(column, value) => format!(".constant {column} {value}"),
            Pin::Equal(left, right) => format!(".equal {left} {right}"),
        })
        .collect::<Vec<_>>()
        .join(",\n   ")
}

fn render_shard(index: usize, instructions: &[Instruction]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated public-pins instruction shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPins.Generated\n\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 262144\n\n\
         def instructions{index} : List Instruction :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPins.Generated\n",
        lean_instructions(instructions),
    )
}

fn render_artifact(
    program: &NormalizedProgram,
    range: &RowFamilyRange,
    pins: &[Pin],
    trivial: &[NormalizedRow],
) -> String {
    let shard_count = program.instructions.len().div_ceil(SHARD_SIZE);
    let imports = (0..shard_count)
        .map(|index| format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPublicPinsInstructions{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let instructions = (0..shard_count)
        .map(|index| format!("Generated.instructions{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    format!(
        "{imports}\n\
         import Nightstream.Implementation.R1CS.Core.AffinePins\n\
         import Nightstream.Implementation.R1CS.Core.TrivialRows\n\n\
         /-! Exact checked program for the full-history public-image pins. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPins\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 524288\n\n\
         def inputColumns : List Nat := {}\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\
         def definitionCount : Nat := {}\n\
         def checkCount : Nat := {}\n\n\
         def instructions : List Instruction :=\n    {instructions}\n\n\
         def rows : List Row := CheckedProgram.rows instructions\n\
         def pins : List AffinePins.Pin :=\n  [{}]\n\n\
         def trivialRows : List Row :=\n  [{}]\n\n\
         theorem instructions_length : instructions.length = rowCount := by native_decide\n\
         theorem rows_length : rows.length = rowCount := by native_decide\n\
         theorem definitions_length :\n\
             (definitions instructions).length = definitionCount := by native_decide\n\
         theorem checks_length :\n\
             (checks instructions).length = checkCount := by native_decide\n\
         theorem definitions_canonical :\n\
             ∀ definition ∈ definitions instructions, definition.Canonical := by native_decide\n\
         theorem definitions_wellFormed :\n\
             WellFormed inputColumns (definitions instructions) := by native_decide\n\
         theorem checks_reference :\n\
             ChecksReference (knownAfter inputColumns (definitions instructions))\n\
               instructions := by native_decide\n\
         theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide\n\
         theorem pin_rows_in_checks :\n\
             rowsIncluded (AffinePins.rows pins) (checks instructions) = true := by\n\
           native_decide\n\
         theorem trivial_rows_in_checks :\n\
             rowsIncluded trivialRows (checks instructions) = true := by native_decide\n\
         theorem checks_covered :\n\
             ∀ row ∈ checks instructions,\n\
               row ∈ AffinePins.rows pins ∨ row ∈ trivialRows := by native_decide\n\
         theorem trivial_rows_valid : TrivialRows.Valid trivialRows := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPins\n",
        lean_nat_list(program.input_columns.iter().copied()),
        range.row_start,
        range.row_end,
        range.row_end - range.row_start,
        program.definition_count,
        program.check_count,
        lean_pins(pins),
        lean_rows(trivial),
    )
}

fn render_hashes(builder: &R1csBuilder, range: &RowFamilyRange, hash: &Poseidon2HashAudit) -> String {
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
                .expect("public-pins sponge permutation call");
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
                lean_nat_list(round.defining_rows.iter().map(|row| row - range.row_start)),
                call.row_start - range.row_start,
                call.row_end - range.row_start,
                lean_nat_list(call.input_cols),
                call.first_allocated_col,
            )
        })
        .collect::<Vec<_>>()
        .join("\n    , ");
    let trace = format!(
        "{{ inputColumns := {}, zeroColumn := {}, zeroRow := {}, rounds := [\n      {}\n    ], \
         outputColumns := {} }}",
        lean_nat_list(hash.input_cols.iter().copied()),
        hash.zero_col,
        hash.zero_row - range.row_start,
        rounds,
        lean_nat_list(hash.output_cols),
    );
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPublicPinsArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge\n\n\
         /-! Generated sponge certificate for the exact public-image pins. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsPoseidonHashes\n\n\
         open Nightstream.Implementation.R1CS.Poseidon2Sponge\n\n\
         set_option maxRecDepth 524288\n\n\
         def xOutTrace : Trace :=\n  {trace}\n\n\
         theorem xOutTrace_valid :\n\
             xOutTrace.Valid FPrimeFullHistoryPublicPins.rows := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsPoseidonHashes\n"
    )
}

pub fn compare_public_pins_artifacts(builder: &R1csBuilder) {
    let range = public_pins_range(builder);
    let hash = public_pins_hash(builder, range);
    let program = normalize_range(builder, range.row_start, range.row_end, hash.input_cols[0]);
    let (pins, trivial) = checks(&program);
    let root = formal_repo_root();
    let mut drifted = Vec::new();
    let mut compare = |path: PathBuf, rendered: String| {
        if fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
            let expected = path.with_extension("lean.expected");
            fs::write(&expected, rendered).expect("write public-pins artifact");
            drifted.push(expected);
        }
    };
    compare(
        root.join(ARTIFACT_PATH),
        render_artifact(&program, range, &pins, &trivial),
    );
    for (index, shard) in program.instructions.chunks(SHARD_SIZE).enumerate() {
        compare(
            root.join(format!("{SHARD_PREFIX}{index}.lean")),
            render_shard(index, shard),
        );
    }
    compare(root.join(HASHES_PATH), render_hashes(builder, range, &hash));
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "full-history public-pins artifacts drifted: {drifted:?}"
    );
}
