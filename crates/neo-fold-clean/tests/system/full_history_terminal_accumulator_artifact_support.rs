use neo_fold_clean::engine::r1cs_circuit::builder::Poseidon2HashAudit;
use p3_field::PrimeField64;

use super::*;

const PARENT_LINK_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalParentLinkArtifact.lean";
const ACCUMULATOR_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryTerminalAccumulatorArtifact.lean";
const ACCUMULATOR_SHARD_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalAccumulatorSegment";
const ACCUMULATOR_HASHES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.lean";
const SHARD_SIZE: usize = 1_200;
const INPUT_SHARD_SIZE: usize = 1_000;
const TERNARY_DIGITS: usize = 41;
const TERNARY_CANONICAL_ROWS: usize = 124;

pub(super) struct AccumulatorCorePaths {
    pub artifact: &'static str,
    pub shard_prefix: &'static str,
    pub hashes: &'static str,
    pub recursive: bool,
}

fn lean_compact_nat_sequence(values: &[usize]) -> String {
    if values.is_empty() {
        return "[]".into();
    }
    let mut pieces = Vec::new();
    let mut literals = Vec::new();
    let flush_literals = |literals: &mut Vec<usize>, pieces: &mut Vec<String>| {
        if !literals.is_empty() {
            pieces.push(lean_nat_list(literals.drain(..)));
        }
    };
    let mut index = 0;
    while index < values.len() {
        if index + 3 < values.len() && values[index + 1] > values[index] {
            let step = values[index + 1] - values[index];
            let mut end = index + 2;
            while end < values.len() && values[end] > values[end - 1] && values[end] - values[end - 1] == step {
                end += 1;
            }
            if end - index >= 4 {
                flush_literals(&mut literals, &mut pieces);
                pieces.push(format!(
                    "((List.range {}).map (fun index => {} + {} * index))",
                    end - index,
                    values[index],
                    step,
                ));
                index = end;
                continue;
            }
        }
        literals.push(values[index]);
        index += 1;
    }
    flush_literals(&mut literals, &mut pieces);
    pieces.join(" ++\n    ")
}

pub(super) struct TernaryMap {
    pub(super) row_start: usize,
    pub(super) field_col: usize,
    pub(super) digit_cols: Vec<usize>,
    pub(super) negative_cols: Vec<usize>,
    pub(super) borrow_cols: Vec<usize>,
}

struct ProgramSegment {
    row_start: usize,
    row_end: usize,
    program: NormalizedProgram,
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

fn first_allocated_column(builder: &R1csBuilder, range: &RowFamilyRange) -> usize {
    let (a, b, c) = builder.sparse_triplets();
    assert_eq!(
        b.iter()
            .filter(|(row, _, _)| *row == range.row_start)
            .map(|(_, column, coefficient)| (*column, coefficient.as_canonical_u64()))
            .collect::<Vec<_>>(),
        vec![(0, 1)],
        "ordinary accumulator segment starts with a linear definition"
    );
    assert!(
        c.iter().all(|(row, _, _)| *row != range.row_start),
        "ordinary accumulator segment first row has nonempty C"
    );
    a.iter()
        .filter(|(row, column, coefficient)| {
            *row == range.row_start && *column != 0 && coefficient.as_canonical_u64() == 1
        })
        .map(|(_, column, _)| *column)
        .min()
        .expect("ordinary accumulator segment first output column")
}

fn accumulator_hashes(builder: &R1csBuilder, range: &RowFamilyRange, expected: usize) -> Vec<Poseidon2HashAudit> {
    let hashes = builder
        .poseidon2_hash_audits()
        .into_iter()
        .filter(|hash| range.row_start <= hash.row_start && hash.row_end <= range.row_end)
        .collect::<Vec<_>>();
    assert_eq!(
        hashes.len(),
        expected,
        "accumulator owner has the expected Poseidon2 sponge count"
    );
    hashes
}

pub(super) fn ternary_maps(builder: &R1csBuilder, range: &RowFamilyRange) -> Vec<TernaryMap> {
    let (a, b, c) = builder.sparse_triplets();
    let mut digit_starts = builder
        .seeded_phi81_a_blocks()
        .iter()
        .filter(|block| range.row_start <= block.row_start() && block.row_end() <= range.row_end)
        .flat_map(|block| block.word_starts().iter().copied())
        .collect::<Vec<_>>();
    digit_starts.sort_unstable();
    digit_starts.dedup();

    let digit_start_set = digit_starts
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let negative_start_to_digit = digit_starts
        .iter()
        .map(|start| (start + TERNARY_DIGITS, *start))
        .collect::<std::collections::HashMap<_, _>>();
    let mut a_candidate_rows = std::collections::HashMap::<usize, Vec<usize>>::new();
    for (row, column, coefficient) in a {
        if range.row_start <= *row
            && *row < range.row_end
            && coefficient.as_canonical_u64() == 1
            && digit_start_set.contains(column)
        {
            a_candidate_rows.entry(*column).or_default().push(*row);
        }
    }
    let mut c_candidate_rows = std::collections::HashMap::<usize, std::collections::HashSet<usize>>::new();
    for (row, column, coefficient) in c {
        if range.row_start <= *row && *row < range.row_end && coefficient.as_canonical_u64() == 2 {
            if let Some(digit_start) = negative_start_to_digit.get(column) {
                c_candidate_rows
                    .entry(*digit_start)
                    .or_default()
                    .insert(*row);
            }
        }
    }
    let first_rows = digit_starts
        .iter()
        .filter_map(|digit_start| {
            let c_rows = c_candidate_rows.get(digit_start)?;
            let first_row = a_candidate_rows
                .get(digit_start)?
                .iter()
                .filter(|row| c_rows.contains(row))
                .min()
                .copied()?;
            Some((*digit_start, first_row))
        })
        .collect::<Vec<_>>();
    let reconstruction_rows = first_rows
        .iter()
        .map(|(_, first_row)| first_row + 2 * TERNARY_DIGITS)
        .collect::<std::collections::HashSet<_>>();
    let mut reconstruction_a = std::collections::HashMap::<usize, Vec<(usize, u64)>>::new();
    let mut reconstruction_b = std::collections::HashMap::<usize, Vec<(usize, u64)>>::new();
    let mut reconstruction_c_rows = std::collections::HashSet::new();
    for (row, column, coefficient) in a {
        if reconstruction_rows.contains(row) {
            reconstruction_a
                .entry(*row)
                .or_default()
                .push((*column, coefficient.as_canonical_u64()));
        }
    }
    for (row, column, coefficient) in b {
        if reconstruction_rows.contains(row) {
            reconstruction_b
                .entry(*row)
                .or_default()
                .push((*column, coefficient.as_canonical_u64()));
        }
    }
    for (row, _, _) in c {
        if reconstruction_rows.contains(row) {
            reconstruction_c_rows.insert(*row);
        }
    }

    let mut maps = first_rows
        .into_iter()
        .map(|(digit_start, first_row)| {
            assert!(
                first_row + TERNARY_CANONICAL_ROWS <= range.row_end,
                "shifted-ternary rows leave accumulator owner"
            );
            let digit_cols = (digit_start..digit_start + TERNARY_DIGITS).collect::<Vec<_>>();
            let negative_cols = (digit_start + TERNARY_DIGITS..digit_start + 2 * TERNARY_DIGITS).collect::<Vec<_>>();
            let borrow_cols = (digit_start + 2 * TERNARY_DIGITS..digit_start + 2 * TERNARY_DIGITS + TERNARY_DIGITS - 1)
                .collect::<Vec<_>>();
            let reconstruction_row = first_row + 2 * TERNARY_DIGITS;
            let reconstruction_a = reconstruction_a
                .get(&reconstruction_row)
                .expect("shifted-ternary reconstruction A");
            let field_candidates = reconstruction_a
                .iter()
                .filter(|(column, coefficient)| *coefficient == 1 && !digit_cols.contains(column))
                .map(|(column, _)| *column)
                .collect::<Vec<_>>();
            let [field_col] = field_candidates.as_slice() else {
                panic!("one shifted-ternary field at row {reconstruction_row}: {reconstruction_a:?}");
            };
            assert_eq!(
                reconstruction_b.get(&reconstruction_row).map(Vec::as_slice),
                Some([(0, 1)].as_slice()),
                "shifted-ternary reconstruction B"
            );
            assert!(
                !reconstruction_c_rows.contains(&reconstruction_row),
                "shifted-ternary reconstruction C"
            );
            TernaryMap {
                row_start: first_row - range.row_start,
                field_col: *field_col,
                digit_cols,
                negative_cols,
                borrow_cols,
            }
        })
        .collect::<Vec<_>>();
    maps.sort_unstable_by_key(|map| map.row_start);

    let mut column_owner = std::collections::HashMap::new();
    for (map_index, map) in maps.iter().enumerate() {
        for column in map
            .digit_cols
            .iter()
            .chain(&map.negative_cols)
            .chain(&map.borrow_cols)
        {
            assert!(
                column_owner.insert(*column, map_index).is_none(),
                "shifted-ternary map columns are disjoint"
            );
        }
    }
    let mut referenced = vec![std::collections::HashSet::new(); maps.len()];
    for (row, column, _) in a.iter().chain(b).chain(c) {
        if let Some(map_index) = column_owner.get(column) {
            let first_row = range.row_start + maps[*map_index].row_start;
            if first_row <= *row && *row < first_row + TERNARY_CANONICAL_ROWS {
                referenced[*map_index].insert(*column);
            }
        }
    }
    for (map_index, map) in maps.iter().enumerate() {
        for column in map
            .digit_cols
            .iter()
            .chain(&map.negative_cols)
            .chain(&map.borrow_cols)
        {
            assert!(
                referenced[map_index].contains(column),
                "shifted-ternary map column {column} unreferenced"
            );
        }
    }
    maps
}

fn render_shard(segment: usize, index: usize, instructions: &[Instruction]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated terminal-accumulator segment {segment} instruction shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator.Generated\n\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 262144\n\n\
         def segment{segment}Instructions{index} : List Instruction :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator.Generated\n",
        lean_instructions(instructions),
    )
}

fn render_input_shard(segment: usize, index: usize, columns: &[usize]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated terminal-accumulator segment {segment} input shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator.Generated\n\n\
         def segment{segment}Inputs{index} : List Nat :=\n  {}\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator.Generated\n",
        lean_compact_nat_sequence(columns),
    )
}

fn render_artifact(prefix: &ProgramSegment, range: &RowFamilyRange, accumulator_digest: &[usize]) -> String {
    assert_eq!(prefix.row_start, 0, "direct accumulator prefix starts at owner");
    let instruction_imports = (0..prefix.program.instructions.len().div_ceil(SHARD_SIZE))
        .map(|index| {
            format!(
                "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions{index}"
            )
        });
    let input_imports = (0..prefix.program.input_columns.len().div_ceil(INPUT_SHARD_SIZE))
        .map(|index| {
            format!(
                "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs{index}"
            )
        });
    let imports = instruction_imports
        .chain(input_imports)
        .chain(std::iter::once(
            "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalAccumulatorPoseidonHashes".to_string(),
        ))
        .collect::<Vec<_>>()
        .join("\n");
    let input_columns = (0..prefix
        .program
        .input_columns
        .len()
        .div_ceil(INPUT_SHARD_SIZE))
        .map(|index| format!("Generated.segment0Inputs{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    let instructions = (0..prefix.program.instructions.len().div_ceil(SHARD_SIZE))
        .map(|index| format!("Generated.segment0Instructions{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    format!(
        "{imports}\n\n\
         /-!\n\
         Exact checked rows for the terminal post-fold accumulator owner.\n\n\
         | Branch | Mathematical obligation | Emits constraints |\n\
         |---|---|---|\n\
         | prefix | Pin the supported-profile constants and inactive-X zero | yes |\n\
         | digest | Poseidon2 over the 1,682-field accumulator-v1 projection | yes |\n\n\
         Owns: the direct accumulator digest computation.\n\
         Does not own: PiRLC parent authority or omitted y_zcol validation.\n\
         -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 1048576\n\n\
         def accumulatorClaimSourceColumns : List Nat :=\n\
         \x20 FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.accumulatorDigestTrace.inputColumns\n\n\
         def accumulatorDigestColumns : List Nat := {}\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\
         def definitionCount : Nat := {}\n\
         def checkCount : Nat := {}\n\n\
         def segment0RowStart : Nat := {}\n\
         def segment0RowEnd : Nat := {}\n\
         def segment0InputColumns : List Nat :=\n    {input_columns}\n\
         def segment0Instructions : List Instruction :=\n    {instructions}\n\
         def segment0Rows : List Row := CheckedProgram.rows segment0Instructions\n\n\
         theorem segment0_instructions_length :\n\
         \x20   segment0Instructions.length = segment0RowEnd - segment0RowStart := by native_decide\n\n\
         theorem segment0_rows_length :\n\
         \x20   segment0Rows.length = segment0RowEnd - segment0RowStart := by\n\
         \x20 simpa [segment0Rows, CheckedProgram.rows] using segment0_instructions_length\n\n\
         theorem segment0_definitions_canonical :\n\
         \x20   ∀ definition ∈ definitions segment0Instructions, definition.Canonical := by native_decide\n\n\
         theorem segment0_definitions_wellFormed :\n\
         \x20   WellFormed segment0InputColumns (definitions segment0Instructions) := by native_decide\n\n\
         theorem segment0_checks_reference :\n\
         \x20   ChecksReference\n\
         \x20     (knownAfter segment0InputColumns (definitions segment0Instructions))\n\
         \x20     segment0Instructions := by native_decide\n\n\
         def rowPieces : List (List Row) :=\n\
         \x20 [segment0Rows, FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows]\n\n\
         def rows : List Row := rowPieces.flatten\n\n\
         theorem rows_length : rows.length = rowCount := by\n\
         \x20 simp only [rows, rowPieces, List.flatten_cons, List.flatten_nil,\n\
         \x20   List.length_append, List.length_nil, segment0_rows_length,\n\
         \x20   FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows_length]\n\
         \x20 native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator\n",
        lean_nat_list(accumulator_digest.iter().copied()),
        range.row_start,
        range.row_end,
        range.row_end - range.row_start,
        prefix.program.definition_count,
        prefix.program.check_count,
        prefix.row_start,
        prefix.row_end,
    )
}

fn render_hash_trace(builder: &R1csBuilder, row_origin: usize, hash: &Poseidon2HashAudit) -> String {
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
                .expect("terminal accumulator sponge permutation call");
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
                lean_nat_list(round.defining_rows.iter().map(|row| row - row_origin)),
                call.row_start - row_origin,
                call.row_end - row_origin,
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
        hash.zero_row - row_origin,
        rounds,
        lean_nat_list(hash.output_cols),
    )
}

fn render_hashes(builder: &R1csBuilder, range: &RowFamilyRange, hashes: &[Poseidon2HashAudit]) -> String {
    let [accumulator] = hashes else {
        unreachable!("direct accumulator owner has one Poseidon2 hash")
    };
    assert!(range.row_start <= accumulator.row_start);
    assert_eq!(accumulator.row_end, range.row_end, "direct hash closes owner");
    let hash_row_count = accumulator.row_end - accumulator.row_start;
    let accumulator = render_hash_trace(builder, accumulator.row_start, accumulator);
    format!(
        "import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge\n\n\
         /-! Generated exact direct accumulator-claim sponge rows and trace. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorPoseidonHashes\n\n\
         open Nightstream.Implementation.R1CS.Poseidon2Sponge\n\n\
         set_option maxRecDepth 1048576\n\n\
         def accumulatorDigestTrace : Trace :=\n  {accumulator}\n\n\
         def rows : List Nightstream.Implementation.R1CS.Row :=\n\
         \x20 accumulatorDigestTrace.rows\n\n\
         theorem rows_length : rows.length = {hash_row_count} := by native_decide\n\n\
         theorem accumulatorDigestTrace_valid :\n\
         \x20   accumulatorDigestTrace.Valid\n\
         \x20     rows := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorPoseidonHashes\n"
    )
}

fn for_variant(rendered: String, recursive: bool) -> String {
    if !recursive {
        return rendered;
    }
    rendered
        .replace(
            "FPrimeFullHistoryTerminalAccumulator",
            "FPrimeFullHistoryRecursiveAccumulatorCore",
        )
        .replace("terminal-accumulator", "recursive-accumulator-core")
        .replace("terminal post-fold accumulator", "recursive accumulator core")
        .replace("terminal accumulator", "recursive accumulator core")
}

pub(super) fn compare_accumulator_core_artifacts(
    builder: &R1csBuilder,
    accumulator: &RowFamilyRange,
    paths: AccumulatorCorePaths,
) {
    let hashes = accumulator_hashes(builder, accumulator, 1);
    let hash = &hashes[0];
    assert_eq!(
        accumulator.row_end - accumulator.row_start,
        254_911,
        "fixed-profile direct accumulator owner rows"
    );
    assert_eq!(hash.input_cols.len(), 1_682, "accumulator-v1 preimage width");
    assert_eq!(hash.rounds.len(), 422, "accumulator-v1 sponge rounds");
    assert_eq!(hash.row_end - hash.row_start, 254_884, "accumulator-v1 sponge rows");
    let accumulator_digest = hashes[0].output_cols.to_vec();
    let ternary_maps = ternary_maps(builder, accumulator);
    assert!(ternary_maps.is_empty(), "direct accumulator has no ternary maps");
    let ordinary_prefix = RowFamilyRange {
        name: "accumulator.direct_hash_prefix",
        row_start: accumulator.row_start,
        row_end: hashes[0].row_start,
    };
    assert_eq!(
        ordinary_prefix.row_end - ordinary_prefix.row_start,
        27,
        "accumulator-v1 ordinary prefix rows"
    );
    let first_allocated = first_allocated_column(builder, &ordinary_prefix);
    let prefix = ProgramSegment {
        row_start: 0,
        row_end: ordinary_prefix.row_end - ordinary_prefix.row_start,
        program: normalize_range(
            builder,
            ordinary_prefix.row_start,
            ordinary_prefix.row_end,
            first_allocated,
        ),
    };
    assert_eq!(prefix.program.definition_count, 27, "prefix definitions");
    assert_eq!(prefix.program.check_count, 0, "prefix checks");

    let root = formal_repo_root();
    let mut drifted = Vec::new();
    let mut compare = |path: PathBuf, rendered: String| {
        if fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
            let expected = path.with_extension("lean.expected");
            fs::write(&expected, rendered).expect("write accumulator core artifact");
            drifted.push(expected);
        }
    };
    for (index, shard) in prefix
        .program
        .input_columns
        .chunks(INPUT_SHARD_SIZE)
        .enumerate()
    {
        compare(
            root.join(format!("{}0Inputs{index}.lean", paths.shard_prefix)),
            for_variant(render_input_shard(0, index, shard), paths.recursive),
        );
    }
    for (index, shard) in prefix.program.instructions.chunks(SHARD_SIZE).enumerate() {
        compare(
            root.join(format!("{}0Instructions{index}.lean", paths.shard_prefix)),
            for_variant(render_shard(0, index, shard), paths.recursive),
        );
    }
    compare(
        root.join(paths.artifact),
        for_variant(
            render_artifact(&prefix, accumulator, &accumulator_digest),
            paths.recursive,
        ),
    );
    compare(
        root.join(paths.hashes),
        for_variant(render_hashes(builder, accumulator, &hashes), paths.recursive),
    );
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "accumulator core artifacts drifted: {drifted:?}"
    );
}

pub fn compare_terminal_accumulator_artifacts(builder: &R1csBuilder) {
    let parent_link = owner(builder, "terminal.parent_link");
    let accumulator = owner(builder, "terminal.accumulator");
    let root = formal_repo_root();
    let rendered = render_equality_artifact(
        builder,
        parent_link,
        "FPrimeFullHistoryTerminalParentLink",
        "terminal previous/current parent-authority link",
        &range_hash(builder, parent_link),
    );
    if fs::read_to_string(root.join(PARENT_LINK_PATH))
        .ok()
        .as_deref()
        != Some(&rendered)
    {
        let expected = root.join(PARENT_LINK_PATH).with_extension("lean.expected");
        fs::write(&expected, rendered).expect("write terminal parent-link artifact");
        assert!(
            STAGE_ALL_ARTIFACTS,
            "terminal parent-link artifact drifted: {expected:?}"
        );
    }
    compare_accumulator_core_artifacts(
        builder,
        accumulator,
        AccumulatorCorePaths {
            artifact: ACCUMULATOR_PATH,
            shard_prefix: ACCUMULATOR_SHARD_PREFIX,
            hashes: ACCUMULATOR_HASHES_PATH,
            recursive: false,
        },
    );
}
