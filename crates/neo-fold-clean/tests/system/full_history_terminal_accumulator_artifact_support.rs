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
const ACCUMULATOR_CHECK_COVERAGE_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryTerminalAccumulatorCheckCoverage.lean";
const ACCUMULATOR_SCHEDULE_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalAccumulatorSchedule";
const ACCUMULATOR_SCHEDULES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryTerminalAccumulatorSchedules.lean";
const SHARD_SIZE: usize = 1_200;
const INPUT_SHARD_SIZE: usize = 1_000;
const SCHEDULE_SHARD_SIZE: usize = 16;
const TERNARY_DIGITS: usize = 41;
const TERNARY_CANONICAL_ROWS: usize = 124;

pub(super) struct AccumulatorCorePaths {
    pub artifact: &'static str,
    pub shard_prefix: &'static str,
    pub hashes: &'static str,
    pub check_coverage: &'static str,
    pub schedule_prefix: &'static str,
    pub schedules: &'static str,
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

struct SeededPlacement {
    block_index: usize,
    row_start: usize,
    row_end: usize,
}

struct ProgramSegment {
    row_start: usize,
    row_end: usize,
    program: NormalizedProgram,
}

struct TernaryCheckCoverage {
    patterns: Vec<Vec<usize>>,
    pattern_tags: Vec<usize>,
    segment_map_indices: Vec<Vec<usize>>,
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

fn accumulator_hashes(builder: &R1csBuilder, range: &RowFamilyRange) -> Vec<Poseidon2HashAudit> {
    let hashes = builder
        .poseidon2_hash_audits()
        .into_iter()
        .filter(|hash| range.row_start <= hash.row_start && hash.row_end <= range.row_end)
        .collect::<Vec<_>>();
    assert_eq!(
        hashes.len(),
        2,
        "terminal accumulator must contain parent-CE and final-accumulator sponges"
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

fn external_digit_word_starts(builder: &R1csBuilder, range: &RowFamilyRange, maps: &[TernaryMap]) -> Vec<usize> {
    let local = maps
        .iter()
        .map(|map| map.digit_cols[0])
        .collect::<std::collections::HashSet<_>>();
    let mut external = builder
        .seeded_phi81_a_blocks()
        .iter()
        .filter(|block| range.row_start <= block.row_start() && block.row_end() <= range.row_end)
        .flat_map(|block| block.word_starts().iter().copied())
        .filter(|start| !local.contains(start))
        .collect::<Vec<_>>();
    external.sort_unstable();
    external.dedup();
    external
}

fn seeded_placements(builder: &R1csBuilder, range: &RowFamilyRange) -> Vec<SeededPlacement> {
    let mut indexed = builder
        .seeded_phi81_a_blocks()
        .iter()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(_, block)| block.row_start());
    indexed
        .into_iter()
        .enumerate()
        .filter(|(_, (_, block))| range.row_start <= block.row_start() && block.row_end() <= range.row_end)
        .map(|(global_index, (_, block))| SeededPlacement {
            block_index: global_index,
            row_start: block.row_start() - range.row_start,
            row_end: block.row_end() - range.row_start,
        })
        .collect()
}

/// Recover each in-range SeededPhi81 block's source fields in exact word
/// order. The map comes from synthesis metadata, never allocation offsets.
fn seeded_source_field_columns(builder: &R1csBuilder, range: &RowFamilyRange) -> Vec<Vec<usize>> {
    let by_word_start = builder
        .balanced_ternary_audits()
        .into_iter()
        .map(|decomposition| {
            let start = decomposition.digit_cols[0];
            assert!(decomposition
                .digit_cols
                .iter()
                .enumerate()
                .all(|(digit, column)| *column == start + digit));
            (start, decomposition.field_col)
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mut blocks = builder
        .seeded_phi81_a_blocks()
        .iter()
        .filter(|block| range.row_start <= block.row_start() && block.row_end() <= range.row_end)
        .collect::<Vec<_>>();
    blocks.sort_by_key(|block| block.row_start());
    blocks
        .into_iter()
        .map(|block| {
            block
                .word_starts()
                .iter()
                .map(|start| {
                    *by_word_start
                        .get(start)
                        .unwrap_or_else(|| panic!("SeededPhi81 word {start} has no source-field audit"))
                })
                .collect()
        })
        .collect()
}

fn program_segments(
    builder: &R1csBuilder,
    range: &RowFamilyRange,
    placements: &[SeededPlacement],
) -> Vec<ProgramSegment> {
    let mut cursor = range.row_start;
    let mut segments = Vec::with_capacity(placements.len() + 1);
    for placement in placements {
        let block_start = range.row_start + placement.row_start;
        assert!(
            cursor < block_start,
            "seeded block must follow a nonempty ordinary segment"
        );
        let ordinary = RowFamilyRange {
            name: "terminal.accumulator.ordinary",
            row_start: cursor,
            row_end: block_start,
        };
        let first_allocated = first_allocated_column(builder, &ordinary);
        segments.push(ProgramSegment {
            row_start: ordinary.row_start - range.row_start,
            row_end: ordinary.row_end - range.row_start,
            program: normalize_range(builder, ordinary.row_start, ordinary.row_end, first_allocated),
        });
        cursor = range.row_start + placement.row_end;
    }
    assert!(
        cursor < range.row_end,
        "terminal accumulator must end in an ordinary segment"
    );
    let ordinary = RowFamilyRange {
        name: "terminal.accumulator.ordinary",
        row_start: cursor,
        row_end: range.row_end,
    };
    let first_allocated = first_allocated_column(builder, &ordinary);
    segments.push(ProgramSegment {
        row_start: ordinary.row_start - range.row_start,
        row_end: ordinary.row_end - range.row_start,
        program: normalize_range(builder, ordinary.row_start, ordinary.row_end, first_allocated),
    });
    segments
}

fn ternary_check_coverage(segments: &[ProgramSegment], maps: &[TernaryMap]) -> TernaryCheckCoverage {
    let mut patterns = Vec::<Vec<usize>>::new();
    let mut pattern_tags = Vec::with_capacity(maps.len());
    let mut segment_map_indices = vec![Vec::new(); segments.len()];
    for (map_index, map) in maps.iter().enumerate() {
        let (segment_index, segment) = segments
            .iter()
            .enumerate()
            .find(|(_, segment)| {
                segment.row_start <= map.row_start && map.row_start + TERNARY_CANONICAL_ROWS <= segment.row_end
            })
            .expect("shifted-ternary check map belongs to one ordinary segment");
        let instruction_start = map.row_start - segment.row_start;
        let pattern = (0..TERNARY_CANONICAL_ROWS)
            .filter(|offset| {
                matches!(
                    segment.program.instructions[instruction_start + offset],
                    Instruction::Check(_)
                )
            })
            .collect::<Vec<_>>();
        let pattern_tag = patterns
            .iter()
            .position(|candidate| candidate == &pattern)
            .unwrap_or_else(|| {
                patterns.push(pattern);
                patterns.len() - 1
            });
        pattern_tags.push(pattern_tag);
        segment_map_indices[segment_index].push(map_index);
    }
    assert_eq!(patterns.len(), 2, "two shifted-ternary normalized-check patterns");
    let mut pattern_lengths = patterns.iter().map(Vec::len).collect::<Vec<_>>();
    pattern_lengths.sort_unstable();
    assert_eq!(pattern_lengths, vec![111, 112], "shifted-ternary check-pattern sizes");
    for (segment_index, segment) in segments.iter().enumerate() {
        let actual_positions = segment
            .program
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(index, instruction)| matches!(instruction, Instruction::Check(_)).then_some(index))
            .collect::<Vec<_>>();
        let expected_positions = segment_map_indices[segment_index]
            .iter()
            .flat_map(|map_index| {
                let map = &maps[*map_index];
                patterns[pattern_tags[*map_index]]
                    .iter()
                    .map(move |offset| map.row_start - segment.row_start + offset)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual_positions, expected_positions,
            "segment {segment_index} check coverage"
        );
    }
    assert_eq!(
        pattern_tags
            .iter()
            .map(|tag| patterns[*tag].len())
            .sum::<usize>(),
        21_438,
        "all terminal accumulator checks are shifted-ternary canonical rows"
    );
    TernaryCheckCoverage {
        patterns,
        pattern_tags,
        segment_map_indices,
    }
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

fn render_artifact(
    segments: &[ProgramSegment],
    range: &RowFamilyRange,
    parent_digest: [usize; 4],
    accumulator_digest: [usize; 4],
    ternary_maps: &[TernaryMap],
    external_digit_word_starts: &[usize],
    seeded_source_columns: &[Vec<usize>],
    seeded_placements: &[SeededPlacement],
) -> String {
    assert_eq!(segments.len(), seeded_placements.len() + 1);
    assert_eq!(seeded_source_columns.len(), seeded_placements.len());
    let imports = segments
        .iter()
        .enumerate()
        .flat_map(|(segment, program)| {
            let instruction_imports = (0..program.program.instructions.len().div_ceil(SHARD_SIZE))
                .map(|index| {
                    format!(
                        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalAccumulatorSegment{segment}Instructions{index}"
                    )
                });
            let input_imports = (0..program.program.input_columns.len().div_ceil(INPUT_SHARD_SIZE))
                .map(|index| {
                    format!(
                        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalAccumulatorSegment{segment}Inputs{index}"
                    )
                });
            instruction_imports.chain(input_imports).collect::<Vec<_>>()
        })
        .chain(std::iter::once(
            "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Artifact".to_string(),
        ))
        .collect::<Vec<_>>()
        .join("\n");
    let segment_definitions = segments
        .iter()
        .enumerate()
        .map(|(segment, program)| {
            let input_columns = (0..program
                .program
                .input_columns
                .len()
                .div_ceil(INPUT_SHARD_SIZE))
                .map(|index| format!("Generated.segment{segment}Inputs{index}"))
                .collect::<Vec<_>>()
                .join(" ++\n    ");
            let instructions = (0..program.program.instructions.len().div_ceil(SHARD_SIZE))
                .map(|index| format!("Generated.segment{segment}Instructions{index}"))
                .collect::<Vec<_>>()
                .join(" ++\n    ");
            format!(
                "def segment{segment}RowStart : Nat := {}\n\
                 def segment{segment}RowEnd : Nat := {}\n\
                 def segment{segment}InputColumns : List Nat :=\n    {input_columns}\n\
                 def segment{segment}DefinitionCount : Nat := {}\n\
                 def segment{segment}CheckCount : Nat := {}\n\
                 def segment{segment}Instructions : List Instruction :=\n    {instructions}\n\
                 def segment{segment}Rows : List Row :=\n\
                 \x20 CheckedProgram.rows segment{segment}Instructions\n\n\
                 theorem segment{segment}_instructions_length :\n\
                 \x20   segment{segment}Instructions.length =\n\
                 \x20     segment{segment}RowEnd - segment{segment}RowStart := by native_decide\n\
                 theorem segment{segment}_rows_length :\n\
                 \x20   segment{segment}Rows.length =\n\
                 \x20     segment{segment}RowEnd - segment{segment}RowStart := by\n\
                 \x20 simpa [segment{segment}Rows, CheckedProgram.rows] using\n\
                 \x20   segment{segment}_instructions_length\n\
                 theorem segment{segment}_definitions_canonical :\n\
                 \x20   ∀ definition ∈ definitions segment{segment}Instructions,\n\
                 \x20     definition.Canonical := by native_decide\n\
                 theorem segment{segment}_definitions_wellFormed :\n\
                 \x20   WellFormed segment{segment}InputColumns\n\
                 \x20     (definitions segment{segment}Instructions) := by native_decide\n\
                 theorem segment{segment}_checks_reference :\n\
                 \x20   ChecksReference\n\
                 \x20     (knownAfter segment{segment}InputColumns\n\
                 \x20       (definitions segment{segment}Instructions))\n\
                 \x20     segment{segment}Instructions := by native_decide\n",
                program.row_start, program.row_end, program.program.definition_count, program.program.check_count,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let ternary_maps = ternary_maps
        .iter()
        .map(|map| {
            format!(
                "{{ rowStart := {}, fieldColumn := {}, digitColumns := {}, negativeColumns := {}, borrowColumns := {} }}",
                map.row_start,
                map.field_col,
                lean_compact_nat_sequence(&map.digit_cols),
                lean_compact_nat_sequence(&map.negative_cols),
                lean_compact_nat_sequence(&map.borrow_cols),
            )
        })
        .collect::<Vec<_>>()
        .join(",\n   ");
    let rendered_seeded_placements = seeded_placements
        .iter()
        .map(|placement| {
            format!(
                "{{ blockIndex := {}, rowStart := {}, rowEnd := {} }}",
                placement.block_index, placement.row_start, placement.row_end
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let rendered_seeded_source_columns = seeded_source_columns
        .iter()
        .map(|columns| lean_compact_nat_sequence(columns))
        .collect::<Vec<_>>()
        .join(",\n   ");
    let segment_entries = (0..segments.len())
        .map(|segment| {
            format!(
                "{{ rowStart := segment{segment}RowStart, rowEnd := segment{segment}RowEnd, \
                 inputColumns := segment{segment}InputColumns, instructions := segment{segment}Instructions }}"
            )
        })
        .collect::<Vec<_>>()
        .join(",\n   ");
    let seeded_blocks = seeded_placements
        .iter()
        .map(|placement| format!("FPrimeFullHistorySeededPhi81.block{}", placement.block_index))
        .collect::<Vec<_>>()
        .join(", ");
    let mut row_pieces = Vec::with_capacity(segments.len() + seeded_placements.len());
    for (index, placement) in seeded_placements.iter().enumerate() {
        row_pieces.push(format!("segment{index}Rows"));
        row_pieces.push(format!(
            "FPrimeFullHistorySeededPhi81.block{}.rows",
            placement.block_index
        ));
    }
    row_pieces.push(format!("segment{}Rows", segments.len() - 1));
    let row_pieces = row_pieces.join(",\n   ");
    let definition_count = segments
        .iter()
        .map(|segment| segment.program.definition_count)
        .sum::<usize>();
    let check_count = segments
        .iter()
        .map(|segment| segment.program.check_count)
        .sum::<usize>();
    let row_length_lemmas = (0..segments.len())
        .map(|segment| format!("segment{segment}_rows_length"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{imports}\n\n\
         /-! Exact checked program for the terminal post-fold accumulator owner. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 1048576\n\n\
         def parentCeDigestColumns : List Nat := {}\n\
         def accumulatorDigestColumns : List Nat := {}\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\
         def definitionCount : Nat := {}\n\
         def checkCount : Nat := {}\n\n\
         {segment_definitions}\n\n\
         structure ShiftedTernaryMap where\n\
         \x20 rowStart : Nat\n\
         \x20 fieldColumn : Nat\n\
         \x20 digitColumns : List Nat\n\
         \x20 negativeColumns : List Nat\n\
         \x20 borrowColumns : List Nat\n\
         deriving DecidableEq, Repr, Inhabited\n\n\
         def shiftedTernaryMaps : List ShiftedTernaryMap :=\n  [{ternary_maps}]\n\n\
         def externalDigitWordStarts : List Nat := {}\n\n\
         def seededPhi81SourceColumns : List (List Nat) :=\n  [{rendered_seeded_source_columns}]\n\n\
         /-- Raw `ce_claim_digest/v2` fields consumed by the first SIS map. -/\n\
         def parentCeClaimSourceColumns : List Nat :=\n\
         \x20 seededPhi81SourceColumns.getD 0 []\n\n\
         /-- First SIS commitment fields consumed by the compression map. -/\n\
         def parentCeDigestCompressionSourceColumns : List Nat :=\n\
         \x20 seededPhi81SourceColumns.getD 1 []\n\n\
         structure SeededPhi81Placement where\n\
         \x20 blockIndex : Nat\n\
         \x20 rowStart : Nat\n\
         \x20 rowEnd : Nat\n\n\
         def seededPhi81Placements : List SeededPhi81Placement :=\n  [{rendered_seeded_placements}]\n\n\
         structure Segment where\n\
         \x20 rowStart : Nat\n\
         \x20 rowEnd : Nat\n\
         \x20 inputColumns : List Nat\n\
         \x20 instructions : List Instruction\n\n\
         def segments : List Segment :=\n  [{segment_entries}]\n\n\
         def seededBlocks : List SeededPhi81.Block :=\n\
         \x20 [{seeded_blocks}]\n\n\
         theorem seededBlocks_length :\n\
         \x20   seededBlocks.length = seededPhi81Placements.length := by native_decide\n\n\
         def rowPieces : List (List Row) :=\n  [{row_pieces}]\n\n\
         def rows : List Row := rowPieces.flatten\n\n\
         theorem rows_length : rows.length = rowCount := by\n\
         \x20 simp only [rows, rowPieces, List.flatten_cons, List.flatten_nil,\n\
         \x20   List.length_append, List.length_nil, {row_length_lemmas},\n\
         \x20   SeededPhi81.Block.rows_length]\n\
         \x20 native_decide\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator\n",
        lean_nat_list(parent_digest),
        lean_nat_list(accumulator_digest),
        range.row_start,
        range.row_end,
        range.row_end - range.row_start,
        definition_count,
        check_count,
        lean_compact_nat_sequence(external_digit_word_starts),
    )
}

fn render_check_coverage(segments: &[ProgramSegment], coverage: &TernaryCheckCoverage) -> String {
    let pattern_definitions = coverage
        .patterns
        .iter()
        .enumerate()
        .map(|(index, pattern)| {
            format!(
                "def checkPattern{index} : List Nat :=\n  {}",
                lean_compact_nat_sequence(pattern),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let pattern_references = (0..coverage.patterns.len())
        .map(|index| format!("checkPattern{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let segment_definitions = segments
        .iter()
        .enumerate()
        .map(|(segment, _)| {
            format!(
                "def segment{segment}MapIndices : List Nat :=\n  {}\n\n\
                 def segment{segment}ExpectedChecks : List Row :=\n\
                 \x20 segment{segment}MapIndices.flatMap checksForMapIndex\n\n\
                 theorem segment{segment}_checks_covered :\n\
                 \x20   CheckedProgram.checks\n\
                 \x20       FPrimeFullHistoryTerminalAccumulator.segment{segment}Instructions =\n\
                 \x20     segment{segment}ExpectedChecks := by native_decide",
                lean_compact_nat_sequence(&coverage.segment_map_indices[segment]),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalAccumulatorArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Relabel\n\
         import Nightstream.Implementation.R1CS.Ownership.ShiftedTernary.ShiftedTernary\n\n\
         /-! Exact classification of every terminal-accumulator assertion row. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorCheckCoverage\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         set_option maxRecDepth 1048576\n\n\
         def defaultRow : Row := ⟨[], [], []⟩\n\n\
         def columnMap\n\
         \x20   (map : FPrimeFullHistoryTerminalAccumulator.ShiftedTernaryMap) :\n\
         \x20   List Nat :=\n\
         \x20 [0, map.fieldColumn] ++ List.replicate 56 0 ++\n\
         \x20   map.digitColumns ++ map.negativeColumns ++ map.borrowColumns\n\n\
         def shiftedOwnerRows\n\
         \x20   (map : FPrimeFullHistoryTerminalAccumulator.ShiftedTernaryMap) :\n\
         \x20   List Row :=\n\
         \x20 if map.rowStart < FPrimeFullHistoryTerminalAccumulator.segment1RowStart then\n\
         \x20   FPrimeFullHistoryTerminalAccumulator.segment0Rows\n\
         \x20 else FPrimeFullHistoryTerminalAccumulator.segment1Rows\n\n\
         def shiftedLocalRowStart\n\
         \x20   (map : FPrimeFullHistoryTerminalAccumulator.ShiftedTernaryMap) : Nat :=\n\
         \x20 if map.rowStart < FPrimeFullHistoryTerminalAccumulator.segment1RowStart then\n\
         \x20   map.rowStart\n\
         \x20 else map.rowStart - FPrimeFullHistoryTerminalAccumulator.segment1RowStart\n\n\
         {pattern_definitions}\n\n\
         def checkPatterns : List (List Nat) := [{pattern_references}]\n\n\
         def checkPatternTags : List Nat :=\n  {}\n\n\
         def checksForMapIndex (mapIndex : Nat) : List Row :=\n\
         \x20 let map := FPrimeFullHistoryTerminalAccumulator.shiftedTernaryMaps.getD mapIndex default\n\
         \x20 let patternTag := checkPatternTags.getD mapIndex 0\n\
         \x20 (checkPatterns.getD patternTag []).map fun rowIndex =>\n\
         \x20   Relabel.row (columnMap map)\n\
         \x20     (ShiftedTernaryCompiler.canonicalRows.getD rowIndex defaultRow)\n\n\
         theorem checkPatternTags_length :\n\
         \x20   checkPatternTags.length =\n\
         \x20     FPrimeFullHistoryTerminalAccumulator.shiftedTernaryMaps.length := by native_decide\n\n\
         theorem checkPatterns_bounded :\n\
         \x20   ∀ pattern ∈ checkPatterns, ∀ rowIndex ∈ pattern,\n\
         \x20     rowIndex < ShiftedTernaryCompiler.canonicalRows.length := by native_decide\n\n\
         def classifiedCheckCount : Nat :=\n\
         \x20 (checkPatternTags.map fun tag => (checkPatterns.getD tag []).length).sum\n\n\
         def residualCheckCount : Nat := 0\n\n\
         theorem classification_count : classifiedCheckCount = 21438 := by native_decide\n\n\
         {segment_definitions}\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorCheckCoverage\n",
        lean_compact_nat_sequence(&coverage.pattern_tags),
    )
}

fn render_schedule_shard(shard: usize, map_start: usize, maps: &[TernaryMap], segments: &[ProgramSegment]) -> String {
    let theorems = maps
        .iter()
        .enumerate()
        .map(|(local_index, map)| {
            let index = map_start + local_index;
            let (segment_index, segment) = segments
                .iter()
                .enumerate()
                .find(|(_, segment)| {
                    segment.row_start <= map.row_start
                        && map.row_start + TERNARY_CANONICAL_ROWS <= segment.row_end
                })
                .expect("schedule map ordinary segment");
            let local_row_start = map.row_start - segment.row_start;
            format!(
                "def map{index} :=\n\
                 \x20 FPrimeFullHistoryTerminalAccumulator.shiftedTernaryMaps.getD {index} default\n\n\
                 theorem map{index}_rows_schedule :\n\
                 \x20   (FPrimeFullHistoryTerminalAccumulator.segment{segment_index}Rows.drop {local_row_start}).take 124 =\n\
                 \x20     ShiftedTernaryCompiler.canonicalRows.map\n\
                 \x20       (Relabel.row\n\
                 \x20         (FPrimeFullHistoryTerminalAccumulatorCheckCoverage.columnMap map{index})) := by\n\
                 \x20 native_decide"
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let index_cases = maps
        .iter()
        .enumerate()
        .map(|(local_index, _)| format!("index = {}", map_start + local_index))
        .collect::<Vec<_>>()
        .join(" ∨ ");
    let case_patterns = maps.iter().map(|_| "rfl").collect::<Vec<_>>().join(" | ");
    let aggregate_cases = maps
        .iter()
        .enumerate()
        .map(|(local_index, _)| {
            let index = map_start + local_index;
            format!(
                "  · simpa [map{index},\n\
                 \x20     FPrimeFullHistoryTerminalAccumulatorCheckCoverage.shiftedOwnerRows,\n\
                 \x20     FPrimeFullHistoryTerminalAccumulatorCheckCoverage.shiftedLocalRowStart] using\n\
                 \x20       map{index}_rows_schedule"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let map_end = map_start + maps.len();
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalAccumulatorCheckCoverage\n\n\
         /-! Generated exact shifted-ternary row schedules, shard {shard}. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSchedule\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         set_option maxRecDepth 1048576\n\n\
         {theorems}\n\n\
         theorem shard{shard}_rows_schedule\n\
         \x20   (index : Nat) (lower : {map_start} ≤ index) (upper : index < {map_end}) :\n\
         \x20   let map := FPrimeFullHistoryTerminalAccumulator.shiftedTernaryMaps.getD index default\n\
         \x20   ((FPrimeFullHistoryTerminalAccumulatorCheckCoverage.shiftedOwnerRows map).drop\n\
         \x20       (FPrimeFullHistoryTerminalAccumulatorCheckCoverage.shiftedLocalRowStart map)).take 124 =\n\
         \x20     ShiftedTernaryCompiler.canonicalRows.map\n\
         \x20       (Relabel.row\n\
         \x20         (FPrimeFullHistoryTerminalAccumulatorCheckCoverage.columnMap map)) := by\n\
         \x20 have cases : {index_cases} := by omega\n\
         \x20 rcases cases with {case_patterns}\n\
         {aggregate_cases}\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSchedule\n"
    )
}

fn render_schedules(map_count: usize) -> String {
    let imports = (0..map_count.div_ceil(SCHEDULE_SHARD_SIZE))
        .map(|shard| {
            format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalAccumulatorSchedule{shard}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let shard_count = map_count.div_ceil(SCHEDULE_SHARD_SIZE);
    let shard_intervals = (0..shard_count)
        .map(|shard| {
            let start = shard * SCHEDULE_SHARD_SIZE;
            let end = ((shard + 1) * SCHEDULE_SHARD_SIZE).min(map_count);
            if shard == 0 {
                format!("index < {end}")
            } else {
                format!("({start} ≤ index ∧ index < {end})")
            }
        })
        .collect::<Vec<_>>()
        .join(" ∨ ");
    let shard_patterns = (0..shard_count)
        .map(|shard| {
            if shard == 0 {
                "first".to_string()
            } else {
                format!("shard{shard}")
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let dispatch = (0..shard_count)
        .map(|shard| {
            if shard == 0 {
                "  · exact shard0_rows_schedule index (by omega) first".to_string()
            } else {
                format!("  · exact shard{shard}_rows_schedule index shard{shard}.1 shard{shard}.2")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{imports}\n\n\
         /-! Aggregate certificate for every exact terminal-accumulator shifted-ternary schedule. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSchedule\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         theorem rows_schedule (index : Nat) (indexLt : index < {map_count}) :\n\
         \x20   let map := FPrimeFullHistoryTerminalAccumulator.shiftedTernaryMaps.getD index default\n\
         \x20   ((FPrimeFullHistoryTerminalAccumulatorCheckCoverage.shiftedOwnerRows map).drop\n\
         \x20       (FPrimeFullHistoryTerminalAccumulatorCheckCoverage.shiftedLocalRowStart map)).take 124 =\n\
         \x20   ShiftedTernaryCompiler.canonicalRows.map\n\
         \x20     (Relabel.row\n\
         \x20       (FPrimeFullHistoryTerminalAccumulatorCheckCoverage.columnMap map)) := by\n\
         \x20 have shardCases : {shard_intervals} := by omega\n\
         \x20 rcases shardCases with {shard_patterns}\n\
         {dispatch}\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSchedule\n"
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

fn render_hashes(
    builder: &R1csBuilder,
    range: &RowFamilyRange,
    hash_segment: &ProgramSegment,
    hash_segment_index: usize,
    hashes: &[Poseidon2HashAudit],
) -> String {
    let row_origin = range.row_start + hash_segment.row_start;
    let row_end = range.row_start + hash_segment.row_end;
    assert!(
        hashes
            .iter()
            .all(|hash| row_origin <= hash.row_start && hash.row_end <= row_end),
        "terminal accumulator hashes belong to the final ordinary segment"
    );
    let parent = render_hash_trace(builder, row_origin, &hashes[0]);
    let accumulator = render_hash_trace(builder, row_origin, &hashes[1]);
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalAccumulatorArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge\n\n\
         /-! Generated exact sponge traces for the terminal post-fold accumulator owner. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorPoseidonHashes\n\n\
         open Nightstream.Implementation.R1CS.Poseidon2Sponge\n\n\
         set_option maxRecDepth 1048576\n\n\
         def parentCeDigestTrace : Trace :=\n  {parent}\n\n\
         def accumulatorDigestTrace : Trace :=\n  {accumulator}\n\n\
         theorem parentCeDigestTrace_valid :\n\
         \x20   parentCeDigestTrace.Valid\n\
         \x20     FPrimeFullHistoryTerminalAccumulator.segment{hash_segment_index}Rows := by native_decide\n\n\
         theorem accumulatorDigestTrace_valid :\n\
         \x20   accumulatorDigestTrace.Valid\n\
         \x20     FPrimeFullHistoryTerminalAccumulator.segment{hash_segment_index}Rows := by native_decide\n\n\
         theorem parentCeDigestTrace_output :\n\
         \x20   parentCeDigestTrace.outputColumns =\n\
         \x20     FPrimeFullHistoryTerminalAccumulator.parentCeDigestColumns := by native_decide\n\n\
         theorem accumulatorDigestTrace_output :\n\
         \x20   accumulatorDigestTrace.outputColumns =\n\
         \x20     FPrimeFullHistoryTerminalAccumulator.accumulatorDigestColumns := by native_decide\n\n\
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
    let hashes = accumulator_hashes(builder, accumulator);
    let parent_digest = hashes[0].output_cols;
    let accumulator_digest = hashes[1].output_cols;
    let ternary_maps = ternary_maps(builder, accumulator);
    let external_digit_word_starts = external_digit_word_starts(builder, accumulator, &ternary_maps);
    let seeded_source_columns = seeded_source_field_columns(builder, accumulator);
    assert_eq!(
        seeded_source_columns
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        vec![1_650, 108],
        "fixed-profile parent CE binding and digest-compression input widths"
    );
    let seeded_placements = seeded_placements(builder, accumulator);
    let segments = program_segments(builder, accumulator, &seeded_placements);
    let check_coverage = ternary_check_coverage(&segments, &ternary_maps);

    let root = formal_repo_root();
    let mut drifted = Vec::new();
    let mut compare = |path: PathBuf, rendered: String| {
        if fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
            let expected = path.with_extension("lean.expected");
            fs::write(&expected, rendered).expect("write accumulator core artifact");
            drifted.push(expected);
        }
    };
    for (segment, program) in segments.iter().enumerate() {
        for (index, shard) in program
            .program
            .input_columns
            .chunks(INPUT_SHARD_SIZE)
            .enumerate()
        {
            compare(
                root.join(format!("{}{segment}Inputs{index}.lean", paths.shard_prefix)),
                for_variant(render_input_shard(segment, index, shard), paths.recursive),
            );
        }
        for (index, shard) in program.program.instructions.chunks(SHARD_SIZE).enumerate() {
            compare(
                root.join(format!("{}{segment}Instructions{index}.lean", paths.shard_prefix)),
                for_variant(render_shard(segment, index, shard), paths.recursive),
            );
        }
    }
    compare(
        root.join(paths.artifact),
        for_variant(
            render_artifact(
                &segments,
                accumulator,
                parent_digest,
                accumulator_digest,
                &ternary_maps,
                &external_digit_word_starts,
                &seeded_source_columns,
                &seeded_placements,
            ),
            paths.recursive,
        ),
    );
    compare(
        root.join(paths.check_coverage),
        for_variant(render_check_coverage(&segments, &check_coverage), paths.recursive),
    );
    for (shard, maps) in ternary_maps.chunks(SCHEDULE_SHARD_SIZE).enumerate() {
        compare(
            root.join(format!("{}{shard}.lean", paths.schedule_prefix)),
            for_variant(
                render_schedule_shard(shard, shard * SCHEDULE_SHARD_SIZE, maps, &segments),
                paths.recursive,
            ),
        );
    }
    compare(
        root.join(paths.schedules),
        for_variant(render_schedules(ternary_maps.len()), paths.recursive),
    );
    compare(
        root.join(paths.hashes),
        for_variant(
            render_hashes(
                builder,
                accumulator,
                segments.last().expect("accumulator core final segment"),
                segments.len() - 1,
                &hashes,
            ),
            paths.recursive,
        ),
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
            check_coverage: ACCUMULATOR_CHECK_COVERAGE_PATH,
            schedule_prefix: ACCUMULATOR_SCHEDULE_PREFIX,
            schedules: ACCUMULATOR_SCHEDULES_PATH,
            recursive: false,
        },
    );
}
