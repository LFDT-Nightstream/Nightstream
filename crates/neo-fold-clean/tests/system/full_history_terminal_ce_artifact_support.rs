use super::*;
use neo_fold_clean::engine::r1cs_circuit::builder::{ProgramRangeAudit, TerminalCeClaimAudit};

const TERMINAL_CE_PROFILE_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalCeProfile.lean";
const TERMINAL_CE_ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryTerminalCeArtifact.lean";
const TERMINAL_CE_PACKED_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalCePacked.lean";
const TERMINAL_CE_LAYOUT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalCeLayout.lean";
const TERMINAL_CE_MAP_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalCeMap";
const TERMINAL_CE_PACKED_TOKEN_CHUNK: usize = 1_000;

fn canonical_input_columns(program: &NormalizedProgram, canonical: &CanonicalizedProgram) -> Vec<usize> {
    program
        .input_columns
        .iter()
        .map(|global| {
            canonical
                .column_map
                .iter()
                .position(|candidate| candidate == global)
                .unwrap_or_else(|| panic!("terminal-CE input column {global} missing from canonical map"))
        })
        .collect()
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
        let mut repeated_end = index + 1;
        while repeated_end < values.len() && values[repeated_end] == values[index] {
            repeated_end += 1;
        }
        if repeated_end - index >= 4 {
            flush_literals(&mut literals, &mut pieces);
            pieces.push(format!("List.replicate {} {}", repeated_end - index, values[index]));
            index = repeated_end;
            continue;
        }
        if index + 3 < values.len() && values[index + 1] > values[index] {
            let step = values[index + 1] - values[index];
            let mut end = index + 2;
            while end < values.len() && values[end] > values[end - 1] && values[end] - values[end - 1] == step {
                end += 1;
            }
            if end - index >= 4 {
                flush_literals(&mut literals, &mut pieces);
                if values[index] == 0 && step == 1 {
                    pieces.push(format!("List.range {}", end - index));
                } else {
                    pieces.push(format!(
                        "((List.range {}).map (fun index => {} + {} * index))",
                        end - index,
                        values[index],
                        step,
                    ));
                }
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

fn stage_artifact(path: &Path, rendered: &str, drifted: &mut Vec<PathBuf>) {
    if fs::read_to_string(path).unwrap_or_default() != rendered {
        fs::write(path.with_extension("lean.expected"), rendered).expect("write reviewed terminal-CE artifact");
        drifted.push(path.to_path_buf());
    }
}

fn push_terms(tokens: &mut Vec<usize>, terms: &[(usize, u64)]) {
    tokens.push(terms.len());
    for &(column, coefficient) in terms {
        tokens.push(column);
        tokens.push(coefficient as usize);
    }
}

fn packed_instruction_tokens(instructions: &[Instruction]) -> Vec<usize> {
    let mut tokens = vec![instructions.len()];
    for instruction in instructions {
        match instruction {
            Instruction::Define(definition) => {
                tokens.push(0);
                tokens.push(definition.output);
                match &definition.rhs {
                    Rhs::Linear(terms) => {
                        tokens.push(0);
                        push_terms(&mut tokens, terms);
                    }
                    Rhs::Product(left, right) => {
                        tokens.push(1);
                        push_terms(&mut tokens, left);
                        push_terms(&mut tokens, right);
                    }
                }
            }
            Instruction::Check(row) => {
                tokens.push(1);
                push_terms(&mut tokens, &row.a);
                push_terms(&mut tokens, &row.b);
                push_terms(&mut tokens, &row.c);
            }
        }
    }
    tokens
}

fn render_packed(instructions: &[Instruction]) -> String {
    let tokens = packed_instruction_tokens(instructions);
    let chunks = tokens
        .chunks(TERMINAL_CE_PACKED_TOKEN_CHUNK)
        .map(|chunk| {
            format!(
                "\"{}\"",
                chunk
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",\n    ");
    format!(
        "import Nightstream.Implementation.R1CS.Core.PackedProgram\n\n\
         /-! Generated packed exact terminal-CE checked program. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe.Generated\n\n\
         def chunks : List String :=\n    [{chunks}]\n\n\
         def decoded : Option (List CheckedProgram.Instruction) :=\n\
           PackedProgram.decode chunks\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe.Generated\n"
    )
}

fn render_map(index: usize, values: &[usize]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.Relabel\n\n\
         /-! Generated exact terminal-CE claim column map {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe.GeneratedMaps\n\n\
         set_option maxRecDepth 262144\n\
         set_option maxHeartbeats 2000000\n\n\
         def claimMap{index} : List Nat :=\n    {}\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe.GeneratedMaps\n",
        lean_compact_nat_sequence(values),
    )
}

fn canonical_column(column: usize, shape: &CanonicalizedProgram) -> usize {
    shape
        .column_map
        .iter()
        .position(|candidate| *candidate == column)
        .unwrap_or_else(|| panic!("terminal-CE semantic column {column} missing from canonical map"))
}

fn canonical_columns(columns: &[usize], shape: &CanonicalizedProgram) -> Vec<usize> {
    columns
        .iter()
        .map(|column| canonical_column(*column, shape))
        .collect()
}

fn lean_k_columns(columns: &[[usize; 2]], shape: &CanonicalizedProgram) -> String {
    format!(
        "[{}]",
        columns
            .iter()
            .map(|columns| format!(
                "{{ c0 := {}, c1 := {} }}",
                canonical_column(columns[0], shape),
                canonical_column(columns[1], shape),
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_nested_columns(columns: &[Vec<usize>], shape: &CanonicalizedProgram) -> String {
    format!(
        "[{}]",
        columns
            .iter()
            .map(|row| format!("({})", lean_compact_nat_sequence(&canonical_columns(row, shape))))
            .collect::<Vec<_>>()
            .join(",\n    ")
    )
}

fn render_layout(audit: &TerminalCeClaimAudit, shape: &CanonicalizedProgram) -> String {
    let expected_public_width = audit
        .expected_public_width
        .map_or_else(|| "none".to_owned(), |width| format!("some {width}"));
    format!(
        "import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalCeCompiler\n\n\
         /-! Generated exact semantic column layout for one terminal-CE claim. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe\n\n\
         open Nightstream.Implementation.R1CS.TerminalCeCompiler\n\n\
         set_option maxHeartbeats 1000000\n\n\
         def layout : Layout where\n\
           normBound := {}\n\
           expectedPublicWidth := {}\n\
           structureRows := {}\n\
           structureColumns := {}\n\
           witnessRows := {}\n\
         witnessColumns := {}\n\
         witnessCols := {}\n\
         normFirstAllocatedColumn := {}\n\
         commitmentCols := {}\n\
           commitmentD := {}\n\
           commitmentKappa := {}\n\
           publicCols := {}\n\
           publicRows := {}\n\
           publicWidth := {}\n\
           publicInputLen := {}\n\
           pointCols := {}\n\
           evaluationCols := {}\n\
           constantTermCols := {}\n\
           ncPointCols := {}\n\
           ncEvaluationCols := {}\n\
           ncEvaluationLanes := {}\n\n\
         theorem layout_shape : ShapeValid layout where\n\
           witnessSize := by native_decide\n\
           commitmentSize := by native_decide\n\
           publicSize := by native_decide\n\
           publicRowsPositive := by native_decide\n\
           publicProjectionWithinStructure := by native_decide\n\
         publicWidthPinned := by rfl\n\
           constantTermSize := by native_decide\n\
           evaluationRowsNonempty := by native_decide\n\
           evaluationRowsEven := by native_decide\n\
           ncEvaluationSize := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe\n",
        audit.norm_bound,
        expected_public_width,
        audit.structure_rows,
        audit.structure_columns,
        audit.witness_rows,
        audit.witness_columns,
        lean_compact_nat_sequence(&canonical_columns(&audit.witness_cols, shape)),
        canonical_column(audit.norm_first_allocated_column, shape),
        lean_compact_nat_sequence(&canonical_columns(&audit.commitment_cols, shape)),
        audit.commitment_d,
        audit.commitment_kappa,
        lean_compact_nat_sequence(&canonical_columns(&audit.public_cols, shape)),
        audit.public_rows,
        audit.public_width,
        audit.public_input_len,
        lean_k_columns(&audit.point_cols, shape),
        lean_nested_columns(&audit.evaluation_cols, shape),
        lean_k_columns(&audit.constant_term_cols, shape),
        lean_k_columns(&audit.nc_point_cols, shape),
        lean_compact_nat_sequence(&canonical_columns(&audit.nc_evaluation_cols, shape)),
        audit.nc_evaluation_lanes,
    )
}

fn phase_schedule(builder: &R1csBuilder, claim: &ProgramRangeAudit) -> Vec<(&'static str, usize, usize)> {
    let mut phases = builder
        .row_family_ranges()
        .iter()
        .filter(|phase| {
            phase.name.starts_with("terminal_ce.claim.")
                && phase.row_start >= claim.row_start
                && phase.row_end <= claim.row_end
        })
        .map(|phase| {
            (
                phase.name,
                phase.row_start - claim.row_start,
                phase.row_end - claim.row_start,
            )
        })
        .collect::<Vec<_>>();
    phases.sort_by_key(|phase| phase.1);
    phases
}

fn render_profile(
    audits: &[&ProgramRangeAudit],
    program: &NormalizedProgram,
    canonical: &CanonicalizedProgram,
    phases: &[(&'static str, usize, usize)],
) -> String {
    let ranges = audits
        .iter()
        .map(|audit| {
            format!(
                "({}, {}, {})",
                audit.row_start, audit.row_end, audit.first_allocated_column
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let phases = phases
        .iter()
        .map(|(name, start, end)| format!("(\"{name}\", {start}, {end})"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifestSchema\n\n\
         /-! Generated exact-shape manifest for the direct terminal-CE claim compiler. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeProfile\n\n\
         def claimCount : Nat := {}\n\
         def rowCount : Nat := {}\n\
         def definitionCount : Nat := {}\n\
         def checkCount : Nat := {}\n\
         def canonicalColumnCount : Nat := {}\n\
         def claimRanges : List (Nat × Nat × Nat) := [{}]\n\
         def phaseRanges : List (String × Nat × Nat) := [{}]\n\n\
         def phaseRange (phase : String × Nat × Nat) :\n\
             FPrimeRecursiveManifest.RowRange where\n\
           name := phase.1\n\
           rowStart := phase.2.1\n\
           rowEnd := phase.2.2\n\
           nonzeroEntries := 0\n\
           sha256 := \"\"\n\n\
         theorem claim_count : claimRanges.length = claimCount := by native_decide\n\
         theorem phase_schedule : phaseRanges.map Prod.fst =\n\
             [\"terminal_ce.claim.commitment\", \"terminal_ce.claim.public_input\",\n\
              \"terminal_ce.claim.norm\", \"terminal_ce.claim.evaluations\",\n\
              \"terminal_ce.claim.constant_term\", \"terminal_ce.claim.nc_channel\"] := by\n\
           native_decide\n\
         theorem phase_coverage :\n\
             let ranges := phaseRanges.map phaseRange\n\
             FPrimeRecursiveManifest.covers 0 rowCount ranges = true := by\n\
           native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeProfile\n",
        audits.len(),
        program.instructions.len(),
        program.definition_count,
        program.check_count,
        canonical.column_map.len(),
        ranges,
        phases,
    )
}

fn render_artifact(
    audits: &[&ProgramRangeAudit],
    program: &NormalizedProgram,
    canonical: &CanonicalizedProgram,
    phases: &[(&'static str, usize, usize)],
) -> String {
    assert_eq!(phases.len(), 6, "terminal-CE compiler phases");
    let map_imports = (0..audits.len())
        .map(|index| format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalCeMap{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let maps = (0..audits.len())
        .map(|index| format!("GeneratedMaps.claimMap{index}"))
        .collect::<Vec<_>>()
        .join(",\n    ");
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalCePacked\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalCeLayout\n\
         {map_imports}\n\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalCeProfile\n\n\
         /-! Exact normalized checked program for every direct terminal-CE claim. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 524288\n\
         set_option maxHeartbeats 4000000\n\n\
         def inputColumns : List Nat :=\n    {}\n\n\
         def packedDecode : Option (List Instruction) := Generated.decoded\n\n\
         def instructions : List Instruction := packedDecode.getD []\n\n\
         def rows : List Row := CheckedProgram.rows instructions\n\
         def columnMaps : List (List Nat) :=\n    [{maps}]\n\
         def claimRows : List (List Row) :=\n\
           columnMaps.map fun columnMap => rows.map (Relabel.row columnMap)\n\
         def terminalCeRows : List Row := claimRows.flatten\n\n\
         def schedule : TerminalCeCompiler.Schedule where\n\
           commitmentStart := {}\n\
           commitmentEnd := {}\n\
           publicInputStart := {}\n\
           publicInputEnd := {}\n\
           normStart := {}\n\
           normEnd := {}\n\
           evaluationsStart := {}\n\
           evaluationsEnd := {}\n\
           constantTermStart := {}\n\
           constantTermEnd := {}\n\
           ncChannelStart := {}\n\
           ncChannelEnd := {}\n\n\
         def program : TerminalCeCompiler.Program where\n\
           layout := layout\n\
           schedule := schedule\n\
           inputColumns := inputColumns\n\
           instructions := instructions\n\n\
         theorem packed_decode_ok : packedDecode = some instructions := by native_decide\n\
         theorem instructions_length : instructions.length = {} := by native_decide\n\
         theorem rows_length : rows.length = {} := by native_decide\n\
         theorem definitions_length : (definitions instructions).length = {} := by native_decide\n\
         theorem checks_length : (checks instructions).length = {} := by native_decide\n\
         theorem definitions_canonical :\n\
             ∀ definition ∈ definitions instructions, definition.Canonical := by native_decide\n\
         theorem definitions_wellFormed :\n\
             WellFormed inputColumns (definitions instructions) := by native_decide\n\
         theorem checks_reference :\n\
             ChecksReference (knownAfter inputColumns (definitions instructions)) instructions := by native_decide\n\
         theorem column_maps_length : columnMaps.length = {} := by native_decide\n\
         theorem column_maps_one :\n\
             ∀ columnMap ∈ columnMaps, Relabel.column columnMap 0 = 0 := by native_decide\n\
         theorem column_maps_injective :\n\
             ∀ columnMap ∈ columnMaps, columnMap.Nodup := by native_decide\n\n\
         theorem commitment_checks_match :\n\
             LinearOutputs.rows program.commitmentChecks =\n\
               checks program.commitmentInstructions := by native_decide\n\
         theorem public_program_match :\n\
             CheckedProgram.rows program.publicInstructions =\n\
               LinearOutputs.rows (TerminalCeCompiler.projectionChecks layout) := by\n\
           native_decide\n\
         theorem norm_program_match :\n\
             program.normInstructionsSlice =\n\
               TerminalCeCompiler.normInstructions layout := by native_decide\n\
         theorem evaluation_checks_match :\n\
             LinearOutputs.rows program.evaluationChecks =\n\
               checks program.evaluationInstructions := by native_decide\n\
         theorem constant_term_program_match :\n\
             CheckedProgram.rows program.constantTermInstructions =\n\
               LinearOutputs.rows (TerminalCeCompiler.constantTermChecks layout) := by\n\
           native_decide\n\
         theorem nc_checks_match :\n\
             LinearOutputs.rows program.ncChecks =\n\
               checks program.ncInstructions := by native_decide\n\n\
         theorem commitment_check_outputs :\n\
             program.commitmentChecks.map LinearOutputs.Check.output =\n\
               layout.commitmentCols := by native_decide\n\
         theorem evaluation_check_outputs :\n\
             program.evaluationChecks.map LinearOutputs.Check.output =\n\
               layout.evaluationCols.flatten := by native_decide\n\
         theorem nc_check_outputs :\n\
             program.ncChecks.map LinearOutputs.Check.output =\n\
               layout.ncEvaluationCols := by native_decide\n\
         theorem linear_checks_canonical :\n\
             LinearOutputs.Canonical program.commitmentChecks ∧\n\
             LinearOutputs.Canonical program.evaluationChecks ∧\n\
             LinearOutputs.Canonical program.ncChecks := by native_decide\n\
         theorem semantic_columns_known :\n\
             ∀ column ∈ TerminalCeCompiler.semanticColumns layout,\n\
               column ∈ knownAfter inputColumns (definitions instructions) := by\n\
           native_decide\n\
         theorem semantic_columns_input :\n\
             ∀ column ∈ TerminalCeCompiler.semanticColumns layout,\n\
               column ∈ inputColumns := by native_decide\n\
         theorem public_rows_reference_input :\n\
             ∀ row ∈ CheckedProgram.rows program.publicInstructions,\n\
               ∀ column ∈ rowRefs row, column ∈ inputColumns := by native_decide\n\
         theorem constant_term_rows_reference_input :\n\
             ∀ row ∈ CheckedProgram.rows program.constantTermInstructions,\n\
               ∀ column ∈ rowRefs row, column ∈ inputColumns := by native_decide\n\
         theorem norm_definitions_in_program :\n\
             ∀ definition ∈ definitions (TerminalCeCompiler.normInstructions layout),\n\
               definition ∈ definitions instructions := by native_decide\n\
         theorem phase_partition :\n\
             instructions = program.commitmentInstructions ++\n\
               program.publicInstructions ++ program.normInstructionsSlice ++\n\
               program.evaluationInstructions ++ program.constantTermInstructions ++\n\
               program.ncInstructions := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe\n",
        lean_compact_nat_sequence(&canonical_input_columns(program, canonical)),
        phases[0].1,
        phases[0].2,
        phases[1].1,
        phases[1].2,
        phases[2].1,
        phases[2].2,
        phases[3].1,
        phases[3].2,
        phases[4].1,
        phases[4].2,
        phases[5].1,
        phases[5].2,
        program.instructions.len(),
        program.instructions.len(),
        program.definition_count,
        program.check_count,
        audits.len(),
    )
}

pub fn compare_terminal_ce_profiles(builder: &R1csBuilder) {
    let mut audits = builder
        .program_range_audits()
        .iter()
        .filter(|audit| audit.name == "terminal_ce.claim")
        .collect::<Vec<_>>();
    audits.sort_by_key(|audit| audit.row_start);
    assert_eq!(audits.len(), 14, "direct terminal-CE child claims");
    let mut semantic_audits = builder
        .terminal_ce_claim_audits()
        .iter()
        .collect::<Vec<_>>();
    semantic_audits.sort_by_key(|audit| audit.row_start);
    assert_eq!(semantic_audits.len(), audits.len(), "terminal-CE semantic layouts");
    for (index, (program, semantic)) in audits.iter().zip(&semantic_audits).enumerate() {
        assert_eq!(
            semantic.row_start, program.row_start,
            "terminal-CE claim {index} semantic start"
        );
        assert_eq!(
            semantic.row_end, program.row_end,
            "terminal-CE claim {index} semantic end"
        );
        assert_eq!(
            semantic.first_allocated_column, program.first_allocated_column,
            "terminal-CE claim {index} semantic allocation boundary",
        );
    }

    let normalized = audits
        .iter()
        .map(|audit| normalize_range(builder, audit.row_start, audit.row_end, audit.first_allocated_column))
        .collect::<Vec<_>>();
    let canonical = normalized
        .iter()
        .map(canonicalize_program)
        .collect::<Vec<_>>();
    let expected_phases = phase_schedule(builder, audits[0]);
    assert_eq!(expected_phases.len(), 6, "terminal-CE semantic phases");

    for index in 0..audits.len() {
        let program = &normalized[index];
        let shape = &canonical[index];
        assert_eq!(
            program.instructions.len(),
            normalized[0].instructions.len(),
            "terminal-CE claim {index} row count",
        );
        if let Some(position) = shape
            .instructions
            .iter()
            .zip(&canonical[0].instructions)
            .position(|(left, right)| left != right)
        {
            panic!("terminal-CE claim {index} canonical instruction drift at row {position}");
        }
        assert_eq!(
            shape.instructions.len(),
            canonical[0].instructions.len(),
            "terminal-CE claim {index} canonical instruction length",
        );
        assert_eq!(
            canonical_input_columns(program, shape),
            canonical_input_columns(&normalized[0], &canonical[0]),
            "terminal-CE claim {index} input ownership",
        );
        assert_eq!(
            phase_schedule(builder, audits[index]),
            expected_phases,
            "terminal-CE claim {index} phase schedule",
        );
        let relabelled = relabel_instructions(&shape.instructions, &shape.column_map);
        if let Some(position) = relabelled
            .iter()
            .zip(&program.instructions)
            .position(|(left, right)| left != right)
        {
            panic!("terminal-CE claim {index} relabel drift at row {position}");
        }
        assert_eq!(
            relabelled.len(),
            program.instructions.len(),
            "terminal-CE claim {index} relabelled instruction length",
        );
        assert_eq!(
            render_layout(semantic_audits[index], shape),
            render_layout(semantic_audits[0], &canonical[0]),
            "terminal-CE claim {index} canonical semantic layout",
        );
    }

    let mut drifted = Vec::new();
    stage_artifact(
        &formal_repo_root().join(TERMINAL_CE_PROFILE_PATH),
        &render_profile(&audits, &normalized[0], &canonical[0], &expected_phases),
        &mut drifted,
    );
    stage_artifact(
        &formal_repo_root().join(TERMINAL_CE_PACKED_PATH),
        &render_packed(&canonical[0].instructions),
        &mut drifted,
    );
    stage_artifact(
        &formal_repo_root().join(TERMINAL_CE_LAYOUT_PATH),
        &render_layout(semantic_audits[0], &canonical[0]),
        &mut drifted,
    );
    for (index, shape) in canonical.iter().enumerate() {
        stage_artifact(
            &formal_repo_root().join(format!("{TERMINAL_CE_MAP_PREFIX}{index}.lean")),
            &render_map(index, &shape.column_map),
            &mut drifted,
        );
    }
    stage_artifact(
        &formal_repo_root().join(TERMINAL_CE_ARTIFACT_PATH),
        &render_artifact(&audits, &normalized[0], &canonical[0], &expected_phases),
        &mut drifted,
    );
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "full-history terminal-CE artifacts drifted: {drifted:?}",
    );
}
