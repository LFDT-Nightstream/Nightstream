use super::*;
use neo_fold_clean::engine::r1cs_circuit::builder::{PiDecCommitmentAudit, PiDecStrictAudit, ProgramRangeAudit};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const PI_DEC_ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryPiDecArtifact.lean";
const PI_DEC_SHARD_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPiDecInstructions";
const PI_DEC_RECURSIVE_MAP_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPiDecRecursiveMap";
const PI_DEC_TERMINAL_MAP_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPiDecTerminalMap";
const PI_DEC_TERMINAL_CE_MAP_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPiDecTerminalCeMap";
const PI_DEC_LAYOUT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPiDecLayout.lean";
const PI_DEC_SHARD_SIZE: usize = 250;
const PI_DEC_MAP_SHARD_SIZE: usize = 1_000;

fn neg_coefficient(coefficient: u64) -> u64 {
    if coefficient == 0 {
        0
    } else {
        F::ORDER_U64 - coefficient
    }
}

fn check_row(a: Vec<(usize, u64)>, b: Vec<(usize, u64)>, c: Vec<(usize, u64)>) -> Instruction {
    Instruction::Check(checked_program_artifact_support::Row { a, b, c })
}

fn equality_check(lhs: usize, rhs: usize) -> Instruction {
    check_row(vec![(lhs, 1), (rhs, F::ORDER_U64 - 1)], vec![(0, 1)], Vec::new())
}

fn zero_check(column: usize) -> Instruction {
    check_row(vec![(column, 1)], vec![(0, 1)], Vec::new())
}

fn recomposition_check(parent: usize, children: impl Iterator<Item = usize>, powers: &[u64]) -> Instruction {
    let mut terms = vec![(parent, 1)];
    terms.extend(
        children
            .zip(powers)
            .map(|(column, &coefficient)| (column, neg_coefficient(coefficient))),
    );
    check_row(terms, vec![(0, 1)], Vec::new())
}

fn commitment_recomposition(
    parent: &PiDecCommitmentAudit,
    children: &[&PiDecCommitmentAudit],
    powers: &[u64],
    out: &mut Vec<Instruction>,
) {
    for child in children {
        out.push(equality_check(parent.d_col, child.d_col));
        out.push(equality_check(parent.kappa_col, child.kappa_col));
    }
    for lane in 0..parent.data_cols.len() {
        out.push(recomposition_check(
            parent.data_cols[lane],
            children.iter().map(|child| child.data_cols[lane]),
            powers,
        ));
    }
}

fn factor_terms(column: usize, alphabet_value: i64) -> Vec<(usize, u64)> {
    let mut terms = vec![(column, 1)];
    let constant = if alphabet_value >= 0 {
        -F::from_u64(alphabet_value as u64)
    } else {
        F::from_u64((-alphabet_value) as u64)
    };
    if constant != F::ZERO {
        terms.push((0, constant.as_canonical_u64()));
    }
    terms
}

fn centered_alphabet_program(column: usize, radix: u32, next_output: &mut usize, out: &mut Vec<Instruction>) {
    let bound = radix as i64 - 1;
    let alphabet = (-bound..=bound).collect::<Vec<_>>();
    let mut previous = factor_terms(column, alphabet[0]);
    for (index, &value) in alphabet.iter().enumerate().skip(1) {
        let factor = factor_terms(column, value);
        if index + 1 == alphabet.len() {
            out.push(check_row(previous, factor, Vec::new()));
            break;
        } else {
            let output = *next_output;
            *next_output += 1;
            out.push(Instruction::Define(checked_program_artifact_support::Definition {
                output,
                rhs: Rhs::Product(previous, factor),
            }));
            previous = vec![(output, 1)];
        }
    }
}

fn unique_in_order(values: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    values.filter(|value| seen.insert(*value)).collect()
}

/// Reconstruct the strict PiDEC compiler from its semantic wire schedule.
/// Equality with the normalized emitted program is the fail-closed bridge:
/// the schedule is never accepted as semantic authority by itself.
fn expected_instructions(audit: &PiDecStrictAudit) -> Vec<Instruction> {
    assert!(audit.radix >= 2, "strict PiDEC radix");
    let mut power = F::ONE;
    let radix = F::from_u64(audit.radix as u64);
    let powers = (0..audit.children.len())
        .map(|_| {
            let result = power.as_canonical_u64();
            power *= radix;
            result
        })
        .collect::<Vec<_>>();
    let children = audit.children.iter().collect::<Vec<_>>();
    let active_cols = audit.parent.m_in.div_ceil(neo_math::D);
    let mut out = Vec::with_capacity(audit.row_end - audit.row_start);

    for lane in 0..audit.parent.commitment.data_cols.len() {
        out.push(recomposition_check(
            audit.parent.commitment.data_cols[lane],
            children
                .iter()
                .map(|child| child.commitment.data_cols[lane]),
            &powers,
        ));
    }
    match &audit.parent.adv {
        None => assert!(children.iter().all(|child| child.adv.is_none())),
        Some(parent) => {
            let child_adv = children
                .iter()
                .map(|child| child.adv.as_ref().expect("PiDEC child adv"))
                .collect::<Vec<_>>();
            for (parent_coordinate, child_coordinates) in [
                (&parent.ops, child_adv.iter().map(|adv| &adv.ops).collect::<Vec<_>>()),
                (&parent.is, child_adv.iter().map(|adv| &adv.is).collect::<Vec<_>>()),
                (&parent.fs, child_adv.iter().map(|adv| &adv.fs).collect::<Vec<_>>()),
            ] {
                commitment_recomposition(parent_coordinate, &child_coordinates, &powers, &mut out);
            }
        }
    }
    for row in 0..audit.parent.x_rows {
        for column in 0..active_cols {
            let lane = row * audit.parent.x_width + column;
            out.push(recomposition_check(
                audit.parent.x_cols[lane],
                children.iter().map(|child| child.x_cols[lane]),
                &powers,
            ));
        }
    }
    for (row, parent) in audit.parent.y_ring_cols.iter().enumerate() {
        for lane in 0..parent.len() {
            out.push(recomposition_check(
                parent[lane],
                children.iter().map(|child| child.y_ring_cols[row][lane]),
                &powers,
            ));
        }
    }

    for child in &children {
        out.push(equality_check(audit.parent.commitment.d_col, child.commitment.d_col));
        out.push(equality_check(
            audit.parent.commitment.kappa_col,
            child.commitment.kappa_col,
        ));
        out.push(equality_check(audit.parent.x_rows_col, child.x_rows_col));
        out.push(equality_check(audit.parent.x_width_col, child.x_width_col));
        out.push(equality_check(audit.parent.m_in_col, child.m_in_col));
    }
    for child in &children {
        for (parent, child) in audit.parent.r_cols.iter().zip(&child.r_cols) {
            out.push(equality_check(parent[0], child[0]));
            out.push(equality_check(parent[1], child[1]));
        }
    }
    for child in &children {
        for (parent, child) in audit.parent.s_col_cols.iter().zip(&child.s_col_cols) {
            out.push(equality_check(parent[0], child[0]));
            out.push(equality_check(parent[1], child[1]));
        }
    }
    for claim in std::iter::once(&audit.parent).chain(children.iter().copied()) {
        let inactive = (0..claim.x_rows)
            .flat_map(|row| (active_cols..claim.x_width).map(move |column| claim.x_cols[row * claim.x_width + column]));
        out.extend(unique_in_order(inactive).into_iter().map(zero_check));
    }

    let mut next_output = audit.first_allocated_column;
    for child in &children {
        for row in 0..child.x_rows {
            for column in 0..active_cols {
                centered_alphabet_program(
                    child.x_cols[row * child.x_width + column],
                    audit.radix,
                    &mut next_output,
                    &mut out,
                );
            }
        }
    }
    for claim in std::iter::once(&audit.parent).chain(children.iter().copied()) {
        for (ct, y_ring) in claim.ct_cols.iter().zip(&claim.y_ring_cols) {
            out.push(equality_check(ct[0], y_ring[0]));
            out.push(equality_check(ct[1], y_ring[1]));
        }
    }
    for claim in std::iter::once(&audit.parent).chain(children.iter().copied()) {
        for row in &claim.y_ring_cols {
            out.extend(row.iter().skip(neo_math::D * 2).copied().map(zero_check));
        }
    }
    for child in &children {
        for lane in 0..audit.parent.fold_digest_cols.len() {
            out.push(equality_check(
                child.fold_digest_cols[lane],
                audit.parent.fold_digest_cols[lane],
            ));
        }
    }
    assert_eq!(
        out.len(),
        audit.row_end - audit.row_start,
        "strict PiDEC reconstructed row count"
    );
    out
}

fn stage_artifact(path: &Path, rendered: &str, drifted: &mut Vec<PathBuf>) {
    if fs::read_to_string(path).unwrap_or_default() != rendered {
        fs::write(path.with_extension("lean.expected"), rendered).expect("write reviewed PiDEC artifact");
        drifted.push(path.to_path_buf());
    }
}

fn canonical_input_columns(program: &NormalizedProgram, canonical: &CanonicalizedProgram) -> Vec<usize> {
    program
        .input_columns
        .iter()
        .map(|global| {
            canonical
                .column_map
                .iter()
                .position(|candidate| candidate == global)
                .unwrap_or_else(|| panic!("PiDEC input column {global} missing from canonical map"))
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
            pieces.push(format!("List.replicate {} {}", repeated_end - index, values[index],));
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

fn local_column(column_map: &[usize], global: usize) -> usize {
    column_map
        .iter()
        .position(|&candidate| candidate == global)
        .unwrap_or_else(|| panic!("strict PiDEC column {global} absent from normalized program"))
}

fn local_columns(column_map: &[usize], globals: &[usize]) -> Vec<usize> {
    globals
        .iter()
        .map(|&global| local_column(column_map, global))
        .collect()
}

fn render_pairs(column_map: &[usize], pairs: &[[usize; 2]]) -> String {
    let values = pairs
        .iter()
        .map(|pair| {
            format!(
                "({}, {})",
                local_column(column_map, pair[0]),
                local_column(column_map, pair[1]),
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", values.join(", "))
}

fn render_commitment(column_map: &[usize], commitment: &PiDecCommitmentAudit) -> String {
    format!(
        "{{ dCol := {}, kappaCol := {}, dataCols := {} }}",
        local_column(column_map, commitment.d_col),
        local_column(column_map, commitment.kappa_col),
        lean_compact_nat_sequence(&local_columns(column_map, &commitment.data_cols)),
    )
}

fn render_claim(
    column_map: &[usize],
    claim: &neo_fold_clean::engine::r1cs_circuit::builder::PiDecClaimAudit,
) -> String {
    let active_width = claim.m_in.div_ceil(neo_math::D);
    let active_x = (0..claim.x_rows)
        .flat_map(|row| (0..active_width).map(move |column| claim.x_cols[row * claim.x_width + column]))
        .collect::<Vec<_>>();
    let inactive_x = (0..claim.x_rows)
        .flat_map(|row| (active_width..claim.x_width).map(move |column| claim.x_cols[row * claim.x_width + column]))
        .collect::<Vec<_>>();
    let inactive_col = *inactive_x
        .first()
        .expect("strict PiDEC has inactive X columns");
    assert!(
        inactive_x.iter().all(|&column| column == inactive_col),
        "strict PiDEC inactive X columns share one constrained zero wire",
    );
    let adv = match &claim.adv {
        None => "none".to_string(),
        Some(adv) => format!(
            "some {{ ops := {}, is := {}, fs := {} }}",
            render_commitment(column_map, &adv.ops),
            render_commitment(column_map, &adv.is),
            render_commitment(column_map, &adv.fs),
        ),
    };
    let y_ring = claim
        .y_ring_cols
        .iter()
        .map(|row| lean_compact_nat_sequence(&local_columns(column_map, row)))
        .collect::<Vec<_>>()
        .join(",\n        ");
    format!(
        "{{\n      commitment := {}\n      adv := {adv}\n      xActiveCols := {}\n      xInactiveCol := {}\n      xRows := {}\n      xWidth := {}\n      xRowsCol := {}\n      xWidthCol := {}\n      mIn := {}\n      mInCol := {}\n      yRingCols :=\n        [{y_ring}]\n      ctCols := {}\n      rCols := {}\n      sColCols := {}\n      foldDigestCols := {} }}",
        render_commitment(column_map, &claim.commitment),
        lean_compact_nat_sequence(&local_columns(column_map, &active_x)),
        local_column(column_map, inactive_col),
        claim.x_rows,
        claim.x_width,
        local_column(column_map, claim.x_rows_col),
        local_column(column_map, claim.x_width_col),
        claim.m_in,
        local_column(column_map, claim.m_in_col),
        render_pairs(column_map, &claim.ct_cols),
        render_pairs(column_map, &claim.r_cols),
        render_pairs(column_map, &claim.s_col_cols),
        lean_compact_nat_sequence(&local_columns(column_map, &claim.fold_digest_cols)),
    )
}

fn render_layout(audit: &PiDecStrictAudit, column_map: &[usize]) -> String {
    let children = audit
        .children
        .iter()
        .map(|child| format!("    {}", render_claim(column_map, child)))
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCompiler\n\n\
         /-! Generated exact strict-PiDEC semantic wire layout. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec\n\n\
         def layout : PiDecStrictCompiler.Layout := {{\n\
           radix := {}\n\
           ringDimension := {}\n\
           extensionLimbs := 2\n\
           firstAllocatedColumn := {}\n\
           parent := {}\n\
           children :=\n\
         [{children}] }}\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec\n",
        audit.radix,
        neo_math::D,
        local_column(column_map, audit.first_allocated_column),
        render_claim(column_map, &audit.parent),
    )
}

fn render_shard(index: usize, instructions: &[Instruction]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated exact PiDEC instruction shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec.Generated\n\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 262144\n\
         set_option maxHeartbeats 2000000\n\n\
         def instructions{index} : List Instruction :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec.Generated\n",
        lean_instructions(instructions),
    )
}

fn render_map_shard(kind: &str, index: usize, values: &[usize]) -> String {
    let definition = format!("{}Map{index}", kind.to_ascii_lowercase());
    format!(
        "import Nightstream.Implementation.R1CS.Core.Relabel\n\n\
         /-! Generated exact PiDEC {kind} column-map shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec.GeneratedMaps\n\n\
         set_option maxRecDepth 262144\n\
         set_option maxHeartbeats 2000000\n\n\
         def {definition} : List Nat := {}\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec.GeneratedMaps\n",
        lean_compact_nat_sequence(values),
    )
}

fn render_artifact(
    program: &NormalizedProgram,
    canonical: &CanonicalizedProgram,
    terminal_ce_input_columns: &[usize],
    recursive: &ProgramRangeAudit,
    terminal: &ProgramRangeAudit,
    terminal_ce: &PiDecStrictAudit,
    recursive_map_count: usize,
    terminal_map_count: usize,
    terminal_ce_map_count: usize,
) -> String {
    let shard_count = canonical.instructions.len().div_ceil(PI_DEC_SHARD_SIZE);
    let instruction_imports = (0..shard_count)
        .map(|index| format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiDecInstructions{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let map_imports =
        (0..recursive_map_count)
            .map(|index| format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiDecRecursiveMap{index}"))
            .chain((0..terminal_map_count).map(|index| {
                format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiDecTerminalMap{index}")
            }))
            .chain((0..terminal_ce_map_count).map(|index| {
                format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiDecTerminalCeMap{index}")
            }))
            .collect::<Vec<_>>()
            .join("\n");
    let instructions = (0..shard_count)
        .map(|index| format!("Generated.instructions{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    let recursive_map = (0..recursive_map_count)
        .map(|index| format!("GeneratedMaps.recursiveMap{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    let terminal_map = (0..terminal_map_count)
        .map(|index| format!("GeneratedMaps.terminalMap{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    let terminal_ce_map = (0..terminal_ce_map_count)
        .map(|index| format!("GeneratedMaps.terminalceMap{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    format!(
        "{instruction_imports}\n\
         {map_imports}\n\n\
         import Nightstream.Implementation.R1CS.Core.Relabel\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPiDecLayout\n\n\
         /-! Exact normalized PiDEC verifier program shared by recursive NIFS, terminal NIFS, and direct terminal CE. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 524288\n\
         set_option maxHeartbeats 2000000\n\n\
         def inputColumns : List Nat := {}\n\
         def terminalCeInputColumns : List Nat := {}\n\
         def recursiveColumnMap : List Nat :=\n    {recursive_map}\n\
         def terminalColumnMap : List Nat :=\n    {terminal_map}\n\
         def terminalCeColumnMap : List Nat :=\n    {terminal_ce_map}\n\
         def recursiveRowStart : Nat := {}\n\
         def recursiveRowEnd : Nat := {}\n\
         def terminalRowStart : Nat := {}\n\
         def terminalRowEnd : Nat := {}\n\
         def terminalCeRowStart : Nat := {}\n\
         def terminalCeRowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\
         def definitionCount : Nat := {}\n\
         def checkCount : Nat := {}\n\n\
         def instructions : List Instruction :=\n    {instructions}\n\n\
         def rows : List Row := CheckedProgram.rows instructions\n\
         def recursiveRows : List Row := rows.map (Relabel.row recursiveColumnMap)\n\
         def terminalRows : List Row := rows.map (Relabel.row terminalColumnMap)\n\n\
         def terminalCeRows : List Row := rows.map (Relabel.row terminalCeColumnMap)\n\n\
         theorem instructions_length : instructions.length = rowCount := by native_decide\n\
         theorem rows_length : rows.length = rowCount := by native_decide\n\
         theorem recursiveRows_length : recursiveRows.length = rowCount := by native_decide\n\
         theorem terminalRows_length : terminalRows.length = rowCount := by native_decide\n\
         theorem terminalCeRows_length : terminalCeRows.length = rowCount := by native_decide\n\
         theorem definitions_length : (definitions instructions).length = definitionCount := by native_decide\n\
         theorem checks_length : (checks instructions).length = checkCount := by native_decide\n\
         theorem definitions_canonical :\n\
             ∀ definition ∈ definitions instructions, definition.Canonical := by native_decide\n\
         theorem definitions_wellFormed :\n\
             WellFormed inputColumns (definitions instructions) := by native_decide\n\
         theorem terminal_ce_definitions_wellFormed :\n\
             WellFormed terminalCeInputColumns (definitions instructions) := by native_decide\n\
         theorem checks_reference :\n\
             ChecksReference (knownAfter inputColumns (definitions instructions)) instructions := by native_decide\n\
         theorem recursive_map_one : Relabel.column recursiveColumnMap 0 = 0 := by native_decide\n\
         theorem terminal_map_one : Relabel.column terminalColumnMap 0 = 0 := by native_decide\n\
         theorem terminal_ce_map_one : Relabel.column terminalCeColumnMap 0 = 0 := by native_decide\n\
         theorem recursive_map_injective : recursiveColumnMap.Nodup := by native_decide\n\
         theorem terminal_map_injective : terminalColumnMap.Nodup := by native_decide\n\
         theorem terminal_ce_map_injective : terminalCeColumnMap.Nodup := by native_decide\n\
         theorem instructions_match_compiler :\n\
             instructions = PiDecStrictCompiler.instructions layout := by native_decide\n\
         theorem checks_match_compiler :\n\
             checks instructions = PiDecStrictCompiler.checkRows layout := by native_decide\n\n\
        end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec\n",
        lean_compact_nat_sequence(&canonical_input_columns(program, canonical)),
        lean_compact_nat_sequence(terminal_ce_input_columns),
        recursive.row_start,
        recursive.row_end,
        terminal.row_start,
        terminal.row_end,
        terminal_ce.row_start,
        terminal_ce.row_end,
        recursive.row_end - recursive.row_start,
        program.definition_count,
        program.check_count,
    )
}

pub fn compare_pi_dec_artifacts(builder: &R1csBuilder) {
    let mut audits = builder
        .program_range_audits()
        .iter()
        .filter(|audit| audit.name == "nifs.pi_dec")
        .collect::<Vec<_>>();
    audits.sort_by_key(|audit| audit.row_start);
    assert_eq!(audits.len(), 2, "recursive and terminal PiDEC programs");

    let normalized = audits
        .iter()
        .map(|audit| normalize_range(builder, audit.row_start, audit.row_end, audit.first_allocated_column))
        .collect::<Vec<_>>();
    let semantic_audits = audits
        .iter()
        .map(|range| {
            let matches = builder
                .pi_dec_strict_audits()
                .iter()
                .filter(|audit| audit.row_start == range.row_start && audit.row_end == range.row_end)
                .collect::<Vec<_>>();
            assert_eq!(
                matches.len(),
                1,
                "strict PiDEC semantic schedule at {}",
                range.row_start
            );
            matches[0]
        })
        .collect::<Vec<_>>();
    for ((range, program), semantic) in audits.iter().zip(&normalized).zip(&semantic_audits) {
        assert_eq!(
            (semantic.row_start, semantic.row_end, semantic.first_allocated_column),
            (range.row_start, range.row_end, range.first_allocated_column),
            "strict PiDEC semantic/program boundary",
        );
        assert_eq!(
            expected_instructions(semantic),
            program.instructions,
            "strict PiDEC schedule must reconstruct every normalized emitted row at {}",
            range.row_start,
        );
    }
    let canonical = normalized
        .iter()
        .map(canonicalize_program)
        .collect::<Vec<_>>();
    for ((audit, program), canonical) in audits.iter().zip(&normalized).zip(&canonical) {
        assert_eq!(
            relabel_instructions(&canonical.instructions, &canonical.column_map),
            program.instructions,
            "{} PiDEC canonicalization must preserve every exact row",
            audit.row_start,
        );
    }
    assert_eq!(
        canonical[0].instructions, canonical[1].instructions,
        "recursive and terminal PiDEC must share one normalized row program"
    );
    assert_eq!(
        canonical_input_columns(&normalized[0], &canonical[0]),
        canonical_input_columns(&normalized[1], &canonical[1]),
        "recursive and terminal PiDEC must share one normalized input layout"
    );
    let recursive_layout = render_layout(semantic_audits[0], &canonical[0].column_map);
    let terminal_layout = render_layout(semantic_audits[1], &canonical[1].column_map);
    assert_eq!(
        recursive_layout, terminal_layout,
        "recursive and terminal PiDEC must share one semantic wire layout"
    );

    let terminal_ce_audits = builder
        .pi_dec_strict_audits()
        .iter()
        .filter(|semantic| {
            !audits
                .iter()
                .any(|range| semantic.row_start == range.row_start && semantic.row_end == range.row_end)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        terminal_ce_audits.len(),
        1,
        "one direct terminal-CE strict PiDEC program"
    );
    let terminal_ce_audit = terminal_ce_audits[0];
    let terminal_ce_normalized = normalize_range(
        builder,
        terminal_ce_audit.row_start,
        terminal_ce_audit.row_end,
        terminal_ce_audit.first_allocated_column,
    );
    assert_eq!(
        expected_instructions(terminal_ce_audit),
        terminal_ce_normalized.instructions,
        "direct terminal-CE strict PiDEC schedule must reconstruct every normalized emitted row",
    );
    let terminal_ce_canonical = canonicalize_program(&terminal_ce_normalized);
    assert_eq!(
        relabel_instructions(&terminal_ce_canonical.instructions, &terminal_ce_canonical.column_map,),
        terminal_ce_normalized.instructions,
        "direct terminal-CE PiDEC canonicalization must preserve every exact row",
    );
    assert_eq!(
        canonical[0].instructions, terminal_ce_canonical.instructions,
        "direct terminal-CE and NIFS PiDEC must share one normalized row program"
    );
    let terminal_ce_layout = render_layout(terminal_ce_audit, &terminal_ce_canonical.column_map);
    assert_eq!(
        recursive_layout, terminal_ce_layout,
        "direct terminal-CE and NIFS PiDEC must share one semantic wire layout"
    );

    let mut drifted = Vec::new();
    stage_artifact(
        &formal_repo_root().join(PI_DEC_LAYOUT_PATH),
        &recursive_layout,
        &mut drifted,
    );
    for (index, shard) in canonical[0]
        .instructions
        .chunks(PI_DEC_SHARD_SIZE)
        .enumerate()
    {
        stage_artifact(
            &formal_repo_root().join(format!("{PI_DEC_SHARD_PREFIX}{index}.lean")),
            &render_shard(index, shard),
            &mut drifted,
        );
    }
    for (index, map) in canonical[0]
        .column_map
        .chunks(PI_DEC_MAP_SHARD_SIZE)
        .enumerate()
    {
        stage_artifact(
            &formal_repo_root().join(format!("{PI_DEC_RECURSIVE_MAP_PREFIX}{index}.lean")),
            &render_map_shard("Recursive", index, map),
            &mut drifted,
        );
    }
    for (index, map) in canonical[1]
        .column_map
        .chunks(PI_DEC_MAP_SHARD_SIZE)
        .enumerate()
    {
        stage_artifact(
            &formal_repo_root().join(format!("{PI_DEC_TERMINAL_MAP_PREFIX}{index}.lean")),
            &render_map_shard("Terminal", index, map),
            &mut drifted,
        );
    }
    for (index, map) in terminal_ce_canonical
        .column_map
        .chunks(PI_DEC_MAP_SHARD_SIZE)
        .enumerate()
    {
        stage_artifact(
            &formal_repo_root().join(format!("{PI_DEC_TERMINAL_CE_MAP_PREFIX}{index}.lean")),
            &render_map_shard("TerminalCe", index, map),
            &mut drifted,
        );
    }
    stage_artifact(
        &formal_repo_root().join(PI_DEC_ARTIFACT_PATH),
        &render_artifact(
            &normalized[0],
            &canonical[0],
            &canonical_input_columns(&terminal_ce_normalized, &terminal_ce_canonical),
            audits[0],
            audits[1],
            terminal_ce_audit,
            canonical[0]
                .column_map
                .len()
                .div_ceil(PI_DEC_MAP_SHARD_SIZE),
            canonical[1]
                .column_map
                .len()
                .div_ceil(PI_DEC_MAP_SHARD_SIZE),
            terminal_ce_canonical
                .column_map
                .len()
                .div_ceil(PI_DEC_MAP_SHARD_SIZE),
        ),
        &mut drifted,
    );
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "full-history PiDEC artifacts drifted: {drifted:?}"
    );
}
