import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.InitialArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.PaddingArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceProgram
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalArtifact

/-!
Ordered checked-program reconstruction for the fixed production combined-NC
source relation.

Owns: the exact 8,021-row source schedule, its classification into 7,969
canonical SSA definitions and 52 verifier checks, and coefficient-level row
agreement with the generated source artifact.

Does not own: source-to-selective lowering, source assignment construction,
check truth, transcript order, parent or raw-child authority, commitment
binding, costs, or permission to remove rows.

The classification is not inferred from stage labels or contiguous ranges.
Padding definitions come from the independently reconstructed zero-padding
schedule; initial and terminal definitions come from their typed programs;
and each round uses the exact generated `RawRoundMap.columnMap` already bound
to its 30 literal source rows by `RoundArtifactValid`.

Assurance tier: artifact-checked for this fixed generated source profile once
this leaf validates.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.StageProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-! ## Generic checked-program composition -/

private theorem rowsPermutationEquivalentList_append
    {leftSource rightSource leftExpected rightExpected : List Row}
    (left : RowsPermutationEquivalentList leftSource leftExpected)
    (right : RowsPermutationEquivalentList rightSource rightExpected) :
    RowsPermutationEquivalentList (leftSource ++ rightSource)
      (leftExpected ++ rightExpected) := by
  induction leftSource generalizing leftExpected with
  | nil =>
      cases leftExpected with
      | nil => simpa using right
      | cons _ _ => simp [RowsPermutationEquivalentList] at left
  | cons source sources inductionHypothesis =>
      cases leftExpected with
      | nil => simp [RowsPermutationEquivalentList] at left
      | cons expected expecteds =>
          change RowsPermutationEquivalent source expected ∧
            RowsPermutationEquivalentList sources expecteds at left
          change RowsPermutationEquivalent source expected ∧
            RowsPermutationEquivalentList
              (sources ++ rightSource) (expecteds ++ rightExpected)
          exact ⟨left.1, inductionHypothesis left.2⟩

private theorem rowsPermutationEquivalentList_ofFn_flatten :
    ∀ count (sources expected : Fin count → List Row),
      (∀ index, RowsPermutationEquivalentList
        (sources index) (expected index)) →
      RowsPermutationEquivalentList
        (List.ofFn sources).flatten (List.ofFn expected).flatten
  | 0, _, _, _ => trivial
  | count + 1, sources, expected, related => by
      rw [List.ofFn_succ, List.ofFn_succ,
        List.flatten_cons, List.flatten_cons]
      exact rowsPermutationEquivalentList_append (related 0)
        (rowsPermutationEquivalentList_ofFn_flatten count
          (fun index => sources index.succ)
          (fun index => expected index.succ)
          (fun index => related index.succ))

private theorem rowsPermutationEquivalentList_length
    {left right : List Row}
    (related : RowsPermutationEquivalentList left right) :
    left.length = right.length := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons _ _ => simp [RowsPermutationEquivalentList] at related
  | cons head tail inductionHypothesis =>
      cases right with
      | nil => simp [RowsPermutationEquivalentList] at related
      | cons rightHead rightTail =>
          change RowsPermutationEquivalent head rightHead ∧
            RowsPermutationEquivalentList tail rightTail at related
          simp [inductionHypothesis related.2]

private theorem definitions_append
    (left right : List Instruction) :
    CheckedProgram.definitions (left ++ right) =
      CheckedProgram.definitions left ++ CheckedProgram.definitions right := by
  induction left with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases head <;>
        simp [CheckedProgram.definitions, inductionHypothesis]

private theorem checks_append
    (left right : List Instruction) :
    CheckedProgram.checks (left ++ right) =
      CheckedProgram.checks left ++ CheckedProgram.checks right := by
  induction left with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases head <;>
        simp [CheckedProgram.checks, inductionHypothesis]

private theorem definitions_defines (values : List Definition) :
    CheckedProgram.definitions (values.map .define) = values := by
  simp [CheckedProgram.definitions, Function.comp_def]

private theorem checks_defines (values : List Definition) :
    CheckedProgram.checks (values.map .define) = [] := by
  simp [CheckedProgram.checks, Function.comp_def]

private theorem definitions_checks (values : List Row) :
    CheckedProgram.definitions (values.map .check) = [] := by
  simp [CheckedProgram.definitions, Function.comp_def]

private theorem checks_checks (values : List Row) :
    CheckedProgram.checks (values.map .check) = values := by
  simp [CheckedProgram.checks, Function.comp_def]

private theorem definitions_flatten (stages : List (List Instruction)) :
    CheckedProgram.definitions stages.flatten =
      (stages.map CheckedProgram.definitions).flatten := by
  induction stages with
  | nil => rfl
  | cons stage stages inductionHypothesis =>
      simp only [List.flatten_cons, List.map_cons]
      rw [definitions_append, inductionHypothesis]

private theorem checks_flatten (stages : List (List Instruction)) :
    CheckedProgram.checks stages.flatten =
      (stages.map CheckedProgram.checks).flatten := by
  induction stages with
  | nil => rfl
  | cons stage stages inductionHypothesis =>
      simp only [List.flatten_cons, List.map_cons]
      rw [checks_append, inductionHypothesis]

private theorem definitionsCanonical_flatten
    {stages : List (List Instruction)}
    (canonical : ∀ stage ∈ stages,
      ∀ definition ∈ CheckedProgram.definitions stage,
        definition.Canonical) :
    ∀ definition ∈ CheckedProgram.definitions stages.flatten,
      definition.Canonical := by
  rw [definitions_flatten]
  intro definition member
  rcases List.mem_flatten.mp member with
    ⟨stageDefinitions, stageDefinitionsMember, definitionMember⟩
  rcases List.mem_map.mp stageDefinitionsMember with
    ⟨stage, stageMember, rfl⟩
  exact canonical stage stageMember definition definitionMember

private theorem instruction_partition_count
    (values : List Instruction) :
    values.length =
      (CheckedProgram.definitions values).length +
        (CheckedProgram.checks values).length := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases head <;>
        simp [CheckedProgram.definitions, CheckedProgram.checks,
          inductionHypothesis, Nat.add_assoc, Nat.add_comm,
          Nat.add_left_comm]

private theorem define_mem_of_definition_mem
    {definition : Definition} {values : List Instruction}
    (member : definition ∈ CheckedProgram.definitions values) :
    Instruction.define definition ∈ values := by
  rcases List.mem_filterMap.mp member with
    ⟨instruction, instructionMember, mapped⟩
  cases instruction with
  | define current =>
      simp only at mapped
      cases mapped
      exact instructionMember
  | check row => simp at mapped

private theorem definition_mem_of_define_mem
    {definition : Definition} {values : List Instruction}
    (member : Instruction.define definition ∈ values) :
    definition ∈ CheckedProgram.definitions values := by
  apply List.mem_filterMap.mpr
  exact ⟨.define definition, member, rfl⟩

private theorem sum_ofFn_constant (count value : Nat) :
    (List.ofFn fun _ : Fin count => value).sum = count * value := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ, List.sum_cons, inductionHypothesis, Nat.succ_mul]
      omega

private theorem flatten_ofFn_length
    {Alpha : Type} {count width : Nat}
    (blocks : Fin count → List Alpha)
    (blockLength : ∀ index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten, List.map_ofFn]
  have lengths :
      List.ofFn (List.length ∘ blocks) =
        List.ofFn (fun _ : Fin count => width) := by
    apply congrArg List.ofFn
    funext index
    exact blockLength index
  rw [lengths, sum_ofFn_constant]

private theorem all_ofFn
    {Alpha : Type} (Property : Alpha → Prop) :
    ∀ count (values : Fin count → Alpha),
      (∀ index, Property (values index)) →
      ∀ value ∈ List.ofFn values, Property value
  | 0, _, _ => by simp
  | count + 1, values, holds => by
      intro value member
      simp only [List.ofFn_succ, List.mem_cons] at member
      rcases member with rfl | member
      · exact holds 0
      · exact all_ofFn Property count
          (fun index => values index.succ)
          (fun index => holds index.succ) value member

private theorem flatMap_congr_left
    {Alpha Beta : Type} (values : List Alpha)
    {left right : Alpha → List Beta}
    (equal : ∀ value ∈ values, left value = right value) :
    values.flatMap left = values.flatMap right := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      calc
        (head :: tail).flatMap left =
            left head ++ tail.flatMap left := rfl
        _ = right head ++ tail.flatMap left := by
          rw [equal head (by simp)]
        _ = right head ++ tail.flatMap right := by
          exact congrArg (fun suffix => right head ++ suffix)
            (inductionHypothesis fun value member =>
              equal value (by simp [member]))
        _ = (head :: tail).flatMap right := rfl

/-! ## Definition-preserving production relabeling -/

def relabelRhs (columnMap : List Nat) : Rhs → Rhs
  | .linear terms => .linear (Relabel.terms columnMap terms)
  | .product left right =>
      .product (Relabel.terms columnMap left)
        (Relabel.terms columnMap right)

def relabelDefinition (columnMap : List Nat)
    (definition : Definition) : Definition where
  output := Relabel.column columnMap definition.output
  rhs := relabelRhs columnMap definition.rhs

def relabelInstruction (columnMap : List Nat) : Instruction → Instruction
  | .define definition => .define (relabelDefinition columnMap definition)
  | .check row => .check (Relabel.row columnMap row)

theorem relabelDefinition_builderRow
    {columnMap : List Nat}
    (mapsOne : Relabel.column columnMap 0 = 0)
    (definition : Definition) :
    (relabelDefinition columnMap definition).builderRow =
      Relabel.row columnMap definition.builderRow := by
  cases definition with
  | mk output rhs =>
      cases rhs with
      | linear terms =>
          simp [relabelDefinition, relabelRhs, Definition.builderRow,
            builderLinearRow, Relabel.row, Relabel.terms,
            Program.negateTerms, List.map_map, Function.comp_def, mapsOne]
      | product left right =>
          rfl

theorem relabelDefinition_canonical
    {columnMap : List Nat} {definition : Definition}
    (canonical : definition.Canonical) :
    (relabelDefinition columnMap definition).Canonical := by
  cases definition with
  | mk output rhs =>
      cases rhs with
      | linear terms =>
          change CanonicalTerms (Relabel.terms columnMap terms)
          unfold Relabel.terms
          intro term member
          rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
          exact canonical source sourceMember
      | product left right =>
          trivial

private theorem relabelInstruction_row
    {columnMap : List Nat}
    (mapsOne : Relabel.column columnMap 0 = 0)
    (instruction : Instruction) :
    (relabelInstruction columnMap instruction).row =
      Relabel.row columnMap instruction.row := by
  cases instruction with
  | define definition =>
      exact relabelDefinition_builderRow mapsOne definition
  | check row => rfl

theorem rows_relabelInstructions
    {columnMap : List Nat}
    (mapsOne : Relabel.column columnMap 0 = 0)
    (source : List Instruction) :
    CheckedProgram.rows (source.map (relabelInstruction columnMap)) =
      (CheckedProgram.rows source).map (Relabel.row columnMap) := by
  unfold CheckedProgram.rows
  rw [List.map_map, List.map_map]
  apply List.map_congr_left
  intro instruction member
  exact relabelInstruction_row mapsOne instruction

theorem definitions_relabelInstructions
    (columnMap : List Nat) (source : List Instruction) :
    CheckedProgram.definitions
        (source.map (relabelInstruction columnMap)) =
      (CheckedProgram.definitions source).map
        (relabelDefinition columnMap) := by
  induction source with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases head with
      | define definition =>
          change relabelDefinition columnMap definition ::
              CheckedProgram.definitions
                (tail.map (relabelInstruction columnMap)) =
            relabelDefinition columnMap definition ::
              (CheckedProgram.definitions tail).map
                (relabelDefinition columnMap)
          exact congrArg (List.cons (relabelDefinition columnMap definition))
            inductionHypothesis
      | check row =>
          change CheckedProgram.definitions
              (tail.map (relabelInstruction columnMap)) =
            (CheckedProgram.definitions tail).map
              (relabelDefinition columnMap)
          exact inductionHypothesis

theorem checks_relabelInstructions
    (columnMap : List Nat) (source : List Instruction) :
    CheckedProgram.checks
        (source.map (relabelInstruction columnMap)) =
      (CheckedProgram.checks source).map (Relabel.row columnMap) := by
  induction source with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases head with
      | define definition =>
          change CheckedProgram.checks
              (tail.map (relabelInstruction columnMap)) =
            (CheckedProgram.checks tail).map (Relabel.row columnMap)
          exact inductionHypothesis
      | check row =>
          change Relabel.row columnMap row ::
              CheckedProgram.checks
                (tail.map (relabelInstruction columnMap)) =
            Relabel.row columnMap row ::
              (CheckedProgram.checks tail).map (Relabel.row columnMap)
          exact congrArg (List.cons (Relabel.row columnMap row))
            inductionHypothesis

/-! ## Padding definitions -/

def paddingDefinition (column : Nat) : Definition :=
  ⟨column, .linear []⟩

def paddingDefinitionsForOutput (output : Nat) : List Definition :=
  match Metadata.boundary.outputYZcolPaddingRows[output]?,
      Metadata.boundary.outputYZcolColumns[output]? with
  | some _, some columns =>
      (List.range paddingLaneCount).flatMap fun paddingOffset =>
        match columns[activeLaneCount + paddingOffset]? with
        | none => []
        | some pair =>
            [paddingDefinition pair.c0, paddingDefinition pair.c1]
  | _, _ => []

def paddingDefinitions : List Definition :=
  (List.range outputCount).flatMap paddingDefinitionsForOutput

def paddingInstructions : List Instruction :=
  paddingDefinitions.map .define

def paddingSourceRows : List RawSourceRow :=
  PaddingArtifact.sourceShard0 ++ PaddingArtifact.sourceShard1 ++
    PaddingArtifact.sourceShard2

private theorem expectedOutputRows_eq_paddingRows (output : Nat) :
    PaddingArtifact.rawRows (PaddingArtifact.expectedOutputRows output) =
      (paddingDefinitionsForOutput output).map Definition.builderRow := by
  cases rangeLookup :
      Metadata.boundary.outputYZcolPaddingRows[output]? with
  | none =>
      simp [PaddingArtifact.expectedOutputRows,
        paddingDefinitionsForOutput, PaddingArtifact.rawRows, rangeLookup]
  | some rowRange =>
      cases columnsLookup :
          Metadata.boundary.outputYZcolColumns[output]? with
      | none =>
          simp [PaddingArtifact.expectedOutputRows,
            paddingDefinitionsForOutput, PaddingArtifact.rawRows,
            rangeLookup, columnsLookup]
      | some columns =>
          simp only [PaddingArtifact.expectedOutputRows,
            paddingDefinitionsForOutput, rangeLookup, columnsLookup,
            PaddingArtifact.rawRows, List.map_flatMap, List.map_map]
          apply flatMap_congr_left
          intro paddingOffset offsetMember
          cases pairLookup : columns[activeLaneCount + paddingOffset]? with
          | none => simp [pairLookup]
          | some pair =>
              simp [pairLookup, PaddingArtifact.rawRow,
                PaddingArtifact.rawTerms, PaddingArtifact.expectedRow,
                paddingDefinition, Definition.builderRow,
                builderLinearRow, Program.negateTerms,
                Instruction.row, Function.comp_def]

theorem paddingRows_exact :
    RowsPermutationEquivalentList
      (PaddingArtifact.rawRows paddingSourceRows)
      (CheckedProgram.rows paddingInstructions) := by
  have sourceEquality :
      paddingSourceRows =
        (List.range outputCount).flatMap
          PaddingArtifact.expectedOutputRows := by
    rw [paddingSourceRows, PaddingArtifact.sourceShard0_exact,
      PaddingArtifact.sourceShard1_exact,
      PaddingArtifact.sourceShard2_exact]
    rw [← PaddingArtifact.outputShard_coverage]
    simp [PaddingArtifact.expectedShard0, PaddingArtifact.expectedShard1,
      PaddingArtifact.expectedShard2, List.flatMap_append,
      List.append_assoc]
  have rowEquality :
      PaddingArtifact.rawRows paddingSourceRows =
        CheckedProgram.rows paddingInstructions := by
    rw [sourceEquality]
    calc
      PaddingArtifact.rawRows
          ((List.range outputCount).flatMap
            PaddingArtifact.expectedOutputRows) =
          (List.range outputCount).flatMap
            (fun output => PaddingArtifact.rawRows
              (PaddingArtifact.expectedOutputRows output)) := by
            simp [PaddingArtifact.rawRows, List.map_flatMap]
      _ = (List.range outputCount).flatMap
          (fun output =>
            (paddingDefinitionsForOutput output).map
              Definition.builderRow) := by
            apply flatMap_congr_left
            intro output outputMember
            exact expectedOutputRows_eq_paddingRows output
      _ = CheckedProgram.rows paddingInstructions := by
            simp [paddingInstructions, paddingDefinitions,
              CheckedProgram.rows, Instruction.row, List.map_flatMap,
              List.map_map, Function.comp_def]
  rw [rowEquality]
  induction CheckedProgram.rows paddingInstructions with
  | nil => trivial
  | cons head tail inductionHypothesis =>
      exact ⟨⟨List.Perm.refl _, List.Perm.refl _, List.Perm.refl _⟩,
        inductionHypothesis⟩

private theorem paddingDefinition_canonical (column : Nat) :
    (paddingDefinition column).Canonical := by
  simp [paddingDefinition, Definition.Canonical, CanonicalTerms]

private theorem paddingDefinitionsForOutput_canonical (output : Nat) :
    ∀ definition ∈ paddingDefinitionsForOutput output,
      definition.Canonical := by
  intro definition member
  generalize rangeLookup :
      Metadata.boundary.outputYZcolPaddingRows[output]? = rangeOption at member
  generalize columnsLookup :
      Metadata.boundary.outputYZcolColumns[output]? = columnsOption at member
  cases rangeOption with
  | none => simp [paddingDefinitionsForOutput, rangeLookup] at member
  | some rowRange =>
      cases columnsOption with
      | none =>
          simp [paddingDefinitionsForOutput, rangeLookup, columnsLookup] at member
      | some columns =>
          simp only [paddingDefinitionsForOutput, rangeLookup,
            columnsLookup] at member
          rcases List.mem_flatMap.mp member with
            ⟨paddingOffset, offsetMember, definitionMember⟩
          cases pairLookup : columns[activeLaneCount + paddingOffset]? with
          | none => simp [pairLookup] at definitionMember
          | some pair =>
              simp [pairLookup] at definitionMember
              rcases definitionMember with rfl | rfl
              · exact paddingDefinition_canonical pair.c0
              · exact paddingDefinition_canonical pair.c1

theorem paddingDefinitions_canonical :
    ∀ definition ∈ paddingDefinitions, definition.Canonical := by
  intro definition member
  rcases List.mem_flatMap.mp member with
    ⟨output, outputMember, definitionMember⟩
  exact paddingDefinitionsForOutput_canonical output definition
    definitionMember

/-! ## Generated production rounds -/

def roundSourceRowsAt : Nat → List RawSourceRow
  | 0 => RoundArtifact.round0Rows
  | 1 => RoundArtifact.round1Rows
  | 2 => RoundArtifact.round2Rows
  | 3 => RoundArtifact.round3Rows
  | 4 => RoundArtifact.round4Rows
  | 5 => RoundArtifact.round5Rows
  | 6 => RoundArtifact.round6Rows
  | 7 => RoundArtifact.round7Rows
  | 8 => RoundArtifact.round8Rows
  | 9 => RoundArtifact.round9Rows
  | 10 => RoundArtifact.round10Rows
  | 11 => RoundArtifact.round11Rows
  | 12 => RoundArtifact.round12Rows
  | 13 => RoundArtifact.round13Rows
  | 14 => RoundArtifact.round14Rows
  | 15 => RoundArtifact.round15Rows
  | 16 => RoundArtifact.round16Rows
  | 17 => RoundArtifact.round17Rows
  | 18 => RoundArtifact.round18Rows
  | 19 => RoundArtifact.round19Rows
  | 20 => RoundArtifact.round20Rows
  | 21 => RoundArtifact.round21Rows
  | 22 => RoundArtifact.round22Rows
  | 23 => RoundArtifact.round23Rows
  | 24 => RoundArtifact.round24Rows
  | _ => []

theorem roundCertificateAt (index : Nat) (bound : index < sumcheckRoundCount) :
    RoundArtifact.Certificate index (roundSourceRowsAt index) := by
  have bound' : index < 25 := by
    simpa [sumcheckRoundCount, blockBitCount, laneBitCount] using bound
  by_cases h0 : index = 0
  · simpa only [h0, roundSourceRowsAt] using RoundArtifact.round0
  by_cases h1 : index = 1
  · simpa only [h1, roundSourceRowsAt] using RoundArtifact.round1
  by_cases h2 : index = 2
  · simpa only [h2, roundSourceRowsAt] using RoundArtifact.round2
  by_cases h3 : index = 3
  · simpa only [h3, roundSourceRowsAt] using RoundArtifact.round3
  by_cases h4 : index = 4
  · simpa only [h4, roundSourceRowsAt] using RoundArtifact.round4
  by_cases h5 : index = 5
  · simpa only [h5, roundSourceRowsAt] using RoundArtifact.round5
  by_cases h6 : index = 6
  · simpa only [h6, roundSourceRowsAt] using RoundArtifact.round6
  by_cases h7 : index = 7
  · simpa only [h7, roundSourceRowsAt] using RoundArtifact.round7
  by_cases h8 : index = 8
  · simpa only [h8, roundSourceRowsAt] using RoundArtifact.round8
  by_cases h9 : index = 9
  · simpa only [h9, roundSourceRowsAt] using RoundArtifact.round9
  by_cases h10 : index = 10
  · simpa only [h10, roundSourceRowsAt] using RoundArtifact.round10
  by_cases h11 : index = 11
  · simpa only [h11, roundSourceRowsAt] using RoundArtifact.round11
  by_cases h12 : index = 12
  · simpa only [h12, roundSourceRowsAt] using RoundArtifact.round12
  by_cases h13 : index = 13
  · simpa only [h13, roundSourceRowsAt] using RoundArtifact.round13
  by_cases h14 : index = 14
  · simpa only [h14, roundSourceRowsAt] using RoundArtifact.round14
  by_cases h15 : index = 15
  · simpa only [h15, roundSourceRowsAt] using RoundArtifact.round15
  by_cases h16 : index = 16
  · simpa only [h16, roundSourceRowsAt] using RoundArtifact.round16
  by_cases h17 : index = 17
  · simpa only [h17, roundSourceRowsAt] using RoundArtifact.round17
  by_cases h18 : index = 18
  · simpa only [h18, roundSourceRowsAt] using RoundArtifact.round18
  by_cases h19 : index = 19
  · simpa only [h19, roundSourceRowsAt] using RoundArtifact.round19
  by_cases h20 : index = 20
  · simpa only [h20, roundSourceRowsAt] using RoundArtifact.round20
  by_cases h21 : index = 21
  · simpa only [h21, roundSourceRowsAt] using RoundArtifact.round21
  by_cases h22 : index = 22
  · simpa only [h22, roundSourceRowsAt] using RoundArtifact.round22
  by_cases h23 : index = 23
  · simpa only [h23, roundSourceRowsAt] using RoundArtifact.round23
  by_cases h24 : index = 24
  · simpa only [h24, roundSourceRowsAt] using RoundArtifact.round24
  omega

def roundInstructionsAt (index : Nat) : List Instruction :=
  match RoundMaps.values[index]? with
  | none => []
  | some round =>
      ProductionRound.instructions.map
        (relabelInstruction round.columnMap)

def roundSourceStages : List (List RawSourceRow) :=
  List.ofFn fun index : Fin sumcheckRoundCount =>
    roundSourceRowsAt index.val

def roundInstructionStages : List (List Instruction) :=
  List.ofFn fun index : Fin sumcheckRoundCount =>
    roundInstructionsAt index.val

def roundSourceRows : List RawSourceRow := roundSourceStages.flatten
def roundInstructions : List Instruction := roundInstructionStages.flatten

theorem roundRowsAt_exact (index : Fin sumcheckRoundCount) :
    RowsPermutationEquivalentList
      ((roundSourceRowsAt index.val).map SourceDecodeBridge.rawRow)
      (CheckedProgram.rows (roundInstructionsAt index.val)) := by
  rcases RoundArtifact.certificate_lookup
      (roundCertificateAt index.val index.isLt) with
    ⟨round, lookup, valid⟩
  have mapsOne : Relabel.column round.columnMap 0 = 0 := by
    simpa [Relabel.column] using valid.1.2.2.2.1
  rw [roundInstructionsAt, lookup,
    rows_relabelInstructions mapsOne]
  simpa [RoundArtifact.rawRows, RoundArtifact.rawRow,
    RoundArtifact.rawTerms, SourceDecodeBridge.rawRow,
    SourceDecodeBridge.rawTerms] using valid.2.2.2.2.2

theorem roundRows_exact :
    RowsPermutationEquivalentList
      (roundSourceRows.map SourceDecodeBridge.rawRow)
      (CheckedProgram.rows roundInstructions) := by
  have related := rowsPermutationEquivalentList_ofFn_flatten
    sumcheckRoundCount
    (fun index =>
      (roundSourceRowsAt index.val).map SourceDecodeBridge.rawRow)
    (fun index => CheckedProgram.rows (roundInstructionsAt index.val))
    roundRowsAt_exact
  simpa [roundSourceRows, roundSourceStages, roundInstructions,
    roundInstructionStages, List.map_flatten, CheckedProgram.rows,
    List.map_ofFn, Function.comp_def] using related

theorem roundInstructionsAt_length (index : Fin sumcheckRoundCount) :
    (roundInstructionsAt index.val).length = 30 := by
  rcases RoundArtifact.certificate_lookup
      (roundCertificateAt index.val index.isLt) with
    ⟨round, lookup, valid⟩
  rw [roundInstructionsAt, lookup, List.length_map]
  simpa [ProductionRound.rows, CheckedProgram.rows] using
    ProductionRound.row_count

theorem roundInstructionsAt_checkCount (index : Fin sumcheckRoundCount) :
    (CheckedProgram.checks (roundInstructionsAt index.val)).length = 2 := by
  rcases RoundArtifact.certificate_lookup
      (roundCertificateAt index.val index.isLt) with
    ⟨round, lookup, valid⟩
  rw [roundInstructionsAt, lookup,
    checks_relabelInstructions]
  simp [ProductionRound.instructions, CheckedProgram.checks]

private theorem roundInstructionsAt_canonical
    (index : Fin sumcheckRoundCount) :
    ∀ definition ∈
        CheckedProgram.definitions (roundInstructionsAt index.val),
      definition.Canonical := by
  rcases RoundArtifact.certificate_lookup
      (roundCertificateAt index.val index.isLt) with
    ⟨round, lookup, valid⟩
  rw [roundInstructionsAt, lookup,
    definitions_relabelInstructions]
  intro definition member
  rcases List.mem_map.mp member with
    ⟨localDefinition, localMember, rfl⟩
  exact relabelDefinition_canonical
    (ProductionRound.definitions_canonical localDefinition localMember)

theorem roundInstructions_canonical :
    ∀ definition ∈ CheckedProgram.definitions roundInstructions,
      definition.Canonical := by
  unfold roundInstructions
  apply definitionsCanonical_flatten
  unfold roundInstructionStages
  exact all_ofFn
    (fun stage => ∀ definition ∈ CheckedProgram.definitions stage,
      definition.Canonical)
    sumcheckRoundCount
    (fun index : Fin sumcheckRoundCount =>
      roundInstructionsAt index.val)
    roundInstructionsAt_canonical

/-! These seven order certificates compare at most 128 source records on
each side (256 proof-free records total). They only join the already checked
30-row round leaves across generated chunk boundaries; coefficient meaning
continues to come from `RoundArtifactValid`. -/

def roundChunk5Rows : List RawSourceRow :=
  RoundArtifact.round0Rows ++ RoundArtifact.round1Rows ++
    RoundArtifact.round2Rows ++ RoundArtifact.round3Rows.take 2

def roundChunk6Rows : List RawSourceRow :=
  RoundArtifact.round3Rows.drop 2 ++ RoundArtifact.round4Rows ++
    RoundArtifact.round5Rows ++ RoundArtifact.round6Rows ++
    RoundArtifact.round7Rows.take 10

def roundChunk7Rows : List RawSourceRow :=
  RoundArtifact.round7Rows.drop 10 ++ RoundArtifact.round8Rows ++
    RoundArtifact.round9Rows ++ RoundArtifact.round10Rows ++
    RoundArtifact.round11Rows.take 18

def roundChunk8Rows : List RawSourceRow :=
  RoundArtifact.round11Rows.drop 18 ++ RoundArtifact.round12Rows ++
    RoundArtifact.round13Rows ++ RoundArtifact.round14Rows ++
    RoundArtifact.round15Rows.take 26

def roundChunk9Rows : List RawSourceRow :=
  RoundArtifact.round15Rows.drop 26 ++ RoundArtifact.round16Rows ++
    RoundArtifact.round17Rows ++ RoundArtifact.round18Rows ++
    RoundArtifact.round19Rows ++ RoundArtifact.round20Rows.take 4

def roundChunk10Rows : List RawSourceRow :=
  RoundArtifact.round20Rows.drop 4 ++ RoundArtifact.round21Rows ++
    RoundArtifact.round22Rows ++ RoundArtifact.round23Rows ++
    RoundArtifact.round24Rows.take 12

def roundChunk11Rows : List RawSourceRow :=
  RoundArtifact.round24Rows.drop 12

set_option maxRecDepth 100000 in
theorem roundChunk5_exact :
    roundChunk5Rows = SourceRows.Chunk5.values.drop 36 := by native_decide
set_option maxRecDepth 100000 in
theorem roundChunk6_exact :
    roundChunk6Rows = SourceRows.Chunk6.values := by native_decide
set_option maxRecDepth 100000 in
theorem roundChunk7_exact :
    roundChunk7Rows = SourceRows.Chunk7.values := by native_decide
set_option maxRecDepth 100000 in
theorem roundChunk8_exact :
    roundChunk8Rows = SourceRows.Chunk8.values := by native_decide
set_option maxRecDepth 100000 in
theorem roundChunk9_exact :
    roundChunk9Rows = SourceRows.Chunk9.values := by native_decide
set_option maxRecDepth 100000 in
theorem roundChunk10_exact :
    roundChunk10Rows = SourceRows.Chunk10.values := by native_decide
set_option maxRecDepth 100000 in
theorem roundChunk11_exact :
    roundChunk11Rows = SourceRows.Chunk11.values.take 18 := by native_decide

def roundSourceRowsByChunk : List RawSourceRow :=
  roundChunk5Rows ++ roundChunk6Rows ++ roundChunk7Rows ++
    roundChunk8Rows ++ roundChunk9Rows ++ roundChunk10Rows ++
    roundChunk11Rows

set_option maxRecDepth 100000 in
private theorem roundSourceRows_explicit :
    roundSourceRows =
      RoundArtifact.round0Rows ++ RoundArtifact.round1Rows ++
      RoundArtifact.round2Rows ++ RoundArtifact.round3Rows ++
      RoundArtifact.round4Rows ++ RoundArtifact.round5Rows ++
      RoundArtifact.round6Rows ++ RoundArtifact.round7Rows ++
      RoundArtifact.round8Rows ++ RoundArtifact.round9Rows ++
      RoundArtifact.round10Rows ++ RoundArtifact.round11Rows ++
      RoundArtifact.round12Rows ++ RoundArtifact.round13Rows ++
      RoundArtifact.round14Rows ++ RoundArtifact.round15Rows ++
      RoundArtifact.round16Rows ++ RoundArtifact.round17Rows ++
      RoundArtifact.round18Rows ++ RoundArtifact.round19Rows ++
      RoundArtifact.round20Rows ++ RoundArtifact.round21Rows ++
      RoundArtifact.round22Rows ++ RoundArtifact.round23Rows ++
      RoundArtifact.round24Rows := by
  rfl

private theorem regroupSixSplits
    {Alpha : Type}
    (a x b y c z d u e v f w : List Alpha) :
    (a ++ x.take 2) ++
      (x.drop 2 ++ b ++ y.take 10) ++
      (y.drop 10 ++ c ++ z.take 18) ++
      (z.drop 18 ++ d ++ u.take 26) ++
      (u.drop 26 ++ e ++ v.take 4) ++
      (v.drop 4 ++ f ++ w.take 12) ++ w.drop 12 =
    a ++ x ++ b ++ y ++ c ++ z ++ d ++ u ++ e ++ v ++ f ++ w := by
  simp only [List.append_assoc]
  rw [← List.append_assoc (List.take 2 x) (List.drop 2 x),
    List.take_append_drop 2 x,
    ← List.append_assoc (List.take 10 y) (List.drop 10 y),
    List.take_append_drop 10 y,
    ← List.append_assoc (List.take 18 z) (List.drop 18 z),
    List.take_append_drop 18 z,
    ← List.append_assoc (List.take 26 u) (List.drop 26 u),
    List.take_append_drop 26 u,
    ← List.append_assoc (List.take 4 v) (List.drop 4 v),
    List.take_append_drop 4 v,
    List.take_append_drop 12 w]

private theorem roundSourceRows_eq_byChunk :
    roundSourceRows = roundSourceRowsByChunk := by
  calc
    roundSourceRows =
        RoundArtifact.round0Rows ++ RoundArtifact.round1Rows ++
        RoundArtifact.round2Rows ++ RoundArtifact.round3Rows ++
        RoundArtifact.round4Rows ++ RoundArtifact.round5Rows ++
        RoundArtifact.round6Rows ++ RoundArtifact.round7Rows ++
        RoundArtifact.round8Rows ++ RoundArtifact.round9Rows ++
        RoundArtifact.round10Rows ++ RoundArtifact.round11Rows ++
        RoundArtifact.round12Rows ++ RoundArtifact.round13Rows ++
        RoundArtifact.round14Rows ++ RoundArtifact.round15Rows ++
        RoundArtifact.round16Rows ++ RoundArtifact.round17Rows ++
        RoundArtifact.round18Rows ++ RoundArtifact.round19Rows ++
        RoundArtifact.round20Rows ++ RoundArtifact.round21Rows ++
        RoundArtifact.round22Rows ++ RoundArtifact.round23Rows ++
        RoundArtifact.round24Rows := roundSourceRows_explicit
    _ = roundChunk5Rows ++ roundChunk6Rows ++ roundChunk7Rows ++
        roundChunk8Rows ++ roundChunk9Rows ++ roundChunk10Rows ++
        roundChunk11Rows := by
      simpa only [roundChunk5Rows, roundChunk6Rows, roundChunk7Rows,
        roundChunk8Rows, roundChunk9Rows, roundChunk10Rows,
        roundChunk11Rows, List.append_assoc] using
        (regroupSixSplits
          (RoundArtifact.round0Rows ++ RoundArtifact.round1Rows ++
            RoundArtifact.round2Rows)
          RoundArtifact.round3Rows
          (RoundArtifact.round4Rows ++ RoundArtifact.round5Rows ++
            RoundArtifact.round6Rows)
          RoundArtifact.round7Rows
          (RoundArtifact.round8Rows ++ RoundArtifact.round9Rows ++
            RoundArtifact.round10Rows)
          RoundArtifact.round11Rows
          (RoundArtifact.round12Rows ++ RoundArtifact.round13Rows ++
            RoundArtifact.round14Rows)
          RoundArtifact.round15Rows
          (RoundArtifact.round16Rows ++ RoundArtifact.round17Rows ++
            RoundArtifact.round18Rows ++ RoundArtifact.round19Rows)
          RoundArtifact.round20Rows
          (RoundArtifact.round21Rows ++ RoundArtifact.round22Rows ++
            RoundArtifact.round23Rows)
          RoundArtifact.round24Rows).symm
    _ = roundSourceRowsByChunk := rfl

theorem roundSourceRows_chunkCoverage :
    roundSourceRows =
      SourceRows.Chunk5.values.drop 36 ++
      SourceRows.Chunk6.values ++ SourceRows.Chunk7.values ++
      SourceRows.Chunk8.values ++ SourceRows.Chunk9.values ++
      SourceRows.Chunk10.values ++ SourceRows.Chunk11.values.take 18 := by
  rw [roundSourceRows_eq_byChunk]
  simp only [roundSourceRowsByChunk, roundChunk5_exact, roundChunk6_exact,
    roundChunk7_exact, roundChunk8_exact, roundChunk9_exact,
    roundChunk10_exact, roundChunk11_exact]

/-! ## Terminal checked stage and complete source coverage -/

def terminalInstructions : List Instruction :=
  TerminalProgram.definitions.map .define ++
    TerminalProgram.finalEqualityRows.map .check

theorem terminalInstructionRows :
    CheckedProgram.rows terminalInstructions = TerminalProgram.rows := by
  simp [terminalInstructions, CheckedProgram.rows, TerminalProgram.rows,
    TerminalProgram.identityRows, Instruction.row, List.map_map,
    Function.comp_def]

theorem terminalRows_exact :
    RowsPermutationEquivalentList
      (TerminalArtifact.generatedTerminalRows.map SourceDecodeBridge.rawRow)
      (CheckedProgram.rows terminalInstructions) := by
  rw [terminalInstructionRows]
  simpa [TerminalArtifact.Certificates.rawRows] using
    TerminalArtifact.terminalProgramRows_exact

theorem terminalInstructions_canonical :
    ∀ definition ∈ CheckedProgram.definitions terminalInstructions,
      definition.Canonical := by
  rw [terminalInstructions, definitions_append, definitions_defines,
    definitions_checks, List.append_nil]
  exact TerminalArtifact.terminalDefinitionsCanonical

/-! This arithmetic certificate evaluates exactly 52 `Nat` shard lengths;
it does not inspect any source row or definition record. -/
private theorem terminalShardLengthSum :
    (List.ofFn fun index : Fin
      TerminalArtifact.Certificates.terminalShardCount =>
        TerminalArtifact.Certificates.shardLength index.val).sum = 6595 := by
  native_decide

set_option maxRecDepth 100000 in
theorem terminalSourceRows_length :
    TerminalArtifact.generatedTerminalRows.length = 6595 := by
  rw [TerminalArtifact.generatedTerminalRows, List.length_flatten,
    TerminalArtifact.generatedTerminalSourceShards, List.map_ofFn]
  have lengths :
      List.ofFn
          (List.length ∘ fun index : Fin
            TerminalArtifact.Certificates.terminalShardCount =>
              TerminalArtifact.Certificates.sourceShard index.val) =
        List.ofFn
          (fun index : Fin
            TerminalArtifact.Certificates.terminalShardCount =>
              TerminalArtifact.Certificates.shardLength index.val) := by
    apply congrArg List.ofFn
    funext index
    exact TerminalArtifact.generatedTerminalShardLength index
  rw [lengths, terminalShardLengthSum]

/-! The following symbolic expansion contains 52 list nodes but no row
evaluation. The final equality certificate compares 85 records on each side
(170 total), below the 256-record ceiling. -/

set_option maxRecDepth 100000 in
private theorem terminalSourceRows_chunkCoverage :
    TerminalArtifact.generatedTerminalRows =
      SourceRows.Chunk11.values.drop 18 ++
      SourceRows.Chunk12.values ++ SourceRows.Chunk13.values ++
      SourceRows.Chunk14.values ++ SourceRows.Chunk15.values ++
      SourceRows.Chunk16.values ++ SourceRows.Chunk17.values ++
      SourceRows.Chunk18.values ++ SourceRows.Chunk19.values ++
      SourceRows.Chunk20.values ++ SourceRows.Chunk21.values ++
      SourceRows.Chunk22.values ++ SourceRows.Chunk23.values ++
      SourceRows.Chunk24.values ++ SourceRows.Chunk25.values ++
      SourceRows.Chunk26.values ++ SourceRows.Chunk27.values ++
      SourceRows.Chunk28.values ++ SourceRows.Chunk29.values ++
      SourceRows.Chunk30.values ++ SourceRows.Chunk31.values ++
      SourceRows.Chunk32.values ++ SourceRows.Chunk33.values ++
      SourceRows.Chunk34.values ++ SourceRows.Chunk35.values ++
      SourceRows.Chunk36.values ++ SourceRows.Chunk37.values ++
      SourceRows.Chunk38.values ++ SourceRows.Chunk39.values ++
      SourceRows.Chunk40.values ++ SourceRows.Chunk41.values ++
      SourceRows.Chunk42.values ++ SourceRows.Chunk43.values ++
      SourceRows.Chunk44.values ++ SourceRows.Chunk45.values ++
      SourceRows.Chunk46.values ++ SourceRows.Chunk47.values ++
      SourceRows.Chunk48.values ++ SourceRows.Chunk49.values ++
      SourceRows.Chunk50.values ++ SourceRows.Chunk51.values ++
      SourceRows.Chunk52.values ++ SourceRows.Chunk53.values ++
      SourceRows.Chunk54.values ++ SourceRows.Chunk55.values ++
      SourceRows.Chunk56.values ++ SourceRows.Chunk57.values ++
      SourceRows.Chunk58.values ++ SourceRows.Chunk59.values ++
      SourceRows.Chunk60.values ++ SourceRows.Chunk61.values ++
      SourceRows.Chunk62.values.take 85 := by
  rfl

set_option maxRecDepth 100000 in
private theorem finalSourceChunk_full :
    SourceRows.Chunk62.values.take 85 = SourceRows.Chunk62.values := by
  native_decide

def stageSourceRows : List RawSourceRow :=
  paddingSourceRows ++ InitialArtifact.claimedInitialRows ++
    roundSourceRows ++ TerminalArtifact.generatedTerminalRows

def sourcePrefixThroughChunk4 : List RawSourceRow :=
  SourceRows.Chunk0.values ++ SourceRows.Chunk1.values ++
    SourceRows.Chunk2.values ++ SourceRows.Chunk3.values ++
    SourceRows.Chunk4.values

def sourceRoundMiddleChunks : List RawSourceRow :=
  SourceRows.Chunk6.values ++ SourceRows.Chunk7.values ++
    SourceRows.Chunk8.values ++ SourceRows.Chunk9.values ++
    SourceRows.Chunk10.values

def sourceTailAfterChunk11 : List RawSourceRow :=
  SourceRows.Chunk12.values ++ SourceRows.Chunk13.values ++
  SourceRows.Chunk14.values ++ SourceRows.Chunk15.values ++
  SourceRows.Chunk16.values ++ SourceRows.Chunk17.values ++
  SourceRows.Chunk18.values ++ SourceRows.Chunk19.values ++
  SourceRows.Chunk20.values ++ SourceRows.Chunk21.values ++
  SourceRows.Chunk22.values ++ SourceRows.Chunk23.values ++
  SourceRows.Chunk24.values ++ SourceRows.Chunk25.values ++
  SourceRows.Chunk26.values ++ SourceRows.Chunk27.values ++
  SourceRows.Chunk28.values ++ SourceRows.Chunk29.values ++
  SourceRows.Chunk30.values ++ SourceRows.Chunk31.values ++
  SourceRows.Chunk32.values ++ SourceRows.Chunk33.values ++
  SourceRows.Chunk34.values ++ SourceRows.Chunk35.values ++
  SourceRows.Chunk36.values ++ SourceRows.Chunk37.values ++
  SourceRows.Chunk38.values ++ SourceRows.Chunk39.values ++
  SourceRows.Chunk40.values ++ SourceRows.Chunk41.values ++
  SourceRows.Chunk42.values ++ SourceRows.Chunk43.values ++
  SourceRows.Chunk44.values ++ SourceRows.Chunk45.values ++
  SourceRows.Chunk46.values ++ SourceRows.Chunk47.values ++
  SourceRows.Chunk48.values ++ SourceRows.Chunk49.values ++
  SourceRows.Chunk50.values ++ SourceRows.Chunk51.values ++
  SourceRows.Chunk52.values ++ SourceRows.Chunk53.values ++
  SourceRows.Chunk54.values ++ SourceRows.Chunk55.values ++
  SourceRows.Chunk56.values ++ SourceRows.Chunk57.values ++
  SourceRows.Chunk58.values ++ SourceRows.Chunk59.values ++
  SourceRows.Chunk60.values ++ SourceRows.Chunk61.values ++
  SourceRows.Chunk62.values

private theorem joinOneSplit
    {Alpha : Type} (headRows values suffix : List Alpha) (count : Nat) :
    (headRows ++ values.take count) ++ (values.drop count ++ suffix) =
      headRows ++ values ++ suffix := by
  calc
    (headRows ++ values.take count) ++
        (values.drop count ++ suffix) =
      headRows ++
        ((values.take count ++ values.drop count) ++ suffix) := by
          simp only [List.append_assoc]
    _ = headRows ++ (values ++ suffix) := by
      rw [List.take_append_drop count values]
    _ = headRows ++ values ++ suffix := by
      simp only [List.append_assoc]

private theorem joinTwoSplits
    {Alpha : Type} (headRows first middle second suffix : List Alpha)
    (firstCount secondCount : Nat) :
    (headRows ++ first.take firstCount) ++
        (first.drop firstCount ++ middle ++ second.take secondCount) ++
        (second.drop secondCount ++ suffix) =
      headRows ++ first ++ middle ++ second ++ suffix := by
  calc
    (headRows ++ first.take firstCount) ++
        (first.drop firstCount ++ middle ++ second.take secondCount) ++
        (second.drop secondCount ++ suffix) =
      headRows ++
        ((first.take firstCount ++ first.drop firstCount) ++
          (middle ++
            ((second.take secondCount ++ second.drop secondCount) ++
              suffix))) := by
          simp only [List.append_assoc]
    _ = headRows ++ (first ++ (middle ++ (second ++ suffix))) := by
      rw [List.take_append_drop firstCount first,
        List.take_append_drop secondCount second]
    _ = headRows ++ first ++ middle ++ second ++ suffix := by
      simp only [List.append_assoc]

private theorem initialSourceRows_chunkCoverage :
    paddingSourceRows ++ InitialArtifact.claimedInitialRows =
      sourcePrefixThroughChunk4 ++ SourceRows.Chunk5.values.take 36 := by
  calc
    paddingSourceRows ++ InitialArtifact.claimedInitialRows =
        (SourceRows.Chunk0.values ++ SourceRows.Chunk1.values ++
          SourceRows.Chunk2.values.take 44) ++
        (SourceRows.Chunk2.values.drop 44 ++
          SourceRows.Chunk3.values ++ SourceRows.Chunk4.values ++
          SourceRows.Chunk5.values.take 36) := by
      rw [paddingSourceRows, PaddingArtifact.generatedPrefix_coverage.symm]
      rfl
    _ = sourcePrefixThroughChunk4 ++
        SourceRows.Chunk5.values.take 36 := by
      simpa only [sourcePrefixThroughChunk4, List.append_assoc] using
        joinOneSplit
          (SourceRows.Chunk0.values ++ SourceRows.Chunk1.values)
          SourceRows.Chunk2.values
          (SourceRows.Chunk3.values ++ SourceRows.Chunk4.values ++
            SourceRows.Chunk5.values.take 36)
          44

private theorem roundSourceRows_compactCoverage :
    roundSourceRows =
      SourceRows.Chunk5.values.drop 36 ++ sourceRoundMiddleChunks ++
        SourceRows.Chunk11.values.take 18 := by
  simpa only [sourceRoundMiddleChunks, List.append_assoc] using
    roundSourceRows_chunkCoverage

private theorem terminalSourceRows_compactCoverage :
    TerminalArtifact.generatedTerminalRows =
      SourceRows.Chunk11.values.drop 18 ++ sourceTailAfterChunk11 := by
  rw [terminalSourceRows_chunkCoverage, finalSourceChunk_full]
  rfl

private theorem generatedSourceRows_chunks :
    sourcePrefixThroughChunk4 ++ SourceRows.Chunk5.values ++
        sourceRoundMiddleChunks ++ SourceRows.Chunk11.values ++
        sourceTailAfterChunk11 =
      SourceRows.values := by
  simp only [sourcePrefixThroughChunk4, sourceRoundMiddleChunks,
    sourceTailAfterChunk11, SourceRows.values, List.append_assoc]

theorem stageSourceRows_coverage :
    stageSourceRows = SourceRows.values := by
  calc
    stageSourceRows =
        (paddingSourceRows ++ InitialArtifact.claimedInitialRows) ++
          roundSourceRows ++ TerminalArtifact.generatedTerminalRows := by
      simp only [stageSourceRows, List.append_assoc]
    _ = (sourcePrefixThroughChunk4 ++
          SourceRows.Chunk5.values.take 36) ++
        (SourceRows.Chunk5.values.drop 36 ++ sourceRoundMiddleChunks ++
          SourceRows.Chunk11.values.take 18) ++
        (SourceRows.Chunk11.values.drop 18 ++
          sourceTailAfterChunk11) := by
      rw [initialSourceRows_chunkCoverage,
        roundSourceRows_compactCoverage,
        terminalSourceRows_compactCoverage]
    _ = sourcePrefixThroughChunk4 ++ SourceRows.Chunk5.values ++
        sourceRoundMiddleChunks ++ SourceRows.Chunk11.values ++
        sourceTailAfterChunk11 := by
      exact joinTwoSplits sourcePrefixThroughChunk4
        SourceRows.Chunk5.values sourceRoundMiddleChunks
        SourceRows.Chunk11.values sourceTailAfterChunk11 36 18
    _ = SourceRows.values := generatedSourceRows_chunks

/-! ## Unified program -/

def initialInstructions : List Instruction :=
  InitialProgram.definitions.map .define

def instructions : List Instruction :=
  paddingInstructions ++ initialInstructions ++ roundInstructions ++
    terminalInstructions

theorem sourceRows_exact :
    SourceProgram.ExactInstructionRows instructions := by
  have combined := rowsPermutationEquivalentList_append paddingRows_exact
    (rowsPermutationEquivalentList_append InitialArtifact.initialProgramRows_exact
      (rowsPermutationEquivalentList_append roundRows_exact terminalRows_exact))
  rw [SourceProgram.ExactInstructionRows, SourceProgram.generatedRows,
    ← stageSourceRows_coverage]
  simpa [stageSourceRows, instructions, initialInstructions,
    paddingInstructions, terminalInstructions, CheckedProgram.rows,
    PaddingArtifact.rawRows, PaddingArtifact.rawRow,
    PaddingArtifact.rawTerms, InitialArtifact.rawRows,
    RoundArtifact.rawRows, RoundArtifact.rawRow, RoundArtifact.rawTerms,
    TerminalArtifact.Certificates.rawRows, List.map_append,
    List.map_map, Function.comp_def] using combined

theorem paddingInstruction_count : paddingInstructions.length = 300 := by
  have sourceLength : paddingSourceRows.length = 300 := by
    unfold paddingSourceRows
    simp only [List.length_append]
    exact PaddingArtifact.sourceShard_total_count
  have rowLengths := rowsPermutationEquivalentList_length paddingRows_exact
  simpa only [PaddingArtifact.rawRows, List.length_map,
    CheckedProgram.rows, sourceLength] using rowLengths.symm

theorem roundInstruction_count : roundInstructions.length = 750 := by
  unfold roundInstructions roundInstructionStages
  simpa [sumcheckRoundCount] using
    flatten_ofFn_length
      (fun index : Fin sumcheckRoundCount =>
        roundInstructionsAt index.val)
      roundInstructionsAt_length

theorem terminalInstruction_count : terminalInstructions.length = 6595 := by
  have rowLengths := rowsPermutationEquivalentList_length terminalRows_exact
  simpa only [List.length_map, CheckedProgram.rows,
    terminalSourceRows_length] using rowLengths.symm

theorem instruction_count : instructions.length = 8021 := by
  simp only [instructions, List.length_append, paddingInstruction_count,
    initialInstructions, List.length_map, InitialProgram.definition_count,
    roundInstruction_count, terminalInstruction_count]

private theorem roundCheck_count :
    (CheckedProgram.checks roundInstructions).length = 50 := by
  have count := flatten_ofFn_length (width := 2)
    (fun index : Fin sumcheckRoundCount =>
      CheckedProgram.checks (roundInstructionsAt index.val))
    roundInstructionsAt_checkCount
  have projection :
      CheckedProgram.checks roundInstructions =
        (List.ofFn fun index : Fin sumcheckRoundCount =>
          CheckedProgram.checks
            (roundInstructionsAt index.val)).flatten := by
    rw [roundInstructions, checks_flatten, roundInstructionStages,
      List.map_ofFn]
    rfl
  rw [projection]
  calc
    (List.ofFn fun index : Fin sumcheckRoundCount =>
      CheckedProgram.checks
        (roundInstructionsAt index.val)).flatten.length =
        sumcheckRoundCount * 2 := count
    _ = 50 := rfl

theorem check_count :
    (CheckedProgram.checks instructions).length = 52 := by
  have paddingChecks : CheckedProgram.checks paddingInstructions = [] := by
    rw [paddingInstructions, checks_defines]
  have initialChecks : CheckedProgram.checks initialInstructions = [] := by
    rw [initialInstructions, checks_defines]
  have terminalChecks :
      CheckedProgram.checks terminalInstructions =
        TerminalProgram.finalEqualityRows := by
    rw [terminalInstructions, checks_append, checks_defines, checks_checks,
      List.nil_append]
  have decomposition :
      CheckedProgram.checks instructions =
        CheckedProgram.checks paddingInstructions ++
        CheckedProgram.checks initialInstructions ++
        CheckedProgram.checks roundInstructions ++
        CheckedProgram.checks terminalInstructions := by
    simp only [instructions, checks_append]
  rw [decomposition]
  simp only [List.length_append]
  rw [paddingChecks, initialChecks, terminalChecks, roundCheck_count]
  rfl

theorem definition_count :
    (CheckedProgram.definitions instructions).length = 7969 := by
  have partition := instruction_partition_count instructions
  rw [instruction_count, check_count] at partition
  omega

theorem definitions_canonical :
    ∀ definition ∈ CheckedProgram.definitions instructions,
      definition.Canonical := by
  intro definition member
  have instructionMember := define_mem_of_definition_mem member
  have instructionSplit :
      Instruction.define definition ∈ instructions ↔
        Instruction.define definition ∈ paddingInstructions ∨
        Instruction.define definition ∈ initialInstructions ∨
        Instruction.define definition ∈ roundInstructions ∨
        Instruction.define definition ∈ terminalInstructions := by
    unfold instructions
    rw [List.mem_append, List.mem_append, List.mem_append]
    grind
  rcases instructionSplit.mp instructionMember with
      paddingMember | initialMember | roundMember |
      terminalMember
  · unfold paddingInstructions at paddingMember
    rcases List.mem_map.mp paddingMember with
      ⟨padding, paddingMember, equal⟩
    cases equal
    exact paddingDefinitions_canonical definition paddingMember
  · unfold initialInstructions at initialMember
    rcases List.mem_map.mp initialMember with
      ⟨initial, initialMember, equal⟩
    cases equal
    exact InitialProgram.definitions_canonical definition initialMember
  · exact roundInstructions_canonical definition
      (definition_mem_of_define_mem roundMember)
  · exact terminalInstructions_canonical definition
      (definition_mem_of_define_mem terminalMember)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.StageProgram
