import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedSourceArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.InputBoundary

/-!
Exact retained-check ownership and visible-reference boundary.

Owns: coefficient-level lockstep between all 52 StageProgram checks and retained
raw source rows, plus the exact visible-column reference boundary.

Does not own: selected-row satisfaction, protocol acceptance, transcript
authority, commitment binding, costs, or permission to remove rows.

Assurance tier: artifact-checked for the fixed generated production profile
once this leaf validates.
-/

/-!
Emits constraints: none; this module classifies existing retained checks.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_disposition.retained_checks` | Prove retained checks remain in lockstep with their source-row obligations. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-! ## Exact retained checks and visible references -/

private theorem checks_append (left right : List Instruction) :
    checks (left ++ right) = checks left ++ checks right := by
  induction left with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases head <;> simp [checks, inductionHypothesis]

private theorem checks_flatten (stages : List (List Instruction)) :
    checks stages.flatten = (stages.map checks).flatten := by
  induction stages with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatten_cons, List.map_cons]
      rw [checks_append, inductionHypothesis]

private theorem checks_defines (values : List Definition) :
    checks (values.map .define) = [] := by
  simp [checks, Function.comp_def]

private theorem checks_checks (values : List Row) :
    checks (values.map .check) = values := by
  simp [checks, Function.comp_def]

def scheduledRetainedChecks : List Row :=
  (List.ofFn fun index : Fin sumcheckRoundCount =>
    checks (StageProgram.roundInstructionsAt index.val)).flatten ++
  TerminalProgram.finalEqualityRows

theorem scheduledRetainedChecks_eq_stageChecks :
    scheduledRetainedChecks = checks StageProgram.instructions := by
  have roundProjection :
      checks StageProgram.roundInstructions =
        (List.ofFn fun index : Fin sumcheckRoundCount =>
          checks (StageProgram.roundInstructionsAt index.val)).flatten := by
    rw [StageProgram.roundInstructions, checks_flatten,
      StageProgram.roundInstructionStages, List.map_ofFn]
    rfl
  have paddingChecks : checks StageProgram.paddingInstructions = [] := by
    rw [StageProgram.paddingInstructions, checks_defines]
  have initialChecks : checks StageProgram.initialInstructions = [] := by
    rw [StageProgram.initialInstructions, checks_defines]
  have terminalChecks : checks StageProgram.terminalInstructions =
      TerminalProgram.finalEqualityRows := by
    rw [StageProgram.terminalInstructions, checks_append, checks_defines,
      checks_checks, List.nil_append]
  have decomposition : checks StageProgram.instructions =
      checks StageProgram.paddingInstructions ++
      checks StageProgram.initialInstructions ++
      checks StageProgram.roundInstructions ++
      checks StageProgram.terminalInstructions := by
    simp only [StageProgram.instructions, checks_append]
  rw [decomposition, paddingChecks, initialChecks, roundProjection,
    terminalChecks]
  rfl

def visibleDefinitionColumns : List Nat :=
  SourceExecution.inputColumns ++
    (physicalDefinitionOutputs ++ terminalPivotColumns)

/-- Compact logical form of the visible-column boundary. Its source-input
branch uses the exact registry-minus-definition predicate rather than
materializing `SourceExecution.inputColumns` during executable checking. -/
def VisibleColumn (column : Nat) : Prop :=
  (column ∈ Provenance.sourceColumns ∧
      column ∉ SourceExecution.definitionOutputs) ∨
    column ∈ physicalDefinitionOutputs ∨
    column ∈ terminalPivotColumns

instance (column : Nat) : Decidable (VisibleColumn column) := by
  unfold VisibleColumn
  infer_instance

private theorem visibleColumn_mem {column : Nat}
    (visible : VisibleColumn column) :
    column ∈ visibleDefinitionColumns := by
  rcases visible with sourceInput | physical | pivot
  · apply List.mem_append_left
    exact (SourceExecution.mem_inputColumns_iff column).mpr sourceInput
  · apply List.mem_append_right
    exact List.mem_append_left _ physical
  · apply List.mem_append_right
    exact List.mem_append_right _ pivot

structure RetainedCheckShape where
  coefficientEquivalent : Bool
  referencesVisible : Bool
deriving DecidableEq, Repr

def retainedCheckShape (raw : RawSourceRow) (checked : Row) :
    RetainedCheckShape :=
  { coefficientEquivalent := decide
      (RowsPermutationEquivalent (SourceDecodeBridge.rawRow raw) checked)
    referencesVisible := decide
      (∀ column ∈ rowRefs checked, VisibleColumn column) }

def checkShapes : List RawSourceRow → List Row → List RetainedCheckShape
  | raw :: raws, checked :: checks =>
      retainedCheckShape raw checked :: checkShapes raws checks
  | _, _ => []

def checkShapesValid (values : List RetainedCheckShape) : Bool :=
  values.all fun shape =>
    shape.coefficientEquivalent && shape.referencesVisible

private theorem lockstep_of_checkShapes :
    ∀ (raws : List RawSourceRow) (checked : List Row),
      raws.length = checked.length →
      checkShapesValid (checkShapes raws checked) = true →
      RowsPermutationEquivalentList
        (raws.map SourceDecodeBridge.rawRow) checked := by
  intro raws
  induction raws with
  | nil =>
      intro checked length _
      have empty : checked = [] :=
        List.eq_nil_of_length_eq_zero (by simpa using length.symm)
      subst checked
      trivial
  | cons raw raws inductionHypothesis =>
      intro checked length valid
      cases checked with
      | nil => simp at length
      | cons row rows =>
          simp only [checkShapes, checkShapesValid, List.all_cons,
            Bool.and_eq_true] at valid
          change RowsPermutationEquivalent
              (SourceDecodeBridge.rawRow raw) row ∧
            RowsPermutationEquivalentList
              (raws.map SourceDecodeBridge.rawRow) rows
          constructor
          · exact of_decide_eq_true (by
              simpa only [retainedCheckShape] using valid.1.1)
          · apply inductionHypothesis rows
            · simpa using length
            · exact valid.2

private theorem references_of_checkShapes :
    ∀ (raws : List RawSourceRow) (checked : List Row),
      raws.length = checked.length →
      checkShapesValid (checkShapes raws checked) = true →
      ∀ row ∈ checked, ∀ column ∈ rowRefs row,
        column ∈ visibleDefinitionColumns := by
  intro raws
  induction raws with
  | nil =>
      intro checked length _
      have empty : checked = [] :=
        List.eq_nil_of_length_eq_zero (by simpa using length.symm)
      subst checked
      simp
  | cons raw raws inductionHypothesis =>
      intro checked length valid
      cases checked with
      | nil => simp at length
      | cons row rows =>
          simp only [checkShapes, checkShapesValid, List.all_cons,
            Bool.and_eq_true] at valid
          intro candidate member column reference
          simp only [List.mem_cons] at member
          rcases member with rfl | member
          · have visible := of_decide_eq_true (by
                simpa only [retainedCheckShape] using valid.1.2)
            exact visibleColumn_mem (visible column reference)
          · exact inductionHypothesis rows (by simpa using length)
              valid.2 candidate member column reference

/-! ## Bounded direct shard certificates

Each of the 25 round subjects contains exactly two raw/check row pairs, 28
sparse coefficient terms, and 14 referenced-column classifications: 44
proof-free observed records total. The terminal subject contains two pairs,
12 sparse terms, and six classifications: 20 records total. Every subject is
therefore below the fixed 256-record native-evaluation ceiling.
-/

private def sparseTermCount (row : Row) : Nat :=
  row.a.length + row.b.length + row.c.length

structure RetainedCheckShardSummary where
  rawRows : Nat
  checkedRows : Nat
  rowPairs : Nat
  sparseCoefficientTerms : Nat
  referencedColumnClassifications : Nat
  totalObservedRecords : Nat
  withinRecordLimit : Bool
  checksValid : Bool
deriving DecidableEq, Repr

def retainedCheckShardSummary
    (raws : List RawSourceRow) (checked : List Row) :
    RetainedCheckShardSummary :=
  let rawRows := raws.map SourceDecodeBridge.rawRow
  let rowPairs := min rawRows.length checked.length
  let sparseCoefficientTerms :=
    (rawRows.map sparseTermCount).sum +
      (checked.map sparseTermCount).sum
  let referencedColumnClassifications :=
    (checked.map fun row => (rowRefs row).length).sum
  let totalObservedRecords := rowPairs + sparseCoefficientTerms +
    referencedColumnClassifications
  { rawRows := rawRows.length
    checkedRows := checked.length
    rowPairs
    sparseCoefficientTerms
    referencedColumnClassifications
    totalObservedRecords
    withinRecordLimit := decide (totalObservedRecords ≤ 256)
    checksValid := checkShapesValid (checkShapes raws checked) }

private def expectedRoundShardSummary : RetainedCheckShardSummary :=
  { rawRows := 2
    checkedRows := 2
    rowPairs := 2
    sparseCoefficientTerms := 28
    referencedColumnClassifications := 14
    totalObservedRecords := 44
    withinRecordLimit := true
    checksValid := true }

private def expectedTerminalShardSummary : RetainedCheckShardSummary :=
  { rawRows := 2
    checkedRows := 2
    rowPairs := 2
    sparseCoefficientTerms := 12
    referencedColumnClassifications := 6
    totalObservedRecords := 20
    withinRecordLimit := true
    checksValid := true }

private theorem round0Certificate :
    retainedCheckShardSummary (RoundArtifact.round0Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 0)) =
        expectedRoundShardSummary := by native_decide
private theorem round1Certificate :
    retainedCheckShardSummary (RoundArtifact.round1Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 1)) =
        expectedRoundShardSummary := by native_decide
private theorem round2Certificate :
    retainedCheckShardSummary (RoundArtifact.round2Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 2)) =
        expectedRoundShardSummary := by native_decide
private theorem round3Certificate :
    retainedCheckShardSummary (RoundArtifact.round3Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 3)) =
        expectedRoundShardSummary := by native_decide
private theorem round4Certificate :
    retainedCheckShardSummary (RoundArtifact.round4Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 4)) =
        expectedRoundShardSummary := by native_decide
private theorem round5Certificate :
    retainedCheckShardSummary (RoundArtifact.round5Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 5)) =
        expectedRoundShardSummary := by native_decide
private theorem round6Certificate :
    retainedCheckShardSummary (RoundArtifact.round6Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 6)) =
        expectedRoundShardSummary := by native_decide
private theorem round7Certificate :
    retainedCheckShardSummary (RoundArtifact.round7Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 7)) =
        expectedRoundShardSummary := by native_decide
private theorem round8Certificate :
    retainedCheckShardSummary (RoundArtifact.round8Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 8)) =
        expectedRoundShardSummary := by native_decide
private theorem round9Certificate :
    retainedCheckShardSummary (RoundArtifact.round9Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 9)) =
        expectedRoundShardSummary := by native_decide
private theorem round10Certificate :
    retainedCheckShardSummary (RoundArtifact.round10Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 10)) =
        expectedRoundShardSummary := by native_decide
private theorem round11Certificate :
    retainedCheckShardSummary (RoundArtifact.round11Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 11)) =
        expectedRoundShardSummary := by native_decide
private theorem round12Certificate :
    retainedCheckShardSummary (RoundArtifact.round12Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 12)) =
        expectedRoundShardSummary := by native_decide
private theorem round13Certificate :
    retainedCheckShardSummary (RoundArtifact.round13Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 13)) =
        expectedRoundShardSummary := by native_decide
private theorem round14Certificate :
    retainedCheckShardSummary (RoundArtifact.round14Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 14)) =
        expectedRoundShardSummary := by native_decide
private theorem round15Certificate :
    retainedCheckShardSummary (RoundArtifact.round15Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 15)) =
        expectedRoundShardSummary := by native_decide
private theorem round16Certificate :
    retainedCheckShardSummary (RoundArtifact.round16Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 16)) =
        expectedRoundShardSummary := by native_decide
private theorem round17Certificate :
    retainedCheckShardSummary (RoundArtifact.round17Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 17)) =
        expectedRoundShardSummary := by native_decide
private theorem round18Certificate :
    retainedCheckShardSummary (RoundArtifact.round18Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 18)) =
        expectedRoundShardSummary := by native_decide
private theorem round19Certificate :
    retainedCheckShardSummary (RoundArtifact.round19Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 19)) =
        expectedRoundShardSummary := by native_decide
private theorem round20Certificate :
    retainedCheckShardSummary (RoundArtifact.round20Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 20)) =
        expectedRoundShardSummary := by native_decide
private theorem round21Certificate :
    retainedCheckShardSummary (RoundArtifact.round21Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 21)) =
        expectedRoundShardSummary := by native_decide
private theorem round22Certificate :
    retainedCheckShardSummary (RoundArtifact.round22Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 22)) =
        expectedRoundShardSummary := by native_decide
private theorem round23Certificate :
    retainedCheckShardSummary (RoundArtifact.round23Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 23)) =
        expectedRoundShardSummary := by native_decide
private theorem round24Certificate :
    retainedCheckShardSummary (RoundArtifact.round24Rows.take 2)
      (checks (StageProgram.roundInstructionsAt 24)) =
        expectedRoundShardSummary := by native_decide
private theorem terminalCertificate :
    retainedCheckShardSummary
      RetainedSourceArtifact.terminalRetainedSourceRows
      TerminalProgram.finalEqualityRows =
        expectedTerminalShardSummary := by native_decide

private def RowsReferenceVisible (rows : List Row) : Prop :=
  ∀ row ∈ rows, ∀ column ∈ rowRefs row,
    column ∈ visibleDefinitionColumns

private def RetainedCheckShardHolds
    (raws : List RawSourceRow) (checked : List Row) : Prop :=
  RowsPermutationEquivalentList
      (raws.map SourceDecodeBridge.rawRow) checked ∧
    RowsReferenceVisible checked

private theorem shardHolds_of_summary
    {raws : List RawSourceRow} {checked : List Row}
    {summary : RetainedCheckShardSummary}
    (certificate : retainedCheckShardSummary raws checked = summary)
    (rawRows : summary.rawRows = 2)
    (checkedRows : summary.checkedRows = 2)
    (valid : summary.checksValid = true) :
    RetainedCheckShardHolds raws checked := by
  have rawLength := congrArg RetainedCheckShardSummary.rawRows certificate
  simp only [retainedCheckShardSummary] at rawLength
  simp only [List.length_map] at rawLength
  have checkedLength :=
    congrArg RetainedCheckShardSummary.checkedRows certificate
  simp only [retainedCheckShardSummary] at checkedLength
  have shapeValidity :=
    congrArg RetainedCheckShardSummary.checksValid certificate
  simp only [retainedCheckShardSummary] at shapeValidity
  have lengthsEqual : raws.length = checked.length := by
    calc
      raws.length = summary.rawRows := rawLength
      _ = 2 := rawRows
      _ = summary.checkedRows := checkedRows.symm
      _ = checked.length := checkedLength.symm
  have shapesValid : checkShapesValid (checkShapes raws checked) = true :=
    shapeValidity.trans valid
  exact ⟨lockstep_of_checkShapes raws checked lengthsEqual shapesValid,
    references_of_checkShapes raws checked lengthsEqual shapesValid⟩

private theorem round0Holds : RetainedCheckShardHolds
    (RoundArtifact.round0Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 0)) :=
  shardHolds_of_summary round0Certificate rfl rfl rfl
private theorem round1Holds : RetainedCheckShardHolds
    (RoundArtifact.round1Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 1)) :=
  shardHolds_of_summary round1Certificate rfl rfl rfl
private theorem round2Holds : RetainedCheckShardHolds
    (RoundArtifact.round2Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 2)) :=
  shardHolds_of_summary round2Certificate rfl rfl rfl
private theorem round3Holds : RetainedCheckShardHolds
    (RoundArtifact.round3Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 3)) :=
  shardHolds_of_summary round3Certificate rfl rfl rfl
private theorem round4Holds : RetainedCheckShardHolds
    (RoundArtifact.round4Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 4)) :=
  shardHolds_of_summary round4Certificate rfl rfl rfl
private theorem round5Holds : RetainedCheckShardHolds
    (RoundArtifact.round5Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 5)) :=
  shardHolds_of_summary round5Certificate rfl rfl rfl
private theorem round6Holds : RetainedCheckShardHolds
    (RoundArtifact.round6Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 6)) :=
  shardHolds_of_summary round6Certificate rfl rfl rfl
private theorem round7Holds : RetainedCheckShardHolds
    (RoundArtifact.round7Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 7)) :=
  shardHolds_of_summary round7Certificate rfl rfl rfl
private theorem round8Holds : RetainedCheckShardHolds
    (RoundArtifact.round8Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 8)) :=
  shardHolds_of_summary round8Certificate rfl rfl rfl
private theorem round9Holds : RetainedCheckShardHolds
    (RoundArtifact.round9Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 9)) :=
  shardHolds_of_summary round9Certificate rfl rfl rfl
private theorem round10Holds : RetainedCheckShardHolds
    (RoundArtifact.round10Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 10)) :=
  shardHolds_of_summary round10Certificate rfl rfl rfl
private theorem round11Holds : RetainedCheckShardHolds
    (RoundArtifact.round11Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 11)) :=
  shardHolds_of_summary round11Certificate rfl rfl rfl
private theorem round12Holds : RetainedCheckShardHolds
    (RoundArtifact.round12Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 12)) :=
  shardHolds_of_summary round12Certificate rfl rfl rfl
private theorem round13Holds : RetainedCheckShardHolds
    (RoundArtifact.round13Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 13)) :=
  shardHolds_of_summary round13Certificate rfl rfl rfl
private theorem round14Holds : RetainedCheckShardHolds
    (RoundArtifact.round14Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 14)) :=
  shardHolds_of_summary round14Certificate rfl rfl rfl
private theorem round15Holds : RetainedCheckShardHolds
    (RoundArtifact.round15Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 15)) :=
  shardHolds_of_summary round15Certificate rfl rfl rfl
private theorem round16Holds : RetainedCheckShardHolds
    (RoundArtifact.round16Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 16)) :=
  shardHolds_of_summary round16Certificate rfl rfl rfl
private theorem round17Holds : RetainedCheckShardHolds
    (RoundArtifact.round17Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 17)) :=
  shardHolds_of_summary round17Certificate rfl rfl rfl
private theorem round18Holds : RetainedCheckShardHolds
    (RoundArtifact.round18Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 18)) :=
  shardHolds_of_summary round18Certificate rfl rfl rfl
private theorem round19Holds : RetainedCheckShardHolds
    (RoundArtifact.round19Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 19)) :=
  shardHolds_of_summary round19Certificate rfl rfl rfl
private theorem round20Holds : RetainedCheckShardHolds
    (RoundArtifact.round20Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 20)) :=
  shardHolds_of_summary round20Certificate rfl rfl rfl
private theorem round21Holds : RetainedCheckShardHolds
    (RoundArtifact.round21Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 21)) :=
  shardHolds_of_summary round21Certificate rfl rfl rfl
private theorem round22Holds : RetainedCheckShardHolds
    (RoundArtifact.round22Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 22)) :=
  shardHolds_of_summary round22Certificate rfl rfl rfl
private theorem round23Holds : RetainedCheckShardHolds
    (RoundArtifact.round23Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 23)) :=
  shardHolds_of_summary round23Certificate rfl rfl rfl
private theorem round24Holds : RetainedCheckShardHolds
    (RoundArtifact.round24Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 24)) :=
  shardHolds_of_summary round24Certificate rfl rfl rfl
private theorem terminalHolds : RetainedCheckShardHolds
    RetainedSourceArtifact.terminalRetainedSourceRows
    TerminalProgram.finalEqualityRows :=
  shardHolds_of_summary terminalCertificate rfl rfl rfl

/-! ## Kernel-only shard composition -/

private theorem rowsPermutationEquivalentList_append
    {leftRaw rightRaw leftChecked rightChecked : List Row}
    (left : RowsPermutationEquivalentList leftRaw leftChecked)
    (right : RowsPermutationEquivalentList rightRaw rightChecked) :
    RowsPermutationEquivalentList (leftRaw ++ rightRaw)
      (leftChecked ++ rightChecked) := by
  induction leftRaw generalizing leftChecked with
  | nil =>
      cases leftChecked with
      | nil => simpa using right
      | cons _ _ => simp [RowsPermutationEquivalentList] at left
  | cons raw raws inductionHypothesis =>
      cases leftChecked with
      | nil => simp [RowsPermutationEquivalentList] at left
      | cons checked checks =>
          change RowsPermutationEquivalent raw checked ∧
            RowsPermutationEquivalentList raws checks at left
          change RowsPermutationEquivalent raw checked ∧
            RowsPermutationEquivalentList
              (raws ++ rightRaw) (checks ++ rightChecked)
          exact ⟨left.1, inductionHypothesis left.2⟩

private theorem retainedCheckShardHolds_append
    {leftRaw rightRaw : List RawSourceRow}
    {leftChecked rightChecked : List Row}
    (left : RetainedCheckShardHolds leftRaw leftChecked)
    (right : RetainedCheckShardHolds rightRaw rightChecked) :
    RetainedCheckShardHolds (leftRaw ++ rightRaw)
      (leftChecked ++ rightChecked) := by
  constructor
  · simpa only [List.map_append] using
      rowsPermutationEquivalentList_append left.1 right.1
  · intro row member column reference
    rcases List.mem_append.mp member with leftMember | rightMember
    · exact left.2 row leftMember column reference
    · exact right.2 row rightMember column reference

private theorem roundRetainedHolds : RetainedCheckShardHolds
    (RoundArtifact.round0Rows.take 2 ++
      RoundArtifact.round1Rows.take 2 ++
      RoundArtifact.round2Rows.take 2 ++
      RoundArtifact.round3Rows.take 2 ++
      RoundArtifact.round4Rows.take 2 ++
      RoundArtifact.round5Rows.take 2 ++
      RoundArtifact.round6Rows.take 2 ++
      RoundArtifact.round7Rows.take 2 ++
      RoundArtifact.round8Rows.take 2 ++
      RoundArtifact.round9Rows.take 2 ++
      RoundArtifact.round10Rows.take 2 ++
      RoundArtifact.round11Rows.take 2 ++
      RoundArtifact.round12Rows.take 2 ++
      RoundArtifact.round13Rows.take 2 ++
      RoundArtifact.round14Rows.take 2 ++
      RoundArtifact.round15Rows.take 2 ++
      RoundArtifact.round16Rows.take 2 ++
      RoundArtifact.round17Rows.take 2 ++
      RoundArtifact.round18Rows.take 2 ++
      RoundArtifact.round19Rows.take 2 ++
      RoundArtifact.round20Rows.take 2 ++
      RoundArtifact.round21Rows.take 2 ++
      RoundArtifact.round22Rows.take 2 ++
      RoundArtifact.round23Rows.take 2 ++
      RoundArtifact.round24Rows.take 2)
    (checks (StageProgram.roundInstructionsAt 0) ++
      checks (StageProgram.roundInstructionsAt 1) ++
      checks (StageProgram.roundInstructionsAt 2) ++
      checks (StageProgram.roundInstructionsAt 3) ++
      checks (StageProgram.roundInstructionsAt 4) ++
      checks (StageProgram.roundInstructionsAt 5) ++
      checks (StageProgram.roundInstructionsAt 6) ++
      checks (StageProgram.roundInstructionsAt 7) ++
      checks (StageProgram.roundInstructionsAt 8) ++
      checks (StageProgram.roundInstructionsAt 9) ++
      checks (StageProgram.roundInstructionsAt 10) ++
      checks (StageProgram.roundInstructionsAt 11) ++
      checks (StageProgram.roundInstructionsAt 12) ++
      checks (StageProgram.roundInstructionsAt 13) ++
      checks (StageProgram.roundInstructionsAt 14) ++
      checks (StageProgram.roundInstructionsAt 15) ++
      checks (StageProgram.roundInstructionsAt 16) ++
      checks (StageProgram.roundInstructionsAt 17) ++
      checks (StageProgram.roundInstructionsAt 18) ++
      checks (StageProgram.roundInstructionsAt 19) ++
      checks (StageProgram.roundInstructionsAt 20) ++
      checks (StageProgram.roundInstructionsAt 21) ++
      checks (StageProgram.roundInstructionsAt 22) ++
      checks (StageProgram.roundInstructionsAt 23) ++
      checks (StageProgram.roundInstructionsAt 24)) := by
  simpa only [List.append_assoc] using
    retainedCheckShardHolds_append round0Holds
    (retainedCheckShardHolds_append round1Holds
    (retainedCheckShardHolds_append round2Holds
    (retainedCheckShardHolds_append round3Holds
    (retainedCheckShardHolds_append round4Holds
    (retainedCheckShardHolds_append round5Holds
    (retainedCheckShardHolds_append round6Holds
    (retainedCheckShardHolds_append round7Holds
    (retainedCheckShardHolds_append round8Holds
    (retainedCheckShardHolds_append round9Holds
    (retainedCheckShardHolds_append round10Holds
    (retainedCheckShardHolds_append round11Holds
    (retainedCheckShardHolds_append round12Holds
    (retainedCheckShardHolds_append round13Holds
    (retainedCheckShardHolds_append round14Holds
    (retainedCheckShardHolds_append round15Holds
    (retainedCheckShardHolds_append round16Holds
    (retainedCheckShardHolds_append round17Holds
    (retainedCheckShardHolds_append round18Holds
    (retainedCheckShardHolds_append round19Holds
    (retainedCheckShardHolds_append round20Holds
    (retainedCheckShardHolds_append round21Holds
    (retainedCheckShardHolds_append round22Holds
    (retainedCheckShardHolds_append round23Holds round24Holds)))))))))))))))))))))))

private theorem retainedSourceRows_explicit :
    RetainedSourceArtifact.retainedSourceRows =
      (RoundArtifact.round0Rows.take 2 ++
        RoundArtifact.round1Rows.take 2 ++
        RoundArtifact.round2Rows.take 2 ++
        RoundArtifact.round3Rows.take 2 ++
        RoundArtifact.round4Rows.take 2 ++
        RoundArtifact.round5Rows.take 2 ++
        RoundArtifact.round6Rows.take 2 ++
        RoundArtifact.round7Rows.take 2 ++
        RoundArtifact.round8Rows.take 2 ++
        RoundArtifact.round9Rows.take 2 ++
        RoundArtifact.round10Rows.take 2 ++
        RoundArtifact.round11Rows.take 2 ++
        RoundArtifact.round12Rows.take 2 ++
        RoundArtifact.round13Rows.take 2 ++
        RoundArtifact.round14Rows.take 2 ++
        RoundArtifact.round15Rows.take 2 ++
        RoundArtifact.round16Rows.take 2 ++
        RoundArtifact.round17Rows.take 2 ++
        RoundArtifact.round18Rows.take 2 ++
        RoundArtifact.round19Rows.take 2 ++
        RoundArtifact.round20Rows.take 2 ++
        RoundArtifact.round21Rows.take 2 ++
        RoundArtifact.round22Rows.take 2 ++
        RoundArtifact.round23Rows.take 2 ++
        RoundArtifact.round24Rows.take 2) ++
      RetainedSourceArtifact.terminalRetainedSourceRows := by
  rfl

/-- One kernel step of finite schedule expansion. Keeping this structural
lemma separate prevents simplification from unfolding any row payload. -/
private theorem flatten_ofFn_succ {Alpha : Type} {count : Nat}
    (values : Fin (count + 1) → List Alpha) :
    (List.ofFn values).flatten =
      values 0 ++
        (List.ofFn fun index : Fin count => values index.succ).flatten := by
  rw [List.ofFn_succ, List.flatten_cons]

private theorem flatten_ofFn_zero {Alpha : Type}
    (values : Fin 0 → List Alpha) :
    (List.ofFn values).flatten = [] := by
  rfl

private theorem scheduledRetainedChecks_explicit :
    scheduledRetainedChecks =
      (checks (StageProgram.roundInstructionsAt 0) ++
        checks (StageProgram.roundInstructionsAt 1) ++
        checks (StageProgram.roundInstructionsAt 2) ++
        checks (StageProgram.roundInstructionsAt 3) ++
        checks (StageProgram.roundInstructionsAt 4) ++
        checks (StageProgram.roundInstructionsAt 5) ++
        checks (StageProgram.roundInstructionsAt 6) ++
        checks (StageProgram.roundInstructionsAt 7) ++
        checks (StageProgram.roundInstructionsAt 8) ++
        checks (StageProgram.roundInstructionsAt 9) ++
        checks (StageProgram.roundInstructionsAt 10) ++
        checks (StageProgram.roundInstructionsAt 11) ++
        checks (StageProgram.roundInstructionsAt 12) ++
        checks (StageProgram.roundInstructionsAt 13) ++
        checks (StageProgram.roundInstructionsAt 14) ++
        checks (StageProgram.roundInstructionsAt 15) ++
        checks (StageProgram.roundInstructionsAt 16) ++
        checks (StageProgram.roundInstructionsAt 17) ++
        checks (StageProgram.roundInstructionsAt 18) ++
        checks (StageProgram.roundInstructionsAt 19) ++
        checks (StageProgram.roundInstructionsAt 20) ++
        checks (StageProgram.roundInstructionsAt 21) ++
        checks (StageProgram.roundInstructionsAt 22) ++
        checks (StageProgram.roundInstructionsAt 23) ++
        checks (StageProgram.roundInstructionsAt 24)) ++
      TerminalProgram.finalEqualityRows := by
  change
    (List.ofFn (fun index : Fin 25 =>
      checks (StageProgram.roundInstructionsAt index.val))).flatten ++
        TerminalProgram.finalEqualityRows = _
  rw [flatten_ofFn_succ, flatten_ofFn_succ, flatten_ofFn_succ,
    flatten_ofFn_succ, flatten_ofFn_succ, flatten_ofFn_succ,
    flatten_ofFn_succ, flatten_ofFn_succ, flatten_ofFn_succ,
    flatten_ofFn_succ, flatten_ofFn_succ, flatten_ofFn_succ,
    flatten_ofFn_succ, flatten_ofFn_succ, flatten_ofFn_succ,
    flatten_ofFn_succ, flatten_ofFn_succ, flatten_ofFn_succ,
    flatten_ofFn_succ, flatten_ofFn_succ, flatten_ofFn_succ,
    flatten_ofFn_succ, flatten_ofFn_succ, flatten_ofFn_succ,
    flatten_ofFn_succ]
  rw [flatten_ofFn_zero]
  simp only [List.append_nil, Fin.val_zero, Fin.val_succ,
    Nat.zero_add, Nat.reduceAdd, List.append_assoc]

private theorem scheduledRetainedHolds : RetainedCheckShardHolds
    RetainedSourceArtifact.retainedSourceRows scheduledRetainedChecks := by
  rw [retainedSourceRows_explicit, scheduledRetainedChecks_explicit]
  exact retainedCheckShardHolds_append roundRetainedHolds terminalHolds

/-! ## Public exactness boundary -/

/-- Exact coefficient-level lockstep ownership of all 52 StageProgram checks.
This is stronger than a count equality and is suitable for satisfaction
transport through row permutation equivalence. -/
theorem retainedChecks_lockstep :
    RowsPermutationEquivalentList
      (RetainedSourceArtifact.retainedSourceRows.map
        SourceDecodeBridge.rawRow)
      (checks StageProgram.instructions) := by
  rw [← scheduledRetainedChecks_eq_stageChecks]
  exact scheduledRetainedHolds.1

/-- Every retained check reads only the source seed, physical compiler-linear
outputs, or rewrite-terminal pivot outputs. Hidden trace outputs are not
silently added to this boundary. -/
theorem retainedChecks_referencesOnly :
    ChecksReference visibleDefinitionColumns StageProgram.instructions := by
  unfold ChecksReference
  rw [← scheduledRetainedChecks_eq_stageChecks]
  exact scheduledRetainedHolds.2

theorem retainedCheck_count :
    (checks StageProgram.instructions).length = 52 :=
  StageProgram.check_count

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition
