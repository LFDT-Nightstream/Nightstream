import NightstreamFPrime.Export.Stage1.Wide.ApplicationPackage
import NightstreamFPrime.Export.Stage1.Wide.RetainedLayout
import NightstreamFPrime.Export.Stage1.Wide.SourceAssignment
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
import NightstreamFPrime.Layout.PiRlcWideSampler.Retained

/-! Executable retained-value transport for the wide physical package.
Common blocks keep their proved order. The sampler retains S-box outputs,
canonical bits and field auxiliaries, and result bits; helpers have no slot.
Phi81 quotients form a derived suffix after the physical source columns. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentTransport

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export.Codec NightstreamFPrime.Export.Package
open Layout.ProductionRelation Layout.Stage1.Wide
open PerApplicationAssignmentPlan PerApplicationAssignmentBlocks

abbrev Program := RetainedLayout.Program

structure Values where
  kind : LowNormSlot.Kind
  count : Nat
  sources : List AffineRuns.Run

def Values.encode (values : Values) : Value := .array [
  slotKindFormat.encode values.kind, .atom values.count, AffineRuns.format.encode values.sources]

def Values.ofSource {count : Nat} (kind : LowNormSlot.Kind) (source : Fin count → Nat) : Values :=
  ⟨kind, count, AffineRuns.compressIndexedTR source⟩

theorem Values.ofSource_source {count : Nat} (kind : LowNormSlot.Kind)
    (source : Fin count → Nat) (index : Fin count) :
    AffineRuns.sourceAt (Values.ofSource kind source).sources index.val = source index := by
  rw [AffineRuns.sourceAt_eq_expand_getD]
  change (AffineRuns.expand (AffineRuns.compressIndexedTR source)).getD index.val 0 = _
  rw [AffineRuns.compressIndexedTR_eq_compress_ofFn, AffineRuns.expand_compress]
  exact Lifecycle.PriorStateHash.ofFn_getD source index 0

def finalColumn (source : Nat) : Except String Nat :=
  if Layout.Stage1.Spartan.privateColumnCount ≤ source then
    .ok (SourceOrder.privateColumns + (source - Layout.Stage1.Spartan.privateColumnCount))
  else
    match Layout.Stage1.Spartan.spartanToSource source with
    | none => .error s!"missing retained source column {source}"
    | some original =>
      match SourceAssignment.source? original with
      | none => .error s!"discarded retained source column {source}"
      | some current =>
        let target := SourceOrder.column current
        if target < SourceOrder.privateColumns then .ok target
        else .error s!"retained private source maps outside its region: {source}"

def finalMap : PhysicalRelabel.Map := ⟨finalColumn, PhysicalRelabel.row⟩

def moveRun (run : AffineRuns.Run) : Except String AffineRuns.Run := do
  let first ← finalColumn run.first
  if (List.range run.count).all (fun index =>
      decide (finalColumn (run.first + run.step * index) = .ok (first + run.step * index))) then
    return { run with first := first }
  else throw "retained source run crosses a removed source interval"

def moveRuns : List AffineRuns.Run → Except String (List AffineRuns.Run)
  | [] => .ok []
  | run :: rest => return (← moveRun run) :: (← moveRuns rest)

theorem moveRun_source (run moved : AffineRuns.Run) (emitted : moveRun run = .ok moved) :
    moved.count = run.count ∧ ∀ index < run.count,
      finalColumn (run.first + run.step * index) = .ok (moved.first + moved.step * index) := by
  cases mapped : finalColumn run.first with
  | error message => simp [Bind.bind, Except.bind, moveRun, mapped] at emitted
  | ok first =>
    simp only [Bind.bind, Pure.pure, moveRun, mapped, Except.bind, Except.pure] at emitted
    split at emitted
    · rename_i checked
      cases emitted
      refine ⟨rfl, ?_⟩
      intro index bound
      have exact := List.all_eq_true.mp checked index (List.mem_range.mpr bound)
      exact of_decide_eq_true exact
    · change Except.error "retained source run crosses a removed source interval" = Except.ok moved at emitted
      cases emitted

/-- Each successful run stream reads the exact relocated source, in order.
The proof is structural in the stream; no circuit rows are enumerated. -/
theorem moveRuns_source (runs moved : List AffineRuns.Run) (emitted : moveRuns runs = .ok moved) :
    (moved.map AffineRuns.Run.count).sum = (runs.map AffineRuns.Run.count).sum ∧
    ∀ index < (runs.map AffineRuns.Run.count).sum,
      finalColumn (AffineRuns.sourceAt runs index) = .ok (AffineRuns.sourceAt moved index) := by
  induction runs generalizing moved with
  | nil =>
    simp [moveRuns] at emitted
    subst moved
    simp
  | cons run rest inductionHypothesis =>
    cases first : moveRun run with
    | error message => simp [Bind.bind, Except.bind, moveRuns, first] at emitted
    | ok head =>
      cases tail : moveRuns rest with
      | error message => simp [Bind.bind, Except.bind, moveRuns, first, tail] at emitted
      | ok suffix =>
        simp [Bind.bind, Pure.pure, Except.bind, Except.pure, moveRuns, first, tail] at emitted
        subst moved
        obtain ⟨headCount, headSource⟩ := moveRun_source run head first
        obtain ⟨tailCount, tailSource⟩ := inductionHypothesis suffix tail
        refine ⟨by simp only [List.map_cons, List.sum_cons, headCount, tailCount], ?_⟩
        intro index bound
        simp only [List.map_cons, List.sum_cons] at bound
        simp only [AffineRuns.sourceAt, headCount]
        split
        · exact headSource index (by assumption)
        · exact tailSource (index - run.count) (by omega)

theorem moveRuns_value (runs moved : List AffineRuns.Run) (emitted : moveRuns runs = .ok moved)
    (target : Env) (index : Nat) (bounded : index < (runs.map AffineRuns.Run.count).sum) :
    target (AffineRuns.sourceAt moved index) =
      finalMap.pullback target (AffineRuns.sourceAt runs index) := by
  have mapped := (moveRuns_source runs moved emitted).2 index bounded
  simp only [PhysicalRelabel.Map.pullback, finalMap, mapped]

def takeRuns : Nat → List AffineRuns.Run → List AffineRuns.Run
  | 0, _ => []
  | _, [] => []
  | count, first :: rest =>
    if first.count ≤ count then first :: takeRuns (count - first.count) rest
    else [{ first with count := count }]

theorem takeRuns_source (runs : List AffineRuns.Run) (count index : Nat)
    (within : index < count) (bounded : index < (runs.map AffineRuns.Run.count).sum) :
    AffineRuns.sourceAt (takeRuns count runs) index = AffineRuns.sourceAt runs index := by
  induction runs generalizing count index with
  | nil => simp at bounded
  | cons run rest inductionHypothesis =>
    cases count with
    | zero => omega
    | succ count =>
      simp only [takeRuns]
      split
      · simp only [AffineRuns.sourceAt]
        split
        · rfl
        · apply inductionHypothesis <;> simp only [List.map_cons, List.sum_cons] at bounded <;> omega
      · have head : index < run.count := by omega
        simp only [AffineRuns.sourceAt, within, head, if_pos]

theorem takeRuns_count (runs : List AffineRuns.Run) (count : Nat) :
    ((takeRuns count runs).map AffineRuns.Run.count).sum =
      min count (runs.map AffineRuns.Run.count).sum := by
  induction runs generalizing count with
  | nil => cases count <;> simp [takeRuns]
  | cons run rest inductionHypothesis =>
    cases count with
    | zero => simp [takeRuns]
    | succ count =>
      simp only [takeRuns]
      split
      · simp only [List.map_cons, List.sum_cons, inductionHypothesis]
        omega
      · simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
        omega

def commonKinds : List BlockKind :=
  [.priorPoseidon, .outputPoseidon, .laterPoseidon,
   .priorPoseidonInput, .outputPoseidonInput, .runningPiDec, .runningInverse,
   .runningFlag, .piCcsFreshPublicInput, .piCcsPriorLast, .piCcsOutputLast,
   .piCcsExpectedContext, .piCcsProofLogical, .piCcsOutputEndpoint, .piCcsFresh,
   .pilotCanonicalLocal, .pilotCanonicalFresh, .pilotOutputDigest,
   .piDecLogical, .piDecFresh, .applicationWitness, .applicationLocal]

def commonLimit (program : Program) (kind : BlockKind) : Nat :=
  if kind = .laterPoseidon then 7604 * 86 else (entry program kind).block.slotCount

theorem commonLimit_le (program : Program) (kind : BlockKind) :
    commonLimit program kind ≤ (BlockPlan.ofKind program kind).slotCount := by
  unfold commonLimit
  split
  · subst kind
    change 7604 * 86 ≤ (PiRLCRetainedGeometry.laterPoseidonBlock program).slotCount
    exact LaterPoseidonRetainedBlocks.piCcsFits program
  · exact Nat.le_refl _

def sourcesBounded (width : Nat) (runs : List AffineRuns.Run) : Bool :=
  runs.all fun run => decide (run.count = 0 ∨ run.first + run.step * (run.count - 1) < width)

theorem sourcesBounded_source (width : Nat) (runs : List AffineRuns.Run)
    (bounded : sourcesBounded width runs = true) (index : Nat)
    (inside : index < (runs.map AffineRuns.Run.count).sum) : AffineRuns.sourceAt runs index < width := by
  induction runs generalizing index with
  | nil => simp at inside
  | cons run rest ih =>
    simp only [sourcesBounded, List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at bounded
    simp only [List.map_cons, List.sum_cons] at inside
    rw [AffineRuns.sourceAt]
    split
    · rename_i within
      have maximum := Nat.mul_le_mul_left run.step (show index ≤ run.count - 1 by omega)
      rcases bounded.1 with empty | bound
      · omega
      · omega
    · apply ih bounded.2 (index - run.count)
      omega

def commonBlock (program : Program) (kind : BlockKind) : Except String Values :=
  let original := BlockPlan.ofKind program kind
  let count := commonLimit program kind
  let runs := takeRuns count original.sourceRuns
  if sourcesBounded (PiRLCProductPlan.baseSourceWidth program) runs then
    Values.mk original.slotKind count <$> moveRuns runs
  else .error "common retained source exceeds the physical base"

theorem commonBlock_checks (program : Program) (kind : BlockKind) (block : Values)
    (emitted : commonBlock program kind = .ok block) :
    sourcesBounded (PiRLCProductPlan.baseSourceWidth program)
      (takeRuns (commonLimit program kind) (BlockPlan.ofKind program kind).sourceRuns) = true ∧
    (Values.mk (BlockPlan.ofKind program kind).slotKind (commonLimit program kind) <$>
      moveRuns (takeRuns (commonLimit program kind) (BlockPlan.ofKind program kind).sourceRuns)) = .ok block := by
  unfold commonBlock at emitted
  dsimp only at emitted
  split at emitted
  · exact ⟨by assumption, emitted⟩
  · cases emitted

/-- An emitted common block selects exactly the reference block's source
slot after checked relocation, including the shortened PiCCS hash prefix. -/
theorem commonBlock_source (program : Program) (kind : BlockKind) (block : Values)
    (emitted : commonBlock program kind = .ok block) (index : Fin (commonLimit program kind)) :
    finalColumn (sourceIndex program kind
      ⟨index.val, lt_of_lt_of_le index.isLt (commonLimit_le program kind)⟩) =
      .ok (AffineRuns.sourceAt block.sources index.val) := by
  let original := BlockPlan.ofKind program kind
  let runs := takeRuns (commonLimit program kind) original.sourceRuns
  have originalCount : (original.sourceRuns.map AffineRuns.Run.count).sum = original.slotCount :=
    BlockPlan.ofKind_sourceRuns_count program kind
  have bounded : index.val < (runs.map AffineRuns.Run.count).sum := by
    dsimp only [runs]
    rw [takeRuns_count, originalCount, Nat.min_eq_left (commonLimit_le program kind)]
    exact index.isLt
  have emitted := (commonBlock_checks program kind block emitted).2
  change (Values.mk original.slotKind (commonLimit program kind) <$> moveRuns runs) = .ok block at emitted
  cases result : moveRuns runs with
  | error message => simp [result] at emitted
  | ok moved =>
    simp only [result, Except.map_ok, Except.ok.injEq] at emitted
    subst block
    have mapped := (moveRuns_source runs moved result).2 index.val bounded
    dsimp only [runs] at mapped
    rw [takeRuns_source original.sourceRuns _ index.val index.isLt
      (by rw [originalCount]; exact lt_of_lt_of_le index.isLt (commonLimit_le program kind))] at mapped
    have selected : AffineRuns.sourceAt original.sourceRuns index.val =
        sourceIndex program kind ⟨index.val, lt_of_lt_of_le index.isLt (commonLimit_le program kind)⟩ := by
      rw [AffineRuns.sourceAt_eq_expand_getD]
      change (AffineRuns.expand (sourceRunsFor program kind)).getD index.val 0 = _
      rw [sourceRuns_expand]
      exact Lifecycle.PriorStateHash.ofFn_getD _
        ⟨index.val, lt_of_lt_of_le index.isLt (commonLimit_le program kind)⟩ 0
    rw [selected] at mapped
    exact mapped

theorem commonBlock_source_lt (program : Program) (kind : BlockKind) (block : Values)
    (emitted : commonBlock program kind = .ok block) (index : Fin (commonLimit program kind)) :
    sourceIndex program kind ⟨index.val, lt_of_lt_of_le index.isLt (commonLimit_le program kind)⟩ <
      PiRLCProductPlan.baseSourceWidth program := by
  let original := BlockPlan.ofKind program kind
  have count := BlockPlan.ofKind_sourceRuns_count program kind
  have read := sourcesBounded_source _ _ (commonBlock_checks program kind block emitted).1 index.val (by
    rw [takeRuns_count, count, Nat.min_eq_left (commonLimit_le program kind)]
    exact index.isLt)
  rw [takeRuns_source original.sourceRuns _ index.val index.isLt
    (by rw [count]; exact lt_of_lt_of_le index.isLt (commonLimit_le program kind)),
    AffineRuns.sourceAt_eq_expand_getD] at read
  change (AffineRuns.expand (sourceRunsFor program kind)).getD index.val 0 < _ at read
  rw [sourceRuns_expand] at read
  change (List.ofFn (sourceIndex program kind)).getD index.val 0 < _ at read
  rw [Lifecycle.PriorStateHash.ofFn_getD (sourceIndex program kind)
    ⟨index.val, lt_of_lt_of_le index.isLt (commonLimit_le program kind)⟩ 0] at read
  exact read

def samplerPoseidonSource (index : Fin (34 * 86)) : Nat :=
    let invocation := index.val / 86
    let start := if invocation % 2 = 0 then PiRLCStarts.entryLogicalStart (invocation / 2)
      else PiRLCStarts.advanceLogicalStart (invocation / 2)
    SourceOrder.column start +
      (PoseidonRetainedSlots.localOutput ⟨index.val % 86, by rw [PoseidonRetainedSlots.rows_length]; omega⟩).val

def samplerPoseidon : Values := Values.ofSource .field samplerPoseidonSource

def rangeBlock (source : Nat) (block : LowNormBlock.Block 2025) : Values :=
  Values.ofSource block.kind fun index : Fin block.slotCount =>
    SourceOrder.column (PiRLCStarts.rangeLogicalStart source) + (block.source index).val - 4

theorem rangeBlock_source (source : Nat) (block : LowNormBlock.Block 2025)
    (index : Fin block.slotCount) :
    AffineRuns.sourceAt (rangeBlock source block).sources index.val =
      SourceOrder.column (PiRLCStarts.rangeLogicalStart source) + (block.source index).val - 4 :=
  Values.ofSource_source _ _ index

theorem rangeBlock_count (source : Nat) (block : LowNormBlock.Block 2025) :
    (((rangeBlock source block).sources).map AffineRuns.Run.count).sum = block.slotCount := by
  rw [← AffineRuns.expand_length]
  change (AffineRuns.expand (AffineRuns.compressIndexedTR _)).length = _
  rw [AffineRuns.compressIndexedTR_eq_compress_ofFn, AffineRuns.expand_compress, List.length_ofFn]

/-- Retained range operands name the same physical cells as the source
permutation used by the proved hint program. -/
theorem rangeBlock_physical_source (source : Fin 17) (block : LowNormBlock.Block 2025)
    (index : Fin block.slotCount) (retained : 4 ≤ (block.source index).val) :
    AffineRuns.sourceAt (rangeBlock source.val block).sources index.val =
      SourceOrder.column (PiRLCStarts.rangeLogicalStart source.val + ((block.source index).val - 4)) := by
  rw [rangeBlock_source, SourceOrder.column_late_add]
  · have lower : 14751804 ≤ PiRLCStarts.rangeLogicalStart source.val := by
      change 14751804 ≤ 19513117 + source.val * 3205 + 592
      omega
    rw [SourceOrder.column_late _ lower (by
      rw [SourceOrder.sourceWidth_eq]
      change 19513117 + source.val * 3205 + 592 < 27859538
      have bound : source.val < 17 := source.isLt
      omega)]
    omega
  · change 14751804 ≤ 19513117 + source.val * 3205 + 592
    omega
  · rw [SourceOrder.sourceWidth_eq]
    change 19513117 + source.val * 3205 + 592 + ((block.source index).val - 4) < 27859538
    have sourceBound : source.val < 17 := source.isLt
    have cellBound : (block.source index).val < 2025 := (block.source index).isLt
    omega

def samplerRanges : List Values :=
  (List.range 17).flatMap fun source =>
    [rangeBlock source PiRlcWideSampler.Retained.canonicalBits,
     rangeBlock source PiRlcWideSampler.Retained.canonicalFields,
     rangeBlock source PiRlcWideSampler.Retained.resultBits]

def challenges : List AffineRuns.Run :=
  (List.range 17).map fun source =>
    ⟨SourceOrder.column (Gadgets.Sampling.WideReduction.digitStart
      (Gadgets.Sampling.WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart source))), 1, 54 * 3⟩

structure Plan where
  blocks : List Values
  families : List PerApplicationAssignmentTransport.Phi81FamilyShape
  valueSources : List AffineRuns.Run
  challengeSources : List AffineRuns.Run
  outputDigestExpressions : List Expr

def Plan.encode (plan : Plan) : Value := .array [
  .atom 4,
  .array (plan.blocks.map Values.encode),
  .array [
    (list PerApplicationAssignmentTransport.Phi81FamilyShape.format).encode plan.families,
    AffineRuns.format.encode plan.valueSources,
    AffineRuns.format.encode plan.challengeSources],
  (list exprFormat).encode plan.outputDigestExpressions]

def Plan.coordinateCount (plan : Plan) : Nat :=
  270 + (plan.blocks.map fun block => block.count * block.kind.width).sum

def plan (program : Program) (physicalWidth : Nat) : Except String Plan := do
  let common ← commonKinds.mapM (commonBlock program)
  let outputs ← commonBlock program .productOutput
  return {
    blocks := common ++ [samplerPoseidon] ++ samplerRanges ++
      [outputs, ⟨.field, 52326, [⟨physicalWidth, 1, 52326⟩]⟩]
    families := PerApplicationAssignmentTransport.phi81FamilyShapes
    valueSources := ← moveRuns (PerApplicationAssignmentTransport.phi81ValueSources program)
    challengeSources := challenges
    outputDigestExpressions := ← (PerApplicationAssignmentTransport.outputDigestExpressions program).mapM
      finalMap.expression }

end NightstreamFPrime.Export.Stage1.Wide.AssignmentTransport
