import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.PiRLCSamplerProjection
import NightstreamFPrime.Export.WitnessEncoding
import NightstreamFPrime.Layout.Stage1.Wide.SourceOrder
import NightstreamFPrime.Layout.Stage1.Wide.PiRLCStarts
import NightstreamFPrime.Lifecycle.PiRLC.Wide.DigitWords

/-! Physical witness data for the wide sampler. The range constraints and
hint program come from the proved gadget; Poseidon uses the existing compact
permutation template. Temporary digit words feed the existing ring recipes. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PhysicalSampler

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Sampling NightstreamFPrime.Lifecycle.PiRLC
open Layout.Stage1.Wide

def expression (value : Expr) : Expr := CompactRows.renameExpr SourceOrder.column value

def hint : Hint → Hint
  | .bit source index => .bit (expression source) index
  | .inverseOrZero source => .inverseOrZero (expression source)
  | .quotientFive source => .quotientFive (expression source)
  | .remainderFive source => .remainderFive (expression source)

def batch (value : WitnessBatch) : WitnessBatch :=
  WitnessEncoding.batch {
    start := SourceOrder.column value.start
    recipes := value.recipes.map expression
    hints := value.hints.map hint }

def beforeState (source : Nat) : Layer.EState :=
  match source with
  | 0 => PiRLCSamplerProjection.productionInitialState
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
  | previous + 1 => Permutation.scheduleOutput (PiRLCStarts.advanceLogicalStart previous)

def enteredState (source : Nat) : Layer.EState :=
  Permutation.scheduleOutput (PiRLCStarts.entryLogicalStart source)

def rangeInterface (source : Nat) : WideReduction.Interface where
  source := fun lane _ => enteredState source ⟨lane.val, lt_trans lane.isLt (by decide)⟩

def rangeOperations (source : Nat) : List Op :=
  WideReduction.Program.operations (rangeInterface source) (PiRLCStarts.rangeLogicalStart source)

def digitOperations : List Op :=
  Wide.DigitWords.operations PiRLCStarts.samplerLogicalStart (PiRLCStarts.challengeWordStart 0)

def compilePacket (rowStart freshStart : Nat) (constraints : List Expr) : List Rows.CompiledRow :=
  Rows.compileRowsTR (SourceOrder.column freshStart) rowStart
    ((Rows.lowerConstraintsTR constraints freshStart).rows.map (R1CS.mapRowColumns SourceOrder.column))

theorem compilePacket_rows (rowStart freshStart : Nat) (constraints : List Expr) :
    (compilePacket rowStart freshStart constraints).map Rows.CompiledRow.toR1CS =
      (R1CS.lowerConstraints constraints freshStart).rows.map (R1CS.mapRowColumns SourceOrder.column) := by
  rw [compilePacket, Rows.compileRowsTR_toR1CS, Rows.lowerConstraintsTR_eq]

theorem expression_eval (value : Expr) (env : Env) :
    (expression value).eval env = value.eval (fun index => env (SourceOrder.column index)) :=
  CompactRows.renameExpr_eval SourceOrder.column value env

theorem hint_eval (value : Hint) (env : Env) :
    (hint value).eval env = value.eval (fun index => env (SourceOrder.column index)) := by
  cases value <;> simp only [hint, Hint.eval, expression_eval]

def rows (_unit : Unit) : List Rows.CompiledRow :=
  (List.range 17).flatMap (fun source =>
    compilePacket (PiRLCStarts.rangeRowStart source) (PiRLCStarts.samplerSourceFreshStart source)
      (flatConstraints (rangeOperations source))) ++
  compilePacket (PiRLCStarts.samplerSourceRowStart 17) PiRLCStarts.commitmentFreshStart
    (flatConstraints digitOperations)

def batches (_unit : Unit) : List WitnessBatch :=
  ((List.range 17).flatMap (fun source => witnesses (rangeOperations source)) ++
    witnesses digitOperations).map batch

def inputCombination (value : Expr) : Except String SparseCombination :=
  match R1CS.lowerAffine value with
  | some lowered => .ok (Rows.sparseCombination (R1CS.mapCombinationColumns SourceOrder.column lowered.combination))
  | none => .error "non-affine wide sampler permutation input"

def invocation (rowStart witnessStart : Nat) (state : Layer.EState) : Except String PermutationInvocation := do
  return {
    phase := 7
    rowStart := rowStart
    witnessStart := SourceOrder.column witnessStart
    inputs := ← (List.ofFn state).mapM inputCombination }

def permutations (_unit : Unit) : Except String (List PermutationInvocation) := do
  let parts ← (List.range 17).mapM fun source => do
    let input := Hash.absorbE (beforeState source)
      (v1_1.TranscriptAbsorption.constantWords (v1_1.TranscriptAbsorption.frameWords source))
    return [← invocation (PiRLCStarts.entryRowStart source) (PiRLCStarts.entryLogicalStart source) input,
      ← invocation (PiRLCStarts.advanceRowStart source) (PiRLCStarts.advanceLogicalStart source) (enteredState source)]
  return parts.flatten

end NightstreamFPrime.Export.Stage1.Wide.PhysicalSampler
