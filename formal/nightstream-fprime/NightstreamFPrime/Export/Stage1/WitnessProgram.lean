import NightstreamFPrime.Export.WitnessEncoding
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
import NightstreamFPrime.Export.Stage1.PiDECArithmetic
import NightstreamFPrime.Export.Stage1.RunningTransitionArithmetic

/-!
Owns the canonical logical witness-program IR through the running transition.

The seven arithmetic children already export `WitnessBatch` recipes through
their opaque `FormalCircuit` interfaces. This module gathers those batches in
protocol order, remaps their symbolic variables through the proved Stage 1
Spartan permutation, and balances expression sums without changing execution. PiCCS Poseidon2 children remain represented
by compact permutation invocations. PiRLC batches come from the checked
wide-reduction and coefficient-word circuits. Permutation invocations execute
the hash steps; ordinary rows check the reduction and coefficient words.
PiDEC contributes only the 54 sign-hint batches of its opaque public-input
split child; R1CS intermediate recipes remain ordinary row instructions.
The running transition contributes its one inverse-or-zero hint batch.
-/

namespace NightstreamFPrime.Export.Stage1.WitnessProgram

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def remapExpr : Expr → Expr
  | .var index => .var (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan index)
  | .const value => .const value
  | .add left right => .add (remapExpr left) (remapExpr right)
  | .mul left right => .mul (remapExpr left) (remapExpr right)

theorem remapExpr_eval (target : Env) (expression : Expr) :
    (remapExpr expression).eval target =
      expression.eval
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback target) := by
  induction expression with
  | var index =>
      rfl
  | const value =>
      rfl
  | add left right leftIH rightIH =>
      simp [remapExpr, Expr.eval, leftIH, rightIH]
  | mul left right leftIH rightIH =>
      simp [remapExpr, Expr.eval, leftIH, rightIH]

def remapBatch (batch : WitnessBatch) : WitnessBatch := WitnessEncoding.batch {
  start := NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan batch.start
  recipes := batch.recipes.map remapExpr
  hints := batch.hints.map fun hint =>
    match hint with
    | .bit source index => .bit (remapExpr source) index
    | .inverseOrZero source => .inverseOrZero (remapExpr source)
    | .quotientFive source => .quotientFive (remapExpr source)
    | .remainderFive source => .remainderFive (remapExpr source) }

@[simp] theorem remapBatch_start (batch : WitnessBatch) :
    (remapBatch batch).start =
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan batch.start := by
  rfl

@[simp] theorem remapBatch_recipes_length (batch : WitnessBatch) :
    (remapBatch batch).recipes.length = batch.recipes.length := by
  simp [remapBatch, WitnessEncoding.batch]

@[simp] theorem remapBatch_hints_length (batch : WitnessBatch) :
    (remapBatch batch).hints.length = batch.hints.length := by
  simp [remapBatch, WitnessEncoding.batch]

def childBatches (main : Circuit Unit) (offset : Nat) : List WitnessBatch :=
  (witnesses (Circuit.ops main offset)).map remapBatch

def initialClaimBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  childBatches
    (Formal.initialClaimCircuit
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).main
    PiCCSArithmetic.initialClaimLogicalStart

def sumcheckBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  childBatches
    (Formal.sumcheckCircuit
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).main
    PiCCSArithmetic.sumcheckLogicalStart

def evalKBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  childBatches
    (Formal.evalKCircuit
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).main
    PiCCSArithmetic.evalKLogicalStart

def evalABatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  childBatches
    (Formal.evalACircuit
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).main
    PiCCSArithmetic.evalALogicalStart

def ccsBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  childBatches
    (Formal.ccsRowMain
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
    PiCCSArithmetic.ccsLogicalStart

def normBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  childBatches
    (Formal.normRowMain
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
    PiCCSArithmetic.normLogicalStart

def finalIdentityBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  childBatches
    (Formal.finalIdentityRowMain
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
    PiCCSArithmetic.finalIdentityLogicalStart

def piRlcSourceBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source : Nat) : List WitnessBatch :=
  childBatches
    (Gadgets.Sampling.WideReduction.Program.circuit
      (PiRLCSamplerOrdinaryRows.rangeInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits) source)).main
    (Layout.Stage1.PiRLCStarts.rangeLogicalStart source) ++
  childBatches
    (PiRLC.v1_1.SamplerWords.circuit (Layout.Stage1.PiRLCStarts.rangeLogicalStart source)).main
    (Layout.Stage1.PiRLCStarts.challengeWordStart source)

theorem piRlcSourceBatches_eq_fromCircuit
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth)
    (source : Nat) :
    piRlcSourceBatches logicalWidth publicFits source =
      childBatches (PiRLC.v1_1.Sampler.rangeCircuit
        (PiRLCSamplerInvocations.sourceInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source) source
        (PiRLCSamplerInvocations.sourceLogicalStart source)).main
        (Layout.Stage1.PiRLCStarts.rangeLogicalStart source) ++
      childBatches
        (PiRLC.v1_1.SamplerWords.circuit (Layout.Stage1.PiRLCStarts.rangeLogicalStart source)).main
        (Layout.Stage1.PiRLCStarts.challengeWordStart source) := by
  unfold piRlcSourceBatches
  rw [PiRLCSamplerOrdinaryRows.rangeInterface_eq]
  rfl

def piRlcSamplerBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  (List.range 17).flatMap
    (piRlcSourceBatches logicalWidth publicFits)

def piCcsBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  initialClaimBatches logicalWidth publicFits ++
    sumcheckBatches logicalWidth publicFits ++
    evalKBatches logicalWidth publicFits ++
    evalABatches logicalWidth publicFits ++
    ccsBatches logicalWidth publicFits ++
    normBatches logicalWidth publicFits ++
    finalIdentityBatches logicalWidth publicFits

def piDecBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  let shared := NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.atOffset
    (PiDECArithmetic.phaseInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset
  childBatches
    (NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.publicInputCircuit
      shared).main
    NightstreamFPrime.Layout.Stage1.PiDECStarts.publicInputLogicalStart

def runningTransitionBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  childBatches
    (NightstreamFPrime.Lifecycle.Stage1.RunningTransition.circuit
    (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
        logicalWidth publicFits)).main
    NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset

private theorem witnesses_assertions (values : List Expr) :
    witnesses (values.map Op.assertZero) = [] := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [witnesses, Op.witnesses]

/-- Closed-form executable form of the sole running-transition witness batch.
It avoids traversing all 45,894 assertion operations during emission. -/
def directRunningTransitionBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  let interface :=
    NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
      logicalWidth publicFits
  let offset :=
    NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset
  [remapBatch (WitnessBatch.hinted offset
    [NightstreamFPrime.Lifecycle.Stage1.RunningTransition.inverseHint
      interface offset])]

theorem directRunningTransitionBatches_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    directRunningTransitionBatches logicalWidth publicFits =
      runningTransitionBatches logicalWidth publicFits := by
  unfold directRunningTransitionBatches runningTransitionBatches childBatches
  change [_] =
    (witnesses
      (NightstreamFPrime.Lifecycle.Stage1.RunningTransition.operations
        (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
          logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset)
      ).map remapBatch
  unfold NightstreamFPrime.Lifecycle.Stage1.RunningTransition.operations
  change [_] =
    ([_] ++ witnesses
      ((NightstreamFPrime.Lifecycle.Stage1.RunningTransition.constraints
        (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
          logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset
      ).map Op.assertZero)).map remapBatch
  rw [witnesses_assertions]
  rfl

/-- Exact logical-witness order through the running transition. -/
def batches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List WitnessBatch :=
  piCcsBatches logicalWidth publicFits ++
    (piRlcSamplerBatches logicalWidth publicFits ++
      (piDecBatches logicalWidth publicFits ++
        runningTransitionBatches logicalWidth publicFits))

theorem batches_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    batches logicalWidth publicFits =
      piCcsBatches logicalWidth publicFits ++
        (piRlcSamplerBatches logicalWidth publicFits ++
          (piDecBatches logicalWidth publicFits ++
            runningTransitionBatches logicalWidth publicFits)) := by
  rfl

end NightstreamFPrime.Export.Stage1.WitnessProgram
