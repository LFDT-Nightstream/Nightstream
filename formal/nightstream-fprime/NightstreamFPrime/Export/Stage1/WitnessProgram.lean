import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
import NightstreamFPrime.Export.Stage1.PiDECArithmetic
import NightstreamFPrime.Export.Stage1.RunningTransitionArithmetic
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.Witness

/-!
Owns the canonical logical witness-program IR through the running transition.

The seven arithmetic children already export `WitnessBatch` recipes through
their opaque `FormalCircuit` interfaces. This module gathers those batches in
protocol order and remaps only their symbolic variable indices through the
proved Stage 1 Spartan permutation. PiCCS Poseidon2 children remain represented
by compact permutation invocations. PiRLC digest-lane batches are built
directly and proved equal to the opaque child traversal. PiRLC permutation and
`First54` outputs are owned by their package invocations and are not written
again here.
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

def remapBatch (batch : WitnessBatch) : WitnessBatch where
  start := NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan batch.start
  recipes := batch.recipes.map remapExpr
  hints := batch.hints.map fun hint =>
    match hint with
    | .bit source index => .bit (remapExpr source) index
    | .inverseOrZero source => .inverseOrZero (remapExpr source)
    | .quotientFive source => .quotientFive (remapExpr source)
    | .remainderFive source => .remainderFive (remapExpr source)

@[simp] theorem remapBatch_start (batch : WitnessBatch) :
    (remapBatch batch).start =
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan batch.start := by
  rfl

@[simp] theorem remapBatch_recipes_length (batch : WitnessBatch) :
    (remapBatch batch).recipes.length = batch.recipes.length := by
  simp [remapBatch]

@[simp] theorem remapBatch_hints_length (batch : WitnessBatch) :
    (remapBatch batch).hints.length = batch.hints.length := by
  simp [remapBatch]

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

private def piRlcDigestLaneBatchesFromCircuit
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) (lane : Fin 4) : List WitnessBatch :=
  childBatches
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.circuit
      (PiRLCSamplerOrdinaryRows.laneInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)).main
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
      source round lane.val)

def digestLaneBatches
    (source : Expr) (offset : Nat) : List WitnessBatch :=
  (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.witnessBatchesForSource
    source offset).map remapBatch

def piRlcDigestLaneBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) (lane : Fin 4) : List WitnessBatch :=
  let interface := PiRLCSamplerOrdinaryRows.laneInterface
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    source round lane
  let offset :=
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
      source round lane.val
  digestLaneBatches (interface.source offset) offset

theorem piRlcDigestLaneBatches_eq_fromCircuit
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) (lane : Fin 4) :
    piRlcDigestLaneBatches logicalWidth publicFits source round lane =
      piRlcDigestLaneBatchesFromCircuit logicalWidth publicFits
        source round lane := by
  simp [piRlcDigestLaneBatches, digestLaneBatches,
    piRlcDigestLaneBatchesFromCircuit,
    childBatches,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.witnessBatchesForSource_eq,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.witnesses_circuit_main]

def piRlcWindowBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) : List WitnessBatch :=
  (List.finRange 4).flatMap
    (piRlcDigestLaneBatches logicalWidth publicFits source round)

def piRlcSourceBatches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source : Nat) : List WitnessBatch :=
  (List.range 8).flatMap
    (piRlcWindowBatches logicalWidth publicFits source)

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
