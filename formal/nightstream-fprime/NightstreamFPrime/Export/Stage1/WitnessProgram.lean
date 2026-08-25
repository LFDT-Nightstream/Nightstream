import NightstreamFPrime.Export.Stage1.PiCCSArithmetic

/-!
Owns the canonical logical witness-program IR for the Stage 1 PiCCS prefix.

The seven arithmetic children already export `WitnessBatch` recipes through
their opaque `FormalCircuit` interfaces. This module gathers those batches in
protocol order and remaps only their symbolic variable indices through the
proved Stage 1 Spartan permutation. Poseidon2 children remain represented by
their compact permutation invocations and are not duplicated here.
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

/-- Exact arithmetic-child order from the sole PiCCS phase assembler. -/
def batches
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

end NightstreamFPrime.Export.Stage1.WitnessProgram
