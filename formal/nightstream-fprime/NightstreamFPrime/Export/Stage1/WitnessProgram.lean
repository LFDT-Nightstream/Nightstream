import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows

/-!
Owns the canonical logical witness-program IR through the PiRLC sampler.

The seven arithmetic children already export `WitnessBatch` recipes through
their opaque `FormalCircuit` interfaces. This module gathers those batches in
protocol order and remaps only their symbolic variable indices through the
proved Stage 1 Spartan permutation. PiCCS Poseidon2 children remain represented
by compact permutation invocations. PiRLC digest-lane batches are built
directly and proved equal to the opaque child traversal. PiRLC permutation and
`First54` outputs are owned by their package invocations and are not written
again here.
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

private def canonicalU64Batches
    (interface : NightstreamFPrime.Gadgets.Range.CanonicalU64.Interface)
    (offset : Nat) : List WitnessBatch :=
  [ WitnessBatch.hinted offset
      (NightstreamFPrime.Gadgets.Range.CanonicalU64.bitHints interface offset),
    WitnessBatch.hinted
      (offset + NightstreamFPrime.Gadgets.Range.CanonicalU64.bitCount)
      [NightstreamFPrime.Gadgets.Range.CanonicalU64.inverseHint offset],
    WitnessBatch.arithmetic
      (offset + NightstreamFPrime.Gadgets.Range.CanonicalU64.bitCount + 1)
      [NightstreamFPrime.Gadgets.Range.CanonicalU64.flagRecipe offset] ]

private def candidate16FiveBatches
    (interface : NightstreamFPrime.Gadgets.Sampling.Candidate16Five.Interface)
    (offset : Nat) : List WitnessBatch :=
  [ WitnessBatch.hinted offset
      (NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientRemainderHints
        interface offset),
    WitnessBatch.hinted (offset + 2)
      (NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBitHints offset),
    WitnessBatch.arithmetic (offset + 16)
      [NightstreamFPrime.Gadgets.Sampling.Candidate16Five.rejectRecipe
        interface offset] ]

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
  let canonicalInterface :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.canonicalInterface
      interface offset
  let canonicalOffset :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.canonicalOffset offset
  let lowInterface :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.decoderInterface
      offset NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.lowPart
  let lowOffset :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.decoderOffset
      offset NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.lowPart
  let highInterface :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.decoderInterface
      offset NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.highPart
  let highOffset :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.decoderOffset
      offset NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.highPart
  (canonicalU64Batches canonicalInterface canonicalOffset ++
    candidate16FiveBatches lowInterface lowOffset ++
    candidate16FiveBatches highInterface highOffset).map remapBatch

theorem piRlcDigestLaneBatches_eq_fromCircuit
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) (lane : Fin 4) :
    piRlcDigestLaneBatches logicalWidth publicFits source round lane =
      piRlcDigestLaneBatchesFromCircuit logicalWidth publicFits
        source round lane := by
  simp [piRlcDigestLaneBatches, piRlcDigestLaneBatchesFromCircuit,
    childBatches, canonicalU64Batches, candidate16FiveBatches,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.circuit,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.main,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.opsAt,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.canonicalOp,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.lowOp,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.highOp,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.canonicalCircuit,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.decoderCircuit,
    NightstreamFPrime.Circuit.Sequence.childOp,
    FormalCircuit.asSubcircuit, witnesses, Op.witnesses,
    NightstreamFPrime.Gadgets.Range.CanonicalU64.circuit,
    NightstreamFPrime.Gadgets.Range.CanonicalU64.main,
    NightstreamFPrime.Gadgets.Range.CanonicalU64.operations,
    NightstreamFPrime.Gadgets.Range.CanonicalU64.booleanOps,
    NightstreamFPrime.Gadgets.Sampling.Candidate16Five.circuit,
    NightstreamFPrime.Gadgets.Sampling.Candidate16Five.main,
    NightstreamFPrime.Gadgets.Sampling.Candidate16Five.operations,
    NightstreamFPrime.Gadgets.Sampling.Candidate16Five.quotientBooleanOps]

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

/-- Exact logical-witness order through the PiRLC sampler prefix. -/
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
    finalIdentityBatches logicalWidth publicFits ++
    piRlcSamplerBatches logicalWidth publicFits

end NightstreamFPrime.Export.Stage1.WitnessProgram
