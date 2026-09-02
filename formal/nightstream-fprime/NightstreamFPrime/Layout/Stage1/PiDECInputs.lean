import NightstreamFPrime.Layout.PiDEC.v1_1.Ownership
import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLC
import NightstreamFPrime.Layout.Stage1.PiRLCStarts

/-!
Owns the zero-copy PiRLC-to-PiDEC bridge and the constrained PiDEC input ABI.

The parent commitment, public input, point, separate `Eval_K`, and 14 separate
`Eval_A` families reuse the exact PiRLC output wires. New private input words
hold the 16 prover messages and the 16×54 verifier-computed child public-input
values. These words are non-authoritative until the PiDEC rows accept them.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

/-- The completed PiRLC private-column endpoint. -/
def proofInputStart : Nat := 28973248

def childCount : Nat := 16
def commitmentWordsPerChild : Nat := 1188
def evalKWordsPerChild : Nat := 108
def evalAWordsPerChild : Nat := 1512
def publicInputWordsPerChild : Nat := 270

def commitmentInputStart : Nat := proofInputStart
def evalKInputStart : Nat :=
  commitmentInputStart + childCount * commitmentWordsPerChild
def evalAInputStart : Nat :=
  evalKInputStart + childCount * evalKWordsPerChild
def publicInputStart : Nat :=
  evalAInputStart + childCount * evalAWordsPerChild

def proofInputColumnCount : Nat :=
  childCount * (commitmentWordsPerChild + evalKWordsPerChild +
    evalAWordsPerChild + publicInputWordsPerChild)

/-- PiDEC logical witnesses start after every constrained input word. -/
def phaseOffset : Nat := proofInputStart + proofInputColumnCount

theorem proofInputStart_matches_piRlc
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    proofInputStart = PilotPiCCSPiRLC.physicalColumnCount relation := by
  rw [PilotPiCCSPiRLC.physicalColumnCount_eq]
  rfl

theorem proofInputColumnCount_eq : proofInputColumnCount = 49248 := by
  rfl

theorem inputStarts_eq :
    [commitmentInputStart, evalKInputStart, evalAInputStart, publicInputStart,
      phaseOffset] =
    [28973248, 28992256, 28993984, 29018176, 29022496] := by
  rfl

def childCommitmentStart (child : Radix.ChildIndex) : Nat :=
  commitmentInputStart + child.val * commitmentWordsPerChild

def childEvalKStart (child : Radix.ChildIndex) : Nat :=
  evalKInputStart + child.val * evalKWordsPerChild

def childEvalAStart (child : Radix.ChildIndex) : Nat :=
  evalAInputStart + child.val * evalAWordsPerChild

def childPublicInputStart (child : Radix.ChildIndex) : Nat :=
  publicInputStart + child.val * publicInputWordsPerChild

def childCommitment (child : Radix.ChildIndex)
    (row : Fin productionProfile.commitmentWidth)
    (lane : Fin ringDegree) : Expr :=
  Expr.var (childCommitmentStart child + row.val * ringDegree + lane.val)

def childEvalK (child : Radix.ChildIndex)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  let start := childEvalKStart child + coefficient.val * 2
  ⟨Expr.var start, Expr.var (start + 1)⟩

def childEvalA (child : Radix.ChildIndex)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  let start := childEvalAStart child +
    matrix.val * evalKWordsPerChild + coefficient.val * 2
  ⟨Expr.var start, Expr.var (start + 1)⟩

def childPublicInput (child : Radix.ChildIndex)
    (coordinate : Fin 270) : Expr :=
  Expr.var (childPublicInputStart child + coordinate.val)

def piRlcInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)

def piRlcSharedInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset
    (piRlcInterface logicalWidth publicFits) PiRLCInputs.phaseOffset

def piRlcOutputInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.outputBindingInterface
    (piRlcSharedInterface logicalWidth publicFits) PiRLCInputs.phaseOffset

def parent
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.ParentExpr
      logicalWidth publicFits where
  commitment := (piRlcOutputInterface logicalWidth publicFits).commitment
    PiRLCStarts.outputLogicalStart
  publicInput := (piRlcOutputInterface logicalWidth publicFits).publicInput
    PiRLCStarts.outputLogicalStart
  evaluation := {
    eval_K := (piRlcOutputInterface logicalWidth publicFits).eval_K
      PiRLCStarts.outputLogicalStart
    eval_A := (piRlcOutputInterface logicalWidth publicFits).eval_A
      PiRLCStarts.outputLogicalStart }

def message (child : Radix.ChildIndex) :
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.ChildMessageExpr where
  commitment := childCommitment child
  evaluation := {
    eval_K := childEvalK child
    eval_A := childEvalA child }

/-- The sole production PiDEC interface in cumulative Stage 1 order. -/
def interface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Interface
      logicalWidth publicFits where
  parent := fun _ => parent logicalWidth publicFits
  point := fun _ =>
    (piRlcOutputInterface logicalWidth publicFits).point
      PiRLCStarts.outputLogicalStart
  message := fun _ => message
  digit := fun _ child coordinate =>
    childPublicInput child
      (Fin.cast
        (NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount_eq
          logicalWidth publicFits)
        coordinate)

def inputShapes
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiDEC.v1_1.InputShapes relation
      (interface logicalWidth publicFits) phaseOffset where
  publicInput := by
    intro childOffset
    refine ⟨?_, ?_, ?_⟩
    · intro coordinate
      rfl
    · intro child coordinate
      rfl
    · intro child coordinate value equality
      change Expr.var _ = Expr.const value at equality
      cases equality
  commitment := by
    intro childOffset
    refine ⟨?_, ?_⟩
    · intro row lane
      rfl
    · intro child row lane
      rfl
  eval_K := by
    intro childOffset
    refine ⟨?_, ?_⟩
    · intro coefficient
      exact ⟨rfl, rfl⟩
    · intro child coefficient
      exact ⟨rfl, rfl⟩
  eval_A := by
    intro childOffset
    refine ⟨?_, ?_⟩
    · intro matrix coefficient
      exact ⟨rfl, rfl⟩
    · intro child matrix coefficient
      exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiDECInputs
