import NightstreamFPrime.Layout.PiRLC.v1_1.Composition

/-!
Owns the exact structural boundary between each opaque PiRLC child and the
constraint list used by the phase lowering. These equalities let later export
proofs preserve child opacity without a large definitional reduction.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

theorem inputConstraints_eq_nil
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    childConstraints
        (Formal.inputBindingCircuit relation (Formal.atOffset interface offset))
        (Formal.inputBindingOffset offset) = [] := by
  rfl

theorem samplerConstraints_eq
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    childConstraints
        (Formal.samplerCircuit (Formal.atOffset interface offset))
        (Formal.samplerOffset offset) =
      SamplerChain.logicalConstraints
        (Formal.samplerInterface (Formal.atOffset interface offset))
        (Formal.samplerOffset offset) := by
  rfl

theorem commitmentConstraints_eq
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    childConstraints
        (Formal.commitmentCircuit (Formal.atOffset interface offset))
        (Formal.commitmentOffset offset) =
      CommitmentCombination.logicalConstraints
        (Formal.commitmentInterface (Formal.atOffset interface offset))
        (Formal.commitmentOffset offset) := by
  rfl

theorem publicInputConstraints_eq
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    childConstraints
        (Formal.publicInputCircuit (Formal.atOffset interface offset))
        (Formal.publicInputOffset offset) =
      PublicInputCombination.logicalConstraints
        (Formal.publicInputInterface (Formal.atOffset interface offset))
        (Formal.publicInputOffset offset) := by
  rfl

theorem evalKConstraints_eq
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    childConstraints
        (Formal.evalKCircuit (Formal.atOffset interface offset))
        (Formal.evalKOffset offset) =
      EvalKCombination.logicalConstraints
        (Formal.evalKInterface (Formal.atOffset interface offset))
        (Formal.evalKOffset offset) := by
  rfl

theorem evalAConstraints_eq
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    childConstraints
        (Formal.evalACircuit (Formal.atOffset interface offset))
        (Formal.evalAOffset offset) =
      EvalACombination.logicalConstraints
        (Formal.evalAInterface (Formal.atOffset interface offset))
        (Formal.evalAOffset offset) := by
  rfl

theorem outputConstraints_eq_nil
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    childConstraints
        (Formal.outputBindingCircuit relation (Formal.atOffset interface offset)
          offset)
        (Formal.outputBindingOffset offset) = [] := by
  rfl

def samplerPacketConstraints
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  SamplerChain.logicalConstraints
    (Formal.samplerInterface (Formal.atOffset interface offset))
    (Formal.samplerOffset offset)

def commitmentPacketConstraints
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  CommitmentCombination.logicalConstraints
    (Formal.commitmentInterface (Formal.atOffset interface offset))
    (Formal.commitmentOffset offset)

def publicInputPacketConstraints
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  PublicInputCombination.logicalConstraints
    (Formal.publicInputInterface (Formal.atOffset interface offset))
    (Formal.publicInputOffset offset)

def evalKPacketConstraints
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  EvalKCombination.logicalConstraints
    (Formal.evalKInterface (Formal.atOffset interface offset))
    (Formal.evalKOffset offset)

def evalAPacketConstraints
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  EvalACombination.logicalConstraints
    (Formal.evalAInterface (Formal.atOffset interface offset))
    (Formal.evalAOffset offset)

def packetConstraintLists
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    List (List Expr) :=
  [[], samplerPacketConstraints interface offset,
    commitmentPacketConstraints interface offset,
    publicInputPacketConstraints interface offset,
    evalKPacketConstraints interface offset,
    evalAPacketConstraints interface offset, []]

def packetConstraints
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  (packetConstraintLists interface offset).flatten

/-- The phase child ledger contains two empty views and the five opaque
nonempty packet lists, in exact parent order. -/
theorem childConstraintLists_eq_packets
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    childConstraintLists relation interface offset =
      packetConstraintLists interface offset := by
  unfold childConstraintLists packetConstraintLists samplerPacketConstraints
    commitmentPacketConstraints publicInputPacketConstraints
    evalKPacketConstraints evalAPacketConstraints
  dsimp only
  rw [inputConstraints_eq_nil, samplerConstraints_eq,
    commitmentConstraints_eq, publicInputConstraints_eq,
    evalKConstraints_eq, evalAConstraints_eq, outputConstraints_eq_nil]

end NightstreamFPrime.Layout.PiRLC.v1_1.PacketBoundaries
