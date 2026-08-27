import NightstreamFPrime.Lifecycle.PriorStateHashCore
import NightstreamFPrime.Lifecycle.Relation

/-!
Connects the proved low-norm prior-state hash circuit to the exact recursive
relation slot. Circuit construction and layout are owned by
`PriorStateHashCore`.
-/

namespace NightstreamFPrime.Lifecycle.PriorStateHash

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle.PaperAlgebra

section Relation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}

def RepresentsPreimage (interface : Interface) (offset : Nat) (env : Env)
    (preimage : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits)) : Prop :=
  Hash.evalList env (interface.preimage offset) =
    serializePreimage (publicFits := publicFits) preimage

def RepresentsPublicInput (interface : Interface) (offset : Nat) (env : Env)
    (publicInput : PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits)) : Prop :=
  ∀ column, (interface.publicInput offset column).eval env = publicInput column

/-- The logical builder's specification implies the exact recursive relation
slot, with the same authoritative preimage and public input. -/
theorem builder_implies_priorPublicInput
    (interface : Interface) (offset : Nat) (env : Env)
    (preimage : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (publicInput : PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (specification : SpecHolds interface offset env)
    (preimageRepresents : RepresentsPreimage interface offset env preimage)
    (publicRepresents : RepresentsPublicInput interface offset env publicInput) :
    publicInput = encHash (publicFits := publicFits)
      (stateHash (publicFits := publicFits) preimage) := by
  funext column
  calc
    publicInput column = (interface.publicInput offset column).eval env :=
      (publicRepresents column).symm
    _ = encodedHash (Poseidon2.hash
        (Hash.evalList env (interface.preimage offset))) column :=
      congrFun specification column
    _ = encodedHash (Poseidon2.hash
        (serializePreimage (publicFits := publicFits) preimage)) column := by
      rw [preimageRepresents]
    _ = encHash (publicFits := publicFits)
        (stateHash (publicFits := publicFits) preimage) column := by
      rfl

/-- Production specialization: the builder proves the exact
`RecursiveHolds.priorPublicInput` equation of `StepHolds`. -/
theorem builder_implies_recursive_slot
    (interface : Interface) (offset : Nat) (env : Env)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (specification : SpecHolds interface offset env)
    (preimageRepresents : RepresentsPreimage interface offset env
      (priorHashPreimage (setup relation ajtai vk) input))
    (publicRepresents : RepresentsPublicInput interface offset env
      ((machine publicFits F).freshPublic input.fresh)) :
    (machine publicFits F).freshPublic input.fresh =
      (machine publicFits F).encodeInstance
        ((machine publicFits F).hash
          (priorHashPreimage (setup relation ajtai vk) input)) := by
  exact builder_implies_priorPublicInput interface offset env
    (priorHashPreimage (setup relation ajtai vk) input)
    ((machine publicFits F).freshPublic input.fresh)
    specification preimageRepresents publicRepresents

end Relation

end NightstreamFPrime.Lifecycle.PriorStateHash
