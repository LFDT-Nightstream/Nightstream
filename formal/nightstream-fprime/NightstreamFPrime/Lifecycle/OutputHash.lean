import NightstreamFPrime.Gadgets.Poseidon2.Formal
import NightstreamFPrime.Lifecycle.Relation

/-!
Owns the logical builder for HyperNova Construction-2's public output digest.
It specializes the proved Poseidon2 child to `nextHashPreimage` and exports the
exact `OutputHolds` relation-slot theorem.
-/

namespace NightstreamFPrime.Lifecycle.OutputHash

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle.PaperAlgebra

/-- External expressions owned by the lifecycle parent. -/
structure Interface where
  preimage : Nat → List Expr
  digest : Nat → Fin 4 → Expr

def hashInterface (interface : Interface) : Formal.Interface where
  input := interface.preimage
  expected := interface.digest

@[simp] theorem hashInterface_input (interface : Interface) (offset : Nat) :
    (hashInterface interface).input offset = interface.preimage offset := by
  rfl

@[simp] theorem hashInterface_expected (interface : Interface) (offset : Nat)
    (lane : Fin 4) :
    (hashInterface interface).expected offset lane =
      interface.digest offset lane := by
  rfl

def Assumptions (interface : Interface) : Nat → Env → Prop :=
  Formal.Assumptions (hashInterface interface)

def SpecHolds (interface : Interface) : Nat → Env → Prop :=
  Formal.SpecHolds (hashInterface interface)

/-- The production logical builder for the `outputHash` phase. -/
def circuit (interface : Interface) : FormalCircuit :=
  Formal.circuit (hashInterface interface)

theorem circuit_localLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) =
      (Hash.compile offset (interface.preimage offset)).recipes.length := by
  exact Formal.opsAt_localLength (hashInterface interface) offset

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (hholds : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  (circuit interface).soundness env offset assumptions hholds

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  (circuit interface).completeness env offset assumptions specification

section Relation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}

def RepresentsPreimage (interface : Interface) (offset : Nat) (env : Env)
    (preimage : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits)) : Prop :=
  Hash.evalList env (interface.preimage offset) =
    serializePreimage (publicFits := publicFits) preimage

def RepresentsDigest (interface : Interface) (offset : Nat) (env : Env)
    (digest : Digest) : Prop :=
  digest = List.ofFn (fun lane => (interface.digest offset lane).eval env)

theorem builder_implies_digest
    (interface : Interface) (offset : Nat) (env : Env)
    (preimage : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (digest : Digest)
    (specification : SpecHolds interface offset env)
    (preimageRepresents : RepresentsPreimage interface offset env preimage)
    (digestRepresents : RepresentsDigest interface offset env digest) :
    digest = stateHash (publicFits := publicFits) preimage := by
  calc
    digest = List.ofFn (fun lane =>
        (interface.digest offset lane).eval env) := digestRepresents
    _ = Poseidon2.hash (Hash.evalList env (interface.preimage offset)) :=
      specification
    _ = Poseidon2.hash
        (serializePreimage (publicFits := publicFits) preimage) := by
      rw [preimageRepresents]
    _ = stateHash (publicFits := publicFits) preimage := rfl

/-- Production specialization: the builder proves the exact
`OutputHolds` equation inside `StepHolds`. -/
theorem builder_implies_output_slot
    (interface : Interface) (offset : Nat) (env : Env)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (specification : SpecHolds interface offset env)
    (preimageRepresents : RepresentsPreimage interface offset env
      (nextHashPreimage (setup relation ajtai vk) input output))
    (digestRepresents : RepresentsDigest interface offset env output.x) :
    OutputHolds (setup relation ajtai vk) (machine publicFits F) input output := by
  exact builder_implies_digest interface offset env
    (nextHashPreimage (setup relation ajtai vk) input output) output.x
    specification preimageRepresents digestRepresents

end Relation

end NightstreamFPrime.Lifecycle.OutputHash
