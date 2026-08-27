import NightstreamFPrime.Lifecycle.OutputHash
import NightstreamFPrime.Lifecycle.PriorStateHash

/-!
Owns the logical composition of the two pilot phases. The prior-state child is
placed first and the output-hash child starts at its exact final offset.
Physical row concatenation belongs to `Layout/`.
-/

namespace NightstreamFPrime.Lifecycle.Pilot

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle.PaperAlgebra

structure Interface where
  prior : PriorStateHash.Interface
  output : OutputHash.Interface

def priorCircuit (interface : Interface) : FormalCircuit :=
  PriorStateHash.circuit interface.prior

def outputCircuit (interface : Interface) : FormalCircuit :=
  OutputHash.circuit interface.output

def outputOffset (interface : Interface) (offset : Nat) : Nat :=
  offset + PriorStateHash.logicalPrivateCount interface.prior offset

theorem priorCircuit_localLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (priorCircuit interface).main offset) =
      PriorStateHash.logicalPrivateCount interface.prior offset :=
  PriorStateHash.circuit_localLength interface.prior offset

theorem outputCircuit_localLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (outputCircuit interface).main offset) =
      OutputHash.hashLength interface.output offset :=
  OutputHash.circuit_localLength interface.output offset

def Assumptions (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  PriorStateHash.Assumptions interface.prior offset env ∧
    OutputHash.Assumptions interface.output (outputOffset interface offset) env

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  PriorStateHash.SpecHolds interface.prior offset env ∧
    OutputHash.SpecHolds interface.output (outputOffset interface offset) env

theorem phase_soundness (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (priorRows : holds env
      (Circuit.ops (priorCircuit interface).main offset))
    (outputRows : holds env
      (Circuit.ops (outputCircuit interface).main
        (outputOffset interface offset))) :
    SpecHolds interface offset env := by
  exact ⟨
    (priorCircuit interface).soundness env offset assumptions.1 priorRows,
    (outputCircuit interface).soundness env (outputOffset interface offset)
      assumptions.2 outputRows⟩

section Relation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Base-case convention for the pilot. At iteration zero, Construction 2
does not consume the prior-state fresh-instance binding. The base branch still
consumes the output-hash slot; its application and default-running slots belong
to their later builders. -/
theorem base_branch_output_slot
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (base : BaseHolds (setup relation ajtai vk) (machine publicFits F)
      functionIndex input output) :
    OutputHolds (setup relation ajtai vk) (machine publicFits F)
      input output :=
  base.outputHash

/-- Both logical pilot builders imply their exact fields of the production
recursive relation. -/
theorem builders_imply_hash_slots
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
    (priorPreimageRepresents : PriorStateHash.RepresentsPreimage
      interface.prior offset env
      (priorHashPreimage (setup relation ajtai vk) input))
    (priorPublicRepresents : PriorStateHash.RepresentsPublicInput
      interface.prior offset env ((machine publicFits F).freshPublic input.fresh))
    (outputPreimageRepresents : OutputHash.RepresentsPreimage interface.output
      (outputOffset interface offset) env
      (nextHashPreimage (setup relation ajtai vk) input output))
    (outputDigestRepresents : OutputHash.RepresentsDigest interface.output
      (outputOffset interface offset) env output.x) :
    (machine publicFits F).freshPublic input.fresh =
        (machine publicFits F).encodeInstance
          ((machine publicFits F).hash
            (priorHashPreimage (setup relation ajtai vk) input)) ∧
      OutputHolds (setup relation ajtai vk) (machine publicFits F) input output := by
  exact ⟨
    PriorStateHash.builder_implies_recursive_slot interface.prior offset env
      relation ajtai vk F input specification.1 priorPreimageRepresents
      priorPublicRepresents,
    OutputHash.builder_implies_output_slot interface.output
      (outputOffset interface offset) env relation ajtai vk F input output
      specification.2 outputPreimageRepresents outputDigestRepresents⟩

end Relation

end NightstreamFPrime.Lifecycle.Pilot
