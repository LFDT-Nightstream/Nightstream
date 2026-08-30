import NightstreamFPrime.Lifecycle.Relation

/-!
Owns the concrete Stage 1 outer terminal relation.

The base proof is the unique empty constructor. A recursive proof checks the
prior public-state link, all 16 running CE openings inside the one uniform-IVC
slot, and the selected fresh CCS opening. It performs no additional NIFS fold.

Physical terminal circuits and package placement belong to later layers.
-/

namespace NightstreamFPrime.Lifecycle.Stage1.Terminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.ProductionKey

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

abbrev RunningWitness := Fin productionShape.runningCount →
  PaperAlgebra.Assignment
    (logicalWidth := logicalWidth) (publicFits := publicFits)

abbrev FreshWitness := PaperAlgebra.Assignment
  (logicalWidth := logicalWidth) (publicFits := publicFits)

abbrev ProofEnvelope := OuterTerminalProof
  (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (RunningWitness (logicalWidth := logicalWidth) (publicFits := publicFits))
  (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (FreshWitness (logicalWidth := logicalWidth) (publicFits := publicFits))
  slotCount

/-- Exact SuperNeo relation-membership checks used by the outer HyperNova
verifier. The key digest selects no data; the full relation and Ajtai key are
verifier-owned arguments of this closed constructor. -/
def relations
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    TerminalRelations KeyDigest
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (RunningWitness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (FreshWitness (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount where
  runningHolds := fun _ _ running witness =>
    ∀ index, CE.Holds (semantics ajtai) productionGlobalParams
      (runningStatement relation running index) (witness index)
  freshHolds := fun _ _ fresh witness =>
    CCS.Holds (semantics ajtai) productionGlobalParams
      (freshStatement relation fresh) witness

/-- Exact Construction-2 terminal verifier for one selected Lean application. -/
noncomputable def HoldsFor
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (application : Application.Program)
    (statement : TerminalStatement AppState)
    (proof : ProofEnvelope
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  OuterTerminalTransition (setup relation ajtai vk)
    (machineFor publicFits application) (relations relation ajtai)
    statement proof

theorem holdsFor_bottom_iff
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (application : Application.Program)
    (statement : TerminalStatement AppState) :
    HoldsFor relation ajtai vk application statement .bottom ↔
      statement.iteration = 0 ∧ statement.zi = statement.z0 := by
  rfl

/-- The relation-membership component is exactly the existing concrete
Nightstream terminal opening predicate. -/
theorem relations_iff_terminalHolds
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest)
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (runningWitness : RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : FreshWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (relations relation ajtai).runningHolds functionIndex vk
          running runningWitness ∧
        (relations relation ajtai).freshHolds functionIndex vk
          fresh freshWitness ↔
      Lifecycle.TerminalHolds relation ajtai running runningWitness
        fresh freshWitness := by
  rfl

theorem holdsFor_recursive_iff
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (application : Application.Program)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (RunningWitness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (FreshWitness (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount) :
    HoldsFor relation ajtai vk application statement (.recursive payload) ↔
      RecursiveTerminalTransition (setup relation ajtai vk)
        (machineFor publicFits application) (relations relation ajtai)
        statement payload := by
  rfl

end

end NightstreamFPrime.Lifecycle.Stage1.Terminal
