import NightstreamFPrime.Lifecycle.Relation

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

def StatementValid (statement : TerminalStatement AppState) : Prop :=
  statement.iteration < goldilocksModulus ∧
    statement.z0.length = Application.stateWordCount ∧
    statement.zi.length = Application.stateWordCount

def ProofCanonical :
    ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits) → Prop
  | .bottom => True
  | .recursive payload => ∀ slot, ChildrenCanonical (payload.running slot)

noncomputable def HoldsFor
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (application : Application.Program)
    (statement : TerminalStatement AppState)
    (proof : ProofEnvelope
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  StatementValid statement ∧ ProofCanonical proof ∧
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
      StatementValid statement ∧ statement.iteration = 0 ∧ statement.zi = statement.z0 :=
  ⟨fun accepted => ⟨accepted.1, accepted.2.2⟩,
    fun accepted => ⟨accepted.1, trivial, accepted.2⟩⟩

theorem rejects_counter_shift
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (application : Application.Program)
    (statement : TerminalStatement AppState)
    (proof : ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ¬ HoldsFor relation ajtai vk application
      { statement with iteration := statement.iteration + goldilocksModulus } proof := by
  intro accepted
  have bound := accepted.1.1
  dsimp only at bound
  omega

theorem rejects_wrong_initial_state_length
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (application : Application.Program)
    (statement : TerminalStatement AppState)
    (proof : ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
    (wrong : statement.z0.length ≠ Application.stateWordCount) :
    ¬ HoldsFor relation ajtai vk application statement proof := by
  exact fun accepted => wrong accepted.1.2.1

theorem rejects_wrong_current_state_length
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (application : Application.Program)
    (statement : TerminalStatement AppState)
    (proof : ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits))
    (wrong : statement.zi.length ≠ Application.stateWordCount) :
    ¬ HoldsFor relation ajtai vk application statement proof := by
  exact fun accepted => wrong accepted.1.2.2

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
      StatementValid statement ∧ (∀ slot, ChildrenCanonical (payload.running slot)) ∧
        RecursiveTerminalTransition (setup relation ajtai vk)
          (machineFor publicFits application) (relations relation ajtai)
          statement payload := by
  rfl

end

end NightstreamFPrime.Lifecycle.Stage1.Terminal
