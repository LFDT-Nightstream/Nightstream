import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Lifecycle.Stage1.Application

/-! HyperNova Construction 2. -/

namespace NightstreamFPrime.Lifecycle

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.ProductionKey

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}

noncomputable def nifsVerifier (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Verifier KeyDigest (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (degreeBound relation)) where
  verify := fun _ running fresh proof =>
    Nifs.PaperNonInteractive.verify (key relation ajtai) running fresh proof

noncomputable def setup (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) :
    Setup KeyDigest (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (degreeBound relation)) slotCount where
  verifierKeys := fun _ => vk
  nifs := nifsVerifier relation ajtai
  defaultRunning := defaultRunning

def machine (publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (F : AppState → AppWitness → AppState) :
    Machine KeyDigest Digest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount where
  control := fun _ _ => functionIndex
  step := fun _ z w => F z w
  freshPublic := fun fresh => fresh.publicInputs ⟨0, by decide⟩
  encodeInstance := encHash (publicFits := publicFits)
  hash := stateHash (publicFits := publicFits)

def machineFor (publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (application : Stage1.Application.Program) :
    Machine KeyDigest Digest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth)
        (publicFits := publicFits)) slotCount :=
  machine publicFits application.step

noncomputable def StepHolds (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount) : Prop :=
  FixedAugmentedTransition (setup relation ajtai vk) (machine publicFits F)
    functionIndex input output

noncomputable def StepHoldsFor
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (application : Stage1.Application.Program)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount) : Prop :=
  FixedAugmentedTransition (setup relation ajtai vk)
    (machineFor publicFits application) functionIndex input output

theorem stepHoldsFor_eq
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (application : Stage1.Application.Program)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount) :
    StepHoldsFor relation ajtai vk application input output =
      StepHolds relation ajtai vk application.step input output := by
  rfl

def runningStatement (relation : LogicalRelation logicalWidth publicFits)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (i : Fin productionShape.runningCount) :
    CE.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment where
  constraintSystem := PaperAlgebra.relationSource
    (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
    relation.system
  commitment := running.commitments i
  publicInput := running.publicInputs i
  point := running.point
  evaluations := #[running.evaluations i]
  stage := .fresh

def freshStatement (relation : LogicalRelation logicalWidth publicFits)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    CCS.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Commitment where
  constraintSystem := PaperAlgebra.relationSource
    (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
    relation.system
  commitment := fresh.commitments ⟨0, by decide⟩
  publicInput := fresh.publicInputs ⟨0, by decide⟩
  stage := .fresh

def TerminalHolds (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (runningWitness : Fin productionShape.runningCount →
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Prop :=
  (∀ i, CE.Holds (semantics ajtai) productionGlobalParams
    (runningStatement relation running i) (runningWitness i)) ∧
  CCS.Holds (semantics ajtai) productionGlobalParams
    (freshStatement relation fresh) freshWitness

end

end NightstreamFPrime.Lifecycle
