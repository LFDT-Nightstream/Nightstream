import NightstreamFPrime.Lifecycle.Relation
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Key

/-! HyperNova's fixed augmented step with the wide-sampler NIFS key, which the
selected production package uses. The application, state hashing, public
encoding and default accumulator are the existing ones. -/

namespace NightstreamFPrime.Lifecycle.Stage1.Wide.Relation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

noncomputable def nifsVerifier (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Verifier KeyDigest (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) (Proof (ProductionKey.degreeBound relation)) where
  verify := fun _ running fresh proof =>
    Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai) running fresh proof

noncomputable def setup (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) (vk : KeyDigest) :
    Setup KeyDigest (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount where
  verifierKeys := fun _ => vk
  nifs := nifsVerifier relation ajtai
  defaultRunning := defaultRunning

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) (vk : KeyDigest)
  (application : Stage1.Application.Program)
  (input : Input KeyDigest AppState AppWitness
    (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (Proof (ProductionKey.degreeBound relation)) slotCount)
  (output : Output Digest AppState
    (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)

noncomputable def StepHoldsFor : Prop :=
  FixedAugmentedTransition (setup relation ajtai vk)
    (Lifecycle.machineFor publicFits application) functionIndex input output

theorem priorPreimage_unchanged :
    priorHashPreimage (setup relation ajtai vk) input =
      priorHashPreimage (Lifecycle.setup relation ajtai vk) input := by
  unfold priorHashPreimage
  rfl

theorem nextPreimage_unchanged :
    nextHashPreimage (setup relation ajtai vk) input output =
      nextHashPreimage (Lifecycle.setup relation ajtai vk) input output := by
  unfold nextHashPreimage
  rfl

end NightstreamFPrime.Lifecycle.Stage1.Wide.Relation
