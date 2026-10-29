import NightstreamFPrime.Layout.Stage1.PiCCSSecurity
import NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.NifsRealSuccess

open scoped BigOperators
attribute [local instance] Classical.propDecidable

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

structure RealOutput (relation : ProductionKey.LogicalRelation logicalWidth publicFits) where
  prior : Lifecycle.HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)
  proof : PaperNonInteractive.Proof K PaperAlgebra.Commitment productionShape
    (ProductionKey.degreeBound relation)
  children : Fin productionShape.runningCount →
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)

variable
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (contextDigest : KeyDigest)

def RealSuccess
    (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Option (RealOutput relation) → Prop
  | none => False
  | some output =>
      let key := ProductionKey.key relation ajtai
      Layout.Stage1.PiCCSSecurity.PriorLink output.prior running fresh contextDigest ∧
      ∃ result attempt,
        PaperNonInteractive.verify key running fresh output.proof = some result ∧
        key.piDecAttempt running fresh output.proof = some attempt ∧
        ∀ child, CE.Holds key.piRlcSemantics key.params
          (PiDEC.OutputWitnessConsumer.runningStatement key result child) (output.children child)

theorem realSuccess_implies_exact_children
    (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (output : RealOutput relation) (success : RealSuccess relation ajtai contextDigest running fresh (some output)) :
    let key := ProductionKey.key relation ajtai
    ∃ result attempt,
      PaperNonInteractive.verify key running fresh output.proof = some result ∧
      key.piDecAttempt running fresh output.proof = some attempt ∧
      ∀ child, CE.Holds key.piRlcSemantics key.params
        (PiDEC.PaperVerifier.children key.piDecPublicInputSplit attempt child)
        (output.children (Fin.cast key.outputCount_eq child)) := by
  dsimp only
  rcases success with ⟨_, result, attempt, accepted, attemptEq, valid⟩
  refine ⟨result, attempt, accepted, attemptEq, ?_⟩
  intro child
  rw [← PiDEC.OutputWitnessConsumer.runningStatement_eq_child
    (ProductionKey.key relation ajtai) running fresh output.proof result attempt attemptEq accepted child]
  exact valid (Fin.cast (ProductionKey.key relation ajtai).outputCount_eq child)

end NightstreamFPrime.Export.Stage1.NifsRealSuccess
