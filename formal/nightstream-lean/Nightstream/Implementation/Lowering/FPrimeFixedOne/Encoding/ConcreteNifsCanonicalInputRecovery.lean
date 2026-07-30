import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofRecovery
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec

/-!
Contract: reconstruct the selected running and fresh NIFS inputs from every
exact-width canonical coordinate list.

Owns: inverse constructions for the canonical running and fresh codecs.

Does not own: physical columns, proof recovery, verifier acceptance, row
constraints, Rust, or generated artifacts.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private theorem completePayloadCodec_exactWidthRecoverable
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (completePayloadCodec shape publicRingColumns verifierRows publicFits
      ).ExactWidthRecoverable := by
  unfold completePayloadCodec
  exact
    Codec.product_exactWidthRecoverable
      (commitmentCodec verifierRows)
      (Codec.product
        (publicInputCodec (ringDegree * publicRingColumns))
        (Codec.product
          (pointCodec shape.rowVariables)
          (evaluationsCodec shape.matrixCount)))
      (commitmentCodec_exactWidthRecoverable verifierRows)
      (Codec.product_exactWidthRecoverable
        (publicInputCodec (ringDegree * publicRingColumns))
        (Codec.product
          (pointCodec shape.rowVariables)
          (evaluationsCodec shape.matrixCount))
        (publicInputCodec_exactWidthRecoverable
          (ringDegree * publicRingColumns))
        (Codec.product_exactWidthRecoverable
          (pointCodec shape.rowVariables)
          (evaluationsCodec shape.matrixCount)
          (pointCodec_exactWidthRecoverable shape.rowVariables)
          (evaluationsCodec_exactWidthRecoverable shape.matrixCount)))

theorem parentPayloadCodec_exactWidthRecoverable
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (parentPayloadCodec shape publicRingColumns verifierRows publicFits
      ).ExactWidthRecoverable := by
  unfold parentPayloadCodec
  apply Codec.pullback_exactWidthRecoverable
    (completePayloadCodec shape publicRingColumns verifierRows publicFits)
    parentPayloadData parentPayloadData_injective
    (fun data => {
      commitment := data.1
      publicInput := data.2.1
      point := data.2.2.1
      evaluations := data.2.2.2
    })
  · intro data _
    rcases data with ⟨commitment, publicInput, point, evaluations⟩
    rfl
  · exact completePayloadCodec_exactWidthRecoverable
      shape publicRingColumns verifierRows publicFits

theorem runningPayloadCodec_exactWidthRecoverable
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (runningPayloadCodec shape publicRingColumns verifierRows publicFits
      ).ExactWidthRecoverable := by
  unfold runningPayloadCodec
  apply Codec.pullback_exactWidthRecoverable
    (completePayloadCodec shape publicRingColumns verifierRows publicFits)
    runningPayloadData runningPayloadData_injective
    (fun data => {
      commitment := data.1
      publicInput := data.2.1
      point := data.2.2.1
      evaluations := data.2.2.2
    })
  · intro data _
    rcases data with ⟨commitment, publicInput, point, evaluations⟩
    rfl
  · exact completePayloadCodec_exactWidthRecoverable
      shape publicRingColumns verifierRows publicFits

theorem freshCodec_exactWidthRecoverable
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (freshCodec shape publicRingColumns verifierRows publicFits
      ).ExactWidthRecoverable := by
  unfold freshCodec
  apply Codec.pullback_exactWidthRecoverable
    (Codec.product
      (commitmentCodec verifierRows)
      (publicInputCodec (ringDegree * publicRingColumns)))
    freshPayloadData freshPayloadData_injective
    (fun data => {
      commitment := data.1
      publicInput := data.2
    })
  · intro data _
    rcases data with ⟨commitment, publicInput⟩
    rfl
  · exact
      Codec.product_exactWidthRecoverable
        (commitmentCodec verifierRows)
        (publicInputCodec (ringDegree * publicRingColumns))
        (commitmentCodec_exactWidthRecoverable verifierRows)
        (publicInputCodec_exactWidthRecoverable
          (ringDegree * publicRingColumns))

theorem runningCodec_exactWidthRecoverable
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (runningCodec shape publicRingColumns verifierRows publicFits
      ).ExactWidthRecoverable := by
  unfold runningCodec
  apply Codec.pullback_exactWidthRecoverable
    (Codec.product
      (parentPayloadCodec shape publicRingColumns verifierRows publicFits)
      (Codec.finFunction productionGlobalParams.k
        (runningPayloadCodec
          shape publicRingColumns verifierRows publicFits)))
    runningData runningData_injective
    (fun data => {
      parent := data.1
      children := data.2
    })
  · intro data _
    rcases data with ⟨parent, children⟩
    rfl
  · exact
      Codec.product_exactWidthRecoverable
        (parentPayloadCodec
          shape publicRingColumns verifierRows publicFits)
        (Codec.finFunction productionGlobalParams.k
          (runningPayloadCodec
            shape publicRingColumns verifierRows publicFits))
        (parentPayloadCodec_exactWidthRecoverable
          shape publicRingColumns verifierRows publicFits)
        (Codec.finFunction_exactWidthRecoverable
          (runningPayloadCodec
            shape publicRingColumns verifierRows publicFits)
          (runningPayloadCodec_exactWidthRecoverable
            shape publicRingColumns verifierRows publicFits)
          productionGlobalParams.k)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery
