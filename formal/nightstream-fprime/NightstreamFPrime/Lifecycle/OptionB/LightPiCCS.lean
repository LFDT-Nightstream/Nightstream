import NightstreamFPrime.Lifecycle.OptionB.Keys
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim

/-! The light Π_CCS leaves (CW-2.2), one per production leaf made generic in the shape. Each
bridge gives a value of the production light key's Π_CCS statement from the leaf's rows. -/

namespace NightstreamFPrime.Lifecycle.OptionB.LightPiCCS

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle.PiCCS.v1_2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The light Π_CCS claimed sum `T` (SuperNeo v1.2 Section 7.3, Step 2) over the single running
claim: 54 + 216 = 270 claimed coefficients, against 4,320 today
(`InitialClaim.coefficientExprs_length`). -/
theorem initial_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (interface : InitialClaim.ShapeInterface LightShape) (offset : Nat) (env : Env)
    (gamma : K) (gamma_eq : (interface.gamma offset).eval env = gamma)
    (eval_K_eq : ∀ coordinate, (interface.eval_K offset coordinate).eval env =
      (((keys relation ajtai).light.statement running fresh).verifierInput
        (keys relation ajtai).light.lift).claimedPadCoefficient coordinate)
    (eval_A_eq : ∀ coordinate, (interface.eval_A offset coordinate).eval env =
      (((keys relation ajtai).light.statement running fresh).verifierInput
        (keys relation ajtai).light.lift).claimedMatrixCoefficient coordinate)
    (specification : InitialClaim.SpecHolds interface offset env) :
    (InitialClaim.output interface offset).eval env =
      (((keys relation ajtai).light.statement running fresh).verifierInput
        (keys relation ajtai).light.lift).initial extensionOps gamma :=
  InitialClaim.spec_implies_initial interface offset env _ gamma gamma_eq eval_K_eq eval_A_eq
    specification

theorem light_coefficientCount (interface : InitialClaim.ShapeInterface LightShape)
    (offset : Nat) : (InitialClaim.coefficientExprs interface offset).length = 270 := by
  simp only [InitialClaim.coefficientExprs, List.length_append, List.length_map,
    canonicalPadCoordinates_length, canonicalMatrixCoordinates_length]
  rfl

end NightstreamFPrime.Lifecycle.OptionB.LightPiCCS
