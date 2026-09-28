import NightstreamFPrime.Export.Stage1.PiCCSFreshPolynomial
import NightstreamFPrime.Export.Stage1.PiCCSPublicReplay
import NightstreamFPrime.Export.Stage1.PiCCSSourceImages

/-! Proof-only zero suffix for the selected fresh CCS polynomial.
The syntax has positive degree in every term; no row or matrix is evaluated. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFreshPolynomial

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- The selected coefficient lift keeps every positive production exponent
vector. This proof inspects only the syntax lift, not matrices or assignments. -/
theorem selected_terms_positive (input : PiCCSPublicReplay.Input)
    (term : CCSResidualTable.Monomial K productionShape.matrixCount)
    (member : term ∈ (PiCCSPublicReplay.verifierInput input).constraintPolynomial.terms) :
    0 < term.totalDegree := by
  change term ∈ Spec.ProductionRelation.polynomial.terms.map
    (ConstraintPolynomialLift.liftMonomial K.embed) at member
  obtain ⟨base, baseMember, rfl⟩ := List.mem_map.mp member
  change 0 < base.totalDegree
  exact Spec.ProductionRelation.SelectivePolynomial.term_totalDegree_pos base baseMember

/-- The selected production fresh kernel has an exact zero padded suffix. -/
theorem selected_ccsPolynomialWithPowers_zero (input : PiCCSPublicReplay.Input)
    (powers : Nat → K) (selector : FixedPolynomial K 1)
    (low high : ProtocolPolynomial.OutputMessage K productionShape)
    (lowZero : ∀ source matrix, low.freshMatrixImage source matrix = extensionOps.zero)
    (highZero : ∀ source matrix, high.freshMatrixImage source matrix = extensionOps.zero) :
    ccsPolynomialWithPowers extensionOps (PiCCSPublicReplay.verifierInput input)
        powers selector low high =
      FixedPolynomial.zero extensionOps.toOps
        (PiCCSPublicReplay.verifierInput input).constraintPolynomial.canonicalEqualityGatedDegreeBound := by
  exact ccsPolynomialWithPowers_zero extensionOps extensionLaws _ powers selector low high
    (selected_terms_positive input) lowZero highZero

/-- Both endpoints are exactly zero after the final active pair. This keeps
the mixed real/padded pair when the program has an odd number of rows. -/
theorem freshMatrixImage?_zero_of_pair_beyond
    (program : NightstreamFPrime.Layout.MatrixProgram.Program)
    (sourceRow : Nat → Option NightstreamFPrime.Layout.R1CS.Row)
    (assignment : Phi81Relation.Assignment PiCCSSourceImages.shape)
    (suffix : BooleanVertex 27)
    (beyond : program.rowCount ≤ 2 * NumericBooleanDomain.index suffix) (bit : Bool) :
    PiCCSSourceImages.freshMatrixImage? program sourceRow assignment
        (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) bit suffix) =
      some (Vector.replicate ProductionRelation.matrixCount (0 : F)) := by
  have outside : ¬ NumericBooleanDomain.index
      (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) bit suffix) <
        program.rowCount := by
    cases bit
    · change ¬ 0 + 2 * NumericBooleanDomain.index suffix < program.rowCount
      omega
    · change ¬ 1 + 2 * NumericBooleanDomain.index suffix < program.rowCount
      omega
  simp only [PiCCSSourceImages.freshMatrixImage?, PiCCSSourceImages.rowValues?, if_neg outside]

end NightstreamFPrime.Export.Stage1.PiCCSFreshPolynomial
