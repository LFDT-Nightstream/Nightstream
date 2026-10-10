import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientObject
import NightstreamFPrime.Spec.SumCheck.GoldilocksCausal

/-!
Two facts about the signed PiCCS coefficient object: negation in `K` is zero
only at zero (`neg_zero_iff`), and every α-dependent signed coefficient comes
from one CCS or norm table of the data (`negative_coefficient_is_table`).
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingProbability

open scoped BigOperators
open NightstreamFPrime.Spec SumCheck.Finite
open GoldilocksCausal ConcreteCarrier SignedCoefficientObject

attribute [local instance] Classical.propDecidable

theorem neg_zero_iff (value : K) : extensionOps.neg value = K.zero ↔ value = K.zero := by
  constructor
  · intro zero
    change extensionOps.neg value = extensionOps.zero at zero
    have inverse := extensionLaws.add_neg value
    rw [zero, extensionLaws.add_zero] at inverse
    exact inverse
  · intro zero
    rw [zero]
    exact extensionZeroLaws.neg_zero

/-- Every alpha-dependent signed coefficient comes from one of the exact CCS
or norm tables in this data object. -/
theorem negative_coefficient_is_table {shape : Shape}
    (data : SignedJointIdentity.JointData K shape)
    (polynomial : AlphaPolynomial K (canonicalAlphaBasis shape))
    (inside : Coefficient.negativeAlpha polynomial ∈ coefficients extensionOps data) :
    ∃ table : BooleanTable K shape.cubeVariables, polynomial = table.toAlphaPolynomial extensionOps := by
  simp only [coefficients, residuals, TableResidualData.toResiduals, List.map_map,
    Function.comp_def, List.mem_append, List.mem_map] at inside
  rcases inside with ⟨value, _member, equal⟩ | ⟨value, _member, equal⟩ |
    ⟨source, _member, equal⟩ | ⟨source, _member, equal⟩
  · cases equal
  · cases equal
  · exact ⟨data.ccs source, (Coefficient.negativeAlpha.inj equal).symm⟩
  · exact ⟨data.norm source, (Coefficient.negativeAlpha.inj equal).symm⟩

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingProbability
