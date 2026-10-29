import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolDataRefinement

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem k_embed_strictNorm (value : F) :
    K.mul (K.mul (K.add (K.embed value) K.one) (K.embed value))
        (K.sub (K.embed value) K.one) =
      K.embed (NormRange.cubicResidual value) := by
  rw [show K.one = K.embed 1 from rfl]
  exact NormRange.embed_cubicResidual value

theorem zeroReflectingLift :
    ConcreteJointData.ZeroReflectingLift baseOps extensionOps K.embed where
  zero_iff := by
    intro value
    constructor
    · intro equal
      have component := congrArg K.c0 equal
      simpa only [K.embed, K.zero, baseOps, extensionOps] using component
    · intro equal
      subst equal
      rfl

theorem embed_strictNorm (value : F) :
    ProtocolPolynomial.strictNormResidual extensionOps (K.embed value) =
      K.embed (NormRange.cubicResidual value) := by
  unfold ProtocolPolynomial.strictNormResidual
  rw [derived_sub_eq_concrete_sub]
  exact k_embed_strictNorm value

theorem protocolLift :
    ProtocolDataRefinement.ProtocolLift baseOps extensionOps K.embed where
  toZeroReflectingLift := zeroReflectingLift
  map_one := embed_one
  map_add := embed_add
  map_mul := embed_mul
  map_strictNorm := embed_strictNorm

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
