import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Norm.Finite
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC

/-! SuperNeo v1.2 Definition 22. -/

namespace NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Norm

export Centered
  (centeredMagnitude_add_le centeredMagnitude_sub_le
    embedCoefficient_mul_le_two)

export Product
  (supportActive supportCount totalSupport totalSupport_le_two_degrees
    rawMulCoeffF_le_support ringFMul_le_expansion)

export Finite
  (act_coordinate_le_expansion combineAssignments_le production_total_bound
    relation_norm_growth)

end NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Norm
