import Mathlib.Analysis.SpecificLimits.Basic
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw

open NightstreamFPrime.Spec
open PaperForkExtraction

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params)

abbrev Challenge := {challenge : Scalar // algebra.challengeValid challenge}

def scalarVector (vector : Fin arity.total → Challenge algebra) : Fin arity.total → Scalar :=
  fun index => (vector index).val

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw
