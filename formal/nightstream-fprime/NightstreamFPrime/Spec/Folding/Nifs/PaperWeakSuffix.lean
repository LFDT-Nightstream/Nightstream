import NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-! SuperNeo v1.2 Appendix B.1, B.3 and B.4. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakSuffix

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

structure Reply (Assignment Evaluation Commitment : Type*) (params : GlobalParams) where
  messages : Fin params.k → PiDEC.PaperVerifier.ChildMessage Evaluation Commitment
  assignments : Fin params.k → Assignment

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (rlc : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)

def attempt (vector : Fin arity.total → Challenge rlc)
    (reply : Reply Assignment Evaluation Commitment params) :
    PiDEC.PaperVerifier.Attempt Structure PublicInput Point Evaluation Commitment params where
  parent := PiRLC.combinedOutput rlc batch.system batch.point batch.inputs (scalarVector rlc vector)
  messages := reply.messages

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakSuffix
