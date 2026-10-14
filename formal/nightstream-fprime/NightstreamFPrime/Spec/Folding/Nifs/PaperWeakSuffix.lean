import NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-!
Owns the reply of one resumed PiRLC/PiDEC suffix call (SuperNeo B.1, B.3,
B.4): its final child messages and witnesses (`Reply`), and the PiDEC attempt
that the verifier forms from the fixed PiCCS batch and the call's challenge
vector (`attempt`). The parent is public; it is not part of the reply.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakSuffix

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

/-- Raw public child messages and the final witness returned by this call. -/
structure Reply (Assignment Evaluation Commitment : Type*) (params : GlobalParams) where
  messages : Fin params.k → PiDEC.PaperVerifier.ChildMessage Evaluation Commitment
  assignments : Fin params.k → Assignment

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (rlc : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)

/-- The verifier computes the public parent; it is absent from the reply. -/
def attempt (vector : Fin arity.total → Challenge rlc)
    (reply : Reply Assignment Evaluation Commitment params) :
    PiDEC.PaperVerifier.Attempt Structure PublicInput Point Evaluation Commitment params where
  parent := PiRLC.combinedOutput rlc batch.system batch.point batch.inputs (scalarVector rlc vector)
  messages := reply.messages

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakSuffix
