import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

namespace NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

open NightstreamFPrime.Spec
open PaperForkExtraction PaperForkAlgebra

universe uValue uScalar uAssignment uStructure uPublicInput uPoint uEvaluation uCommitment

structure Result (Value : Type uValue) where
  value : Value
  work : Nat

end NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
