import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

/-!
Owns `Result`: a value together with its declared mathematical clock. Charged
programs return it. The clock is a declared count, not a measured Lean or Rust
runtime.
-/

namespace NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

open NightstreamFPrime.Spec
open PaperForkExtraction PaperForkAlgebra

universe uValue uScalar uAssignment uStructure uPublicInput uPoint uEvaluation uCommitment

structure Result (Value : Type uValue) where
  value : Value
  work : Nat

end NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
