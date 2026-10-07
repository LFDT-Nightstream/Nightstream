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

/-- A value and its declared mathematical clock. A program adds its callees'
clocks and its own explicit charges. Function application, allocation,
representation conversion and field arithmetic count only where a charge
includes them. No Lean/Rust runtime or machine-operation count follows without
a separate execution-model refinement. -/
structure Result (Value : Type uValue) where
  value : Value
  work : Nat

end NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
