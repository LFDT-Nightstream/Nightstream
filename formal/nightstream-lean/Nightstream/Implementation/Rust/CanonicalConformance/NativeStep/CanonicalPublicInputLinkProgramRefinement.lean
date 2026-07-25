import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Generated.PublicInputLinkProgram

/-!
Contract: artifact-checked refinement of the Rust-emitted native public-link
program into the typed Lean source program.

Owns:
- exact equality of the generated six-instruction value and the canonical
  typed program;
- exact scalar-obligation cost;
- universal reduction of generated-program acceptance to the logical
  HyperNova public-input equality.

Does not own: formal semantics of compiled Rust, lifecycle call arguments,
the complete fixed-one verifier program, R1CS rows, or a whole-verifier
obligation-11 theorem.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement

open Nightstream.Implementation.Encoding.FPrime
open CanonicalPlainCarrierLink
open CanonicalPlainCarrierSource
open CanonicalPublicInputLinkProgram

abbrev generatedPlain : Program :=
  Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Generated.CanonicalPublicInputLinkProgram.plain

/-- The value emitted by Rust is definitionally the selected typed schedule. -/
theorem generated_plain_eq_canonical :
    generatedPlain = plain := by
  decide

/-- Program cost is computed from the emitted instruction data. -/
theorem generated_plain_cost :
    cost generatedPlain = 273 := by
  rw [generated_plain_eq_canonical]
  exact plain_cost

/-- The emitted program accepts exactly the source-shaped native predicate. -/
theorem generated_run_eq_sourceCheck
    (digest : Digest)
    (claim : RawClaim) :
    run generatedPlain digest carrierWidth claim =
      sourceCheck digest claim := by
  rw [generated_plain_eq_canonical]
  exact run_plain_eq_sourceCheck digest claim

/-- Concrete refinement chain for the Rust-emitted public-link program. -/
theorem generated_run_reduces_to_logicalPaperLink
    (digest : Digest)
    (raw : RawClaim) :
    run generatedPlain digest carrierWidth raw = true <->
      exists typed logical,
        check digest typed = true /\
          CanonicalPublicInputLink.check digest logical = true /\
          raw.mIn = typed.mIn /\
          raw.x = typed.x.coordinates /\
          typed = completeClaim logical := by
  rw [generated_run_eq_sourceCheck]
  exact sourceCheck_reduces_to_logicalPaperLink digest raw

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgramRefinement
