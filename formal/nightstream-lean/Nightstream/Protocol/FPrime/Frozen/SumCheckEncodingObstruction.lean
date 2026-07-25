import Nightstream.SuperNeo.SumCheck.FixedPhase
import Nightstream.SuperNeo.SumCheck.VerifierCertificate

/-!
Kernel obstruction to identifying typed fixed-width SumCheck acceptance with
the current canonical variable-length certificate checker.

Owns: one degree-one padded-zero polynomial accepted by the fixed-width chain
checker and rejected by the raw checker solely because the latter requires
its coefficient list to be canonical.

Does not own: a choice of production transcript encoding, a paper theorem,
Fiat--Shamir, `Pi_CCS` soundness, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

The two messages represent the same zero polynomial.  Therefore a bridge
between the current interactive composition and the frozen NIFS must either
prove a transcript-preserving canonicalization theorem or use one common
paper-owned polynomial-message relation; replay equality alone is
insufficient.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.SumCheckEncodingObstruction

open Nightstream.SuperNeo.SumCheck.Finite

/-- A one-element carrier makes the rejection depend only on message shape,
not on field arithmetic. -/
def ops : Ops Unit where
  zero := ()
  one := ()
  add := fun _ _ => ()
  mul := fun _ _ => ()

/-- Degree-one storage of the zero polynomial, including one redundant high
zero slot. -/
def paddedZero : FixedPolynomial Unit 1 where
  coefficients := [(), ()]
  coefficients_length := by decide

def rawCertificate : Certificate Unit where
  rounds := [paddedZero.toMessage]

/-- Static width makes the recurrence and terminal equation sufficient. -/
theorem fixedWidth_accepts :
    FixedPhase.checkChain ops () [paddedZero] [()] () = true := by
  decide

/-- The raw checker rejects the same coefficient vector because its final
coefficient equals `ops.zero`. -/
theorem canonicalRaw_rejects :
    check ops 1 () [()] () rawCertificate = false := by
  decide

/-- Headline countermodel: current fixed-width and canonical-raw acceptance
cannot be identified without an additional encoding theorem or a shared
message relation. -/
theorem fixed_width_acceptance_is_not_canonical_raw_acceptance :
    FixedPhase.checkChain ops () [paddedZero] [()] () = true /\
      check ops 1 () [()] () rawCertificate = false := by
  exact ⟨fixedWidth_accepts, canonicalRaw_rejects⟩

end Nightstream.Protocol.FPrime.Frozen.SumCheckEncodingObstruction
