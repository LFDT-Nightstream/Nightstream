import Nightstream.Implementation.Encoding.FPrime
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.PaperFreshLinkBoundary

/-!
Contract: typed plain-profile realization of the paper fresh-public equality.

Owns:
- the executable link check over the canonical 257-coordinate logical public
  input `[1 | enc_inst(digest)]`;
- exact equivalence of that check with equality to the independently defined
  canonical encoder;
- the positive equality factorization missing from the generic native
  lifecycle callback.

Does not own: Rust-source refinement, raw vector decoding, the physical
257-to-270 carrier completion, Nebula suffixes, R1CS rows, or proof that a
native boundary receipt uses this check.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLink

open Nightstream.Implementation.Encoding.FPrime
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.PaperFreshLinkBoundary

/-- Compare the two paper-owned coordinates of the logical F-prime public
input without accepting a digest or Boolean result from the caller. -/
def check (digest : Digest) (fresh : PublicInput) : Bool :=
  decide (fresh.one = 1) &&
    decide (forall lane, fresh.body lane = encodeEncInst digest lane)

/-- The typed check accepts exactly the canonical public-input encoder. -/
theorem check_eq_true_iff
    (digest : Digest)
    (fresh : PublicInput) :
    check digest fresh = true <->
      fresh = encodePublicInput digest := by
  constructor
  · intro accepted
    simp only [check, Bool.and_eq_true, decide_eq_true_eq] at accepted
    cases fresh with
    | mk one body =>
        simp only [encodePublicInput] at accepted |- 
        have bodyEqual : body = encodeEncInst digest :=
          funext accepted.2
        rw [accepted.1, bodyEqual]
  · intro equal
    subst fresh
    simp [check, encodePublicInput]

/-- Positive plain-profile discharge of the current-interface obstruction:
the actual typed public value and canonical instance encoder generate the
link by equality. -/
theorem equalityFactorization :
    EqualityFactorization check id encodePublicInput := by
  intro digest fresh
  exact check_eq_true_iff digest fresh

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLink
