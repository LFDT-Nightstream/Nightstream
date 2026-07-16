import Mathlib.Algebra.Polynomial.Eval.Defs

/-!
Owns: the deterministic soundness boundary of one sampled Pi_RLC polynomial
identity.

Does not own: transcript derivation of `beta`, a root-counting probability
bound, or any Rust/R1CS row correspondence.

Emits constraints: no.

Authority boundary: both polynomials and `beta` are supplied values. Equality
at one point establishes polynomial equality only outside the explicitly named
nonzero-difference root event.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `ProjectionEvaluationAccepted` | `identities.*.final_limb_checks` | Both sides agree at `beta` | Supplied commutative-ring polynomials and point | No |
| `ProjectionBadRoot` | probabilistic security boundary | The nonzero difference vanishes at `beta` | Same supplied values | No |
| `projectionEvaluationAccepted_implies_exact_or_badRoot` | `identities.*` | Acceptance implies exact polynomial equality or the named bad-root event | No Fiat-Shamir or probability assumption | No |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

open Polynomial

universe u

/-- The exact equality enforced by a one-point projection check. -/
def ProjectionEvaluationAccepted
    {K : Type u} [CommRing K]
    (lhs rhs : Polynomial K) (beta : K) : Prop :=
  lhs.eval beta = rhs.eval beta

/-- The sole deterministic failure branch of one-point projection: the
polynomials differ, but their nonzero difference vanishes at `beta`. -/
structure ProjectionBadRoot
    {K : Type u} [CommRing K]
    (lhs rhs : Polynomial K) (beta : K) : Prop where
  difference_ne_zero : lhs - rhs ≠ 0
  root : (lhs - rhs).IsRoot beta

/-- A sampled evaluation equality is either exact polynomial equality or an
explicit root of the nonzero difference polynomial. -/
theorem projectionEvaluationAccepted_implies_exact_or_badRoot
    {K : Type u} [CommRing K]
    (lhs rhs : Polynomial K) (beta : K)
    (accepted : ProjectionEvaluationAccepted lhs rhs beta) :
    lhs = rhs ∨ ProjectionBadRoot lhs rhs beta := by
  by_cases exact : lhs = rhs
  · exact Or.inl exact
  · right
    refine ⟨sub_ne_zero.mpr exact, ?_⟩
    simpa [Polynomial.IsRoot, Polynomial.eval_sub] using sub_eq_zero.mpr accepted

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
