import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicCarrier
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC

/-!
Shared fixed-profile scalar recomposition bridge for production `PiDEC`.

Protocol: SuperNeo `Pi_DEC` at production radix two and fourteen children.
Phase: implementation-to-semantic weight and scalar-fold correspondence.
Constraint family: verifier-owned recomposition coefficients only; this file
emits no rows.

Owns: equality of the production and independent Lean radix weights; one
canonical scalar fold; and equality between the production list fold and that
semantic fold.

Does not own: commitment, public-input, or evaluation carriers; strict gadget
acceptance; Rust/R1CS row satisfaction; costs; or row removal.

Emits constraints: no.

Authority boundary: both weight families are verifier-computed. No witness,
digest, or caller-supplied equivalence selects a recomposition coefficient.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.refinement.weights` | implementation weight `i` equals typed `2^i` in Goldilocks | checked | `radixWeights_eq` |
| `nifs.pi_dec.verify.refinement.scalar_fold` | production list fold equals the canonical finite semantic fold | checked | `combineScalar_eq` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.Weights

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper

/-- Fixed-profile weight lists agree entry-for-entry. -/
theorem radixWeights_eq :
    radixWeights = List.ofFn PiDEC.radixWeight := by
  decide

/-- Canonical head-first scalar fold used as the common semantic target. -/
def combineScalars : {count : Nat} ->
    (Fin count -> F) -> (Fin count -> F) -> F
  | 0, _, _ => 0
  | _ + 1, weights, items =>
      weights 0 * items 0 +
        combineScalars
          (fun index => weights index.succ)
          (fun index => items index.succ)

private theorem foldr_zip_ofFn_eq_combineScalars
    {count : Nat} (weights items : Fin count -> F) :
    ((List.ofFn items).zip (List.ofFn weights)).foldr
        (fun pair suffix => pair.2 * pair.1 + suffix) 0 =
      combineScalars weights items := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [List.ofFn_succ, combineScalars, inductionHypothesis]

/-- Production scalar recomposition is exactly the independent typed
production-radix fold. -/
theorem combineScalar_eq
    (items : Fin productionGlobalParams.k -> F) :
    combineScalar items = combineScalars PiDEC.radixWeight items := by
  rw [combineScalar, radixWeights_eq,
    foldr_zip_ofFn_eq_combineScalars]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.Weights
