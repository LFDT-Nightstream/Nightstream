import Nightstream.Implementation.Nebula.NIFS.PiDEC.Rows

/-!
Contract: arithmetic bridge from V2 PiDEC sparse-row recomposition to the
typed Goldilocks and quadratic-extension folds.

Owns canonical wire decoding, the finite field fold used by PiDEC, its exact
agreement with sparse R1CS linear-combination evaluation, and coordinate
projections of the typed commitment and evaluation recomposition functions.

Does not own column placement, row inclusion in a larger artifact, or paper
verifier acceptance.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductPiDecLinearCombination

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.Nebula.ProductPiDecRows
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

/-- Interpret one canonical R1CS wire as a Goldilocks element. -/
def fieldAt (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (column : Nat) : F :=
  ⟨assignment column, by
    simpa [goldilocksP, goldilocksModulus] using canonical column⟩

/-- Head-first finite base-field fold, in the exact order used by PiDEC. -/
def combineFields : {count : Nat} ->
    (Fin count -> F) -> (Fin count -> F) -> F
  | 0, _, _ => 0
  | _ + 1, weights, values =>
      weights 0 * values 0 +
        combineFields
          (fun index => weights index.succ)
          (fun index => values index.succ)

/-- Sparse R1CS evaluation of a typed finite coordinate family is exactly the
same Goldilocks element as the typed head-first field fold. -/
theorem lcEval_ofFn_zip
    {count : Nat} (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (columns : Fin count -> Nat) (weights : Fin count -> F) :
    (⟨lcEval assignment
        ((List.ofFn columns).zip
          (List.ofFn fun index => (weights index).val)), by
        unfold lcEval
        simpa [goldilocksP, goldilocksModulus] using
          Nat.mod_lt
            (List.foldl
              (fun accumulated term => accumulated + term.2 * assignment term.1)
              0
              ((List.ofFn columns).zip
                (List.ofFn fun index => (weights index).val)))
            (by decide : 0 < goldilocksP)⟩ : F) =
      combineFields weights (fun index => fieldAt assignment canonical (columns index)) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      apply Fin.ext
      simp only [List.ofFn_succ, List.zip_cons_cons, lcEval_eq_raw_mod,
        rawLcEval, combineFields, fieldAt, Fin.val_add, Fin.val_mul]
      have tail := congrArg Fin.val
        (inductionHypothesis
          (fun index => columns index.succ)
          (fun index => weights index.succ))
      simp only at tail
      rw [lcEval_eq_raw_mod] at tail
      rw [Nat.add_mod, Nat.mul_mod]
      rw [tail]
      have weightLt : (weights 0).val < goldilocksP := by
        simp [goldilocksP, goldilocksModulus]
      rw [Nat.mod_eq_of_lt weightLt]
      rw [Nat.mod_eq_of_lt (canonical (columns 0))]
      rfl

/-- One satisfied recomposition equation gives the corresponding typed field
equation. The only assumptions are canonical wire values and the exact
independent row meaning. -/
theorem recomposes_field
    {count : Nat} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (parent : Nat) (columns : Fin count -> Nat) (weights : Fin count -> F)
    (recomposes : Recomposes assignment parent
      (List.ofFn columns) (List.ofFn fun index => (weights index).val)) :
    fieldAt assignment canonical parent =
      combineFields weights
        (fun index => fieldAt assignment canonical (columns index)) := by
  rw [show fieldAt assignment canonical parent =
      (⟨lcEval assignment
        ((List.ofFn columns).zip
          (List.ofFn fun index => (weights index).val)), by
          unfold lcEval
          simpa [goldilocksP, goldilocksModulus] using
            Nat.mod_lt
              (List.foldl
                (fun accumulated term =>
                  accumulated + term.2 * assignment term.1)
                0
                ((List.ofFn columns).zip
                  (List.ofFn fun index => (weights index).val)))
              (by decide : 0 < goldilocksP)⟩ : F) by
    apply Fin.ext
    simpa [fieldAt, Recomposes] using recomposes]
  exact lcEval_ofFn_zip assignment canonical columns weights

/-! ## Agreement with the typed product algebra -/

/-- Typed Ajtai commitment recomposition is the identical coordinatewise
field fold. -/
theorem combineCommitments_coordinate
    {count verifierRows : Nat}
    (weights : Fin count -> F)
    (values : Fin count ->
      Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.Value
        verifierRows)
    (row : Fin verifierRows) (lane : Fin ringDegree) :
    (Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.combineCommitments
      weights values row lane) =
      combineFields weights (fun index => values index row lane) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.combineCommitments,
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd,
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.commitmentScale,
        Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.ringFScale,
        ringFAdd, combineFields]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => values index.succ)]

/-- Product-bundle PiDEC uses the same fold for all four components, all 18
Ajtai rows, and all 54 ring coefficients. -/
theorem recomposeBundles_coordinate
    (values : ProductPiDecRows.ChildIndex ->
      ProductCommitmentAlgebra.BundleValue)
    (component : Nightstream.Protocol.Nebula.CommitmentBundle.Component)
    (row : Fin ProductCommitmentAlgebra.Rank) (lane : Fin ringDegree) :
    ProductCommitmentAlgebra.recomposeBundles values component row lane =
      combineFields PiDEC.radixWeight
        (fun child => values child component row lane) := by
  unfold ProductCommitmentAlgebra.recomposeBundles
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment.recomposeCommitment
  exact combineCommitments_coordinate PiDEC.radixWeight
    (fun child => values child component) row lane

/-- The low extension limb of evaluation recomposition is the identical
base-field fold. -/
theorem combineEvaluations_c0
    {count : Nat} (weights : Fin count -> F)
    (values : Fin count -> RingK) (lane : Fin ringDegree) :
    (BaseLinear.combineEvaluations weights values lane).c0 =
      combineFields weights (fun index => (values index lane).c0) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [BaseLinear.combineEvaluations, BaseLinear.evaluationAdd,
        BaseLinear.evaluationScale, K.add, K.mul, K.embed, combineFields]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => values index.succ)]
      simp

/-- The high extension limb of evaluation recomposition is the identical
base-field fold. -/
theorem combineEvaluations_c1
    {count : Nat} (weights : Fin count -> F)
    (values : Fin count -> RingK) (lane : Fin ringDegree) :
    (BaseLinear.combineEvaluations weights values lane).c1 =
      combineFields weights (fun index => (values index lane).c1) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [BaseLinear.combineEvaluations, BaseLinear.evaluationAdd,
        BaseLinear.evaluationScale, K.add, K.mul, K.embed, combineFields]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => values index.succ)]
      simp

end Nightstream.Implementation.Nebula.ProductPiDecLinearCombination
