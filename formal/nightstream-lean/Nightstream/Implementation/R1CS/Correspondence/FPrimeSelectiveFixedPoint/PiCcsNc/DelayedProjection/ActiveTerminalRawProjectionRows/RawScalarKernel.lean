import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC

/-!
Exact scalar kernel for the fourteen raw-child radix terms.

This leaf evaluates only fourteen proof-free `(column, coefficient)` terms.
It never constructs a production witness table, projection row family, or
block-domain list.

Assurance tier: model-level arithmetic kernel.

Owns: exact base-field and quadratic-extension interpretation of the
fourteen-term radix-weighted raw-child scalar expression used by the terminal
projection rows.

Does not own: row indices, compiler schedules, witness-column provenance,
terminal CE, commitment binding, trace composition, or production costs.

Emits constraints: no; proof-only scalar algebra.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.terminal_rows.scalar.residue` | raw natural-number linear-combination evaluation agrees with base-field evaluation | derived algebra |
| `f_prime.pi_ccs_nc.delayed.terminal_rows.scalar.radix` | fourteen generated child coefficients equal the semantic radix progression | derived algebra |
| `f_prime.pi_ccs_nc.delayed.terminal_rows.scalar.recompose` | the row scalar equals the corresponding coordinate of raw-child recomposition | derived algebra |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjectionRows.RawScalarKernel

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance

private theorem residue_add (left right : Nat) :
    residue (left + right) = residue left + residue right := by
  apply Fin.ext
  exact Nat.add_mod left right goldilocksP

private theorem residue_mul (left right : Nat) :
    residue (left * right) = residue left * residue right := by
  apply Fin.ext
  exact Nat.mul_mod left right goldilocksP

private theorem residue_mod (value : Nat) :
    residue (value % goldilocksP) = residue value := by
  apply Fin.ext
  simp [residue]

private theorem residue_rawLcEval (assignment : Nat -> Nat) :
    forall terms : List (Nat × Nat),
      residue (Program.rawLcEval assignment terms) =
        terms.foldr (fun term suffix =>
          residue term.2 * residue (assignment term.1) + suffix) 0 := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [Program.rawLcEval, List.foldr_cons]
      rw [residue_add, residue_mul, inductionHypothesis]

private theorem residue_lcEval
    (assignment : Nat -> Nat) (terms : List (Nat × Nat)) :
    residue (lcEval assignment terms) =
      terms.foldr (fun term suffix =>
        residue term.2 * residue (assignment term.1) + suffix) 0 := by
  rw [Program.lcEval_eq_raw_mod, residue_mod]
  exact residue_rawLcEval assignment terms

private theorem toConcreteField_foldr
    (assignment : Nat -> Nat) (terms : List (Nat × Nat)) :
    toConcreteField
        (terms.foldr (fun term suffix =>
          residue term.2 * residue (assignment term.1) + suffix) 0) =
      terms.foldr (fun term suffix =>
        toConcreteField (residue term.2) *
          toConcreteField (residue (assignment term.1)) + suffix) 0 := by
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, toConcreteField_add, toConcreteField_mul,
        inductionHypothesis]

private def combineScalars : {count : Nat} ->
    (Fin count -> Concrete.F) -> (Fin count -> Concrete.F) -> Concrete.F
  | 0, _, _ => 0
  | _ + 1, weights, values =>
      weights 0 * values 0 +
        combineScalars
          (fun index => weights index.succ)
          (fun index => values index.succ)

private theorem foldr_ofFn_eq_combineScalars {count : Nat}
    (weights values : Fin count -> Concrete.F) :
    ((List.ofFn fun index => (weights index, values index))).foldr
        (fun pair suffix => pair.1 * pair.2 + suffix) 0 =
      combineScalars weights values := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [List.ofFn_succ, combineScalars, inductionHypothesis]

private theorem rawCombineAssignments_apply
    {columns count : Nat}
    (weights : Fin count -> Concrete.F)
    (assignments : Fin count ->
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.Assignment
        Concrete.F columns)
    (column : Fin columns) :
    BaseLinear.Raw.combineAssignments weights assignments column =
      combineScalars weights (fun index => assignments index column) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [BaseLinear.Raw.combineAssignments,
        BaseLinear.Raw.assignmentAdd, BaseLinear.Raw.assignmentScale,
        combineScalars]
      rw [inductionHypothesis]

private theorem coefficient_eq_radixWeight
    (child : Fin productionGlobalParams.k) :
    toConcreteField
        (residue
          (productionGlobalParams.b ^ child.val % goldilocksP)) =
      PiDEC.radixWeight child := by
  apply Fin.ext
  simp [toConcreteField, residue, PiDEC.radixWeight, goldilocksP,
    goldilocksModulus]

/-- A compiler linear combination with the production powers-of-two
coefficients is exactly the independent PiDEC scalar recomposition. -/
theorem lcEval_radixTerms_eq_recomposeScalar
    {width : Nat}
    (assignment : Nat -> Nat)
    (columns : Fin productionGlobalParams.k -> Nat)
    (childAssignments : Fin productionGlobalParams.k ->
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.Assignment
        Concrete.F width)
    (column : Fin width)
    (columnsMatch : forall child,
      assignment (columns child) = (childAssignments child column).val) :
    toConcreteField
        (residue
          (lcEval assignment
            (List.ofFn fun child : Fin productionGlobalParams.k =>
              (columns child,
                productionGlobalParams.b ^ child.val % goldilocksP)))) =
      BaseLinear.Raw.combineAssignments PiDEC.radixWeight
        childAssignments column := by
  rw [residue_lcEval, toConcreteField_foldr]
  calc
    (List.ofFn fun child : Fin productionGlobalParams.k =>
        (columns child,
          productionGlobalParams.b ^ child.val % goldilocksP)).foldr
        (fun term suffix =>
          toConcreteField (residue term.2) *
            toConcreteField (residue (assignment term.1)) + suffix) 0 =
      combineScalars
        (fun child => toConcreteField
          (residue
            (productionGlobalParams.b ^ child.val % goldilocksP)))
        (fun child => toConcreteField
          (residue (assignment (columns child)))) := by
      simpa using foldr_ofFn_eq_combineScalars
        (fun child : Fin productionGlobalParams.k => toConcreteField
          (residue
            (productionGlobalParams.b ^ child.val % goldilocksP)))
        (fun child : Fin productionGlobalParams.k => toConcreteField
          (residue (assignment (columns child))))
    _ = combineScalars PiDEC.radixWeight
        (fun child => childAssignments child column) := by
      congr 1
      · funext child
        exact coefficient_eq_radixWeight child
      · funext child
        rw [columnsMatch child]
        apply Fin.ext
        simp [toConcreteField, residue, goldilocksP, goldilocksModulus]
    _ = BaseLinear.Raw.combineAssignments PiDEC.radixWeight
        childAssignments column := by
      exact (rawCombineAssignments_apply PiDEC.radixWeight
        childAssignments column).symm

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjectionRows.RawScalarKernel
