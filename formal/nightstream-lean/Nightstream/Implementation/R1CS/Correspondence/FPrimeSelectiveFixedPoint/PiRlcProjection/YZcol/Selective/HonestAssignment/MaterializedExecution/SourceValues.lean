import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.AbstractProgram

/-!
Source/compiler value transport for the bounded selective fixed-point
`y_zcol` rewrite program.

Owns: agreement of one decoded source linear form, product factor, and bounded
five-factor sum with the centered-word materialized assignment.

Does not own: derived-slot materialization, full-program recurrence evidence,
retained checks, or selected-row completeness.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `materialized.source_value` | compiler-backed source forms equal abstract source values | derived |
| `materialized.factor_sum` | every bounded factor sum preserves its abstract field value | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

private theorem rawLcEval_eq_of_agree
    {left right : Nat → Nat} : ∀ terms : List (Nat × Nat),
    (∀ term ∈ terms, left term.1 = right term.1) →
      rawLcEval left terms = rawLcEval right terms := by
  intro terms
  induction terms with
  | nil => intro _; rfl
  | cons head tail inductionHypothesis =>
      intro agree
      simp only [rawLcEval]
      rw [agree head (by simp)]
      rw [inductionHypothesis]
      intro term member
      exact agree term (by simp [member])

private theorem lcEval_eq_of_agree
    {left right : Nat → Nat} (terms : List (Nat × Nat))
    (agree : ∀ term ∈ terms, left term.1 = right term.1) :
    lcEval left terms = lcEval right terms := by
  rw [Program.lcEval_eq_raw_mod, Program.lcEval_eq_raw_mod,
    rawLcEval_eq_of_agree terms agree]

theorem sourceValue_eq_abstract
    {source derived : Nat → Nat}
    (honest : HonestSourceBoundary source)
    (linear : SourceDecode.DecodedSourceLinearCombination)
    (known : LinearKnown linear) :
    sourceValue (materializedAssignment source derived) linear =
      abstractSourceValue source linear := by
  unfold sourceValue abstractSourceValue
  apply congrArg Materialized.Semantics.fieldResidue
  apply lcEval_eq_of_agree
  intro term member
  exact compilerAssignment_agrees honest term.1 (known term member)

private theorem factorValue_eq_abstract
    {source derived : Nat → Nat}
    (honest : HonestSourceBoundary source)
    (factor : DecodedProductFactor) (known : FactorKnown factor) :
    factorValue (materializedAssignment source derived) factor =
      abstractFactorValue source factor := by
  unfold factorValue abstractFactorValue
  rw [sourceValue_eq_abstract honest factor.left known.1,
    sourceValue_eq_abstract honest factor.right known.2]

private theorem factorValueAt_eq_abstract
    {source derived : Nat → Nat}
    (honest : HonestSourceBoundary source)
    (factors : List DecodedProductFactor)
    (known : ∀ factor ∈ factors, FactorKnown factor) (index : Nat) :
    factorValueAt (materializedAssignment source derived) factors index =
      abstractFactorValueAt source factors index := by
  unfold factorValueAt abstractFactorValueAt
  cases selected : factors[index]? with
  | none => rfl
  | some factor =>
      apply factorValue_eq_abstract honest factor
      rcases List.getElem_of_getElem? selected with ⟨bound, equality⟩
      rw [← equality]
      exact known factors[index] (List.get_mem factors ⟨index, bound⟩)

theorem factorSum_eq_abstract
    {source derived : Nat → Nat}
    (honest : HonestSourceBoundary source)
    (factors : List DecodedProductFactor)
    (known : ∀ factor ∈ factors, FactorKnown factor) :
    factorSum (materializedAssignment source derived) factors =
      abstractFactorSum source factors := by
  unfold factorSum abstractFactorSum
  rw [factorValueAt_eq_abstract honest factors known 0,
    factorValueAt_eq_abstract honest factors known 1,
    factorValueAt_eq_abstract honest factors known 2,
    factorValueAt_eq_abstract honest factors known 3,
    factorValueAt_eq_abstract honest factors known 4]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
