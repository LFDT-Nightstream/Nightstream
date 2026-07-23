import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteChain

/-!
Kernel agreement lemmas for the concrete combined-NC source/compiler join.

Owns: transport of assignment equality on an explicit column set through the
repository sparse-linear evaluator, decoded product factors, and the exact
rewrite contribution used by `RewriteChain`.

Does not own: generated dependency certificates, rewrite-chain ownership,
source-program execution, selected-row satisfaction, protocol acceptance,
commitment binding, costs, or row removal.

This leaf is deliberately non-executable.  Artifact leaves must prove that
the exact generated columns referenced by a contribution belong to the
already-established agreement set; no list certificate is normalized here.
-/

/-!
Emits constraints: none; this module proves agreement between existing assignments.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.assignment_agreement` | Relate decoded artifact columns to the materialized assignment used by execution proofs. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.AssignmentAgreement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Decoder
open Semantics
open SelectiveCompilerBridge
open RewriteChain

theorem agreeOn_mono {left right : Nat → Nat} {small large : List Nat}
    (agreement : AgreeOn left right large)
    (subset : ∀ column ∈ small, column ∈ large) :
    AgreeOn left right small := by
  intro column member
  exact agreement column (subset column member)

theorem agreeOn_append {left right : Nat → Nat} {first second : List Nat}
    (firstAgreement : AgreeOn left right first)
    (secondAgreement : AgreeOn left right second) :
    AgreeOn left right (first ++ second) := by
  intro column member
  rcases List.mem_append.mp member with member | member
  · exact firstAgreement column member
  · exact secondAgreement column member

theorem agreeOn_cons {left right : Nat → Nat} {known : List Nat}
    {column : Nat}
    (head : left column = right column)
    (tail : AgreeOn left right known) :
    AgreeOn left right (column :: known) := by
  intro candidate member
  simp only [List.mem_cons] at member
  rcases member with rfl | member
  · exact head
  · exact tail candidate member

theorem lcEval_eq_of_agreeOn {left right : Nat → Nat} {known : List Nat}
    (agreement : AgreeOn left right known) (terms : List (Nat × Nat))
    (references : ∀ term ∈ terms, term.1 ∈ known) :
    lcEval left terms = lcEval right terms := by
  unfold lcEval
  have foldAgreement : ∀ initial,
      terms.foldl (fun accumulator term =>
          accumulator + term.2 * left term.1) initial =
        terms.foldl (fun accumulator term =>
          accumulator + term.2 * right term.1) initial := by
    intro initial
    induction terms generalizing initial with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.foldl]
        rw [agreement head.1 (references head (by simp))]
        apply inductionHypothesis
        intro term member
        exact references term (by simp [member])
  rw [foldAgreement 0]

theorem fieldResidue_injective_of_canonical {left right : Nat}
    (leftCanonical : left < goldilocksP)
    (rightCanonical : right < goldilocksP)
    (equal : fieldResidue left = fieldResidue right) :
    left = right := by
  have values := congrArg Fin.val equal
  change left % goldilocksModulus = right % goldilocksModulus at values
  have modulusEquality : goldilocksP = goldilocksModulus := rfl
  rw [← modulusEquality, Nat.mod_eq_of_lt leftCanonical,
    Nat.mod_eq_of_lt rightCanonical] at values
  exact values

theorem rhsEval_eq_of_agreeOn {left right : Nat → Nat} {known : List Nat}
    (agreement : AgreeOn left right known) (rhs : Rhs)
    (references : ∀ column ∈ rhs.refs, column ∈ known) :
    rhs.eval left = rhs.eval right := by
  cases rhs with
  | linear terms =>
      apply lcEval_eq_of_agreeOn agreement terms
      intro term member
      apply references term.1
      exact List.mem_map.mpr ⟨term, member, rfl⟩
  | product lhs rhs =>
      simp only [Rhs.eval]
      rw [lcEval_eq_of_agreeOn agreement lhs (by
        intro term member
        apply references term.1
        apply List.mem_append_left
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]
      rw [lcEval_eq_of_agreeOn agreement rhs (by
        intro term member
        apply references term.1
        apply List.mem_append_right
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]

theorem definitionOutput_eq_of_holds
    {left right : Nat → Nat} {known : List Nat}
    {definition : Definition}
    (agreement : AgreeOn left right known)
    (references : ReferencesOnly known definition)
    (leftHolds : definition.Holds left)
    (rightHolds : definition.Holds right) :
    left definition.output = right definition.output := by
  unfold Definition.Holds at leftHolds rightHolds
  rw [leftHolds, rightHolds]
  exact rhsEval_eq_of_agreeOn agreement definition.rhs references

def LinearCombinationReferencesOnly {columns : Nat} (known : List Nat)
    (value : DecodedLinearCombination columns) : Prop :=
  ∀ term ∈ linearCombinationTerms value, term.1 ∈ known

instance {columns : Nat} (known : List Nat)
    (value : DecodedLinearCombination columns) :
    Decidable (LinearCombinationReferencesOnly known value) := by
  unfold LinearCombinationReferencesOnly
  infer_instance

theorem linearCombinationValue_eq_of_agreeOn
    {columns : Nat} {left right : Nat → Nat} {known : List Nat}
    {value : DecodedLinearCombination columns}
    (agreement : AgreeOn left right known)
    (references : LinearCombinationReferencesOnly known value) :
    linearCombinationValue value left = linearCombinationValue value right := by
  unfold linearCombinationValue
  rw [lcEval_eq_of_agreeOn agreement _ references]

def ProductFactorReferencesOnly {columns : Nat} (known : List Nat)
    (factor : DecodedProductFactor columns) : Prop :=
  LinearCombinationReferencesOnly known factor.left ∧
    LinearCombinationReferencesOnly known factor.right

instance {columns : Nat} (known : List Nat)
    (factor : DecodedProductFactor columns) :
    Decidable (ProductFactorReferencesOnly known factor) := by
  unfold ProductFactorReferencesOnly
  infer_instance

theorem productFactorValue_eq_of_agreeOn
    {columns : Nat} {left right : Nat → Nat} {known : List Nat}
    {factor : DecodedProductFactor columns}
    (agreement : AgreeOn left right known)
    (references : ProductFactorReferencesOnly known factor) :
    productFactorValue factor left = productFactorValue factor right := by
  unfold productFactorValue
  rw [linearCombinationValue_eq_of_agreeOn agreement references.1,
    linearCombinationValue_eq_of_agreeOn agreement references.2]

private theorem factorValueAt_eq_of_agreeOn
    {columns : Nat} {left right : Nat → Nat} {known : List Nat}
    (agreement : AgreeOn left right known)
    (factors : List (DecodedProductFactor columns))
    (references : ∀ factor ∈ factors,
      ProductFactorReferencesOnly known factor) :
    ∀ index,
      factorValueAt left factors index = factorValueAt right factors index := by
  intro index
  induction factors generalizing index with
  | nil => simp [factorValueAt]
  | cons head tail inductionHypothesis =>
      cases index with
      | zero =>
          simp only [factorValueAt, List.getElem?_cons_zero]
          exact productFactorValue_eq_of_agreeOn agreement
            (references head (by simp))
      | succ index =>
          simp only [factorValueAt, List.getElem?_cons_succ]
          exact inductionHypothesis
            (fun factor member => references factor (by simp [member])) index

theorem factorSum_eq_of_agreeOn
    {columns : Nat} {left right : Nat → Nat} {known : List Nat}
    {factors : List (DecodedProductFactor columns)}
    (agreement : AgreeOn left right known)
    (references : ∀ factor ∈ factors,
      ProductFactorReferencesOnly known factor) :
    factorSum left factors = factorSum right factors := by
  unfold factorSum
  rw [factorValueAt_eq_of_agreeOn agreement factors references 0,
    factorValueAt_eq_of_agreeOn agreement factors references 1,
    factorValueAt_eq_of_agreeOn agreement factors references 2,
    factorValueAt_eq_of_agreeOn agreement factors references 3,
    factorValueAt_eq_of_agreeOn agreement factors references 4]

def ContributionReferencesOnly {columns : Nat} (known : List Nat)
    (step : DecodedRewriteStep columns) : Prop :=
  LinearCombinationReferencesOnly known step.base ∧
    ∀ factor ∈ step.factors,
      ProductFactorReferencesOnly known factor

instance {columns : Nat} (known : List Nat)
    (step : DecodedRewriteStep columns) :
    Decidable (ContributionReferencesOnly known step) := by
  unfold ContributionReferencesOnly
  infer_instance

theorem contribution_eq_of_agreeOn
    {columns : Nat} {left right : Nat → Nat} {known : List Nat}
    {step : DecodedRewriteStep columns}
    (agreement : AgreeOn left right known)
    (references : ContributionReferencesOnly known step) :
    contribution left step = contribution right step := by
  unfold contribution
  rw [linearCombinationValue_eq_of_agreeOn agreement references.1,
    factorSum_eq_of_agreeOn agreement references.2]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.AssignmentAgreement
