import Nightstream.Implementation.Lowering.Goldilocks.NormalForm
import Nightstream.Implementation.Lowering.Goldilocks.ColumnPlan

/-!
Contract: finite normal-form selection for concrete non-call Goldilocks
primitive encodings.

Owns:
- the two admitted branch-join encodings: one selected mux row or two gated
  equality rows;
- the two admitted gated-assertion encodings: one direct row or a materialized
  residual followed by a zero pin;
- independent local semantics and correctness proofs for every admitted
  candidate;
- exact finite-list minima in the fixed order
  `(rows, committed columns, public columns, auxiliary columns)`.

Does not own: call lowering, whole-program compilation, generated artifacts,
Rust behavior, or global minimality over other arithmetizations.  Literal pin
and branch activation are not given singleton candidate lists here: this leaf
has no independently justified competing canonical forms for them.

Emits constraints: no.  The listed candidates describe row recipes; selection
itself emits nothing.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks.PrimitiveNormalForm

open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete

/-! ## Branch join -/

namespace BranchJoin

/-- Source-level data for one field-coordinate branch join.

Selector booleanity is an upstream obligation.  Join rows own only the
selection equation, so both candidates are compared under the same boolean
selector premise. -/
structure Specification where
  owner : PhysicalOwner
  firstOrdinal : Nat
  assignment : ColumnId -> F
  one : ColumnId
  selector : ColumnId
  joined : ColumnId
  onTrue : ColumnId
  onFalse : ColumnId
  constantOne : assignment one = 1
  selectorBoolean :
    assignment selector = 0 ∨ assignment selector = 1

/-- Candidate-independent meaning of a branch join. -/
def Semantics (specification : Specification) : Prop :=
  (specification.assignment specification.selector = 1 ∧
      specification.assignment specification.joined =
        specification.assignment specification.onTrue) ∨
    (specification.assignment specification.selector = 0 ∧
      specification.assignment specification.joined =
        specification.assignment specification.onFalse)

/-- The complete admitted branch-join candidate class. -/
inductive Candidate where
  | selectedMux
  | twoGatedEqualities
deriving DecidableEq, Repr

private def ownedRow
    (specification : Specification)
    (offset : Nat)
    (row : Row) : OwnedRow :=
  { id :=
      { owner := specification.owner
        ordinal := specification.firstOrdinal + offset }
    row := row }

/-- Gate `joined = onTrue` by `selector`. -/
private def trueGatedEquality (specification : Specification) : Row :=
  { a := singleton specification.selector 1
    b := difference specification.joined specification.onTrue
    c := [] }

/-- Gate `joined = onFalse` by `1 - selector`. -/
private def falseGatedEquality (specification : Specification) : Row :=
  { a := oneMinus specification.one specification.selector
    b := difference specification.joined specification.onFalse
    c := [] }

/-- Exact physical rows for each admitted branch-join candidate. -/
def Candidate.rows :
    Candidate -> Specification -> List OwnedRow
  | .selectedMux, specification =>
      [ownedRow specification 0
        (CanonicalRow.mux
          specification.joined
          specification.selector
          specification.onTrue
          specification.onFalse).row]
  | .twoGatedEqualities, specification =>
      [ownedRow specification 0
          (trueGatedEquality specification),
        ownedRow specification 1
          (falseGatedEquality specification)]

/-- Semantic correctness is deliberately independent of candidate cost. -/
def Implements
    (candidate : Candidate)
    (specification : Specification) : Prop :=
  Satisfies (candidate.rows specification) specification.assignment ↔
    Semantics specification

private theorem trueGatedEquality_iff
    (specification : Specification) :
    (trueGatedEquality specification).Holds specification.assignment ↔
      specification.assignment specification.selector *
          (specification.assignment specification.joined -
            specification.assignment specification.onTrue) = 0 := by
  simp only [trueGatedEquality, Row.Holds, singleton, difference,
    LinearCombination.eval, Fin.one_mul, Fin.add_zero,
    Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg]

private theorem falseGatedEquality_iff
    (specification : Specification) :
    (falseGatedEquality specification).Holds specification.assignment ↔
      (1 - specification.assignment specification.selector) *
          (specification.assignment specification.joined -
            specification.assignment specification.onFalse) = 0 := by
  simp only [falseGatedEquality, Row.Holds, difference, oneMinus,
    LinearCombination.eval, specification.constantOne, Fin.one_mul,
    Fin.add_zero, Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg]

private theorem mux_complete_true
    (specification : Specification)
    (selectorOne :
      specification.assignment specification.selector = 1)
    (joinedTrue :
      specification.assignment specification.joined =
        specification.assignment specification.onTrue) :
    (CanonicalRow.mux
      specification.joined
      specification.selector
      specification.onTrue
      specification.onFalse).row.Holds specification.assignment := by
  apply (CanonicalRow.mux_iff
    specification.assignment
    specification.joined
    specification.selector
    specification.onTrue
    specification.onFalse).mpr
  rw [selectorOne, Fin.one_mul, joinedTrue]

private theorem mux_complete_false
    (specification : Specification)
    (selectorZero :
      specification.assignment specification.selector = 0)
    (joinedFalse :
      specification.assignment specification.joined =
        specification.assignment specification.onFalse) :
    (CanonicalRow.mux
      specification.joined
      specification.selector
      specification.onTrue
      specification.onFalse).row.Holds specification.assignment := by
  apply (CanonicalRow.mux_iff
    specification.assignment
    specification.joined
    specification.selector
    specification.onTrue
    specification.onFalse).mpr
  rw [selectorZero, Fin.zero_mul, joinedFalse]
  exact
    (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr rfl).symm

theorem selectedMux_correct
    (specification : Specification) :
    Implements .selectedMux specification := by
  constructor
  · intro satisfies
    have rowHolds :
        (CanonicalRow.mux
          specification.joined
          specification.selector
          specification.onTrue
          specification.onFalse).row.Holds
            specification.assignment :=
      satisfies.1
    rcases specification.selectorBoolean with selectorZero | selectorOne
    · exact Or.inr ⟨selectorZero,
        CanonicalRow.mux_selects_false
          specification.assignment
          specification.joined
          specification.selector
          specification.onTrue
          specification.onFalse
          selectorZero rowHolds⟩
    · exact Or.inl ⟨selectorOne,
        CanonicalRow.mux_selects_true
          specification.assignment
          specification.joined
          specification.selector
          specification.onTrue
          specification.onFalse
          selectorOne rowHolds⟩
  · intro semantics
    rcases semantics with
      ⟨selectorOne, joinedTrue⟩ | ⟨selectorZero, joinedFalse⟩
    · exact ⟨mux_complete_true specification selectorOne joinedTrue,
        True.intro⟩
    · exact ⟨mux_complete_false specification selectorZero joinedFalse,
        True.intro⟩

theorem twoGatedEqualities_correct
    (specification : Specification) :
    Implements .twoGatedEqualities specification := by
  constructor
  · intro satisfies
    have trueEquation :=
      (trueGatedEquality_iff specification).mp satisfies.1
    have falseEquation :=
      (falseGatedEquality_iff specification).mp satisfies.2.1
    rcases specification.selectorBoolean with selectorZero | selectorOne
    · right
      refine ⟨selectorZero, ?_⟩
      rw [selectorZero] at falseEquation
      have differenceZero :
          specification.assignment specification.joined -
            specification.assignment specification.onFalse = 0 := by
        rw [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_zero,
          Fin.add_zero, Fin.one_mul] at falseEquation
        exact falseEquation
      exact
        Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp differenceZero
    · left
      refine ⟨selectorOne, ?_⟩
      rw [selectorOne] at trueEquation
      have differenceZero :
          specification.assignment specification.joined -
            specification.assignment specification.onTrue = 0 := by
        rw [Fin.one_mul] at trueEquation
        exact trueEquation
      exact
        Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp differenceZero
  · intro semantics
    rcases semantics with
      ⟨selectorOne, joinedTrue⟩ | ⟨selectorZero, joinedFalse⟩
    · refine ⟨
        (trueGatedEquality_iff specification).mpr ?_,
        (falseGatedEquality_iff specification).mpr ?_,
        True.intro⟩
      · rw [selectorOne, Fin.one_mul]
        exact
          Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr joinedTrue
      · rw [selectorOne]
        have oneSubOne : (1 : F) - 1 = 0 :=
          Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr rfl
        rw [oneSubOne, Fin.zero_mul]
    · refine ⟨
        (trueGatedEquality_iff specification).mpr ?_,
        (falseGatedEquality_iff specification).mpr ?_,
        True.intro⟩
      · rw [selectorZero]
        rw [Fin.zero_mul]
      · rw [selectorZero, Fin.sub_eq_add_neg,
          Lean.Grind.AddCommGroup.neg_zero, Fin.add_zero, Fin.one_mul]
        exact
          Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr joinedFalse

/-- Neither join form allocates a candidate-specific column.  The selector,
arms, and joined output belong to the common surrounding column plan. -/
def Candidate.allocations :
    Candidate -> Specification -> List OwnedColumn
  | .selectedMux, _ => []
  | .twoGatedEqualities, _ => []

/-- Resource use is folded from the candidate's actual owned allocations and
row occurrences, rather than copied into an editable cost table. -/
def Candidate.cost
    (specification : Specification)
    (candidate : Candidate) : Cost :=
  physicalCost
    (candidate.allocations specification)
    (candidate.rows specification)

theorem selectedMux_cost (specification : Specification) :
    Candidate.cost specification .selectedMux = ⟨1, 0, 0, 0⟩ :=
  rfl

theorem twoGatedEqualities_cost (specification : Specification) :
    Candidate.cost specification .twoGatedEqualities = ⟨2, 0, 0, 0⟩ :=
  rfl

/-- The explicit, non-singleton finite class over which minimum is proved. -/
def candidates (specification : Specification) :
    NormalForm.FiniteCandidates
      Candidate Specification Implements specification where
  head := .selectedMux
  tail := [.twoGatedEqualities]
  correct := by
    intro candidate _
    cases candidate
    · exact selectedMux_correct specification
    · exact twoGatedEqualities_correct specification

/-- Selected branch-join normal form. -/
def canonical (specification : Specification) : Candidate :=
  (candidates specification).canonical (Candidate.cost specification)

theorem canonical_eq_selectedMux
    (specification : Specification) :
    canonical specification = .selectedMux :=
  rfl

theorem canonical_cost
    (specification : Specification) :
    Candidate.cost specification (canonical specification) =
      ⟨1, 0, 0, 0⟩ :=
  rfl

theorem canonical_correct
    (specification : Specification) :
    Implements (canonical specification) specification :=
  (candidates specification).canonical_correct
    (Candidate.cost specification)

/-- Inclusion-minimality inside exactly the two admitted branch-join forms. -/
theorem canonical_minimum
    (specification : Specification)
    (candidate : Candidate)
    (member : candidate ∈ (candidates specification).members) :
    Cost.LexLe
      (Candidate.cost specification (canonical specification))
      (Candidate.cost specification candidate) :=
  (candidates specification).canonical_minimum
    (Candidate.cost specification) member

end BranchJoin

/-! ## Gated assertion -/

namespace GatedAssertion

/-- Source data plus a fresh auxiliary identity available to the materialized
candidate.  Freshness prevents its synthesized witness from changing any
source coordinate. -/
structure Specification where
  owner : PhysicalOwner
  firstOrdinal : Nat
  assignment : ColumnId -> F
  one : ColumnId
  active : ColumnId
  condition : ColumnId
  residual : ColumnId
  constantOne : assignment one = 1
  residual_ne_one : residual ≠ one
  residual_ne_active : residual ≠ active
  residual_ne_condition : residual ≠ condition

/-- Candidate-independent gated-assertion equation. -/
def Semantics (specification : Specification) : Prop :=
  specification.assignment specification.active *
      (1 - specification.assignment specification.condition) = 0

/-- Direct assertion or an explicitly materialized residual. -/
inductive Candidate where
  | direct
  | materializedResidual
deriving DecidableEq, Repr

private def residualValue (specification : Specification) : F :=
  specification.assignment specification.active *
    (1 - specification.assignment specification.condition)

/-- Candidate witness construction.  Only the materialized form allocates and
writes the fresh residual coordinate. -/
def Candidate.witness :
    Candidate -> Specification -> ColumnId -> F
  | .direct, specification => specification.assignment
  | .materializedResidual, specification =>
      fun column =>
        if column = specification.residual then
          residualValue specification
        else
          specification.assignment column

private def ownedRow
    (specification : Specification)
    (offset : Nat)
    (row : Row) : OwnedRow :=
  { id :=
      { owner := specification.owner
        ordinal := specification.firstOrdinal + offset }
    row := row }

/-- Define the residual as `active * (1 - condition)`. -/
private def materializeResidualRow
    (specification : Specification) : Row :=
  { a := singleton specification.active 1
    b := oneMinus specification.one specification.condition
    c := singleton specification.residual 1 }

/-- Exact physical rows for each admitted assertion candidate. -/
def Candidate.rows :
    Candidate -> Specification -> List OwnedRow
  | .direct, specification =>
      [ownedRow specification 0
        (CanonicalRow.gatedAssert
          specification.one
          specification.active
          specification.condition).row]
  | .materializedResidual, specification =>
      [ownedRow specification 0
          (materializeResidualRow specification),
        ownedRow specification 1
          (CanonicalRow.pin
            specification.one specification.residual 0).row]

/-- Semantic correctness includes each candidate's explicit witness
construction but remains independent of its cost. -/
def Implements
    (candidate : Candidate)
    (specification : Specification) : Prop :=
  Satisfies
      (candidate.rows specification)
      (candidate.witness specification) ↔
    Semantics specification

private theorem directRow_iff
    (specification : Specification) :
    (CanonicalRow.gatedAssert
      specification.one
      specification.active
      specification.condition).row.Holds specification.assignment ↔
      Semantics specification := by
  simp only [CanonicalRow.row, Row.Holds, singleton, oneMinus,
    LinearCombination.eval, specification.constantOne, Fin.one_mul,
    Fin.add_zero, Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg, Semantics]

private theorem materializedWitness_one
    (specification : Specification) :
    Candidate.witness .materializedResidual specification specification.one =
      1 := by
  simp [Candidate.witness, specification.residual_ne_one.symm,
    specification.constantOne]

private theorem materializedWitness_residual
    (specification : Specification) :
    Candidate.witness
        .materializedResidual specification specification.residual =
      residualValue specification := by
  simp [Candidate.witness]

private theorem materializeResidualRow_holds
    (specification : Specification) :
    (materializeResidualRow specification).Holds
      (Candidate.witness .materializedResidual specification) := by
  simp only [materializeResidualRow, Row.Holds, singleton, oneMinus,
    LinearCombination.eval, Fin.one_mul, Fin.add_zero,
    Lean.Grind.Fin.neg_mul]
  rw [materializedWitness_one, materializedWitness_residual]
  simp only [Candidate.witness,
    specification.residual_ne_active.symm,
    specification.residual_ne_condition.symm,
    if_false, residualValue]
  rw [Fin.sub_eq_add_neg]

private theorem materializedPin_iff
    (specification : Specification) :
    (CanonicalRow.pin
      specification.one specification.residual 0).row.Holds
        (Candidate.witness .materializedResidual specification) ↔
      Semantics specification := by
  rw [CanonicalRow.pin_iff
    (Candidate.witness .materializedResidual specification)
    specification.one specification.residual 0
    (materializedWitness_one specification)]
  rw [materializedWitness_residual]
  rfl

theorem direct_correct
    (specification : Specification) :
    Implements .direct specification := by
  constructor
  · intro satisfies
    exact (directRow_iff specification).mp satisfies.1
  · intro semantics
    exact ⟨(directRow_iff specification).mpr semantics, True.intro⟩

theorem materializedResidual_correct
    (specification : Specification) :
    Implements .materializedResidual specification := by
  constructor
  · intro satisfies
    exact (materializedPin_iff specification).mp satisfies.2.1
  · intro semantics
    exact ⟨materializeResidualRow_holds specification,
      (materializedPin_iff specification).mpr semantics,
      True.intro⟩

/-- Candidate-specific owned allocations.  Only the residual form introduces
the fresh auxiliary coordinate written by its witness construction. -/
def Candidate.allocations :
    Candidate -> Specification -> List OwnedColumn
  | .direct, _ => []
  | .materializedResidual, specification =>
      [{ id := specification.residual
         ownership := .auxiliaryColumn }]

/-- Resource use is folded from actual candidate allocations and rows. -/
def Candidate.cost
    (specification : Specification)
    (candidate : Candidate) : Cost :=
  physicalCost
    (candidate.allocations specification)
    (candidate.rows specification)

theorem direct_cost (specification : Specification) :
    Candidate.cost specification .direct = ⟨1, 0, 0, 0⟩ :=
  rfl

theorem materializedResidual_cost (specification : Specification) :
    Candidate.cost specification .materializedResidual =
      ⟨2, 0, 0, 1⟩ :=
  rfl

/-- The explicit, non-singleton finite class over which minimum is proved. -/
def candidates (specification : Specification) :
    NormalForm.FiniteCandidates
      Candidate Specification Implements specification where
  head := .direct
  tail := [.materializedResidual]
  correct := by
    intro candidate _
    cases candidate
    · exact direct_correct specification
    · exact materializedResidual_correct specification

/-- Selected gated-assertion normal form. -/
def canonical (specification : Specification) : Candidate :=
  (candidates specification).canonical (Candidate.cost specification)

theorem canonical_eq_direct
    (specification : Specification) :
    canonical specification = .direct :=
  rfl

theorem canonical_cost
    (specification : Specification) :
    Candidate.cost specification (canonical specification) =
      ⟨1, 0, 0, 0⟩ :=
  rfl

theorem canonical_correct
    (specification : Specification) :
    Implements (canonical specification) specification :=
  (candidates specification).canonical_correct
    (Candidate.cost specification)

/-- Inclusion-minimality inside exactly the two admitted assertion forms. -/
theorem canonical_minimum
    (specification : Specification)
    (candidate : Candidate)
    (member : candidate ∈ (candidates specification).members) :
    Cost.LexLe
      (Candidate.cost specification (canonical specification))
      (Candidate.cost specification candidate) :=
  (candidates specification).canonical_minimum
    (Candidate.cost specification) member

end GatedAssertion

end Nightstream.Implementation.Lowering.Goldilocks.PrimitiveNormalForm
