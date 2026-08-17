import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Contract: proof-carrying straight-line normalization for large R1CS artifacts.

This module is the scaling path beyond hand-proving thousands of generated
rows. A definition is either a linear combination or one field
multiplication. `WellFormed` makes dependencies and single assignment
explicit. `run_agrees_of_holds` proves that every satisfying execution agrees
with the deterministic interpreter on every derived column.

The normalizer covers both production row shapes: multiplication rows use
`lhs * rhs = output`, while Rust's `enforce_eq(output, rhs)` emits the
algebraically equivalent `(output-rhs) * 1 = 0`. The cancellation proof for
that subtraction form is part of this module rather than an exporter trust
assumption.
-/

namespace Nightstream.Implementation.R1CS.Program

open Nightstream.Implementation.R1CS

inductive Rhs where
  | linear (terms : List (Nat × Nat))
  | product (left right : List (Nat × Nat))
deriving DecidableEq, Repr

def Rhs.refs : Rhs → List Nat
  | .linear terms => terms.map Prod.fst
  | .product left right => left.map Prod.fst ++ right.map Prod.fst

def Rhs.eval (z : Nat → Nat) : Rhs → Nat
  | .linear terms => lcEval z terms
  | .product left right => lcEval z left * lcEval z right % goldilocksP

structure Definition where
  output : Nat
  rhs : Rhs
deriving DecidableEq, Repr

/-- Normalized one-row representation of a deterministic definition. -/
def Definition.row (definition : Definition) : Row :=
  match definition.rhs with
  | .linear terms => ⟨terms, [(0, 1)], [(definition.output, 1)]⟩
  | .product left right => ⟨left, right, [(definition.output, 1)]⟩

/-- Unreduced integer value of an LC. `lcEval` is this value modulo p. -/
def rawLcEval (z : Nat → Nat) : List (Nat × Nat) → Nat
  | [] => 0
  | term :: tail => term.2 * z term.1 + rawLcEval z tail

private theorem foldl_lc_eq_add_raw (z : Nat → Nat) (terms : List (Nat × Nat))
    (initial : Nat) :
    terms.foldl (fun acc term => acc + term.2 * z term.1) initial =
      initial + rawLcEval z terms := by
  induction terms generalizing initial with
  | nil => simp [rawLcEval]
  | cons head tail inductionHypothesis =>
      simp only [List.foldl]
      rw [inductionHypothesis]
      simp only [rawLcEval]
      omega

theorem lcEval_eq_raw_mod (z : Nat → Nat) (terms : List (Nat × Nat)) :
    lcEval z terms = rawLcEval z terms % goldilocksP := by
  unfold lcEval
  rw [foldl_lc_eq_add_raw]
  simp

/-- Sparse LC evaluation is insensitive to term order. This is the semantic
bridge between builder emission order and CSC's canonical sparse order. -/
theorem rawLcEval_eq_of_perm (z : Nat → Nat) {left right : List (Nat × Nat)}
    (permutation : left.Perm right) :
    rawLcEval z left = rawLcEval z right := by
  induction permutation with
  | nil => rfl
  | cons _ _ inductionHypothesis =>
      simp [rawLcEval, inductionHypothesis]
  | swap _ _ _ =>
      simp [rawLcEval]
      omega
  | trans _ _ first second =>
      omega

/-- Canonical LC evaluation is insensitive to term order. -/
theorem lcEval_eq_of_perm (z : Nat → Nat) {left right : List (Nat × Nat)}
    (permutation : left.Perm right) :
    lcEval z left = lcEval z right := by
  rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod,
    rawLcEval_eq_of_perm z permutation]

def negCoeff (coefficient : Nat) : Nat :=
  if coefficient = 0 then 0 else goldilocksP - coefficient

def negateTerms (terms : List (Nat × Nat)) : List (Nat × Nat) :=
  terms.map (fun term => (term.1, negCoeff term.2))

/-- Exact precondition of a canonical nonzero Rust sparse LC. -/
def CanonicalTerms (terms : List (Nat × Nat)) : Prop :=
  ∀ term ∈ terms, 0 < term.2 ∧ term.2 < goldilocksP

instance (terms : List (Nat × Nat)) : Decidable (CanonicalTerms terms) := by
  unfold CanonicalTerms
  infer_instance

private theorem term_cancel_mod (z : Nat → Nat) {term : Nat × Nat}
    (canonical : 0 < term.2 ∧ term.2 < goldilocksP) :
    (term.2 * z term.1 + negCoeff term.2 * z term.1) % goldilocksP = 0 := by
  have coefficientLe : term.2 ≤ goldilocksP := Nat.le_of_lt canonical.2
  have coefficientNe : term.2 ≠ 0 := Nat.ne_of_gt canonical.1
  simp only [negCoeff, coefficientNe, ↓reduceIte]
  rw [← Nat.add_mul, Nat.add_sub_of_le coefficientLe]
  simp

private theorem rawLc_cancel_mod (z : Nat → Nat) (terms : List (Nat × Nat))
    (canonical : CanonicalTerms terms) :
    (rawLcEval z terms + rawLcEval z (negateTerms terms)) % goldilocksP = 0 := by
  induction terms with
  | nil => simp [rawLcEval, negateTerms]
  | cons head tail inductionHypothesis =>
      have headCanonical := canonical head (by simp)
      have tailCanonical : CanonicalTerms tail := by
        intro term member
        exact canonical term (by simp [member])
      have headCancel := term_cancel_mod z headCanonical
      have tailCancel := inductionHypothesis tailCanonical
      simp only [negateTerms] at tailCancel
      simp only [rawLcEval, negateTerms, List.map_cons]
      have reorder :
          head.2 * z head.1 + rawLcEval z tail +
              (negCoeff head.2 * z head.1 + rawLcEval z (List.map (fun term =>
                (term.1, negCoeff term.2)) tail)) =
            (head.2 * z head.1 + negCoeff head.2 * z head.1) +
              (rawLcEval z tail + rawLcEval z (List.map (fun term =>
                (term.1, negCoeff term.2)) tail)) := by
        omega
      rw [reorder, Nat.add_mod, headCancel, tailCancel]
      decide

/-- Appending the canonical coefficient-wise negation of an LC yields zero
under every assignment. This is the semantic counterpart of CSC duplicate
coalescing when a source column and its decoded alias cancel. -/
theorem lcEval_append_negateTerms_eq_zero
    (z : Nat → Nat) (terms : List (Nat × Nat))
    (canonical : CanonicalTerms terms) :
    lcEval z (terms ++ negateTerms terms) = 0 := by
  have rawAppend (left right : List (Nat × Nat)) :
      rawLcEval z (left ++ right) =
        rawLcEval z left + rawLcEval z right := by
    induction left with
    | nil => simp [rawLcEval]
    | cons head tail inductionHypothesis =>
        simp only [List.cons_append, rawLcEval]
        rw [inductionHypothesis]
        omega
  rw [lcEval_eq_raw_mod, rawAppend]
  exact rawLc_cancel_mod z terms canonical

/-- Rust `enforce_eq(output, rhs)` row: `(output - rhs) * 1 = 0`.
`CanonicalTerms` ensures coefficient negation exactly matches the sparse
Goldilocks representation (no zero terms). -/
def builderLinearRow (output : Nat) (terms : List (Nat × Nat)) : Row :=
  ⟨(output, 1) :: negateTerms terms, [(0, 1)], []⟩

def Definition.builderRow (definition : Definition) : Row :=
  match definition.rhs with
  | .linear terms => builderLinearRow definition.output terms
  | .product left right => ⟨left, right, [(definition.output, 1)]⟩

def Definition.Canonical (definition : Definition) : Prop :=
  match definition.rhs with
  | .linear terms => CanonicalTerms terms
  | .product _ _ => True

instance (definition : Definition) : Decidable definition.Canonical := by
  unfold Definition.Canonical
  cases definition.rhs <;> infer_instance

def Definition.Holds (z : Nat → Nat) (definition : Definition) : Prop :=
  z definition.output = definition.rhs.eval z

instance (z : Nat → Nat) (definition : Definition) :
    Decidable (Definition.Holds z definition) := by
  unfold Definition.Holds
  infer_instance

def ReferencesOnly (known : List Nat) (definition : Definition) : Prop :=
  ∀ column ∈ definition.rhs.refs, column ∈ known

instance (known : List Nat) (definition : Definition) :
    Decidable (ReferencesOnly known definition) := by
  unfold ReferencesOnly
  infer_instance

/-- SSA discipline: every RHS reads only known columns and every output is
fresh. This makes the artifact interpreter deterministic. -/
inductive WellFormed : List Nat → List Definition → Prop where
  | nil (known) : WellFormed known []
  | cons {known : List Nat} {head : Definition} {tail : List Definition}
      (references : ReferencesOnly known head)
      (fresh : head.output ∉ known)
      (rest : WellFormed (head.output :: known) tail) :
      WellFormed known (head :: tail)

private def wellFormedDecidable (known : List Nat) :
    (definitions : List Definition) → Decidable (WellFormed known definitions)
  | [] => isTrue (.nil known)
  | head :: tail =>
      if references : ReferencesOnly known head then
        if fresh : head.output ∉ known then
          match wellFormedDecidable (head.output :: known) tail with
          | isTrue rest => isTrue (.cons references fresh rest)
          | isFalse notRest => isFalse (fun wellFormed => by
              cases wellFormed with
              | cons _ _ rest => exact notRest rest)
        else
          isFalse (fun wellFormed => by
            cases wellFormed with
            | cons _ actualFresh _ => exact fresh actualFresh)
      else
        isFalse (fun wellFormed => by
          cases wellFormed with
          | cons actualReferences _ _ => exact references actualReferences)

instance (known : List Nat) (definitions : List Definition) :
    Decidable (WellFormed known definitions) :=
  wellFormedDecidable known definitions

def AgreeOn (left right : Nat → Nat) (columns : List Nat) : Prop :=
  ∀ column ∈ columns, left column = right column

def setColumn (state : Nat → Nat) (column value : Nat) : Nat → Nat :=
  fun candidate => if candidate = column then value else state candidate

@[simp] theorem setColumn_same (state : Nat → Nat) (column value : Nat) :
    setColumn state column value column = value := by
  simp [setColumn]

theorem setColumn_other (state : Nat → Nat) {column other value : Nat}
    (different : other ≠ column) :
    setColumn state column value other = state other := by
  simp [setColumn, different]

def execute (state : Nat → Nat) (definition : Definition) : Nat → Nat :=
  setColumn state definition.output (definition.rhs.eval state)

def run : (Nat → Nat) → List Definition → Nat → Nat
  | state, [] => state
  | state, head :: tail => run (execute state head) tail

def knownAfter : List Nat → List Definition → List Nat
  | known, [] => known
  | known, head :: tail => knownAfter (head.output :: known) tail

theorem mem_knownAfter {known : List Nat} {definitions : List Definition}
    {column : Nat} (member : column ∈ known) :
    column ∈ knownAfter known definitions := by
  induction definitions generalizing known with
  | nil => exact member
  | cons head tail inductionHypothesis =>
      exact inductionHypothesis (List.mem_cons_of_mem head.output member)

private theorem rhsEval_lt (z : Nat → Nat) (rhs : Rhs) :
    rhs.eval z < goldilocksP := by
  have modulusPositive : 0 < goldilocksP := by decide
  cases rhs <;> simp only [Rhs.eval] <;> exact Nat.mod_lt _ modulusPositive

private theorem lcEval_agree {left right : Nat → Nat} {known : List Nat}
    (agreement : AgreeOn left right known) (terms : List (Nat × Nat))
    (references : ∀ term ∈ terms, term.1 ∈ known) :
    lcEval left terms = lcEval right terms := by
  unfold lcEval
  have foldAgree : ∀ initial,
      terms.foldl (fun acc term => acc + term.2 * left term.1) initial =
        terms.foldl (fun acc term => acc + term.2 * right term.1) initial := by
    intro initial
    induction terms generalizing initial with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.foldl]
        rw [agreement head.1 (references head (by simp))]
        apply inductionHypothesis
        intro term member
        exact references term (by simp [member])
  rw [foldAgree 0]

private theorem rhsEval_agree {left right : Nat → Nat} {known : List Nat}
    (agreement : AgreeOn left right known) (rhs : Rhs)
    (references : ∀ column ∈ rhs.refs, column ∈ known) :
    rhs.eval left = rhs.eval right := by
  cases rhs with
  | linear terms =>
      apply lcEval_agree agreement terms
      intro term member
      apply references term.1
      exact List.mem_map.mpr ⟨term, member, rfl⟩
  | product lhs rhs =>
      simp only [Rhs.eval]
      rw [lcEval_agree agreement lhs (by
        intro term member
        apply references term.1
        apply List.mem_append_left
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]
      rw [lcEval_agree agreement rhs (by
        intro term member
        apply references term.1
        apply List.mem_append_right
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]

private theorem execute_preserves_known {known : List Nat} {definition : Definition}
    (fresh : definition.output ∉ known) (state : Nat → Nat) :
    AgreeOn (execute state definition) state known := by
  intro column member
  rw [execute, setColumn_other state]
  intro equal
  apply fresh
  rw [← equal]
  exact member

theorem run_preserves_known {known : List Nat} {definitions : List Definition}
    (wellFormed : WellFormed known definitions) (state : Nat → Nat) :
    AgreeOn (run state definitions) state known := by
  induction wellFormed generalizing state with
  | nil => exact fun _ _ => rfl
  | @cons known head tail references fresh rest inductionHypothesis =>
      intro column member
      have tailPreserves := inductionHypothesis (execute state head)
        column (by simp [member])
      exact tailPreserves.trans (execute_preserves_known fresh state column member)

theorem run_canonical {definitions : List Definition} {state : Nat → Nat}
    (canonical : ∀ column, state column < goldilocksP) :
    ∀ column, run state definitions column < goldilocksP := by
  induction definitions generalizing state with
  | nil => exact canonical
  | cons head tail inductionHypothesis =>
      apply inductionHypothesis
      intro column
      by_cases isOutput : column = head.output
      · subst column
        simp only [execute, setColumn_same]
        exact rhsEval_lt state head.rhs
      · rw [execute, setColumn_other state isOutput]
        exact canonical column

private theorem definitionHolds_execute {known : List Nat} {definition : Definition}
    (references : ReferencesOnly known definition)
    (fresh : definition.output ∉ known) (state : Nat → Nat) :
    Definition.Holds (execute state definition) definition := by
  unfold Definition.Holds
  simp only [execute, setColumn_same]
  exact (rhsEval_agree (right := state) (known := known)
    (execute_preserves_known fresh state) definition.rhs references).symm

private theorem definitionHolds_of_agree {known : List Nat}
    {definition : Definition} {left right : Nat → Nat}
    (references : ReferencesOnly known definition)
    (agreement : AgreeOn left right (definition.output :: known))
    (holds : Definition.Holds right definition) :
    Definition.Holds left definition := by
  unfold Definition.Holds at holds ⊢
  rw [agreement definition.output (by simp)]
  rw [rhsEval_agree agreement definition.rhs]
  · exact holds
  · intro column member
    exact List.mem_cons_of_mem definition.output (references column member)

/-- The interpreter makes every checked SSA definition true in its final
state; later definitions cannot overwrite earlier dependencies or outputs. -/
theorem run_definitions_hold {known : List Nat} {definitions : List Definition}
    (wellFormed : WellFormed known definitions) (state : Nat → Nat) :
    ∀ definition ∈ definitions,
      Definition.Holds (run state definitions) definition := by
  induction wellFormed generalizing state with
  | nil => simp
  | @cons known head tail references fresh rest inductionHypothesis =>
      intro definition member
      simp only [List.mem_cons] at member
      rcases member with isHead | inTail
      · subst definition
        apply definitionHolds_of_agree references
        · exact run_preserves_known rest (execute state head)
        · exact definitionHolds_execute references fresh state
      · exact inductionHypothesis (execute state head) definition inTail

/-- One normalized R1CS row yields its deterministic column definition. -/
theorem definition_sound {z : Nat → Nat} (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (definition : Definition)
    (holds : RowHolds z definition.row) : Definition.Holds z definition := by
  cases definition with
  | mk output rhs =>
      cases rhs with
      | linear terms =>
          have outputLt := hcanon output
          simpa [Definition.Holds, Definition.row, Rhs.eval, RowHolds,
            lcEval, hone, Nat.mod_eq_of_lt outputLt] using holds.symm
      | product left right =>
          have outputLt := hcanon output
          simpa [Definition.Holds, Definition.row, Rhs.eval, RowHolds,
            lcEval, Nat.mod_eq_of_lt outputLt] using holds.symm

/-- Soundness of Rust's exact subtraction-form linear equality row. -/
theorem builderLinearRow_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP) (hone : z 0 = 1)
    (output : Nat) (terms : List (Nat × Nat)) (canonical : CanonicalTerms terms)
    (holds : RowHolds z (builderLinearRow output terms)) :
    z output = lcEval z terms := by
  have modulusPositive : 0 < goldilocksP := by decide
  have outputLt := hcanon output
  have rhsLt : lcEval z terms < goldilocksP := by
    rw [lcEval_eq_raw_mod]
    exact Nat.mod_lt _ modulusPositive
  have claimedCancel :
      (z output + rawLcEval z (negateTerms terms)) % goldilocksP = 0 := by
    simpa [builderLinearRow, RowHolds, lcEval_eq_raw_mod, rawLcEval, hone]
      using holds
  have rawCancel := rawLc_cancel_mod z terms canonical
  have rhsCancel :
      (lcEval z terms + rawLcEval z (negateTerms terms)) % goldilocksP = 0 := by
    rw [lcEval_eq_raw_mod, Nat.add_comm, Nat.add_mod_mod, Nat.add_comm]
    exact rawCancel
  simp only [goldilocksP] at claimedCancel rhsCancel outputLt rhsLt ⊢
  omega

/-- Every exact Rust builder definition row yields its normalized equation. -/
theorem builderDefinition_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP) (hone : z 0 = 1)
    (definition : Definition) (canonical : definition.Canonical)
    (holds : RowHolds z definition.builderRow) : Definition.Holds z definition := by
  cases definition with
  | mk output rhs =>
      cases rhs with
      | linear terms =>
          exact builderLinearRow_sound hcanon hone output terms canonical holds
      | product left right =>
          have outputLt := hcanon output
          simpa [Definition.Holds, Definition.builderRow, Rhs.eval, RowHolds,
            lcEval, Nat.mod_eq_of_lt outputLt] using holds.symm

theorem builderLinearRow_complete {z : Nat → Nat} (hone : z 0 = 1)
    (output : Nat) (terms : List (Nat × Nat)) (canonical : CanonicalTerms terms)
    (holds : z output = lcEval z terms) :
    RowHolds z (builderLinearRow output terms) := by
  have rawCancel := rawLc_cancel_mod z terms canonical
  have claimedCancel :
      (z output + rawLcEval z (negateTerms terms)) % goldilocksP = 0 := by
    rw [holds, lcEval_eq_raw_mod, Nat.add_comm, Nat.add_mod_mod, Nat.add_comm]
    exact rawCancel
  simpa [builderLinearRow, RowHolds, lcEval_eq_raw_mod, rawLcEval, hone]
    using claimedCancel

theorem builderDefinition_complete {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP) (hone : z 0 = 1)
    (definition : Definition) (canonical : definition.Canonical)
    (holds : Definition.Holds z definition) : RowHolds z definition.builderRow := by
  cases definition with
  | mk output rhs =>
      cases rhs with
      | linear terms =>
          exact builderLinearRow_complete hone output terms canonical holds
      | product left right =>
          have outputLt := hcanon output
          simpa [Definition.Holds, Definition.builderRow, Rhs.eval, RowHolds,
            lcEval, Nat.mod_eq_of_lt outputLt] using holds.symm

/-- Satisfaction of the normalized definition rows yields every SSA equation. -/
theorem definitions_sound {definitions : List Definition} {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP) (hone : z 0 = 1)
    (hsat : Satisfies (definitions.map Definition.row) z) :
    ∀ definition ∈ definitions, Definition.Holds z definition := by
  intro definition member
  apply definition_sound hcanon hone definition
  exact hsat _ (List.mem_map.mpr ⟨definition, member, rfl⟩)

theorem builderDefinitions_sound {definitions : List Definition} {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP) (hone : z 0 = 1)
    (canonical : ∀ definition ∈ definitions, definition.Canonical)
    (hsat : Satisfies (definitions.map Definition.builderRow) z) :
    ∀ definition ∈ definitions, Definition.Holds z definition := by
  intro definition member
  apply builderDefinition_sound hcanon hone definition (canonical definition member)
  exact hsat _ (List.mem_map.mpr ⟨definition, member, rfl⟩)

theorem builderDefinitions_complete {definitions : List Definition} {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP) (hone : z 0 = 1)
    (canonical : ∀ definition ∈ definitions, definition.Canonical)
    (holds : ∀ definition ∈ definitions, Definition.Holds z definition) :
    Satisfies (definitions.map Definition.builderRow) z := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨definition, member, rowEq⟩
  subst row
  exact builderDefinition_complete hcanon hone definition
    (canonical definition member) (holds definition member)

/-- Core scaling theorem: under checked SSA dependencies, a satisfying
assignment and the executable interpreter agree on every input and derived
column. -/
theorem run_agrees_of_holds {known : List Nat} {definitions : List Definition}
    {z state : Nat → Nat} (wellFormed : WellFormed known definitions)
    (initialAgreement : AgreeOn state z known)
    (holds : ∀ definition ∈ definitions, Definition.Holds z definition) :
    AgreeOn (run state definitions) z (knownAfter known definitions) := by
  induction wellFormed generalizing state with
  | nil known => exact initialAgreement
  | @cons known head tail references fresh rest inductionHypothesis =>
      have rhsAgreement : head.rhs.eval state = head.rhs.eval z :=
        rhsEval_agree initialAgreement head.rhs references
      have headHolds : Definition.Holds z head := holds head (by simp)
      have nextAgreement : AgreeOn (execute state head) z (head.output :: known) := by
        intro column member
        simp only [List.mem_cons] at member
        rcases member with output | old
        · subst column
          simp only [execute, setColumn_same]
          exact rhsAgreement.trans headHolds.symm
        · have different : column ≠ head.output := by
            intro equal
            apply fresh
            rw [← equal]
            exact old
          rw [execute, setColumn_other state different]
          exact initialAgreement column old
      apply inductionHypothesis nextAgreement
      intro definition member
      exact holds definition (List.mem_cons_of_mem head member)

/-- A well-formed extracted program is extensional in its declared inputs.
Values initially stored in fresh output columns are irrelevant because every
such column is assigned exactly once by `run`. -/
theorem run_congr {known : List Nat} {definitions : List Definition}
    (wellFormed : WellFormed known definitions)
    {left right : Nat → Nat}
    (inputsAgree : AgreeOn left right known) :
    AgreeOn (run left definitions) (run right definitions)
      (knownAfter known definitions) := by
  induction wellFormed generalizing left right with
  | nil => exact inputsAgree
  | @cons known head tail references fresh rest inductionHypothesis =>
      apply inductionHypothesis
      intro column member
      simp only [List.mem_cons] at member
      rcases member with output | old
      · subst column
        simp only [execute, setColumn_same]
        exact rhsEval_agree inputsAgree head.rhs references
      · have different : column ≠ head.output := by
          intro equal
          apply fresh
          rw [← equal]
          exact old
        simpa [execute, setColumn, different] using inputsAgree column old

/-- Combined artifact rule: exact normalized row satisfaction fixes all
derived columns to the deterministic interpreter. -/
theorem run_agrees_of_satisfies {known : List Nat} {definitions : List Definition}
    {z state : Nat → Nat} (wellFormed : WellFormed known definitions)
    (initialAgreement : AgreeOn state z known)
    (hcanon : ∀ column, z column < goldilocksP) (hone : z 0 = 1)
    (hsat : Satisfies (definitions.map Definition.row) z) :
    AgreeOn (run state definitions) z (knownAfter known definitions) :=
  run_agrees_of_holds wellFormed initialAgreement
    (definitions_sound hcanon hone hsat)

/-- Exact Rust-row variant of `run_agrees_of_satisfies`. This is the theorem
the generated Poseidon2/NIFS normalizer can instantiate without changing the
authoritative sparse row representation. -/
theorem run_agrees_of_builder_satisfies
    {known : List Nat} {definitions : List Definition} {z state : Nat → Nat}
    (wellFormed : WellFormed known definitions)
    (initialAgreement : AgreeOn state z known)
    (hcanon : ∀ column, z column < goldilocksP) (hone : z 0 = 1)
    (canonical : ∀ definition ∈ definitions, definition.Canonical)
    (hsat : Satisfies (definitions.map Definition.builderRow) z) :
    AgreeOn (run state definitions) z (knownAfter known definitions) :=
  run_agrees_of_holds wellFormed initialAgreement
    (builderDefinitions_sound hcanon hone canonical hsat)

/-- Exact agreement with a checked SSA execution is sufficient for all of its
builder rows. This is the reverse transport used by compact rewrites: they can
reconstruct the deterministic execution without retaining every source row. -/
theorem run_agrees_implies_builder_satisfies
    {known : List Nat} {definitions : List Definition} {z state : Nat → Nat}
    (wellFormed : WellFormed known definitions)
    (zCanonical : ∀ column, z column < goldilocksP) (zOne : z 0 = 1)
    (canonical : ∀ definition ∈ definitions, definition.Canonical)
    (agreement : AgreeOn (run state definitions) z
      (knownAfter known definitions)) :
    Satisfies (definitions.map Definition.builderRow) z := by
  apply builderDefinitions_complete zCanonical zOne canonical
  induction wellFormed generalizing state with
  | nil => simp
  | @cons known head tail references fresh rest inductionHypothesis =>
      intro definition member
      simp only [List.mem_cons] at member
      rcases member with isHead | inTail
      · subst definition
        apply definitionHolds_of_agree references
        · intro column columnKnown
          have finalKnown :
              column ∈ knownAfter (head.output :: known) tail :=
            mem_knownAfter columnKnown
          exact (agreement column finalKnown).symm.trans
            (run_preserves_known rest (execute state head) column columnKnown)
        · exact definitionHolds_execute references fresh state
      · exact inductionHypothesis (state := execute state head)
          (fun definition member =>
            canonical definition (List.mem_cons_of_mem head member))
          agreement definition inTail

/-- `CIR-COMPLETE` scaling rule for a deterministic row block: interpreting
any canonical input assignment yields a satisfying witness for all exact
builder rows. -/
theorem run_satisfies_builder_rows
    {known : List Nat} {definitions : List Definition} {state : Nat → Nat}
    (wellFormed : WellFormed known definitions)
    (stateCanonical : ∀ column, state column < goldilocksP)
    (constantOne : 0 ∈ known) (hone : state 0 = 1)
    (canonical : ∀ definition ∈ definitions, definition.Canonical) :
    Satisfies (definitions.map Definition.builderRow) (run state definitions) := by
  have finalCanonical := run_canonical (definitions := definitions) stateCanonical
  have preservesKnown := run_preserves_known wellFormed state
  have finalOne : run state definitions 0 = 1 :=
    (preservesKnown 0 constantOne).trans hone
  exact builderDefinitions_complete finalCanonical finalOne canonical
    (run_definitions_hold wellFormed state)

end Nightstream.Implementation.R1CS.Program
