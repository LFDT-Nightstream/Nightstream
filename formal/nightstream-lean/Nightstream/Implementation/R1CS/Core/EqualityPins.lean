import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Contract: universal semantics of lists of Rust `enforce_eq(left, right)`
rows.  Generated artifacts carry only column pairs and exact-row inclusion;
satisfaction derives the equalities.
-/

namespace Nightstream.Implementation.R1CS.EqualityPins

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

def equalityRow (pair : Nat × Nat) : Row :=
  builderLinearRow pair.1 [(pair.2, 1)]

def rows (pairs : List (Nat × Nat)) : List Row :=
  pairs.map equalityRow

/-- Compact representation of a consecutive equality block. -/
structure PairRun where
  leftStart : Nat
  rightStart : Nat
  leftStep : Nat
  rightStep : Nat
  count : Nat
deriving DecidableEq, Repr

def PairRun.pairs (run : PairRun) : List (Nat × Nat) :=
  (List.range run.count).map fun offset =>
    (run.leftStart + offset * run.leftStep,
      run.rightStart + offset * run.rightStep)

def expandRuns (runs : List PairRun) : List (Nat × Nat) :=
  runs.flatMap PairRun.pairs

theorem PairRun.pairs_length (run : PairRun) :
    run.pairs.length = run.count := by
  simp [PairRun.pairs]

theorem expandRuns_length (runs : List PairRun) :
    (expandRuns runs).length = (runs.map PairRun.count).sum := by
  induction runs with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [expandRuns, PairRun.pairs_length]

def transferPins (pairs sourcePins : List (Nat × Nat)) : List (Nat × Nat) :=
  pairs.map fun pair => (pair.1, ConstantPins.lookup sourcePins pair.2)

def SourcesCovered (pairs sourcePins : List (Nat × Nat)) : Prop :=
  ∀ pair ∈ pairs, ∃ pin ∈ sourcePins, pin.1 = pair.2

def SourceKeysCovered (pairs : List (Nat × Nat)) (sourceKeys : List Nat) : Prop :=
  ∀ pair ∈ pairs, pair.2 ∈ sourceKeys

instance (pairs sourcePins : List (Nat × Nat)) :
    Decidable (SourcesCovered pairs sourcePins) := by
  unfold SourcesCovered
  infer_instance

instance (pairs : List (Nat × Nat)) (sourceKeys : List Nat) :
    Decidable (SourceKeysCovered pairs sourceKeys) := by
  unfold SourceKeysCovered
  infer_instance

theorem sourcesCovered_iff_keys {pairs sourcePins : List (Nat × Nat)} :
    SourcesCovered pairs sourcePins ↔
      SourceKeysCovered pairs (ConstantPins.keys sourcePins) := by
  constructor
  · intro covered pair member
    rcases covered pair member with ⟨pin, pinMember, key⟩
    exact List.mem_map.mpr ⟨pin, pinMember, key⟩
  · intro covered pair member
    rcases List.mem_map.mp (covered pair member) with
      ⟨pin, pinMember, key⟩
    exact ⟨pin, pinMember, key⟩

theorem transferPins_keys (pairs sourcePins : List (Nat × Nat)) :
    ConstantPins.keys (transferPins pairs sourcePins) = pairs.map Prod.fst := by
  simp [ConstantPins.keys, transferPins, List.map_map, Function.comp_def]

private theorem equalityRow_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {pair : Nat × Nat}
    (holds : RowHolds assignment (equalityRow pair)) :
    assignment pair.1 = assignment pair.2 := by
  have defined := builderLinearRow_sound canonical one pair.1 [(pair.2, 1)]
    (by simp [CanonicalTerms]; decide) holds
  simpa [lcEval, Nat.mod_eq_of_lt (canonical pair.2)] using defined

theorem rows_sound
    {pairs : List (Nat × Nat)} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows pairs) assignment) :
    ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 := by
  intro pair member
  apply equalityRow_sound canonical one
  exact satisfies _ (List.mem_map.mpr ⟨pair, member, rfl⟩)

private theorem equalityRow_complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {pair : Nat × Nat}
    (equal : assignment pair.1 = assignment pair.2) :
    RowHolds assignment (equalityRow pair) := by
  apply builderLinearRow_complete one pair.1 [(pair.2, 1)]
    (by simp [CanonicalTerms]; decide)
  simpa [lcEval, Nat.mod_eq_of_lt (canonical pair.2)] using equal

theorem rows_complete
    {pairs : List (Nat × Nat)} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (equalities : ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2) :
    Satisfies (rows pairs) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨pair, pairMember, rfl⟩
  exact equalityRow_complete canonical one (equalities pair pairMember)

theorem sound
    {pairs : List (Nat × Nat)} {programRows : List Row}
    {assignment : Nat → Nat}
    (included : rowsIncluded (rows pairs) programRows = true)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 := by
  intro pair member
  apply equalityRow_sound canonical one
  apply satisfies
  apply rowsIncluded_sound included
  exact List.mem_map.mpr ⟨pair, member, rfl⟩

theorem transfer_sound
    {pairs sourcePins : List (Nat × Nat)} {assignment : Nat → Nat}
    (equalities : ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2)
    (sourceFacts : ∀ pin ∈ sourcePins, assignment pin.1 = pin.2)
    (sourcesCovered : SourcesCovered pairs sourcePins) :
    ∀ pin ∈ transferPins pairs sourcePins, assignment pin.1 = pin.2 := by
  intro pin member
  rcases List.mem_map.mp member with ⟨pair, pairMember, pairEq⟩
  subst pin
  exact (equalities pair pairMember).trans
    (sourceFacts (pair.2, ConstantPins.lookup sourcePins pair.2)
      (ConstantPins.lookup_pair_mem (sourcesCovered pair pairMember)))

end Nightstream.Implementation.R1CS.EqualityPins
