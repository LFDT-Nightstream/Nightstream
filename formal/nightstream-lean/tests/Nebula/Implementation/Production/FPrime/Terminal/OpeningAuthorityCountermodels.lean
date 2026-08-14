import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Hostile countermodels for two terminal-opening authority failures.

The first model omits the row that aliases a terminal commitment coordinate
to the verified trailing NIFS output. The NIFS row and the opening row then
accept different values in one assignment.

The second model evaluates the same column name in two different assignments.
Both local row sets accept, even though their values disagree. This shows why
all terminal children and the trailing NIFS carrier must use one assignment.
-/

set_option autoImplicit false

namespace tests.NebulaProductionTerminalOpeningAuthorityCountermodels

open Nightstream.Implementation.R1CS

def nifsOutputRow : Row :=
  ⟨[(2, 1)], [(0, 1)], [(0, 5)]⟩

def openingCommitmentRow : Row :=
  ⟨[(3, 1)], [(0, 1)], [(0, 7)]⟩

/-- This is the missing authority row in the first hostile model. -/
def commitmentAliasRow : Row :=
  ⟨[(3, 1)], [(0, 1)], [(2, 1)]⟩

def wrongSingleAssignment : Nat -> Nat
  | 0 => 1
  | 2 => 5
  | 3 => 7
  | _ => 0

/-- Local NIFS and opening checks do not bind each other when the exact alias
row is absent. Adding that row rejects the false terminal opening. -/
theorem omitted_commitment_alias_accepts_wrong_opening :
    Satisfies [nifsOutputRow, openingCommitmentRow] wrongSingleAssignment /\
      wrongSingleAssignment 2 ≠ wrongSingleAssignment 3 /\
      ¬ Satisfies [nifsOutputRow, openingCommitmentRow, commitmentAliasRow]
        wrongSingleAssignment := by
  decide

/-- Both local checks intentionally use column 2. They are unsafe when each is
evaluated against a different assignment. -/
def openingOnSharedColumnRow : Row :=
  ⟨[(2, 1)], [(0, 1)], [(0, 7)]⟩

def nifsAssignment : Nat -> Nat
  | 0 => 1
  | 2 => 5
  | _ => 0

def openingAssignment : Nat -> Nat
  | 0 => 1
  | 2 => 7
  | _ => 0

/-- Reusing a column identifier is not an authority link when the two row sets
are evaluated on different assignments. -/
theorem separate_assignments_accept_conflicting_shared_column :
    Satisfies [nifsOutputRow] nifsAssignment /\
      Satisfies [openingOnSharedColumnRow] openingAssignment /\
      nifsAssignment 2 ≠ openingAssignment 2 /\
      ¬ Satisfies [nifsOutputRow, openingOnSharedColumnRow] nifsAssignment /\
      ¬ Satisfies [nifsOutputRow, openingOnSharedColumnRow] openingAssignment := by
  decide

end tests.NebulaProductionTerminalOpeningAuthorityCountermodels
