import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Hostile countermodel for an omitted terminal public-result link.

The retained row constrains an unrelated column. The terminal value column is
free, so two canonical assignments satisfy the retained relation and disagree
on the terminal value. Adding the missing equality row rejects the mismatch.
The same construction applies to invocation count, both application-state
vectors, real-row count, segment count, timestamp, and each memory-root lane.
-/

set_option autoImplicit false

namespace tests.NebulaV2TerminalPublicRowsCountermodels

open Nightstream.Implementation.R1CS

def retainedRow : Row :=
  ⟨[(1, 1)], [(0, 1)], [(0, 7)]⟩

def terminalLink : Row :=
  ⟨[(2, 1)], [(0, 1)], [(3, 1)]⟩

def retainedRows : List Row := [retainedRow]

def completeRows : List Row := [retainedRow, terminalLink]

def goodAssignment : Nat -> Nat
  | 0 => 1
  | 1 => 7
  | 2 => 11
  | 3 => 11
  | _ => 0

def badAssignment : Nat -> Nat
  | 0 => 1
  | 1 => 7
  | 2 => 11
  | 3 => 12
  | _ => 0

/-- An omitted terminal link permits a verifier-owned public value to differ
from the terminal state while all retained rows still hold. -/
theorem omitted_terminal_link_allows_public_mismatch :
    Satisfies retainedRows goodAssignment /\
      Satisfies retainedRows badAssignment /\
      goodAssignment 2 = goodAssignment 3 /\
      badAssignment 2 ≠ badAssignment 3 /\
      Satisfies completeRows goodAssignment /\
      ¬ Satisfies completeRows badAssignment := by
  decide

end tests.NebulaV2TerminalPublicRowsCountermodels
