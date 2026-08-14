import Nightstream.Implementation.Nebula.Core.BoundedWordRows

/-!
Hostile countermodels for terminal public-word recomposition.

The first model omits the recomposition row. The public bit remains fixed,
but the field value can change. The second model shows the exact 64-bit
Goldilocks alias: the binary word `q` satisfies ordinary field recomposition
with output zero. A sound 64-bit bridge must therefore use the independent
fact that the parsed integer is below `q`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace tests.NebulaTerminalStatementRowsCountermodels

open Nightstream.Implementation.Nebula.BoundedWordRows
open Nightstream.Implementation.R1CS

def oneBitLayout : Layout :=
  { width := 1
    valueColumn := 1
    bitStart := 2 }

def goodOneBitAssignment : Nat -> Nat
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | _ => 0

def badOneBitAssignment : Nat -> Nat
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | _ => 0

/-- Bit placement without recomposition leaves the authority-bearing field
value unconstrained. Adding the recomposition row rejects the mismatch. -/
theorem omitted_recomposition_allows_public_field_substitution :
    Satisfies oneBitLayout.bitRows goodOneBitAssignment /\
      Satisfies oneBitLayout.bitRows badOneBitAssignment /\
      goodOneBitAssignment 2 = badOneBitAssignment 2 /\
      goodOneBitAssignment 1 ≠ badOneBitAssignment 1 /\
      Satisfies (rows oneBitLayout) goodOneBitAssignment /\
      ¬ Satisfies (rows oneBitLayout) badOneBitAssignment := by
  decide

def word64Layout : Layout :=
  { width := 64
    valueColumn := 1
    bitStart := 2 }

/-- Value column 1 is zero. Columns 2 through 65 contain the little-endian
binary expansion of the Goldilocks modulus. -/
def modulusAliasAssignment : Nat -> Nat
  | 0 => 1
  | 1 => 0
  | Nat.succ (Nat.succ offset) =>
      if offset < 64 then (goldilocksP / 2 ^ offset) % 2 else 0

/-- Ordinary 64-bit field recomposition accepts the noncanonical word `q` as
field zero. This is why the terminal bridge proves `decoded < q` from the
typed public digest before it invokes the row-soundness theorem. -/
theorem modulus_word_aliases_zero_without_integer_bound :
    Satisfies (rows word64Layout) modulusAliasAssignment /\
      decoded word64Layout modulusAliasAssignment = goldilocksP /\
      modulusAliasAssignment word64Layout.valueColumn = 0 /\
      ¬ decoded word64Layout modulusAliasAssignment < goldilocksP := by
  decide

end tests.NebulaTerminalStatementRowsCountermodels
