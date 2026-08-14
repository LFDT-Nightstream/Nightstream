import Nightstream.Implementation.Nebula.Memory.Carry.PublicRows
import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: one exact numeric R1CS row that requires the V2 terminal
intermediate carry to be closed.

Assurance tier: implementation-to-protocol bridge.

Owns the unconditional phase-equals-zero row and its link to the exact parsed
carry phase. The complete carry validator separately forces every closed-only
inactive field to its canonical zero encoding.

Does not own delayed claim consumption, product balance, carry parsing,
terminal opening, or public-result checks.

Emits constraints: one row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.TerminalClosedCarryRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

structure Layout where
  carry : MemoryCarryPublicRows.Layout

def Layout.phaseColumn (layout : Layout) : Nat :=
  layout.carry.carry.fieldColumn .phase

def row (layout : Layout) : Row :=
  builderLinearRow layout.phaseColumn []

def rows (layout : Layout) : List Row :=
  [row layout]

theorem rows_length (layout : Layout) :
    (rows layout).length = 1 :=
  rfl

theorem phase_zero
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment layout.phaseColumn = 0 := by
  have holds : RowHolds assignment (row layout) :=
    satisfied _ (by simp [rows])
  have exact := builderLinearRow_sound canonical one layout.phaseColumn []
    (by simp [CanonicalTerms]) holds
  simpa [lcEval] using exact

/-- The exact parsed carry is closed. This conclusion is derived from the row
and the parser-column link; it is not a caller-selected phase premise. -/
theorem parsed_phase_closed
    {layout : Layout} {assignment : Nat → Nat}
    {headers : Nightstream.Protocol.Nebula.FPrime.ChainHeaders
      Nightstream.Protocol.Nebula.Digest.Value}
    {value : MemoryCarryCodec.Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryCarryPublicRows.ParsedColumnsMatch
      layout.carry assignment headers value)
    (satisfied : Satisfies (rows layout) assignment) :
    value.phase = .closed := by
  have zero := phase_zero canonical one satisfied
  have placed := parsed.placed MemoryCarryCodec.FieldTag.phase
  have valueZero : value.fieldValue .phase = 0 := placed.symm.trans zero
  cases phaseExact : value.phase with
  | closed => rfl
  | active =>
      simp [MemoryCarryCodec.Value.fieldValue,
        MemoryCarryCodec.phaseValue, phaseExact] at valueZero

theorem rows_honest
    {layout : Layout} {assignment : Nat → Nat}
    (phaseZero : assignment layout.phaseColumn = 0) :
    Satisfies (rows layout) assignment := by
  intro candidate member
  have equal : candidate = row layout := by simpa [rows] using member
  subst candidate
  simp [row, builderLinearRow, negateTerms, RowHolds, lcEval, phaseZero]

end Nightstream.Implementation.Nebula.TerminalClosedCarryRows
