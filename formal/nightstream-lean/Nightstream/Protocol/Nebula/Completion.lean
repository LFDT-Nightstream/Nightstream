import Nightstream.Protocol.Nebula.FPrime

/-!
Contract: canonical completed-execution semantics for V2.

Assurance tier: model-level.

Owns typed returned/trapped outcomes, real-row counting, canonical padding,
and the smallest complete segment count.

Does not own WASM operational semantics, the generated application-state
schema, row-to-port refinement, result codecs, or terminal proof rows.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Completion

open Nightstream.Protocol.Nebula.Lifecycle

def applicationRowsPerClaim : Nat := 3

def applicationRowsPerSegment : Nat :=
  applicationRowsPerClaim * claimsPerSegment

def realApplicationRowLimit : Nat := 2 ^ 18

structure OutputValue where
  low : Nat
  high : Nat
  lowInRange : low < 2 ^ 32
  highInRange : high < 2 ^ 32

inductive Trap where
  | unreachable
  | divisionByZero
  | signedDivisionOverflow
  | indirectCallNull
  | indirectCallTypeMismatch
  | tableOutOfBounds
  | memoryOutOfBounds
deriving DecidableEq, Repr

def Trap.exitCode : Trap → Nat
  | .unreachable => 1
  | .divisionByZero => 2
  | .signedDivisionOverflow => 3
  | .indirectCallNull => 4
  | .indirectCallTypeMismatch => 5
  | .tableOutOfBounds => 6
  | .memoryOutOfBounds => 7

theorem Trap.exitCode_in_range (trap : Trap) :
    1 ≤ trap.exitCode ∧ trap.exitCode ≤ 7 := by
  cases trap <;> decide

theorem Trap.exitCode_injective
    {left right : Trap}
    (equal : left.exitCode = right.exitCode) :
    left = right := by
  cases left <;> cases right <;> simp_all [Trap.exitCode]

inductive Outcome where
  | returned (output : Option OutputValue)
  | trapped (trap : Trap)

def Outcome.exitCode : Outcome → Nat
  | .returned _ => 0
  | .trapped trap => trap.exitCode

inductive RowKind where
  | active
  | returned
  | trapped
  | padding
deriving DecidableEq, Repr

def Outcome.terminalRow : Outcome → RowKind
  | .returned _ => .returned
  | .trapped _ => .trapped

structure ExecutionResult (ApplicationState Digest : Type) where
  realApplicationRowCount : Nat
  finalApplicationState : ApplicationState
  outcome : Outcome
  finalMemoryRoot : Digest

def segmentCapacity (segmentCount : Nat) : Nat :=
  segmentCount * applicationRowsPerSegment

/-- Ceiling division by the nonzero per-segment row capacity. -/
def minimumSegmentCount (realRowCount : Nat) : Nat :=
  (realRowCount + applicationRowsPerSegment - 1) /
    applicationRowsPerSegment

def canonicalRows
    {ApplicationState Digest : Type}
    (result : ExecutionResult ApplicationState Digest)
    (segmentCount : Nat) : List RowKind :=
  List.replicate (result.realApplicationRowCount - 1) .active ++
    [result.outcome.terminalRow] ++
    List.replicate
      (segmentCapacity segmentCount - result.realApplicationRowCount)
      .padding

/-- Complete trace classification. The last real row authenticates the typed
outcome, and all remaining rows are canonical padding. -/
structure ValidCompletedTrace
    {ApplicationState Digest : Type}
    (result : ExecutionResult ApplicationState Digest)
    (segmentCount : Nat)
    (rows : List RowKind) : Prop where
  segmentCountPositive : 0 < segmentCount
  segmentCountBound : segmentCount ≤ maximumSegments
  realRowCountPositive : 0 < result.realApplicationRowCount
  realRowCountBound : result.realApplicationRowCount < realApplicationRowLimit
  fitsDeclaredSegments :
    result.realApplicationRowCount ≤ segmentCapacity segmentCount
  smallestSegmentCount :
    segmentCount = minimumSegmentCount result.realApplicationRowCount
  rowsCanonical : rows = canonicalRows result segmentCount

/-- This predicate omits only the smallest-segment condition. It is used to
show why that condition is necessary. -/
structure ValidWithoutMinimality
    {ApplicationState Digest : Type}
    (result : ExecutionResult ApplicationState Digest)
    (segmentCount : Nat)
    (rows : List RowKind) : Prop where
  segmentCountPositive : 0 < segmentCount
  segmentCountBound : segmentCount ≤ maximumSegments
  realRowCountPositive : 0 < result.realApplicationRowCount
  realRowCountBound : result.realApplicationRowCount < realApplicationRowLimit
  fitsDeclaredSegments :
    result.realApplicationRowCount ≤ segmentCapacity segmentCount
  rowsCanonical : rows = canonicalRows result segmentCount

theorem canonicalRows_length
    {ApplicationState Digest : Type}
    (result : ExecutionResult ApplicationState Digest)
    (segmentCount : Nat)
    (positive : 0 < result.realApplicationRowCount)
    (fits : result.realApplicationRowCount ≤ segmentCapacity segmentCount) :
    (canonicalRows result segmentCount).length =
      segmentCapacity segmentCount := by
  unfold canonicalRows
  simp only [List.length_append, List.length_replicate, List.length_singleton]
  omega

theorem valid_trace_has_exact_capacity
    {ApplicationState Digest : Type}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    {rows : List RowKind}
    (valid : ValidCompletedTrace result segmentCount rows) :
    rows.length = segmentCapacity segmentCount := by
  rw [valid.rowsCanonical]
  exact canonicalRows_length result segmentCount
    valid.realRowCountPositive valid.fitsDeclaredSegments

theorem maximum_application_rows :
    segmentCapacity maximumSegments = 208896 := by
  decide

theorem maximum_application_rows_fit_18_bits :
    segmentCapacity maximumSegments < realApplicationRowLimit := by
  decide

end Nightstream.Protocol.Nebula.Completion
