import Nightstream.Protocol.Nebula.Fingerprint

/-!
Contract: exact delayed-claim indexes for the V2 F-prime lifecycle.

Assurance tier: model-level.

Owns the factor-one claim, fold, augmented-invocation, segment, and step
indexes. It makes the base and terminal off-by-one rules explicit.

Does not own NIFS soundness, memory-carry transition contents, recursive-size
closure, or the terminal backend.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Lifecycle

def claimsPerSegment : Nat := 1088
def maximumSegments : Nat := 64

def totalClaims (segmentCount : Nat) : Nat :=
  segmentCount * claimsPerSegment

/-- An augmented invocation index is in `0..T`, inclusive. -/
abbrev InvocationIndex (claimCount : Nat) := Fin (claimCount + 1)

/-- Invocation `i<T` produces `C[i]`; terminal invocation `T` produces none. -/
def producedAt
    {claimCount : Nat} (invocation : InvocationIndex claimCount) :
    Option (Fin claimCount) :=
  if beforeTerminal : invocation.val < claimCount then
    some ⟨invocation.val, beforeTerminal⟩
  else
    none

/-- Every invocation after base consumes the immediately prior claim. -/
def consumedAt
    {claimCount : Nat} (invocation : InvocationIndex claimCount) :
    Option (Fin claimCount) :=
  if afterBase : 0 < invocation.val then
    some ⟨invocation.val - 1, by omega⟩
  else
    none

def baseIndex (claimCount : Nat) : InvocationIndex claimCount :=
  ⟨0, by omega⟩

def terminalIndex (claimCount : Nat) : InvocationIndex claimCount :=
  ⟨claimCount, by omega⟩

theorem base_consumes_none (claimCount : Nat) :
    consumedAt (baseIndex claimCount) = none := by
  simp [consumedAt, baseIndex]

theorem base_produces_first
    {claimCount : Nat} (positive : 0 < claimCount) :
    producedAt (baseIndex claimCount) = some ⟨0, positive⟩ := by
  simp [producedAt, baseIndex, positive]

theorem recursive_consumes_prior_and_produces_current
    {claimCount : Nat}
    (invocation : InvocationIndex claimCount)
    (afterBase : 0 < invocation.val)
    (beforeTerminal : invocation.val < claimCount) :
    consumedAt invocation =
        some ⟨invocation.val - 1, by omega⟩ ∧
      producedAt invocation = some ⟨invocation.val, beforeTerminal⟩ := by
  simp [consumedAt, producedAt, afterBase, beforeTerminal]

theorem terminal_produces_none (claimCount : Nat) :
    producedAt (terminalIndex claimCount) = none := by
  simp [producedAt, terminalIndex]

/-- The terminal invocation consumes the trailing claim `C[T-1]`. -/
theorem terminal_consumes_trailing
    {claimCount : Nat} (positive : 0 < claimCount) :
    consumedAt (terminalIndex claimCount) =
      some ⟨claimCount - 1, by omega⟩ := by
  simp [consumedAt, terminalIndex, positive]

def claimSegment (claimIndex : Nat) : Nat :=
  claimIndex / claimsPerSegment

def claimStep (claimIndex : Nat) : Nat :=
  claimIndex % claimsPerSegment

theorem claimStep_lt (claimIndex : Nat) :
    claimStep claimIndex < claimsPerSegment := by
  unfold claimStep claimsPerSegment
  exact Nat.mod_lt _ (by decide)

theorem claim_index_decompose (claimIndex : Nat) :
    claimSegment claimIndex * claimsPerSegment + claimStep claimIndex =
      claimIndex := by
  unfold claimSegment claimStep
  simpa [Nat.mul_comm] using Nat.div_add_mod claimIndex claimsPerSegment

theorem segment_boundary_locations
    {segmentIndex : Nat} (positive : 0 < segmentIndex) :
    claimSegment (segmentIndex * claimsPerSegment - 1) = segmentIndex - 1 ∧
      claimStep (segmentIndex * claimsPerSegment - 1) =
        claimsPerSegment - 1 ∧
      claimSegment (segmentIndex * claimsPerSegment) = segmentIndex ∧
      claimStep (segmentIndex * claimsPerSegment) = 0 := by
  have priorIndex :
      segmentIndex * claimsPerSegment - 1 =
        (segmentIndex - 1) * claimsPerSegment + (claimsPerSegment - 1) := by
    unfold claimsPerSegment
    omega
  rw [priorIndex]
  unfold claimSegment claimStep claimsPerSegment
  constructor
  · rw [Nat.mul_comm (segmentIndex - 1) 1088, Nat.add_comm]
    rw [Nat.add_mul_div_left 1087 (segmentIndex - 1) (by decide)]
    simp
  constructor
  · rw [Nat.mul_comm (segmentIndex - 1) 1088, Nat.add_comm]
    simp
  constructor <;> simp

theorem final_claim_location
    {segmentCount : Nat} (positive : 0 < segmentCount) :
    claimSegment (totalClaims segmentCount - 1) = segmentCount - 1 ∧
      claimStep (totalClaims segmentCount - 1) = claimsPerSegment - 1 := by
  unfold totalClaims
  exact ⟨(segment_boundary_locations positive).1,
    (segment_boundary_locations positive).2.1⟩

theorem augmented_invocation_count (claimCount : Nat) :
    List.length (List.range (claimCount + 1)) = claimCount + 1 := by
  simp

theorem fold_invocation_count (claimCount : Nat) :
    List.length ((List.range claimCount).map (fun index => index + 1)) =
      claimCount := by
  simp

/-- The four endpoint facts that distinguish the delayed F-prime schedule
from a schedule that stops before it consumes the trailing claim. -/
structure CompleteSchedule (claimCount : Nat) : Prop where
  positive : 0 < claimCount
  baseConsumesNone : consumedAt (baseIndex claimCount) = none
  baseProducesFirst :
    ∃ first : Fin claimCount,
      producedAt (baseIndex claimCount) = some first ∧ first.val = 0
  terminalConsumesLast :
    ∃ last : Fin claimCount,
      consumedAt (terminalIndex claimCount) = some last ∧
        last.val + 1 = claimCount
  terminalProducesNone : producedAt (terminalIndex claimCount) = none

theorem completeSchedule
    {claimCount : Nat} (positive : 0 < claimCount) :
    CompleteSchedule claimCount := by
  refine
    { positive := positive
      baseConsumesNone := base_consumes_none claimCount
      baseProducesFirst := ?_
      terminalConsumesLast := ?_
      terminalProducesNone := terminal_produces_none claimCount }
  · let first : Fin claimCount := ⟨0, positive⟩
    exact ⟨first, base_produces_first positive, rfl⟩
  · let last : Fin claimCount := ⟨claimCount - 1, by omega⟩
    refine ⟨last, terminal_consumes_trailing positive, ?_⟩
    simp only [last]
    omega

theorem maximum_claim_count : totalClaims maximumSegments = 69632 := by
  decide

theorem maximum_augmented_invocation_count :
    totalClaims maximumSegments + 1 = 69633 := by
  decide

end Nightstream.Protocol.Nebula.Lifecycle
