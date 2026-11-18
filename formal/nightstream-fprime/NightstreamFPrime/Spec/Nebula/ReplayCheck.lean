import NightstreamFPrime.Spec.Nebula.MemoryTrace

/-! Owns the decision procedures of the memory-event copy: replay as a function, equal to
`Replay`, and the decidability of `boundedMemoryEvents`. It does not own the golden traces. -/

namespace NightstreamFPrime.Spec.Nebula.MemoryTrace

/-- The prior check of a write as a test. -/
def priorHolds (state : MemoryState) (memory : ℕ) (address : Address) : Option ℤ → Bool
  | none => true
  | some value => state memory address == some value

/-- The state after the events, or `none` when an event does not step. -/
def replay? (state : MemoryState) : List Event → Option MemoryState
  | [] => some state
  | .read memory address value :: rest =>
    if state memory address == some value then replay? state rest else none
  | .write memory address value prior :: rest =>
    if priorHolds state memory address prior then replay? (update state memory address value) rest
    else none

private theorem priorHolds_iff {state : MemoryState} {memory : ℕ} {address : Address}
    {prior : Option ℤ} : priorHolds state memory address prior = true ↔
      priorMatches state memory address prior := by
  cases prior <;> simp [priorHolds, priorMatches]

/-- `replay?` is `Replay`. -/
theorem replay?_eq_some_iff {state final : MemoryState} {events : List Event} :
    replay? state events = some final ↔ Replay state events final := by
  induction events generalizing state with
  | nil =>
    constructor
    · intro h
      cases h
      exact .nil _
    · rintro ⟨⟩
      rfl
  | cons event rest ih =>
    constructor
    · intro h
      cases event with
      | read memory address value =>
        simp only [replay?] at h
        split at h
        next holds => exact .cons (.read (by simpa using holds)) (ih.mp h)
        next => cases h
      | write memory address value prior =>
        simp only [replay?] at h
        split at h
        next holds => exact .cons (.write (priorHolds_iff.mp holds)) (ih.mp h)
        next => cases h
    · intro h
      cases h with
      | cons step tail =>
        cases step with
        | read holds => simp [replay?, holds, ih.mpr tail]
        | write holds => simp [replay?, priorHolds_iff.mpr holds, ih.mpr tail]

/-- Memory consistency is a successful `replay?`. -/
theorem memoryConsistent_iff {catalog : List MemorySpec} {initial : MemoryState}
    {rows : List Witness} :
    MemoryConsistent catalog initial rows ↔ (replay? initial (eventsForTrace catalog rows)).isSome := by
  unfold MemoryConsistent
  constructor
  · rintro ⟨final, replay⟩
    rw [replay?_eq_some_iff.mpr replay]
    rfl
  · intro some
    obtain ⟨final, shown⟩ := Option.isSome_iff_exists.mp some
    exact ⟨final, replay?_eq_some_iff.mp shown⟩

instance : DecidablePred u32Value := fun _ => by unfold u32Value; infer_instance

instance : DecidablePred eventValuesBounded := fun e => by
  cases e with
  | read => unfold eventValuesBounded; infer_instance
  | write _ _ _ prior => cases prior <;> (unfold eventValuesBounded; infer_instance)

instance : DecidablePred boundedMemoryEvents := fun _ => by
  unfold boundedMemoryEvents; infer_instance

end NightstreamFPrime.Spec.Nebula.MemoryTrace
