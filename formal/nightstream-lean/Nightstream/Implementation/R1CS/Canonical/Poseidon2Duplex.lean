import Nightstream.Implementation.R1CS.Canonical.TranscriptModeBoundary

/-!
Contract: the value-level duplex the Fiat–Shamir transcript actually runs.

Owns: the cursored state, overwrite absorption, the permute-and-reset step, the
cursor invariant that makes the write index in range, the capacity guarantee
that follows from it, and the pre-squeeze domain gate.

Does **not** own a row program.  This is the model
`TRANSCRIPT-MODE-BOUNDARY` said had to exist before any transcript recipe could
be written, and it is deliberately value-level: emitting rows before the model
is right is how a recipe ends up computing a different function from the
verifier.

## Transcribed from `neo-transcript/src/poseidon2.rs`

```text
fn absorb_elem(&mut self, x) {
    if self.absorbed >= RATE { self.permute(); }
    self.st[self.absorbed] = x;      // overwrite
    self.absorbed += 1;
}
fn permute(&mut self) { self.st = perm(self.st); self.absorbed = 0; }
```

and, before squeezing a challenge,

```text
self.absorb_elem(ONE);
self.permute();
```

with the source's own comment on that pair: "Domain gate before squeezing to
avoid state reuse issues."

## The cursor invariant is the whole well-formedness argument

`st[self.absorbed]` is only in range because `absorbed ≤ rate` always, and the
guard restores it before every write.  `cursor_le_rate` proves the invariant is
maintained; `write_index_lt_rate` is the consequence that makes the write legal;
`capacity_untouched` is the security-relevant corollary — absorption can never
reach a capacity lane, because the guard fires first.

Rust does not carry this as a proof.  It carries it as an `assert!` in
`from_state_and_absorbed` and as the arithmetic working out.  Here it is a
theorem.

## Not the binding sponge

`TranscriptModeBoundary` proves the two absorb modes diverge from any carried
state.  `duplex_absorb_is_overwrite` ties this model to the overwrite side of
that boundary, so the two constructions cannot be silently conflated later.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Canonical.TranscriptModeBoundary

/-- The duplex state: the permutation state plus the absorb cursor. -/
structure State where
  lanes : Values
  absorbed : Nat

/-- The empty transcript: all-zero state, cursor at zero. -/
def empty : State := ⟨fun _ => 0, 0⟩

/-- Permute and reset the cursor. -/
def permute (constants : Constants) (s : State) : State :=
  ⟨referencePermutation constants s.lanes, 0⟩

theorem permute_absorbed (constants : Constants) (s : State) :
    (permute constants s).absorbed = 0 := rfl

/-- The state a write happens on: permuted first when the rate is full. -/
def guarded (constants : Constants) (s : State) : State :=
  if rate ≤ s.absorbed then permute constants s else s

/-- **Overwrite absorption.**  One element into the lane the cursor names. -/
def absorbElem (constants : Constants) (x : Nat) (s : State) : State :=
  let target := guarded constants s
  ⟨fun lane =>
      if lane.val = target.absorbed then x % goldilocksP else target.lanes lane,
    target.absorbed + 1⟩

/-- Absorb a list, left to right. -/
def absorbList (constants : Constants) : List Nat → State → State
  | [], s => s
  | x :: rest, s => absorbList constants rest (absorbElem constants x s)

/-! ## The cursor invariant

Everything else rests on this. -/

/-- **The guard restores the cursor below the rate, unconditionally.**

Note the absence of a hypothesis.  This is not an invariant that callers must
maintain: either the guard fires and the cursor becomes `0`, or it did not fire
and the cursor was already below the rate.  Rust carries the same fact as an
`assert!` on a reconstructed state; here nothing has to be assumed. -/
theorem guarded_absorbed_lt (constants : Constants) (s : State) :
    (guarded constants s).absorbed < rate := by
  unfold guarded
  by_cases full : rate ≤ s.absorbed
  · rw [if_pos full]
    simp only [permute]
    decide
  · rw [if_neg full]
    omega

/-- **The write index is always in range.**  This is what makes
`st[self.absorbed]` legal in Rust, carried there by an `assert!` and by the
arithmetic working out. -/
theorem write_index_lt_rate (constants : Constants) (s : State) :
    (guarded constants s).absorbed < rate :=
  guarded_absorbed_lt constants s

/-- **The cursor stays in range, whatever it was before.** -/
theorem cursor_le_rate (constants : Constants) (x : Nat) (s : State) :
    (absorbElem constants x s).absorbed ≤ rate := by
  have bound := guarded_absorbed_lt constants s
  simp only [absorbElem]
  omega

/-- **The invariant survives a whole list.** -/
theorem cursor_le_rate_list (constants : Constants) (input : List Nat) :
    ∀ s : State, s.absorbed ≤ rate → (absorbList constants input s).absorbed ≤ rate := by
  induction input with
  | nil => intro s invariant; exact invariant
  | cons x rest inductionHypothesis =>
      intro s invariant
      exact inductionHypothesis _ (cursor_le_rate constants x s)

theorem empty_cursor : empty.absorbed ≤ rate := by
  simp only [empty]
  decide

/-! ## The capacity guarantee

Absorption can never reach a capacity lane, because the guard fires first. -/

/-- **No capacity lane is written by absorption.**

The security-relevant corollary of the cursor invariant: with the cursor below
the rate at every write, the overwritten lane is a rate lane, so the capacity
carries only what the permutation put there. -/
theorem capacity_untouched
    (constants : Constants) (x : Nat) (s : State)
    (lane : Fin width) (isCapacity : rate ≤ lane.val) :
    (absorbElem constants x s).lanes lane = (guarded constants s).lanes lane := by
  have bound := guarded_absorbed_lt constants s
  simp only [absorbElem]
  rw [if_neg (by omega)]

/-! ## The pre-squeeze domain gate

`absorb_elem(ONE); permute();` — the source's own comment calls it a "domain
gate before squeezing to avoid state reuse issues".  It is the transcript's
separation mechanism, and it is *not* the binding sponge's padding rule: this
one fires before every squeeze, that one once at the end. -/

/-- The gate: absorb a one, then permute. -/
def gate (constants : Constants) (s : State) : State :=
  permute constants (absorbElem constants 1 s)

/-- The gate leaves the cursor at zero, so a squeeze always reads a freshly
permuted state. -/
theorem gate_absorbed (constants : Constants) (s : State) :
    (gate constants s).absorbed = 0 := rfl

/-- One challenge field element: the gated state's lane zero. -/
def challengeField (constants : Constants) (s : State) : Nat × State :=
  let gated := gate constants s
  (gated.lanes ⟨0, by decide⟩, gated)

theorem challengeField_state (constants : Constants) (s : State) :
    (challengeField constants s).2 = gate constants s := rfl

/-- **A challenge is read from a permuted state**, never from one that
absorption has just written into. -/
theorem challengeField_cursor (constants : Constants) (s : State) :
    (challengeField constants s).2.absorbed = 0 := rfl

/-! ## Tied to the mode boundary

So the two constructions cannot be silently conflated later. -/

/-- **This model's absorption is the overwrite mode**, on the lane the cursor
names — the side of `TranscriptModeBoundary` the transcript is on, not the
binding sponge's. -/
theorem duplex_absorb_is_overwrite
    (constants : Constants) (x : Nat) (s : State) :
    (absorbElem constants x s).lanes
        ⟨(guarded constants s).absorbed,
          Nat.lt_of_lt_of_le (guarded_absorbed_lt constants s) (by decide)⟩
      = x % goldilocksP := by
  simp only [absorbElem]
  simp

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex
