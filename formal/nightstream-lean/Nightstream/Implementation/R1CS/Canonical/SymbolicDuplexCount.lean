import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplex

/-!
Contract: count symbolic Poseidon2 duplex permutations without constructing
field expressions, permutation entries, or R1CS rows.

Assurance tier: model-level canonical encoding.

Owns: the value-independent entry-count and cursor recurrence, and its exact
refinement to `SymbolicDuplex.Builder`.

Does not own: a protocol transcript, field serialization, physical columns,
rows, or a security claim.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCount

open Nightstream.Implementation.R1CS.Canonical

/-- Complete value-independent duplex control state. -/
structure Control where
  entries : Nat
  absorbed : Nat
deriving DecidableEq, Repr

/-- Project the control state of a symbolic builder. -/
def ofBuilder (builder : SymbolicDuplex.Builder) : Control :=
  ⟨builder.entries.length, builder.absorbed⟩

/-- One forced permutation adds one entry and resets the rate cursor. -/
def permute (control : Control) : Control :=
  ⟨control.entries + 1, 0⟩

/-- Restore the rate cursor before one overwrite absorption. -/
def guarded (control : Control) : Control :=
  if Poseidon2Sponge.rate ≤ control.absorbed
  then permute control
  else control

/-- Count one overwrite absorption. -/
def absorb (control : Control) : Control :=
  let ready := guarded control
  ⟨ready.entries, ready.absorbed + 1⟩

/-- Count an exact number of consecutive absorptions. -/
def absorbMany : Nat → Control → Control
  | 0, control => control
  | count + 1, control => absorbMany count (absorb control)

/-- Constant-time form of `absorbMany`. The first absorption applies the
same guard as `absorb`; the quotient counts later rate crossings. -/
def absorbManyFast : Nat → Control → Control
  | 0, control => control
  | count + 1, control =>
      let ready := guarded control
      ⟨ready.entries +
          (ready.absorbed + count) / Poseidon2Sponge.rate,
        (ready.absorbed + count) % Poseidon2Sponge.rate + 1⟩

/-- Count the pre-squeeze one absorption and forced permutation. -/
def gate (control : Control) : Control :=
  permute (absorb control)

private theorem guarded_absorbed_lt (control : Control) :
    (guarded control).absorbed < Poseidon2Sponge.rate := by
  unfold guarded
  split
  · simp [permute, Poseidon2Sponge.rate]
  · omega

private theorem absorbManyFast_succ
    (count : Nat) (control : Control) :
    absorbManyFast (count + 1) control =
      absorbManyFast count (absorb control) := by
  let ready := guarded control
  have readyLt : ready.absorbed < Poseidon2Sponge.rate := by
    exact guarded_absorbed_lt control
  change
    {
      entries :=
        ready.entries +
          (ready.absorbed + count) / Poseidon2Sponge.rate
      absorbed :=
        (ready.absorbed + count) % Poseidon2Sponge.rate + 1
    } =
      absorbManyFast count
        {
          entries := ready.entries
          absorbed := ready.absorbed + 1
        }
  have readyCases :
      ready.absorbed = 0 ∨ ready.absorbed = 1 ∨
        ready.absorbed = 2 ∨ ready.absorbed = 3 := by
    simp only [Poseidon2Sponge.rate] at readyLt
    omega
  rcases readyCases with readyZero | readyOne | readyTwo | readyThree
  ·
    rw [readyZero]
    cases count with
    | zero =>
        rfl
    | succ count =>
        simp [absorbManyFast, guarded, Poseidon2Sponge.rate] <;> omega
  ·
    rw [readyOne]
    cases count with
    | zero =>
        rfl
    | succ count =>
        simp [absorbManyFast, guarded, Poseidon2Sponge.rate] <;> omega
  ·
    rw [readyTwo]
    cases count with
    | zero =>
        rfl
    | succ count =>
        simp [absorbManyFast, guarded, Poseidon2Sponge.rate] <;> omega
  ·
    rw [readyThree]
    cases count with
    | zero =>
        rfl
    | succ count =>
        simp [absorbManyFast, guarded, permute, Poseidon2Sponge.rate,
          Nat.add_div_right, Nat.add_mod] <;> omega

/-- The constant-time counter is exactly the recursive duplex control. -/
theorem absorbMany_eq_fast (count : Nat) (control : Control) :
    absorbMany count control = absorbManyFast count control := by
  induction count generalizing control with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [absorbMany, inductionHypothesis]
      exact (absorbManyFast_succ count control).symm

@[simp] theorem ofBuilder_permute
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    ofBuilder (SymbolicDuplex.permute base builder) =
      permute (ofBuilder builder) := by
  simp [ofBuilder, permute]

@[simp] theorem ofBuilder_guarded
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    ofBuilder (SymbolicDuplex.guarded base builder) =
      guarded (ofBuilder builder) := by
  unfold SymbolicDuplex.guarded guarded ofBuilder
  by_cases full : Poseidon2Sponge.rate ≤ builder.absorbed
  · simp [full, SymbolicDuplex.permute, permute]
  · simp [full]

@[simp] theorem ofBuilder_absorb
    (base : Nat) (value : LinCombNormal.LinComb)
    (builder : SymbolicDuplex.Builder) :
    ofBuilder (SymbolicDuplex.absorb base value builder) =
      absorb (ofBuilder builder) := by
  let ready := SymbolicDuplex.guarded base builder
  let counted := guarded (ofBuilder builder)
  have readyCount : ofBuilder ready = counted := by
    exact ofBuilder_guarded base builder
  have entriesEqual : ready.entries.length = counted.entries :=
    congrArg Control.entries readyCount
  have absorbedEqual : ready.absorbed = counted.absorbed :=
    congrArg Control.absorbed readyCount
  change
    Control.mk ready.entries.length (ready.absorbed + 1) =
      Control.mk counted.entries (counted.absorbed + 1)
  rw [entriesEqual, absorbedEqual]

/-- Field values do not affect either control coordinate. -/
theorem ofBuilder_absorbMany
    (base : Nat) :
    ∀ (values : List LinCombNormal.LinComb)
      (builder : SymbolicDuplex.Builder),
      ofBuilder (SymbolicDuplex.absorbMany base values builder) =
        absorbMany values.length (ofBuilder builder)
  | [], _ => rfl
  | value :: rest, builder => by
      simp only [SymbolicDuplex.absorbMany, List.length_cons, absorbMany]
      rw [ofBuilder_absorbMany base rest, ofBuilder_absorb]

@[simp] theorem ofBuilder_gate
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    ofBuilder (SymbolicDuplex.gate base builder) =
      gate (ofBuilder builder) := by
  unfold SymbolicDuplex.gate gate
  rw [ofBuilder_permute, ofBuilder_absorb]

end Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCount
