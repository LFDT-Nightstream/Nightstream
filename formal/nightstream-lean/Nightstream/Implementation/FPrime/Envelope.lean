import Nightstream.HyperNova.Construction2.State

/-!
Contract: F' state-envelope integrity.

Assumes: decidable equality for the digest representation.
Guarantees: a successful executable check establishes the exact branch,
counter, fixed-program-counter, immutable-boundary, and trace-copy equations
implemented by `state_base_case_check` and `advance_state`.

Non-goals: NIFS correctness, application-state authenticity, accumulator
opening authority, Poseidon2 collision resistance, Nebula transition validity,
R1CS soundness, and whole-Rust-function refinement.

Maps to:
- `crates/neo-fold-clean/src/paper/construction2/transition.rs`
  (`enforce_pc_in_range`, `state_base_case_check`, `advance_state`)
- `crates/neo-fold-clean/src/paper/f_prime/native.rs`
  (`prove`, `verify` branch selection)
-/

namespace Nightstream.Implementation.FPrime.Envelope

open Nightstream.HyperNova.Construction2

universe uDigest uRunning uFresh

abbrev Carrier
    (Digest : Type uDigest)
    (Running : Type uRunning)
    (Fresh : Type uFresh) :=
  State Digest Running Fresh

/-- Input-state branch coherence checked before native F' verification. -/
def InputCoherent
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    (state : Carrier Digest Running Fresh) : Prop :=
  state.pc = 1 ∧
  match state.proof with
  | .initial =>
      state.chunkCount = 0 ∧
      state.stepCount = 0 ∧
      state.z0 = state.zi
  | .active _ _ =>
      state.chunkCount ≠ 0 ∧
      state.stepCount ≠ 0

/-- Public equations owned by `advance_state`; content authority is separate. -/
def AdvanceCoherent
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    (freshCount : Nat)
    (prior next : Carrier Digest Running Fresh) : Prop :=
  next.chunkCount = prior.chunkCount + 1 ∧
  next.stepCount = prior.stepCount + freshCount ∧
  next.z0 = prior.z0 ∧
  next.pc = prior.pc ∧
  next.initialSemanticState = prior.initialSemanticState ∧
  next.publicTrace = next.zi ∧
  match next.proof with
  | .initial => False
  | .active _ _ => True

/-- Full scope of the first executable assurance check. -/
def Holds
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    (freshCount : Nat)
    (prior next : Carrier Digest Running Fresh) : Prop :=
  InputCoherent prior ∧ AdvanceCoherent freshCount prior next

private instance holdsDecidable
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    [DecidableEq Digest]
    (freshCount : Nat)
    (prior next : Carrier Digest Running Fresh) :
    Decidable (Holds freshCount prior next) := by
  unfold Holds InputCoherent AdvanceCoherent
  cases prior.proof <;> cases next.proof <;> infer_instance

/-- Executable checker for the theorem's deliberately narrow envelope. -/
def check
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    [DecidableEq Digest]
    (freshCount : Nat)
    (prior next : Carrier Digest Running Fresh) : Bool :=
  decide (Holds freshCount prior next)

/--
An accepted envelope cannot conceal a base/active, counter, `pc`, or immutable
boundary mismatch. The conclusion is computed by `check`; it is not supplied by
the caller as an authority field.
-/
theorem check_sound
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    [DecidableEq Digest]
    {freshCount : Nat}
    {prior next : Carrier Digest Running Fresh}
    (accepted : check freshCount prior next = true) :
    Holds freshCount prior next := by
  unfold check at accepted
  exact of_decide_eq_true accepted

end Nightstream.Implementation.FPrime.Envelope
