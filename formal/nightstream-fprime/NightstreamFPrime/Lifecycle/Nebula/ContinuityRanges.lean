import Std

/-! SPEC.md section 7.6. Private output decompositions can be restored from checked
inputs at internal batch boundaries. The payload and input auxiliaries stay fixed.
The representation excludes output auxiliaries from every other constraint and
observable; the Rust recognizer must establish that separation. -/

namespace NightstreamFPrime.Lifecycle.Nebula.ContinuityRanges

variable {Payload Value Aux Key : Type}

def Full (check : Key → Value → Aux → Prop) (output : Payload → Key → Nat → Value)
    (n : Nat) (payload : Payload) (aux : Key → Nat → Aux) : Prop :=
  ∀ k i, i < n → check k (output payload k i) (aux k i)

def Retained (check : Key → Value → Aux → Prop) (output : Payload → Key → Nat → Value)
    (n : Nat) (payload : Payload) (aux : Key → Nat → Aux) : Prop :=
  ∀ k i, i < n → ¬ i + 1 < n → check k (output payload k i) (aux k i)

def restore (n : Nat) (inputAux outputAux : Key → Nat → Aux) (k : Key) (i : Nat) : Aux :=
  if i + 1 < n then inputAux k (i + 1) else outputAux k i

theorem sound
    (check : Key → Value → Aux → Prop)
    (input output : Payload → Key → Nat → Value)
    (rest : Payload → (Key → Nat → Aux) → Prop)
    (n : Nat) (payload : Payload) (inputAux outputAux : Key → Nat → Aux)
    (other : rest payload inputAux)
    (inputs : Full check input n payload inputAux)
    (last : Retained check output n payload outputAux)
    (links : ∀ k i, i + 1 < n → output payload k i = input payload k (i + 1)) :
    rest payload inputAux ∧ Full check input n payload inputAux ∧
      Full check output n payload (restore n inputAux outputAux) := by
  refine ⟨other, inputs, ?_⟩
  intro k i inside
  by_cases internal : i + 1 < n
  · simp only [restore, ite_eq_left internal, links k i internal]
    exact inputs k (i + 1) internal
  · simp only [restore, ite_eq_right internal]
    exact last k i inside internal

theorem complete
    (check : Key → Value → Aux → Prop) (output : Payload → Key → Nat → Value)
    (n : Nat) (payload : Payload) (outputAux : Key → Nat → Aux)
    (outputs : Full check output n payload outputAux) :
    Retained check output n payload outputAux := by
  intro k i inside _
  exact outputs k i inside

end NightstreamFPrime.Lifecycle.Nebula.ContinuityRanges
