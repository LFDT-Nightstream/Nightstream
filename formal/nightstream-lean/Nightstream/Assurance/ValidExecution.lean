/-!
Top-level assurance target for the active formalization.

This file owns the meaning of a valid recursive execution and the reduction
target expected of the concrete verifier. It deliberately proves no verifier
soundness theorem before concrete SuperNeo, F', circuit, and Rust refinement
obligations exist.
-/

namespace Nightstream.Assurance

universe uState uPublic uProof

/-- Reachability by exactly `steps` semantic transitions. -/
inductive Reachable
    {State : Type uState}
    (Step : State → State → Prop)
    (initial : State) : Nat → State → Prop where
  | zero : Reachable Step initial 0 initial
  | succ {steps : Nat} {prior next : State} :
      Reachable Step initial steps prior →
      Step prior next →
      Reachable Step initial (steps + 1) next

/-- The semantic result an accepted terminal proof must establish. -/
def ValidExecution
    {State : Type uState}
    (Step : State → State → Prop)
    (TerminalValid : State → Prop)
    (initial final : State)
    (steps : Nat) : Prop :=
  Reachable Step initial steps final ∧ TerminalValid final

/--
The intended cryptographic theorem shape: acceptance reduces to semantic
validity or an explicitly modeled bad event. This is a target predicate, not an
assumption package and not a claim that the current implementation satisfies it.
-/
def VerifierReductionTarget
    {State : Type uState}
    {PublicInput : Type uPublic}
    {Proof : Type uProof}
    (verify : PublicInput → Proof → Bool)
    (initialState : PublicInput → State)
    (finalState : PublicInput → State)
    (stepCount : PublicInput → Nat)
    (Step : State → State → Prop)
    (TerminalValid : State → Prop)
    (BadEvent : PublicInput → Proof → Prop) : Prop :=
  ∀ statement proof,
    verify statement proof = true →
      ValidExecution Step TerminalValid
          (initialState statement) (finalState statement) (stepCount statement) ∨
        BadEvent statement proof

end Nightstream.Assurance
