import Nightstream.HyperNova.Construction2.Paper

/-!
Canonical executable terminal verifier for HyperNova Construction 2.

Owns: executable base/recursive terminal branching, the prior public-link
check, finite validation of every running slot, validation of the selected
fresh slot, and extensional equality with the independent terminal relation.

Does not own: an additional NIFS fold (the terminal verifier performs none),
the concrete running/fresh relation checkers, SuperNeo security, Rust, R1CS,
lowering, or costs.

Emits constraints: no.

Relation membership is supplied through Boolean checkers with exactness
proofs.  Those checkers are the terminal analogue of the deterministic
`NIFS.V` function; no caller-provided terminal-acceptance proposition is an
input to `eval`.
-/

namespace Nightstream.Protocol.FPrime.CanonicalTerminalVerifier

open Nightstream.HyperNova.Construction2.Paper

universe uKey uDigest uState uWitness uRunning uRunningWitness uFresh
  uFreshWitness uProof uEncoded

/-- Executable membership checks for the two terminal relation families. -/
structure RelationChecks
    {Key : Type uKey}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount) where
  runningCheck : (slot : Fin slotCount) ->
    Key -> Running -> RunningWitness -> Bool
  freshCheck : (slot : Fin slotCount) ->
    Key -> Fresh -> FreshWitness -> Bool
  runningCheck_iff : forall slot key value witness,
    runningCheck slot key value witness = true <->
      relations.runningHolds slot key value witness
  freshCheck_iff : forall slot key value witness,
    freshCheck slot key value witness = true <->
      relations.freshHolds slot key value witness

/-- Execute all running-relation checks in canonical slot order. -/
def allRunningAccepted
    {Key : Type uKey}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount}
    (checks : RelationChecks relations)
    (setupKeys : Fin slotCount -> Key)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Bool :=
  (List.finRange slotCount).all fun slot =>
    checks.runningCheck slot (setupKeys slot)
      (proof.running slot) (proof.runningWitness slot)

theorem finRange_all_eq_true_iff
    {slotCount : Nat}
    (predicate : Fin slotCount -> Bool) :
    (List.finRange slotCount).all predicate = true <->
      forall slot, predicate slot = true := by
  simp

theorem allRunningAccepted_eq_true_iff
    {Key : Type uKey}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount}
    (checks : RelationChecks relations)
    (setupKeys : Fin slotCount -> Key)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    allRunningAccepted checks setupKeys proof = true <->
      forall slot, relations.runningHolds slot (setupKeys slot)
        (proof.running slot) (proof.runningWitness slot) := by
  rw [allRunningAccepted, finRange_all_eq_true_iff]
  constructor
  · intro accepted slot
    exact (checks.runningCheck_iff slot (setupKeys slot)
      (proof.running slot) (proof.runningWitness slot)).1 (accepted slot)
  · intro holds slot
    exact (checks.runningCheck_iff slot (setupKeys slot)
      (proof.running slot) (proof.runningWitness slot)).2 (holds slot)

/-- Compact terminal evaluation.  Iteration zero checks only the endpoint.
Every positive iteration checks the prior link, every running relation, and
the selected fresh relation.  It never calls `NIFS.V`. -/
def eval
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (checks : RelationChecks relations)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Bool :=
  letI : Decidable (InRange slotCount proof.pc) := by
    unfold InRange
    infer_instance
  if statement.iteration = 0 then
    decide (statement.zi = statement.z0)
  else if pcValid : InRange slotCount proof.pc then
    (decide (machine.freshPublic proof.fresh =
      machine.encodeInstance (machine.hash {
        verifierKeys := setup.verifierKeys
        iteration := statement.iteration
        z0 := statement.z0
        current := statement.zi
        running := proof.running
        pc := proof.pc
      })) &&
    allRunningAccepted checks setup.verifierKeys proof) &&
    checks.freshCheck (selectedIndex pcValid)
      (setup.verifierKeys (selectedIndex pcValid)) proof.fresh
        proof.freshWitness
  else
    false

/-- The executable terminal verifier accepts exactly the independent terminal
transition, including explicit base and recursive boundaries. -/
theorem eval_eq_true_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (checks : RelationChecks relations)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    eval setup machine relations checks statement proof = true <->
      TerminalTransition setup machine relations statement proof := by
  unfold eval
  by_cases iterationZero : statement.iteration = 0
  · rw [if_pos iterationZero]
    constructor
    · intro accepted
      exact Or.inl ⟨iterationZero, of_decide_eq_true accepted⟩
    · intro transition
      rcases transition with base | recursive
      · exact decide_eq_true base.2
      · have impossible : 0 < 0 := iterationZero ▸ recursive.2.1
        exact False.elim (Nat.lt_irrefl 0 impossible)
  · rw [if_neg iterationZero]
    by_cases pcValid : InRange slotCount proof.pc
    · rw [dif_pos pcValid]
      constructor
      · intro accepted
        simp only [Bool.and_eq_true] at accepted
        rcases accepted with
          ⟨⟨publicInputAccepted, runningAccepted⟩, freshAccepted⟩
        exact Or.inr ⟨pcValid, Nat.pos_of_ne_zero iterationZero,
          of_decide_eq_true publicInputAccepted,
          (allRunningAccepted_eq_true_iff checks setup.verifierKeys proof).1
            runningAccepted,
          (checks.freshCheck_iff (selectedIndex pcValid)
            (setup.verifierKeys (selectedIndex pcValid)) proof.fresh
            proof.freshWitness).1 freshAccepted⟩
      · intro transition
        rcases transition with base | recursive
        · exact False.elim (iterationZero base.1)
        · rcases recursive with
            ⟨otherPcValid, iterationPositive, priorPublicInput,
              runningValid, freshValid⟩
          simp only [Bool.and_eq_true]
          exact ⟨⟨decide_eq_true priorPublicInput,
            (allRunningAccepted_eq_true_iff checks setup.verifierKeys proof).2
              runningValid⟩,
            (checks.freshCheck_iff (selectedIndex pcValid)
              (setup.verifierKeys (selectedIndex pcValid)) proof.fresh
              proof.freshWitness).2 (by
                simpa only using freshValid)⟩
    · rw [dif_neg pcValid]
      constructor
      · intro accepted
        contradiction
      · intro transition
        rcases transition with base | recursive
        · exact False.elim (iterationZero base.1)
        · exact False.elim (pcValid recursive.1)

end Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
