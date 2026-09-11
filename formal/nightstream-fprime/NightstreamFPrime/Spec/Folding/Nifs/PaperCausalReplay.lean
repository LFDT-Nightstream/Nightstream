import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiatShamir
import NightstreamFPrime.Spec.SumCheck.FixedPhase.Sequential

/-!
Generates a causal PiCCS message list with the verifier's actual transcript.
Each strategy call precedes its message absorption and challenge squeeze.
The returned list uses the existing fixed polynomials and finite certificate
view. This is deterministic replay, with no sampling or cryptographic law.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperCausalReplay

open NightstreamFPrime.Spec
open SumCheck.Finite
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal (Strategy)
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace (issued)
open Folding.PiCCS.PaperJoint
open StrongReduction

universe uContext uState

variable {Context : Type uContext} {State : Type uState}
  {shape : Shape} {width : Nat}

/-- Emit one fixed polynomial from past challenges, then absorb it and derive
the next challenge. The supplied index list is the only recursion axis. -/
def generatedMessages (oracle : FiatShamir.Oracle Context K State shape)
    (strategy : Strategy width) (fixed : List K) :
    State → List (Fin shape.cubeVariables) →
      Option (List (FixedPolynomial K width) × State)
  | state, [] => some ([], state)
  | state, index :: indices =>
      match strategy fixed with
      | none => none
      | some message =>
          let sample := oracle.squeeze
            (oracle.absorbRound state index message.toMessage) (.sumcheck index)
          (generatedMessages oracle strategy (fixed ++ [sample.1]) sample.2 indices).map
            fun result => (message :: result.1, result.2)

private theorem issued_tail_exists (strategy : Strategy width)
    (fixed : List K) (challenge : K) (challenges : List K)
    (message : FixedPolynomial K width) (chosen : strategy fixed = some message)
    (rounds : List (FixedPolynomial K width))
    (returned : issued strategy fixed (challenge :: challenges) = some rounds) :
    ∃ tail, issued strategy (fixed ++ [challenge]) challenges = some tail := by
  cases later : issued strategy (fixed ++ [challenge]) challenges with
  | none =>
      simp only [issued, chosen, later, Option.map_none] at returned
      cases returned
  | some tail => exact ⟨tail, rfl⟩

private theorem generatedMessages_total
    (oracle : FiatShamir.Oracle Context K State shape) (strategy : Strategy width)
    (indices : List (Fin shape.cubeVariables)) (fixed : List K) (state : State)
    (total : ∀ challenges : List K, challenges.length = indices.length →
      ∃ rounds, issued strategy fixed challenges = some rounds) :
    ∃ rounds finalState,
      generatedMessages oracle strategy fixed state indices = some (rounds, finalState) ∧
      rounds.length = indices.length := by
  induction indices generalizing fixed state with
  | nil => exact ⟨[], state, rfl, rfl⟩
  | cons index indices ih =>
      obtain ⟨probeRounds, probeReturned⟩ := total
        (K.zero :: List.replicate indices.length K.zero) (by simp)
      cases chosen : strategy fixed with
      | none =>
          simp only [issued, chosen] at probeReturned
          cases probeReturned
      | some message =>
          let sample := oracle.squeeze
            (oracle.absorbRound state index message.toMessage) (.sumcheck index)
          have tailTotal : ∀ challenges : List K, challenges.length = indices.length →
              ∃ rounds, issued strategy (fixed ++ [sample.1]) challenges = some rounds := by
            intro challenges length
            obtain ⟨rounds, returned⟩ := total (sample.1 :: challenges)
              (by simp only [List.length_cons, length])
            exact issued_tail_exists strategy fixed sample.1 challenges message chosen rounds returned
          obtain ⟨tail, finalState, later, length⟩ :=
            ih (fixed ++ [sample.1]) sample.2 tailTotal
          refine ⟨message :: tail, finalState, ?_, ?_⟩
          · simp only [generatedMessages, chosen]
            change (generatedMessages oracle strategy (fixed ++ [sample.1]) sample.2 indices).map
              (fun result => (message :: result.1, result.2)) = some (message :: tail, finalState)
            rw [later]
            rfl
          · simp only [List.length_cons, length]

private theorem generatedMessages_replay
    (oracle : FiatShamir.Oracle Context K State shape) (strategy : Strategy width)
    (messages : Fin shape.cubeVariables → FixedPolynomial K width)
    (indices : List (Fin shape.cubeVariables)) (fixed : List K) (state : State)
    (rounds : List (FixedPolynomial K width)) (finalState : State)
    (returned : generatedMessages oracle strategy fixed state indices = some (rounds, finalState))
    (ordered : indices.map messages = rounds) :
    let replay := FiatShamir.deriveRoundsFrom oracle
      (fun index => (messages index).toMessage) state indices
    issued strategy fixed replay.1 = some rounds ∧ replay.2 = finalState := by
  induction indices generalizing fixed state rounds finalState with
  | nil =>
      have equal : ([], state) = (rounds, finalState) := Option.some.inj returned
      have roundsEq : rounds = [] := (congrArg Prod.fst equal).symm
      have stateEq : state = finalState := congrArg Prod.snd equal
      subst rounds
      exact ⟨rfl, stateEq⟩
  | cons index indices ih =>
      cases chosen : strategy fixed with
      | none =>
          simp only [generatedMessages, chosen] at returned
          cases returned
      | some message =>
          let sample := oracle.squeeze
            (oracle.absorbRound state index message.toMessage) (.sumcheck index)
          simp only [generatedMessages, chosen] at returned
          change (generatedMessages oracle strategy (fixed ++ [sample.1]) sample.2 indices).map
            (fun result => (message :: result.1, result.2)) = some (rounds, finalState) at returned
          cases later : generatedMessages oracle strategy (fixed ++ [sample.1]) sample.2 indices with
          | none =>
              simp only [later, Option.map_none] at returned
              cases returned
          | some result =>
              obtain ⟨tail, tailState⟩ := result
              have equal : (message :: tail, tailState) = (rounds, finalState) := by
                apply Option.some.inj
                simpa only [later, Option.map_some] using returned
              have roundsEq : rounds = message :: tail := (congrArg Prod.fst equal).symm
              have stateEq : tailState = finalState := congrArg Prod.snd equal
              cases roundsEq
              have orderedHead : messages index = message := (List.cons.inj ordered).1
              have orderedTail : indices.map messages = tail := (List.cons.inj ordered).2
              have following := ih (fixed ++ [sample.1]) sample.2 tail tailState later orderedTail
              constructor
              · simp only [FiatShamir.deriveRoundsFrom, orderedHead]
                change issued strategy fixed
                  (sample.1 :: (FiatShamir.deriveRoundsFrom oracle
                    (fun item => (messages item).toMessage) sample.2 indices).1) = some (message :: tail)
                simp only [issued, chosen]
                rw [following.1]
                rfl
              · simpa only [FiatShamir.deriveRoundsFrom, orderedHead, sample] using
                  following.2.trans stateEq

private theorem map_canonicalFinIndices {Value : Type*} (count : Nat)
    (value : Fin count → Value) :
    (canonicalFinIndices count).map value = List.ofFn value := by
  simp only [canonicalFinIndices, List.map_ofFn, Function.comp_def]
  rfl

/-- A strategy that returns on every full challenge list also returns on its
own transcript. The generated messages have the same issued receipt and final
state when the verifier replays them in canonical round order. -/
theorem generated_messages_replay
    (oracle : FiatShamir.Oracle Context K State shape) (strategy : Strategy width)
    (state : State)
    (total : ∀ challenges : List K, challenges.length = shape.cubeVariables →
      ∃ rounds, issued strategy [] challenges = some rounds) :
    ∃ (messages : Fin shape.cubeVariables → FixedPolynomial K width) (finalState : State),
      generatedMessages oracle strategy [] state (canonicalFinIndices shape.cubeVariables) =
        some (List.ofFn messages, finalState) ∧
      let replay := FiatShamir.deriveRoundsFrom oracle
        (fun index => (messages index).toMessage) state (canonicalFinIndices shape.cubeVariables)
      issued strategy [] replay.1 = some (List.ofFn messages) ∧ replay.2 = finalState := by
  have totalIndices : ∀ challenges : List K,
      challenges.length = (canonicalFinIndices shape.cubeVariables).length →
      ∃ rounds, issued strategy [] challenges = some rounds := by
    intro challenges length
    exact total challenges (length.trans (canonicalFinIndices_length _))
  obtain ⟨rounds, finalState, returned, length⟩ := generatedMessages_total
    oracle strategy (canonicalFinIndices shape.cubeVariables) [] state totalIndices
  have lengthExact : rounds.length = shape.cubeVariables :=
    length.trans (canonicalFinIndices_length _)
  let messages := FixedPhase.Sequential.functionOfExactList rounds lengthExact
  have listEq : List.ofFn messages = rounds :=
    FixedPhase.Sequential.ofFn_functionOfExactList rounds lengthExact
  refine ⟨messages, finalState, ?_, ?_⟩
  · rw [listEq]
    exact returned
  · have ordered : (canonicalFinIndices shape.cubeVariables).map messages = rounds :=
      (map_canonicalFinIndices _ messages).trans listEq
    have replay := generatedMessages_replay oracle strategy messages
      (canonicalFinIndices shape.cubeVariables) [] state rounds finalState returned ordered
    simpa only [listEq] using replay

/-- The existing all-coin causal-prover success property supplies the totality
used above. No additional transcript, output-correctness, or sampling premise
is introduced; the honest PiCCS theorem supplies this property. -/
theorem prover_generated_messages_replay {columns : Nat}
    (oracle : FiatShamir.Oracle Context K State shape)
    (prover : CausalExecution.Prover shape columns width)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K) (state : State)
    (total : ∀ point : CubePoint K shape.cubeVariables,
      ∃ (probe : Probe K shape) (witness : OutputWitness shape columns),
        CausalExecution.run prover alpha gamma point = some (probe, witness)) :
    ∃ (messages : Fin shape.cubeVariables → FixedPolynomial K width) (finalState : State),
      generatedMessages oracle (prover.rounds alpha gamma) [] state
        (canonicalFinIndices shape.cubeVariables) = some (List.ofFn messages, finalState) ∧
      let replay := FiatShamir.deriveRoundsFrom oracle
        (fun index => (messages index).toMessage) state (canonicalFinIndices shape.cubeVariables)
      issued (prover.rounds alpha gamma) [] replay.1 = some (List.ofFn messages) ∧
        replay.2 = finalState := by
  apply generated_messages_replay oracle (prover.rounds alpha gamma) state
  intro challenges length
  obtain ⟨probe, witness, returned⟩ := total ⟨challenges, length⟩
  cases execution : issued (prover.rounds alpha gamma) [] challenges with
  | none =>
      simp only [CausalExecution.run, execution] at returned
      cases returned
  | some rounds => exact ⟨rounds, rfl⟩

end NightstreamFPrime.Spec.Folding.Nifs.PaperCausalReplay
