import Nightstream.Protocol.FPrime.CanonicalVerifier
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
import Nightstream.Protocol.FPrime.Frozen.Obligations

/-!
Construction-2 refinement from an independently stated NIFS transition.

Owns: the semantic augmented-function transition obtained by expanding the
paper equations with an independent selected NIFS transition; soundness of
the canonical evaluator modulo exactly the selected NIFS bad event; honest
completeness by replacing only the sole NIFS proof; and terminal exactness
without another fold.

Does not own: a concrete NIFS proof, SuperNeo, Fiat--Shamir, Rust, R1CS,
artifacts, lowering, or costs.

Emits constraints: no.
-/

namespace Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement

open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Protocol.FPrime.Frozen.Obligations

universe uKey uDigest uState uWitness uRunning uRunningWitness uFresh
  uFreshWitness uProof uEncoded

/-- Replace only the single prover message consumed by the recursive NIFS
call. Every public and witness field outside that message is definitionally
unchanged. -/
def withNifsProof
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {slotCount : Nat}
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (proof : Proof) :
    Input Key State Witness Running Fresh Proof slotCount :=
  { input with nifsProof := proof }

/-- Independently expanded Construction-2 transition. The base branch performs
no fold. The recursive branch states the selected fold through the supplied
semantic NIFS transition, never through verifier acceptance. -/
def SemanticTransition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (nifsTransition : Key -> Running -> Fresh -> Running -> Prop)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) : Prop :=
  machine.control input.zi input.witness = functionIndex /\
  output.pcNext = functionIndex /\
  output.zNext = machine.step functionIndex input.zi input.witness /\
  output.x = machine.hash (nextHashPreimage setup input output) /\
  ((input.iteration = 0 /\
      input.z0 = input.zi /\
      output.runningNext = fun _ => setup.defaultRunning) \/
    exists priorPcValid : InRange slotCount input.priorPc,
      0 < input.iteration /\
      machine.freshPublic input.fresh =
        machine.encodeInstance (machine.hash (priorHashPreimage setup input)) /\
      nifsTransition
        (setup.verifierKeys (selectedIndex priorPcValid))
        (input.running (selectedIndex priorPcValid)) input.fresh
        (output.runningNext (selectedIndex priorPcValid)) /\
      forall slot, slot ≠ selectedIndex priorPcValid ->
        output.runningNext slot = input.running slot)

/-- The sole failure admitted by augmented-function soundness: the concrete
NIFS verifier's bad event at the checked prior-program-counter slot. -/
def SelectedNifsBadEvent
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (badEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) : Prop :=
  exists priorPcValid : InRange slotCount input.priorPc,
    0 < input.iteration /\
    badEvent
      (setup.verifierKeys (selectedIndex priorPcValid))
      (input.running (selectedIndex priorPcValid)) input.fresh input.nifsProof
      (output.runningNext (selectedIndex priorPcValid))

/-- A fixed canonical acceptance realizes the independently expanded
Construction-2 transition or exactly the selected NIFS bad event. -/
theorem accepts_implies_semanticTransition_or_selectedNifsBadEvent
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (nifsTransition : Key -> Running -> Fresh -> Running -> Prop)
    (nifsBadEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop)
    (nifsCorrect : NifsSoundAndCompleteModulo setup.nifs
      nifsTransition nifsBadEvent)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount)
    (accepted :
      Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts
        setup machine functionIndex input output) :
    SemanticTransition setup machine nifsTransition functionIndex input output \/
      SelectedNifsBadEvent setup nifsBadEvent input output := by
  have paperTransition :=
    (Nightstream.Protocol.FPrime.CanonicalVerifier.accepts_iff_transition
      setup machine functionIndex input output).mp accepted
  rcases paperTransition with
    ⟨dispatch, pcNext, application, outputHash, branch⟩
  rcases branch with base | recursive
  · exact Or.inl ⟨dispatch, pcNext, application, outputHash, Or.inl base⟩
  · rcases recursive with
      ⟨priorPcValid, iterationPositive, priorPublicInput,
        selectedNifs, unchanged⟩
    rcases nifsCorrect.1
        (setup.verifierKeys (selectedIndex priorPcValid))
        (input.running (selectedIndex priorPcValid)) input.fresh
        input.nifsProof (output.runningNext (selectedIndex priorPcValid))
        selectedNifs with selectedTransition | selectedBadEvent
    · exact Or.inl ⟨dispatch, pcNext, application, outputHash,
        Or.inr ⟨priorPcValid, iterationPositive, priorPublicInput,
          selectedTransition, unchanged⟩⟩
    · exact Or.inr ⟨priorPcValid, iterationPositive, selectedBadEvent⟩

/-- Every independently valid semantic transition is accepted after replacing
only the recursive NIFS message. In the base branch the existing message is
retained and never inspected. -/
theorem semanticTransition_implies_exists_nifsProof_accepts
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (nifsTransition : Key -> Running -> Fresh -> Running -> Prop)
    (nifsBadEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop)
    (nifsCorrect : NifsSoundAndCompleteModulo setup.nifs
      nifsTransition nifsBadEvent)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount)
    (semantic :
      SemanticTransition setup machine nifsTransition functionIndex input output) :
    exists nifsProof : Proof,
      Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts
        setup machine functionIndex (withNifsProof input nifsProof) output := by
  rcases semantic with
    ⟨dispatch, pcNext, application, outputHash, branch⟩
  rcases branch with base | recursive
  · refine ⟨input.nifsProof, ?_⟩
    apply (Nightstream.Protocol.FPrime.CanonicalVerifier.accepts_iff_transition
      setup machine functionIndex (withNifsProof input input.nifsProof)
        output).mpr
    simpa [withNifsProof] using
      (show Transition setup machine functionIndex input output from
        ⟨dispatch, pcNext, application, outputHash, Or.inl base⟩)
  · rcases recursive with
      ⟨priorPcValid, iterationPositive, priorPublicInput,
        selectedTransition, unchanged⟩
    rcases nifsCorrect.2
        (setup.verifierKeys (selectedIndex priorPcValid))
        (input.running (selectedIndex priorPcValid)) input.fresh
        (output.runningNext (selectedIndex priorPcValid))
        selectedTransition with ⟨nifsProof, selectedNifs⟩
    refine ⟨nifsProof, ?_⟩
    apply (Nightstream.Protocol.FPrime.CanonicalVerifier.accepts_iff_transition
      setup machine functionIndex (withNifsProof input nifsProof) output).mpr
    refine ⟨?_, ?_, ?_, ?_, Or.inr ⟨?_, ?_, ?_, ?_, ?_⟩⟩
    · simpa [withNifsProof] using dispatch
    · exact pcNext
    · simpa [withNifsProof] using application
    · simpa [withNifsProof, nextHashPreimage] using outputHash
    · simpa [withNifsProof] using priorPcValid
    · simpa [withNifsProof] using iterationPositive
    · simpa [withNifsProof, priorHashPreimage] using priorPublicInput
    · simpa [withNifsProof] using selectedNifs
    · intro slot different
      simpa [withNifsProof] using unchanged slot different

/-- The executable terminal verifier is exact without an NIFS correctness
premise, an NIFS bad event, or a final fold. -/
theorem terminal_exact_without_nifs
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
    (checks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        relations)
    (statement : TerminalStatement State)
    (proof :
      OuterTerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.evalOuter
        setup machine relations checks statement proof = true <->
      OuterTerminalTransition setup machine relations statement proof := by
  exact
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.evalOuter_eq_true_iff_transition
      setup machine relations checks statement proof

end Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement
