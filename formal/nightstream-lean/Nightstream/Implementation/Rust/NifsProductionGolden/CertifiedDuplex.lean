import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
import Nightstream.Implementation.Rust.NifsProductionGolden.Poseidon2Trace
import Nightstream.Implementation.Rust.PiCcsExecution.Checker

/-!
Fail-closed transcript replay backed by local Poseidon2 round certificates.

The executable path reads a recorded permutation trace only after Lean checks
all thirty round transitions. The simulation theorems show that every
successful replay equals the canonical cached duplex used by the paper
checker.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Rust.NifsProductionGolden.CertifiedDuplex

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
open Nightstream.Implementation.Rust.NifsProductionGolden
open Nightstream.Implementation.Rust.NifsProductionGolden.Poseidon2Trace
open Nightstream.Implementation.Rust.PiCcsExecution
open Nightstream.Implementation.Rust.PiCcsExecution.CachedDuplex

abbrev DuplexState := Poseidon2Duplex.State

structure ReplayState where
  transcript : DuplexState
  nextTrace : Nat

def initial (state : DuplexState) (nextTrace : Nat := 0) : ReplayState where
  transcript := state
  nextTrace := nextTrace

def fastPermute? (receipt : ProductionReceipt)
    (state : ReplayState) : Option ReplayState :=
  if available : state.nextTrace < receipt.poseidonPermutationTraces.length then
    let trace := receipt.poseidonPermutationTraces.get
      ⟨state.nextTrace, available⟩
      if Poseidon2Trace.check Poseidon2CanonicalConstants.selected
          state.transcript.lanes trace then
        some {
          transcript := {
            lanes := Poseidon2Trace.output trace
            absorbed := 0 }
          nextTrace := state.nextTrace + 1 }
      else
        none
  else
    none

theorem fastPermute?_sound (receipt : ProductionReceipt)
    (state outputState : ReplayState)
    (accepted : fastPermute? receipt state = some outputState) :
    outputState.transcript =
        CachedDuplex.permute Poseidon2CanonicalConstants.selected
          state.transcript /\
      outputState.nextTrace = state.nextTrace + 1 := by
  unfold fastPermute? at accepted
  split at accepted
  · rename_i available
    dsimp only at accepted
    split at accepted
    · rename_i checked
      let trace := receipt.poseidonPermutationTraces.get
        ⟨state.nextTrace, available⟩
      have outputEq := Poseidon2Trace.output_eq_referenceCached
        Poseidon2CanonicalConstants.selected state.transcript.lanes trace
        (Poseidon2Trace.check_sound _ _ _ checked)
      cases accepted
      constructor
      · unfold CachedDuplex.permute
        rw [<- outputEq]
      · rfl
    · contradiction
  · contradiction

def overwrite (value : Nat) (state : DuplexState) : DuplexState where
  lanes := fun lane =>
    if lane.val = state.absorbed then value % goldilocksP else state.lanes lane
  absorbed := state.absorbed + 1

def guarded? (receipt : ProductionReceipt)
    (state : ReplayState) : Option ReplayState :=
  if rate <= state.transcript.absorbed then
    fastPermute? receipt state
  else
    some state

theorem guarded?_sound (receipt : ProductionReceipt)
    (state outputState : ReplayState)
    (accepted : guarded? receipt state = some outputState) :
    outputState.transcript =
        CachedDuplex.guarded Poseidon2CanonicalConstants.selected
          state.transcript := by
  unfold guarded? at accepted
  split at accepted
  · rename_i full
    have sound := (fastPermute?_sound receipt state outputState accepted).1
    unfold CachedDuplex.guarded
    rw [if_pos full]
    exact sound
  · rename_i available
    cases accepted
    unfold CachedDuplex.guarded
    rw [if_neg available]

def absorbElem? (receipt : ProductionReceipt) (value : Nat)
    (state : ReplayState) : Option ReplayState := do
  let target <- guarded? receipt state
  some {
    transcript := overwrite value target.transcript
    nextTrace := target.nextTrace }

theorem absorbElem?_sound (receipt : ProductionReceipt) (value : Nat)
    (state outputState : ReplayState)
    (accepted : absorbElem? receipt value state = some outputState) :
    outputState.transcript =
        CachedDuplex.absorbElem Poseidon2CanonicalConstants.selected value
          state.transcript := by
  cases targetEq : guarded? receipt state with
  | none => simp [absorbElem?, targetEq] at accepted
  | some target =>
    simp [absorbElem?, targetEq] at accepted
    cases accepted
    have guardedEq := guarded?_sound receipt state target targetEq
    unfold CachedDuplex.absorbElem
    unfold overwrite
    rw [guardedEq]

def absorbFields? (receipt : ProductionReceipt) :
    List Nat -> ReplayState -> Option ReplayState
  | [], state => some state
  | value :: values, state => do
      let next <- absorbElem? receipt value state
      absorbFields? receipt values next

theorem absorbFields?_sound (receipt : ProductionReceipt) :
    forall values state outputState,
      absorbFields? receipt values state = some outputState ->
        outputState.transcript =
          CachedDuplex.absorbList Poseidon2CanonicalConstants.selected values
            state.transcript := by
  intro values
  induction values with
  | nil =>
      intro state outputState accepted
      simp only [absorbFields?] at accepted
      cases accepted
      rfl
  | cons value values inductionHypothesis =>
      intro state outputState accepted
      cases nextEq : absorbElem? receipt value state with
      | none => simp [absorbFields?, nextEq] at accepted
      | some next =>
        simp [absorbFields?, nextEq] at accepted
        rw [CachedDuplex.absorbList]
        rw [<- absorbElem?_sound receipt value state next nextEq]
        exact inductionHypothesis next outputState accepted

def gate? (receipt : ProductionReceipt)
    (state : ReplayState) : Option ReplayState := do
  let absorbed <- absorbElem? receipt 1 state
  fastPermute? receipt absorbed

theorem gate?_sound (receipt : ProductionReceipt)
    (state outputState : ReplayState)
    (accepted : gate? receipt state = some outputState) :
    outputState.transcript =
        CachedDuplex.gate Poseidon2CanonicalConstants.selected
          state.transcript := by
  cases absorbedEq : absorbElem? receipt 1 state with
  | none => simp [gate?, absorbedEq] at accepted
  | some absorbed =>
    simp [gate?, absorbedEq] at accepted
    have permutationEq := (fastPermute?_sound receipt absorbed outputState accepted).1
    have absorptionEq := absorbElem?_sound receipt 1 state absorbed absorbedEq
    unfold CachedDuplex.gate
    rw [<- absorptionEq]
    exact permutationEq

def challenge (state : DuplexState) : K where
  c0 := ⟨state.lanes ⟨0, by decide⟩ % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩
  c1 := ⟨state.lanes ⟨1, by decide⟩ % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

theorem challenge_gate_pair (state : DuplexState) :
    (challenge (CachedDuplex.gate Poseidon2CanonicalConstants.selected state),
      CachedDuplex.gate Poseidon2CanonicalConstants.selected state) =
      squeezeK state := by
  rfl

def squeezeK? (receipt : ProductionReceipt)
    (state : ReplayState) : Option (K × ReplayState) := do
  let next <- gate? receipt state
  some (challenge next.transcript, next)

theorem squeezeK?_sound (receipt : ProductionReceipt)
    (state outputState : ReplayState) (value : K)
    (accepted : squeezeK? receipt state = some (value, outputState)) :
    (value, outputState.transcript) = squeezeK state.transcript := by
  cases nextEq : gate? receipt state with
  | none => simp [squeezeK?, nextEq] at accepted
  | some next =>
    have pairEq : (challenge next.transcript, next) = (value, outputState) :=
      Option.some.inj (by simpa [squeezeK?, nextEq] using accepted)
    have valueEq := congrArg Prod.fst pairEq
    have stateEq := congrArg Prod.snd pairEq
    dsimp only at valueEq stateEq
    subst value
    subst outputState
    have gateEq := gate?_sound receipt state next nextEq
    rw [gateEq]
    exact challenge_gate_pair state.transcript

def squeezeIndexed? (receipt : ProductionReceipt) (tag index : Nat)
    (state : ReplayState) : Option (K × ReplayState) := do
  let absorbed <- absorbFields? receipt [tag, index] state
  squeezeK? receipt absorbed

def squeezeSingle? (receipt : ProductionReceipt) (tag : Nat)
    (state : ReplayState) : Option (K × ReplayState) := do
  let absorbed <- absorbFields? receipt [tag] state
  squeezeK? receipt absorbed

theorem squeezeIndexed?_sound (receipt : ProductionReceipt) (tag index : Nat)
    (state outputState : ReplayState) (value : K)
    (accepted : squeezeIndexed? receipt tag index state =
      some (value, outputState)) :
    (value, outputState.transcript) = squeezeIndexed tag index state.transcript := by
  cases absorbedEq : absorbFields? receipt [tag, index] state with
  | none => simp [squeezeIndexed?, absorbedEq] at accepted
  | some absorbed =>
    have squeezed := squeezeK?_sound receipt absorbed outputState value
      (by simpa [squeezeIndexed?, absorbedEq] using accepted)
    have absorptionEq := absorbFields?_sound receipt [tag, index]
      state absorbed absorbedEq
    unfold squeezeIndexed
    unfold absorbFields
    rw [<- absorptionEq]
    exact squeezed

theorem squeezeSingle?_sound (receipt : ProductionReceipt) (tag : Nat)
    (state outputState : ReplayState) (value : K)
    (accepted : squeezeSingle? receipt tag state = some (value, outputState)) :
    (value, outputState.transcript) = squeezeSingle tag state.transcript := by
  cases absorbedEq : absorbFields? receipt [tag] state with
  | none => simp [squeezeSingle?, absorbedEq] at accepted
  | some absorbed =>
    have squeezed := squeezeK?_sound receipt absorbed outputState value
      (by simpa [squeezeSingle?, absorbedEq] using accepted)
    have absorptionEq := absorbFields?_sound receipt [tag]
      state absorbed absorbedEq
    unfold squeezeSingle
    unfold absorbFields
    rw [<- absorptionEq]
    exact squeezed

def deriveIndexed? (receipt : ProductionReceipt) (tag start : Nat) :
    Nat -> ReplayState -> Option (List K × ReplayState)
  | 0, state => some ([], state)
  | count + 1, state => do
      let sampled <- squeezeIndexed? receipt tag start state
      let rest <- deriveIndexed? receipt tag (start + 1) count sampled.2
      some (sampled.1 :: rest.1, rest.2)

theorem deriveIndexed?_sound (receipt : ProductionReceipt) (tag start : Nat) :
    forall count state values outputState,
      deriveIndexed? receipt tag start count state = some (values, outputState) ->
        (values, outputState.transcript) =
          deriveIndexed tag start count state.transcript := by
  intro count
  induction count generalizing start with
  | zero =>
      intro state values outputState accepted
      simp only [deriveIndexed?] at accepted
      cases accepted
      rfl
  | succ count inductionHypothesis =>
      intro state values outputState accepted
      cases sampledEq : squeezeIndexed? receipt tag start state with
      | none => simp [deriveIndexed?, sampledEq] at accepted
      | some sampled =>
        cases restEq : deriveIndexed? receipt tag (start + 1) count sampled.2 with
        | none => simp [deriveIndexed?, sampledEq, restEq] at accepted
        | some rest =>
          have acceptedEq : (sampled.1 :: rest.1, rest.2) =
              (values, outputState) :=
            Option.some.inj (by
              simpa [deriveIndexed?, sampledEq, restEq] using accepted)
          have sampledSound := squeezeIndexed?_sound receipt tag start
            state sampled.2 sampled.1 (by simpa using sampledEq)
          have restSound := inductionHypothesis (start + 1)
            sampled.2 rest.1 rest.2 (by simpa using restEq)
          have valuesEq := congrArg Prod.fst acceptedEq
          have stateEq := congrArg Prod.snd acceptedEq
          dsimp only at valuesEq stateEq
          subst values
          subst outputState
          unfold deriveIndexed
          rw [<- sampledSound]
          dsimp only
          rw [<- restSound]

def deriveRoundChallenges? (receipt : ProductionReceipt) :
    Nat -> List (Nightstream.SuperNeo.SumCheck.Finite.Message K) ->
      ReplayState -> Option (List K × ReplayState)
  | _, [], state => some ([], state)
  | index, message :: messages, state => do
      let absorbed <- absorbFields? receipt (roundFields index message) state
      let sampled <- squeezeIndexed? receipt 46 index absorbed
      let rest <- deriveRoundChallenges? receipt (index + 1) messages sampled.2
      some (sampled.1 :: rest.1, rest.2)

theorem deriveRoundChallenges?_sound (receipt : ProductionReceipt) :
    forall index messages state values outputState,
      deriveRoundChallenges? receipt index messages state =
          some (values, outputState) ->
        (values, outputState.transcript) =
          deriveRoundChallenges index messages state.transcript := by
  intro index messages
  induction messages generalizing index with
  | nil =>
      intro state values outputState accepted
      simp only [deriveRoundChallenges?] at accepted
      cases accepted
      rfl
  | cons message messages inductionHypothesis =>
      intro state values outputState accepted
      cases absorbedEq : absorbFields? receipt (roundFields index message) state with
      | none => simp [deriveRoundChallenges?, absorbedEq] at accepted
      | some absorbed =>
        cases sampledEq : squeezeIndexed? receipt 46 index absorbed with
        | none =>
          simp [deriveRoundChallenges?, absorbedEq, sampledEq] at accepted
        | some sampled =>
          cases restEq : deriveRoundChallenges? receipt (index + 1)
              messages sampled.2 with
          | none =>
            simp [deriveRoundChallenges?, absorbedEq, sampledEq, restEq] at accepted
          | some rest =>
            have acceptedEq : (sampled.1 :: rest.1, rest.2) =
                (values, outputState) :=
              Option.some.inj (by
                simpa [deriveRoundChallenges?, absorbedEq, sampledEq, restEq]
                  using accepted)
            have absorptionSound := absorbFields?_sound receipt
              (roundFields index message) state absorbed absorbedEq
            have sampledSound := squeezeIndexed?_sound receipt 46 index
              absorbed sampled.2 sampled.1 (by simpa using sampledEq)
            have restSound := inductionHypothesis (index + 1)
              sampled.2 rest.1 rest.2 (by simpa using restEq)
            have valuesEq := congrArg Prod.fst acceptedEq
            have stateEq := congrArg Prod.snd acceptedEq
            dsimp only at valuesEq stateEq
            subst values
            subst outputState
            unfold deriveRoundChallenges
            unfold absorbFields
            rw [<- absorptionSound]
            dsimp only
            rw [<- sampledSound]
            dsimp only
            rw [<- restSound]

end Nightstream.Implementation.Rust.NifsProductionGolden.CertifiedDuplex
