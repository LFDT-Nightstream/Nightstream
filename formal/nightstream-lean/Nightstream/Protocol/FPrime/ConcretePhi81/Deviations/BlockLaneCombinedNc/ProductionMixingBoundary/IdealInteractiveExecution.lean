import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveCarrier

/-!
Transcript replay for the selected production ideal-interactive carrier.

Assurance tier: model-level protocol refinement.

Owns: exact FE and NC challenge-list replay from the selected finite seed,
construction of the physical message-only production certificate from a
prefix-causal strategy, and transport of the existing Split-NC collision
event to the actual `FeFailure`/`NcFailure` constructors.

Does not own: algebraic mixing-root probability, Fiat--Shamir, Poseidon2,
concrete field certificates, Rust/R1CS, artifacts, costs, or rows.

Emits constraints: no.

| Replay | Authority | Result |
|---|---|---|
| FE | message absorbed before `seed.feWord[i]` | exact `List.ofFn seed.feWord` |
| NC | starts after complete FE replay; message before `seed.ncWord[i]` | exact `List.ofFn seed.ncWord` |
| collision transport | existing physical fixed-phase predicates | actual named production failure |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveExecution

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveCarrier

universe uState

/-- Consecutive ideal challenges beginning at one replay cursor. -/
def wordSlice
    {count : Nat}
    (word : Fin count -> K)
    (start length : Nat) : List K :=
  (List.range' start length).map (wordAt word)

@[simp] theorem wordSlice_zero
    {count : Nat}
    (word : Fin count -> K) :
    wordSlice word 0 count = List.ofFn word := by
  apply List.ext_get
  · simp [wordSlice]
  · intro index leftBound rightBound
    have inRange : index < count := by
      simpa using rightBound
    simp [wordSlice, wordAt, inRange]

/-- Replaying any FE message suffix reveals exactly the corresponding ideal
word slice; message contents cannot affect the selected challenge. -/
theorem fe_runRoundsFrom_challenges
    {BaseState : Type uState}
    {shape : SemanticShape}
    {VerifierKey Input : Type}
    (baseSchedule :
      Schedule VerifierKey Input shape PiCcsDomains.production BaseState)
    (initialClaim : K)
    (state : ReplayState BaseState shape)
    (messages : List (Nightstream.SuperNeo.SumCheck.Finite.Message K)) :
    (Transcript.Fe.runRoundsFrom
      (feMachine (schedule baseSchedule) initialClaim)
      state messages).1 =
      wordSlice state.seed.feWord state.feIndex messages.length := by
  induction messages generalizing state with
  | nil =>
      rfl
  | cons message messages inductionHypothesis =>
      simp only [Transcript.Fe.runRoundsFrom, Transcript.Fe.runRound]
      change
        wordAt state.seed.feWord state.feIndex ::
            (Transcript.Fe.runRoundsFrom
              (feMachine (schedule baseSchedule) initialClaim)
              { state with
                base :=
                  (baseSchedule.squeezeFeChallenge
                    (baseSchedule.absorbFeRound state.base message)).2
                feIndex := state.feIndex + 1 }
              messages).1 =
          wordSlice state.seed.feWord state.feIndex
            (messages.length + 1)
      rw [inductionHypothesis]
      simp only [wordSlice, List.range'_succ, List.map_cons]

/-- FE entry resets the cursor, so the physical certificate replay produces
the complete sampled FE word in canonical row-then-lane order. -/
theorem fe_derive_coordinates
    {BaseState : Type uState}
    {shape : SemanticShape}
    {VerifierKey Input : Type}
    {statementInput :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.PublicInput shape}
    (baseSchedule :
      Schedule VerifierKey Input shape PiCcsDomains.production BaseState)
    (initialClaim : K)
    (state : ReplayState BaseState shape)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        statementInput PiCcsDomains.production.fe) :
    (Transcript.Fe.derive
      (feMachine (schedule baseSchedule) initialClaim)
      state certificate).challengePoint.coordinates =
      List.ofFn state.seed.feWord := by
  rw [Transcript.Fe.derive_point_coordinates]
  rw [fe_runRoundsFrom_challenges]
  change
    wordSlice
        ((schedule baseSchedule).enterFe state initialClaim).seed.feWord
        ((schedule baseSchedule).enterFe state initialClaim).feIndex
        certificate.rawRounds.length =
      List.ofFn state.seed.feWord
  simp only [schedule,
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate.rawRounds_length]
  exact wordSlice_zero state.seed.feWord

/-- Replaying any NC suffix similarly reveals exactly its selected word
slice. -/
theorem nc_runRoundsFrom_challenges
    {BaseState : Type uState}
    {shape : SemanticShape}
    {VerifierKey Input : Type}
    (baseSchedule :
      Schedule VerifierKey Input shape PiCcsDomains.production BaseState)
    (state : ReplayState BaseState shape)
    (messages : List Transcript.Nc.RoundMessage) :
    (Transcript.Nc.runRoundsFrom
      (ncMachine (schedule baseSchedule))
      state messages).1 =
      wordSlice state.seed.ncWord state.ncIndex messages.length := by
  induction messages generalizing state with
  | nil =>
      rfl
  | cons message messages inductionHypothesis =>
      simp only [Transcript.Nc.runRoundsFrom, Transcript.Nc.runRound]
      change
        wordAt state.seed.ncWord state.ncIndex ::
            (Transcript.Nc.runRoundsFrom
              (ncMachine (schedule baseSchedule))
              { state with
                base :=
                  (baseSchedule.squeezeNcChallenge
                    (baseSchedule.absorbNcRound state.base message)).2
                ncIndex := state.ncIndex + 1 }
              messages).1 =
          wordSlice state.seed.ncWord state.ncIndex
            (messages.length + 1)
      rw [inductionHypothesis]
      simp only [wordSlice, List.range'_succ, List.map_cons]

/-- NC entry occurs only after FE and resets only the NC cursor; the selected
NC word is reproduced exactly. -/
theorem nc_derive_coordinates
    {BaseState : Type uState}
    {shape : SemanticShape}
    {VerifierKey Input : Type}
    (baseSchedule :
      Schedule VerifierKey Input shape PiCcsDomains.production BaseState)
    (state : ReplayState BaseState shape)
    (certificate :
      Transcript.Nc.BlockLane.Certificate PiCcsDomains.production.nc) :
    (Transcript.Nc.BlockLane.derive
      (ncMachine (schedule baseSchedule))
      state certificate).challengePoint.coordinates =
      List.ofFn state.seed.ncWord := by
  rw [Transcript.Nc.BlockLane.derive_point_coordinates]
  rw [nc_runRoundsFrom_challenges]
  change
    wordSlice
        ((schedule baseSchedule).enterNc state).seed.ncWord
        ((schedule baseSchedule).enterNc state).ncIndex
        certificate.rawRounds.length =
      List.ofFn state.seed.ncWord
  simp only [schedule, Transcript.Nc.BlockLane.Certificate.rawRounds_length]
  exact wordSlice_zero state.seed.ncWord

/-! ## Prefix-causal physical certificate -/

variable
  {shape : SemanticShape}
  {BaseState : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Prover strategy selected after the complete pre-SumCheck verifier prefix.
It cannot depend on either FE or NC challenge word.  Its NC callback may still
depend on the completed FE word through the existing `SplitStrategy` type. -/
abbrev Strategy
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits)) :=
  PreSeed shape ->
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.SplitStrategy
      (PublicInput.ofSources baseInput.data)
      PiCcsDomains.production.fe CausalSoundness.ncRoundCount

/-- Later PiRLC/PiDEC payloads.  They occur after both SumChecks and may
depend on the complete ideal seed; they are irrelevant to the Split-NC
mixing and collision events. -/
structure Suffix
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth) where
  piRlcChallenges : Fin FixedActive.arity.total -> RingF
  piDecPayloads : Fin productionGlobalParams.k ->
    PiDecChildPayload (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)

/-- The exact message-only production certificate induced by a prefix-causal
strategy and one complete ideal seed. -/
def certificate
    (alphabet : Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy baseInput)
    (suffix : Seed shape -> Suffix shape publicRingColumns verifierRows publicFits)
    (seed : Seed shape) :
    ProductionRefinement.Certificate (input alphabet baseInput seed) where
  fe := (strategy seed.1).fe.physicalCertificate seed.feWord
  nc := CausalSoundness.ncCertificate
    (input := input alphabet baseInput seed)
    (strategy seed.1) seed.feWord seed.ncWord
  piRlcChallenges := (suffix seed).piRlcChallenges
  piDecPayloads := (suffix seed).piDecPayloads

private theorem fe_runRoundsFrom_seed
    {VerifierKey Input : Type}
    (baseSchedule :
      Schedule VerifierKey Input shape PiCcsDomains.production BaseState)
    (initialClaim : K)
    (state : ReplayState BaseState shape)
    (messages : List (Nightstream.SuperNeo.SumCheck.Finite.Message K)) :
    (Transcript.Fe.runRoundsFrom
      (feMachine (schedule baseSchedule) initialClaim)
      state messages).2.seed = state.seed := by
  induction messages generalizing state with
  | nil =>
      rfl
  | cons message messages inductionHypothesis =>
      simp only [Transcript.Fe.runRoundsFrom, Transcript.Fe.runRound]
      rw [inductionHypothesis]
      rfl

/-- FE replay preserves the immutable ideal seed while advancing only the
base state and FE cursor. -/
@[simp] theorem certificate_feFinal_seed
    (alphabet : Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy baseInput)
    (suffix : Seed shape -> Suffix shape publicRingColumns verifierRows publicFits)
    (seed : Seed shape) :
    (certificate alphabet baseInput strategy suffix seed).feExecution.finalState.seed =
      seed := by
  unfold ProductionRefinement.Certificate.feExecution
  apply fe_runRoundsFrom_seed

/-- Physical FE replay yields exactly the sampled FE word. -/
@[simp] theorem certificate_fe_coordinates
    (alphabet : Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy baseInput)
    (suffix : Seed shape -> Suffix shape publicRingColumns verifierRows publicFits)
    (seed : Seed shape) :
    ((certificate alphabet baseInput strategy suffix seed).feExecution
        ).challengePoint.coordinates =
      List.ofFn seed.feWord := by
  unfold ProductionRefinement.Certificate.feExecution
  apply fe_derive_coordinates

/-- Physical NC replay starts from the completed FE state and yields exactly
the sampled NC word. -/
@[simp] theorem certificate_nc_coordinates
    (alphabet : Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy baseInput)
    (suffix : Seed shape -> Suffix shape publicRingColumns verifierRows publicFits)
    (seed : Seed shape) :
    ((certificate alphabet baseInput strategy suffix seed).ncExecution
        ).challengePoint.coordinates =
      List.ofFn seed.ncWord := by
  unfold ProductionRefinement.Certificate.ncExecution
  have replay :=
    nc_derive_coordinates
      baseInput.context.piCcsSchedule
      (certificate alphabet baseInput strategy suffix seed).feExecution.finalState
      (certificate alphabet baseInput strategy suffix seed).nc
  simpa using replay

/-! ## Exact collision transport -/

/-- Exact physical FE-or-NC SumCheck collision for the certificate induced by
one ideal seed.  The strategy is selected only from the pre-SumCheck prefix. -/
def SplitCollision
    (alphabet : Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy baseInput)
    (seed : Seed shape) : Prop :=
  CausalSoundness.SplitCollision
    (input alphabet baseInput seed) (strategy seed.1) seed.2

/-- The existing finite-root collision predicate maps to the exact production
failure constructors.  No replacement event family is introduced. -/
theorem splitCollision_implies_namedFailure
    (alphabet : Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support K)
    (baseInput : ProductionRefinement.AuthoritativeInput
      (shape := shape) (State := BaseState)
      (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows)
      (publicFits := publicFits))
    (strategy : Strategy baseInput)
    (suffix : Seed shape -> Suffix shape publicRingColumns verifierRows publicFits)
    (seed : Seed shape)
    (collision : SplitCollision alphabet baseInput strategy seed) :
    ProductionRefinement.FeFailure
        (input alphabet baseInput seed)
        (certificate alphabet baseInput strategy suffix seed) ∨
      ProductionRefinement.NcFailure
        (input alphabet baseInput seed)
        (certificate alphabet baseInput strategy suffix seed) := by
  rcases collision with fe | nc
  · apply Or.inl
    rcases fe with ⟨round, bad⟩
    let bound :=
      (input alphabet baseInput seed).publicInput_eq_sources
    refine .sumcheck bound (.roundCollision round ?_)
    rw [ProductionRefinement.Certificate.fePoint_materialize]
    rw [certificate_fe_coordinates]
    have physicalCertificate :
        (Protocol.BlockLane.certificateAtSources
          (input alphabet baseInput seed).data
          (certificate alphabet baseInput strategy suffix seed).materialize.piCcs
          bound).fe =
        (strategy seed.1).fe.physicalCertificate seed.feWord := by
      have boundEq : bound = rfl := Subsingleton.elim _ _
      rw [boundEq]
      rfl
    rw [physicalCertificate]
    exact bad
  · apply Or.inr
    rcases nc with ⟨round, bad⟩
    refine .roundCollision round ?_
    rw [ProductionRefinement.Certificate.ncPoint_materialize]
    rw [certificate_nc_coordinates]
    simpa [certificate] using bad

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveExecution
