import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshSchedule

/-! The light fold's transcript as a random-oracle `Schedule` (`schedule`), and the light key's
reads of it. The light transcript absorbs only the fresh claim (`OptionB.lightAbsorbPublicInput`);
the prior state hash in the fresh public input binds the running claim. Its Π_CCS coins
(`coins_eq_reads`, `probe_coins`) and its one Π_RLC challenge (`challenge_seal`, the first
challenge of the production sampler) are the schedule's reads. The light and refresh
transcripts share no sponge call prefix (`challengeCalls_separated_refresh`). This is
`TranscriptCoverage` for the light key of `OptionB.keys`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightSchedule

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Sponge

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

section Schedule

variable (logicalWidth publicFits) in
/-- What the light transcript absorbs before the Π_CCS coins: the fresh claim. -/
abbrev Statement :=
  Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) LightShape

/-- The light fresh claim as a production fresh claim: both shapes have one fresh source. -/
def toProduction (fresh : Statement logicalWidth publicFits) :
    Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  ⟨fresh.commitments, fresh.publicInputs⟩

def statementWords (fresh : Statement logicalWidth publicFits) : List F :=
  (ProductionKey.publicInputBlocks (toProduction fresh)).flatten

theorem statementWords_length (fresh fresh' : Statement logicalWidth publicFits) :
    (statementWords fresh).length = (statementWords fresh').length :=
  RefreshSchedule.freshWords_length _ _

theorem statementWords_injective {fresh fresh' : Statement logicalWidth publicFits}
    (same : statementWords fresh = statementWords fresh') : fresh = fresh' := by
  have production := RefreshSchedule.freshWords_injective same
  cases fresh
  cases fresh'
  simp only [toProduction] at production
  obtain ⟨rfl, rfl⟩ := production
  rfl

theorem lightOutputWords_eq (output : FullOutputCoordinates.FullOutput K LightShape) :
    lightOutputWords output = Schedule.fullOutputWords output :=
  rfl

variable (logicalWidth publicFits) in
/-- The light transcript's schedule: light shape, the light domain, and one Π_RLC challenge. -/
def schedule : Schedule where
  shape := LightShape
  domainChunk := lightFoldDomainChunk
  domainChunk_length := by simp [lightFoldDomainChunk, domainChunk]
  Statement := Statement logicalWidth publicFits
  statementWords := statementWords
  statementWords_length := statementWords_length
  statementWords_injective := statementWords_injective
  Output := FullOutputCoordinates.FullOutput K LightShape
  outputWords := lightOutputWords
  outputWords_length := Schedule.fullOutputWords_length
  outputWords_injective := Schedule.fullOutputWords_injective
  rhoCount := 1
  rhoCount_le := by decide

def messages {width : Nat} (proof : Light.Proof LightShape width) :
    (schedule logicalWidth publicFits).Messages width :=
  ⟨proof.piCcsRounds, proof.piCcsOutput⟩

/-- The light and refresh transcripts share no sponge call prefix. -/
theorem challengeCalls_separated_refresh {width width' : Nat}
    {fresh : (schedule logicalWidth publicFits).Statement}
    {lightMessages : (schedule logicalWidth publicFits).Messages width}
    {challenge : (schedule logicalWidth publicFits).Challenge}
    {statement : (RefreshSchedule.schedule logicalWidth publicFits).Statement}
    {refreshMessages : (RefreshSchedule.schedule logicalWidth publicFits).Messages width'}
    {refreshChallenge : (RefreshSchedule.schedule logicalWidth publicFits).Challenge}
    {calls : List Call}
    (light : calls <+:
      (schedule logicalWidth publicFits).challengeCalls fresh lightMessages challenge)
    (refresh : calls <+: (RefreshSchedule.schedule logicalWidth publicFits).challengeCalls statement
      refreshMessages refreshChallenge) :
    calls = [] :=
  Schedule.challengeCalls_separated _ foldDomainChunks_distinct.1 light refresh

end Schedule

section Seals

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) LightShape)
  (fresh : Statement logicalWidth publicFits)
  (proof : Light.Proof LightShape (ProductionKey.degreeBound relation))

theorem publicInputState_eq_run :
    ((keys relation ajtai).light.transcriptStatement running fresh).priorState =
      run Transcript.initialState ((schedule logicalWidth publicFits).statementCalls fresh) := by
  show Transcript.absorb lightFoldInitialState (statementWords fresh) = _
  rw [lightFoldInitialState, absorb_eq_run, absorb_eq_run, ← run_append]
  rfl

theorem coins_eq_reads :
    (keys relation ajtai).light.coins running fresh proof.piCcsRounds =
      { alpha := ⟨List.ofFn fun coordinate => readK (slotOf coordinate.val)
            (run Transcript.initialState ((schedule logicalWidth publicFits).challengeCalls fresh
              (messages proof) (.alpha coordinate))),
          List.length_ofFn⟩
        gamma := readK (slotOf LightShape.cubeVariables) (run Transcript.initialState
          ((schedule logicalWidth publicFits).challengeCalls fresh (messages proof) .gamma))
        roundPoint := ⟨List.ofFn fun index => readK (slotOf 0) (run Transcript.initialState
            ((schedule logicalWidth publicFits).challengeCalls fresh (messages proof)
              (.round index))),
          List.length_ofFn⟩
        finalState := run Transcript.initialState
          ((schedule logicalWidth publicFits).roundsCalls fresh (messages proof)) } :=
  Schedule.derive_eq_reads (schedule logicalWidth publicFits) fresh (messages proof)
    (publicInputState_eq_run relation ajtai running fresh)

theorem probe_coins :
    ((keys relation ajtai).light.probe running fresh proof).coins =
      (schedule logicalWidth publicFits).coinsFrom Schedule.spongeRead fresh (messages proof) := by
  simp only [Light.Key.probe, coins_eq_reads relation ajtai running fresh proof]
  rfl

/-- The light Π_RLC challenge (`OptionB.lightChallenge`) is the read at the schedule's one rho
address. -/
theorem challenge_seal :
    (keys relation ajtai).light.challenge running fresh proof =
      some (readRho (run Transcript.initialState
        ((schedule logicalWidth publicFits).challengeCalls fresh (messages proof)
          (.rho ⟨0, Nat.zero_lt_one⟩)))) := by
  unfold Light.Key.challenge
  rw [coins_eq_reads relation ajtai running fresh proof]
  show some (lightChallenge (Transcript.absorb (run Transcript.initialState
      ((schedule logicalWidth publicFits).roundsCalls fresh (messages proof)))
      ((schedule logicalWidth publicFits).outputWords (messages proof).output))) = _
  rw [Schedule.absorbOutput_eq_run (schedule logicalWidth publicFits) fresh (messages proof),
    lightChallenge, Transcript.PiRlcSampler.piRlcChallenges,
    Transcript.PiRlcSampler.piRlcChallengesWithState_challenges]
  exact congrArg some (Schedule.rho_read (schedule logicalWidth publicFits) fresh (messages proof)
    ⟨0, Nat.zero_lt_one⟩)

end Seals

end NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightSchedule
