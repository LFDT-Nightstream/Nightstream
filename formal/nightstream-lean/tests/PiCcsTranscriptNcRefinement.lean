import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.NcRefinement

/-!
Focused model-level regressions for exact-width NC transcript refinement.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.transcript.transport` | `K` and implementation extension values round-trip | silent modulus or limb-order mismatch |
| `nifs.pi_ccs.nc.transcript.message.coefficients` | five coefficients produce ten payload fields | trimming, missing limb, or extra payload |
| `nifs.pi_ccs.nc.transcript.round` | semantic absorb/squeeze equals concrete `runRound` | tuple-order or state-thread mismatch |
| `nifs.pi_ccs.nc.transcript.phase` | semantic derive equals concrete `runRounds` and `runNc` | prologue, ordering, challenge, or final-state drift |
-/

namespace NightstreamTests.PiCcsTranscriptNcRefinement

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc

private def semanticValue : K :=
  { c0 := ⟨3, by decide⟩
    c1 := ⟨5, by decide⟩ }

private def implementationValue : Extension :=
  { c0 := wordField 7
    c1 := wordField 11 }

example : toK (toExtension semanticValue) = semanticValue := by
  simp

example : toExtension (toK implementationValue) = implementationValue := by
  simp

private def zeroRound : RoundMessage where
  coefficients := [K.zero, K.zero, K.zero, K.zero, K.zero]
  coefficients_length := rfl

example : (toConcreteRound zeroRound).coefficients.length = 5 := by
  exact toConcreteRound_coefficients_length zeroRound

example :
    (SumCheck.roundFields (toConcreteRound zeroRound)).length = 10 := by
  exact toConcreteRound_fields_length zeroRound

example (state : State) :
    runRound machine state zeroRound =
      let concrete := SumCheck.runRound state (toConcreteRound zeroRound)
      (toK concrete.2, concrete.1) :=
  runRound_refines state zeroRound

private def twoRoundDomain : FlatNcDomain where
  columnVariables := 1
  laneVariables := 1

private def certificate : Certificate twoRoundDomain where
  rounds := fun _ => zeroRound

private def concreteMessages : SumCheck.Messages where
  feInitial := Extension.zero
  feRounds := []
  ncRounds := concreteRounds certificate

example (state : State) :
    ((derive machine state certificate).challengePoint.coordinates,
        (derive machine state certificate).finalState) =
      let concrete := SumCheck.runRounds (SumCheck.ncPrologue state)
        (concreteRounds certificate)
      (concrete.2.map toK, concrete.1) :=
  derive_refines_runRounds state certificate

example (state : State) :
    ((derive machine state certificate).challengePoint.coordinates,
        (derive machine state certificate).finalState) =
      let concrete := SumCheck.runNc state concreteMessages
      (concrete.2.map toK, concrete.1) :=
  derive_refines_runNc state certificate concreteMessages rfl

end NightstreamTests.PiCcsTranscriptNcRefinement
