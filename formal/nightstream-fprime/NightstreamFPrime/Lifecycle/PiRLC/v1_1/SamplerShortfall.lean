import NightstreamFPrime.Lifecycle.Transcript
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ShortfallBound

/-!
Owns the link from the actual scalar decoder to its 64-candidate failure
event. The probability bound concerns an explicit uniform-bit comparison
of that decoder; it assigns no law to the Poseidon2 source or its state.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerShortfall

open NightstreamFPrime.Spec
open Sampling
open Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet

/-- Every candidate comes from the actual coordinate's verifier-owned source. -/
def candidateWindow (initial : Transcript.State) (coordinate : Nat) :
    ShortfallBound.Window :=
  fun index => (sourceAt Transcript.PiRlcSampler.specification initial coordinate).stream index.val

private theorem sourcePrefix_eq_window (initial : Transcript.State) (coordinate : Nat) :
    FirstAccepted.streamPrefix
        (sourceAt Transcript.PiRlcSampler.specification initial coordinate).stream candidateBound =
      List.ofFn (candidateWindow initial coordinate) := by
  apply List.ext_getElem
  · simp
  · intro index leftBound rightBound
    simp [FirstAccepted.streamPrefix, candidateWindow]

/-- The selected transcript sampler calls the same bounded decoder on the
actual ordered window and then performs its existing scalar conversion. -/
theorem sampleScalar_eq_windowDecode (initial : Transcript.State) (coordinate : Nat) :
    Transcript.PiRlcSampler.sampleScalar initial coordinate =
      (FirstAccepted.boundedSample verifier coefficientCount
        (List.ofFn (candidateWindow initial coordinate))).map
          Transcript.PiRlcSampler.scalarOfList := by
  unfold Transcript.PiRlcSampler.sampleScalar
  dsimp only
  rw [sourcePrefix_eq_window]

/-- Actual scalar failure is exactly the eleven-rejection event on its
own transcript-derived candidates. -/
theorem sampleScalar_none_iff_eleven_rejections
    (initial : Transcript.State) (coordinate : Nat) :
    Transcript.PiRlcSampler.sampleScalar initial coordinate = none ↔
      11 ≤ (ShortfallBound.rejectedPositions (candidateWindow initial coordinate)).card := by
  rw [sampleScalar_eq_windowDecode, Option.map_eq_none_iff]
  exact ShortfallBound.bounded_sample_none_iff_eleven_rejections _

/-- The actual decoder's failure bound when its input is replaced only in
the comparison experiment by 64 independent uniform 16-bit candidates. -/
theorem iid_bit_decoder_failure_probability_le :
    (Nat.card {window : ShortfallBound.Window //
      (FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)).map
        Transcript.PiRlcSampler.scalarOfList = none} : ℚ) /
        (chunkModulus : ℚ) ^ candidateBound ≤
      (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11 := by
  have sameFailure (window : ShortfallBound.Window) :
      (FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)).map
          Transcript.PiRlcSampler.scalarOfList = none ↔
        FirstAccepted.Shortfall verifier coefficientCount (List.ofFn window) := by
    rw [Option.map_eq_none_iff, FirstAccepted.boundedSample_eq_none_iff_shortfall]
  have sameCount := Nat.card_congr (Equiv.subtypeEquivRight sameFailure)
  rw [sameCount]
  exact ShortfallBound.iid_bit_shortfall_probability_le

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerShortfall
