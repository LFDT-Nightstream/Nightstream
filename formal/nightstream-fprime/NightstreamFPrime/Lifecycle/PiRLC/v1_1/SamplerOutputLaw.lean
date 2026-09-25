import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerFieldShortfall
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldOutputLaw
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.BitOutputLaw

/-!
Owns the output-law consumer for the existing scalar conversion. Actual
execution is identified pointwise; frequencies concern explicit uniform
field and bit inputs, with abort retained. No law is assigned to Poseidon2.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerOutputLaw

open NightstreamFPrime.Spec
open Sampling Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet

def fieldDecode (fields : FieldShortfall.FieldWindow) : Option Scalar :=
  (FirstAccepted.boundedSample verifier coefficientCount
    (List.ofFn (FieldShortfall.fieldCandidates fields))).map
      Transcript.PiRlcSampler.scalarOfList

def bitDecode (window : ShortfallBound.Window) : Option Scalar :=
  (FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)).map
    Transcript.PiRlcSampler.scalarOfList

/-- The comparison decoder uses the exact conversion of the selected sampler. -/
theorem sampleScalar_eq_fieldDecode (initial : Transcript.State) (coordinate : Nat) :
    Transcript.PiRlcSampler.sampleScalar initial coordinate =
      fieldDecode (SamplerFieldShortfall.fieldWindow initial coordinate) := by
  rw [SamplerShortfall.sampleScalar_eq_windowDecode,
    SamplerFieldShortfall.candidateWindow_eq_fieldCandidates]
  rfl

/-- Successful exact-length lists never use the scalar conversion's default. -/
theorem scalarOfList_ofFn (scalar : Scalar) :
    Transcript.PiRlcSampler.scalarOfList (List.ofFn scalar) = scalar := by
  funext position
  simp [Transcript.PiRlcSampler.scalarOfList, List.getD_eq_getElem?_getD]

/-- Any event on actual scalar values, including failure, has the field/bit
comparison bound. The two denominators name the independent input spaces. -/
theorem field_bit_event_error_le (event : Option Scalar → Prop) :
    |(Nat.card {fields : FieldShortfall.FieldWindow // event (fieldDecode fields)} : ℚ) /
        (goldilocksModulus : ℚ) ^ FieldShortfall.fieldLaneCount -
      (Nat.card {window : ShortfallBound.Window // event (bitDecode window)} : ℚ) /
        (chunkModulus : ℚ) ^ candidateBound| ≤
      (FieldShortfall.fieldLaneCount : ℚ) * FieldPairLaw.pairDeviation := by
  exact FieldOutputLaw.boundedSample_event_frequency_error_le
    (fun output => event (output.map Transcript.PiRlcSampler.scalarOfList))

/-- The selected scalar decoder on independent uniform fields is compared
with a uniform successful scalar in the same Option space. Failure remains
part of the left distribution and contributes the proved shortfall bound. -/
theorem field_output_event_error_le (event : Option Scalar → Prop) :
    |(Nat.card {fields : FieldShortfall.FieldWindow // event (fieldDecode fields)} : ℚ) /
        (goldilocksModulus : ℚ) ^ FieldShortfall.fieldLaneCount -
      (Nat.card {scalar : Scalar // event (some scalar)} : ℚ) /
        (alphabetSize : ℚ) ^ coefficientCount| ≤
      32 * FieldPairLaw.pairDeviation +
        (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11 := by
  simpa only [FieldOutputLaw.fieldDecodedFrequency,
    BitOutputLaw.uniformSomeFrequency, fieldDecode,
    Option.map_some, scalarOfList_ofFn] using
      BitOutputLaw.boundedSample_field_event_error_le_upper
        (fun output => event (output.map Transcript.PiRlcSampler.scalarOfList))

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerOutputLaw
