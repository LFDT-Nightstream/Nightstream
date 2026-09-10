import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerOutputLaw
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldDensityLaw

/-!
Owns successful-event density for the selected scalar decoder on independent
uniform field inputs. The existing sampleScalar_eq_fieldDecode theorem
identifies actual execution pointwise; no transcript law is asserted here.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerDensityLaw

open NightstreamFPrime.Spec
open Sampling Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet SamplerOutputLaw

/-- The exact deployed scalar conversion preserves the finite density bound.
Singleton events give the bound for any requested scalar; abort is excluded. -/
theorem field_output_event_le (event : Option Scalar → Prop) (noAbort : ¬ event none) :
    (Nat.card {fields : FieldShortfall.FieldWindow // event (fieldDecode fields)} : ℚ) /
        (goldilocksModulus : ℚ) ^ FieldShortfall.fieldLaneCount ≤
      ((FieldPairLaw.pairModulus : ℚ) ^ 2 / goldilocksModulus) ^ FieldShortfall.fieldLaneCount *
        ((Nat.card {scalar : Scalar // event (some scalar)} : ℚ) /
          (alphabetSize : ℚ) ^ coefficientCount) := by
  have rejects : ¬ (fun output : Option (List Coefficient) =>
      event (output.map Transcript.PiRlcSampler.scalarOfList)) none := by
    simpa only [Option.map_none] using noAbort
  simpa only [FieldOutputLaw.fieldDecodedFrequency, BitOutputLaw.uniformSomeFrequency,
    fieldDecode, Option.map_some, scalarOfList_ofFn] using
    FieldDensityLaw.boundedSample_success_event_le
      (fun output => event (output.map Transcript.PiRlcSampler.scalarOfList)) rejects

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerDensityLaw
