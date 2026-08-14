import Nightstream.Implementation.Nebula.NIFS.PiRLC.PostPiCcsBridge
import Nightstream.Implementation.Nebula.NIFS.PiRLC.SamplerResponseSound

/-! Regression surface for the exact V2 full-field PiRLC sampler rows. -/

set_option autoImplicit false

namespace tests.NebulaProductPiRlcSamplerRows

#check Nightstream.Implementation.Nebula.ProductPiCcsTranscriptCursor.afterFullOutput_absorbed
#check Nightstream.Implementation.Nebula.ProductPiRlcPostPiCcsBridge.rows_imply_candidate_exact
#check Nightstream.Implementation.Nebula.ProductPiRlcCandidateClassificationSound.all_candidates_sound
#check Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedSound.sound
#check Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedBatchSound.sampleCoefficient_eq_some_output
#check Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound.sampler_available
#check Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound.samplerSucceeded_eq_true
#check Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound.output_eq_scalarResponse

example :
    Nightstream.Implementation.Nebula.ProductPiRlcTranscriptRows.aggregateRowCount =
      1710720 :=
  Nightstream.Implementation.Nebula.ProductPiRlcTranscriptRows.aggregateRowCount_eq

example :
    Nightstream.Implementation.Nebula.ProductPiRlcCandidateClassificationRows.aggregateRowCount =
      216270 :=
  Nightstream.Implementation.Nebula.ProductPiRlcCandidateClassificationRows.aggregateRowCount_eq

example :
    Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount =
      7290 :=
  Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount_eq

end tests.NebulaProductPiRlcSamplerRows
