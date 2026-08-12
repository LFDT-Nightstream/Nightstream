import Nightstream.Protocol.NebulaV2.Transcript

set_option autoImplicit false

namespace tests.NebulaV2Transcript

open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.Transcript

def frame : Frame Nat :=
  { profile := Nightstream.Protocol.NebulaV2.Profile.v2
    verifierKeyDigest := 1
    applicationRelationDigest := 2
    programDigest := 3
    memoryPlanDigest := 4
    laneLayoutDigest := 5
    priorStateDigest := 6
    runningAccumulatorDigest := 7
    segmentIndex := 8
    segmentStartTimestamp := 9
    activeAccessCount := 10
    segmentEndTimestamp := 19
    roots := ⟨11, 12, 13⟩ }

theorem exact_frame_length : (encode frame).length = 19 :=
  encode_length frame

theorem four_coordinates_are_distinct_positions :
    (coordinateIndex 0 0).val = 0 ∧
      (coordinateIndex 0 1).val = 1 ∧
      (coordinateIndex 1 0).val = 2 ∧
      (coordinateIndex 1 1).val = 3 := by
  decide

/-- The model names the exact security boundary: complete framing alone does
not make a constant challenge function secure. -/
theorem constant_challenge_countermodel :
    derive (constantOracle 7) frame =
      fun _ => { gamma1 := 7, gamma2 := 7 } := by
  rfl

theorem constant_oracle_reuses_both_repetitions :
    derive (constantOracle 7) frame 0 =
      derive (constantOracle 7) frame 1 := by
  rfl

end tests.NebulaV2Transcript
