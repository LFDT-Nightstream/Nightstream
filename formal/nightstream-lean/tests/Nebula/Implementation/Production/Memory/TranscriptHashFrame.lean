import Nightstream.Implementation.Nebula.Production.Memory.TranscriptHashFrame

/-! Regression surface for the successor memory challenge frame. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryTranscriptHashFrame

open Nightstream.Implementation.Nebula.ProductionMemoryTranscriptHashFrame

#check encode_injective
#check encode_joint_injective
#check encode_eq_encodeFor
#check encode_ne_v2
#check candidate_eq_of_encode_eq

/-- Two different checked-step profiles cannot have one pre-hash frame. -/
example (left right : Input) : encode .e4 left ≠ encode .e8 right := by
  intro equal
  have candidates := candidate_eq_of_encode_eq equal
  cases candidates

end tests.NebulaProductionMemoryTranscriptHashFrame
