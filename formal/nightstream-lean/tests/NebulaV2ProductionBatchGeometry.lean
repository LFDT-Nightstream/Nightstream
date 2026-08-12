import Nightstream.Protocol.NebulaV2.ProductionBatchGeometry

/-! Regression surface for successor batch indexing and geometry. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionBatchGeometry

open Nightstream.Protocol.NebulaV2.ProductionBatchGeometry
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

#check encode_bijective
#check candidate_geometry_table

example : encode .e8 (decode .e8 ⟨1087, by decide⟩) =
    ⟨1087, by decide⟩ := encode_decode .e8 _

example : (decode .e8 ⟨1087, by decide⟩).batch.val = 135 := by decide
example : (decode .e8 ⟨1087, by decide⟩).within.val = 7 := by decide

end tests.NebulaV2ProductionBatchGeometry
