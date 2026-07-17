import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCompiler
import Nightstream.SuperNeo.Concrete.Phi81Relation.Types

/-!
Low-level physical CE-claim shape contract.

Assurance tier: model-level schema.

Owns: the point length, matrix-evaluation count, and active evaluation-row
width required before one strict-`Pi_DEC` claim can be decoded at a typed
Phi81 relation shape.

Does not own: any generated artifact, production relation profile, point or
evaluation values, padding zeroes, transcript authority, CE membership, Rust
conformance, R1CS rows, costs, or row removal.

Emits constraints: no.

Authority boundary: all three facts are explicit checked premises. Decoding,
a digest, or a generated count cannot manufacture them.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.claim.shape.point` | physical `r` pair count equals semantic row variables | checked | `Holds.pointLength` |
| `nifs.claim.shape.evaluations` | physical `y_ring` row count equals semantic matrix count | checked | `Holds.evaluationCount` |
| `nifs.claim.shape.evaluation_width` | every physical `y_ring` row exposes all 54 extension-field coefficients | checked | `Holds.activeEvaluationWidth` |
| `nifs.claim.shape.three_row_rejection` | a three-row claim cannot align with a non-three-matrix relation | derived | `not_aligned_of_threeRows` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShape

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.Implementation.R1CS.PiDecStrictCompiler

/-- Exact physical layout facts required to decode one claim at one semantic
relation shape. -/
structure Holds (shape : Shape) (claim : ClaimLayout) : Prop where
  pointLength : claim.rCols.length = shape.rowVariables
  evaluationCount : claim.yRingCols.length = shape.matrixCount
  activeEvaluationWidth : ∀ row, row ∈ claim.yRingCols →
    2 * ringDegree <= row.length

/-- The evaluation-count branch alone rejects a fixed three-row claim when
the semantic relation has another matrix count. -/
theorem not_aligned_of_threeRows
    {shape : Shape} {claim : ClaimLayout}
    (threeRows : claim.yRingCols.length = 3)
    (matrixCount : shape.matrixCount ≠ 3) :
    Not (Holds shape claim) := by
  intro alignment
  exact matrixCount (alignment.evaluationCount.symm.trans threeRows)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShape
