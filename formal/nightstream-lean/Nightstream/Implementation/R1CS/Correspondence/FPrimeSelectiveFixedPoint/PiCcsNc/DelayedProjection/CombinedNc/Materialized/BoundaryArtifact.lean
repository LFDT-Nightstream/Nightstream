import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Metadata
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder

/-!
Fail-closed decoding of the generated production delayed combined-NC boundary.

Owns: one typed interpretation of the exact generated boundary record and
proof that decoding preserves that record verbatim.

Does not own: row satisfaction, meanings of the named columns, transcript
ordering, raw-child authority, commitment binding, costs, or row removal.

Emits constraints: none.

The executable certificate examines one proof-free `RawBoundaryMap`. Its
nested payload consists of fifteen rows of sixty-four output column pairs,
fifty-four pending-parent pairs, nineteen pending/block-point/beta pairs,
six lane-point/beta pairs, and scalar/range metadata. No assignment or proof
object is evaluated by the certificate.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.BoundaryArtifact

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder

abbrev rawBoundary : RawBoundaryMap := Generated.Metadata.boundary

private theorem rawBoundary_isSome :
    (decodeBoundaryMap rawBoundary).isSome := by
  native_decide

/-- The generated value paired with its successful fail-closed decode. -/
def certificate :
    { decoded : DecodedBoundaryMap //
      decodeBoundaryMap rawBoundary = some decoded } :=
  let decoded := (decodeBoundaryMap rawBoundary).get rawBoundary_isSome
  ⟨decoded, Option.eq_some_of_isSome rawBoundary_isSome⟩

def decodedBoundary : DecodedBoundaryMap := certificate.1

theorem decodedBoundary_exact :
    decodeBoundaryMap rawBoundary = some decodedBoundary :=
  certificate.2

/-- Decoding validates but never substitutes a different boundary record. -/
theorem decodedBoundary_raw : decodedBoundary.raw = rawBoundary := by
  have exact := decodedBoundary_exact
  unfold decodeBoundaryMap at exact
  split at exact
  · exact (congrArg DecodedBoundaryMap.raw (Option.some.inj exact)).symm
  · contradiction

theorem boundary_valid : boundaryMapValid rawBoundary := by
  rw [← decodedBoundary_raw]
  exact decodedBoundary.valid

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.BoundaryArtifact
