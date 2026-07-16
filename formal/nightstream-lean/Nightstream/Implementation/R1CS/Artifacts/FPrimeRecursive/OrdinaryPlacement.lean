import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.Generated.FPrimeBranchOrdinaryPlacementData
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.SourceRoleCensus

/-!
Contract: checked ordinary-private encoded placement summaries for the fixed
F-prime base and recursive gadget-native branches.

Owns: one executable metadata certificate per branch, exact source-phase ends,
final encoded-column bounds, ordinary coordinate totals, and first/last start
regressions derived from the checked source-role runs.

Does not own: centered witness words, parser/materializer invertibility,
constraint rows, CE coordinates, selector composition, row removal, or
lifecycle authority.

Emits constraints: no.

Authority boundary: exact starts come from `placementStart?`, which scans the
already checked source-role runs using fixed production allocation widths.
The generated metadata supplies only the source-phase end and final encoded
bound. Rust conformance additionally requires the production observer and
byte-for-byte generator drift test.

| Branch/result | Mathematical obligation | Guarantee | Assurance boundary |
|---|---|---|---|
| metadata checks | recomputed source phase equals Rust summary and fits final width | `base_data_check`, `recursive_data_check` | artifact-checked |
| ordinary totals | 3,050/154,747 fields occupy 125,050/6,344,627 ABI coordinates | `base_ordinaryCoordinateCount`, `recursive_ordinaryCoordinateCount` | artifact-checked arithmetic |
| placement endpoints | first and last starts follow the complete checked run scan | `base_firstPlacement`, `base_lastPlacement`, `recursive_firstPlacement`, `recursive_lastPlacement` | artifact-checked; Rust-conformant after drift gate |
| open bridge | accepted words need not equal deterministic re-encodings | explicit non-goal | no NIVC-invertibility or row-removal claim |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement

open Nightstream.Implementation.R1CS.FPrimeFieldLayout
open Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement
open Nightstream.Implementation.R1CS.FPrimeBranchOrdinaryPlacementData
open Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus

theorem base_data_check : baseData.check baseSourceCensus = true := by
  native_decide

theorem recursive_data_check :
    recursiveData.check recursiveSourceCensus = true := by
  native_decide

theorem base_data_valid : baseData.ValidFor baseSourceCensus :=
  baseData.check_sound baseSourceCensus base_data_check

theorem recursive_data_valid :
    recursiveData.ValidFor recursiveSourceCensus :=
  recursiveData.check_sound recursiveSourceCensus recursive_data_check

theorem base_publicInputLength : publicInputLength baseSourceCensus = 257 := by
  native_decide

theorem recursive_publicInputLength :
    publicInputLength recursiveSourceCensus = 257 := by
  native_decide

theorem base_sourcePhaseEnd : sourcePhaseEnd baseSourceCensus = 125695 := by
  have exactPhase := base_data_valid.2.1
  simpa [baseData] using exactPhase.symm

theorem recursive_sourcePhaseEnd :
    sourcePhaseEnd recursiveSourceCensus = 7830083 := by
  have exactPhase := recursive_data_valid.2.1
  simpa [recursiveData] using exactPhase.symm

theorem base_sourcePhase_fits_encoded :
    sourcePhaseEnd baseSourceCensus ≤ 125695 := by
  simpa [baseData] using
    base_data_valid.sourcePhaseEnd_le_encodedColumnCount

theorem recursive_sourcePhase_fits_encoded :
    sourcePhaseEnd recursiveSourceCensus ≤ 8137378 := by
  simpa [recursiveData] using
    recursive_data_valid.sourcePhaseEnd_le_encodedColumnCount

theorem base_ordinaryCoordinateCount :
    ordinaryCoordinateCount baseSourceCensus = 125050 := by
  simp [ordinaryCoordinateCount, ordinaryWordWidth, base_eligible_count]

theorem recursive_ordinaryCoordinateCount :
    ordinaryCoordinateCount recursiveSourceCensus = 6344627 := by
  simp [ordinaryCoordinateCount, ordinaryWordWidth,
    recursive_eligible_count]

theorem base_firstPlacement :
    placementStart? baseSourceCensus 1 = some 257 := by
  native_decide

theorem base_lastPlacement :
    placementStart? baseSourceCensus 22336 = some 125654 := by
  native_decide

theorem recursive_firstPlacement :
    placementStart? recursiveSourceCensus 1 = some 257 := by
  native_decide

theorem recursive_lastPlacement :
    placementStart? recursiveSourceCensus 2399090 = some 7830042 := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement
