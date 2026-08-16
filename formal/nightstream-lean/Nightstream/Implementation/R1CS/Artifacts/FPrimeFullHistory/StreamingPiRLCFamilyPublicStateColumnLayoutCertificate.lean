import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicCertificateSupport

/-!
Contract: structural state-column layout certificate for both Rust-emitted
PiRLC public-family arms.

Owns four exact decompositions of the 937-column state layouts into short,
disjoint intervals. It owns no row data or state semantics.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport

def evenBeforeSegments : List Segment :=
  [{ start := 146224, length := 8 },
   { start := 275374, length := 1 },
   { start := 144386, length := 108 },
   { start := 146232, length := 8 },
   { start := 275375, length := 1 },
   { start := 144602, length := 810 },
   { start := 146222, length := 1 }]

def evenAfterSegments : List Segment :=
  [{ start := 1619, length := 2 },
   { start := 267434, length := 6 },
   { start := 275376, length := 1 },
   { start := 144494, length := 108 },
   { start := 1673, length := 2 },
   { start := 275234, length := 6 },
   { start := 275377, length := 1 },
   { start := 145412, length := 810 },
   { start := 146223, length := 1 }]

def oddBeforeSegments : List Segment :=
  [{ start := 146224, length := 8 },
   { start := 276574, length := 1 },
   { start := 144386, length := 108 },
   { start := 146232, length := 8 },
   { start := 276575, length := 1 },
   { start := 144602, length := 810 },
   { start := 146222, length := 1 }]

def oddAfterSegments : List Segment :=
  [{ start := 268032, length := 8 },
   { start := 276576, length := 1 },
   { start := 144494, length := 108 },
   { start := 276432, length := 8 },
   { start := 276577, length := 1 },
   { start := 145412, length := 810 },
   { start := 146223, length := 1 }]

theorem evenBefore_exact :
    evenArm.beforeStateColumns = expandSegments evenBeforeSegments := by
  rfl

theorem evenAfter_exact :
    evenArm.afterStateColumns = expandSegments evenAfterSegments := by
  rfl

theorem oddBefore_exact :
    oddArm.beforeStateColumns = expandSegments oddBeforeSegments := by
  rfl

theorem oddAfter_exact :
    oddArm.afterStateColumns = expandSegments oddAfterSegments := by
  rfl

theorem evenBefore_length :
    (evenBeforeSegments.map Segment.length).sum = 937 := by
  rfl

theorem evenAfter_length :
    (evenAfterSegments.map Segment.length).sum = 937 := by
  rfl

theorem oddBefore_length :
    (oddBeforeSegments.map Segment.length).sum = 937 := by
  rfl

theorem oddAfter_length :
    (oddAfterSegments.map Segment.length).sum = 937 := by
  rfl

theorem evenBefore_segments_valid :
    SegmentsValid evenArm.columnCount evenBeforeSegments := by
  norm_num [SegmentsValid, Segment.Disjoint, evenBeforeSegments, evenArm]

theorem evenAfter_segments_valid :
    SegmentsValid evenArm.columnCount evenAfterSegments := by
  norm_num [SegmentsValid, Segment.Disjoint, evenAfterSegments, evenArm]

theorem oddBefore_segments_valid :
    SegmentsValid oddArm.columnCount oddBeforeSegments := by
  norm_num [SegmentsValid, Segment.Disjoint, oddBeforeSegments, oddArm]

theorem oddAfter_segments_valid :
    SegmentsValid oddArm.columnCount oddAfterSegments := by
  norm_num [SegmentsValid, Segment.Disjoint, oddAfterSegments, oddArm]

theorem evenArm_stateColumnLayout_valid : evenArm.StateColumnLayoutValid :=
  ⟨columnsValid_of_segments evenBefore_exact evenBefore_length
      evenBefore_segments_valid,
    columnsValid_of_segments evenAfter_exact evenAfter_length
      evenAfter_segments_valid⟩

theorem oddArm_stateColumnLayout_valid : oddArm.StateColumnLayoutValid :=
  ⟨columnsValid_of_segments oddBefore_exact oddBefore_length
      oddBefore_segments_valid,
    columnsValid_of_segments oddAfter_exact oddAfter_length
      oddAfter_segments_valid⟩

theorem evenArm_beforeState_last_is_cursor :
    evenArm.beforeStateColumns.getD 936 0 =
      evenArm.beforeFamilyCursorColumn := by
  rw [evenBefore_exact]
  rfl

theorem evenArm_afterState_last_is_cursor :
    evenArm.afterStateColumns.getD 936 0 =
      evenArm.afterFamilyCursorColumn := by
  rw [evenAfter_exact]
  rfl

theorem oddArm_beforeState_last_is_cursor :
    oddArm.beforeStateColumns.getD 936 0 =
      oddArm.beforeFamilyCursorColumn := by
  rw [oddBefore_exact]
  rfl

theorem oddArm_afterState_last_is_cursor :
    oddArm.afterStateColumns.getD 936 0 =
      oddArm.afterFamilyCursorColumn := by
  rw [oddAfter_exact]
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate
