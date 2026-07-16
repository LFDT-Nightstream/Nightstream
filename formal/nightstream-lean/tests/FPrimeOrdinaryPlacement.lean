import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive

/-!
Contract: executable geometry, mutation, and concrete production regressions
for ordinary-private source-loop placement.

Owns: a small mixed-role allocation fixture, exact 41-word ordering checks,
compact metadata mutations, and stable facade theorem-shape checks.

Does not own: Rust generation, chosen centered words, constraint emission, CE
authority, row removal, or NIVC invertibility.

Emits constraints: no.

| Test family | Mathematical obligation | Expected result |
|---|---|---|
| mixed-role scan | public prefix, two ordinary runs, canonical 95, and Boolean 1 compose in order | starts 2, 43, and 180; phase end 221 |
| word geometry | increasing offsets produce disjoint 41-coordinate words | ordered end/start inequality |
| metadata mutations | version, phase end, and encoded bound drift fail closed | `check = false` |
| production pins | fixed branch totals and endpoint placements stay exact | facade equalities |
-/

namespace NightstreamTests.FPrimeOrdinaryPlacement

open Nightstream.Implementation.R1CS.FPrimeFieldLayout
open Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement

def tinySegments : List SourceSegment :=
  [{ ownerPath := "constant", role := .constantOne,
     source := { start := 0, length := 1 } },
   { ownerPath := "ordinary.a", role := .ordinaryPrivateField,
     source := { start := 1, length := 2 } },
   { ownerPath := "canonical", role := .canonicalU64,
     source := { start := 3, length := 1 } },
   { ownerPath := "boolean", role := .privateBoolean,
     source := { start := 4, length := 1 } },
   { ownerPath := "ordinary.b", role := .ordinaryPrivateField,
     source := { start := 5, length := 1 } },
   { ownerPath := "public", role := .publicBit,
     source := { start := 6, length := 1 } }]

def tinyRoleCount : SlotRole → Nat
  | .constantOne => 1
  | .ordinaryPrivateField => 3
  | .privateBoolean => 1
  | .publicBit => 1
  | .canonicalU64 => 1
  | .sisOpening => 0
  | .linearlyDerived => 0
  | .structuralBalancedAlias => 0
  | .gadgetDerived => 0
  | .productDerived => 0
  | .gadgetTemporary => 0

def tinyArtifact : SourceCensusArtifact where
  sourceColumnCount := 7
  sourceSegments := tinySegments
  declaredRoleCount := tinyRoleCount
  sourcePartition := by
    simp [tinySegments, ExactPartition, ExactPartitionFrom,
      CoordinateRun.endExclusive]
  roleCensusExact := by
    intro role
    cases role <;> native_decide

def tinyMetadata : Metadata where
  formatVersion := currentFormatVersion
  sourcePhaseEnd := 221
  encodedColumnCount := 225

example : publicInputLength tinyArtifact = 2 := by native_decide
example : sourcePhaseEnd tinyArtifact = 221 := by native_decide
example : ordinaryCoordinateCount tinyArtifact = 123 := by native_decide
example : placementStart? tinyArtifact 1 = some 2 := by native_decide
example : placementStart? tinyArtifact 2 = some 43 := by native_decide
example : placementStart? tinyArtifact 3 = none := by native_decide
example : placementStart? tinyArtifact 5 = some 180 := by native_decide
example : tinyMetadata.check tinyArtifact = true := by native_decide

example : ({ tinyMetadata with formatVersion := 2 }).check tinyArtifact = false := by
  native_decide

example : ({ tinyMetadata with sourcePhaseEnd := 220 }).check tinyArtifact = false := by
  native_decide

example : ({ tinyMetadata with encodedColumnCount := 220 }).check tinyArtifact = false := by
  native_decide

example : (wordRun (sameSegmentStart 2 0)).endExclusive ≤
    (wordRun (sameSegmentStart 2 1)).start :=
  sameSegment_wordRun_before 2 0 1 (by omega)

namespace Concrete

open Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement
open Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus

example : sourcePhaseEnd baseSourceCensus = 125695 := base_sourcePhaseEnd
example : sourcePhaseEnd recursiveSourceCensus = 7830083 :=
  recursive_sourcePhaseEnd
example : ordinaryCoordinateCount baseSourceCensus = 125050 :=
  base_ordinaryCoordinateCount
example : ordinaryCoordinateCount recursiveSourceCensus = 6344627 :=
  recursive_ordinaryCoordinateCount
example : placementStart? baseSourceCensus 1 = some 257 := base_firstPlacement
example : placementStart? baseSourceCensus 22336 = some 125654 :=
  base_lastPlacement
example : placementStart? recursiveSourceCensus 1 = some 257 :=
  recursive_firstPlacement
example : placementStart? recursiveSourceCensus 2399090 = some 7830042 :=
  recursive_lastPlacement

end Concrete
end NightstreamTests.FPrimeOrdinaryPlacement
