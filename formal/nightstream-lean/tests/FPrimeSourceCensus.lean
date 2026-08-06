import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.SourceRoleCensus

/-!
Contract: fail-closed parser mutations and public theorem-shape regressions
for the source-only fixed-F-prime census.

Owns: small malformed-input examples and stable-facade checks for the concrete
eligible counts and conditional per-field-41 floors.

Does not own: production artifact generation, Rust trace conformance, encoded
or CE layouts, selector composition, or permission to remove constraints.

Emits constraints: no.

Authority boundary: examples consume the stable Lean facade. The two large
certificates are checked in the imported artifact module and remain
non-authoritative without the Rust generator drift gate.

| Test family | Obligation | Expected result |
|---|---|---|
| packed mutations | malformed metadata, tokens, runs, counts, and initial ownership fail closed | `Data.check = false` |
| concrete census | base/recursive ordinary counts remain 3,226/99,314 | exact equality |
| conditional capacity | separate candidates retain the 4,204,140 floor; the recursive per-field-41 candidate alone retains the 1M no-go | theorem-shape regression |
-/

namespace NightstreamTests.FPrimeSourceCensus

open Nightstream.Implementation.R1CS.FPrimeFieldLayout

namespace Packed

open Nightstream.Implementation.R1CS.FPrimeFieldLayout.PackedSourceCensus

def validCounts : RoleCounts where
  constantOne := 1
  ordinaryPrivateField := 2
  privateBoolean := 0
  publicBit := 1
  canonicalU64 := 0
  sisOpening := 0
  linearlyDerived := 0
  structuralBalancedAlias := 0
  gadgetDerived := 0
  productDerived := 0
  gadgetTemporary := 0

/-- Three runs split across two chunks:
`(1, constantOne, stage 0)`, `(2, ordinary, stage 1)`, and
`(1, publicBit, stage 1)`. -/
def validData : Data where
  formatVersion := currentFormatVersion
  sourceColumnCount := 4
  runCount := 3
  declaredRoleCounts := validCounts
  stagePaths := #[constantOneOwnerPath, "application"]
  packedChunks := ["22,47", "29"]

theorem valid_check : validData.check = true := by native_decide

def validArtifact : SourceCensusArtifact :=
  validData.toSourceCensusArtifact valid_check

example : validArtifact.sourceSegments =
    [{ ownerPath := constantOneOwnerPath, role := .constantOne,
       source := { start := 0, length := 1 } },
     { ownerPath := "application", role := .ordinaryPrivateField,
       source := { start := 1, length := 2 } },
     { ownerPath := "application", role := .publicBit,
       source := { start := 3, length := 1 } }] := by
  native_decide

example : validArtifact.declaredRoleCount .ordinaryPrivateField = 2 := rfl

example : ({ validData with formatVersion := 2 }).check = false := by
  native_decide

example : ({ validData with stagePaths := #[] }).check = false := by
  native_decide

example : ({ validData with stagePaths := #["", "application"] }).check =
    false := by
  native_decide

example : ({ validData with stagePaths := #["same", "same"] }).check =
    false := by
  native_decide

example : ({ validData with packedChunks := ["22,x"] }).check = false := by
  native_decide

example : ({ validData with packedChunks := ["22,"] }).check = false := by
  native_decide

example : ({ validData with packedChunks := [""] }).check = false := by
  native_decide

example :
    ({ validData with stagePaths := #["wrong.constant", "application"] }).check =
      false := by
  native_decide

example : ({ validData with packedChunks := ["0"] }).check = false := by
  native_decide

example :
    ({ validData with
      runCount := 4
      packedChunks := ["22,25", "25,29"] }).check = false := by
  native_decide

example : ({ validData with sourceColumnCount := 5 }).check = false := by
  native_decide

example : ({ validData with runCount := 4 }).check = false := by
  native_decide

example :
    ({ validData with
      sourceColumnCount := 5
      runCount := 4
      declaredRoleCounts := { validCounts with constantOne := 2 }
      packedChunks := ["22,47,29,23"] }).check = false := by
  native_decide

example :
    ({ validData with
      declaredRoleCounts := { validCounts with ordinaryPrivateField := 3 } }).check =
      false := by
  native_decide

example :
    ({ validData with
      declaredRoleCounts :=
        { validCounts with constantOne := 0, ordinaryPrivateField := 3 }
      packedChunks := ["24,47,29"] }).check = false := by
  native_decide

example : slotRoleOfIndex? 11 = none := rfl

example : decodePackedRun #[] 22 = none := rfl

end Packed

namespace Concrete

open Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus

example : baseSourceCensus.eligibleCount = 3226 := base_eligible_count

example : recursiveSourceCensus.eligibleCount = 99314 :=
  recursive_eligible_count

example :
    SourceSegment.eligibleCountOf baseSourceCensus.sourceSegments = 3226 :=
  base_ordinaryRunSubtotal_count

example :
    SourceSegment.eligibleCountOf recursiveSourceCensus.sourceSegments =
      99314 :=
  recursive_ordinaryRunSubtotal_count

example {baseWidth recursiveWidth : Nat}
    (baseCapacity :
      baseSourceCensus.PerField41CapacityRequirement baseWidth)
    (recursiveCapacity :
      recursiveSourceCensus.PerField41CapacityRequirement recursiveWidth) :
    4204140 ≤ baseWidth + recursiveWidth :=
  combined_perField41_width_floor baseCapacity recursiveCapacity

example {recursiveWidth : Nat}
    (recursiveCapacity :
      recursiveSourceCensus.PerField41CapacityRequirement recursiveWidth) :
    ¬ recursiveWidth ≤ 1000000 :=
  recursive_one_million_perField41_budget_is_no_go recursiveCapacity

end Concrete

example (artifact : SourceCensusArtifact)
    {sourceColumn : Nat}
    (sourceColumnLt : sourceColumn < artifact.sourceColumnCount) :
    ∃ role : SlotRole,
      artifact.RoleOwnsSourceColumn role sourceColumn ∧
        ∀ otherRole : SlotRole,
          artifact.RoleOwnsSourceColumn otherRole sourceColumn →
            otherRole = role :=
  artifact.sourceColumn_hasUniqueRole sourceColumnLt

example (artifact : SourceCensusArtifact) :
    artifact.eligibleCount =
      SourceSegment.eligibleCountOf artifact.sourceSegments :=
  artifact.eligibleCount_eq_ordinaryRunSubtotal

example (artifact : SourceCensusArtifact) :
    artifact.declaredRoleTotal = artifact.sourceColumnCount :=
  artifact.declaredRoleTotal_eq_sourceColumnCount

example (artifact : SourceCensusArtifact) :
    artifact.sourceColumnCount =
      artifact.eligibleCount + artifact.excludedCount :=
  artifact.sourceColumnCount_eq_eligibleCount_add_excludedCount

example (artifact : SourceCensusArtifact) {budget : Nat}
    (tooSmall : budget < artifact.eligibleCount * 41) :
    ¬ artifact.PerField41CapacityRequirement budget :=
  artifact.budget_below_perField41_is_no_go tooSmall

end NightstreamTests.FPrimeSourceCensus
