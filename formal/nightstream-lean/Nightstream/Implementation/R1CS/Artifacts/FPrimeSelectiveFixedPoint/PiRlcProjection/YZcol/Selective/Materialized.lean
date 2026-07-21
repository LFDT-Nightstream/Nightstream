import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Census
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SourceColumns

/-!
Stable artifact surface for the exact compact selective rows of the bounded
fixed-point `y_zcol` projection slice.

Owns: one handwritten import boundary for all generated compact rows and their
exact fragment-ownership census.

Does not own: decoding, row satisfaction, semantic validation of the exported
source-column provenance, selector truth, Rust-wide conformance, or
row-removal authority.

Emits constraints: no.

Assurance tier: artifact-checked for this bounded fixture only.

| Artifact leaf | Mathematical obligation | Authority class |
|---|---|---|
| compact rows | exact final A/B/C port payload | checked |
| source provenance | exact source closure, slots, definitions, and rewrites | computed |
| facade | generated payload is exposed through one stable artifact boundary | direct dataflow |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Checked

abbrev rawRows : List Materialized.RawRow :=
  Generated.SelectiveMatrixRows.rawRows

abbrev finalRelationRows : Nat :=
  Generated.SelectiveMatrixRows.finalRelationRows

abbrev finalRelationColumns : Nat :=
  Generated.SelectiveMatrixRows.finalRelationColumns

abbrev constantOneColumn : Nat :=
  Generated.SelectiveMatrixRows.constantOneColumn

abbrev steadySelectorColumn : Nat :=
  Generated.SelectiveMatrixRows.steadySelectorColumn

abbrev rewriteSteps : List Materialized.RawRewriteStep :=
  Generated.SourceColumns.rewriteSteps

abbrev retainedSteps : List Materialized.RawRetainedStep :=
  Generated.SourceColumns.retainedSteps

abbrev sourceArtifact := Generated.Metadata.artifact

abbrev sourceArm : Nat := Generated.SourceColumns.sourceArm

abbrev sourceColumns : List Nat := Generated.SourceColumns.sourceColumns

abbrev traceEliminatedColumns : List Nat :=
  Generated.SourceColumns.traceEliminatedColumns

abbrev retainedSlots : List Materialized.RawSourceSlot :=
  Generated.SourceColumns.retainedSlots

abbrev linearDefinitions : List Materialized.RawSourceDefinition :=
  Generated.SourceColumns.linearDefinitions

abbrev derivedProductSums : List Materialized.RawDerivedProductSum :=
  Generated.SourceColumns.derivedProductSums

theorem rowCount : rawRows.length = 1254 := Census.rowCount

theorem relationRowCountAgreement :
    finalRelationRows = Generated.SelectiveRows.artifact.finalRelationRowCount :=
  Census.relationRowCountAgreement

theorem distinguishedColumns :
    constantOneColumn = 0 ∧ steadySelectorColumn = 272 :=
  Census.distinguishedColumns

theorem fixedShape :
    ∀ row ∈ rawRows,
      row.schemaVersion = 1 ∧
      row.rows = finalRelationRows ∧
      row.columns = finalRelationColumns ∧
      row.emittedRow < row.rows ∧
      row.arm = some 2 ∧
      row.ports.length = 13 :=
  Census.fixedShape

theorem exactEmittedRows :
    Census.emittedRows.mergeSort
        (fun left right => decide (left ≤ right)) =
      Census.expectedEmittedRows.mergeSort
        (fun left right => decide (left ≤ right)) :=
  Census.exactEmittedRows

theorem uniqueOwner :
    ∀ row ∈ rawRows, (Census.owners row.emittedRow).length = 1 :=
  Census.uniqueOwner

theorem uniqueOwnerAndFamily :
    ∀ row ∈ rawRows, Census.ownerAndFamilyAgree row = true :=
  Census.uniqueOwnerAndFamily

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Checked
