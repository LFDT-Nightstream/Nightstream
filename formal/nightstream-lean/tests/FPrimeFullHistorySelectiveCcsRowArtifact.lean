import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCcsSelectorDomainRow

/-!
External checks for the one-row Rust-to-Lean selective-CCS boundary.

| Case | Expected result |
|---|---|
| Rust-rendered selector-domain fixture | decodes and has Boolean residual |
| Boolean coefficients with false family/arm metadata | decodes and validates from coefficients |
| arbitrary assignment | decoded residual has independent Boolean semantics |
| modular coefficient alias | rejected |
| duplicate column | rejected |
| wrong port count | rejected |
-/

namespace Tests.FPrimeFullHistorySelectiveCcsRowArtifact

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean

private def compilerRawRow : RawRow :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCcsSelectorDomainRow.rawRow

private def decodedCompilerRow : DecodedRow :=
  (decodeRow compilerRawRow).get (by decide)

private def validatedCompilerRow : ValidatedBooleanRow decodedCompilerRow :=
  (validateBooleanAt decodedCompilerRow
      (⟨54, by decide⟩ : Fin decodedCompilerRow.columns)
      (⟨0, by decide⟩ : Fin decodedCompilerRow.columns)).get (by decide)

theorem compiler_selector_row_residual
    (assignment : Fin decodedCompilerRow.columns → F) :
    residual decodedCompilerRow assignment =
      Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components.booleanResidual
        (Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.booleanPoint
          (assignment validatedCompilerRow.selectorColumn)
          (assignment validatedCompilerRow.bitColumn)) := by
  exact residual_eq_booleanResidual decodedCompilerRow validatedCompilerRow assignment

private def emptyPort : RawPort := ⟨[]⟩
private def bitPort : RawPort := ⟨[⟨2, 1⟩]⟩
private def selectorPort : RawPort := ⟨[⟨0, 1⟩]⟩

private def rawBooleanRow : RawRow where
  schemaVersion := 1
  rows := 1
  columns := 3
  emittedRow := 0
  runIndex := 7
  -- Intentionally false: classification must ignore provenance labels.
  family := .poseidon2
  arm := some 9
  ports := [bitPort, selectorPort] ++ List.replicate 11 emptyPort

example : (decodeRow rawBooleanRow).isSome = true := by
  decide

private def decodedBooleanRow : DecodedRow :=
  (decodeRow rawBooleanRow).get (by decide)

private def validatedBooleanRow : ValidatedBooleanRow decodedBooleanRow :=
  (validateBooleanAt decodedBooleanRow
      (⟨2, by decide⟩ : Fin decodedBooleanRow.columns)
      (⟨0, by decide⟩ : Fin decodedBooleanRow.columns)).get (by decide)

example (assignment : Fin decodedBooleanRow.columns → F) :
    residual decodedBooleanRow assignment =
      Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components.booleanResidual
        (Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.booleanPoint
          (assignment validatedBooleanRow.selectorColumn)
          (assignment validatedBooleanRow.bitColumn)) := by
  exact residual_eq_booleanResidual decodedBooleanRow validatedBooleanRow assignment

private def noncanonicalRow : RawRow :=
  { rawBooleanRow with
    ports :=
      [⟨[⟨2, goldilocksModulus⟩]⟩, selectorPort] ++
        List.replicate 11 emptyPort }

example : decodeRow noncanonicalRow = none := by
  decide

private def duplicateColumnRow : RawRow :=
  { rawBooleanRow with
    ports :=
      [⟨[⟨2, 1⟩, ⟨2, 1⟩]⟩, selectorPort] ++
        List.replicate 11 emptyPort }

example : decodeRow duplicateColumnRow = none := by
  decide

private def wrongPortCountRow : RawRow :=
  { rawBooleanRow with ports := List.replicate 12 emptyPort }

example : decodeRow wrongPortCountRow = none := by
  decide

end Tests.FPrimeFullHistorySelectiveCcsRowArtifact
