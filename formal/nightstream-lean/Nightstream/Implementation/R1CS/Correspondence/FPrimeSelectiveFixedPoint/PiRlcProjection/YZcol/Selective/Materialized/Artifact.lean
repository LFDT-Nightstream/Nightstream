import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Gating

/-!
Decoded artifact boundary for the compact rows in the bounded fixed-point
`y_zcol` projection slice.

Owns: successful fail-closed decoding of every generated row, one stable
decoded-row list, exact alignment with rewrite/retained provenance, and
coefficient-derived selector-gate coverage.

Does not own: satisfaction, selector truth, rewrite correctness,
source-column provenance, protocol authority, or row-removal authority.

Emits constraints: no.

Assurance tier: artifact-checked for this bounded fixture only.

| Correspondence leaf | Mathematical obligation | Authority class |
|---|---|---|
| row decoding | every raw row has one bounded typed interpretation | checked |
| provenance alignment | decoded rows and rewrite records share emitted order | checked |
| gate classification | selector ports determine evaluation/general gating | derived from coefficients |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Artifact

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating

abbrev rawRows : List Materialized.RawRow := Checked.rawRows

def decodedRows : List DecodedRow := rawRows.filterMap decodeRow

def allRowsDecode : Bool :=
  rawRows.all fun row => (decodeRow row).isSome

theorem allRowsDecode_true : allRowsDecode = true := by
  native_decide

theorem rowCount : decodedRows.length = 1254 := by
  native_decide

/-- Successful decoding preserves emitted-row identity and order. -/
theorem decodedEmittedRows_eq_raw :
    (decodedRows.map fun row => row.emittedRow.val) = Census.emittedRows := by
  native_decide

/-- The decoded list retains every emitted-row index exactly once. -/
theorem emittedRowsStrictlyIncreasing :
    YZcol.StrictlyIncreasing (decodedRows.map fun row => row.emittedRow.val) := by
  rw [decodedEmittedRows_eq_raw]
  exact Census.emittedRowsStrictlyIncreasing

theorem decodedFixedShape :
    ∀ row ∈ decodedRows,
      row.rows = Checked.finalRelationRows ∧
      row.columns = Checked.finalRelationColumns ∧
      row.arm = some 2 := by
  native_decide

def decodedRowsAt (emittedRows : List Nat) : List DecodedRow :=
  decodedRows.filter fun row => emittedRows.contains row.emittedRow.val

def rewriteRows : List DecodedRow :=
  decodedRowsAt (Checked.rewriteSteps.map RawRewriteStep.emittedRow)

def retainedRows : List DecodedRow :=
  decodedRowsAt (Checked.retainedSteps.map RawRetainedStep.emittedRow)

theorem rewriteRowCount : rewriteRows.length = 1250 := by
  native_decide

theorem retainedRowCount : retainedRows.length = 4 := by
  native_decide

/-- Provenance and decoded matrix records name the same rewrite rows in the
same strictly increasing emitted-row order. -/
theorem rewriteRowAlignment :
    (rewriteRows.map fun row => row.emittedRow.val) =
      Checked.rewriteSteps.map RawRewriteStep.emittedRow := by
  native_decide

theorem retainedRowAlignment :
    (retainedRows.map fun row => row.emittedRow.val) =
      Checked.retainedSteps.map RawRetainedStep.emittedRow := by
  native_decide

/-- The generated selector coordinate is in range for every decoded row. -/
theorem steadySelectorBound :
    ∀ row ∈ decodedRows, Checked.steadySelectorColumn < row.columns := by
  native_decide

/-- Coefficient-only gate validation at the generated steady-selector
coordinate. Failure is explicit if the coordinate is out of range. -/
def validateSteadyGate (row : DecodedRow) (gate : GatePort) :
    Option (Gating.ValidatedGateRow row) :=
  if bound : Checked.steadySelectorColumn < row.columns then
    Gating.validateGateAt row gate
      ⟨Checked.steadySelectorColumn, bound⟩
  else
    none

def steadyGateValid (row : DecodedRow) (gate : GatePort) : Bool :=
  (validateSteadyGate row gate).isSome

/-- All rewritten rows select the evaluation component by their decoded
selector-port coefficients. No family label participates in this check. -/
theorem rewriteRowsEvaluationGated :
    ∀ row ∈ rewriteRows, steadyGateValid row .evaluation = true := by
  native_decide

/-- The four physically retained final checks select the general component by
their decoded selector-port coefficients. -/
theorem retainedRowsGeneralGated :
    ∀ row ∈ retainedRows, steadyGateValid row .general = true := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Artifact
