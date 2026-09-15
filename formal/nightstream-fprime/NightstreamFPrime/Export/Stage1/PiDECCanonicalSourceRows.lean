import NightstreamFPrime.Export.Stage1.PiDECSourceIndex
import NightstreamFPrime.Export.Stage1.PiDECPiCCSPacketSource
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage

/-!
Read canonical PiCCS source rows through their existing packet schedule.
The exact source-row equality preserves package authority. Other indices
continue through the existing canonical accessor.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceRows

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram

private def findOrdinal (schedule : IndexSchedule) (source : Nat) : Option Nat :=
  match schedule with
  | .rangeList ranges => PiDECSourceIndex.findRange ranges source
  | .indexTable _ => none

private theorem findOrdinal_value (schedule : IndexSchedule) (source ordinal : Nat)
    (found : findOrdinal schedule source = some ordinal) :
    schedule.index? ordinal = some source := by
  cases schedule with
  | rangeList ranges => exact PiDECSourceIndex.findRange_value ranges source found
  | indexTable indices =>
      simp only [findOrdinal] at found
      cases found

/-- Select the existing PiCCS packet from range headers before constructing
its rows. The complete source index space retains the canonical result. -/
def sourceRow (program : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program)
    (source : Nat) : Option R1CS.Row :=
  match findOrdinal PiCCSOrdinaryMatrixProgram.rowSchedule source with
  | none => PerApplicationCanonicalPackage.sourceRow program fits source
  | some ordinal =>
      if ordinal < PiCCSOrdinaryMatrixProgram.rowSchedule.count then
        (PiDECPiCCSPacketSource.row? (PerApplicationFixedPoint.logicalWidth program)
          (PerApplicationFixedPoint.publicFits program) ordinal).map
          (PerApplicationSourceProjection.basePackageRow program)
      else PerApplicationCanonicalPackage.sourceRow program fits source

/-- The optimized source accessor is equal at every index to the canonical
package accessor. No caller supplies a row or a source-agreement premise. -/
theorem sourceRow_value (program : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program) :
    sourceRow program fits = PerApplicationCanonicalPackage.sourceRow program fits := by
  funext source
  unfold sourceRow
  cases found : findOrdinal PiCCSOrdinaryMatrixProgram.rowSchedule source with
  | none => rfl
  | some ordinal =>
      dsimp only
      by_cases bounded : ordinal < PiCCSOrdinaryMatrixProgram.rowSchedule.count
      · rw [if_pos bounded]
        have bound : ordinal < 811669 := by
          simpa only [PiCCSOrdinaryMatrixProgram.rowSchedule_count] using bounded
        have selected := findOrdinal_value PiCCSOrdinaryMatrixProgram.rowSchedule
          source ordinal found
        let relation := PerApplicationFixedPoint.relation program fits
        rw [PiDECPiCCSPacketSource.row?_eq_programRow relation ⟨ordinal, bound⟩,
          Option.map_some, PerApplicationCanonicalPackage.sourceRow_eq_packageSource]
        exact (PerApplicationPackageSourceRows.piCcsPackageSourceRow?_eq_some
          program relation ⟨ordinal, bound⟩ source selected).symm
      · rw [if_neg bounded]

end NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceRows
