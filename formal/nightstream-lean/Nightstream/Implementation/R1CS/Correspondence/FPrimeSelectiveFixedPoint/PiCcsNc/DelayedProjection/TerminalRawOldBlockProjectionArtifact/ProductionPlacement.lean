import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.Contract
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.ColumnMap
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.ProductionEmitterLayout

/-!
Fixed runtime placement for the generated production raw-old-block rows.

This leaf checks the concrete Rust-emitted column layout, Ajtai witness join,
profile identity, and recursive selector.  It also proves that adding the
generated row base is a bijection from the compact local row index onto the
exact half-open physical interval.  No assignment values or semantic
acceptance propositions occur here.

Owns: the fixed emitter-layout validity certificate, equality with the Ajtai
child witness bases, production profile and selector pins, and exact physical
row-interval ownership.

Does not own: witness values, row satisfaction, commitment binding, projection
semantics, terminal CE, transcript soundness, costs, or row-removal authority.

Emits constraints: no; owns the generated physical placement certificate.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.execution.placement.columns` | the concrete emitter column map is structurally valid and uses the ordered Ajtai child bases | checked artifact |
| `f_prime.pi_ccs_nc.delayed.execution.placement.profile` | the fixed profile, pending-projection join, and recursive selector identities match production | checked artifact |
| `f_prime.pi_ccs_nc.delayed.execution.placement.rows` | local row addition bijects onto the exact generated half-open physical interval | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler

/-- The active Rust emitter layout satisfies the fail-closed structural
column-map check.  The certificate contains 19 old-block pairs, 54 parent
pairs, 14 child bases, and three derived-column interval starts. -/
theorem productionEmitterLayout_checked :
    productionEmitterLayoutChecked = true := by
  decide

/-- Fixed valid emitter layout; never a theorem-caller premise. -/
def productionEmitterValid :
    EmitterLayoutValid productionEmitterLayout :=
  ⟨productionEmitterLayout_checked⟩

/-- The checked production inverse recovers every canonical source column,
including the 270 final-scale columns introduced by the optimized emitter. -/
theorem productionColumnRoundTrip
    (column : Nat) (columnInRange : column < canonicalColumnCount) :
    emitterColumnInverse productionEmitterLayout
        (emitterColumnMap productionEmitterLayout column) =
      some column :=
  productionEmitterValid.columnRoundTrip column columnInRange

/-- The projection rows read the same fourteen ordered raw witness bases as
the terminal Ajtai opening. -/
theorem productionAjtaiJoin_checked :
    productionAjtaiJoinChecked = true := by
  decide

/-- The emitted profile, pending-projection join, and recursive-arm tag are
the fixed production identities. -/
theorem productionProfile_checked :
    productionProfileChecked = true := by
  decide

/-- The active encoder selected the recursive arm at the generated selector
columns. -/
theorem productionSelector_checked :
    productionSelectorChecked = true := by
  decide

@[simp] theorem productionRowFirst :
    productionEmitterLayout.rowFirst = 22834865 := by
  rfl

@[simp] theorem productionRowStop :
    productionEmitterLayout.rowStop = 47020034 := by
  rfl

/-- Every local compiler row is placed inside the exact generated physical
interval. -/
theorem productionPhysicalRow_mem
    (row : Fin totalRows) :
    productionEmitterLayout.rowFirst <=
        physicalRow productionEmitterLayout row /\
      physicalRow productionEmitterLayout row <
        productionEmitterLayout.rowStop := by
  constructor
  · simp [physicalRow]
  · simp only [physicalRow, productionRowFirst, productionRowStop,
      totalRows]
    omega

/-- Distinct local rows have distinct generated physical row numbers. -/
theorem productionPhysicalRow_injective :
    Function.Injective (physicalRow productionEmitterLayout) := by
  intro left right equal
  apply Fin.ext
  simp only [physicalRow] at equal
  omega

/-- Every physical row in the generated interval has exactly one compact
local owner. -/
theorem productionPhysicalRow_unique
    (physical : Nat)
    (lower : productionEmitterLayout.rowFirst <= physical)
    (upper : physical < productionEmitterLayout.rowStop) :
    ∃ row : Fin totalRows,
      physicalRow productionEmitterLayout row = physical /\
      ∀ other : Fin totalRows,
        physicalRow productionEmitterLayout other = physical ->
          other = row := by
  let row : Fin totalRows :=
    ⟨physical - productionEmitterLayout.rowFirst, by
      simp only [productionRowFirst, productionRowStop, totalRows] at lower upper ⊢
      omega⟩
  refine ⟨row, ?_, ?_⟩
  · simp only [row, physicalRow]
    omega
  · intro other otherEqual
    apply productionPhysicalRow_injective
    rw [otherEqual]
    simp only [row, physicalRow]
    omega

/-- Every conceptual tensor, coordinate-product, final-scale, or terminal
owner is placed inside the generated physical interval. -/
theorem productionConceptualPhysicalRow_mem
    (index : RowIndex productionFactoredLayout) :
    productionEmitterLayout.rowFirst <=
        physicalRow productionEmitterLayout
          (productionPhysicalIndex index) /\
      physicalRow productionEmitterLayout
          (productionPhysicalIndex index) <
        productionEmitterLayout.rowStop :=
  productionPhysicalRow_mem (productionPhysicalIndex index)

/-- The physical placement remains injective after composing the generated
four-family compiler owner with its exact local row permutation. -/
theorem productionConceptualPhysicalRow_injective :
    Function.Injective (fun index : RowIndex productionFactoredLayout =>
      physicalRow productionEmitterLayout
        (productionPhysicalIndex index)) :=
  productionPhysicalRow_injective.comp productionPhysicalIndex_injective

/-- Every physical row in the generated interval has exactly one conceptual
owner among tensor, coordinate-product, final-scale, and terminal rows. -/
theorem productionConceptualPhysicalRow_unique
    (physical : Nat)
    (lower : productionEmitterLayout.rowFirst <= physical)
    (upper : physical < productionEmitterLayout.rowStop) :
    exists index : RowIndex productionFactoredLayout,
      physicalRow productionEmitterLayout
          (productionPhysicalIndex index) = physical /\
      forall other : RowIndex productionFactoredLayout,
        physicalRow productionEmitterLayout
            (productionPhysicalIndex other) = physical ->
          other = index := by
  obtain ⟨row, rowEqual, rowUnique⟩ :=
    productionPhysicalRow_unique physical lower upper
  obtain ⟨index, indexEqual⟩ := productionPhysicalIndex_surjective row
  refine ⟨index, ?_, ?_⟩
  · rw [indexEqual]
    exact rowEqual
  · intro other otherEqual
    apply productionPhysicalIndex_injective
    exact (rowUnique (productionPhysicalIndex other) otherEqual).trans
      indexEqual.symm

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
