import NightstreamFPrime.Export.PilotData
import NightstreamFPrime.Layout.PilotSpartan
import NightstreamFPrime.Layout.Poseidon2

/-!
Owns the proofs that connect the executable pilot package in `PilotData` to
the production lifecycle layout. It owns no second package, row list, or
schedule.
-/

namespace NightstreamFPrime.Export.Pilot

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package

theorem canonicalState_affine :
    Poseidon2.StateAffine PilotData.canonicalState := by
  intro lane
  exact R1CS.isAffine_var lane.val

theorem canonicalRows_length :
    (PilotData.canonicalRows ()).length = 592 := by
  calc
    (PilotData.canonicalRows ()).length =
        R1CS.totalRowCount (PilotData.canonicalConstraints ()) := by
      exact R1CS.lowerConstraints_rows_length
        (PilotData.canonicalConstraints ()) 600
    _ = (PilotData.canonicalRecipes ()).length := by
      exact R1CS.recipeConstraints_totalRowCount 8
        (PilotData.canonicalRecipes ())
        (Poseidon2.compile_schedule_direct 8 PilotData.canonicalState
          canonicalState_affine)
    _ = 592 := by
      exact Permutation.compile_schedule_recipe_count 8
        PilotData.canonicalState

theorem templateRowsFrom_length (output : Nat) (rows : List R1CS.Row) :
    (PilotData.templateRowsFrom output rows).length = rows.length := by
  induction rows generalizing output with
  | nil => rfl
  | cons row rest ih =>
      simp [PilotData.templateRowsFrom, ih]

theorem templateRows_length :
    (PilotData.templateRows ()).length = 592 := by
  rw [PilotData.templateRows, templateRowsFrom_length,
    canonicalRows_length]

@[simp] theorem digestRows_length (chain : HashChain) :
    (PilotData.digestRows chain).length = 4 := by
  simp [PilotData.digestRows]

theorem tailBindingRows_length :
    PilotData.tailBindingRows.length = 49 := by
  simp [PilotData.tailBindingRows]

theorem bindingRows_length :
    (PilotData.bindingRows ()).length = 50 := by
  simp [PilotData.bindingRows, tailBindingRows_length]

theorem assertionRows_length :
    (PilotData.assertionRows ()).length = 58 := by
  simp [PilotData.assertionRows, bindingRows_length]

theorem circuitPackage_decode_encode :
    CircuitPackage.format.decode
      (CircuitPackage.format.encode (PilotData.circuitPackage ())) =
        .ok (PilotData.circuitPackage ()) :=
  Package.decode_encode (PilotData.circuitPackage ())

theorem circuitPackage_template_rows :
    (PilotData.circuitPackage ()).permutation.rows.length = 592 :=
  templateRows_length

theorem circuitPackage_assertion_rows :
    (PilotData.circuitPackage ()).assertionRows.length = 58 :=
  assertionRows_length

theorem circuitPackage_row_coverage :
    PilotData.priorChain.witnessLength +
      PilotData.outputChain.witnessLength +
      (PilotData.circuitPackage ()).assertionRows.length =
        (PilotData.circuitPackage ()).layout.rowCount := by
  rw [circuitPackage_assertion_rows]
  rfl

/-- The executable package uses the exact proved pilot row and Spartan column
counts. No Rust-selected layout value enters this statement. -/
theorem circuitPackage_layout_matches :
    let layout := (PilotData.circuitPackage ()).layout
    layout.rowCount =
        Layout.Pilot.physicalRowCount PilotProduction.interface
          PilotProduction.witnessOffset ∧
      layout.privateColumnCount = PilotSpartan.privateColumnCount ∧
      layout.constantColumn = PilotSpartan.constantColumn ∧
      layout.publicColumnCount = PilotSpartan.publicColumnCount ∧
      layout.totalColumnCount = PilotSpartan.spartanColumnCount := by
  dsimp [PilotData.circuitPackage, PilotData.physicalLayout]
  exact ⟨PilotProduction.physicalRowCount_eq.symm, rfl, rfl, rfl, rfl⟩

theorem artifact_identifier :
    (PilotData.artifact ()).claimedIdentifier =
      (PilotData.relationIdentifier ()).map (fun word => word.val) := by
  rfl

end NightstreamFPrime.Export.Pilot
