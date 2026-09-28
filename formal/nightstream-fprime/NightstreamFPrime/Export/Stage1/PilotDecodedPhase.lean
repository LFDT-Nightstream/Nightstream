import NightstreamFPrime.Export.Stage1.PilotDecodedEnvironment
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint

/-!
Owns the arbitrary-assignment connection from the selected pilot rows to
the existing two-child pilot specification. The decoder preserves the actual
hash preimages and gives each arithmetic location its compiled value.
-/

namespace NightstreamFPrime.Export.Stage1.PilotDecodedPhase

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- The direct hash, arithmetic, and digest-binding rows imply the complete
pilot specification for arbitrary assignments with the enforced constant. -/
theorem rowsZero_implies_specHolds
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PilotOrdinaryDirectPlan.oneColumn geometry) = 1)
    (hashRows : (PilotPoseidonPlan.plan (PilotDirectSemantics.poseidonGeometry geometry)
      ).RowsZero assignment)
    (ordinaryRows : (PilotOrdinaryDirectPlan.plan geometry).RowsZero assignment)
    (bindingRows : (PilotDigestBindingPlan.plan geometry).RowsZero assignment) :
    Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset
      (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment)) := by
  have poseidonOne : assignment (PiRLCPoseidonGeometry.oneColumn
      (PilotDirectSemantics.poseidonGeometry geometry)) = 1 := by
    have same : PiRLCPoseidonGeometry.oneColumn
        (PilotDirectSemantics.poseidonGeometry geometry) =
          PilotOrdinaryDirectPlan.oneColumn geometry := by
      apply Fin.ext
      rfl
    rw [same]
    exact one
  have hashes := PilotPoseidonPreservation.semantics_imply_hashFacts
    (PilotDirectSemantics.poseidonGeometry geometry) assignment
    (PilotDecodedEnvironment.env geometry assignment) poseidonOne
    (PilotDecodedEnvironment.priorInputForm_eval geometry assignment)
    (PilotDecodedEnvironment.outputInputForm_eval geometry assignment)
    (PilotPoseidonPlan.rowsZero_implies_semantics
      (PilotDirectSemantics.poseidonGeometry geometry) assignment poseidonOne hashRows)
  exact PilotDirectSemantics.implies_spec geometry assignment
    (PilotDecodedEnvironment.env geometry assignment) hashes
    (PilotDecodedEnvironment.rowsZero_implies_sourceRows geometry assignment one ordinaryRows)
    ((PilotDigestBindingPlan.rowsZero_iff_matches geometry assignment one).mp bindingRows)
    (fun lane => (PilotDecodedEnvironment.env_location geometry assignment
      (.priorDigest lane)).symm)
    (fun lane => (PilotDecodedEnvironment.env_location geometry assignment
      (.outputState lane)).symm)

/-- The selected complete Stage 1 rows force the actual prior and output
hash contracts. No raw packet, encoding, or representation is supplied. -/
theorem selectedRowsZero_implies_specHolds
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (one : assignment (ApplicationRetainedGeometry.oneColumn
      (PerApplicationFixedPoint.geometry application)) = 1)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
      assignment) :
    Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset
      (PilotSpartan.pullback (PilotDecodedEnvironment.env
        (DirectApplicationPrefixPlan.pilotOrdinaryGeometry
          (PerApplicationFixedPoint.geometry application)) assignment)) := by
  let relation := PerApplicationFixedPoint.relation application fits
  let geometry := PerApplicationFixedPoint.geometry application
  have selected : (DirectApplicationPrefixPlan.plan relation fits.package geometry
      ).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact accepted
  have applicationRows := (DirectApplicationPrefixPlan.rowsZero_iff relation
    fits.package geometry assignment).mp selected
  have prefixRows := (DirectPiRLCSamplerCompletePrefixPlan.rowsZero_iff relation
    (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment).mp
      applicationRows.1.1.1
  have samplerPrefixRows := prefixRows.1
  simp only [DirectPiRLCSamplerCompletePrefixPlan.samplerPrefixPlan,
    DirectPiDECPrefixPlan.samplerPrefixPlan, DirectPiDECPrefixPlan.piCcsCompletePlan,
    DirectPiDECPrefixPlan.pilotBindingPrefixPlan,
    DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan,
    DirectPiDECPrefixPlan.piCcsCorePlan, DirectPiDECPrefixPlan.piCcsPoseidonPrefix,
    ProductionRelation.Plan.append_rowsZero_iff] at samplerPrefixRows
  rcases samplerPrefixRows with
    ⟨⟨⟨⟨⟨⟨hashRows, _transcript⟩, _piCcsOrdinary⟩, ordinaryRows⟩,
      bindingRows⟩, _endpoints⟩, _sampler⟩
  exact rowsZero_implies_specHolds
    (DirectApplicationPrefixPlan.pilotOrdinaryGeometry geometry) assignment one
    hashRows ordinaryRows bindingRows

end NightstreamFPrime.Export.Stage1.PilotDecodedPhase
