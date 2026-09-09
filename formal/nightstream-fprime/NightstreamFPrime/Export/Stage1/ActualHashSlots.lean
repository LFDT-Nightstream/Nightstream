import NightstreamFPrime.Export.Stage1.ActualNextPreimage
import NightstreamFPrime.Export.Stage1.PilotDecodedPhase

/-!
Owns the two Construction 2 hash equations on arbitrary accepted assignments.
The next preimage uses the decoded prior counter plus one. Its serialization
is derived from the actual hashed output words, without a non-wrap premise.
The final typed NIFS and verifier-context connections remain separate.
-/

namespace NightstreamFPrime.Export.Stage1.ActualHashSlots

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def publicInput (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Fin PriorStateHash.publicWidth → F :=
  fun column => (PilotProduction.priorInterface.publicInput
    PilotProduction.witnessOffset column).eval
      (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment))

def outputDigest (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Digest :=
  List.ofFn fun lane => (PilotProduction.outputInterface.digest
    (Lifecycle.Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset)
    lane).eval
      (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment))

/-- The digest read by the hash slot and the final public pin rows uses the
same four owned forms in the arbitrary assignment. -/
theorem outputDigest_eq_forms
    (geometry : ApplicationRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    outputDigest (RecursivePublicOutputPlan.pilotOrdinaryGeometry geometry) assignment =
      List.ofFn (fun word : Fin 4 =>
        (RecursivePublicOutputPlan.outputWordForm geometry word).eval assignment) := by
  unfold outputDigest
  apply congrArg List.ofFn
  funext word
  exact PilotDecodedEnvironment.env_location
    (RecursivePublicOutputPlan.pilotOrdinaryGeometry geometry) assignment (.outputDigest word)

/-- The next hash retains the prior key and initial state, increments the
prior natural counter, and uses the actual next current/running state. -/
def nextPreimage (relationLogicalWidth : Nat)
    (relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth)
    (prior next : Nat → F) : Lifecycle.HashPreimage
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) :=
  { StateDecoder.preimage relationLogicalWidth relationPublicFits next with
    verifierKeys := fun _ => StateDecoder.keyDigest prior
    iteration := StateDecoder.iteration prior + 1
    z0 := StateDecoder.initialState prior }

/-- The constrained next words serialize the exact incremented preimage.
Only the counter's field encoding is compared; natural-number wrap is not
assumed or needed by the hash equation. -/
theorem rowsZero_implies_nextPreimageSerialization
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (ordinaryRows : (PiCCSOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (nextRows : (NextPreimageDirectPlan.plan geometry).RowsZero assignment) :
    serializePreimage (publicFits := relationPublicFits)
        (nextPreimage relationLogicalWidth relationPublicFits
          (ActualPreimageFraming.priorState geometry assignment)
          (ActualPreimageFraming.outputState geometry assignment)) =
      serializePreimage (publicFits := relationPublicFits)
        (StateDecoder.preimage relationLogicalWidth relationPublicFits
          (ActualPreimageFraming.outputState geometry assignment)) := by
  have context := ActualPreimageFraming.rowsZero_implies_contextKeys relation
    geometry assignment one ordinaryRows
  have headers := ActualNextPreimage.rowsZero_implies_decodedHeaders
    geometry assignment one nextRows
  simp only [nextPreimage, StateDecoder.preimage, serializePreimage]
  rw [← context, ← headers.2, headers.1]
  rw [StateDecoder.iteration, StateDecoder.natWord_val]

/-- The opaque pilot contract and accepted framing/next rows imply both
typed hash slots. No representation fact is supplied by the caller. -/
theorem specAndRows_imply_hashSlots
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PilotOrdinaryDirectPlan.oneColumn geometry) = 1)
    (pilot : Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset
      (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment)))
    (ordinaryRows : (PiCCSOrdinaryDirectPlan.plan relation
      (PilotOrdinaryDirectPlan.piCcsGeometry geometry)).RowsZero assignment)
    (nextRows : (NextPreimageDirectPlan.plan
      (PilotOrdinaryDirectPlan.piCcsGeometry geometry)).RowsZero assignment) :
    publicInput geometry assignment = encHash (publicFits := relationPublicFits)
        (stateHash (publicFits := relationPublicFits)
          (StateDecoder.preimage relationLogicalWidth relationPublicFits
            (ActualPreimageFraming.priorState
              (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment))) ∧
      outputDigest geometry assignment = stateHash (publicFits := relationPublicFits)
        (nextPreimage relationLogicalWidth relationPublicFits
          (ActualPreimageFraming.priorState
            (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment)
          (ActualPreimageFraming.outputState
            (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment)) := by
  have represents := ActualPreimageFraming.rowsZero_implies_preimageRepresentations
    relation geometry assignment one ordinaryRows
  constructor
  · exact PriorStateHash.builder_implies_priorPublicInput
      PilotProduction.priorInterface PilotProduction.witnessOffset
      (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment))
      _ (publicInput geometry assignment) pilot.1 represents.1 (fun _ => rfl)
  · have output := OutputHash.builder_implies_digest
      PilotProduction.outputInterface
      (Lifecycle.Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset)
      (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment))
      _ (outputDigest geometry assignment) pilot.2 represents.2 rfl
    rw [stateHash, rowsZero_implies_nextPreimageSerialization relation
      (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment one ordinaryRows nextRows]
    exact output

/-- The selected complete Stage 1 rows force both hash equations on values
decoded from that assignment. The public boundary must enforce the one cell. -/
theorem selectedRowsZero_implies_hashSlots
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (one : assignment (ApplicationRetainedGeometry.oneColumn
      (PerApplicationFixedPoint.geometry application)) = 1)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
      assignment) :
    let geometry := PerApplicationFixedPoint.geometry application
    publicInput (DirectApplicationPrefixPlan.pilotOrdinaryGeometry geometry) assignment =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application)
          (stateHash (publicFits := PerApplicationFixedPoint.publicFits application)
            (StateDecoder.preimage (PerApplicationFixedPoint.logicalWidth application)
              (PerApplicationFixedPoint.publicFits application)
              (ActualPreimageFraming.priorState
                (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment))) ∧
      outputDigest (DirectApplicationPrefixPlan.pilotOrdinaryGeometry geometry) assignment =
        stateHash (publicFits := PerApplicationFixedPoint.publicFits application)
          (nextPreimage (PerApplicationFixedPoint.logicalWidth application)
            (PerApplicationFixedPoint.publicFits application)
            (ActualPreimageFraming.priorState
              (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
            (ActualPreimageFraming.outputState
              (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)) := by
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
    ⟨⟨⟨⟨⟨⟨_hashRows, _transcript⟩, ordinaryRows⟩, _pilotOrdinary⟩,
      _binding⟩, _endpoints⟩, _sampler⟩
  exact specAndRows_imply_hashSlots relation
    (DirectApplicationPrefixPlan.pilotOrdinaryGeometry geometry) assignment one
    (PilotDecodedPhase.selectedRowsZero_implies_specHolds
      application fits assignment one accepted)
    ordinaryRows applicationRows.1.2

/-- Accepted selected rows and their actual CCS public projection force the
claimed digest to equal the next-state hash. The one cell is derived from the
public marker, rather than supplied as an additional value premise. -/
theorem selectedRowsAndPublic_imply_outputHash
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest) (fixed : digest.length = 4)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
      assignment) :
    digest = stateHash (publicFits := PerApplicationFixedPoint.publicFits application)
      (nextPreimage (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (ActualPreimageFraming.priorState
          (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
            (PerApplicationFixedPoint.geometry application)) assignment)
        (ActualPreimageFraming.outputState
          (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
            (PerApplicationFixedPoint.geometry application)) assignment)) := by
  let geometry := PerApplicationFixedPoint.geometry application
  let relation := PerApplicationFixedPoint.relation application fits
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one
    geometry assignment digest publicBound
  have selected : (DirectApplicationPrefixPlan.plan relation fits.package geometry
      ).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact accepted
  have rows := (DirectApplicationPrefixPlan.rowsZero_iff relation
    fits.package geometry assignment).mp selected
  have matching := (RecursivePublicOutputPlan.rowsZero_iff_matches
    geometry assignment one).mp rows.2
  have digestEq : outputDigest (DirectApplicationPrefixPlan.pilotOrdinaryGeometry geometry)
      assignment = digest :=
    (outputDigest_eq_forms geometry assignment).trans
      (matching.outputDigest_eq_of_encHash digest fixed publicBound)
  exact digestEq.symm.trans
    (selectedRowsZero_implies_hashSlots application fits assignment one accepted).2

end NightstreamFPrime.Export.Stage1.ActualHashSlots
