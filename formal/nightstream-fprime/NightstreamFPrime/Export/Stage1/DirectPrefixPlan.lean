import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation
import NightstreamFPrime.Export.Stage1.PiRLCRetainedPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPreservation
import NightstreamFPrime.Export.Stage1.PilotPoseidonPlan

/-!
Owns the canonical direct-plan prefix through PiRLC. Plan order is pilot
Poseidon2, PiCCS Poseidon2, PiRLC sampler Poseidon2, then PiRLC product and
First54 rows.

This is a phase-local compiler plan. It does not include ordinary rows,
PiDEC, accumulator, application, terminal work, or a final package identity.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPrefixPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def pilotGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    PiRLCPoseidonGeometry.Geometry program logicalWidth :=
  PiCCSPoseidonPlan.pilotGeometry geometry

def prefixGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    PiRLCRetainedGeometry.Geometry program logicalWidth :=
  PiCCSPoseidonPlan.prefixGeometry geometry

def pilotPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PilotPoseidonPlan.plan (pilotGeometry geometry)

def piCcsPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PiCCSPoseidonPlan.plan payloadForms geometry

def samplerPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PiRLCSamplerPoseidonPlan.plan geometry

def piRlcPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PiRLCRetainedPlan.plan (prefixGeometry geometry)

theorem pilotPiCcsRowCount_le
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (pilotPlan geometry).rowCount + (piCcsPlan payloadForms geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  simp [pilotPlan, piCcsPlan]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def pilotPiCcsPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (pilotPlan geometry) (piCcsPlan payloadForms geometry)
    (pilotPiCcsRowCount_le payloadForms geometry)

theorem poseidonRowCount_le
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (pilotPiCcsPlan payloadForms geometry).rowCount + (samplerPlan geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  simp [pilotPiCcsPlan, pilotPlan, piCcsPlan, samplerPlan]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def poseidonPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (pilotPiCcsPlan payloadForms geometry)
    (samplerPlan geometry) (poseidonRowCount_le payloadForms geometry)

theorem totalRowCount_le
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (poseidonPlan payloadForms geometry).rowCount + (piRlcPlan geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  simp [poseidonPlan, pilotPiCcsPlan, pilotPlan, piCcsPlan,
    samplerPlan, piRlcPlan]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (poseidonPlan payloadForms geometry) (piRlcPlan geometry)
    (totalRowCount_le payloadForms geometry)

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth) :
    (plan payloadForms geometry).rowCount = 4964947 := by
  simp [plan, poseidonPlan, pilotPiCcsPlan, pilotPlan, piCcsPlan,
    samplerPlan, piRlcPlan]

theorem rowsZero_iff
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (plan payloadForms geometry).RowsZero assignment ↔
      (pilotPlan geometry).RowsZero assignment ∧
        (piCcsPlan payloadForms geometry).RowsZero assignment ∧
          (samplerPlan geometry).RowsZero assignment ∧
            (piRlcPlan geometry).RowsZero assignment := by
  rw [plan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [poseidonPlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [pilotPiCcsPlan, ProductionRelation.Plan.append_rowsZero_iff]
  constructor
  · rintro ⟨⟨⟨pilot, piCcs⟩, sampler⟩, piRlc⟩
    exact ⟨pilot, piCcs, sampler, piRlc⟩
  · rintro ⟨pilot, piCcs, sampler, piRlc⟩
    exact ⟨⟨⟨pilot, piCcs⟩, sampler⟩, piRlc⟩

structure Semantics {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  pilot : PilotPoseidonPlan.Semantics (pilotGeometry geometry) assignment
  piCcsEncoding : PiCCSPoseidonPreservation.Encoding payloadForms geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment
      program base groupValue products)
  piCcs : PiCCSPoseidonPreservation.CanonicalSemantics geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment
      program base groupValue products)
  samplerEncoding : PiRLCSamplerPoseidonPreservation.Encoding geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment
      program base groupValue products)
  sampler : PiRLCSamplerPoseidonPreservation.CanonicalSemantics geometry assignment
  piRlc : PiRLCRetainedPlan.Semantics program base

structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  retained : PiRLCRetainedPreservation.Encodes
    (prefixGeometry geometry) assignment base groupValue products
  pilotPriorInput :
    (PiRLCPoseidonGeometry.priorInputBlock program).EncodesAt
      (PiRLCPoseidonGeometry.priorInputStart program)
      (PiRLCPoseidonGeometry.priorInputFits (pilotGeometry geometry)) assignment
      (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products)
  pilotOutputInput :
    (PiRLCPoseidonGeometry.outputInputBlock program).EncodesAt
      (PiRLCPoseidonGeometry.outputInputStart program)
      (PiRLCPoseidonGeometry.outputInputFits (pilotGeometry geometry)) assignment
      (PiRLCRetainedPreservation.sourceAssignment
        program base groupValue products)
  payload : ∀ index, (payloadForms index).eval assignment =
    PiCCSActionPayloadBlock.payloadValue program
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products)
      index

theorem rowsZero_implies_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (payloadForms : PiCCSPoseidonPlan.Payload logicalWidth)
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (PiCCSPoseidonPlan.oneColumn geometry) = 1)
    (encodes : Encodes payloadForms geometry assignment base groupValue products)
    (rowsZero : (plan payloadForms geometry).RowsZero assignment) :
    Semantics payloadForms geometry assignment base groupValue products := by
  have children := (rowsZero_iff payloadForms geometry assignment).mp rowsZero
  let prefixAssignment := PiRLCRetainedPreservation.sourceAssignment
    program base groupValue products
  have piCcsEncoding : PiCCSPoseidonPreservation.Encoding payloadForms geometry assignment
      prefixAssignment :=
    PiCCSPoseidonPreservation.encodingOfRetained payloadForms geometry assignment
      prefixAssignment
      (PiRLCRetainedGeometry.laterPoseidonFits (prefixGeometry geometry))
      encodes.retained.laterPoseidon encodes.payload
  have samplerEncoding : PiRLCSamplerPoseidonPreservation.Encoding geometry
      assignment prefixAssignment :=
    PiRLCSamplerPoseidonPreservation.encodingOfRetained geometry assignment
      prefixAssignment
      (PiRLCRetainedGeometry.laterPoseidonFits (prefixGeometry geometry))
      encodes.retained.laterPoseidon
  refine ⟨?_, piCcsEncoding, ?_, samplerEncoding, ?_, ?_⟩
  · exact PilotPoseidonPlan.rowsZero_implies_semantics
      (pilotGeometry geometry) assignment one children.1
  · exact PiCCSPoseidonPreservation.rowsZero_implies_canonicalSemantics
      payloadForms geometry assignment prefixAssignment piCcsEncoding one children.2.1
  · exact PiRLCSamplerPoseidonPreservation.rowsZero_implies_canonicalSemantics
      geometry assignment one children.2.2.1
  · exact PiRLCRetainedPlan.rowsZero_implies_semantics
      (prefixGeometry geometry) assignment base groupValue products one
        encodes.retained children.2.2.2

end NightstreamFPrime.Export.Stage1.DirectPrefixPlan
