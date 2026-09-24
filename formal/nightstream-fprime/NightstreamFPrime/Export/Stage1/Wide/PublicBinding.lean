import NightstreamFPrime.Export.Stage1.Wide.FixedPointSoundness

/-! Bind the candidate's decoded step to its verifier-supplied public input. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PublicBinding

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Spec.HyperNova.Construction2.Paper ProductionRelation Layout.Stage1
open FixedPointSoundness

variable (program : RetainedLayout.Program)
  (assignment : Assignment F (RetainedLayout.logicalWidth program))

/-- The public input is the actual aligned-carrier projection. -/
def publicInput : PaperAlgebra.PublicInput
    (logicalWidth := RetainedLayout.logicalWidth program) (publicFits := FixedPoint.publicFits program) :=
  Phi81Relation.projectPublicInput (Phi81CarrierLayout.extendAssignment 0 assignment)

private theorem public_readback (index : Fin 270) :
    RecursivePublicOutputPlan.publicInput (Stage1Plan.referenceGeometry program)
      (AssignmentPullback.assignment program assignment) index = publicInput program assignment index := by
  have fits : 270 ≤ RetainedLayout.logicalWidth program := by
    rw [RetainedLayout.logicalWidth_eq]
    omega
  let source := RecursivePublicOutputPlan.publicColumn (Stage1Plan.referenceGeometry program) index
  have bound : source.val < 270 := index.isLt
  have mapped := RetainedLayout.publicColumn program source.val bound
  have live : RetainedLayout.Live program source.val :=
    (RetainedLayout.live_iff_mapped program source.val).mpr (by rw [mapped]; rfl)
  let target : Fin (RetainedLayout.logicalWidth program) := ⟨index.val, lt_of_lt_of_le index.isLt fits⟩
  have targetEq : RetainedLayout.column program source live = target :=
    Fin.ext (RetainedLayout.column_of_some program source live source.val mapped)
  change AssignmentPullback.assignment program assignment source = _
  rw [AssignmentPullback.at_live program assignment source live, targetEq]
  have carrierEq : (FullShape (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program)).publicColumn index =
      Phi81CarrierLayout.embedLogical target := Fin.ext rfl
  change assignment target = Phi81CarrierLayout.extendAssignment 0 assignment
    ((FullShape (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program)).publicColumn index)
  rw [carrierEq, Phi81CarrierLayout.extendAssignment_embedLogical]

private theorem digest_readback :
    (List.ofFn fun lane : Fin 4 =>
      (RecursivePublicOutputPlan.outputWordForm (Stage1Plan.referenceGeometry program) lane).eval
        (AssignmentPullback.assignment program assignment)) = digest program assignment := by
  apply congrArg List.ofFn
  funext lane
  exact (PilotDecodedEnvironment.env_location _ _ (.outputDigest lane)).symm

private theorem encoded_public_value {logicalWidth : Nat}
    {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (values : PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := fits))
    (expected : Digest) (equal : values = Lifecycle.encHash (publicFits := fits) expected)
    (index : Fin 270) : values index = Lifecycle.encodedHashCells expected index := by
  rw [equal]
  rfl

private theorem decode_public {logicalWidth : Nat}
    {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (values : PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := fits))
    (expected : Digest) (fixed : expected.length = 4)
    (agrees : ∀ index : Fin 270, values index = Lifecycle.encodedHashCells expected index) :
    Lifecycle.decodeHash values = expected := by
  have same : values = Lifecycle.encHash (publicFits := fits) expected := funext agrees
  rw [same]
  exact Lifecycle.decodeHash_encHash expected fixed

private theorem public_cell (expected : Digest)
    (publicEqual : publicInput program assignment = encHash expected) (index : Fin 270) :
    RecursivePublicOutputPlan.publicInput (Stage1Plan.referenceGeometry program)
      (AssignmentPullback.assignment program assignment) index = Lifecycle.encodedHashCells expected index :=
  (public_readback program assignment index).trans
    (encoded_public_value (logicalWidth := RetainedLayout.logicalWidth program)
      (fits := FixedPoint.publicFits program) (publicInput program assignment) expected publicEqual index)

theorem one (expected : Digest)
    (publicEqual : publicInput program assignment = encHash expected) :
    assignment (Stage1Plan.piRlcInterface program).oneColumn = 1 := by
  have marker := public_cell program assignment expected publicEqual ⟨0, by decide⟩
  have encoded : Lifecycle.encodedHashCells expected ⟨0, by decide⟩ = 1 := rfl
  have referenceOne : AssignmentPullback.assignment program assignment
      (ApplicationRetainedGeometry.oneColumn (Stage1Plan.referenceGeometry program)) = 1 := marker.trans encoded
  rw [AssignmentPullback.at_live program assignment _ (ReadSupport.one program _ rfl)] at referenceOne
  exact referenceOne

theorem output_digest (expected : Digest) (fixed : expected.length = 4)
    (publicEqual : publicInput program assignment = encHash expected)
    (rows : (Stage1Plan.publicOutput program).RowsZero assignment) :
    digest program assignment = expected := by
  have referenceOne := DecodedPrefix.reference_one program assignment (one program assignment expected publicEqual)
  have referenceRows := (AssignmentPullback.rowsZero_iff program assignment _ _).mp rows
  have matching := (RecursivePublicOutputPlan.rowsZero_iff_matches (Stage1Plan.referenceGeometry program)
    (AssignmentPullback.assignment program assignment) referenceOne).mp referenceRows
  have decoded := decode_public (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
    (fits := RecursivePublicOutputPlan.carrierPublicFits (Stage1Plan.referenceGeometry program))
    (RecursivePublicOutputPlan.publicInput (Stage1Plan.referenceGeometry program)
      (AssignmentPullback.assignment program assignment)) expected fixed
    (public_cell program assignment expected publicEqual)
  have words := RecursivePublicOutputPlan.Matches.outputDigest_eq_decodeHash
    (application := program) (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
    (geometry := Stage1Plan.referenceGeometry program)
    (assignment := AssignmentPullback.assignment program assignment) matching
  exact (digest_readback program assignment).symm.trans (words.trans decoded)

/-- The actual carrier public input determines the constant-one coordinate
and the step's public output. No canonical witness premise is required. -/
theorem step (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program)
    (ajtai : AjtaiKey (logicalWidth := RetainedLayout.logicalWidth program)
      (publicFits := FixedPoint.publicFits program))
    (expected : Digest) (fixed : expected.length = 4)
    (publicEqual : publicInput program assignment = encHash expected)
    (rows : (FixedPoint.structuralPlan program compiled fits).RowsZero assignment) :
    Lifecycle.Stage1.Wide.Relation.StepHoldsFor (FixedPoint.relation program compiled fits) ajtai
      (contextKey program assignment) program
      (input program assignment (FixedPoint.relation program compiled fits))
      (output program assignment (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program)) ∧
      digest program assignment = expected := by
  have selected : (Stage1Plan.plan program compiled (FixedPoint.relation program compiled fits) fits.package).RowsZero assignment := by
    rwa [FixedPoint.plan_fixedPoint]
  have parts := (Stage1Plan.rows_iff program compiled (FixedPoint.relation program compiled fits) fits.package assignment).mp selected
  exact ⟨rowsZero_implies_stepHoldsFor program assignment compiled fits ajtai
      (one program assignment expected publicEqual) rows,
    output_digest program assignment expected fixed publicEqual parts.2.2.2.2.2.2⟩

end NightstreamFPrime.Export.Stage1.Wide.PublicBinding
