import Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalRows

/-!
Contract: separate the static Split-NC row shape from the dynamic public
claims carried by one NIFS occurrence.

The emitted FE program depends on the verifier-owned constraint polynomial;
the prior point and carried `y_ring` claims affect values but not row shape.
`layoutInput` supplies zero placeholders for those dynamic families.
`withDynamicClaims` restores the exact decoded families without changing the
constraint polynomial, FE degree, transcript columns, or emitted rows.

This is a type-index transport only.  It emits no rows and does not assert
that an arbitrary proof uses the selected constraint polynomial.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 800000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcStaticInput

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- Additive identity of the concrete quadratic extension. -/
def zeroK : K := { c0 := 0, c1 := 0 }

/-- Canonical zero point used only to inhabit the row-layout index. -/
def zeroPoint (count : Nat) : CubePoint K count where
  coordinates := List.replicate count zeroK
  dimension := by simp

/-- Static FE layout input.  Only `constraintPolynomial` is row authority;
the other two fields are placeholders and are never decoded as claims. -/
def layoutInput
    {shape : SemanticShape}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount) :
    PublicInput shape where
  constraintPolynomial := constraintPolynomial
  priorPoint := zeroPoint shape.rowVariables
  claimedYRing := fun _ _ _ => zeroK

/-- Restore the dynamic public families while retaining the selected static
constraint polynomial exactly. -/
def withDynamicClaims
    {shape : SemanticShape}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (input : PublicInput shape) :
    PublicInput shape where
  constraintPolynomial := constraintPolynomial
  priorPoint := input.priorPoint
  claimedYRing := input.claimedYRing

@[simp] theorem withDynamicClaims_constraintPolynomial
    {shape : SemanticShape}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (input : PublicInput shape) :
    (withDynamicClaims constraintPolynomial input).constraintPolynomial =
      constraintPolynomial :=
  rfl

@[simp] theorem withDynamicClaims_priorPoint
    {shape : SemanticShape}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (input : PublicInput shape) :
    (withDynamicClaims constraintPolynomial input).priorPoint =
      input.priorPoint :=
  rfl

@[simp] theorem withDynamicClaims_claimedYRing
    {shape : SemanticShape}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (input : PublicInput shape) :
    (withDynamicClaims constraintPolynomial input).claimedYRing =
      input.claimedYRing :=
  rfl

/-- Once the selected constraint polynomial is decoded exactly, restoring
the other two fields reconstructs the complete public input. -/
theorem withDynamicClaims_eq
    {shape : SemanticShape}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (input : PublicInput shape)
    (selected : input.constraintPolynomial = constraintPolynomial) :
    withDynamicClaims constraintPolynomial input = input := by
  apply PublicInput.ext
  · exact selected.symm
  · rfl
  · rfl

/-- The FE row degree is definitionally unchanged by restoring dynamic
claims. -/
@[simp] theorem drow_withDynamicClaims
    {shape : SemanticShape}
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (input : PublicInput shape) :
    SumCheck.Fe.Drow (withDynamicClaims constraintPolynomial input) =
      SumCheck.Fe.Drow (layoutInput constraintPolynomial) :=
  rfl

/-- Re-index one transcript layout at the restored public input.  Its physical
fields are unchanged because the only dependent index is the FE degree. -/
def retargetTranscript
    {shape : SemanticShape}
    {domains : Domains}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    (input : PublicInput shape)
    (transcript :
      KSplitNcTranscript.Input (layoutInput constraintPolynomial) domains) :
    KSplitNcTranscript.Input
      (withDynamicClaims constraintPolynomial input) domains where
  transcriptBase := transcript.transcriptBase
  priorLanes := transcript.priorLanes
  priorAbsorbed := transcript.priorAbsorbed
  statementFields := transcript.statementFields
  outputFields := transcript.outputFields
  fe := {
    initial := transcript.fe.initial
    rowRounds := transcript.fe.rowRounds
    boundary := transcript.fe.boundary
    laneRounds := transcript.fe.laneRounds
    terminal := transcript.fe.terminal
  }
  nc := transcript.nc

/-- Re-index the complete operational row input without moving or copying any
column. -/
def retarget
    {shape : SemanticShape}
    {domains : Domains}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    (publicInput : PublicInput shape)
    (input :
      KSplitNcOperationalRows.Input
        (layoutInput constraintPolynomial) domains) :
    KSplitNcOperationalRows.Input
      (withDynamicClaims constraintPolynomial publicInput) domains where
  transcript := retargetTranscript publicInput input.transcript
  authority := input.authority

/-- Retargeting changes no emitted operational row. -/
theorem rows_retarget
    {shape : SemanticShape}
    {domains : Domains}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    (constants : Poseidon2Schedule.Constants)
    (publicInput : PublicInput shape)
    (input :
      KSplitNcOperationalRows.Input
        (layoutInput constraintPolynomial) domains) :
    KSplitNcOperationalRows.rows constants
        (retarget publicInput input) =
      KSplitNcOperationalRows.rows constants input :=
  rfl

/-- The endpoint view after retargeting reads the same physical authority
columns. -/
@[simp] theorem retarget_authority
    {shape : SemanticShape}
    {domains : Domains}
    {constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount}
    (publicInput : PublicInput shape)
    (input :
      KSplitNcOperationalRows.Input
        (layoutInput constraintPolynomial) domains) :
    (retarget publicInput input).authority = input.authority :=
  rfl

end Nightstream.Implementation.R1CS.Canonical.KSplitNcStaticInput
