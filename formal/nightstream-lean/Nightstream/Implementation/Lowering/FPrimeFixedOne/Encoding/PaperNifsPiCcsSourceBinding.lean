import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Types

/-!
Contract: reconstruct the paper `Pi_CCS` verifier input from the exact
physical coordinates of a decoded running NIFS operand.

The executable `Pi_CCS` input contains only three authorities:

* the key-owned lifted constraint polynomial;
* the running claim's prior evaluation point; and
* the running claim's complete carried-evaluation family.

The first is selected by the verifier key.  The latter two are projected from
the running operand codec through `KView`; successful whole-value decoding
therefore binds every projected pair to the corresponding semantic field.

This module emits no rows.  It does not decode the NIFS proof, derive
Fiat--Shamir challenges, construct hidden source assignments, assume
`SourceAuthority`, or assert verifier acceptance.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.Nifs

universe uCommitment uPublicInput uScalar uState

abbrev K := Nightstream.SuperNeo.Concrete.K

/-- The exact point coordinate selected by one finite index. -/
def runningPointCoordinate
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    (index : Fin shape.cubeVariables)
    (running : PaperNonInteractive.Running K Commitment PublicInput shape) : K :=
  running.point.coordinates.get
    ⟨index.val, by
      rw [running.point.dimension]
      exact index.isLt⟩

/-- Serialization ownership for every `K` value retained by the executable
`Pi_CCS` input.

Each field identifies two in-range coordinates of the selected running codec
and proves their exact semantic meaning.  It contains no acceptance
proposition and no row-level conclusion. -/
structure RunningViews
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    (codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)) where
  priorPoint :
    ∀ index : Fin shape.cubeVariables,
      KView codec (runningPointCoordinate index)
  claimedCoefficient :
    ∀ coordinate : CarriedCoordinate shape,
      KView codec (fun running =>
        running.evaluations coordinate.running coordinate.matrix
          coordinate.coefficient)

/-- The point read from the physical running bundle in canonical coordinate
order. -/
def RunningViews.physicalPriorPoint
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    (views : RunningViews codec)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field) :
    CubePoint K shape.cubeVariables where
  coordinates :=
    (canonicalFinIndices shape.cubeVariables).map fun index =>
      ((views.priorPoint index).columns bundle widthsAgree).value assignment
  dimension := by
    rw [List.length_map, canonicalFinIndices_length]

/-- The carried-evaluation family read from the same decoded bundle. -/
def RunningViews.physicalClaimedCoefficient
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    (views : RunningViews codec)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field) :
    CarriedCoordinate shape → K :=
  fun coordinate =>
    ((views.claimedCoefficient coordinate).columns bundle widthsAgree).value
      assignment

/-- The exact executable `Pi_CCS` input reconstructed from key authority and
the physical running bundle. -/
def RunningViews.physicalVerifierInput
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key :
      PaperNonInteractive.Key K Commitment PublicInput Scalar State shape
        columns blockCount degreeBound)
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    (views : RunningViews codec)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field) :
    ProtocolPolynomial.VerifierInput K shape where
  constraintPolynomial :=
    ConstraintPolynomialLift.liftConstraintPolynomial key.lift
      key.matrixSource.constraintPolynomial
  priorPoint :=
    views.physicalPriorPoint bundle widthsAgree assignment
  claimedCoefficient :=
    views.physicalClaimedCoefficient bundle widthsAgree assignment

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat}
    {left right : CubePoint K variables}
    (equal : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp only at equal
  subst equal
  rfl

/-- Decoding the whole running operand binds the reconstructed point to the
semantic running point. -/
theorem RunningViews.physicalPriorPoint_eq
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    (views : RunningViews codec)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field)
    (running : PaperNonInteractive.Running K Commitment PublicInput shape)
    (decoded : codec.decode (bundle.values assignment) = some running) :
    views.physicalPriorPoint bundle widthsAgree assignment =
      running.point := by
  apply cubePoint_eq_of_coordinates_eq
  apply List.ext_getElem
  · rw [running.point.dimension]
    exact (views.physicalPriorPoint bundle widthsAgree assignment).dimension
  · intro index leftBound rightBound
    have indexLt : index < shape.cubeVariables := by
      rw [←
        (views.physicalPriorPoint bundle widthsAgree assignment).dimension]
      exact leftBound
    simp only [RunningViews.physicalPriorPoint, List.getElem_map]
    have finIndex :
        (canonicalFinIndices shape.cubeVariables)[index]'(by
            rw [canonicalFinIndices_length]
            exact indexLt) =
          (⟨index, indexLt⟩ : Fin shape.cubeVariables) := by
      simp [canonicalFinIndices]
    rw [finIndex]
    have projected :=
      (views.priorPoint ⟨index, indexLt⟩).value_eq_of_decodes
        bundle widthsAgree assignment running decoded
    simpa only [runningPointCoordinate] using projected

/-- Decoding the whole running operand binds every carried evaluation to the
corresponding semantic running evaluation. -/
theorem RunningViews.physicalClaimedCoefficient_eq
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    (views : RunningViews codec)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field)
    (running : PaperNonInteractive.Running K Commitment PublicInput shape)
    (decoded : codec.decode (bundle.values assignment) = some running) :
    views.physicalClaimedCoefficient bundle widthsAgree assignment =
      fun coordinate =>
        running.evaluations coordinate.running coordinate.matrix
          coordinate.coefficient := by
  funext coordinate
  exact
    (views.claimedCoefficient coordinate).value_eq_of_decodes
      bundle widthsAgree assignment running decoded

theorem verifierInput_eq_of_fields_eq
    {shape : Shape}
    {left right : ProtocolPolynomial.VerifierInput K shape}
    (polynomial :
      left.constraintPolynomial = right.constraintPolynomial)
    (point : left.priorPoint = right.priorPoint)
    (coefficients :
      left.claimedCoefficient = right.claimedCoefficient) :
    left = right := by
  cases left with
  | mk leftPolynomial leftPoint leftCoefficients =>
      cases right with
      | mk rightPolynomial rightPoint rightCoefficients =>
          simp only at polynomial point coefficients
          cases polynomial
          cases point
          cases coefficients
          rfl

/-- **Authoritative physical-to-paper source binding for `Pi_CCS`.**

The conclusion is the unchanged verifier input selected by
`Key.statement`.  It follows solely from the key, the selected codec
projection laws, and successful decoding of the actual running operand.
There is no caller-provided verifier input or source-binding equality. -/
theorem RunningViews.physicalVerifierInput_eq_statement
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key :
      PaperNonInteractive.Key K Commitment PublicInput Scalar State shape
        columns blockCount degreeBound)
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    (views : RunningViews codec)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (assignment : ColumnId → Field)
    (running : PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (decoded : codec.decode (bundle.values assignment) = some running) :
    views.physicalVerifierInput key bundle widthsAgree assignment =
      (key.statement running fresh).verifierInput key.lift := by
  apply verifierInput_eq_of_fields_eq
  · rfl
  · exact
      views.physicalPriorPoint_eq
        bundle widthsAgree assignment running decoded
  · exact
      views.physicalClaimedCoefficient_eq
        bundle widthsAgree assignment running decoded

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding
