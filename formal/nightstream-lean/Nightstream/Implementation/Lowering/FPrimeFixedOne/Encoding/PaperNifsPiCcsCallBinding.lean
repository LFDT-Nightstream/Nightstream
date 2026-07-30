import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding
import Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscript

/-!
Contract: bind the numeric `Pi_CCS` occurrence to the running operand of one
typed `nifsVerify` call.

`RunningPlacement` locates every public `Pi_CCS` value in one shared
numeric-to-typed column map.  `bindInput` then overwrites the three
verifier-input fields of a transcript occurrence from their sole authorities:
the key-owned polynomial and those located running-operand coordinates.
Satisfaction cannot choose an independent prior point or carried-evaluation
family.

The remaining transcript fields are still supplied by the proof/output
binding layers.  This module does not claim complete NIFS soundness,
construct hidden source assignments, or introduce a source-binding premise.

Emits constraints: none beyond the rows already owned by
`KPiCcsTranscript`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsSourceBinding
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.Nifs

universe uCommitment uPublicInput uScalar uState

abbrev K := Nightstream.SuperNeo.Concrete.K

/-- All public `Pi_CCS` running values located in the single call-local
numeric namespace. -/
structure RunningPlacement
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    (views : RunningViews codec)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (widthsAgree : codec.width = layout.owners.length)
    (columnMap : Nat → ColumnId) where
  priorPoint :
    ∀ index : Fin shape.cubeVariables,
      KLocation columnMap
        ((views.priorPoint index).columns bundle widthsAgree)
  claimedCoefficient :
    ∀ coordinate : CarriedCoordinate shape,
      KLocation columnMap
        ((views.claimedCoefficient coordinate).columns bundle widthsAgree)

def RunningPlacement.priorPointCarried
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    {views : RunningViews codec}
    {layout : Layout}
    {bundle : ColumnBundle layout}
    {widthsAgree : codec.width = layout.owners.length}
    {columnMap : Nat → ColumnId}
    (placement :
      RunningPlacement views bundle widthsAgree columnMap)
    (index : Fin shape.cubeVariables) :
    KMul.Carried :=
  (placement.priorPoint index).carried

def RunningPlacement.claimedCoefficientCarried
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {codec :
      Codec (PaperNonInteractive.Running K Commitment PublicInput shape)}
    {views : RunningViews codec}
    {layout : Layout}
    {bundle : ColumnBundle layout}
    {widthsAgree : codec.width = layout.owners.length}
    {columnMap : Nat → ColumnId}
    (placement :
      RunningPlacement views bundle widthsAgree columnMap)
    (coordinate : CarriedCoordinate shape) :
    KMul.Carried :=
  (placement.claimedCoefficient coordinate).carried

/-- Replace exactly the public verifier-input fields of a transcript
occurrence.  All proof/output and transcript-placement fields are preserved
from `template`. -/
def RunningPlacement.bindInput
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
    {views : RunningViews codec}
    {layout : Layout}
    {bundle : ColumnBundle layout}
    {widthsAgree : codec.width = layout.owners.length}
    {columnMap : Nat → ColumnId}
    (placement :
      RunningPlacement views bundle widthsAgree columnMap)
    (template : KPiCcsTranscript.Input shape degreeBound) :
    KPiCcsTranscript.Input shape degreeBound :=
  { template with
    constraintPolynomial :=
      ConstraintPolynomialLift.liftConstraintPolynomial key.lift
        key.matrixSource.constraintPolynomial
    priorPoint := placement.priorPointCarried
    claimedCoefficient := placement.claimedCoefficientCarried }

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

/-- The numeric occurrence's decoded prior point is the exact physical point
read from the typed running bundle. -/
theorem RunningPlacement.decodedPriorPoint_eq
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
    {views : RunningViews codec}
    {layout : Layout}
    {bundle : ColumnBundle layout}
    {widthsAgree : codec.width = layout.owners.length}
    {columnMap : Nat → ColumnId}
    (placement :
      RunningPlacement views bundle widthsAgree columnMap)
    (template : KPiCcsTranscript.Input shape degreeBound)
    (assignment : ColumnId → Field) :
    (KPiCcsOccurrence.decodedVerifierInput
        (KPiCcsTranscript.occurrenceInput
          (placement.bindInput key template))
        (numericAssignment columnMap assignment)).priorPoint =
      views.physicalPriorPoint bundle widthsAgree assignment := by
  apply cubePoint_eq_of_coordinates_eq
  unfold KPiCcsOccurrence.decodedVerifierInput
    KPiCcsOccurrence.terminalInput
    KPiCcsTranscript.occurrenceInput
    KPiCcsTerminal.decodedInput
    KPiCcsTerminal.priorEqualityInput
    KPointEquality.decodedRight
    RunningViews.physicalPriorPoint
    RunningPlacement.bindInput
  simp only [List.map_map, Function.comp_apply]
  apply List.map_congr_left
  intro index _
  exact (placement.priorPoint index).decodeCarried_eq assignment

/-- The numeric occurrence's decoded carried coefficients are exactly the
physical family read from the typed running bundle. -/
theorem RunningPlacement.decodedClaimedCoefficient_eq
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
    {views : RunningViews codec}
    {layout : Layout}
    {bundle : ColumnBundle layout}
    {widthsAgree : codec.width = layout.owners.length}
    {columnMap : Nat → ColumnId}
    (placement :
      RunningPlacement views bundle widthsAgree columnMap)
    (template : KPiCcsTranscript.Input shape degreeBound)
    (assignment : ColumnId → Field) :
    (KPiCcsOccurrence.decodedVerifierInput
        (KPiCcsTranscript.occurrenceInput
          (placement.bindInput key template))
        (numericAssignment columnMap assignment)).claimedCoefficient =
      views.physicalClaimedCoefficient bundle widthsAgree assignment := by
  funext coordinate
  exact (placement.claimedCoefficient coordinate).decodeCarried_eq assignment

/-- The complete numeric row occurrence decodes to the exact typed
physical verifier input. -/
theorem RunningPlacement.decodedVerifierInput_eq_physical
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
    {views : RunningViews codec}
    {layout : Layout}
    {bundle : ColumnBundle layout}
    {widthsAgree : codec.width = layout.owners.length}
    {columnMap : Nat → ColumnId}
    (placement :
      RunningPlacement views bundle widthsAgree columnMap)
    (template : KPiCcsTranscript.Input shape degreeBound)
    (assignment : ColumnId → Field) :
    KPiCcsOccurrence.decodedVerifierInput
        (KPiCcsTranscript.occurrenceInput
          (placement.bindInput key template))
        (numericAssignment columnMap assignment) =
      views.physicalVerifierInput key bundle widthsAgree assignment := by
  apply verifierInput_eq_of_fields_eq
  · rfl
  · exact placement.decodedPriorPoint_eq key template assignment
  · exact placement.decodedClaimedCoefficient_eq key template assignment

/-- **The canonical PiCCS occurrence is internally source-bound to the
decoded running operand.**

Unlike the earlier generic event theorem, the source-binding equality is a
conclusion here, not a caller premise. -/
theorem RunningPlacement.decodedVerifierInput_eq_statement
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
    {views : RunningViews codec}
    {layout : Layout}
    {bundle : ColumnBundle layout}
    {widthsAgree : codec.width = layout.owners.length}
    {columnMap : Nat → ColumnId}
    (placement :
      RunningPlacement views bundle widthsAgree columnMap)
    (template : KPiCcsTranscript.Input shape degreeBound)
    (assignment : ColumnId → Field)
    (running : PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
    (decoded : codec.decode (bundle.values assignment) = some running) :
    KPiCcsOccurrence.decodedVerifierInput
        (KPiCcsTranscript.occurrenceInput
          (placement.bindInput key template))
        (numericAssignment columnMap assignment) =
      (key.statement running fresh).verifierInput key.lift := by
  rw [placement.decodedVerifierInput_eq_physical key template assignment]
  exact views.physicalVerifierInput_eq_statement
    key bundle widthsAgree assignment running fresh decoded

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiCcsCallBinding
