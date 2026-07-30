import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection

/-!
Contract: complete codec-owned coordinate views for the public carriers of the
selected fixed-active NIFS.

The views below are representation data.  They name every base-field or
quadratic-extension coordinate used by incoming running authority, outgoing
`Pi_DEC`, and result materialization.  Successful decoding supplies the
admissibility hypotheses used to fix evaluation-array sizes.

No field carries verifier acceptance, a recomposition equation, an output
claim, source authority, or a paper-event branch.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

universe uTranscriptState

def pointCoordinate
    {variables : Nat}
    (coordinate : Fin variables)
    (point : CubePoint Nightstream.SuperNeo.Concrete.K variables) :
    Nightstream.SuperNeo.Concrete.K :=
  point.coordinates.get
    ⟨coordinate.val, by
      rw [point.dimension]
      exact coordinate.isLt⟩

/-! ## Exact semantic coordinate selectors -/

def parentCommitmentCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (row : Fin verifierRows)
    (lane : Fin ringDegree)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) : F :=
  running.parent.commitment row lane

def childCommitmentCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (child : Fin productionGlobalParams.k)
    (row : Fin verifierRows)
    (lane : Fin ringDegree)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) : F :=
  (running.children child).commitment row lane

def parentPublicCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (column : Fin (ringDegree * publicRingColumns))
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) : F :=
  running.parent.publicInput column

def childPublicCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (child : Fin productionGlobalParams.k)
    (column : Fin (ringDegree * publicRingColumns))
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) : F :=
  (running.children child).publicInput column

def parentPointCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (coordinate : Fin shape.rowVariables)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    Nightstream.SuperNeo.Concrete.K :=
  pointCoordinate coordinate running.parent.point

def childPointCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin shape.rowVariables)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    Nightstream.SuperNeo.Concrete.K :=
  pointCoordinate coordinate (running.children child).point

def parentEvaluationCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    Nightstream.SuperNeo.Concrete.K :=
  running.parent.evaluations.getD matrix.val ringKZero lane

def childEvaluationCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (child : Fin productionGlobalParams.k)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    Nightstream.SuperNeo.Concrete.K :=
  (running.children child).evaluations.getD matrix.val ringKZero lane

def freshCommitmentCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (row : Fin verifierRows)
    (lane : Fin ringDegree)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows) : F :=
  fresh.commitment row lane

def freshPublicCoordinate
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (column : Fin (ringDegree * publicRingColumns))
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows) : F :=
  fresh.publicInput column

def payloadCommitmentCoordinate
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (child : Fin productionGlobalParams.k)
    (row : Fin verifierRows)
    (lane : Fin ringDegree)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : F :=
  (proof.certificate.piDecPayloads child).commitment row lane

def payloadPublicCoordinate
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (child : Fin productionGlobalParams.k)
    (column : Fin (ringDegree * publicRingColumns))
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : F :=
  (proof.certificate.piDecPayloads child).publicInput column

def payloadEvaluationCoordinate
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (child : Fin productionGlobalParams.k)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Nightstream.SuperNeo.Concrete.K :=
  (proof.certificate.piDecPayloads child).evaluations.getD
    matrix.val ringKZero lane

/-! ## Proof-carrying codec projections -/

/-- Complete coordinate map for one selected running carrier.  The same
structure applies to the running operand and to the running output codec. -/
structure RunningViews
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (codec :
      Codec
        (SelectedRunning shape publicRingColumns publicFits verifierRows)) where
  parentCommitment :
    ∀ row lane,
      PaperNifsCodecProjection.FView codec
        (parentCommitmentCoordinate row lane)
  childCommitment :
    ∀ child row lane,
      PaperNifsCodecProjection.FView codec
        (childCommitmentCoordinate child row lane)
  parentPublic :
    ∀ column,
      PaperNifsCodecProjection.FView codec (parentPublicCoordinate column)
  childPublic :
    ∀ child column,
      PaperNifsCodecProjection.FView codec
        (childPublicCoordinate child column)
  parentPoint :
    ∀ coordinate,
      PaperNifsCodecProjection.KView codec
        (parentPointCoordinate coordinate)
  childPoint :
    ∀ child coordinate,
      PaperNifsCodecProjection.KView codec
        (childPointCoordinate child coordinate)
  parentEvaluation :
    ∀ matrix lane,
      PaperNifsCodecProjection.KView codec
        (parentEvaluationCoordinate matrix lane)
  childEvaluation :
    ∀ child matrix lane,
      PaperNifsCodecProjection.KView codec
        (childEvaluationCoordinate child matrix lane)
  parentEvaluationsSize :
    ∀ running,
      codec.Admissible running →
      running.parent.evaluations.size = shape.matrixCount
  childEvaluationsSize :
    ∀ running,
      codec.Admissible running →
      ∀ child,
        (running.children child).evaluations.size = shape.matrixCount

/-! ## Complete running-codec coverage -/

/-- One base-field half of a quadratic-extension coordinate. -/
inductive KComponent where
  | c0
  | c1
deriving DecidableEq, Repr

namespace KComponent

def value (component : KComponent) (input : Nightstream.SuperNeo.Concrete.K) :
    F :=
  match component with
  | .c0 => input.c0
  | .c1 => input.c1

/-- Restrict a two-coordinate `KView` to one exact base-field coordinate. -/
def view
    {α : Type}
    {codec : Codec α}
    {source : α → Nightstream.SuperNeo.Concrete.K}
    (component : KComponent)
    (sourceView : PaperNifsCodecProjection.KView codec source) :
    PaperNifsCodecProjection.FView codec
      (fun input => component.value (source input)) :=
  match component with
  | .c0 => {
      index := sourceView.c0Index
      encodeValue := sourceView.encodeC0
    }
  | .c1 => {
      index := sourceView.c1Index
      encodeValue := sourceView.encodeC1
    }

end KComponent

/-- Every semantic coordinate family in the selected running carrier.

This is a coordinate descriptor, not a serialized index.  A selected codec
coverage certificate below maps every physical codec index to one descriptor,
so an unmentioned padding or authority coordinate cannot hide outside the
row-derived output proof. -/
inductive RunningCoordinate
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat) where
  | parentCommitment
      (row : Fin verifierRows) (lane : Fin ringDegree)
  | childCommitment
      (child : Fin productionGlobalParams.k)
      (row : Fin verifierRows) (lane : Fin ringDegree)
  | parentPublic
      (column : Fin (ringDegree * publicRingColumns))
  | childPublic
      (child : Fin productionGlobalParams.k)
      (column : Fin (ringDegree * publicRingColumns))
  | parentPoint
      (coordinate : Fin shape.rowVariables) (component : KComponent)
  | childPoint
      (child : Fin productionGlobalParams.k)
      (coordinate : Fin shape.rowVariables) (component : KComponent)
  | parentEvaluation
      (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
      (component : KComponent)
  | childEvaluation
      (child : Fin productionGlobalParams.k)
      (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
      (component : KComponent)

namespace RunningCoordinate

def value
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (coordinate :
      RunningCoordinate shape publicRingColumns verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) : F :=
  match coordinate with
  | .parentCommitment row lane =>
      parentCommitmentCoordinate row lane running
  | .childCommitment child row lane =>
      childCommitmentCoordinate child row lane running
  | .parentPublic column =>
      parentPublicCoordinate column running
  | .childPublic child column =>
      childPublicCoordinate child column running
  | .parentPoint index component =>
      component.value (parentPointCoordinate index running)
  | .childPoint child index component =>
      component.value (childPointCoordinate child index running)
  | .parentEvaluation matrix lane component =>
      component.value (parentEvaluationCoordinate matrix lane running)
  | .childEvaluation child matrix lane component =>
      component.value
        (childEvaluationCoordinate child matrix lane running)

def view
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {codec :
      Codec
        (SelectedRunning shape publicRingColumns publicFits verifierRows)}
    (views : RunningViews codec)
    (coordinate :
      RunningCoordinate shape publicRingColumns verifierRows) :
    PaperNifsCodecProjection.FView codec coordinate.value :=
  match coordinate with
  | .parentCommitment row lane =>
      views.parentCommitment row lane
  | .childCommitment child row lane =>
      views.childCommitment child row lane
  | .parentPublic column =>
      views.parentPublic column
  | .childPublic child column =>
      views.childPublic child column
  | .parentPoint index component =>
      component.view (views.parentPoint index)
  | .childPoint child index component =>
      component.view (views.childPoint child index)
  | .parentEvaluation matrix lane component =>
      component.view (views.parentEvaluation matrix lane)
  | .childEvaluation child matrix lane component =>
      component.view (views.childEvaluation child matrix lane)

end RunningCoordinate

/-- Proof that the named running views cover the complete physical codec.

`coordinateAt` is total over `Fin codec.width`: every serialized coordinate
is one of the explicit semantic families above.  `resultAdmissible` converts
the verifier-owned exact evaluation-array sizes into the codec-domain fact
needed to decode the computed result.  Neither field contains a row equation
or a verifier conclusion. -/
structure RunningCodecCoverage
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (codec :
      Codec
        (SelectedRunning shape publicRingColumns publicFits verifierRows))
    (views : RunningViews codec) where
  coordinateAt :
    Fin codec.width →
      RunningCoordinate shape publicRingColumns verifierRows
  indexExact :
    ∀ index,
      ((coordinateAt index).view views).index = index
  resultAdmissible :
    ∀ result :
      FixedActive.FoldResult
        shape publicRingColumns publicFits verifierRows,
      result.parent.evaluations.size = shape.matrixCount →
      (∀ child,
        (result.children child).evaluations.size = shape.matrixCount) →
      codec.Admissible (SelectedRunning.ofResult result)

namespace RunningCodecCoverage

/-- Exact values at all covered views determine the complete serialized
running carrier.  This theorem is the fail-closed use of `coordinateAt`: its
quantifier ranges over every physical codec index, not merely the views
needed by one arithmetic gadget. -/
theorem coordinates_eq_encode
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {codec :
      Codec
        (SelectedRunning shape publicRingColumns publicFits verifierRows)}
    {views : RunningViews codec}
    (coverage : RunningCodecCoverage codec views)
    (coordinates : List F)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (lengthExact : coordinates.length = codec.width)
    (valuesExact :
      ∀ coordinate :
        RunningCoordinate shape publicRingColumns verifierRows,
        coordinates.getD (coordinate.view views).index.val 0 =
          coordinate.value running) :
    coordinates = codec.encode running := by
  apply List.ext_get
  · rw [lengthExact, codec.encode_length]
  · intro index coordinatesLt encodedLt
    let physical : Fin codec.width := ⟨index, by omega⟩
    let coordinate := coverage.coordinateAt physical
    have coordinateIndex :
        (coordinate.view views).index.val = index := by
      exact congrArg Fin.val (coverage.indexExact physical)
    have leftExact := valuesExact coordinate
    have rightExact := (coordinate.view views).encodeValue running
    rw [coordinateIndex] at leftExact rightExact
    rw [← List.getElem_eq_getD
        (l := coordinates) (i := index) (h := coordinatesLt) 0]
      at leftExact
    rw [← List.getElem_eq_getD
        (l := codec.encode running) (i := index) (h := encodedLt) 0]
      at rightExact
    exact leftExact.trans rightExact.symm

end RunningCodecCoverage

/-- Complete coordinate map for the fresh source payload. -/
structure FreshViews
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (codec :
      Codec
        (SelectedFresh shape publicRingColumns publicFits verifierRows)) where
  commitment :
    ∀ row lane,
      PaperNifsCodecProjection.FView codec
        (freshCommitmentCoordinate row lane)
  publicInput :
    ∀ column,
      PaperNifsCodecProjection.FView codec (freshPublicCoordinate column)

/-- Complete coordinate map for all fourteen prover-supplied `Pi_DEC`
payloads.  Points are deliberately absent: the verifier inherits each child
point from the computed parent. -/
structure PayloadViews
    {shape : SemanticShape}
    {TranscriptState : Type uTranscriptState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (codec :
      Codec
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows)) where
  commitment :
    ∀ child row lane,
      PaperNifsCodecProjection.FView codec
        (payloadCommitmentCoordinate child row lane)
  publicInput :
    ∀ child column,
      PaperNifsCodecProjection.FView codec
        (payloadPublicCoordinate child column)
  evaluation :
    ∀ child matrix lane,
      PaperNifsCodecProjection.KView codec
        (payloadEvaluationCoordinate child matrix lane)
  evaluationsSize :
    ∀ proof,
      codec.Admissible proof →
      ∀ child,
        (proof.certificate.piDecPayloads child).evaluations.size =
          shape.matrixCount

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
