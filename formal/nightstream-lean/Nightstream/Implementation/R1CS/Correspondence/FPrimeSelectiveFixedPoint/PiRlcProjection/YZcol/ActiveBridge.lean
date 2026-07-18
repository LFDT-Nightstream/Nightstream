import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ArtifactRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ProducerBinding
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity

/-!
Active 15-source/13-matrix bridge for the fixed-point PiRLC `y_zcol`
projection rows.

Owns: the shape-generic low/high trace pair, its fixed-fixture shape proof,
physical consumer-column identity, and composition of exact satisfied source
rows with the independent typed projection theorem.

Does not own: PiCCS source truth, transcript authority, whole-program
source-to-selective lowering, bad-root probability, production conformance,
costs, necessity, or permission to remove rows.

Emits constraints: no.

| Correspondence obligation | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `active.y_zcol.trace_shape` | two 15-pair Phi81 traces have valid layouts, widths, and shared challenges | artifact-checked | `tracePairShapeValid` |
| `active.y_zcol.serializer_index` | serializer indices follow the 15-source/13-matrix typed layout | artifact-checked | `serializerIndicesMatch` |
| `active.y_zcol.consumer_binding` | typed consumer columns are the exact trace input columns | artifact-checked + checked upstream | `inputConsumer_eq_traceConsumerColumns`, `consumerMatches` |
| `active.y_zcol.source_rows` | satisfied source rows across `projection_shared` and `identities.y_zcol` imply the typed aggregate or one named bad root | artifact-checked + model-level | `rows_decodedOutput_eq_messageAggregate_or_badRoot` |

Assurance tier: source-artifact-checked for the bounded tiny fixture, followed
by a conditional model-level consequence. Serializer indices and source
columns are separate refinements; no final selectively emitted row count or
source-to-selective theorem is established here.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ActiveBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout
open Nightstream.Implementation.R1CS.ProjectionPhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.ProjectionCheck

private theorem checkedLowPairCount : Checked.lowTrace.pairs.length = 15 := by
  set_option maxRecDepth 100000 in
    decide

private theorem checkedHighPairCount : Checked.highTrace.pairs.length = 15 := by
  set_option maxRecDepth 100000 in
    decide

private def checkedLowPair (index : Fin 15) : ProjectionProgram.PairTrace :=
  Checked.lowTrace.pairs.get (Fin.cast checkedLowPairCount.symm index)

private def checkedHighPair (index : Fin 15) : ProjectionProgram.PairTrace :=
  Checked.highTrace.pairs.get (Fin.cast checkedHighPairCount.symm index)

private def pairWidthsValid
    (trace : ProjectionProgram.ProjectionTrace) : Bool :=
  trace.pairs.all fun pair =>
    decide (pair.rhoColumns.length = ringDegree /\
      pair.inputColumns.length = ringDegree)

private theorem checkedLowPairWidthsCheck :
    pairWidthsValid Checked.lowTrace = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem checkedHighPairWidthsCheck :
    pairWidthsValid Checked.highTrace = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem checkedLowPairWidths :
    forall candidate, candidate ∈ Checked.lowTrace.pairs →
      candidate.rhoColumns.length = ringDegree /\
        candidate.inputColumns.length = ringDegree := by
  intro candidate member
  have checked :=
    (List.all_eq_true.mp checkedLowPairWidthsCheck) candidate member
  exact of_decide_eq_true checked

private theorem checkedHighPairWidths :
    forall candidate, candidate ∈ Checked.highTrace.pairs →
      candidate.rhoColumns.length = ringDegree /\
        candidate.inputColumns.length = ringDegree := by
  intro candidate member
  have checked :=
    (List.all_eq_true.mp checkedHighPairWidthsCheck) candidate member
  exact of_decide_eq_true checked

/-- The fixed traces share every complete rho coefficient list. This is a
bounded 15-source fact over trace metadata, not assignment values. -/
private theorem checkedChallengeColumnsShared :
    forall index : Fin 15,
      (checkedHighPair index).rhoColumns =
        (checkedLowPair index).rhoColumns := by
  set_option maxRecDepth 100000 in
    decide

/-- Trace reconstruction and the producer audit name the same complete input
coefficient lists. These two bounded statements are the physical bridge used
below; no decoded assignment values occur in them. -/
private theorem checkedLowInputColumns :
    forall index : Fin 15,
      (checkedLowPair index).inputColumns =
        ProducerBinding.rawTraceInputColumns 0 index.val := by
  set_option maxRecDepth 100000 in
    decide

private theorem checkedHighInputColumns :
    forall index : Fin 15,
      (checkedHighPair index).inputColumns =
        ProducerBinding.rawTraceInputColumns 1 index.val := by
  set_option maxRecDepth 100000 in
    decide

private theorem checkedLayouts
    (census : Checked.artifact.StructureValid) :
    Checked.lowTrace.LayoutValid /\ Checked.highTrace.LayoutValid := by
  rcases census with ⟨_, _, _, _, limbs, _, _, _⟩
  have lowMember : Checked.lowLimb ∈ Checked.artifact.limbs := by
    change Checked.lowLimb ∈ [Checked.lowLimb, Checked.highLimb]
    simp
  have highMember : Checked.highLimb ∈ Checked.artifact.limbs := by
    change Checked.highLimb ∈ [Checked.lowLimb, Checked.highLimb]
    simp
  have lowValid := limbs Checked.lowLimb lowMember
  have highValid := limbs Checked.highLimb highMember
  rcases lowValid with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, lowLayout⟩
  rcases highValid with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, highLayout⟩
  exact ⟨lowLayout, highLayout⟩

/-- Stable checked traces specialized only by the independent semantic source
count. Other shape fields do not influence the projection identity. -/
def tracePair
    (shape : SemanticShape)
    (sourceCount : shape.sourceCount = 15) :
    ProjectionIdentity.TracePair shape where
  low := Checked.lowTrace
  high := Checked.highTrace
  lowPairCount := checkedLowPairCount.trans sourceCount.symm
  highPairCount := checkedHighPairCount.trans sourceCount.symm

theorem tracePair_traces
    (shape : SemanticShape)
    (sourceCount : shape.sourceCount = 15) :
    (tracePair shape sourceCount).traces = Checked.traces := by
  rfl

private theorem tracePair_lowPair_eq_checked
    (shape : SemanticShape)
    (sourceCount : shape.sourceCount = 15)
    (index : Fin shape.sourceCount) :
    (tracePair shape sourceCount).lowPair index =
      checkedLowPair (Fin.cast sourceCount index) := by
  unfold ProjectionIdentity.TracePair.lowPair tracePair checkedLowPair
  apply congrArg (fun position => Checked.lowTrace.pairs.get position)
  apply Fin.ext
  rfl

private theorem tracePair_highPair_eq_checked
    (shape : SemanticShape)
    (sourceCount : shape.sourceCount = 15)
    (index : Fin shape.sourceCount) :
    (tracePair shape sourceCount).highPair index =
      checkedHighPair (Fin.cast sourceCount index) := by
  unfold ProjectionIdentity.TracePair.highPair tracePair checkedHighPair
  apply congrArg (fun position => Checked.highTrace.pairs.get position)
  apply Fin.ext
  rfl

/-- Fixed-fixture structure plus the independent source-count equality imply
the complete generic projection shape contract. -/
theorem tracePairShapeValid
    (shape : SemanticShape)
    (sourceCount : shape.sourceCount = 15) :
    ProjectionIdentity.ShapeValid (tracePair shape sourceCount) := by
  have layouts := checkedLayouts Checked.structureValid
  refine
    { sourceCountPositive := ?_
      lowLayout := layouts.1
      highLayout := layouts.2
      lowPairWidths := ?_
      highPairWidths := ?_
      challengeColumnsShared := ?_ }
  · omega
  · intro candidate member
    exact checkedLowPairWidths candidate member
  · intro candidate member
    exact checkedHighPairWidths candidate member
  · intro index
    rw [tracePair_highPair_eq_checked, tracePair_lowPair_eq_checked]
    exact checkedChallengeColumnsShared (Fin.cast sourceCount index)

private theorem consumerColumns_ext
    {shape : SemanticShape}
    {left right : YZcolConsumer.ConsumerColumns shape}
    (columns : forall limb source lane,
      left.column limb source lane = right.column limb source lane) :
    left = right := by
  cases left with
  | mk leftColumn =>
      cases right with
      | mk rightColumn =>
          have same : leftColumn = rightColumn := by
            funext limb source lane
            exact columns limb source lane
          cases same
          rfl

/-- The generic trace pair and the producer audit reconstruct the same
physical consumer. This theorem compares column lists directly and is
independent of any assignment. -/
theorem inputConsumer_eq_traceConsumerColumns
    (shape : SemanticShape)
    (sourceCount : shape.sourceCount = 15) :
    (tracePair shape sourceCount).inputConsumer =
      ProducerBinding.traceConsumerColumns shape := by
  apply consumerColumns_ext
  intro limb source lane
  let fixedSource : Fin 15 := Fin.cast sourceCount source
  cases limb with
  | c0 =>
      change
        ((tracePair shape sourceCount).lowPair source).inputColumns.getD
            lane.val 0 =
          ProducerBinding.rawTraceConsumerColumn 0 source.val lane.val
      calc
        ((tracePair shape sourceCount).lowPair source).inputColumns.getD
              lane.val 0 =
            (checkedLowPair fixedSource).inputColumns.getD lane.val 0 := by
          rw [tracePair_lowPair_eq_checked]
        _ = (ProducerBinding.rawTraceInputColumns 0 fixedSource.val).getD
              lane.val 0 := by
          rw [checkedLowInputColumns fixedSource]
        _ = ProducerBinding.rawTraceConsumerColumn 0 source.val lane.val := by
          rfl
  | c1 =>
      change
        ((tracePair shape sourceCount).highPair source).inputColumns.getD
            lane.val 0 =
          ProducerBinding.rawTraceConsumerColumn 1 source.val lane.val
      calc
        ((tracePair shape sourceCount).highPair source).inputColumns.getD
              lane.val 0 =
            (checkedHighPair fixedSource).inputColumns.getD lane.val 0 := by
          rw [tracePair_highPair_eq_checked]
        _ = (ProducerBinding.rawTraceInputColumns 1 fixedSource.val).getD
              lane.val 0 := by
          rw [checkedHighInputColumns fixedSource]
        _ = ProducerBinding.rawTraceConsumerColumn 1 source.val lane.val := by
          rfl

/-- Serializer-coordinate refinement is deliberately separate from physical
source-column identity. -/
theorem serializerIndicesMatch
    (shape : SemanticShape)
    (sourceCount : shape.sourceCount = 15)
    (matrixCount : shape.matrixCount = 13) :
    ProducerBinding.SerializerIndicesMatch shape :=
  ProducerBinding.serializerIndicesMatch shape sourceCount matrixCount

/-- Upstream PiCCS producer authority transports through the exact physical
consumer equality. -/
theorem consumerMatches
    {shape : SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    {producer : SourceRole shape -> Nat}
    (upstream : ProducerBinding.UpstreamProducerColumnsBound producer) :
    YZcolConsumer.ConsumerMatches producer
      (tracePair shape sourceCount).inputConsumer := by
  rw [inputConsumer_eq_traceConsumerColumns shape sourceCount]
  exact ProducerBinding.consumerMatches_of_upstreamProducerColumns
    sourceCount upstream

/-- Exact artifact rows enter the generic semantic interface only through
explicit canonicality, constant-one, and selected source-row satisfaction. -/
theorem rowsSatisfied_of_sourceRows
    {shape : SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    {assignment : Nat -> Nat}
    (assignmentCanonical : forall column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies
      Checked.artifact.certificate.sourceRowValues assignment) :
    ProjectionIdentity.RowsSatisfied (tracePair shape sourceCount)
      assignment := by
  exact ArtifactRows.rowsSatisfied_of_sourceRows
    (tracePair_traces shape sourceCount) Checked.structureValid
    assignmentCanonical constantOne sourceSatisfies

/-- Embedding in a complete source field-R1CS is a convenience path only. The
production selective relation must instead refine its rewritten rows to
`rowsSatisfied_of_sourceRows` (or directly to `RowsSatisfied`). -/
theorem rowsSatisfied_of_embedded
    {shape : SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    {fullRows : List Row}
    {assignment : Nat -> Nat}
    (assignmentCanonical : forall column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (embedded : Checked.artifact.certificate.EmbeddedIn fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    ProjectionIdentity.RowsSatisfied (tracePair shape sourceCount)
      assignment := by
  exact ArtifactRows.rowsSatisfied_of_embedded
    (tracePair_traces shape sourceCount) Checked.structureValid
    assignmentCanonical constantOne embedded fullSatisfies

/-- Conditional source-arm projection consequence. The left branch is the independently
typed PiCCS-message aggregate; the right branch is the exact batch bad-root
event. No probability or transcript claim is hidden in either branch. -/
theorem rows_decodedOutput_eq_messageAggregate_or_badRoot
    {shape : SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    {fullRows : List Row}
    {assignment : Nat -> Nat}
    (assignmentCanonical : forall column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (embedded : Checked.artifact.certificate.EmbeddedIn fullRows)
    (fullSatisfies : Satisfies fullRows assignment)
    {producer : SourceRole shape -> Nat}
    (upstream : ProducerBinding.UpstreamProducerColumnsBound producer)
    {message : OutputMessage shape}
    (yZcolBound : BindingsHoldFor .yZcolOutput
      (semanticAssignment assignment) producer message) :
    ProjectionIdentity.decodedOutput (tracePair shape sourceCount)
          assignment =
        sourceAggregate
          (ProjectionIdentity.decodedChallenges
            (tracePair shape sourceCount) assignment)
          message.yZcol \/
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity
          (tracePair shape sourceCount).traces assignment) := by
  have rows := rowsSatisfied_of_embedded sourceCount assignmentCanonical
    constantOne embedded fullSatisfies
  exact ProjectionIdentity.rows_decodedOutput_eq_messageAggregate_or_badRoot
    (tracePairShapeValid shape sourceCount) constantOne rows
    (consumerMatches sourceCount upstream) yZcolBound

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ActiveBridge
