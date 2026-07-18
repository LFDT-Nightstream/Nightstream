import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.SerializerIndex
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer

/-!
Artifact-checked producer-coordinate correspondence for the bounded 15-source/13-matrix PiRLC
`y_zcol` projection fixture.

Owns: independent checks that raw serializer indices follow the typed
serializer formula and that raw producer columns equal the coefficient
columns reconstructed from the projection traces.

Does not own: producer-column authority, assignment satisfaction, exact-row
soundness, selective-lowering refinement, PiCCS output truth, transcript
binding, security bounds, costs, or row removal.

Emits constraints: no.

| Correspondence obligation | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `active.y_zcol.serializer_index` | raw field index equals the independent shape-indexed serializer formula | artifact-checked | `serializerIndicesMatch` |
| `active.y_zcol.producer_consumer_column` | raw producer source column equals the input column reconstructed from the projection trace | artifact-checked | `producerColumnsMatchTrace` |
| `active.y_zcol.upstream_authority` | an arbitrary upstream producer map is preserved as an explicit premise | checked upstream | `consumerMatches_of_upstreamProducerColumns` |

The six finite checks below each cover exactly five sources and 54 lanes for
one limb. They use ordinary kernel reduction; there is no monolithic decision
over all producer coordinates and no native-code proof shortcut.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ProducerBinding

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout

private abbrev SerializerLimb :=
  Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.Limb

/-- The checked artifact remains behind the stable facade; generated modules
are not part of this correspondence module's import surface. -/
private abbrev fixture : Artifact := Checked.artifact

private def missingProducer : ProducerVector where
  sourceIndex := 0
  limb := 0
  entries := []

private def missingEntry : ProducerEntry where
  serializerFieldIndex := 0
  sourceColumn := 0

/-- Raw artifact producer vector at the fixed limb-major coordinate. Missing
data maps to an empty sentinel, so any required coordinate fails closed. -/
def rawProducerVector (limb source : Nat) : ProducerVector :=
  fixture.producers.getD (limb * 15 + source) missingProducer

/-- Raw serializer field index retained by Rust for one producer leaf. -/
def rawProducerSerializerFieldIndex
    (limb source lane : Nat) : Nat :=
  ((rawProducerVector limb source).entries.getD lane missingEntry).serializerFieldIndex

/-- Raw source-R1CS column retained by Rust for the same producer leaf. -/
def rawProducerSourceColumn (limb source lane : Nat) : Nat :=
  ((rawProducerVector limb source).entries.getD lane missingEntry).sourceColumn

/-- Input coefficient columns reconstructed from the artifact's projection
traces, not copied from the producer map. Missing structure becomes `[]`. -/
def rawTraceInputColumns (limb source : Nat) : List Nat :=
  match fixture.traces[limb]? with
  | none => []
  | some trace =>
      match trace.pairs[source]? with
      | none => []
      | some pair => pair.inputColumns

/-- One trace-derived PiRLC consumer column. -/
def rawTraceConsumerColumn (limb source lane : Nat) : Nat :=
  (rawTraceInputColumns limb source).getD lane 0

/-- The independent fixed-parameter specialization of the generic serializer
formula. This definition contains no generated coordinate. -/
def expectedSerializerFieldIndex
    (limb : SerializerLimb) (source lane : Nat) : Nat :=
  8 + source *
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSemantics.sourceFieldCount 13 +
    (9 + 13 *
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Encoding.kVectorFieldCount ringDegree +
      1 + 2 * lane + limbOffset limb)

private def sourceInGroup (group offset : Nat) : Nat :=
  group * 5 + offset

/-- One leaf check. Its two conjuncts intentionally keep serializer-coordinate
refinement separate from producer/consumer columns. -/
private def coordinateValid
    (limb : SerializerLimb) (group offset lane : Nat) : Bool :=
  decide
    (rawProducerSerializerFieldIndex (limbOffset limb)
          (sourceInGroup group offset) lane =
        expectedSerializerFieldIndex limb (sourceInGroup group offset) lane ∧
      rawProducerSourceColumn (limbOffset limb)
          (sourceInGroup group offset) lane =
        rawTraceConsumerColumn (limbOffset limb)
          (sourceInGroup group offset) lane)

/-- Executable bounded certificate over exactly five sources and 54 lanes. -/
private def groupValid (limb : SerializerLimb) (group : Nat) : Bool :=
  (List.range 5).all fun offset =>
    (List.range ringDegree).all fun lane =>
      coordinateValid limb group offset lane

private theorem coordinate_of_groupValid
    {limb : SerializerLimb} {group : Nat}
    (valid : groupValid limb group = true)
    (offset : Fin 5)
    (lane : Fin ringDegree) :
    rawProducerSerializerFieldIndex (limbOffset limb)
        (sourceInGroup group offset.val) lane.val =
      expectedSerializerFieldIndex limb (sourceInGroup group offset.val)
        lane.val ∧
    rawProducerSourceColumn (limbOffset limb)
        (sourceInGroup group offset.val) lane.val =
      rawTraceConsumerColumn (limbOffset limb)
        (sourceInGroup group offset.val) lane.val := by
  have offsetChecked := (List.all_eq_true.mp valid) offset.val
    (List.mem_range.mpr offset.isLt)
  have laneChecked := (List.all_eq_true.mp offsetChecked) lane.val
    (List.mem_range.mpr lane.isLt)
  exact of_decide_eq_true laneChecked

private theorem c0_group0 : groupValid .c0 0 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem c0_group1 : groupValid .c0 1 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem c0_group2 : groupValid .c0 2 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem c1_group0 : groupValid .c1 0 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem c1_group1 : groupValid .c1 1 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem c1_group2 : groupValid .c1 2 = true := by
  set_option maxRecDepth 100000 in
    decide

private theorem fixedBinding_of_groups
    (limb : SerializerLimb)
    (group0 : groupValid limb 0 = true)
    (group1 : groupValid limb 1 = true)
    (group2 : groupValid limb 2 = true)
    (source : Fin 15)
    (lane : Fin ringDegree) :
    rawProducerSerializerFieldIndex (limbOffset limb)
        source.val lane.val =
      expectedSerializerFieldIndex limb source.val lane.val ∧
    rawProducerSourceColumn (limbOffset limb)
        source.val lane.val =
      rawTraceConsumerColumn (limbOffset limb)
        source.val lane.val := by
  by_cases inFirst : source.val < 5
  · have checked := coordinate_of_groupValid group0
      ⟨source.val, inFirst⟩ lane
    have sourceEq : sourceInGroup 0 source.val = source.val := by
      simp [sourceInGroup]
    simpa only [sourceEq] using checked
  · by_cases inSecond : source.val < 10
    · have offsetLt : source.val - 5 < 5 := by omega
      have checked := coordinate_of_groupValid group1
        ⟨source.val - 5, offsetLt⟩ lane
      have sourceEq :
          sourceInGroup 1 (source.val - 5) = source.val := by
        unfold sourceInGroup
        omega
      simpa only [sourceEq] using checked
    · have offsetLt : source.val - 10 < 5 := by omega
      have checked := coordinate_of_groupValid group2
        ⟨source.val - 10, offsetLt⟩ lane
      have sourceEq :
          sourceInGroup 2 (source.val - 10) = source.val := by
        unfold sourceInGroup
        omega
      simpa only [sourceEq] using checked

private theorem fixedBinding
    (limb : SerializerLimb)
    (source : Fin 15)
    (lane : Fin ringDegree) :
    rawProducerSerializerFieldIndex (limbOffset limb)
        source.val lane.val =
      expectedSerializerFieldIndex limb source.val lane.val ∧
    rawProducerSourceColumn (limbOffset limb)
        source.val lane.val =
      rawTraceConsumerColumn (limbOffset limb)
        source.val lane.val := by
  cases limb with
  | c0 =>
      exact fixedBinding_of_groups .c0
        c0_group0 c0_group1 c0_group2 source lane
  | c1 =>
      exact fixedBinding_of_groups .c1
        c1_group0 c1_group1 c1_group2 source lane

/-- Shape-generic statement that every retained raw serializer coordinate is
the coordinate selected by the independent typed serializer. -/
def SerializerIndicesMatch (shape : SemanticShape) : Prop :=
  forall (source : Fin shape.sourceCount) (lane : Fin ringDegree)
      (limb : SerializerLimb),
    rawProducerSerializerFieldIndex (limbOffset limb)
        source.val lane.val =
      yZcolLimbFieldIndex shape source lane limb

/-- Artifact-checked serializer-index refinement for any semantic shape with
the active 15-source/13-matrix dimensions. Other shape fields are irrelevant
to this serializer subtree and remain unconstrained. -/
theorem serializerIndicesMatch
    (shape : SemanticShape)
    (sourceCount : shape.sourceCount = 15)
    (matrixCount : shape.matrixCount = 13) :
    SerializerIndicesMatch shape := by
  intro source lane limb
  have sourceLt : source.val < 15 := by
    simpa [sourceCount] using source.isLt
  have checked := (fixedBinding limb ⟨source.val, sourceLt⟩ lane).1
  rw [checked]
  unfold expectedSerializerFieldIndex yZcolLimbFieldIndex
    yZcolLimbSourceOffset
  rw [matrixCount]

/-- Trace-derived physical consumer for an arbitrary semantic shape. Only the
source indices are shape-dependent; the columns come from the checked tiny
artifact. -/
def traceConsumerColumns (shape : SemanticShape) :
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer.ConsumerColumns
      shape where
  column limb source lane :=
    match limb with
    | .c0 => rawTraceConsumerColumn 0 source.val lane.val
    | .c1 => rawTraceConsumerColumn 1 source.val lane.val

/-- The raw producer source columns match the independently reconstructed
trace consumers. This statement deliberately does not mention serializer
field indices or claim that the producer columns are authoritative. -/
def ProducerColumnsMatchTrace (shape : SemanticShape) : Prop :=
  forall (source : Fin shape.sourceCount) (lane : Fin ringDegree)
      (limb : SerializerLimb),
    rawProducerSourceColumn (limbOffset limb)
        source.val lane.val =
      (traceConsumerColumns shape).column
        (match limb with | .c0 => .c0 | .c1 => .c1) source lane

/-- Artifact-checked producer/consumer equality for every 15-source shape.
The matrix count is intentionally absent: physical column identity does not
depend on serializer dimensions. -/
theorem producerColumnsMatchTrace
    (shape : SemanticShape)
    (sourceCount : shape.sourceCount = 15) :
    ProducerColumnsMatchTrace shape := by
  intro source lane limb
  have sourceLt : source.val < 15 := by
    simpa [sourceCount] using source.isLt
  have checked := (fixedBinding limb ⟨source.val, sourceLt⟩ lane).2
  cases limb <;> exact checked

/-- Explicit authority premise supplied by the PiCCS producer refinement.
This module never manufactures it from artifact self-consistency. -/
def UpstreamProducerColumnsBound
    {shape : SemanticShape}
    (producer : SourceRole shape -> Nat) : Prop :=
  forall (source : Fin shape.sourceCount) (lane : Fin ringDegree)
      (limb : SerializerLimb),
    producer (.yZcolLimb source lane limb) =
      rawProducerSourceColumn (limbOffset limb)
        source.val lane.val

/-- Exact trace consumer matching, conditional on arbitrary upstream
producer-column binding. This is the constructor consumed by the active
PiCCS-to-PiRLC semantic bridge. -/
theorem consumerMatches_of_upstreamProducerColumns
    {shape : SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    {producer : SourceRole shape -> Nat}
    (upstream : UpstreamProducerColumnsBound producer) :
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer.ConsumerMatches
      producer (traceConsumerColumns shape) := by
  have traceMatch := producerColumnsMatchTrace shape sourceCount
  constructor
  · intro source lane
    calc
      (traceConsumerColumns shape).column .c0 source lane =
          rawProducerSourceColumn 0 source.val lane.val :=
        (traceMatch source lane .c0).symm
      _ = producer (.yZcolLimb source lane .c0) :=
        (upstream source lane .c0).symm
  · intro source lane
    calc
      (traceConsumerColumns shape).column .c1 source lane =
          rawProducerSourceColumn 1 source.val lane.val :=
        (traceMatch source lane .c1).symm
      _ = producer (.yZcolLimb source lane .c1) :=
        (upstream source lane .c1).symm

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ProducerBinding
