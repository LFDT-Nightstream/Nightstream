import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority

/-!
Fresh public-`X` subfield refinement for the fixed-active source product.

Assurance tier: artifact-checked for the generated column map and model-level
for the conditional equality transport below.

Owns: decoding one 270-coordinate public vector through the exact generated
normalized source columns; an explicit coordinate-value binding leaf; an
explicit direct-dataflow leaf from those values into `input.fresh[0].publicInput`;
and the theorem that these two leaves establish exactly the `publicInput`
field required by `InputAuthority.FreshSourceBound`.

Does not own: either premise's physical rows, the Split-NC polynomial input,
the rest of `InputBound`, fresh structure/stage, the full private witness `Z`,
commitment/Ajtai authority, or row removal.

Emits constraints: none; correspondence theorem only.

No selective disposition is interpreted as a value. In particular, a
`constantOne`, `linearDefinition`, or `traceEliminated` record remains inert
until `CoordinateValueBindings` is proved from concrete row/dataflow
semantics.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.nc.fresh_x.decode` | read the public vector through exact generated normalized columns | computed from artifact | `decodedPublicX` |
| `pi_ccs.nc.fresh_x.values` | each decoded field equals the authoritative fresh assignment | open row/dataflow boundary | `CoordinateValueBindings` |
| `pi_ccs.nc.fresh_x.input` | exact field dataflow reaches the sole fresh source product | direct dataflow boundary | `DirectFreshPublicXDataflow` |
| `pi_ccs.nc.fresh_x.refinement` | both boundaries imply exactly the fresh public-input field | derived | `coordinateValueBindings_and_dataflow_imply_freshPublicInput` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Refinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

universe uCommitment

/-- Exact sole fresh source of `FixedActive.arity`. -/
def freshSource : Fin FixedActive.arity.freshCount :=
  ⟨0, by decide⟩

/-- The typed Phi81 public carrier at five public ring columns is
definitionally the generated 270-coordinate domain. -/
def logicalColumnOfPublic
    {shape : SemanticShape}
    {publicFits :
      ringDegree * FPrimeCarrier270.publicRingColumns <= shape.carrierWidth}
    (column : Fin
      (RelationShape shape FPrimeCarrier270.publicRingColumns
        publicFits).publicWidth) : LogicalColumn :=
  Fin.cast (by rfl) column

/-- Read the public vector from the exact generated normalized source-column
map. This definition does not assert that the source-arm assignment satisfies
any R1CS row. -/
def decodedPublicX
    {shape : SemanticShape}
    {publicFits :
      ringDegree * FPrimeCarrier270.publicRingColumns <= shape.carrierWidth}
    (sourceArmAssignment : Nat -> F) :
    Phi81Relation.PublicInput
      (RelationShape shape FPrimeCarrier270.publicRingColumns publicFits) :=
  fun column =>
    sourceArmAssignment
      (Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Exact.sourceColumn
        (logicalColumnOfPublic column))

/-- Open per-coordinate value leaf. Every generated source column must equal
the public projection of the independent fresh assignment at that coordinate.
The generated artifact alone cannot inhabit this proposition. -/
def CoordinateValueBindings
    {shape : SemanticShape}
    {publicFits :
      ringDegree * FPrimeCarrier270.publicRingColumns <= shape.carrierWidth}
    (data : Data shape)
    (alignment :
      SourceAlignment shape productionGlobalParams FixedActive.arity)
    (sourceArmAssignment : Nat -> F) : Prop :=
  forall column,
    decodedPublicX (shape := shape) (publicFits := publicFits)
        sourceArmAssignment column =
      sourcePublicInput FPrimeCarrier270.publicRingColumns publicFits
        (data.freshAssignment
          (alignment.semanticFreshIndex freshSource)) column

/-- Direct-dataflow leaf from `prior_link.fresh_public_inputs[0]` into the
sole fresh source-product public input. This is equality of field values, not
an equality of digests and not commitment authority. -/
def DirectFreshPublicXDataflow
    {shape : SemanticShape}
    {Commitment : Type uCommitment}
    {publicFits :
      ringDegree * FPrimeCarrier270.publicRingColumns <= shape.carrierWidth}
    (input : SourceProduct shape FPrimeCarrier270.publicRingColumns
      publicFits Commitment productionGlobalParams FixedActive.arity)
    (sourceArmAssignment : Nat -> F) : Prop :=
  forall column,
    (input.fresh freshSource).publicInput column =
      decodedPublicX (shape := shape) (publicFits := publicFits)
        sourceArmAssignment column

/-- Exact `FreshSourceBound.publicInput` subfield for the sole fresh source. -/
def FreshPublicInputFieldBound
    {shape : SemanticShape}
    {Commitment : Type uCommitment}
    {publicFits :
      ringDegree * FPrimeCarrier270.publicRingColumns <= shape.carrierWidth}
    (data : Data shape)
    (alignment :
      SourceAlignment shape productionGlobalParams FixedActive.arity)
    (input : SourceProduct shape FPrimeCarrier270.publicRingColumns
      publicFits Commitment productionGlobalParams FixedActive.arity) : Prop :=
  sourcePublicInput FPrimeCarrier270.publicRingColumns publicFits
      (data.freshAssignment (alignment.semanticFreshIndex freshSource)) =
    (input.fresh freshSource).publicInput

/-- Explicit coordinate bindings plus exact public-`X` dataflow establish
only the fresh source's public-input field. No premise or conclusion mentions
`PublicInputBound`, complete `InputBound`, a private witness, or a
commitment. -/
theorem coordinateValueBindings_and_dataflow_imply_freshPublicInput
    {shape : SemanticShape}
    {Commitment : Type uCommitment}
    {publicFits :
      ringDegree * FPrimeCarrier270.publicRingColumns <= shape.carrierWidth}
    (data : Data shape)
    (alignment :
      SourceAlignment shape productionGlobalParams FixedActive.arity)
    (input : SourceProduct shape FPrimeCarrier270.publicRingColumns
      publicFits Commitment productionGlobalParams FixedActive.arity)
    (sourceArmAssignment : Nat -> F)
    (bindings : CoordinateValueBindings (publicFits := publicFits)
      data alignment sourceArmAssignment)
    (dataflow : DirectFreshPublicXDataflow input sourceArmAssignment) :
    FreshPublicInputFieldBound data alignment input := by
  funext column
  exact (bindings column).symm.trans (dataflow column).symm

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Refinement
