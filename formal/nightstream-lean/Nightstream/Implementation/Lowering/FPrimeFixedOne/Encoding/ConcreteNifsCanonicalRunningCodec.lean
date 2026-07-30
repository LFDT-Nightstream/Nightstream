import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews

/-!
Contract: canonical codecs for the selected fixed-active NIFS running and
fresh carriers.

Owns: one field order for the incoming parent, fourteen running children, and
the fresh payload.  Each statement is encoded as commitment, public input,
point, then evaluation array.  The fresh payload omits the fields that the
verifier constructs.

Does not own: application data, the prover certificate, a relation system,
acceptance, transcript replay, physical columns, Rust, or artifacts.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev Commitment (verifierRows : Nat) :=
  CommitmentValue verifierRows

private abbrev PublicInput
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Phi81Relation.PublicInput
    (RelationShape shape publicRingColumns publicFits)

private abbrev Point
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Phi81Relation.Point
    (RelationShape shape publicRingColumns publicFits)

private abbrev CompletePayload
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Commitment verifierRows ×
    (PublicInput shape publicRingColumns publicFits ×
      (Point shape publicRingColumns publicFits × Array RingK))

noncomputable def completePayloadCodec
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (CompletePayload shape publicRingColumns verifierRows publicFits) :=
  Codec.product
    (commitmentCodec verifierRows)
    (Codec.product
      (publicInputCodec (ringDegree * publicRingColumns))
      (Codec.product
        (pointCodec shape.rowVariables)
        (evaluationsCodec shape.matrixCount)))

def parentPayloadData
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (payload :
      FixedActive.Canonical.ParentPayload
        shape publicRingColumns publicFits verifierRows) :
    CompletePayload shape publicRingColumns verifierRows publicFits :=
  (payload.commitment,
    (payload.publicInput, (payload.point, payload.evaluations)))

theorem parentPayloadData_injective
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth} :
    Function.Injective
      (parentPayloadData
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)) := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def parentPayloadCodec
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (FixedActive.Canonical.ParentPayload
        shape publicRingColumns publicFits verifierRows) :=
  Codec.pullback
    (completePayloadCodec shape publicRingColumns verifierRows publicFits)
    parentPayloadData parentPayloadData_injective

def runningPayloadData
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (payload :
      FixedActive.Canonical.RunningPayload
        shape publicRingColumns publicFits verifierRows) :
    CompletePayload shape publicRingColumns verifierRows publicFits :=
  (payload.commitment,
    (payload.publicInput, (payload.point, payload.evaluations)))

theorem runningPayloadData_injective
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth} :
    Function.Injective
      (runningPayloadData
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)) := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def runningPayloadCodec
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (FixedActive.Canonical.RunningPayload
        shape publicRingColumns publicFits verifierRows) :=
  Codec.pullback
    (completePayloadCodec shape publicRingColumns verifierRows publicFits)
    runningPayloadData runningPayloadData_injective

private abbrev FreshData
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Commitment verifierRows ×
    PublicInput shape publicRingColumns publicFits

def freshPayloadData
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (payload :
      FixedActive.Canonical.FreshPayload
        shape publicRingColumns publicFits verifierRows) :
    FreshData shape publicRingColumns verifierRows publicFits :=
  (payload.commitment, payload.publicInput)

theorem freshPayloadData_injective
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth} :
    Function.Injective
      (freshPayloadData
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)) := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

/-- Canonical fresh codec: commitment, then the complete 270-coordinate
public input. -/
noncomputable def freshCodec
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (SelectedFresh shape publicRingColumns publicFits verifierRows) :=
  Codec.pullback
    (Codec.product
      (commitmentCodec verifierRows)
      (publicInputCodec (ringDegree * publicRingColumns)))
    freshPayloadData freshPayloadData_injective

private abbrev RunningData
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  FixedActive.Canonical.ParentPayload
      shape publicRingColumns publicFits verifierRows ×
    (Fin productionGlobalParams.k →
      FixedActive.Canonical.RunningPayload
        shape publicRingColumns publicFits verifierRows)

def runningData
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    RunningData shape publicRingColumns verifierRows publicFits :=
  (running.parent, running.children)

theorem runningData_injective
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth} :
    Function.Injective
      (runningData
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)) := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

/-- Canonical running codec: checked parent first, then fourteen children in
increasing child index order. -/
noncomputable def runningCodec
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    Codec
      (SelectedRunning shape publicRingColumns publicFits verifierRows) :=
  Codec.pullback
    (Codec.product
      (parentPayloadCodec shape publicRingColumns verifierRows publicFits)
      (Codec.finFunction productionGlobalParams.k
        (runningPayloadCodec
          shape publicRingColumns verifierRows publicFits)))
    runningData runningData_injective

theorem freshCodec_admissible
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows) :
    (freshCodec shape publicRingColumns verifierRows publicFits).Admissible
      fresh := by
  constructor
  · exact commitmentCodec_admissible fresh.commitment
  · exact publicInputCodec_admissible fresh.publicInput

theorem runningCodec_admissible_iff
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    (runningCodec shape publicRingColumns verifierRows publicFits).Admissible
        running ↔
      running.parent.evaluations.size = shape.matrixCount ∧
        ∀ child,
          (running.children child).evaluations.size = shape.matrixCount := by
  constructor
  · intro admissible
    exact ⟨admissible.1.2.2.2.1,
      fun child => (admissible.2 child).2.2.2.1⟩
  · intro sizes
    constructor
    · exact ⟨commitmentCodec_admissible running.parent.commitment,
        publicInputCodec_admissible running.parent.publicInput,
        pointCodec_admissible running.parent.point,
        evaluationsCodec_admissible running.parent.evaluations sizes.1⟩
    · intro child
      exact ⟨commitmentCodec_admissible
            (running.children child).commitment,
        publicInputCodec_admissible (running.children child).publicInput,
        pointCodec_admissible (running.children child).point,
        evaluationsCodec_admissible
          (running.children child).evaluations (sizes.2 child)⟩

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
