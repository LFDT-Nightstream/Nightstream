import Nightstream.Implementation.Nebula.NIFS.PiDEC.TypedBridgeFor
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutAuthorityLayout
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutSequence

/-!
Contract: exact lifecycle binding for one recursive physical PiRLC family arm.

Owns the equality between both decoded 32-field physical XOut source frames
and the verifier-derived lifecycle frames, the selected-family identity, and
the conversion to the full-state arm consumed by the exact family sequence.

Does not own generated lifecycle rows, Rust assignment conformance, adjacent
family continuity, terminal closure, or Poseidon2 collision resistance.

Assurance tier: artifact-checked conditional adapter for property
`FPRIME-STREAMING-PIRLC-FAMILY-XOUT-PREIMAGE-V1`, Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyLifecycleBridge

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutAuthorityLayout
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutPreimage
open Nightstream.Protocol.FPrime

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen

abbrev RingF := Nightstream.SuperNeo.Concrete.RingF

/-- The four physical Nebula-digest lanes on one side of one family arm. -/
def physicalNebulaDigest
    {setup : InputBindingSetup} {family : Family}
    (physical : AcceptedArm setup family) (side : StateSide) :
    StateOutputAuthorityRows.Digest :=
  fun lane => physical.bodyAssignment
    (xOutPreimageColumn (kindForFamily family) side (28 + lane.val))

/-- Commitment-bundle view of one complete physical family-output function. -/
def outputBundleOfFamilies (outputs : Family → RingF) :
    ProductCommitmentAlgebra.BundleValue :=
  fun component row => outputs (.commitment component row)

/-- Public-input view of one complete physical family-output function. -/
def outputPublicOfFamilies
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <=
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (outputs : Family → RingF) :
    ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits :=
  fun column =>
    outputs (.publicInput
      (Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex
        (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)
        column))
      (Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex
        column)

/-- Evaluation-family view of one complete physical family-output function. -/
def outputEvaluationOfFamilies
    {rowVariables : Nat} (outputs : Family → RingF) :
    ProductPaperAlgebraFor.Evaluation rowVariables :=
  fun matrix lane =>
    ⟨outputs (.evaluation matrix 0) lane,
      outputs (.evaluation matrix 1) lane⟩

/-- Pointwise equality for all 110 families reconstructs the three exact
typed PiRLC output fields. -/
theorem family_outputs_typed_exact
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <=
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    {outputs : Family → RingF}
    {challenges : Source → RingF}
    {inputs : ProductionStreamingPiRlc.TypedInputs rowVariables logicalWidth
      publicFits}
    (outputsExact : ∀ family,
      outputs family =
        ProductionStreamingPiRlc.familyOutput challenges
          (ProductionStreamingPiRlc.typedInputRings inputs) family) :
    outputBundleOfFamilies outputs =
        ProductionStreamingPiRlc.outputBundle challenges inputs ∧
      outputPublicOfFamilies outputs =
        ProductionStreamingPiRlc.outputPublic challenges inputs ∧
      outputEvaluationOfFamilies outputs =
        ProductionStreamingPiRlc.outputEvaluation challenges inputs := by
  constructor
  · funext component row lane
    exact congrFun (outputsExact (.commitment component row)) lane
  constructor
  · funext column
    exact congrFun
      (outputsExact (.publicInput
        (Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex
          (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
            publicFits) column)))
      (Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex
        column)
  · funext matrix lane
    change
      (⟨outputs (.evaluation matrix 0) lane,
        outputs (.evaluation matrix 1) lane⟩ :
          Nightstream.SuperNeo.Concrete.K) =
      (⟨ProductionStreamingPiRlc.familyOutput challenges
          (ProductionStreamingPiRlc.typedInputRings inputs)
          (.evaluation matrix 0) lane,
        ProductionStreamingPiRlc.familyOutput challenges
          (ProductionStreamingPiRlc.typedInputRings inputs)
          (.evaluation matrix 1) lane⟩ : Nightstream.SuperNeo.Concrete.K)
    rw [outputsExact (.evaluation matrix 0),
      outputsExact (.evaluation matrix 1)]

/-- Exact trust-boundary placement for one lifecycle state. The payload
equality uses the canonical-u64 values decoded by the authority rows. The
Nebula equality uses the recomputed digest of the typed present state. -/
structure SideBinding
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount)
    {setup : InputBindingSetup}
    {family : Family}
    (physical : AcceptedArm setup family)
    (state : OuterState Running Fresh Nebula)
    (nebula : Nebula)
    (side : StateSide) : Prop where
  payloadExact :
    StateOutputAuthorityRows.payload
        (authorityLayout (kindForFamily family) side)
        physical.bodyAssignment =
      payload configuration state
  nebulaDigestExact :
    physicalNebulaDigest physical side =
      digestValues (configuration.hashSemantics.nebulaDigest nebula)

namespace SideBinding

/-- The exact physical source frame is the lifecycle frame before hashing.
This equality uses no digest as authority. -/
theorem source_frame_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {setup : InputBindingSetup}
    {family : Family}
    {physical : AcceptedArm setup family}
    {state : OuterState Running Fresh Nebula}
    {nebula : Nebula}
    {side : StateSide}
    (binding : SideBinding configuration physical state nebula side) :
    StateOutputFrameRows.sourceFrame
        (frameLayout (kindForFamily family) side)
        physical.bodyAssignment (physicalNebulaDigest physical side) =
      frame configuration state nebula := by
  have authorityRows := authority_rows_satisfied (kindForFamily family) side
    physical.bodyAssignment physical.suffixSatisfied
  calc
    StateOutputFrameRows.sourceFrame
          (frameLayout (kindForFamily family) side)
          physical.bodyAssignment (physicalNebulaDigest physical side) =
        StateOutputAuthorityRows.fullFrame
          (StateOutputAuthorityRows.payload
            (authorityLayout (kindForFamily family) side)
            physical.bodyAssignment)
          (physicalNebulaDigest physical side) := by
      simpa [authorityLayout] using
        (StateOutputAuthorityRows.sourceFrame_eq_fullFrame
          (authorityLayout_valid (kindForFamily family) side)
          physical.bodyCanonical physical.bodyOne authorityRows
          (physicalNebulaDigest physical side))
    _ = StateOutputAuthorityRows.fullFrame (payload configuration state)
          (digestValues
            (configuration.hashSemantics.nebulaDigest nebula)) := by
      rw [binding.payloadExact, binding.nebulaDigestExact]
    _ = frame configuration state nebula := rfl

end SideBinding

/-- One selected recursive lifecycle invocation bound to one accepted physical
family arm on both sides of the same assignment. -/
structure BoundRecursiveArm
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount)
    (setup : InputBindingSetup)
    (family : Family) where
  physical : AcceptedArm setup family
  recursive : Recursive configuration
  selectedFamily : family =
    ProductionStreamingPiRlcInputBinding.familyAtOrdinal
      recursive.activeArm.selected
  before : SideBinding configuration physical recursive.prior
    recursive.priorNebula .before
  after : SideBinding configuration physical recursive.next
    recursive.nextNebula .after

namespace BoundRecursiveArm

/-- Both physical 32-field source frames equal the verifier-derived lifecycle
frames before either Poseidon2 hash is evaluated. -/
theorem source_frames_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount}
    {setup : InputBindingSetup}
    {family : Family}
    (bound : BoundRecursiveArm configuration setup family) :
    StateOutputFrameRows.sourceFrame
        (frameLayout (kindForFamily family) .before)
        bound.physical.bodyAssignment
        (physicalNebulaDigest bound.physical .before) =
        frame configuration bound.recursive.prior
          bound.recursive.priorNebula /\
      StateOutputFrameRows.sourceFrame
        (frameLayout (kindForFamily family) .after)
        bound.physical.bodyAssignment
        (physicalNebulaDigest bound.physical .after) =
        frame configuration bound.recursive.next
          bound.recursive.nextNebula :=
  ⟨bound.before.source_frame_exact, bound.after.source_frame_exact⟩

/-- The bound recursive invocation supplies the exact full-state arm used by
the 110-family PiRLC sequence. -/
def toAcceptedFullStateArm
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount}
    {setup : InputBindingSetup}
    {family : Family}
    (bound : BoundRecursiveArm configuration setup family) :
    FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateArm
      Running Fresh configuration.hashSemantics configuration.context setup
        family where
  physical := bound.physical
  beforeOuter := bound.recursive.prior
  afterOuter := bound.recursive.next
  beforePinned :=
    (Recursive.recursiveLocalHolds bound.recursive).1.2.2.2.2.2
  afterPinned := Step.next_state_pinned_of_local configuration.hashSemantics
    configuration.stepSemantics .stateful configuration.context
    bound.recursive.prior bound.recursive.next bound.recursive.input
    bound.recursive.proof bound.recursive.localHolds
  beforeSemanticPlaced := by
    intro lane
    have placed := congrArg
      (fun value : StateOutputAuthorityRows.Payload => value.semanticState lane)
      bound.before.payloadExact
    simpa [StateOutputAuthorityRows.payload, authorityLayout, payload,
      digestValues] using placed.symm
  afterSemanticPlaced := by
    intro lane
    have placed := congrArg
      (fun value : StateOutputAuthorityRows.Payload => value.semanticState lane)
      bound.after.payloadExact
    simpa [StateOutputAuthorityRows.payload, authorityLayout, payload,
      digestValues] using placed.symm

end BoundRecursiveArm

/-- Exact lifecycle chain for all 110 verifier-owned physical families.
Adjacent steps share the complete typed outer state and the exact delayed
phase payload. -/
structure BoundRecursiveRun
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount)
    (setup : InputBindingSetup) where
  arm : ∀ ordinal : Fin
      ProductionStreamingPiRlcFamilySequence.exactFamilyCount,
    BoundRecursiveArm configuration setup
      (ProductionStreamingPiRlcInputBinding.familyAtOrdinal ordinal)
  stateContinuous : ∀ (ordinal : Fin
      ProductionStreamingPiRlcFamilySequence.exactFamilyCount)
      (hasNext : ordinal.val + 1 <
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount),
    (arm ordinal).recursive.next =
      (arm ⟨ordinal.val + 1, hasNext⟩).recursive.prior
  payloadContinuous : ∀ (ordinal : Fin
      ProductionStreamingPiRlcFamilySequence.exactFamilyCount)
      (hasNext : ordinal.val + 1 <
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount),
    (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.acceptedPhasePayload
      (arm ordinal).physical).values =
      (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.acceptedPhasePayload
        (arm ⟨ordinal.val + 1, hasNext⟩).physical).values

namespace BoundRecursiveRun

/-- A complete bound lifecycle chain supplies the exact full-state run used
by the family-sequence soundness theorem. -/
def toAcceptedFullStateRun
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount}
    {setup : InputBindingSetup}
    (bound : BoundRecursiveRun configuration setup) :
    FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun
      Running Fresh configuration.hashSemantics configuration.context setup where
  arm ordinal := (bound.arm ordinal).toAcceptedFullStateArm
  continuous := by
    intro ordinal hasNext
    simpa [BoundRecursiveArm.toAcceptedFullStateArm] using
      congrArg
        (XOut.compute configuration.hashSemantics .stateful
          configuration.context)
        (bound.stateContinuous ordinal hasNext)
  payloadContinuous := by
    intro ordinal hasNext
    simpa [BoundRecursiveArm.toAcceptedFullStateArm] using
      bound.payloadContinuous ordinal hasNext

/-- The complete bound lifecycle chain refines the exact semantic PiRLC
family run, or exposes the existing named three-layer binding failure. -/
theorem semantic_run_or_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount}
    {setup : InputBindingSetup}
    (bound : BoundRecursiveRun configuration setup) :
    Nonempty
        (ProductionStreamingPiRlcFamilySequence.AcceptedRun setup
          bound.toAcceptedFullStateRun.inputRings) ∨
      FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity.ContinuityFailure
        configuration.hashSemantics :=
  FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.semanticRun_or_failure
    bound.toAcceptedFullStateRun

/-- Exact family start and finish authority recover all PiCCS-derived inputs,
or expose the existing Module-SIS or continuity failure. -/
theorem start_finish_recovers_inputs_or_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount}
    {setup : InputBindingSetup}
    {authoritative : ProductionStreamingPiRlcInputBinding.InputRings}
    {authoritativeChallenges :
      ProductionStreamingPiRlcInputBinding.Source →
        Nightstream.SuperNeo.Concrete.RingF}
    (bound : BoundRecursiveRun configuration setup)
    (start : FamilyStartRelation
      (bound.toAcceptedFullStateRun.boundaryState 0)
      authoritativeChallenges
      (ProductionStreamingPiRlcInputBindingSetup.concreteBinding setup
        authoritative))
    (finish : ProductionStreamingPiRlcFamilySequence.FamilyFinishRelation
      (bound.toAcceptedFullStateRun.boundaryState
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount)) :
    bound.toAcceptedFullStateRun.inputRings = authoritative ∨
      ProductionStreamingPiRlcInputBindingSetup.ConcreteBindingFailure setup ∨
        FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity.ContinuityFailure
          configuration.hashSemantics :=
  FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.start_finish_recovers_inputs_or_failure
    bound.toAcceptedFullStateRun start finish

/-- In the non-failure branch, a bound lifecycle run produces every exact
monolithic PiRLC family output from the authoritative PiCCS inputs. -/
theorem outputs_exact_or_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount}
    {setup : InputBindingSetup}
    {authoritative : ProductionStreamingPiRlcInputBinding.InputRings}
    {authoritativeChallenges :
      ProductionStreamingPiRlcInputBinding.Source →
        Nightstream.SuperNeo.Concrete.RingF}
    (bound : BoundRecursiveRun configuration setup)
    (start : FamilyStartRelation
      (bound.toAcceptedFullStateRun.boundaryState 0)
      authoritativeChallenges
      (ProductionStreamingPiRlcInputBindingSetup.concreteBinding setup
        authoritative))
    (finish : ProductionStreamingPiRlcFamilySequence.FamilyFinishRelation
      (bound.toAcceptedFullStateRun.boundaryState
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount)) :
    (∀ family,
        bound.toAcceptedFullStateRun.outputs family =
          ProductionStreamingPiRlc.familyOutput
            authoritativeChallenges authoritative family) ∨
      ProductionStreamingPiRlcInputBindingSetup.ConcreteBindingFailure setup ∨
        FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity.ContinuityFailure
          configuration.hashSemantics :=
  FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.outputs_exact_or_failure
    bound.toAcceptedFullStateRun start finish

/-- Accepted physical family outputs are the exact three fields of the
authoritative PiDEC parent, or expose the existing binding or continuity
failure. The parent authority is normally supplied by
`complete_family_run_eq_parent`. -/
theorem piDec_parent_exact_or_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <=
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount}
    {setup : InputBindingSetup}
    {typedInputs : ProductionStreamingPiRlc.TypedInputs rowVariables
      logicalWidth publicFits}
    {challenges : Source → RingF}
    {parentBundle : ProductCommitmentAlgebra.BundleValue}
    {parentPublic :
      ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits}
    {parentEvaluations :
      Array (ProductPaperAlgebraFor.Evaluation rowVariables)}
    (bound : BoundRecursiveRun configuration setup)
    (start : FamilyStartRelation
      (bound.toAcceptedFullStateRun.boundaryState 0) challenges
      (ProductionStreamingPiRlcInputBindingSetup.concreteBinding setup
        (ProductionStreamingPiRlc.typedInputRings typedInputs)))
    (finish : ProductionStreamingPiRlcFamilySequence.FamilyFinishRelation
      (bound.toAcceptedFullStateRun.boundaryState
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount))
    (parentAuthority :
      ProductionStreamingPiRlc.outputBundle challenges typedInputs =
          parentBundle ∧
        ProductionStreamingPiRlc.outputPublic challenges typedInputs =
          parentPublic ∧
        #[ProductionStreamingPiRlc.outputEvaluation challenges typedInputs] =
          parentEvaluations) :
    (outputBundleOfFamilies bound.toAcceptedFullStateRun.outputs =
          parentBundle ∧
        outputPublicOfFamilies bound.toAcceptedFullStateRun.outputs =
          parentPublic ∧
        #[outputEvaluationOfFamilies bound.toAcceptedFullStateRun.outputs] =
          parentEvaluations) ∨
      ProductionStreamingPiRlcInputBindingSetup.ConcreteBindingFailure setup ∨
        FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity.ContinuityFailure
          configuration.hashSemantics := by
  rcases bound.outputs_exact_or_failure
      (authoritative := ProductionStreamingPiRlc.typedInputRings typedInputs)
      (authoritativeChallenges := challenges) start finish with
    outputsExact | failure
  · have typedExact := family_outputs_typed_exact outputsExact
    exact Or.inl
      ⟨typedExact.1.trans parentAuthority.1,
        typedExact.2.1.trans parentAuthority.2.1,
        (congrArg (fun value => #[value]) typedExact.2.2).trans
          parentAuthority.2.2⟩
  · exact Or.inr failure

/-- The accepted physical family outputs are the exact parent consumed by the
operational PiDEC attempt derived from `Key.piCcsOutputs`. Satisfied k16
recomposition rows then establish Section-7.5 PiDEC acceptance. Otherwise,
the theorem exposes the existing binding or continuity failure. -/
theorem authoritative_piDec_parent_exact_or_failure
    (candidate : Nightstream.Protocol.Nebula.ProductionProfileCandidates.Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <=
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (piDecLayout : ProductPiDecRows.Layout)
    (piDecAssignment : Nat -> Nat)
    (piDecCanonical : forall column,
      piDecAssignment column < goldilocksP)
    (constantOne : piDecAssignment 0 = 1)
    (piDecRows : Nightstream.Implementation.R1CS.Satisfies
      (ProductPiDecRows.rows piDecLayout) piDecAssignment)
    (piDecPlacement : ProductPiDecTypedBridgeFor.Placement piDecLayout
      piDecAssignment piDecCanonical
      ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact).piDecAttempt running fresh proof))
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount}
    {setup : InputBindingSetup}
    (bound : BoundRecursiveRun configuration setup)
    (start : FamilyStartRelation
      (bound.toAcceptedFullStateRun.boundaryState 0)
      ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact).piRlcChallenges running fresh proof)
      (ProductionStreamingPiRlcInputBindingSetup.concreteBinding setup
        (authoritativeInputRings candidate statementId config artifact running
          fresh proof)))
    (finish : ProductionStreamingPiRlcFamilySequence.FamilyFinishRelation
      (bound.toAcceptedFullStateRun.boundaryState
        ProductionStreamingPiRlcFamilySequence.exactFamilyCount)) :
    let key := ProductionProductPiCcsTypedBridgeFor.paperKey candidate
      statementId config artifact
    let attempt := key.piDecAttempt running fresh proof
    ((outputBundleOfFamilies bound.toAcceptedFullStateRun.outputs =
            attempt.parent.commitment ∧
          outputPublicOfFamilies bound.toAcceptedFullStateRun.outputs =
            attempt.parent.publicInput ∧
          #[outputEvaluationOfFamilies bound.toAcceptedFullStateRun.outputs] =
            attempt.parent.evaluations) ∧
        Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.Accepted
          (ProductPaperAlgebraFor.piDecAlgebra config)
          (ProductPaperAlgebraFor.evaluationArity config) attempt) ∨
      ProductionStreamingPiRlcInputBindingSetup.ConcreteBindingFailure setup ∨
        FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity.ContinuityFailure
          configuration.hashSemantics := by
  let key := ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
    config artifact
  let attempt := key.piDecAttempt running fresh proof
  let typedInputs := authoritativeInputs candidate statementId config artifact
    running fresh proof
  let challenges := key.piRlcChallenges running fresh proof
  have piDecAccepted :
      Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier.Accepted
        (ProductPaperAlgebraFor.piDecAlgebra config)
        (ProductPaperAlgebraFor.evaluationArity config) attempt := by
    exact ProductPiDecTypedBridgeFor.paperAccepted_of_rows_for_attempt config
      attempt piDecCanonical constantOne piDecRows
      (by simpa [key, attempt] using piDecPlacement) rfl rfl
  have startExact : FamilyStartRelation
      (bound.toAcceptedFullStateRun.boundaryState 0) challenges
      (ProductionStreamingPiRlcInputBindingSetup.concreteBinding setup
        (ProductionStreamingPiRlc.typedInputRings typedInputs)) := by
    simpa [key, typedInputs, challenges, authoritativeInputRings] using start
  have parentAuthority :
      ProductionStreamingPiRlc.outputBundle challenges typedInputs =
          attempt.parent.commitment ∧
        ProductionStreamingPiRlc.outputPublic challenges typedInputs =
          attempt.parent.publicInput ∧
        #[ProductionStreamingPiRlc.outputEvaluation challenges typedInputs] =
          attempt.parent.evaluations := by
    simpa [key, attempt, typedInputs, challenges] using
      (complete_family_run_eq_parent candidate statementId config artifact
        running fresh proof)
  have result := bound.piDec_parent_exact_or_failure
    (typedInputs := typedInputs) (challenges := challenges)
    startExact finish parentAuthority
  have parentResult :
      (outputBundleOfFamilies bound.toAcceptedFullStateRun.outputs =
            attempt.parent.commitment ∧
          outputPublicOfFamilies bound.toAcceptedFullStateRun.outputs =
            attempt.parent.publicInput ∧
          #[outputEvaluationOfFamilies bound.toAcceptedFullStateRun.outputs] =
            attempt.parent.evaluations) ∨
        ProductionStreamingPiRlcInputBindingSetup.ConcreteBindingFailure setup ∨
          FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity.ContinuityFailure
            configuration.hashSemantics := by
    simpa [key, attempt, typedInputs, challenges] using result
  rcases parentResult with parentExact | failure
  · exact Or.inl ⟨parentExact, piDecAccepted⟩
  · exact Or.inr failure

end BoundRecursiveRun

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyLifecycleBridge
