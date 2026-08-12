import Nightstream.HyperNova.NIVCCompatibility
import Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec
import Nightstream.Implementation.NebulaV2.ProductionVerifierArtifactFor

/-!
Contract: finite assignment-capacity evidence for one generated Nebula V2
F-prime artifact.

The recursive payload is not caller-selected. It is the complete finite
source assignment of the generated F-prime program. Its canonical codec uses
one Goldilocks coordinate per source value. The fixed overhead is the exact
difference between that source width and the low-norm carrier width.

The generated-artifact schema proves that its selector-gated row program and
its carrier fit the same Boolean cube. A release must instantiate that schema
with the complete generated F-prime rows, prove that those rows implement all
required application and recursive-verifier operations, and bind the result
to the verifier key. This module does not create such an instance and does not
derive semantic completeness from capacity.

Assurance tier: generated-artifact capacity theorem.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.RecursiveSizeClosure

open Nightstream.HyperNova.NIVCCompatibility
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec
open Nightstream.Implementation.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

universe uParameters uCircuit uInput uOutput uRunningStructure uFreshStructure
  uRunningInstance uRunningWitness uFreshInstance uFreshWitness

/-- Exact recursive payload for the generated relation. It contains every
source coordinate consumed by the row program. This conservative choice also
includes the public and lane prefix, so no smaller caller-selected advice type
can be used to pass the closure check. -/
abbrev Payload
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths) :=
  ProductionFreshLowNormEncoding.SourceAssignment artifact.privateWidth

/-- Canonical fixed-width codec for the exact source assignment. -/
noncomputable def payloadGoldCodec
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths) :
    Nightstream.Implementation.Lowering.Goldilocks.Codec
      (Payload artifact) :=
  Nightstream.Implementation.Lowering.Goldilocks.Codec.finFunction
    (ProductionFreshLowNormEncoding.sourceWidth artifact.privateWidth)
    fieldCodec

theorem payloadGoldCodec_admissible
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths)
    (payload : Payload artifact) :
    (payloadGoldCodec artifact).Admissible payload := by
  intro _index
  trivial

/-- HyperNova-facing canonical codec for the exact generated payload. -/
noncomputable def payloadCodec
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths) :
    Nightstream.HyperNova.NIVCCompatibility.Codec (Payload artifact) F :=
  toTotalNivcCodec (payloadGoldCodec artifact)

theorem payloadCodec_canonical
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths) :
    (payloadCodec artifact).Canonical :=
  toTotalNivcCodec_canonical (payloadGoldCodec artifact)
    (payloadGoldCodec_admissible artifact)

theorem sourceWidth_le_logicalWidth (privateWidth : Nat) :
    ProductionFreshLowNormEncoding.sourceWidth privateWidth <=
      ProductionFreshLowNormEncoding.logicalWidth privateWidth := by
  have sourceFitsPayload :
      ProductionFreshLowNormEncoding.sourceWidth privateWidth <=
        ProductionFreshLowNormEncoding.payloadWidth privateWidth := by
    simp only [ProductionFreshLowNormEncoding.sourceWidth,
      ProductionFreshLowNormEncoding.payloadWidth,
      Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.digitCount]
    omega
  exact sourceFitsPayload.trans
    (ProductionFreshLowNormEncoding.payloadWidth_le_logicalWidth privateWidth)

theorem payloadCodec_length
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths)
    (payload : Payload artifact) :
    ((payloadCodec artifact).encode payload).length =
      ProductionFreshLowNormEncoding.sourceWidth artifact.privateWidth := by
  change ((payloadGoldCodec artifact).encode payload).length = _
  calc
    ((payloadGoldCodec artifact).encode payload).length =
        (payloadGoldCodec artifact).width :=
      (payloadGoldCodec artifact).encode_length payload
    _ = ProductionFreshLowNormEncoding.sourceWidth artifact.privateWidth := by
      simp only [payloadGoldCodec,
        Nightstream.Implementation.Lowering.Goldilocks.Codec.finFunction,
        Nightstream.Implementation.Lowering.Goldilocks.Codec.ofInjectiveEncoding,
        fieldCodec, Nat.mul_one]

/-- Exact number of low-norm carrier coordinates that are not source
coordinates. This includes digit expansion and final whole-ring padding. -/
def encodingOverhead
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths) :
    Nat :=
  ProductionFreshLowNormEncoding.logicalWidth artifact.privateWidth -
    ProductionFreshLowNormEncoding.sourceWidth artifact.privateWidth

/-- Concrete Property-6 object for the exact generated source payload. The
capacity does not depend on runtime parameters. -/
noncomputable def closure
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths)
    (Parameters : Type uParameters) :
    Nightstream.HyperNova.NIVCCompatibility.RecursiveSizeClosure
      F Parameters (Payload artifact) where
  payloadCodec := payloadCodec artifact
  payloadWords := ProductionFreshLowNormEncoding.sourceWidth
    artifact.privateWidth
  fixedOverheadWords := encodingOverhead artifact
  internalWireWords := 0
  privateAssignmentCapacity := fun _ =>
    ProductionFreshLowNormEncoding.logicalWidth artifact.privateWidth

theorem requiredWords_exact
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths)
    (Parameters : Type uParameters) :
    (closure artifact Parameters).requiredWords =
      ProductionFreshLowNormEncoding.logicalWidth artifact.privateWidth := by
  rw [Nightstream.HyperNova.NIVCCompatibility.RecursiveSizeClosure.requiredWords]
  simp only [closure, encodingOverhead, Nat.add_zero]
  exact Nat.add_sub_of_le (sourceWidth_le_logicalWidth artifact.privateWidth)

/-- Finite source and lowered-assignment capacity owned by one artifact.

This structure proves scope and size. It does not prove that the row list
contains every operation required by the F-prime specification. That separate
generated-row refinement theorem is required before this capacity evidence can
be used in a full Definition-12 instantiation. -/
structure FiniteArtifactCapacity
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths) :
    Prop where
  payloadCanonical : (payloadCodec artifact).Canonical
  payloadLength : forall payload : Payload artifact,
    ((payloadCodec artifact).encode payload).length =
      ProductionFreshLowNormEncoding.sourceWidth artifact.privateWidth
  sourceColumns :
    ProductionFreshRelationCompilerFor.NumericBridge.RowsBelow
      (ProductionFreshLowNormEncoding.sourceWidth artifact.privateWidth)
      artifact.fPrimeProgram.rows
  loweredColumns :
    ProductionFreshRelationCompilerFor.NumericBridge.RowsBelow
      (ProductionFreshLowNormEncoding.logicalWidth artifact.privateWidth)
      (Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler.loweredRows
        (ProductionFreshLinearSubstitution.layout artifact.privateWidth)
        artifact.fPrimeProgram.rows)
  exactRowDomain :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.ExactRowDomain
      artifact.fPrimeProgram.rows.length
      artifact.dimensions.relationRowVariables
  carrierFits :
    ProductionFreshLowNormEncoding.logicalWidth artifact.privateWidth <=
      2 ^ artifact.dimensions.relationRowVariables

/-- Every inhabitant of the generated-artifact schema has finite, scoped
source and lowered assignments. A final release still needs a concrete,
complete generated artifact; this theorem does not manufacture one. -/
theorem finiteArtifactCapacity
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths) :
    FiniteArtifactCapacity artifact where
  payloadCanonical := payloadCodec_canonical artifact
  payloadLength := payloadCodec_length artifact
  sourceColumns := artifact.compiler.sourceColumns
  loweredColumns := artifact.compiler.loweredColumns
  exactRowDomain := artifact.compiler.rowDomain
  carrierFits := artifact.compiler.carrierFits

/-- Exact compiler capacities needed to interpret the concrete closure as
corrected HyperNova Definition 12 Property 6. This predicate contains only
capacity equalities. It does not contain the closure conclusion. -/
def MatchesCapacities
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths)
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    {semantics : Semantics Parameters Circuit Input (Payload artifact) Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness}
    {encoding : Encoding Circuit Input (Payload artifact) Output
      RunningStructure FreshStructure FreshInstance FreshWitness}
    (layout : CompilerLayout semantics encoding) : Prop :=
  (forall parameters,
    layout.rowCapacity parameters =
      2 ^ artifact.dimensions.relationRowVariables) /\
  (forall parameters,
    layout.columnCapacity parameters =
      ProductionFreshLowNormEncoding.logicalWidth artifact.privateWidth)

/-- The exact source-assignment codec and matching finite capacities satisfy
the numeric `RecursiveSizeClosure.Holds` contract.

This theorem does not show that `semantics` and `encoding` are the complete
F-prime compiler. A full Definition-12 result must separately prove
`NPComplete`, compiler-layout correctness, and generated-row refinement for
those exact objects. -/
theorem capacityHoldsForMatchingLayout
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    {artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths}
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    {semantics : Semantics Parameters Circuit Input (Payload artifact) Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness}
    {encoding : Encoding Circuit Input (Payload artifact) Output
      RunningStructure FreshStructure FreshInstance FreshWitness}
    (layout : CompilerLayout semantics encoding)
    (capacityMatches : MatchesCapacities artifact layout) :
    (closure artifact Parameters).Holds layout := by
  refine ⟨payloadCodec_canonical artifact, payloadCodec_length artifact, ?_⟩
  intro parameters
  rw [requiredWords_exact artifact Parameters]
  constructor
  · rfl
  · exact (capacityMatches.2 parameters).symm.le

/-- The artifact row list, source payload, low-norm carrier, and assignment
columns fit one finite Boolean cube. Semantic row completeness is separate. -/
theorem completeCompilerFit
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    {artifact : ProductionVerifierArtifactFor.Artifact candidate baseWidths}
    {Parameters : Type uParameters}
    {Circuit : Type uCircuit}
    {Input : Type uInput}
    {Output : Type uOutput}
    {RunningStructure : Type uRunningStructure}
    {FreshStructure : Type uFreshStructure}
    {RunningInstance : Type uRunningInstance}
    {RunningWitness : Type uRunningWitness}
    {FreshInstance : Type uFreshInstance}
    {FreshWitness : Type uFreshWitness}
    {semantics : Semantics Parameters Circuit Input (Payload artifact) Output
      RunningStructure FreshStructure RunningInstance RunningWitness
      FreshInstance FreshWitness}
    {encoding : Encoding Circuit Input (Payload artifact) Output
      RunningStructure FreshStructure FreshInstance FreshWitness}
    (layout : CompilerLayout semantics encoding)
    (capacityMatches : MatchesCapacities artifact layout)
    (parameters : Parameters) :
    artifact.fPrimeProgram.rows.length <= layout.rowCapacity parameters /\
      (closure artifact Parameters).requiredWords <=
        (closure artifact Parameters).privateAssignmentCapacity parameters /\
      (closure artifact Parameters).privateAssignmentCapacity parameters <=
        layout.columnCapacity parameters /\
      layout.columnCapacity parameters <= layout.rowCapacity parameters := by
  have rowsFit := artifact.augmentedRowsFit
  have carrierFit := artifact.compiler.carrierFits
  rw [capacityMatches.1 parameters, capacityMatches.2 parameters,
    requiredWords_exact artifact Parameters]
  exact ⟨rowsFit, le_rfl, le_rfl, carrierFit⟩

/-! ## Necessity countermodels -/

/-- The old, incomplete check that considers only relation rows. -/
def RowFitOnly (rows rowVariables : Nat) : Prop :=
  rows <= 2 ^ rowVariables

/-- All finite capacity inequalities required for one rectangular recursive
assignment. -/
def FullCapacityFit
    (rows rowVariables required privateColumns compilerColumns : Nat) : Prop :=
  RowFitOnly rows rowVariables /\
  required <= privateColumns /\
  privateColumns <= compilerColumns /\
  compilerColumns <= 2 ^ rowVariables

/-- A one-row relation fits a one-variable row cube, but a two-word payload
does not fit one private column. Thus row fit cannot imply Property 6. -/
theorem rowFitOnly_does_not_imply_fullCapacity :
    RowFitOnly 1 1 /\
      ¬ FullCapacityFit 1 1 2 1 1 := by
  constructor <;> norm_num [RowFitOnly, FullCapacityFit]

/-- Required-row containment is a semantic property that no capacity theorem
can derive from equal lengths or a common finite cube. -/
def RequiredRowsPresent
    (required generated : List Nightstream.Implementation.R1CS.Row) : Prop :=
  required.Sublist generated

/-- Two one-row programs have identical finite capacities, but the generated
program can omit the required rejecting equation. This is the exact reason
`FiniteArtifactCapacity` is not a generated-row completeness theorem. -/
theorem finiteCapacity_does_not_imply_requiredRowsPresent :
    RowFitOnly 1 1 /\
      [ProductionRecursiveCoreManifestFor.rejectingConstantRow].length =
        [ProductionRecursiveCoreManifestFor.zeroRow].length /\
      ¬ RequiredRowsPresent
        [ProductionRecursiveCoreManifestFor.rejectingConstantRow]
        [ProductionRecursiveCoreManifestFor.zeroRow] := by
  simp [RowFitOnly, RequiredRowsPresent,
    ProductionRecursiveCoreManifestFor.rejectingConstantRow,
    ProductionRecursiveCoreManifestFor.zeroRow]

end Nightstream.Implementation.NebulaV2.RecursiveSizeClosure
