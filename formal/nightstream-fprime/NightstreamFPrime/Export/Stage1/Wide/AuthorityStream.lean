import NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource
import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransport
import NightstreamFPrime.Export.Stage1.PerApplicationVerifierContextStreaming
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-! The exact typed children of the wide sealed envelope and its native
Poseidon2 identity stream. The stream visits stored rows and compact plans;
it does not expand the matrices or retained coordinates. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AuthorityStream

open NightstreamFPrime.Circuit NightstreamFPrime.Export NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Codec NightstreamFPrime.Layout NightstreamFPrime.Lifecycle

abbrev application := Poseidon2HashChainV1Package.application

/-- The typed children emitted together. `prepare` selects their constructors. -/
structure Parts where
  package : CircuitPackage
  matrix : Layout.MatrixProgram.Program
  application : Stage1.ApplicationPackage.Plan
  transport : AssignmentTransport.Plan

/-- Join already prepared children. The matrix supplies the logical relation
metadata; final terminal metadata refers to that same relation. -/
def ofChildren (physical : CircuitPackage) (matrix : Layout.MatrixProgram.Program)
    (applicationPlan : Stage1.ApplicationPackage.Plan) (transport : AssignmentTransport.Plan) : Parts where
  package := TerminalPackage.install { physical with
    relation := productionCcsRelation matrix.rowCount (RetainedLayout.logicalWidth application) cubeVariables }
  matrix := matrix
  application := applicationPlan
  transport := transport

theorem ofChildren_relation (physical : CircuitPackage) (matrix : Layout.MatrixProgram.Program)
    (applicationPlan : Stage1.ApplicationPackage.Plan) (transport : AssignmentTransport.Plan) :
    (ofChildren physical matrix applicationPlan transport).package.relation =
      productionCcsRelation matrix.rowCount (RetainedLayout.logicalWidth application) cubeVariables := rfl

def prepare (compiled : PiRlcWideSampler.RangePlan.Compiled) : Except String Parts := do
  let (physical, applicationPlan) ← ApplicationPackage.package application
  let matrix ← PhysicalMatrixSource.program application compiled
  let width := RetainedLayout.logicalWidth application
  let transport ← AssignmentTransport.plan application physical.layout.totalColumnCount
  unless transport.coordinateCount = width do throw "wide transport width differs from matrix width"
  return ofChildren physical matrix applicationPlan transport

/-- The pure builder and the emitter share the same final assembly. -/
theorem prepare_uses_ofChildren (compiled : PiRlcWideSampler.RangePlan.Compiled) :
    prepare compiled = (do
      let (physical, applicationPlan) ← ApplicationPackage.package application
      let matrix ← PhysicalMatrixSource.program application compiled
      let transport ← AssignmentTransport.plan application physical.layout.totalColumnCount
      unless transport.coordinateCount = RetainedLayout.logicalWidth application do
        throw "wide transport width differs from matrix width"
      return ofChildren physical matrix applicationPlan transport) := rfl

def nextPreimageRange (parts : Parts) : Layout.MatrixProgram.IndexRange :=
  ⟨parts.application.rowStart + parts.application.rowCount, 5⟩

/-- Proof view of the emitter's seven-child sealed envelope. -/
def sealedValue (parts : Parts) : Value := .array [
  .atom PerApplicationCanonicalPackage.sealedPackageSchema,
  CircuitPackage.format.encode parts.package,
  Layout.MatrixProgram.Program.format.encode parts.matrix,
  Stage1.ApplicationPackage.Plan.format.encode parts.application,
  parts.transport.encode,
  Layout.MatrixProgram.IndexRange.format.encode (nextPreimageRange parts),
  .atom PerApplicationCanonicalPackage.logicalPublicInputCount]

@[specialize push] private def processPackageWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State) (package : CircuitPackage) : State :=
  let state := push state ⟨1, 14⟩
  let state := push state ⟨0, package.schemaVersion⟩
  let state := StreamingIdentity.processValueWith push (Profile.format.encode package.profile) state
  let state := StreamingIdentity.processValueWith push (PoseidonSchedule.format.encode package.poseidon) state
  let state := StreamingIdentity.processValueWith push (PhysicalLayout.format.encode package.layout) state
  let state := StreamingIdentity.processValueWith push (CcsRelation.format.encode package.relation) state
  let state := StreamingIdentity.processValueWith push (PermutationTemplate.format.encode package.permutation) state
  let state := StreamingIdentity.processEncodedListWith push state HashChain.format package.hashChains
  let state := StreamingIdentity.processEncodedListWith push state PermutationInvocation.format package.permutationInvocations
  let state := StreamingIdentity.processEncodedListWith push state CompactRowTemplate.format package.compactRowTemplates
  let state := StreamingIdentity.processEncodedListWith push state CompactRowInvocation.format package.compactRowInvocations
  let state := StreamingIdentity.processEncodedListWith push state WitnessBatch.format package.witnessBatches
  let state := StreamingIdentity.processEncodedListWith push state WitnessInstruction.format package.witnessInstructions
  let state := StreamingIdentity.processEncodedListWith push state SparseRow.format package.assertionRows
  StreamingIdentity.processValueWith push ((option TerminalLayout.format).encode package.terminal) state

private theorem processPackageWith_eq {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State) (package : CircuitPackage) :
    processPackageWith push state package =
      StreamingIdentity.processValueWith push (CircuitPackage.format.encode package) state := by
  unfold processPackageWith
  simp only [StreamingIdentity.processEncodedListWith_eq_processValueWith]
  simp only [CircuitPackage.format, StreamingIdentity.processValueWith,
    List.foldl_cons, List.foldl_nil, List.length_cons, List.length_nil]

@[specialize push] private def processValuesWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State) (values : AssignmentTransport.Values) : State :=
  let state := push state ⟨1, 3⟩
  let state := StreamingIdentity.processValueWith push (PerApplicationAssignmentBlocks.slotKindFormat.encode values.kind) state
  let state := push state ⟨0, values.count⟩
  StreamingIdentity.processEncodedListWith push state AffineRuns.Run.format values.sources

private theorem processValuesWith_eq {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State) (values : AssignmentTransport.Values) :
    processValuesWith push state values =
      StreamingIdentity.processValueWith push values.encode state := by
  unfold processValuesWith
  rw [StreamingIdentity.processEncodedListWith_eq_processValueWith]
  simp only [AssignmentTransport.Values.encode, AffineRuns.format, StreamingIdentity.processValueWith,
    List.foldl_cons, List.foldl_nil, List.length_cons, List.length_nil]

@[specialize push] private def processTransportWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State) (plan : AssignmentTransport.Plan) : State :=
  let state := push state ⟨1, 4⟩
  let state := push state ⟨0, 4⟩
  let state := plan.blocks.foldl (processValuesWith push) (push state ⟨1, plan.blocks.length⟩)
  let state := push state ⟨1, 3⟩
  let state := StreamingIdentity.processEncodedListWith push state
    PerApplicationAssignmentTransport.Phi81FamilyShape.format plan.families
  let state := StreamingIdentity.processEncodedListWith push state AffineRuns.Run.format plan.valueSources
  let state := StreamingIdentity.processEncodedListWith push state AffineRuns.Run.format plan.challengeSources
  StreamingIdentity.processEncodedListWith push state exprFormat plan.outputDigestExpressions

private theorem processTransportWith_eq {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State) (plan : AssignmentTransport.Plan) :
    processTransportWith push state plan = StreamingIdentity.processValueWith push plan.encode state := by
  have values : processValuesWith push = fun state values =>
      StreamingIdentity.processValueWith push values.encode state := by
    funext state values
    exact processValuesWith_eq push state values
  simp only [processTransportWith, values, StreamingIdentity.processEncodedListWith_eq_processValueWith,
    AssignmentTransport.Plan.encode, StreamingIdentity.processValueWith, List.foldl_cons,
    List.foldl_nil, List.length_cons, List.length_nil, List.length_map, List.foldl_map]
  rfl

@[specialize push] def processWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State) (parts : Parts) : State :=
  let state := push state ⟨1, 7⟩
  let state := push state ⟨0, PerApplicationCanonicalPackage.sealedPackageSchema⟩
  let state := processPackageWith push state parts.package
  let state := PerApplicationStreamingIdentity.processMatrixProgramWith push state parts.matrix
  let state := PerApplicationStreamingIdentity.processApplicationPlanWith push state parts.application
  let state := processTransportWith push state parts.transport
  let state := StreamingIdentity.processValueWith push
    (Layout.MatrixProgram.IndexRange.format.encode (nextPreimageRange parts)) state
  push state ⟨0, PerApplicationCanonicalPackage.logicalPublicInputCount⟩

theorem processWith_eq {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State) (parts : Parts) :
    processWith push state parts = StreamingIdentity.processValueWith push (sealedValue parts) state := by
  dsimp only [processWith]
  rw [processPackageWith_eq, PerApplicationStreamingIdentity.processMatrixProgramWith_eq_processValueWith,
    PerApplicationStreamingIdentity.processApplicationPlanWith_eq_processValueWith, processTransportWith_eq]
  simp only [sealedValue, StreamingIdentity.processValueWith,
    List.foldl_cons, List.foldl_nil, List.length_cons, List.length_nil]

def nativeState (parts : Parts) : NativePoseidon2.HashState64 :=
  processWith NativePoseidon2.pushNode64 NativePoseidon2.initialState64 parts

theorem nativeState_denote (parts : Parts) :
    (nativeState parts).denote = StreamingIdentity.processValue (sealedValue parts) StreamingIdentity.initialState := by
  rw [nativeState, processWith_eq]
  have simulation := StreamingIdentity.processValueWith_simulates
    NativePoseidon2.HashState64.denote NativePoseidon2.pushNode64 StreamingIdentity.pushNode
    NativePoseidon2.pushNode64_denote (sealedValue parts) NativePoseidon2.initialState64
  rw [NativePoseidon2.initialState64_denote] at simulation
  exact simulation

def structuralIdentity (parts : Parts) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList (NativePoseidon2.finalize64 (nativeState parts)).denote

/-- Native streaming hashes exactly the complete typed envelope. Digests do
not replace any child or its successful-builder equation. -/
theorem structuralIdentity_recomputed (parts : Parts) :
    structuralIdentity parts = VerifierContext.Digest4.ofList
      (Package.relationIdentifierValue (sealedValue parts)) := by
  rw [structuralIdentity, NativePoseidon2.finalize64_denote, nativeState_denote]
  exact congrArg VerifierContext.Digest4.ofList
    (StreamingIdentity.relationIdentifierValueFast_eq (sealedValue parts))

end NightstreamFPrime.Export.Stage1.Wide.AuthorityStream
