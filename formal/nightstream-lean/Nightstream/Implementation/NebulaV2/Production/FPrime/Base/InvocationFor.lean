import Nightstream.Implementation.NebulaV2.Memory.Carry.InitialRows
import Nightstream.Implementation.NebulaV2.Memory.Transition.OpenSegment
import Nightstream.Implementation.NebulaV2.Production.FPrime.Fresh.ClaimProducerFor
import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.DefaultRunningFor
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout

/-!
Contract: exact base F-prime invocation at the generated relation exponent.

The base invocation installs verifier-authoritative initial state, evaluates
the first application batch, opens the first segment, and produces claim zero.
No prior claim, NIFS proof, verifier result, or consumed receipt appears in
the base type.

Assurance tier: exponent-indexed implementation model.

Does not own generated base rows, root computation, application lowering,
compiled-relation refinement, recursive-size closure, Rust, or cryptography.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperBaseInvocationFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ApplicationBatch
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.Protocol.NebulaV2.WasmStatement
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

local instance concreteKOne : One K := ⟨K.one⟩

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

abbrev FreshAssignment
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.Assignment rowVariables logicalWidth publicFits

/-- HyperNova Construction 2's verifier-owned universal default. -/
def defaultRunning
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth} :
    ProductNifsCodec.RunningFor rowVariables
      (FullShape rowVariables logicalWidth publicFits) :=
  ProductionPaperDefaultRunningFor.value rowVariables logicalWidth publicFits

def initialClosed (initialMemoryRoot : Digest.Value) : ClosedCarry Digest.Value :=
  { segmentIndex := 0
    globalTimestamp := 0
    memoryRoot := initialMemoryRoot }

theorem initialClosed_canOpen (initialMemoryRoot : Digest.Value) :
    (initialClosed initialMemoryRoot).CanOpen := by
  constructor
  · norm_num [initialClosed, Lifecycle.maximumSegments]
  · norm_num [initialClosed, timestampLimit, timestampBits]

/-- Canonical Construction-2 state before the base invocation.  This value is
the authority-bearing base analogue of the prior successor parsed by a
recursive invocation.  Both memory carries are the unique closed chain-start
carry, and the running product is the verifier-owned universal default. -/
noncomputable def initialState
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (initialMemoryRoot : Digest.Value) :
    ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits) :=
  { augmentedInvocationIndex := 0
    realApplicationRowCount := 0
    initialApplicationState :=
      WasmStateEncoding.encode statement.base.initialApplicationState
    applicationState :=
      WasmStateEncoding.encode statement.base.initialApplicationState
    running := defaultRunning
    initialMemoryCarry := InitialMemoryCarryRows.expectedValue headers
      initialMemoryRoot
    memoryCarry := InitialMemoryCarryRows.expectedValue headers
      initialMemoryRoot }

/-- The canonical base input is in the exact domain of the state transcript.
This proof is needed when a concrete Poseidon2 reduction distinguishes a
changed base input from a transcript collision. -/
theorem initialState_canonical
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (initialMemoryRoot : Digest.Value) :
    (initialState (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement
      initialMemoryRoot).Canonical headers := by
  refine
    { invocationIndex := ?_
      realApplicationRowCount := by simp [initialState]
      initialApplicationState := ?_
      applicationState := ?_
      initialMemoryCarry :=
        InitialMemoryCarryRows.expectedValue_canonical headers
          initialMemoryRoot
      memoryCarry :=
        InitialMemoryCarryRows.expectedValue_canonical headers
          initialMemoryRoot }
  · cases candidate <;>
      norm_num [initialState, maximumAugmentedInvocations, maximumClaims,
        ProductionProfileCandidates.maximumSegments, claimsPerSegment,
        stepsPerSegment, checkedStepsPerFreshClaim]
  · exact (WasmStateEncoding.canonical_encode_iff
      statement.base.initialApplicationState).2
      statement.initialApplicationStateValid
  · exact (WasmStateEncoding.canonical_encode_iff
      statement.base.initialApplicationState).2
      statement.initialApplicationStateValid

structure Opening where
  initialMemoryRoot : Digest.Value
  authority : MemoryOpenSegment.Authority
  precommit : Roots Digest.Value
  activeAccessCount : Nat
  activeCountInRange : activeAccessCount < operationCountLimit
  endTimestampInRange : activeAccessCount < timestampLimit

namespace Opening

theorem initialEndTimestampInRange (opening : Opening) :
    (initialClosed opening.initialMemoryRoot).globalTimestamp +
        opening.activeAccessCount < timestampLimit := by
  simpa [initialClosed] using opening.endTimestampInRange

/-- Candidate-profile base opening. Its challenge frame contains the exact
candidate version and checked-step factor. -/
def activeFor (candidate : Id) (headers : ChainHeaders Digest.Value)
    (opening : Opening) :
    ActiveCarry Digest.Value (ProductState.Challenges K)
      (ProductState.State K) :=
  { segmentIndex := 0
    stepIndex := ⟨0, by decide⟩
    globalTimestamp := 0
    segmentStartTimestamp := 0
    segmentActiveAccessCount := opening.activeAccessCount
    segmentEndTimestamp := opening.activeAccessCount
    challenge := MemoryOpenSegment.deriveFor (identity candidate)
      opening.authority (initialClosed opening.initialMemoryRoot)
      opening.precommit opening.activeAccessCount
    products := ProductState.one
    dPre := opening.precommit
    dSeen := headers.roots
    memoryRoot := opening.initialMemoryRoot }

theorem open_exact_for
    (candidate : Id) (headers : ChainHeaders Digest.Value)
    (opening : Opening) :
    MemoryOpenSegment.openCarryFor (identity candidate) opening.authority
        headers opening.precommit opening.activeAccessCount
        (initialClosed opening.initialMemoryRoot)
        (initialClosed_canOpen opening.initialMemoryRoot)
        opening.activeCountInRange opening.initialEndTimestampInRange =
      .active (opening.activeFor candidate headers) := by
  simpa [activeFor, initialClosed] using
    MemoryOpenSegment.open_exact_for (identity candidate) opening.authority
      headers opening.precommit opening.activeAccessCount
      (initialClosed opening.initialMemoryRoot)
      (initialClosed_canOpen opening.initialMemoryRoot)
      opening.activeCountInRange opening.initialEndTimestampInRange

theorem activeWire_canonical_for
    (candidate : Id) (headers : ChainHeaders Digest.Value)
    (opening : Opening) :
    MemoryCarryCodec.Value.Canonical headers
      (CarryEncoding.encodeActive (opening.activeFor candidate headers)) := by
  refine
    { segmentIndex := by
        simp [CarryEncoding.encodeActive, activeFor,
          MemoryWireGeometry.segmentIndexBits]
      stepIndex := by exact (opening.activeFor candidate headers).stepIndex.isLt
      globalTimestamp := by
        simp [CarryEncoding.encodeActive, activeFor,
          MemoryWireGeometry.timestampBits]
      segmentStartTimestamp := by
        simp [CarryEncoding.encodeActive, activeFor,
          MemoryWireGeometry.timestampBits]
      segmentActiveAccessCount := ?_
      segmentEndTimestamp := ?_
      closedFields := ?_ }
  · simpa [CarryEncoding.encodeActive, activeFor,
      MemoryWireGeometry.segmentActiveAccessCountBits,
      operationCountLimit, operationCountBits] using opening.activeCountInRange
  · simpa [CarryEncoding.encodeActive, activeFor,
      MemoryWireGeometry.timestampBits, timestampLimit, timestampBits] using
      opening.endTimestampInRange
  · intro phase
    cases phase

end Opening

noncomputable def state
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (opening : Opening)
    {machine : Machine Program} {after : AppStateVector}
    (batch : Batch candidate machine statement.base.program
      statement.base.initialApplicationState after) :
    ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits) :=
  { augmentedInvocationIndex := 1
    realApplicationRowCount := realRowCount batch.rows
    initialApplicationState :=
      WasmStateEncoding.encode statement.base.initialApplicationState
    applicationState := WasmStateEncoding.encode after
    running := defaultRunning
    initialMemoryCarry := InitialMemoryCarryRows.expectedValue headers
      opening.initialMemoryRoot
    memoryCarry := CarryEncoding.encodeActive
      (opening.activeFor candidate headers) }

/-- Convert the canonical four-lane state-transcript output to the digest type
used by the memory transcript.  This conversion changes no lane or field
representative. -/
def digestValue
    (digest : ProductionSuccessorStateBinding.CanonicalDigest) : Digest.Value :=
  { lanes := digest }

/-- The only base memory-challenge authority admitted by a candidate profile.
It binds the
same verifier-owned statement identity as recursive invocations, the exact
canonical base input state, and the challenge-independent prefix of the base
successor after its application batch.  The successor prefix omits both
memory carries, so this definition has no challenge/output cycle. -/
noncomputable def challengeAuthority
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (statementId : ProductPoseidon2.StatementId)
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (opening : Opening)
    {machine : Machine Program} {after : AppStateVector}
    (batch : Batch candidate machine statement.base.program
      statement.base.initialApplicationState after) :
    MemoryOpenSegment.Authority :=
  MemoryOpenSegment.Authority.ofIdentityAndState statement.base.identity
    (digestValue
      (ProductionSuccessorStateBinding.outputDigest statementId
        (initialState (rowVariables := rowVariables)
          (logicalWidth := logicalWidth) (publicFits := publicFits) candidate
          headers statement opening.initialMemoryRoot)))
    (digestValue
      (ProductionSuccessorStateBinding.preCarryDigest statementId
        (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
          (publicFits := publicFits) candidate headers statement opening
          batch).preCarry))

/-- The base accumulator-authority input is independent of the opening
authority, precommitment roots, and derived memory challenges. -/
theorem state_preCarry_exact
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (opening : Opening)
    {machine : Machine Program} {after : AppStateVector}
    (batch : Batch candidate machine statement.base.program
      statement.base.initialApplicationState after) :
    (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement opening
      batch).preCarry =
      { augmentedInvocationIndex := 1
        realApplicationRowCount := realRowCount batch.rows
        initialApplicationState :=
          WasmStateEncoding.encode statement.base.initialApplicationState
        applicationState := WasmStateEncoding.encode after
        running := defaultRunning } := by
  rfl

theorem state_canonical
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (opening : Opening)
    {machine : Machine Program} {after : AppStateVector}
    (batch : Batch candidate machine statement.base.program
      statement.base.initialApplicationState after) :
    (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement opening batch).Canonical
      headers := by
  refine
    { invocationIndex := by
        cases candidate <;>
          norm_num [state, maximumAugmentedInvocations, maximumClaims,
            ProductionProfileCandidates.maximumSegments, claimsPerSegment,
            stepsPerSegment, checkedStepsPerFreshClaim]
      realApplicationRowCount := ?_
      initialApplicationState := ?_
      applicationState := ?_
      initialMemoryCarry :=
        InitialMemoryCarryRows.expectedValue_canonical headers
          opening.initialMemoryRoot
      memoryCarry := opening.activeWire_canonical_for candidate headers }
  · exact batch.realRowCount_le_rowsPerFreshClaim.trans_lt (by
      cases candidate <;> decide)
  · exact (WasmStateEncoding.canonical_encode_iff
      statement.base.initialApplicationState).2
      statement.initialApplicationStateValid
  · exact (WasmStateEncoding.canonical_encode_iff after).2
      (batch.after_valid statement.initialApplicationStateValid)

noncomputable def claim
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (opening : Opening)
    {machine : Machine Program} {after : AppStateVector}
    (batch : Batch candidate machine statement.base.program
      statement.base.initialApplicationState after)
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :=
  ProductionFreshClaimProducerFor.value candidate statementId config
    (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement opening batch) memory
    assignment

structure Evidence
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact)
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (opening : Opening) (machine : Machine Program) (after : AppStateVector)
    (batch : Batch candidate machine statement.base.program
      statement.base.initialApplicationState after)
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {sourceAssignment : Nat -> Nat}
    (memoryResult : ProductionMemoryCheckedBatchRows.Result layout
      sourceAssignment headers)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) : Prop where
  statementCanonical :
    (PublicImage.ofStatement statement).DecodesFor (identity candidate) statement
  challengeAuthorityExact : opening.authority =
    challengeAuthority (rowVariables := rowVariables)
      (logicalWidth := logicalWidth) (publicFits := publicFits) candidate
      statementId headers statement opening batch
  memoryStartsAt : memoryResult.semantic 0 =
    .active (opening.activeFor candidate headers)
  applicationMatched :
    ProductionApplicationBatchBridge.Matches memoryResult batch
  freshRelation : ProductionFreshClaimProducerFor.FreshRelationWitnessForRows
    statementId config artifact relationAuthority
    (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement opening batch)
    memoryResult.suffixBatch assignment sourceAssignment

structure ExactInvocation
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact)
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (opening : Opening) (machine : Machine Program) (after : AppStateVector)
    (batch : Batch candidate machine statement.base.program
      statement.base.initialApplicationState after)
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {sourceAssignment : Nat -> Nat}
    (memoryResult : ProductionMemoryCheckedBatchRows.Result layout
      sourceAssignment headers)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (evidence : Evidence candidate statementId config artifact relationAuthority headers
      statement opening machine after batch memoryResult
      assignment) : Prop where
  baseInvocationIndex :
    (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement opening batch
    ).augmentedInvocationIndex = 1
  defaultRunningExact :
    (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement opening batch).running =
      defaultRunning (rowVariables := rowVariables) (logicalWidth := logicalWidth)
        (publicFits := publicFits)
  initialApplicationExact :
    (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement opening batch
    ).initialApplicationState =
      WasmStateEncoding.encode statement.base.initialApplicationState
  initialMemoryExact :
    (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement opening batch
    ).initialMemoryCarry =
      InitialMemoryCarryRows.expectedValue headers opening.initialMemoryRoot
  challengeAuthorityExact : opening.authority =
    challengeAuthority (rowVariables := rowVariables)
      (logicalWidth := logicalWidth) (publicFits := publicFits) candidate
      statementId headers statement opening batch
  applicationRun : Runs machine statement.base.program
    statement.base.initialApplicationState batch.rows after
      (realRowCount batch.rows)
  memoryOpen : MemoryOpenSegment.openCarryFor (identity candidate)
      opening.authority headers
      opening.precommit opening.activeAccessCount
      (initialClosed opening.initialMemoryRoot)
      (initialClosed_canOpen opening.initialMemoryRoot)
      opening.activeCountInRange opening.initialEndTimestampInRange =
    .active (opening.activeFor candidate headers)
  delayedCurrentMemory :
    memoryResult.semantic 0 = .active (opening.activeFor candidate headers)
  currentPortsExact : ApplicationBatch.accesses batch.rows =
    ProductionApplicationBatchBridge.memoryAccesses memoryResult
  stateCanonical :
    (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement opening batch).Canonical
      headers
  claimCanonical :
    (claim candidate statementId config headers statement opening batch
      memoryResult.suffixBatch assignment).Canonical
  claimMemoryExact :
    (claim candidate statementId config headers statement opening batch
      memoryResult.suffixBatch assignment).memory =
      memoryResult.suffixBatch
  claimMemoryBound :
    (claim candidate statementId config headers statement opening batch
      memoryResult.suffixBatch assignment).MemoryBound
  freshRelationHolds : CCS.Holds (ProductPaperAlgebraFor.semantics config)
    productionGlobalParams
    (ProductionFreshClaimProducerFor.freshStatement candidate statementId config
      artifact
      (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
        (publicFits := publicFits) candidate headers statement opening batch)
      memoryResult.suffixBatch assignment)
    assignment

theorem exact
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact)
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement Program)
    (opening : Opening) (machine : Machine Program) (after : AppStateVector)
    (batch : Batch candidate machine statement.base.program
      statement.base.initialApplicationState after)
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {sourceAssignment : Nat -> Nat}
    (memoryResult : ProductionMemoryCheckedBatchRows.Result layout
      sourceAssignment headers)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (evidence : Evidence candidate statementId config artifact relationAuthority headers
      statement opening machine after batch memoryResult
      assignment) :
    ExactInvocation candidate statementId config artifact relationAuthority headers statement
      opening machine after batch memoryResult assignment
      evidence := by
  exact
    { baseInvocationIndex := rfl
      defaultRunningExact := rfl
      initialApplicationExact := rfl
      initialMemoryExact := rfl
      challengeAuthorityExact := evidence.challengeAuthorityExact
      applicationRun := batch.run
      memoryOpen := opening.open_exact_for candidate headers
      delayedCurrentMemory := evidence.memoryStartsAt
      currentPortsExact := evidence.applicationMatched.accesses_exact
      stateCanonical := state_canonical candidate headers statement
        opening batch
      claimCanonical := ProductionFreshClaimProducerFor.value_canonical
        candidate statementId config
        (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
          (publicFits := publicFits) candidate headers statement opening batch)
        memoryResult assignment
      claimMemoryExact := rfl
      claimMemoryBound := ProductionFreshClaimProducerFor.value_memoryBound
        candidate statementId config
        (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
          (publicFits := publicFits) candidate headers statement opening batch)
        memoryResult.suffixBatch assignment
      freshRelationHolds :=
        ProductionFreshClaimProducerFor.freshStatement_holds_from_rows candidate
          statementId config artifact relationAuthority
          (state (rowVariables := rowVariables) (logicalWidth := logicalWidth)
            (publicFits := publicFits) candidate headers statement opening batch)
          memoryResult.suffixBatch assignment sourceAssignment
          evidence.freshRelation }

end Nightstream.Implementation.NebulaV2.ProductionPaperBaseInvocationFor
