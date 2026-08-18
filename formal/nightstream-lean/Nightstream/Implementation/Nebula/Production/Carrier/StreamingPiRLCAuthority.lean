import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFusedPass
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLC
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingSetup
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputFamilyRows
import Nightstream.Implementation.Nebula.Production.NIFS.PiRLC.ParentBridgeFor

/-!
Contract: authoritative family-major input replay for production PiRLC.

Assurance tier: model-level exact refinement and cryptographic-reduction
boundary.

Owns the canonical 17-source serialization used by every one of the 110
PiRLC family phases, its exact order and size, the direct projection from
`Key.piCcsOutputs`, one fused replay-and-algebra phase, and exact equality
between the complete authoritative family result and the monolithic paper
parent.

Does not own generated selective-CCS rows, Rust assignment conformance,
Poseidon2 collision resistance, the complete F-prime schedule, or terminal
lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows
open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlc
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev Family := ProductionStreamingPiRlc.Family
abbrev Source := ProductionStreamingPiRlc.Source
abbrev InputRings := ProductionStreamingPiRlc.InputRings
abbrev BindingState := ProductPoseidon2.State
abbrev InputBindingSetup :=
  SeededAjtai.Setup
    ProductionStreamingPiRlcInputBinding.verifierRows
    ProductionStreamingPiRlcInputBinding.messageColumnCount
abbrev InputResidual :=
  ProductionStreamingPiRlcInputBindingSetup.ResidualFields

/-! ## Canonical family-major input frame -/

private theorem flatMap_singletons
    {Alpha Beta : Type} (items : List Alpha) (value : Alpha -> Beta) :
    items.flatMap (fun item => [value item]) = items.map value := by
  induction items with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [inductionHypothesis]

private theorem flatMap_eq_map_flatten
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha -> List Beta) :
    items.flatMap values = (items.map values).flatten := by
  induction items with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [inductionHypothesis]

private theorem map_eq_of_pointwise
    {Alpha Beta : Type} (items : List Alpha) (left right : Alpha -> Beta)
    (equal : forall item, left item = right item) :
    items.map left = items.map right := by
  induction items with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [equal, inductionHypothesis]

/-- Canonical coefficient order for one base ring. -/
def ringFields (value : RingF) : List Nat :=
  List.ofFn fun lane => (value lane).val

@[simp] theorem ringFields_length (value : RingF) :
    (ringFields value).length = 54 := by
  simp [ringFields, ringDegree]

theorem ringFields_injective : Function.Injective ringFields := by
  intro left right equal
  have valuesEqual :
      (fun lane => (left lane).val) = (fun lane => (right lane).val) :=
    List.ofFn_injective equal
  funext lane
  apply Fin.ext
  exact congrFun valuesEqual lane

/-- The proof-oriented ring view is exactly the Poseidon2 protocol encoding
used by the canonical streaming input frame. -/
theorem ringFields_eq_protocol (value : RingF) :
    ringFields value = ProductPoseidon2.ringFFields value := by
  unfold ringFields ProductPoseidon2.ringFFields ProductPoseidon2.finFields
    ProductPoseidon2.fFields
  rw [flatMap_singletons]
  simp [ProductPoseidon2.fFields, canonicalFinIndices]

/-- Verifier-owned source order `0, ..., 16`. -/
def sourceSchedule : List Source := canonicalFinIndices 17

@[simp] theorem sourceSchedule_length : sourceSchedule.length = 17 := by
  simpa [sourceSchedule] using canonicalFinIndices_length 17

theorem sourceSchedule_values : sourceSchedule.map Fin.val = List.range 17 := by
  exact canonicalFinIndices_values 17

/-- Seventeen equal-width ring blocks for one family. -/
def sourceBlocks (inputs : Source -> RingF) : List (List Nat) :=
  List.ofFn fun source => ringFields (inputs source)

theorem sourceBlocks_lengths (inputs : Source -> RingF) :
    (sourceBlocks inputs).map List.length = List.replicate 17 54 := by
  calc
    (sourceBlocks inputs).map List.length =
        List.ofFn (fun _ : Source => 54) := by
      rw [sourceBlocks, List.map_ofFn]
      apply congrArg List.ofFn
      funext source
      exact ringFields_length (inputs source)
    _ = List.replicate ProductPiRlcRingCombinationRows.sourceCount 54 := by
      exact List.ofFn_const _ _
    _ = List.replicate 17 54 := by
      rw [ProductPiRlcRingCombinationRows.sourceCount_eq]

/-- One family phase reads all seventeen sources in source-major order. -/
def phaseFields (inputs : Source -> RingF) : List Nat :=
  (sourceBlocks inputs).flatten

@[simp] theorem phaseFields_length (inputs : Source -> RingF) :
    (phaseFields inputs).length = 918 := by
  rw [phaseFields, List.length_flatten, sourceBlocks_lengths]
  decide

theorem phaseFields_injective : Function.Injective phaseFields := by
  intro left right equal
  have blocksEqual : sourceBlocks left = sourceBlocks right :=
    WasmResultCodec.flatten_injective_of_lengths
      (sourceBlocks_lengths left) (sourceBlocks_lengths right) equal
  have pointwise :
      (fun source => ringFields (left source)) =
        (fun source => ringFields (right source)) :=
    List.ofFn_injective blocksEqual
  funext source
  exact ringFields_injective (congrFun pointwise source)

/-- Input rings selected for one verifier-owned family. -/
def familyInputs (inputs : InputRings) (family : Family) : Source -> RingF :=
  fun source => inputs source family

def familyInputFields (inputs : InputRings) (family : Family) : List Nat :=
  phaseFields (familyInputs inputs family)

@[simp] theorem familyInputFields_length
    (inputs : InputRings) (family : Family) :
    (familyInputFields inputs family).length = 918 := by
  exact phaseFields_length _

/-- The authority layer and the semantic PiRLC machine use one exact
family-major serialization. -/
theorem familyInputFields_eq_canonical
    (inputs : InputRings) (family : Family) :
    familyInputFields inputs family =
      ProductionStreamingPiRlc.familyInputFrame inputs family := by
  unfold familyInputFields phaseFields
  unfold ProductionStreamingPiRlc.familyInputFrame
  rw [flatMap_eq_map_flatten]
  apply congrArg List.flatten
  unfold sourceBlocks ProductionStreamingPiRlc.sourceSchedule
  rw [canonicalFinIndices, List.map_ofFn]
  apply congrArg List.ofFn
  funext source
  change ringFields (inputs source family) =
    ProductPoseidon2.ringFFields (inputs source family)
  exact ringFields_eq_protocol _

/-- One exact input chunk per verifier-owned family. -/
def inputChunks (inputs : InputRings) : List (List Nat) :=
  familySchedule.map (familyInputFields inputs)

private theorem map_lengths_of_uniform
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha -> List Beta)
    (width : Nat) (uniform : forall item, (values item).length = width) :
    (items.map values).map List.length =
      List.replicate items.length width := by
  induction items with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [uniform, inductionHypothesis, List.replicate_succ]

theorem inputChunks_lengths (inputs : InputRings) :
    (inputChunks inputs).map List.length = List.replicate 110 918 := by
  calc
    (inputChunks inputs).map List.length =
        List.replicate familySchedule.length 918 := by
      exact map_lengths_of_uniform familySchedule (familyInputFields inputs)
        918 (familyInputFields_length inputs)
    _ = List.replicate 110 918 := by rw [familySchedule_length]

/-- Complete semantic PiRLC input stream. It is not carried between phases. -/
def inputFrame (inputs : InputRings) : List Nat :=
  (inputChunks inputs).flatten

@[simp] theorem inputFrame_length (inputs : InputRings) :
    (inputFrame inputs).length = 100980 := by
  rw [inputFrame, List.length_flatten, inputChunks_lengths]
  decide

/-- The complete authority frame is the canonical semantic input frame, not
a second trust-boundary encoding. -/
theorem inputFrame_eq_canonical (inputs : InputRings) :
    inputFrame inputs = ProductionStreamingPiRlc.inputFrame inputs := by
  unfold inputFrame inputChunks ProductionStreamingPiRlc.inputFrame
  rw [flatMap_eq_map_flatten]
  apply congrArg List.flatten
  exact map_eq_of_pointwise familySchedule (familyInputFields inputs)
    (ProductionStreamingPiRlc.familyInputFrame inputs)
    (familyInputFields_eq_canonical inputs)

/-- The family schedule is exactly ordinal order `0, ..., 109`. -/
theorem familySchedule_ordinals :
    familySchedule.map ProductPiRlcAlgebraRows.familyOrdinal =
      List.range 110 := by
  decide

/-! ## Authoritative projection from PiCCS -/

/-- The exact PiRLC input values computed by the paper PiCCS output. -/
noncomputable def authoritativeInputs
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) :
    TypedInputs rowVariables logicalWidth publicFits where
  commitments source :=
    ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
      artifact).piCcsOutputs running fresh proof source).commitment
  publicInputs source :=
    ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
      artifact).piCcsOutputs running fresh proof source).publicInput
  evaluations source :=
    ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
      artifact).piCcsOutputs running fresh proof source).evaluations.getD 0
        (ProductPaperAlgebraFor.evaluationZero rowVariables)

/-- Exact 17-by-110 ring projection of the typed PiCCS output. This is the
ordered value bound by the PiRLC input residual. -/
noncomputable def authoritativeInputRings
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) :
    InputRings :=
  typedInputRings
    (authoritativeInputs candidate statementId config artifact running fresh
      proof)

@[simp] theorem piCcsOutputs_evaluations_eq_singleton
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
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
    (source : Source) :
    ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
      artifact).piCcsOutputs running fresh proof source).evaluations =
      #[(authoritativeInputs candidate statementId config artifact running fresh
        proof).evaluations source] := by
  rfl

/-- The complete 110-family authoritative result is the current monolithic
paper PiRLC parent. -/
theorem complete_family_run_eq_parent
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) :
    let key := ProductionProductPiCcsTypedBridgeFor.paperKey candidate
      statementId config artifact
    let inputs := authoritativeInputs candidate statementId config artifact
      running fresh proof
    let challenges := key.piRlcChallenges running fresh proof
    outputBundle challenges inputs = (key.parent running fresh proof).commitment /\
      outputPublic challenges inputs =
        (key.parent running fresh proof).publicInput /\
      #[outputEvaluation challenges inputs] =
        (key.parent running fresh proof).evaluations := by
  dsimp only
  let key := ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
    config artifact
  let inputs := authoritativeInputs candidate statementId config artifact
    running fresh proof
  let challenges := key.piRlcChallenges running fresh proof
  have exactOutput := typedOutput_exact challenges inputs
  constructor
  · calc
      outputBundle challenges inputs =
          ProductCommitmentAlgebra.combineBundles challenges
            inputs.commitments := exactOutput.1
      _ = (key.parent running fresh proof).commitment := by rfl
  constructor
  · calc
      outputPublic challenges inputs =
          Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs challenges
            inputs.publicInputs := exactOutput.2.1
      _ = (key.parent running fresh proof).publicInput := by rfl
  · calc
      #[outputEvaluation challenges inputs] =
          #[ProductPaperAlgebraFor.combineEvaluationFamily challenges
            inputs.evaluations] := congrArg (fun value => #[value])
              exactOutput.2.2
      _ = ProductPaperAlgebraFor.combineEvaluations rowVariables challenges
          (fun source => #[inputs.evaluations source]) :=
        (ProductionProductPiRlcParentBridgeFor.combineEvaluations_singletons
          rowVariables challenges inputs.evaluations).symm
      _ = ProductPaperAlgebraFor.combineEvaluations rowVariables challenges
          (fun source => (key.piCcsOutputs running fresh proof source
            ).evaluations) := by
        congr 2
      _ = (key.parent running fresh proof).evaluations := by rfl

/-! ## One fused family phase -/

/-- A proof-only collector for one bounded 918-field phase. It is not a
persistent circuit carrier. -/
def collectField (fields : List Nat) (value : Nat) : List Nat :=
  fields ++ [value]

theorem foldl_collectField (values initial : List Nat) :
    values.foldl collectField initial = initial ++ values := by
  induction values generalizing initial with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      simp [collectField, List.append_assoc]

def phaseRuntime (prior : BindingState) :
    ProductionStreamingFusedPass.Runtime (List Nat) where
  transcript := prior
  cursor := 0
  accumulator := []

/-- Named failure event for one family phase. The family label is
verifier-owned metadata; the collision is over that phase's exact input
frame from its carried prior state. -/
structure InputReplayCollision
    (prior : BindingState) (family : Family)
    (authoritative : Source -> RingF) : Prop where
  frameCollision : ProductionStreamingFusedPass.FrameReplayCollisionAt prior
    (phaseFields authoritative)

/-- A deterministic 918-field schedule has exactly one phase chunk. -/
theorem phase_chunk_count_exact
    {inputs : Source -> RingF} {chunks : List (List Nat)}
    (schedule : ProductionFullClaimStreaming.ChunkSchedule 918
      (phaseFields inputs) chunks) :
    chunks.length = 1 := by
  have lower := schedule.values_length_le_chunk_capacity
  have upper := schedule.chunk_capacity_lt_values_plus_width (by decide)
  rw [phaseFields_length] at lower upper
  omega

/-- Equality with the authoritative phase transcript recovers the exact
source rings used by the algebra, or exposes a named Poseidon2 collision. -/
theorem fused_phase_recovers_inputs_or_collision
    (prior : BindingState) (family : Family)
    (authoritative supplied : Source -> RingF)
    (chunks : List (List Nat))
    (schedule : ProductionFullClaimStreaming.ChunkSchedule 918
      (phaseFields supplied) chunks)
    (normalized : prior.absorbed < Poseidon2Sponge.rate)
    (transcriptExact :
      (ProductionStreamingFusedPass.run collectField chunks
        (phaseRuntime prior)).transcript =
          Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (phaseFields authoritative) prior) :
    supplied = authoritative \/
      InputReplayCollision prior family authoritative := by
  have recovered :=
    ProductionStreamingFusedPass.accepted_run_recovers_fold_or_collision_at
      collectField (phaseRuntime prior) schedule normalized transcriptExact
  rcases recovered with exactFold | collision
  · left
    apply phaseFields_injective
    calc
      phaseFields supplied =
          (ProductionStreamingFusedPass.run collectField chunks
            (phaseRuntime prior)).accumulator := by
        rw [ProductionStreamingFusedPass.run_accumulator,
          schedule.flatten_eq, foldl_collectField]
        simp [phaseRuntime]
      _ = (phaseFields authoritative).foldl collectField [] := by
        simpa [phaseRuntime] using exactFold
      _ = phaseFields authoritative := by
        rw [foldl_collectField]
        simp
  · right
    exact ⟨by simpa [phaseRuntime] using collision⟩

theorem accepted_different_phase_implies_collision
    (prior : BindingState) (family : Family)
    (authoritative supplied : Source -> RingF)
    (different : supplied ≠ authoritative)
    (chunks : List (List Nat))
    (schedule : ProductionFullClaimStreaming.ChunkSchedule 918
      (phaseFields supplied) chunks)
    (normalized : prior.absorbed < Poseidon2Sponge.rate)
    (transcriptExact :
      (ProductionStreamingFusedPass.run collectField chunks
        (phaseRuntime prior)).transcript =
          Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (phaseFields authoritative) prior) :
    InputReplayCollision prior family authoritative := by
  rcases fused_phase_recovers_inputs_or_collision prior family authoritative
      supplied chunks schedule normalized transcriptExact with equal | collision
  · exact False.elim (different equal)
  · exact collision

/-! ## Concrete local-row phase -/

/-- Complete persistent state for the 110 family phases.

The rank-two residual is the algebraic authority for the 100,980 PiCCS-derived
input fields. The input and output duplex states are checked compression and
collision boundaries only. The exact 918 challenge coefficients are carried
as values because a digest cannot authorize them. -/
structure FamilyState where
  inputReplay : BindingState
  inputResidual : InputResidual
  outputReplay : BindingState
  challenges : Source -> RingF
  familyCursor : Nat

def initialFamilyState
    (inputReplay : BindingState) (inputResidual : InputResidual)
    (outputReplay : BindingState)
    (challenges : Source -> RingF) : FamilyState where
  inputReplay := inputReplay
  inputResidual := inputResidual
  outputReplay := outputReplay
  challenges := challenges
  familyCursor := 0

/-- Physical facts that the `piRlcStart` rows must place into the family
continuation. The challenge values come from the existing row-derived sampler;
they are not prover-selected inputs to this structure. -/
structure FamilyStartTransition
    (after : FamilyState) (challenges : Source -> RingF)
    (inputResidual : InputResidual) : Prop where
  inputReplay : after.inputReplay = ProductPoseidon2.initialState
  inputResidual : after.inputResidual = inputResidual
  outputReplay : after.outputReplay = ProductPoseidon2.initialState
  challenges : after.challenges = challenges
  cursor : after.familyCursor = 0

/-- Exact semantic result of the verifier-owned `piRlcStart` phase. -/
def FamilyStartRelation
    (after : FamilyState) (authoritativeChallenges : Source -> RingF)
    (authoritativeResidual : InputResidual) : Prop :=
  after.inputReplay = ProductPoseidon2.initialState /\
    after.inputResidual = authoritativeResidual /\
      after.outputReplay = ProductPoseidon2.initialState /\
        after.challenges = authoritativeChallenges /\
          after.familyCursor = 0

theorem familyStart_of_transition
    {after : FamilyState}
    {derivedChallenges authoritativeChallenges : Source -> RingF}
    {derivedResidual authoritativeResidual : InputResidual}
    (transition : FamilyStartTransition after derivedChallenges
      derivedResidual)
    (challengeExact : derivedChallenges = authoritativeChallenges)
    (residualExact : derivedResidual = authoritativeResidual) :
    FamilyStartRelation after authoritativeChallenges
      authoritativeResidual := by
  exact ⟨transition.inputReplay,
    transition.inputResidual.trans residualExact,
    transition.outputReplay,
    transition.challenges.trans challengeExact, transition.cursor⟩

/-- Canonical row-major field order for the carried rank-two residual. -/
def inputResidualFields (residual : InputResidual) : List Nat :=
  List.ofFn fun output => (residual output).val

@[simp] theorem inputResidualFields_length (residual : InputResidual) :
    (inputResidualFields residual).length = 108 := by
  simp [inputResidualFields,
    ProductionStreamingPiRlcInputBindingSetup.exact_output_width]

/-- Exact source serialization of the complete PiRLC family continuation. -/
def familyStateFields (state : FamilyState) : List Nat :=
  bindingFields state.inputReplay ++
    inputResidualFields state.inputResidual ++
      bindingFields state.outputReplay ++
        phaseFields state.challenges ++ [state.familyCursor]

@[simp] theorem familyStateFields_length (state : FamilyState) :
    (familyStateFields state).length = 1045 := by
  simp [familyStateFields]

private theorem bindingFields_injective : Function.Injective bindingFields := by
  intro left right equal
  have blocksEqual :
      [List.ofFn left.lanes, [left.absorbed]] =
        [List.ofFn right.lanes, [right.absorbed]] := by
    apply WasmResultCodec.flatten_injective_of_lengths
        (widths := [8, 1])
    · simp [Poseidon2Core.width]
    · simp [Poseidon2Core.width]
    · simpa [bindingFields] using equal
  have lanesEqual : left.lanes = right.lanes := by
    apply List.ofFn_injective
    simpa using congrArg (fun blocks => blocks.getD 0 []) blocksEqual
  have absorbedEqual : left.absorbed = right.absorbed := by
    simpa using congrArg (fun blocks => blocks.getD 1 []) blocksEqual
  cases left
  cases right
  simp_all

private theorem inputResidualFields_injective :
    Function.Injective inputResidualFields := by
  intro left right equal
  have valuesEqual :
      (fun output => (left output).val) =
        (fun output => (right output).val) :=
    List.ofFn_injective equal
  funext output
  apply Fin.ext
  exact congrFun valuesEqual output

/-- The complete 1,045-field continuation encoding has no serialization
ambiguity. Any two different family states have different field lists. -/
theorem familyStateFields_injective : Function.Injective familyStateFields := by
  intro left right equal
  have blocksEqual :
      [bindingFields left.inputReplay,
       inputResidualFields left.inputResidual,
       bindingFields left.outputReplay,
       phaseFields left.challenges,
       [left.familyCursor]] =
        [bindingFields right.inputReplay,
         inputResidualFields right.inputResidual,
         bindingFields right.outputReplay,
         phaseFields right.challenges,
         [right.familyCursor]] := by
    apply WasmResultCodec.flatten_injective_of_lengths
        (widths := [9, 108, 9, 918, 1])
    · simp
    · simp
    · simpa [familyStateFields] using equal
  have inputReplayEqual : left.inputReplay = right.inputReplay :=
    bindingFields_injective (by
      simpa using congrArg (fun blocks => blocks.getD 0 []) blocksEqual)
  have inputResidualEqual : left.inputResidual = right.inputResidual :=
    inputResidualFields_injective (by
      simpa using congrArg (fun blocks => blocks.getD 1 []) blocksEqual)
  have outputReplayEqual : left.outputReplay = right.outputReplay :=
    bindingFields_injective (by
      simpa using congrArg (fun blocks => blocks.getD 2 []) blocksEqual)
  have challengesEqual : left.challenges = right.challenges :=
    phaseFields_injective (by
      simpa using congrArg (fun blocks => blocks.getD 3 []) blocksEqual)
  have cursorEqual : left.familyCursor = right.familyCursor := by
    simpa using congrArg (fun blocks => blocks.getD 4 []) blocksEqual
  cases left
  cases right
  simp_all

/-- Exact state links owned by one family phase. Concrete generated glue rows
must prove each field of this structure. -/
structure FamilyTransition
    (setup : InputBindingSetup) (before after : FamilyState)
    (family : Family) (inputs : Source -> RingF) (output : RingF) : Prop where
  inputReplay :
    after.inputReplay =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (phaseFields inputs) before.inputReplay
  inputResidual :
    ProductionStreamingPiRlcInputBindingSetup.ConcreteResidualTransition
      setup before.inputResidual after.inputResidual family inputs
  outputReplay :
    after.outputReplay =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (ringFields output) before.outputReplay
  challenges : after.challenges = before.challenges
  cursor : after.familyCursor = before.familyCursor + 1

/-- One concrete phase reads the carried verifier-derived challenges, replays
the same seventeen input rings used by the algebra, and binds the output ring. -/
def FamilyPhaseRelation
    (setup : InputBindingSetup) (before after : FamilyState) (family : Family)
    (inputs : Source -> RingF) (output : RingF) : Prop :=
  before.familyCursor = ProductPiRlcAlgebraRows.familyOrdinal family /\
    output = combineOne before.challenges inputs /\
    FamilyTransition setup before after family inputs output

theorem familyPhase_uses_authoritative_challenges
    {setup : InputBindingSetup} {before after : FamilyState} {family : Family}
    {inputs : Source -> RingF} {output : RingF}
    (phase : FamilyPhaseRelation setup before after family inputs output)
    (authoritative : Source -> RingF)
    (challengeExact : before.challenges = authoritative) :
    output = combineOne authoritative inputs := by
  rw [← challengeExact]
  exact phase.2.1

/-- An accepted phase replays the authoritative PiCCS input family, or gives
one explicit Poseidon2 collision from the carried input duplex state. -/
theorem familyPhase_recovers_authoritative_inputs_or_collision
    {setup : InputBindingSetup} {before after : FamilyState} {family : Family}
    {authoritative supplied : Source -> RingF} {output : RingF}
    (phase : FamilyPhaseRelation setup before after family supplied output)
    (transcriptExact :
      after.inputReplay =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (phaseFields authoritative) before.inputReplay) :
    (supplied = authoritative /\
        FamilyPhaseRelation setup before after family authoritative output) \/
      InputReplayCollision before.inputReplay family authoritative := by
  by_cases exact : supplied = authoritative
  · subst supplied
    exact Or.inl ⟨rfl, phase⟩
  · apply Or.inr
    refine { frameCollision := ?_ }
    refine ⟨phaseFields supplied, ?_, ?_⟩
    · intro fieldsExact
      exact exact (phaseFields_injective fieldsExact)
    · exact phase.2.2.inputReplay.symm.trans transcriptExact

def decodedChallenges
    (layout : ProductPiRlcRingCombinationRows.Layout)
    (assignment : Nat -> Nat)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5) : Source -> RingF :=
  challengeRing layout assignment range

def decodedInputs
    (layout : ProductPiRlcRingCombinationRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Source -> RingF :=
  inputRing layout assignment canonical

/-- The existing 49,626 handwritten rows imply one exact fused phase. The
same decoded input rings occur in the replay frame and in `combineOne`. -/
theorem local_rows_imply_concrete_phase
    {layout : ProductPiRlcRingCombinationRows.Layout}
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (satisfied : Satisfies
      (ProductPiRlcRingCombinationRows.rows layout) assignment)
    (setup : InputBindingSetup) (before after : FamilyState) (family : Family)
    (challengesExact :
      decodedChallenges layout assignment range = before.challenges)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (transition : FamilyTransition setup before after family
      (decodedInputs layout assignment canonical)
      (outputRing layout assignment canonical)) :
    FamilyPhaseRelation setup before after family
      (decodedInputs layout assignment canonical)
      (outputRing layout assignment canonical) := by
  refine ⟨cursorExact, ?_, transition⟩
  rw [← challengesExact]
  exact local_rows_imply_combineOne canonical one range satisfied

/-- The arithmetic rows and the exact family input rows use one assignment.
The family input rows derive the residual transition from the same 918 input
fields used by `combineOne`. Only the four non-residual state links remain as
explicit glue facts. -/
theorem local_rows_imply_concrete_phase_from_input_rows
    {algebraLayout : ProductPiRlcRingCombinationRows.Layout}
    {inputLayout :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.Layout}
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (algebraLayout.challengeSymbol source lane) < 5)
    (algebraSatisfied : Satisfies
      (ProductPiRlcRingCombinationRows.rows algebraLayout) assignment)
    (setup : InputBindingSetup) (before after : FamilyState) (family : Family)
    (inputsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.InputsPlaced
        inputLayout.phase assignment
        (decodedInputs algebraLayout assignment canonical))
    (residualsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
        inputLayout assignment before.inputResidual after.inputResidual)
    (inputRowsSatisfied : Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.rows
        setup inputLayout family) assignment)
    (challengesExact :
      decodedChallenges algebraLayout assignment range = before.challenges)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (inputReplayExact :
      after.inputReplay =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (phaseFields (decodedInputs algebraLayout assignment canonical))
          before.inputReplay)
    (outputReplayExact :
      after.outputReplay =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (ringFields (outputRing algebraLayout assignment canonical))
          before.outputReplay)
    (challengesCarry : after.challenges = before.challenges)
    (cursorIncrement : after.familyCursor = before.familyCursor + 1) :
    FamilyPhaseRelation setup before after family
      (decodedInputs algebraLayout assignment canonical)
      (outputRing algebraLayout assignment canonical) := by
  have inputExact :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.rows_sound
      canonical one inputsPlaced residualsPlaced inputRowsSatisfied
  have transition : FamilyTransition setup before after family
      (decodedInputs algebraLayout assignment canonical)
      (outputRing algebraLayout assignment canonical) := by
    exact {
      inputReplay := inputReplayExact
      inputResidual := inputExact.transition
      outputReplay := outputReplayExact
      challenges := challengesCarry
      cursor := cursorIncrement
    }
  exact local_rows_imply_concrete_phase canonical one range algebraSatisfied
    setup before after family challengesExact cursorExact transition

/-! ## Row-derived start authority -/

/-- The existing post-PiCCS transcript, candidate-classification, and
first-accepted rows derive all 918 challenge coefficients placed by
`piRlcStart`. Generated start glue still has to prove
`FamilyStartTransition` on the same assignment. -/
theorem sampler_rows_imply_authoritative_start
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
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
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (piCcsPlacement : ProductionProductPiCcsTypedBridgeFor.Placement candidate
      statementId config artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRowsFor.rows
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (classificationRows : ProductPiRlcCandidateClassificationRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (selectorRows : ProductPiRlcFirstAcceptedBatchRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (challengePlacement : ProductPiRlcChallengeBridge.Placement
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) algebraLayout)
    (inputSetup : InputBindingSetup) :
    let range := ProductPiRlcChallengeBridge.challengeSymbol_range
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase)
      algebraLayout assignment canonical one transcriptRows classificationRows
      selectorRows challengePlacement
    forall after,
      FamilyStartTransition after
          (ProductPiRlcAlgebraSound.decodeChallenges algebraLayout assignment
            range)
          (ProductionStreamingPiRlcInputBindingSetup.concreteBinding inputSetup
            (authoritativeInputRings candidate statementId config artifact
              running fresh proof)) ->
        FamilyStartRelation after
          ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact).piRlcChallenges running fresh proof)
          (ProductionStreamingPiRlcInputBindingSetup.concreteBinding inputSetup
            (authoritativeInputRings candidate statementId config artifact
              running fresh proof)) := by
  dsimp only
  intro after transition
  apply familyStart_of_transition transition
  · exact ProductionProductPiRlcParentBridgeFor.challenges_eq_selected
      candidate statementId config artifact running fresh proof wires
      samplerBase algebraLayout assignment canonical one piCcsPlacement
      piCcsRows transcriptRows classificationRows selectorRows
      challengePlacement
  · rfl

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
