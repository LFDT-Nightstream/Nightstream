import Batteries.Data.Fin.Coding
import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.PiCCSProjection

/-!
Owns the Lean-authored generative plan for Stage 1 Poseidon2 invocations.

Action blocks carry source-layout expressions and exact physical starts. Their
interpreter is the existing invocation compiler. Direct blocks cover the
PiRLC digest windows. The canonical expansion theorem fixes every invocation
and keeps affine lowering out of package emission.
-/

namespace NightstreamFPrime.Export.Stage1.PermutationPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.PiRLC.v1_1

abbrev EState := Invocations.EState
abbrev ActionShape := Formal.ActionShape

/-- Recover an eight-lane state from its wire list. Canonical plans always
carry exactly eight values; Rust rejects any other length. -/
def stateOfList (values : List Expr) : EState :=
  fun lane => values.getD lane.val 0

theorem stateOfList_ofFn (state : EState) :
    stateOfList (List.ofFn state) = state := by
  funext lane
  exact NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD
    state lane 0

namespace ActionShape

/-- Squeeze expectations own assertion rows and do not affect permutation
invocations. Zero is the canonical erased expectation. -/
def toAction : ActionShape → Formal.Action
  | .absorb input => .absorb input
  | .squeezeK => .squeezeK KExpr.zero

@[simp] theorem toAction_shape (shape : ActionShape) :
    shape.toAction.shape = shape := by
  cases shape <;> rfl

def format : Format ActionShape where
  encode
    | .absorb input => .array [
        .atom 0,
        (list Package.exprFormat).encode input]
    | .squeezeK => .array [.atom 1]
  decode
    | .array [.atom 0, input] => do
      pure (.absorb (← (list Package.exprFormat).decode input))
    | .array [.atom 1] => .ok .squeezeK
    | _ => .error "invalid permutation action shape"
  decode_encode := by
    intro value
    cases value with
    | absorb input =>
        simp only
        rw [(list Package.exprFormat).decode_encode]
        rfl
    | squeezeK => rfl

end ActionShape

/-- One exact invocation trace compiled from a verifier-owned Duplex action
schedule. The state and absorb inputs remain in Lean source-column order. -/
structure ActionBlock where
  phase : Nat
  rowStart : Nat
  witnessStart : Nat
  initialState : List Expr
  actionShapes : List ActionShape

def ActionBlock.format : Format ActionBlock where
  encode := fun value => .array [
    .atom value.phase,
    .atom value.rowStart,
    .atom value.witnessStart,
    (list Package.exprFormat).encode value.initialState,
    (list ActionShape.format).encode value.actionShapes]
  decode
    | .array [.atom phase, .atom rowStart, .atom witnessStart,
        initialState, actionShapes] => do
      pure ⟨phase, rowStart, witnessStart,
        ← (list Package.exprFormat).decode initialState,
        ← (list ActionShape.format).decode actionShapes⟩
    | _ => .error "invalid permutation action block"
  decode_encode := by
    intro value
    cases value
    simp only
    rw [(list Package.exprFormat).decode_encode,
      (list ActionShape.format).decode_encode]
    rfl

def ActionBlock.ofActions (phase rowStart witnessStart : Nat)
    (state : EState) (actions : List Formal.Action) : ActionBlock where
  phase := phase
  rowStart := rowStart
  witnessStart := witnessStart
  initialState := List.ofFn state
  actionShapes := actions.map Formal.Action.shape

def ActionBlock.expand (block : ActionBlock) :
    List PermutationInvocation :=
  (Invocations.compileActions block.phase block.rowStart block.witnessStart
    (stateOfList block.initialState)
    (block.actionShapes.map ActionShape.toAction)).invocations

theorem ActionBlock.ofActions_expand
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Formal.Action) :
    (ActionBlock.ofActions phase rowStart witnessStart state actions).expand =
      (Invocations.compileActions phase rowStart witnessStart state
        actions).invocations := by
  unfold ActionBlock.expand ActionBlock.ofActions
  rw [stateOfList_ofFn]
  exact congrArg (fun trace : Invocations.Trace => trace.invocations)
    (Invocations.compileActions_eq_of_shapes phase rowStart witnessStart state
      ((actions.map Formal.Action.shape).map ActionShape.toAction) actions
      (by simp))

/-- One direct invocation whose eight inputs remain as source expressions. -/
structure DirectBlock where
  phase : Nat
  rowStart : Nat
  witnessStart : Nat
  state : List Expr

def DirectBlock.format : Format DirectBlock where
  encode := fun value => .array [
    .atom value.phase,
    .atom value.rowStart,
    .atom value.witnessStart,
    (list Package.exprFormat).encode value.state]
  decode
    | .array [.atom phase, .atom rowStart, .atom witnessStart, state] => do
      pure ⟨phase, rowStart, witnessStart,
        ← (list Package.exprFormat).decode state⟩
    | _ => .error "invalid direct permutation block"
  decode_encode := by
    intro value
    cases value
    simp only
    rw [(list Package.exprFormat).decode_encode]
    rfl

def DirectBlock.ofState (phase rowStart witnessStart : Nat)
    (state : EState) : DirectBlock where
  phase := phase
  rowStart := rowStart
  witnessStart := witnessStart
  state := List.ofFn state

def DirectBlock.expand (block : DirectBlock) : PermutationInvocation :=
  Invocations.invocation block.phase block.rowStart block.witnessStart
    (stateOfList block.state)

theorem DirectBlock.ofState_expand
    (phase rowStart witnessStart : Nat) (state : EState) :
    (DirectBlock.ofState phase rowStart witnessStart state).expand =
      Invocations.invocation phase rowStart witnessStart state := by
  unfold DirectBlock.expand DirectBlock.ofState
  rw [stateOfList_ofFn]

/-- Each block expands by one fixed Lean interpreter. -/
inductive Block where
  | actions (block : ActionBlock)
  | direct (block : DirectBlock)

def Block.expand : Block → List PermutationInvocation
  | .actions block => block.expand
  | .direct block => [block.expand]

def Block.format : Format Block where
  encode
    | .actions block => .array [.atom 0, ActionBlock.format.encode block]
    | .direct block => .array [.atom 1, DirectBlock.format.encode block]
  decode
    | .array [.atom 0, block] => do
      pure (.actions (← ActionBlock.format.decode block))
    | .array [.atom 1, block] => do
      pure (.direct (← DirectBlock.format.decode block))
    | _ => .error "invalid permutation plan block"
  decode_encode := by
    intro value
    cases value <;> simp [Format.decode_encode]

/-! ## Exact witness-start projection -/

/-- Consecutive final-layout witness starts for one compiled action trace. -/
private def sequentialWitnessStarts : Nat → Nat → List Nat
  | _witnessStart, 0 => []
  | witnessStart, count + 1 =>
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan witnessStart ::
        sequentialWitnessStarts (witnessStart + 592) count

private theorem sequentialWitnessStarts_add
    (witnessStart left right : Nat) :
    sequentialWitnessStarts witnessStart (left + right) =
      sequentialWitnessStarts witnessStart left ++
        sequentialWitnessStarts (witnessStart + left * 592) right := by
  induction left generalizing witnessStart with
  | zero => simp [sequentialWitnessStarts]
  | succ left inductionHypothesis =>
      simp only [Nat.succ_add, sequentialWitnessStarts, List.cons_append]
      rw [inductionHypothesis]
      rw [show witnessStart + 592 + left * 592 =
        witnessStart + Nat.succ left * 592 by omega]

private theorem sequentialWitnessStarts_length (witnessStart count : Nat) :
    (sequentialWitnessStarts witnessStart count).length = count := by
  induction count generalizing witnessStart with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [sequentialWitnessStarts, List.length_cons, inductionHypothesis]

private theorem sequentialWitnessStarts_getD
    (witnessStart count index : Nat) (bound : index < count) :
    (sequentialWitnessStarts witnessStart count).getD index 0 =
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        (witnessStart + index * 592) := by
  induction count generalizing witnessStart index with
  | zero => omega
  | succ count inductionHypothesis =>
      cases index with
      | zero => simp [sequentialWitnessStarts]
      | succ index =>
          rw [sequentialWitnessStarts, List.getD_cons_succ,
            inductionHypothesis (witnessStart + 592) index (by omega)]
          apply congrArg NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
          omega

private theorem compileBlocks_witnessStarts
    (phase rowStart witnessStart : Nat) (state : EState)
    (blocks : List (List Expr)) :
    (Invocations.compileBlocks phase rowStart witnessStart state blocks).invocations.map
        (fun invocation => invocation.witnessStart) =
      sequentialWitnessStarts witnessStart blocks.length := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp only [Invocations.compileBlocks, List.map_cons,
        Invocations.invocation_witnessStart, List.length_cons,
        sequentialWitnessStarts]
      rw [inductionHypothesis
        (rowStart := rowStart + 592)
        (witnessStart := witnessStart + 592)
        (state := Invocations.permutationOutput witnessStart)]

private theorem compileActions_witnessStarts
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Formal.Action) :
    (Invocations.compileActions phase rowStart witnessStart state actions).invocations.map
        (fun invocation => invocation.witnessStart) =
      sequentialWitnessStarts witnessStart
        (Invocations.invocationCount actions) := by
  induction actions generalizing rowStart witnessStart state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simp only [Invocations.compileActions, List.map_append]
          rw [compileBlocks_witnessStarts, inductionHypothesis,
            Invocations.compileBlocks_witnessNext]
          simp only [Invocations.invocationCount,
            Invocations.Action.invocationCount, List.map_cons, List.sum_cons]
          exact (sequentialWitnessStarts_add witnessStart
            (Hash.inputChunks input).length
            ((actions.map Invocations.Action.invocationCount).sum)).symm
      | squeezeK expected =>
          simp only [Invocations.compileActions, List.map_cons,
            Invocations.invocation_witnessStart]
          rw [inductionHypothesis]
          simp only [Invocations.invocationCount,
            Invocations.Action.invocationCount, List.map_cons, List.sum_cons]
          rw [sequentialWitnessStarts_add witnessStart 2
            ((actions.map Invocations.Action.invocationCount).sum)]
          simp only [sequentialWitnessStarts, List.cons_append,
            List.nil_append]

/-! ## Canonical PiCCS blocks -/

def statementBlock (_unit : Unit) : ActionBlock :=
  ActionBlock.ofActions PiCCSInvocations.statementPhase
    PiCCSInvocations.statementRowStart PiCCSInvocations.statementWitnessStart
    Hash.zeroE
    (PiCCSInvocations.statementActions Data.logicalWidth Data.publicFits)

def challengeBlock (_unit : Unit) : ActionBlock :=
  ActionBlock.ofActions PiCCSInvocations.challengePhase
    PiCCSInvocations.challengeRowStart PiCCSInvocations.challengeWitnessStart
    (PiCCSProjection.fastStatementState Data.logicalWidth Data.publicFits)
    (PiCCSInvocations.challengeActions Data.logicalWidth Data.publicFits)

def roundBlock (_unit : Unit) : ActionBlock :=
  ActionBlock.ofActions PiCCSInvocations.roundPhase
    PiCCSInvocations.roundRowStart PiCCSInvocations.roundWitnessStart
    (PiCCSProjection.fastChallengeState Data.logicalWidth Data.publicFits)
    (PiCCSInvocations.roundActions Data.logicalWidth Data.publicFits)

def outputBlock (_unit : Unit) : ActionBlock :=
  ActionBlock.ofActions PiCCSInvocations.outputPhase
    PiCCSInvocations.outputRowStart PiCCSInvocations.outputWitnessStart
    (PiCCSProjection.fastRoundState Data.logicalWidth Data.publicFits)
    (PiCCSInvocations.outputActions Data.logicalWidth Data.publicFits)

theorem statementBlock_expand :
    (statementBlock ()).expand =
      (PiCCSInvocations.statementTrace Data.logicalWidth
        Data.publicFits).invocations := by
  simpa [statementBlock, PiCCSInvocations.statementTrace] using
    ActionBlock.ofActions_expand PiCCSInvocations.statementPhase
      PiCCSInvocations.statementRowStart
      PiCCSInvocations.statementWitnessStart Hash.zeroE
      (PiCCSInvocations.statementActions Data.logicalWidth Data.publicFits)

theorem challengeBlock_expand :
    (challengeBlock ()).expand =
      (PiCCSInvocations.challengeTrace Data.logicalWidth
        Data.publicFits).invocations := by
  calc
    (challengeBlock ()).expand =
        (Invocations.compileActions PiCCSInvocations.challengePhase
          PiCCSInvocations.challengeRowStart
          PiCCSInvocations.challengeWitnessStart
          (PiCCSProjection.fastStatementState Data.logicalWidth
            Data.publicFits)
          (PiCCSInvocations.challengeActions Data.logicalWidth
            Data.publicFits)).invocations :=
      (by
        simpa [challengeBlock] using
          ActionBlock.ofActions_expand PiCCSInvocations.challengePhase
            PiCCSInvocations.challengeRowStart
            PiCCSInvocations.challengeWitnessStart
            (PiCCSProjection.fastStatementState Data.logicalWidth
              Data.publicFits)
            (PiCCSInvocations.challengeActions Data.logicalWidth
              Data.publicFits))
    _ = _ := by
      rw [PiCCSProjection.fastStatementState_eq]
      rfl

theorem roundBlock_expand :
    (roundBlock ()).expand =
      (PiCCSInvocations.roundTrace Data.logicalWidth
        Data.publicFits).invocations := by
  calc
    (roundBlock ()).expand =
        (Invocations.compileActions PiCCSInvocations.roundPhase
          PiCCSInvocations.roundRowStart PiCCSInvocations.roundWitnessStart
          (PiCCSProjection.fastChallengeState Data.logicalWidth
            Data.publicFits)
          (PiCCSInvocations.roundActions Data.logicalWidth
            Data.publicFits)).invocations :=
      (by
        simpa [roundBlock] using
          ActionBlock.ofActions_expand PiCCSInvocations.roundPhase
            PiCCSInvocations.roundRowStart PiCCSInvocations.roundWitnessStart
            (PiCCSProjection.fastChallengeState Data.logicalWidth
              Data.publicFits)
            (PiCCSInvocations.roundActions Data.logicalWidth
              Data.publicFits))
    _ = _ := by
      rw [PiCCSProjection.fastChallengeState_eq]
      rfl

theorem outputBlock_expand :
    (outputBlock ()).expand =
      (PiCCSInvocations.outputTrace Data.logicalWidth
        Data.publicFits).invocations := by
  calc
    (outputBlock ()).expand =
        (Invocations.compileActions PiCCSInvocations.outputPhase
          PiCCSInvocations.outputRowStart PiCCSInvocations.outputWitnessStart
          (PiCCSProjection.fastRoundState Data.logicalWidth Data.publicFits)
          (PiCCSInvocations.outputActions Data.logicalWidth
            Data.publicFits)).invocations :=
      (by
        simpa [outputBlock] using
          ActionBlock.ofActions_expand PiCCSInvocations.outputPhase
            PiCCSInvocations.outputRowStart
            PiCCSInvocations.outputWitnessStart
            (PiCCSProjection.fastRoundState Data.logicalWidth Data.publicFits)
            (PiCCSInvocations.outputActions Data.logicalWidth
              Data.publicFits))
    _ = _ := by
      rw [PiCCSProjection.fastRoundState_eq]
      rfl

def piCcsBlocks (_unit : Unit) : List Block :=
  [.actions (statementBlock ()), .actions (challengeBlock ()),
   .actions (roundBlock ()), .actions (outputBlock ())]

theorem piCcsBlocks_expand :
    (piCcsBlocks ()).flatMap Block.expand =
      PiCCSInvocations.invocations Data.logicalWidth Data.publicFits := by
  rw [PiCCSInvocations.invocations]
  simp only [piCcsBlocks, List.flatMap_cons, List.flatMap_nil, Block.expand,
    List.append_nil, statementBlock_expand, challengeBlock_expand,
    roundBlock_expand, outputBlock_expand]
  simp only [List.append_assoc]

/-- Lightweight exact witness-start schedule for all four PiCCS packets. -/
def piCcsWitnessStarts (_unit : Unit) : List Nat :=
  sequentialWitnessStarts PiCCSInvocations.statementWitnessStart 379 ++
    sequentialWitnessStarts PiCCSInvocations.challengeWitnessStart 87 ++
    sequentialWitnessStarts PiCCSInvocations.roundWitnessStart 252 ++
    sequentialWitnessStarts PiCCSInvocations.outputWitnessStart 6886

theorem piCcsWitnessStarts_materializes :
    piCcsWitnessStarts () =
      (PiCCSInvocations.invocations Data.logicalWidth Data.publicFits).map
        (fun invocation => invocation.witnessStart) := by
  unfold piCcsWitnessStarts PiCCSInvocations.invocations
  simp only [List.map_append, PiCCSInvocations.statementTrace,
    PiCCSInvocations.challengeTrace, PiCCSInvocations.roundTrace,
    PiCCSInvocations.outputTrace, compileActions_witnessStarts]
  rw [PiCCSInvocations.statementInvocationCount_eq,
    PiCCSInvocations.challengeInvocationCount_eq,
    PiCCSInvocations.roundInvocationCount_eq,
    PiCCSInvocations.outputInvocationCount_eq]

private theorem piCcsWitnessStarts_transcriptPrefix :
    piCcsWitnessStarts () =
      sequentialWitnessStarts PiCCSInvocations.statementWitnessStart 718 ++
        sequentialWitnessStarts PiCCSInvocations.outputWitnessStart 6886 := by
  have challengeStart : PiCCSInvocations.statementWitnessStart + 379 * 592 =
      PiCCSInvocations.challengeWitnessStart := by
    simp only [PiCCSInvocations.statementWitnessStart,
      PiCCSInvocations.challengeWitnessStart,
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq,
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeWitnessStart_eq]
  have roundStart : PiCCSInvocations.statementWitnessStart +
      (379 + 87) * 592 = PiCCSInvocations.roundWitnessStart := by
    simp only [PiCCSInvocations.statementWitnessStart,
      PiCCSInvocations.roundWitnessStart,
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart_eq,
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptWitnessStart_eq]
  unfold piCcsWitnessStarts
  rw [← challengeStart,
    ← sequentialWitnessStarts_add PiCCSInvocations.statementWitnessStart 379 87]
  rw [← roundStart,
    ← sequentialWitnessStarts_add PiCCSInvocations.statementWitnessStart
      (379 + 87) 252]

private theorem piCcsWitnessStarts_length :
    (piCcsWitnessStarts ()).length = 7604 := by
  rw [piCcsWitnessStarts_transcriptPrefix, List.length_append,
    sequentialWitnessStarts_length, sequentialWitnessStarts_length]

/-! ## Canonical PiRLC sampler blocks -/

def samplerEntryBlock (source : Nat) : ActionBlock :=
  ActionBlock.ofActions PiRLCSamplerInvocations.phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.entryRowStart source)
    (PiRLCSamplerInvocations.sourceLogicalStart source)
    (PiRLCSamplerInvocations.fastEntryState
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
      source)
    (TranscriptAbsorption.actions source)

def samplerWindowBlock (source round : Nat) : DirectBlock :=
  DirectBlock.ofState PiRLCSamplerInvocations.phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationRowStart
      source round)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
      source round)
    (PiRLCSamplerInvocations.fastWindowState
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
      source round)

theorem samplerEntryBlock_expand (source : Nat) :
    (samplerEntryBlock source).expand =
      PiRLCSamplerInvocations.entryInvocations
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source := by
  calc
    (samplerEntryBlock source).expand =
        (Invocations.compileActions PiRLCSamplerInvocations.phase
          (NightstreamFPrime.Layout.Stage1.PiRLCStarts.entryRowStart source)
          (PiRLCSamplerInvocations.sourceLogicalStart source)
          (PiRLCSamplerInvocations.fastEntryState
            (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
            source)
          (TranscriptAbsorption.actions source)).invocations := by
      simpa [samplerEntryBlock] using
        ActionBlock.ofActions_expand PiRLCSamplerInvocations.phase
          (NightstreamFPrime.Layout.Stage1.PiRLCStarts.entryRowStart source)
          (PiRLCSamplerInvocations.sourceLogicalStart source)
          (PiRLCSamplerInvocations.fastEntryState
            (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
            source)
          (TranscriptAbsorption.actions source)
    _ = _ := by rfl

theorem samplerWindowBlock_expand (source round : Nat) :
    (samplerWindowBlock source round).expand =
      PiRLCSamplerInvocations.windowInvocation
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source round := by
  simpa [samplerWindowBlock, PiRLCSamplerInvocations.windowInvocation,
    PiRLCSamplerInvocations.fastWindowState_eq_windowState] using
    DirectBlock.ofState_expand PiRLCSamplerInvocations.phase
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationRowStart
        source round)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
        source round)
      (PiRLCSamplerInvocations.fastWindowState
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source round)

def samplerSourceBlocks (source : Nat) : List Block :=
  .actions (samplerEntryBlock source) ::
    (List.range PiRLCSamplerInvocations.digestRoundCount).map fun round =>
      .direct (samplerWindowBlock source round)

private theorem directBlocks_expand (blocks : List DirectBlock) :
    (blocks.map Block.direct).flatMap Block.expand =
      blocks.map DirectBlock.expand := by
  induction blocks with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp only [List.map_cons, List.flatMap_cons, Block.expand,
        inductionHypothesis, List.singleton_append]

theorem samplerSourceBlocks_expand (source : Nat) :
    (samplerSourceBlocks source).flatMap Block.expand =
      PiRLCSamplerInvocations.sourceInvocations
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source := by
  unfold samplerSourceBlocks PiRLCSamplerInvocations.sourceInvocations
    PiRLCSamplerInvocations.windowInvocations
  simp only [List.flatMap_cons, Block.expand]
  rw [samplerEntryBlock_expand]
  change _ ++
      (((List.range PiRLCSamplerInvocations.digestRoundCount).map
        (samplerWindowBlock source)).map Block.direct).flatMap Block.expand =
    _ ++ _
  rw [directBlocks_expand, List.map_map]
  apply congrArg₂ (· ++ ·) rfl
  apply List.map_congr_left
  intro round _member
  exact samplerWindowBlock_expand source round

def piRlcSamplerBlocks (_unit : Unit) : List Block :=
  (List.range PiRLCSamplerInvocations.sourceCount).flatMap
    samplerSourceBlocks

private theorem samplerBlocks_expand (sources : List Nat) :
    (sources.flatMap samplerSourceBlocks).flatMap Block.expand =
      sources.flatMap fun source =>
        PiRLCSamplerInvocations.sourceInvocations
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
          source := by
  induction sources with
  | nil => rfl
  | cons source sources inductionHypothesis =>
      simp only [List.flatMap_cons, List.flatMap_append]
      rw [samplerSourceBlocks_expand, inductionHypothesis]

theorem piRlcSamplerBlocks_expand :
    (piRlcSamplerBlocks ()).flatMap Block.expand =
      PiRLCSamplerInvocations.invocations
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) := by
  unfold piRlcSamplerBlocks PiRLCSamplerInvocations.invocations
  simpa only using
    samplerBlocks_expand (List.range PiRLCSamplerInvocations.sourceCount)

/-! ## Random-access sampler witness starts -/

def samplerStepsPerSource : Nat := 9

@[simp] theorem samplerStepsPerSource_eq : samplerStepsPerSource = 9 := by
  rfl

/-- One source-local block in entry-then-window order. -/
def samplerSourceBlockAt (source : Nat) (step : Fin samplerStepsPerSource) :
    Block :=
  if step.val = 0 then
    .actions (samplerEntryBlock source)
  else
    .direct (samplerWindowBlock source (step.val - 1))

/-- One source-local final invocation witness start after the canonical
Spartan permutation. -/
def samplerSourceWitnessStartAt (source : Nat)
    (step : Fin samplerStepsPerSource) : Nat :=
  if step.val = 0 then
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (PiRLCSamplerInvocations.sourceLogicalStart source)
  else
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
        source (step.val - 1))

theorem samplerSourceBlockAt_witnessStarts (source : Nat)
    (step : Fin samplerStepsPerSource) :
    ((samplerSourceBlockAt source step).expand.map
      fun invocation => invocation.witnessStart) =
      [samplerSourceWitnessStartAt source step] := by
  by_cases entry : step.val = 0
  · unfold samplerSourceBlockAt samplerSourceWitnessStartAt
    rw [if_pos entry, if_pos entry]
    change ((samplerEntryBlock source).expand.map
      fun invocation => invocation.witnessStart) = _
    rw [samplerEntryBlock_expand]
    simp [PiRLCSamplerInvocations.entryInvocations,
      PiRLCSamplerInvocations.entryTrace, Invocations.compileActions,
      Invocations.compileBlocks, TranscriptAbsorption.actions,
      TranscriptAbsorption.constantWords, TranscriptAbsorption.frameWords,
      Hash.inputChunks, Spec.Poseidon2.rate]
  · unfold samplerSourceBlockAt samplerSourceWitnessStartAt
    rw [if_neg entry, if_neg entry]
    simp [Block.expand, samplerWindowBlock, DirectBlock.expand,
      DirectBlock.ofState, Invocations.invocation_witnessStart]

theorem samplerSourceBlockAt_materializes (source : Nat) :
    List.ofFn (samplerSourceBlockAt source) = samplerSourceBlocks source := by
  change List.ofFn (fun step : Fin 9 => samplerSourceBlockAt source step) = _
  rw [List.ofFn_succ]
  unfold samplerSourceBlocks
  apply congrArg₂ List.cons
  · simp [samplerSourceBlockAt]
  · rw [List.ofFn_eq_map, ← List.map_coe_finRange_eq_range, List.map_map]
    apply List.map_congr_left
    intro step _member
    simp [samplerSourceBlockAt, samplerStepsPerSource]

private theorem ofFn_decodeProd_eq_range_flatMap {Alpha : Type}
    (m n : Nat) (value : Nat → Fin n → Alpha) :
    List.ofFn (fun index : Fin (m * n) =>
        let decoded : Fin m × Fin n := Fin.decodeProd index
        value decoded.1.val decoded.2) =
      (List.range m).flatMap fun outer =>
        List.ofFn fun inner : Fin n => value outer inner := by
  rw [List.ofFn_mul]
  simp only [List.flatten_eq_flatMap]
  rw [List.ofFn_eq_map]
  rw [List.flatMap_map]
  rw [← List.map_coe_finRange_eq_range]
  rw [List.flatMap_map]
  apply List.flatMap_congr
  intro outer _member
  simp only [id_eq]
  apply congrArg List.ofFn
  funext inner
  let combined : Fin (m * n) :=
    ⟨outer.val * n + inner.val, by
      calc
        outer.val * n + inner.val < (outer.val + 1) * n := by
          simpa [Nat.add_mul] using Nat.add_lt_add_left inner.isLt (outer.val * n)
        _ ≤ m * n := Nat.mul_le_mul_right n outer.isLt⟩
  change value (Fin.decodeProd combined).1.val (Fin.decodeProd combined).2 =
    value outer.val inner
  have combined_eq : combined = Fin.encodeProd (outer, inner) := by
    apply Fin.ext
    simp [combined, Fin.encodeProd, Nat.mul_comm]
  rw [combined_eq, Fin.decodeProd_encodeProd]

def samplerBlockAt
    (index : Fin (PiRLCSamplerInvocations.sourceCount *
      samplerStepsPerSource)) : Block :=
  let decoded : Fin PiRLCSamplerInvocations.sourceCount ×
      Fin samplerStepsPerSource := Fin.decodeProd index
  samplerSourceBlockAt decoded.1.val decoded.2

def samplerWitnessStartAt
    (index : Fin (PiRLCSamplerInvocations.sourceCount *
      samplerStepsPerSource)) : Nat :=
  let decoded : Fin PiRLCSamplerInvocations.sourceCount ×
      Fin samplerStepsPerSource := Fin.decodeProd index
  samplerSourceWitnessStartAt decoded.1.val decoded.2

theorem samplerBlockAt_materializes :
    List.ofFn samplerBlockAt = piRlcSamplerBlocks () := by
  calc
    List.ofFn samplerBlockAt =
        (List.range PiRLCSamplerInvocations.sourceCount).flatMap fun source =>
          List.ofFn fun step : Fin samplerStepsPerSource =>
            samplerSourceBlockAt source step :=
      ofFn_decodeProd_eq_range_flatMap PiRLCSamplerInvocations.sourceCount
        samplerStepsPerSource samplerSourceBlockAt
    _ = (List.range PiRLCSamplerInvocations.sourceCount).flatMap
        samplerSourceBlocks := by
      apply List.flatMap_congr
      intro source _member
      exact samplerSourceBlockAt_materializes source
    _ = piRlcSamplerBlocks () := rfl

private theorem ofFn_flatMap_witnessStarts {count : Nat}
    (block : Fin count → Block) (value : Fin count → Nat)
    (each : ∀ index,
      ((block index).expand.map fun invocation => invocation.witnessStart) =
        [value index]) :
    (List.ofFn block).flatMap
        (fun current => current.expand.map fun invocation =>
          invocation.witnessStart) =
      List.ofFn value := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ (f := block), List.flatMap_cons,
        List.ofFn_succ (f := value)]
      have head := each (0 : Fin (count + 1))
      rw [head, List.singleton_append]
      apply congrArg (fun tail => value 0 :: tail)
      exact inductionHypothesis
        (fun index => block index.succ) (fun index => value index.succ)
        (fun index => each index.succ)

/-- The random-access witness-start schedule materializes to the exact
canonical sampler invocation list. -/
theorem samplerWitnessStartAt_materializes :
    List.ofFn samplerWitnessStartAt =
      (PiRLCSamplerInvocations.invocations
        (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)).map
          (fun invocation => invocation.witnessStart) := by
  calc
    List.ofFn samplerWitnessStartAt =
        (List.ofFn samplerBlockAt).flatMap
          (fun current => current.expand.map fun invocation =>
            invocation.witnessStart) := by
      exact (ofFn_flatMap_witnessStarts samplerBlockAt samplerWitnessStartAt
        (fun index => by
          exact samplerSourceBlockAt_witnessStarts
            (Fin.decodeProd index).1.val (Fin.decodeProd index).2)).symm
    _ = (piRlcSamplerBlocks ()).flatMap
          (fun current => current.expand.map fun invocation =>
            invocation.witnessStart) := by
      rw [samplerBlockAt_materializes]
    _ = ((piRlcSamplerBlocks ()).flatMap Block.expand).map
          (fun invocation => invocation.witnessStart) := by
      rw [List.map_flatMap]
    _ = (PiRLCSamplerInvocations.invocations
          (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits)).map
            (fun invocation => invocation.witnessStart) := by
      rw [piRlcSamplerBlocks_expand]

/-- Lightweight exact witness-start schedule for every non-pilot invocation. -/
def canonicalWitnessStarts (_unit : Unit) : List Nat :=
  piCcsWitnessStarts () ++ List.ofFn samplerWitnessStartAt

theorem canonicalWitnessStarts_materializes :
    canonicalWitnessStarts () =
      (Data.permutationInvocations ()).map
        (fun invocation => invocation.witnessStart) := by
  unfold canonicalWitnessStarts
  rw [Data.permutationInvocations_eq, List.map_append,
    piCcsWitnessStarts_materializes, samplerWitnessStartAt_materializes]

private theorem canonicalWitnessStarts_transcript_getD
    (index : Nat) (bound : index < 718) :
    (canonicalWitnessStarts ()).getD index 0 =
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        (NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset +
          index * 592) := by
  rw [canonicalWitnessStarts, List.getD_append _ _ _ _ (by
    rw [piCcsWitnessStarts_length]
    omega)]
  rw [piCcsWitnessStarts_transcriptPrefix, List.getD_append _ _ _ _ (by
    rw [sequentialWitnessStarts_length]
    exact bound)]
  simpa only [PiCCSInvocations.statementWitnessStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementWitnessStart] using
      sequentialWitnessStarts_getD PiCCSInvocations.statementWitnessStart
        718 index bound

/-- The selected pre-ordinary PiCCS invocation has its exact affine source
address. The proof uses the structural schedule and does not expand it. -/
theorem canonicalInvocation_witnessStart_of_transcript
    (index : Fin (Data.permutationInvocations ()).length)
    (bound : index.val < 718) :
    ((Data.permutationInvocations ()).get index).witnessStart =
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
        (NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset +
          index.val * 592) := by
  have selected := canonicalWitnessStarts_transcript_getD index.val bound
  rw [canonicalWitnessStarts_materializes] at selected
  have mappedBound : index.val <
      ((Data.permutationInvocations ()).map
        (fun invocation => invocation.witnessStart)).length := by
    simpa only [List.length_map] using index.isLt
  rw [List.getD_eq_getElem (l := _) (d := 0) mappedBound] at selected
  simpa only [List.getElem_map, List.get_eq_getElem] using selected

def canonicalBlocks (_unit : Unit) : List Block :=
  piCcsBlocks () ++ piRlcSamplerBlocks ()

/-- The generative plan expands to the exact package invocation list. -/
theorem canonicalBlocks_expand :
    (canonicalBlocks ()).flatMap Block.expand =
      Data.permutationInvocations () := by
  rw [canonicalBlocks, List.flatMap_append, piCcsBlocks_expand,
    piRlcSamplerBlocks_expand, Data.permutationInvocations_eq]

end NightstreamFPrime.Export.Stage1.PermutationPlan
