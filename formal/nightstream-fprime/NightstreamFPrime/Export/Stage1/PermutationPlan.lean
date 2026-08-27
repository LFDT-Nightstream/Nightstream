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
    _ = _ := by
      rw [PiRLCSamplerInvocations.fastEntryState_eq_entryState]
      rfl

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

def canonicalBlocks (_unit : Unit) : List Block :=
  piCcsBlocks () ++ piRlcSamplerBlocks ()

/-- The generative plan expands to the exact package invocation list. -/
theorem canonicalBlocks_expand :
    (canonicalBlocks ()).flatMap Block.expand =
      Data.permutationInvocations () := by
  rw [canonicalBlocks, List.flatMap_append, piCcsBlocks_expand,
    piRlcSamplerBlocks_expand, Data.permutationInvocations_eq]

end NightstreamFPrime.Export.Stage1.PermutationPlan
