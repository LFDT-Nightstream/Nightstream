import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
import NightstreamFPrime.Gadgets.SumCheck.FixedChain
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
import NightstreamFPrime.Spec.Folding.PiCCS.Transcript

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Step 2; Fiat–Shamir transform.
Obligation: For every SumCheck round, absorb the fixed-width prover message
before deriving that round's verifier challenge.

Inputs:
- the state after `α` and `γ` derivation;
- 28 fixed-width SumCheck polynomial messages.

Outputs:
- 28 verifier-derived challenges, carried in the shared `FixedChain.Round`
  interfaces;
- the post-round transcript state.

Constraint groups:
- C1: absorb `len(round-index || coefficients)` for each indexed round;
- C2: absorb label `[3, i]`;
- C3: derive `r_i` from the next Duplex squeeze;
- C4: expose the owned final eight-lane state.

Parent coverage:
- the round suffix of `v1_1.Coverage.transcript`;
- `Key.piCcsExecution.coins.roundPoint` and `.finalState`.

The generic Duplex child owns Poseidon2 operations. The generic FixedChain
child owns SumCheck equations. This leaf owns only transcript order and shared
wire mapping.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev State := NightstreamFPrime.Lifecycle.Transcript.State
abbrev EState := Layer.EState
abbrev Context :=
  NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Statement
    K State productionShape

def oracle : FiatShamir.Oracle Context K State productionShape :=
  NightstreamFPrime.Lifecycle.Transcript.piCcsOracle.transcript

/-- One prover-supplied SumCheck polynomial message. It contains no verifier
challenge. -/
structure Message (degreeBound : Nat) where
  coefficient : Fin (degreeBound + 1) → KExpr

def Message.asRound {degreeBound : Nat} (message : Message degreeBound)
    (challenge : KExpr) :
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round degreeBound where
  coefficient := message.coefficient
  challenge := challenge

def Message.semanticPolynomial {degreeBound : Nat} (message : Message degreeBound)
    (env : Env) : NightstreamFPrime.Spec.SumCheck.Finite.FixedPolynomial K degreeBound :=
  (message.asRound KExpr.zero).semanticPolynomial env

/-- External inputs are the prior state and 28 prover messages. Challenges
and the outgoing state are child-owned outputs. -/
structure Interface (degreeBound : Nat) where
  initialState : Nat → EState
  round : Nat → Fin productionShape.cubeVariables → Message degreeBound

def serializeKExpr (value : KExpr) : List Expr := [value.c0, value.c1]

def serializeKExprs (values : List KExpr) : List Expr :=
  values.flatMap serializeKExpr

def serializeRoundExpr {degreeBound : Nat} (round : Message degreeBound) :
    List Expr :=
  serializeKExprs (List.ofFn round.coefficient)

def constantWords (words : List F) : List Expr := words.map Expr.const

def blockExpr (words : List Expr) : List Expr :=
  Expr.const (NightstreamFPrime.Lifecycle.natWord words.length) :: words

/-- One paper round with an explicit expected sample. Layout uses zero only to
fix positions; production uses the compiler-owned sample at this position. -/
def roundActionsWithExpected {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (roundIndex : Fin productionShape.cubeVariables)
    (expected : KExpr) : List Formal.Action :=
  let round := interface.round offset roundIndex
  [.absorb (blockExpr
      (Expr.const (NightstreamFPrime.Lifecycle.natWord roundIndex.val) ::
        serializeRoundExpr round)),
    .absorb (constantWords (NightstreamFPrime.Lifecycle.Transcript.labelWord
      (.sumcheck roundIndex))),
    .squeezeK expected]

def layoutActions {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) : List Formal.Action :=
  (canonicalFinIndices productionShape.cubeVariables).flatMap fun roundIndex =>
    roundActionsWithExpected interface offset roundIndex KExpr.zero

def layoutProgram {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) : Formal.Program :=
  Formal.compile offset (interface.initialState offset)
    (layoutActions interface offset)

/-- Recipe-free executable projection of the fixed round transcript. -/
def layoutWiring {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) : Formal.Wiring :=
  Formal.compileWiringLazy offset (fun _ => interface.initialState offset)
    (layoutActions interface offset)

theorem layoutWiring_eq_compileWiring {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    layoutWiring interface offset =
      Formal.compileWiring offset (interface.initialState offset)
        (layoutActions interface offset) := by
  exact Formal.compileWiringLazy_eq offset
    (fun _ => interface.initialState offset) (interface.initialState offset)
    (layoutActions interface offset) rfl

theorem layoutWiring_samples_eq {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (layoutWiring interface offset).samples =
      (layoutProgram interface offset).samples := by
  calc
    (layoutWiring interface offset).samples =
        (Formal.compileWiring offset (interface.initialState offset)
          (layoutActions interface offset)).samples :=
      congrArg Formal.Wiring.samples
        (layoutWiring_eq_compileWiring interface offset)
    _ = (Formal.compile offset (interface.initialState offset)
          (layoutActions interface offset)).samples :=
      (Formal.compileWiring_matches offset (interface.initialState offset)
        (layoutActions interface offset)).1
    _ = (layoutProgram interface offset).samples := by
      rfl

private theorem layoutActions_squeezeCount {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    ((layoutActions interface offset).filterMap fun action => match action with
      | .absorb _ => none
      | .squeezeK _ => some ()).length =
      (canonicalFinIndices productionShape.cubeVariables).length := by
  unfold layoutActions
  generalize canonicalFinIndices productionShape.cubeVariables = indices
  induction indices with
  | nil => rfl
  | cons roundIndex indices inductionHypothesis =>
      rw [List.flatMap_cons, List.filterMap_append, List.length_append,
        inductionHypothesis]
      simp [roundActionsWithExpected]
      omega

@[simp] theorem layoutProgram_samples_length {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (layoutProgram interface offset).samples.length = 28 := by
  change (Formal.compile offset (interface.initialState offset)
    (layoutActions interface offset)).samples.length = 28
  rw [Formal.compile_samples_length]
  calc
    _ = (canonicalFinIndices productionShape.cubeVariables).length :=
      layoutActions_squeezeCount interface offset
    _ = 28 := canonicalFinIndices_length productionShape.cubeVariables

@[simp] theorem layoutWiring_samples_length {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (layoutWiring interface offset).samples.length = 28 := by
  rw [layoutWiring_samples_eq, layoutProgram_samples_length]

/-- Verifier-derived challenge for one indexed round. -/
def challenge {degreeBound : Nat} (interface : Interface degreeBound)
    (offset : Nat) (roundIndex : Fin productionShape.cubeVariables) : KExpr :=
  (layoutProgram interface offset).samples.get ⟨roundIndex.val, by
    have samplesLength := layoutProgram_samples_length interface offset
    rw [samplesLength]
    simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
      roundIndex.isLt⟩

/-- Executable verifier-derived round challenge from the recipe-free wiring. -/
def challengeFast {degreeBound : Nat} (interface : Interface degreeBound)
    (offset : Nat) (roundIndex : Fin productionShape.cubeVariables) : KExpr :=
  (layoutWiring interface offset).samples.getD roundIndex.val KExpr.zero

theorem challenge_eq_challengeFast_pointwise {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (roundIndex : Fin productionShape.cubeVariables) :
    challenge interface offset roundIndex =
      challengeFast interface offset roundIndex := by
  have roundBound : roundIndex.val <
      (layoutProgram interface offset).samples.length := by
    rw [layoutProgram_samples_length]
    simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
      roundIndex.isLt
  calc
    challenge interface offset roundIndex =
        (layoutProgram interface offset).samples.get
          ⟨roundIndex.val, roundBound⟩ := by
      rfl
    _ = (layoutProgram interface offset).samples.getD
          roundIndex.val KExpr.zero :=
      (List.getD_eq_get (layoutProgram interface offset).samples KExpr.zero
        ⟨roundIndex.val, roundBound⟩).symm
    _ = (layoutWiring interface offset).samples.getD
          roundIndex.val KExpr.zero := by
      rw [layoutWiring_samples_eq]
    _ = challengeFast interface offset roundIndex := by
      rfl

@[csimp] theorem challenge_eq_challengeFast : @challenge = @challengeFast := by
  funext degreeBound interface offset roundIndex
  exact challenge_eq_challengeFast_pointwise interface offset roundIndex

/-- The one round view consumed by the fixed SumCheck chain. -/
def round {degreeBound : Nat} (interface : Interface degreeBound)
    (offset : Nat) (roundIndex : Fin productionShape.cubeVariables) :
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round degreeBound :=
  (interface.round offset roundIndex).asRound
    (challenge interface offset roundIndex)

def roundActions {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (roundIndex : Fin productionShape.cubeVariables) : List Formal.Action :=
  roundActionsWithExpected interface offset roundIndex
    (challenge interface offset roundIndex)

def actions {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) : List Formal.Action :=
  (canonicalFinIndices productionShape.cubeVariables).flatMap fun roundIndex =>
    roundActions interface offset roundIndex

def program {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) : Formal.Program :=
  Formal.compile offset (interface.initialState offset)
    (actions interface offset)

def finalState {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) : EState :=
  (program interface offset).output

private theorem actions_shape_eq_layout_list {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (indices : List (Fin productionShape.cubeVariables)) :
    (indices.flatMap fun roundIndex =>
        roundActions interface offset roundIndex).map Formal.Action.shape =
      (indices.flatMap fun roundIndex =>
        roundActionsWithExpected interface offset roundIndex KExpr.zero).map
          Formal.Action.shape := by
  induction indices with
  | nil => rfl
  | cons roundIndex indices inductionHypothesis =>
      rw [List.flatMap_cons, List.flatMap_cons, List.map_append,
        List.map_append, inductionHypothesis]
      rfl

theorem actions_shape_eq_layout {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (actions interface offset).map Formal.Action.shape =
      (layoutActions interface offset).map Formal.Action.shape := by
  exact actions_shape_eq_layout_list interface offset _

theorem program_shape_eq_layout {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (program interface offset).recipes =
        (layoutProgram interface offset).recipes ∧
      (program interface offset).samples =
        (layoutProgram interface offset).samples ∧
      (program interface offset).output =
        (layoutProgram interface offset).output := by
  exact Formal.compile_shape_eq offset (interface.initialState offset)
    (actions interface offset) (layoutActions interface offset)
      (actions_shape_eq_layout interface offset)

/-- Executable final-state projection from the recipe-free round wiring. -/
def finalStateFast {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) : EState :=
  (layoutWiring interface offset).output

theorem finalState_eq_finalStateFast_pointwise {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    finalState interface offset = finalStateFast interface offset := by
  calc
    finalState interface offset = (program interface offset).output := by
      rfl
    _ = (layoutProgram interface offset).output :=
      (program_shape_eq_layout interface offset).2.2
    _ = (Formal.compileWiring offset (interface.initialState offset)
          (layoutActions interface offset)).output :=
      (Formal.compileWiring_matches offset (interface.initialState offset)
        (layoutActions interface offset)).2.symm
    _ = (layoutWiring interface offset).output :=
      congrArg Formal.Wiring.output
        (layoutWiring_eq_compileWiring interface offset).symm
    _ = finalStateFast interface offset := by
      rfl

@[csimp] theorem finalState_eq_finalStateFast :
    @finalState = @finalStateFast := by
  funext degreeBound interface offset
  exact finalState_eq_finalStateFast_pointwise interface offset

theorem layoutSamples_eq_challenges {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (layoutProgram interface offset).samples =
      (canonicalFinIndices productionShape.cubeVariables).map
        (challenge interface offset) := by
  apply List.ext_get
  · simp [layoutProgram_samples_length, canonicalFinIndices_length]
    norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
  · intro index leftLt rightLt
    simp [canonicalFinIndices, challenge]

private theorem expectedSamples_rounds {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (indices : List (Fin productionShape.cubeVariables)) :
    Formal.expectedSamples (indices.flatMap fun roundIndex =>
      roundActions interface offset roundIndex) =
      indices.map (challenge interface offset) := by
  induction indices with
  | nil => rfl
  | cons roundIndex indices inductionHypothesis =>
      rw [List.flatMap_cons]
      change [challenge interface offset roundIndex] ++
          Formal.expectedSamples (indices.flatMap fun current =>
            roundActions interface offset current) =
        challenge interface offset roundIndex ::
          indices.map (challenge interface offset)
      rw [inductionHypothesis]
      rfl

theorem expectedSamples_eq_samples {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    Formal.expectedSamples (actions interface offset) =
      (program interface offset).samples := by
  rw [actions, expectedSamples_rounds]
  rw [(program_shape_eq_layout interface offset).2.1]
  exact (layoutSamples_eq_challenges interface offset).symm

def duplexInterface {degreeBound : Nat}
    (interface : Interface degreeBound) : Formal.Interface where
  initial := interface.initialState
  actions := actions interface
  final := finalState interface

def evalState (env : Env) (state : EState) : State :=
  List.ofFn (Layer.evalState env state)

def semanticRounds {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) (env : Env) :
    Fin productionShape.cubeVariables → SumCheck.Finite.Message K :=
  fun roundIndex =>
    (interface.round offset roundIndex).semanticPolynomial env |>.toMessage

def evalRoundPoint {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) (env : Env) :
  CubePoint K productionShape.cubeVariables where
  coordinates := (canonicalFinIndices productionShape.cubeVariables).map
    fun roundIndex => (challenge interface offset roundIndex).eval env
  dimension := by
    rw [List.length_map, canonicalFinIndices_length]

def Assumptions {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) (_env : Env) : Prop :=
  (∀ lane, (interface.initialState offset lane).VarsBelow offset) ∧
    ∀ roundIndex coefficient,
      (interface.round offset roundIndex).coefficient coefficient |>.VarsBelow offset

private theorem serializeKExprs_below (values : List KExpr) (bound : Nat)
    (below : ∀ value ∈ values, value.VarsBelow bound) :
    ∀ expression ∈ serializeKExprs values, expression.VarsBelow bound := by
  induction values with
  | nil => simp [serializeKExprs]
  | cons value values inductionHypothesis =>
      intro expression member
      change expression ∈ serializeKExpr value ++ serializeKExprs values at member
      rw [List.mem_append] at member
      rcases member with head | tail
      · simp [serializeKExpr] at head
        rcases head with rfl | rfl
        · exact (below value (by simp)).1
        · exact (below value (by simp)).2
      · exact inductionHypothesis
          (fun current currentMember => below current (by simp [currentMember]))
          expression tail

private theorem serializeRoundExpr_below {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (assumptions : Assumptions interface offset (fun _ => 0))
    (roundIndex : Fin productionShape.cubeVariables) :
    ∀ expression ∈ serializeRoundExpr (interface.round offset roundIndex),
      expression.VarsBelow offset := by
  apply serializeKExprs_below
  intro value member
  rw [List.mem_ofFn'] at member
  rcases member with ⟨coefficient, rfl⟩
  exact assumptions.2 roundIndex coefficient

private theorem layoutRoundActions_below {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (assumptions : Assumptions interface offset (fun _ => 0))
    (roundIndex : Fin productionShape.cubeVariables) :
    Formal.ActionsBelow offset
      (roundActionsWithExpected interface offset roundIndex KExpr.zero) := by
  intro action member
  simp only [roundActionsWithExpected, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl | rfl
  · intro expression expressionMember
    unfold blockExpr at expressionMember
    rw [List.mem_cons] at expressionMember
    rcases expressionMember with rfl | expressionMember
    · trivial
    rw [List.mem_cons] at expressionMember
    rcases expressionMember with rfl | expressionMember
    · trivial
    exact serializeRoundExpr_below interface offset assumptions roundIndex
      expression expressionMember
  · intro expression expressionMember
    simp [constantWords] at expressionMember
    rcases expressionMember with ⟨_, _, rfl⟩
    trivial
  · exact ⟨trivial, trivial⟩

theorem layoutActions_below {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (assumptions : Assumptions interface offset (fun _ => 0)) :
    Formal.ActionsBelow offset (layoutActions interface offset) := by
  intro action member
  rw [layoutActions, List.mem_flatMap] at member
  rcases member with ⟨roundIndex, _, actionMember⟩
  exact layoutRoundActions_below interface offset assumptions roundIndex
    action actionMember

theorem program_causal {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    RecipesCausal offset (program interface offset).recipes := by
  rw [(program_shape_eq_layout interface offset).1]
  apply Formal.compile_causal offset (interface.initialState offset)
    (layoutActions interface offset) assumptions.1
  exact layoutActions_below interface offset (by
    simpa [Assumptions] using assumptions)

def SpecHolds {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) (env : Env) : Prop :=
  NightstreamFPrime.Spec.Folding.PiCCS.Transcript.RoundsHolds
    oracle (semanticRounds interface offset env)
    (evalState env (interface.initialState offset))
    (evalRoundPoint interface offset env)
    (evalState env (finalState interface offset))

private theorem eval_constantWords (env : Env) (words : List F) :
    Hash.evalList env (constantWords words) = words := by
  simp [Hash.evalList, constantWords, Function.comp_def]

private theorem eval_serializeKExprs (env : Env) (values : List KExpr) :
    Hash.evalList env (serializeKExprs values) =
      (values.map (KExpr.eval env)).flatMap
        NightstreamFPrime.Lifecycle.serializeK := by
  induction values with
  | nil => rfl
  | cons value values inductionHypothesis =>
      calc
        Hash.evalList env (serializeKExprs (value :: values)) =
            Hash.evalList env (serializeKExpr value) ++
              Hash.evalList env (serializeKExprs values) := by
          simp [serializeKExprs, Hash.evalList]
        _ = NightstreamFPrime.Lifecycle.serializeK (value.eval env) ++
              (values.map (KExpr.eval env)).flatMap
                NightstreamFPrime.Lifecycle.serializeK := by
          rw [inductionHypothesis]
          rfl
        _ = _ := rfl

private theorem eval_serializeRoundExpr {degreeBound : Nat}
    (env : Env)
    (round : Message degreeBound) :
    Hash.evalList env (serializeRoundExpr round) =
      NightstreamFPrime.Lifecycle.Transcript.serializeMessage
        (round.semanticPolynomial env).toMessage := by
  unfold serializeRoundExpr
  rw [eval_serializeKExprs]
  rfl

private theorem eval_roundPayload {degreeBound : Nat}
    (env : Env) (roundIndex : Fin productionShape.cubeVariables)
    (round : Message degreeBound) :
    Hash.evalList env
        (Expr.const (NightstreamFPrime.Lifecycle.natWord roundIndex.val) ::
          serializeRoundExpr round) =
      NightstreamFPrime.Lifecycle.natWord roundIndex.val ::
        NightstreamFPrime.Lifecycle.Transcript.serializeMessage
          (round.semanticPolynomial env).toMessage := by
  simpa [Hash.evalList] using congrArg
    (List.cons (NightstreamFPrime.Lifecycle.natWord roundIndex.val))
    (eval_serializeRoundExpr env round)

private theorem eval_blockExpr (env : Env) (words : List Expr) :
    Hash.evalList env (blockExpr words) =
      NightstreamFPrime.Lifecycle.block (Hash.evalList env words) := by
  simp [Hash.evalList, blockExpr, NightstreamFPrime.Lifecycle.block]

private theorem roundActions_trace_iff {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) (env : Env)
    (state final : State)
    (roundIndex : Fin productionShape.cubeVariables)
    (tail : List Formal.Action) :
    Formal.TraceHolds state
        ((roundActions interface offset roundIndex ++ tail).map
          (Formal.Action.eval env)) final ↔
      let absorbed := oracle.absorbRound state roundIndex
        (semanticRounds interface offset env roundIndex)
      let sample := oracle.squeeze absorbed (.sumcheck roundIndex)
      (challenge interface offset roundIndex).eval env = sample.1 ∧
        Formal.TraceHolds sample.2
          (tail.map (Formal.Action.eval env)) final := by
  simp [roundActions, roundActionsWithExpected, Formal.Action.eval,
    Formal.TraceHolds,
    eval_constantWords, eval_blockExpr, eval_roundPayload,
    semanticRounds, oracle,
    NightstreamFPrime.Lifecycle.Transcript.piCcsOracle,
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Absorb.reference,
    NightstreamFPrime.Lifecycle.Transcript.absorb,
    NightstreamFPrime.Lifecycle.Transcript.absorbBlock,
    NightstreamFPrime.Lifecycle.block,
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Squeeze.referenceSample,
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Squeeze.referenceState,
    NightstreamFPrime.Lifecycle.Transcript.squeezeK,
    NightstreamFPrime.Lifecycle.Transcript.squeezeF, Hash.inputChunks]

private theorem rounds_trace_iff {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) (env : Env)
    (state final : State)
    (indices : List (Fin productionShape.cubeVariables))
    (tail : List Formal.Action) :
    Formal.TraceHolds state
        (((indices.flatMap fun roundIndex =>
          roundActions interface offset roundIndex) ++ tail).map
            (Formal.Action.eval env)) final ↔
      let result := FiatShamir.deriveRoundsFrom oracle
        (semanticRounds interface offset env) state indices
      indices.map (fun roundIndex =>
        (challenge interface offset roundIndex).eval env) = result.1 ∧
        Formal.TraceHolds result.2
          (tail.map (Formal.Action.eval env)) final := by
  induction indices generalizing state with
  | nil => simp [FiatShamir.deriveRoundsFrom]
  | cons roundIndex indices inductionHypothesis =>
      rw [List.flatMap_cons, List.append_assoc,
        roundActions_trace_iff]
      dsimp only
      rw [inductionHypothesis]
      simp only [List.map_cons, FiatShamir.deriveRoundsFrom,
        List.cons.injEq]
      aesop

private theorem rounds_trace_terminal_iff {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) (env : Env)
    (state final : State) :
    Formal.TraceHolds state
        ((actions interface offset).map (Formal.Action.eval env)) final ↔
      let result := FiatShamir.deriveRoundsFrom oracle
        (semanticRounds interface offset env) state
        (canonicalFinIndices productionShape.cubeVariables)
      (evalRoundPoint interface offset env).coordinates = result.1 ∧
        final = result.2 := by
  unfold actions
  have replay := rounds_trace_iff interface offset env state final
    (canonicalFinIndices productionShape.cubeVariables) []
  simpa [evalRoundPoint, Formal.TraceHolds, eq_comm] using replay

theorem trace_iff_specHolds {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) (env : Env) :
    Formal.TraceHolds
      (evalState env (interface.initialState offset))
      ((actions interface offset).map (Formal.Action.eval env))
      (evalState env (finalState interface offset)) ↔
      SpecHolds interface offset env := by
  constructor
  · intro trace
    rcases (rounds_trace_terminal_iff interface offset env
      (evalState env (interface.initialState offset))
      (evalState env (finalState interface offset))).1 trace with
      ⟨roundPointEq, finalStateEq⟩
    exact ⟨roundPointEq, finalStateEq⟩
  · rintro ⟨roundPointEq, finalStateEq⟩
    exact (rounds_trace_terminal_iff interface offset env
      (evalState env (interface.initialState offset))
      (evalState env (finalState interface offset))).2
        ⟨roundPointEq, finalStateEq⟩

def opsAt {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset
    (program interface offset).recipes)]

def main {degreeBound : Nat}
    (interface : Interface degreeBound) : Circuit Unit := fun offset =>
  ((), offset + (program interface offset).recipes.length,
    opsAt interface offset)

@[simp] theorem main_ops {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

@[simp] theorem opsAt_localLength {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    localLength (opsAt interface offset) =
      (program interface offset).recipes.length := by
  simp [opsAt, localLength, Op.localLength]

@[simp] theorem flatConstraints_opsAt {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (program interface offset).recipes := by
  simp [opsAt, flatConstraints, Op.flatConstraints]

theorem build {degreeBound : Nat}
    (interface : Interface degreeBound) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (program interface offset).recipes.length ∧
      holdsFlat completed (opsAt interface offset) := by
  let compiled := program interface offset
  let completed := executeRecipes env offset compiled.recipes
  have causal : RecipesCausal offset compiled.recipes :=
    program_causal interface offset assumptions
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset compiled.recipes) :=
    executeRecipes_holds_recipeConstraints env offset compiled.recipes causal
  refine ⟨completed, executeRecipes_agreesOutside env offset compiled.recipes, ?_⟩
  change ConstraintsHold completed (flatConstraints (opsAt interface offset))
  rw [flatConstraints_opsAt]
  exact recipeRows

def circuit {degreeBound : Nat}
    (interface : Interface degreeBound) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := by
    intro env offset _assumptions rows
    let compiled := program interface offset
    have recipeRows : ConstraintsHold env
        (recipeConstraints offset compiled.recipes) := by
      exact rows (Op.witness (WitnessBatch.arithmetic offset compiled.recipes)) (by
        change Op.witness (WitnessBatch.arithmetic offset compiled.recipes) ∈
          opsAt interface offset
        simp [opsAt, compiled])
    have assertionRows : ConstraintsHold env compiled.assertions := by
      apply (Formal.compile_assertions_hold_iff env offset
        (interface.initialState offset) (actions interface offset)).2
      rw [expectedSamples_eq_samples]
      rfl
    have trace := Formal.compile_sound env offset
      (interface.initialState offset) (actions interface offset)
        recipeRows assertionRows
    exact (trace_iff_specHolds interface offset env).1 trace
  completeness := by
    intro env offset assumptions _specification
    simpa only [main_ops, opsAt_localLength] using
      build interface env offset assumptions

theorem soundness {degreeBound : Nat}
    (interface : Interface degreeBound) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  (circuit interface).soundness env offset assumptions rows

theorem completeness {degreeBound : Nat}
    (interface : Interface degreeBound) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  (circuit interface).completeness env offset assumptions specification

private theorem cubePoint_eq_of_coordinates_owned
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem fixedPolynomial_eq_of_coefficients
    {Field : Type} {degree : Nat}
    (left right : SumCheck.Finite.FixedPolynomial Field degree)
    (coefficients : left.coefficients = right.coefficients) : left = right := by
  cases left
  cases right
  simp_all

theorem specHolds_of_agree_below {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index,
      index < offset + (program interface offset).recipes.length →
        after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have initialEq : evalState after (interface.initialState offset) =
      evalState before (interface.initialState offset) := by
    apply congrArg List.ofFn
    funext lane
    exact (interface.initialState offset lane).eval_eq_of_agree_below offset
      after before (assumptions.1 lane)
        (fun index below => agrees index (by omega))
  have semanticRoundsEq : semanticRounds interface offset after =
      semanticRounds interface offset before := by
    funext roundIndex
    apply congrArg SumCheck.Finite.FixedPolynomial.toMessage
    apply fixedPolynomial_eq_of_coefficients
    change (List.ofFn (interface.round offset roundIndex).coefficient).map
        (KExpr.eval after) =
      (List.ofFn (interface.round offset roundIndex).coefficient).map
        (KExpr.eval before)
    apply List.map_congr_left
    intro coefficient member
    rw [List.mem_ofFn'] at member
    rcases member with ⟨index, rfl⟩
    exact ((interface.round offset roundIndex).coefficient index).eval_eq_of_agree_below
      offset after before (assumptions.2 roundIndex index)
        (fun current below => agrees current (by omega))
  have roundPointEq : evalRoundPoint interface offset after =
      evalRoundPoint interface offset before := by
    apply cubePoint_eq_of_coordinates_owned
    change (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex => (challenge interface offset roundIndex).eval after) =
      (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex => (challenge interface offset roundIndex).eval before)
    apply List.map_congr_left
    intro roundIndex _
    have sampleScope := Formal.compile_samples_scope offset
      (interface.initialState offset) (layoutActions interface offset)
      assumptions.1 (layoutActions_below interface offset (by
        simpa [Assumptions] using assumptions))
    have below := sampleScope (challenge interface offset roundIndex)
      (List.get_mem _ _)
    have recipesEq := (program_shape_eq_layout interface offset).1
    have belowProgram : (challenge interface offset roundIndex).VarsBelow
        (offset + (program interface offset).recipes.length) := by
      rw [recipesEq]
      exact below
    exact (challenge interface offset roundIndex).eval_eq_of_agree_below
      (offset + (program interface offset).recipes.length) after before
      belowProgram agrees
  have finalEq : evalState after (finalState interface offset) =
      evalState before (finalState interface offset) := by
    apply congrArg List.ofFn
    funext lane
    have outputScope := (Formal.compile_scope offset
      (interface.initialState offset) (layoutActions interface offset)
      assumptions.1 (layoutActions_below interface offset (by
        simpa [Assumptions] using assumptions))).1 lane
    have recipesEq := (program_shape_eq_layout interface offset).1
    have outputEq := (program_shape_eq_layout interface offset).2.2
    change ((program interface offset).output lane).eval after =
      ((program interface offset).output lane).eval before
    rw [outputEq]
    exact ((layoutProgram interface offset).output lane).eval_eq_of_agree_below
      (offset + (layoutProgram interface offset).recipes.length) after before
      outputScope (by
        intro index below
        apply agrees index
        rw [recipesEq]
        exact below)
  unfold SpecHolds at specification ⊢
  rw [semanticRoundsEq, initialEq, roundPointEq, finalEq]
  exact specification

theorem flatConstraints_varsBelow {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  have causal := program_causal interface offset assumptions
  have scope := recipeConstraints_varsBelow_of_causal offset
    (program interface offset).recipes causal
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + localLength (opsAt interface offset))
  rw [flatConstraints_opsAt, opsAt_localLength]
  exact scope

/-- Every verifier-derived round challenge lies inside this transcript
child's declared symbolic interval. -/
theorem challenge_varsBelow {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (roundIndex : Fin productionShape.cubeVariables) :
    (challenge interface offset roundIndex).VarsBelow
      (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  have sampleScope := Formal.compile_samples_scope offset
    (interface.initialState offset) (layoutActions interface offset)
      assumptions.1 (layoutActions_below interface offset (by
        simpa [Assumptions] using assumptions))
  have below := sampleScope (challenge interface offset roundIndex)
    (List.get_mem _ _)
  have recipesEq := (program_shape_eq_layout interface offset).1
  change (challenge interface offset roundIndex).VarsBelow
    (offset + (program interface offset).recipes.length)
  rw [recipesEq]
  exact below

/-- The compiler-owned outgoing transcript state lies inside this child's
declared symbolic interval. -/
theorem finalState_varsBelow {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ lane, (finalState interface offset lane).VarsBelow
      (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  have outputScope := (Formal.compile_scope offset
    (interface.initialState offset) (layoutActions interface offset)
      assumptions.1 (layoutActions_below interface offset (by
        simpa [Assumptions] using assumptions))).1
  have recipesEq := (program_shape_eq_layout interface offset).1
  have outputEq := (program_shape_eq_layout interface offset).2.2
  intro lane
  change ((program interface offset).output lane).VarsBelow
    (offset + localLength (opsAt interface offset))
  rw [opsAt_localLength, recipesEq, outputEq]
  exact outputScope lane

private theorem actions_length_list {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (indices : List (Fin productionShape.cubeVariables)) :
    (indices.flatMap fun roundIndex =>
      roundActions interface offset roundIndex).length = 3 * indices.length := by
  induction indices with
  | nil => rfl
  | cons roundIndex indices inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append, inductionHypothesis]
      simp [roundActions, roundActionsWithExpected]
      omega

theorem actions_length {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (actions interface offset).length = 84 := by
  rw [actions, actions_length_list, canonicalFinIndices_length]
  rfl

def perRoundRecipeCount (degreeBound : Nat) : Nat :=
  (((2 * degreeBound + 7) / 4) + 3) * 592

private theorem serializeKExprs_length (values : List KExpr) :
    (serializeKExprs values).length = 2 * values.length := by
  induction values with
  | nil => rfl
  | cons value values inductionHypothesis =>
      simp [serializeKExprs, serializeKExpr]
      omega

private theorem serializeRoundExpr_length {degreeBound : Nat}
    (round : Message degreeBound) :
    (serializeRoundExpr round).length = 2 * (degreeBound + 1) := by
  rw [serializeRoundExpr, serializeKExprs_length]
  simp

private theorem inputChunks_length (input : List Expr) :
    (Hash.inputChunks input).length = (input.length + 3) / 4 := by
  unfold Hash.inputChunks
  rw [List.length_map, List.length_range]
  rfl

private theorem absorb_recipeCount (input : List Expr) :
    Formal.Action.recipeCount (.absorb input) =
      ((input.length + 3) / 4) * 592 := by
  change (Hash.inputChunks input).length * 592 = _
  rw [inputChunks_length]

private theorem constantWords_length (words : List F) :
    (constantWords words).length = words.length := by
  simp [constantWords]

private theorem blockExpr_length (words : List Expr) :
    (blockExpr words).length = words.length + 1 := by
  simp [blockExpr]

private theorem roundPayload_length {degreeBound : Nat}
    (roundIndex : Fin productionShape.cubeVariables)
    (round : Message degreeBound) :
    (Expr.const (NightstreamFPrime.Lifecycle.natWord roundIndex.val) ::
      serializeRoundExpr round).length = 2 * degreeBound + 3 := by
  rw [List.length_cons, serializeRoundExpr_length]
  omega

private theorem messageAbsorb_recipeCount {degreeBound : Nat}
    (roundIndex : Fin productionShape.cubeVariables)
    (round : Message degreeBound) :
    Formal.Action.recipeCount (.absorb (blockExpr
      (Expr.const (NightstreamFPrime.Lifecycle.natWord roundIndex.val) ::
        serializeRoundExpr round))) =
      ((2 * degreeBound + 7) / 4) * 592 := by
  rw [absorb_recipeCount, blockExpr_length,
    roundPayload_length roundIndex round]

private theorem labelAbsorb_recipeCount
    (roundIndex : Fin productionShape.cubeVariables) :
    Formal.Action.recipeCount (.absorb (constantWords
      (NightstreamFPrime.Lifecycle.Transcript.labelWord
        (.sumcheck roundIndex)))) = 592 := by
  rw [absorb_recipeCount, constantWords_length]
  simp [NightstreamFPrime.Lifecycle.Transcript.labelWord]

private theorem roundActions_recipeCount {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat)
    (roundIndex : Fin productionShape.cubeVariables) :
    Formal.recipeCount (roundActions interface offset roundIndex) =
      perRoundRecipeCount degreeBound := by
  unfold roundActions
  unfold roundActionsWithExpected
  dsimp only
  unfold Formal.recipeCount
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Nat.add_zero]
  rw [messageAbsorb_recipeCount, labelAbsorb_recipeCount]
  simp only [Formal.Action.recipeCount]
  unfold perRoundRecipeCount
  omega

def recipeCount {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) : Nat :=
  Formal.recipeCount (actions interface offset)

/-- Exact symbolic footprint: 28 indexed round groups, with no copied round
implementation. -/
theorem recipeCount_eq {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    recipeCount interface offset =
      productionShape.cubeVariables * perRoundRecipeCount degreeBound := by
  unfold recipeCount actions
  have each : ∀ roundIndex ∈
      canonicalFinIndices productionShape.cubeVariables,
      Formal.recipeCount (roundActions interface offset roundIndex) =
        perRoundRecipeCount degreeBound := by
    intro roundIndex _
    exact roundActions_recipeCount interface offset roundIndex
  rw [Formal.recipeCount_flatMap_constant _ _ _ each]
  rw [canonicalFinIndices_length]

theorem localLength_eq {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) =
      productionShape.cubeVariables * perRoundRecipeCount degreeBound := by
  change localLength (opsAt interface offset) = _
  rw [opsAt_localLength]
  unfold program
  rw [Formal.compile_recipes_length]
  simpa [program, recipeCount] using recipeCount_eq interface offset

@[simp] theorem program_recipes_length {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (program interface offset).recipes.length =
      productionShape.cubeVariables * perRoundRecipeCount degreeBound := by
  unfold program
  rw [Formal.compile_recipes_length]
  simpa [recipeCount] using recipeCount_eq interface offset

theorem operations_length {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 := by
  rfl

theorem flatConstraints_length {degreeBound : Nat}
    (interface : Interface degreeBound) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      productionShape.cubeVariables * perRoundRecipeCount degreeBound := by
  change (flatConstraints (opsAt interface offset)).length = _
  rw [flatConstraints_opsAt, recipeConstraints_length,
    program_recipes_length]

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- Concrete parent coverage: the shared round messages and this transcript
leaf produce the production key's exact SumCheck point and pre-output state. -/
theorem spec_implies_keyExecution_rounds
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : Interface (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (initial_eq : evalState env (interface.initialState offset) =
      (FiatShamir.derivePreSumcheck oracle
        (ChallengeDerivation.productionContext relation ajtai running fresh)).state)
    (rounds_eq : ∀ roundIndex,
      (interface.round offset roundIndex).semanticPolynomial env =
        proof.piCcsRounds roundIndex)
    (specification : SpecHolds interface offset env) :
    evalRoundPoint interface offset env =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint ∧
      evalState env (finalState interface offset) =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.finalState := by
  let key := ProductionKey.key relation ajtai
  let context := ChallengeDerivation.productionContext
    relation ajtai running fresh
  let certificate : FiatShamir.Certificate K productionShape :=
    { rounds := fun roundIndex => (proof.piCcsRounds roundIndex).toMessage }
  have semanticRounds_eq : semanticRounds interface offset env =
      certificate.rounds := by
    funext roundIndex
    exact congrArg SumCheck.Finite.FixedPolynomial.toMessage
      (rounds_eq roundIndex)
  rcases specification with ⟨roundPointEq, finalStateEq⟩
  rw [initial_eq, semanticRounds_eq] at roundPointEq finalStateEq
  let derived := FiatShamir.derive oracle context certificate
  have canonicalRounds :=
    NightstreamFPrime.Spec.Folding.PiCCS.Transcript.derive_rounds_holds
      oracle context certificate
  have roundPointDerived : evalRoundPoint interface offset env =
      derived.roundPoint := by
    apply cubePoint_eq_of_coordinates
    exact roundPointEq.trans canonicalRounds.roundPoint_eq.symm
  have finalStateDerived : evalState env (finalState interface offset) =
      derived.finalState :=
    finalStateEq.trans canonicalRounds.finalState_eq.symm
  have coinsEq := key.piCcsExecution_coins_eq_derive running fresh proof
  have productionEq : (key.piCcsExecution running fresh proof).coins =
      derived := by
    simpa [key, context, certificate, derived,
      ChallengeDerivation.productionContext, oracle] using coinsEq
  exact ⟨
    roundPointDerived.trans
      (congrArg (fun coins => coins.roundPoint) productionEq).symm,
    finalStateDerived.trans
      (congrArg (fun coins => coins.finalState) productionEq).symm⟩

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript
