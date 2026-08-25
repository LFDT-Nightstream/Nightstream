import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.Transcript

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Step 1; Fiat–Shamir transform.
Obligation: Derive `α ∈ K^25`, then `γ ∈ K`, from the state after the
complete public statement absorption.

Inputs:
- the statement leaf's final Poseidon2 state.

Outputs:
- the state immediately before the first SumCheck round message.

Constraint groups:
- C1: absorb label `[1, i]`, then derive `α_i` from one Duplex squeeze;
- C2: absorb label `[2]`, then derive `γ` from one Duplex squeeze.

Parent coverage:
- the pre-SumCheck prefix of `v1_1.Coverage.transcript`;
- `Key.piCcsExecution.coins.alpha` and `.gamma`.

No challenge is an unconstrained witness value. The generic Duplex child owns
all Poseidon2 operations. This leaf owns only labels, order, and wiring.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation

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

/-- The exact production pre-SumCheck oracle. -/
def oracle : FiatShamir.Oracle Context K State productionShape :=
  NightstreamFPrime.Lifecycle.Transcript.piCcsOracle.transcript

/-- The only external input is the state produced by statement absorption. -/
structure Interface where
  initialState : Nat → EState

def constantWords (words : List F) : List Expr := words.map Expr.const

def labelActions (label : FiatShamir.ChallengeLabel productionShape)
    (expected : KExpr) : List Formal.Action :=
  [.absorb (constantWords
      (NightstreamFPrime.Lifecycle.Transcript.labelWord label)),
    .squeezeK expected]

/-- Pair labels and samples without copying a fixed constraint block. -/
def labelledActions :
    List (FiatShamir.ChallengeLabel productionShape) →
      List KExpr → List Formal.Action
  | label :: labels, sample :: samples =>
      labelActions label sample ++ labelledActions labels samples
  | _, _ => []

/-- Exact verifier-owned order: all 25 `α` labels, then `γ`. -/
def challengeLabels : List (FiatShamir.ChallengeLabel productionShape) :=
  FiatShamir.alphaLabels productionShape ++ [.gamma]

/-- The layout schedule fixes only sample positions. Its expected values have
no authority and do not affect recipes, samples, or final state. -/
def layoutActions : List Formal.Action :=
  labelledActions challengeLabels (List.replicate 26 KExpr.zero)

def layoutProgram (interface : Interface) (offset : Nat) : Formal.Program :=
  Formal.compile offset (interface.initialState offset) layoutActions

/-- Recipe-free executable projection of the fixed challenge schedule. The
incoming state remains delayed and is replaced by the first label absorb. -/
def layoutWiring (interface : Interface) (offset : Nat) : Formal.Wiring :=
  Formal.compileWiringLazy offset (fun _ => interface.initialState offset)
    layoutActions

theorem layoutWiring_eq_compileWiring (interface : Interface) (offset : Nat) :
    layoutWiring interface offset =
      Formal.compileWiring offset (interface.initialState offset)
        layoutActions := by
  exact Formal.compileWiringLazy_eq offset
    (fun _ => interface.initialState offset) (interface.initialState offset)
    layoutActions rfl

theorem layoutWiring_samples_eq (interface : Interface) (offset : Nat) :
    (layoutWiring interface offset).samples =
      (layoutProgram interface offset).samples := by
  calc
    (layoutWiring interface offset).samples =
        (Formal.compileWiring offset (interface.initialState offset)
          layoutActions).samples :=
      congrArg Formal.Wiring.samples
        (layoutWiring_eq_compileWiring interface offset)
    _ = (Formal.compile offset (interface.initialState offset)
          layoutActions).samples :=
      (Formal.compileWiring_matches offset (interface.initialState offset)
        layoutActions).1
    _ = (layoutProgram interface offset).samples := by
      rfl

def scheduleActions
    (schedule : List (FiatShamir.ChallengeLabel productionShape × KExpr)) :
    List Formal.Action :=
  schedule.flatMap fun item => labelActions item.1 item.2

private theorem scheduleActions_zip_eq_labelled
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (samples : List KExpr) (sameLength : samples.length = labels.length) :
    scheduleActions (labels.zip samples) = labelledActions labels samples := by
  induction labels generalizing samples with
  | nil =>
      have : samples = [] := List.eq_nil_of_length_eq_zero sameLength
      subst samples
      rfl
  | cons label labels inductionHypothesis =>
      cases samples with
      | nil => simp at sameLength
      | cons sample samples =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          change labelActions label sample ++
              scheduleActions (labels.zip samples) =
            labelActions label sample ++ labelledActions labels samples
          rw [inductionHypothesis samples sameLength]

@[simp] theorem challengeLabels_length : challengeLabels.length = 26 := by
  unfold challengeLabels FiatShamir.alphaLabels
  rw [List.length_append, List.length_map, canonicalFinIndices_length]
  rfl

private theorem labelledActions_expectedSamples
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (samples : List KExpr) (sameLength : samples.length = labels.length) :
    Formal.expectedSamples (labelledActions labels samples) = samples := by
  induction labels generalizing samples with
  | nil =>
      have : samples = [] := List.eq_nil_of_length_eq_zero sameLength
      subst samples
      rfl
  | cons label labels inductionHypothesis =>
      cases samples with
      | nil => simp at sameLength
      | cons sample samples =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          simp [labelledActions, labelActions, Formal.expectedSamples,
            inductionHypothesis samples sameLength]

private theorem labelledActions_shape_eq
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (left right : List KExpr)
    (leftLength : left.length = labels.length)
    (rightLength : right.length = labels.length) :
    (labelledActions labels left).map Formal.Action.shape =
      (labelledActions labels right).map Formal.Action.shape := by
  induction labels generalizing left right with
  | nil =>
      have leftNil : left = [] := List.eq_nil_of_length_eq_zero leftLength
      have rightNil : right = [] := List.eq_nil_of_length_eq_zero rightLength
      subst left
      subst right
      rfl
  | cons label labels inductionHypothesis =>
      cases left with
      | nil => simp at leftLength
      | cons leftSample left =>
          cases right with
          | nil => simp at rightLength
          | cons rightSample right =>
              simp only [List.length_cons, Nat.succ.injEq] at leftLength
              simp only [List.length_cons, Nat.succ.injEq] at rightLength
              simp [labelledActions, labelActions, Formal.Action.shape,
                inductionHypothesis left right leftLength rightLength]

private theorem labelledActions_squeezeCount
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (samples : List KExpr) (sameLength : samples.length = labels.length) :
    ((labelledActions labels samples).filterMap fun action => match action with
      | .absorb _ => none
      | .squeezeK _ => some ()).length = labels.length := by
  induction labels generalizing samples with
  | nil =>
      have : samples = [] := List.eq_nil_of_length_eq_zero sameLength
      subst samples
      rfl
  | cons label labels inductionHypothesis =>
      cases samples with
      | nil => simp at sameLength
      | cons sample samples =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          simp [labelledActions, labelActions,
            inductionHypothesis samples sameLength]

private theorem labelledActions_length
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (samples : List KExpr) (sameLength : samples.length = labels.length) :
    (labelledActions labels samples).length = labels.length * 2 := by
  induction labels generalizing samples with
  | nil =>
      have : samples = [] := List.eq_nil_of_length_eq_zero sameLength
      subst samples
      rfl
  | cons label labels inductionHypothesis =>
      cases samples with
      | nil => simp at sameLength
      | cons sample samples =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          rw [labelledActions, List.length_append,
            inductionHypothesis samples sameLength]
          simp [labelActions]
          omega

private theorem labelledActions_append
    (leftLabels rightLabels :
      List (FiatShamir.ChallengeLabel productionShape))
    (leftSamples rightSamples : List KExpr)
    (leftLength : leftSamples.length = leftLabels.length) :
    labelledActions (leftLabels ++ rightLabels)
        (leftSamples ++ rightSamples) =
      labelledActions leftLabels leftSamples ++
        labelledActions rightLabels rightSamples := by
  induction leftLabels generalizing leftSamples with
  | nil =>
      have : leftSamples = [] := List.eq_nil_of_length_eq_zero leftLength
      subst leftSamples
      rfl
  | cons label labels inductionHypothesis =>
      cases leftSamples with
      | nil => simp at leftLength
      | cons sample samples =>
          simp only [List.length_cons, Nat.succ.injEq] at leftLength
          simp [labelledActions, List.append_assoc,
            inductionHypothesis samples leftLength]

@[simp] theorem layoutProgram_samples_length (interface : Interface)
    (offset : Nat) : (layoutProgram interface offset).samples.length = 26 := by
  change (Formal.compile offset (interface.initialState offset)
    layoutActions).samples.length = 26
  rw [Formal.compile_samples_length]
  calc
    ((layoutActions.filterMap fun action => match action with
      | .absorb _ => none
      | .squeezeK _ => some ()).length) = challengeLabels.length := by
        unfold layoutActions
        apply labelledActions_squeezeCount
        simp [challengeLabels_length]
    _ = 26 := challengeLabels_length

@[simp] theorem layoutWiring_samples_length (interface : Interface)
    (offset : Nat) : (layoutWiring interface offset).samples.length = 26 := by
  rw [layoutWiring_samples_eq, layoutProgram_samples_length]

/-- Derived `α` sample; no caller field exists. -/
def alpha (interface : Interface) (offset : Nat)
    (coordinate : Fin productionShape.cubeVariables) : KExpr :=
  let values := (layoutProgram interface offset).samples.take 25
  values.get ⟨coordinate.val, by
    have valuesLength : values.length = 25 := by
      simp [values]
    rw [valuesLength]
    simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
      coordinate.isLt⟩

/-- Derived `γ` sample follows the 25 `α` coordinates. -/
def gamma (interface : Interface) (offset : Nat) : KExpr :=
  (layoutProgram interface offset).samples.get ⟨25, by simp⟩

/-- Executable `α` projection. Its agreement theorem preserves `alpha` as the
semantic authority. -/
def alphaFast (interface : Interface) (offset : Nat)
    (coordinate : Fin productionShape.cubeVariables) : KExpr :=
  let values := (layoutWiring interface offset).samples.take 25
  values.getD coordinate.val KExpr.zero

/-- Executable `γ` projection from the same recipe-free wiring trace. -/
def gammaFast (interface : Interface) (offset : Nat) : KExpr :=
  (layoutWiring interface offset).samples.getD 25 KExpr.zero

theorem alpha_eq_alphaFast_pointwise (interface : Interface) (offset : Nat)
    (coordinate : Fin productionShape.cubeVariables) :
    alpha interface offset coordinate =
      alphaFast interface offset coordinate := by
  let values := (layoutProgram interface offset).samples.take 25
  have coordinateBound : coordinate.val < values.length := by
    simp [values]
    simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
      coordinate.isLt
  calc
    alpha interface offset coordinate = values.get ⟨coordinate.val,
        coordinateBound⟩ := by
      rfl
    _ = values.getD coordinate.val KExpr.zero :=
      (List.getD_eq_get values KExpr.zero
        ⟨coordinate.val, coordinateBound⟩).symm
    _ = ((layoutWiring interface offset).samples.take 25).getD
          coordinate.val KExpr.zero := by
      rw [layoutWiring_samples_eq]
    _ = alphaFast interface offset coordinate := by
      rfl

theorem gamma_eq_gammaFast_pointwise (interface : Interface) (offset : Nat) :
    gamma interface offset = gammaFast interface offset := by
  have gammaBound : 25 < (layoutProgram interface offset).samples.length := by
    simp
  calc
    gamma interface offset =
        (layoutProgram interface offset).samples.get ⟨25, gammaBound⟩ := by
      rfl
    _ = (layoutProgram interface offset).samples.getD 25 KExpr.zero :=
      (List.getD_eq_get (layoutProgram interface offset).samples KExpr.zero
        ⟨25, gammaBound⟩).symm
    _ = (layoutWiring interface offset).samples.getD 25 KExpr.zero := by
      rw [layoutWiring_samples_eq]
    _ = gammaFast interface offset := by
      rfl

@[csimp] theorem alpha_eq_alphaFast : @alpha = @alphaFast := by
  funext interface offset coordinate
  exact alpha_eq_alphaFast_pointwise interface offset coordinate

@[csimp] theorem gamma_eq_gammaFast : @gamma = @gammaFast := by
  funext interface offset
  exact gamma_eq_gammaFast_pointwise interface offset

/-- Canonical labelled `α` schedule, with one constrained output per label. -/
def alphaSchedule (interface : Interface) (offset : Nat) :
    List (FiatShamir.ChallengeLabel productionShape × KExpr) :=
  (FiatShamir.alphaLabels productionShape).zip
    ((layoutProgram interface offset).samples.take 25)

/-- Exact action order: every `α` coordinate, then `γ`. -/
def actions (interface : Interface) (offset : Nat) : List Formal.Action :=
  scheduleActions (alphaSchedule interface offset) ++
    labelActions .gamma (gamma interface offset)

/-- The one owned production program. -/
def program (interface : Interface) (offset : Nat) : Formal.Program :=
  Formal.compile offset (interface.initialState offset)
    (actions interface offset)

/-- Post-challenge state computed by the same owned program. -/
def finalState (interface : Interface) (offset : Nat) : EState :=
  (program interface offset).output

theorem layoutSamples_take_gamma (interface : Interface) (offset : Nat) :
    (layoutProgram interface offset).samples.take 25 ++
        [gamma interface offset] =
      (layoutProgram interface offset).samples := by
  let samples := (layoutProgram interface offset).samples
  have indexBelow : 25 < samples.length := by
    simp [samples]
  have gammaEq : gamma interface offset = samples[25] := by
    rfl
  have tailEq : samples.drop 25 = [gamma interface offset] := by
    rw [List.drop_eq_getElem_cons indexBelow]
    have restNil : samples.drop 26 = [] := by
      exact List.drop_eq_nil_of_le (by simp [samples])
    rw [show 25 + 1 = 26 by omega, restNil]
    simp [gammaEq]
  change samples.take 25 ++ [gamma interface offset] = samples
  calc
    samples.take 25 ++ [samples.getD 25 KExpr.zero] =
        samples.take 25 ++ samples.drop 25 :=
      congrArg (fun tail => samples.take 25 ++ tail) tailEq.symm
    _ = samples := List.take_append_drop 25 samples

theorem actions_eq_labelled (interface : Interface) (offset : Nat) :
    actions interface offset =
      labelledActions challengeLabels (layoutProgram interface offset).samples := by
  have alphaLength :
      ((layoutProgram interface offset).samples.take 25).length =
        (FiatShamir.alphaLabels productionShape).length := by
    rw [List.length_take, layoutProgram_samples_length]
    norm_num [FiatShamir.alphaLabels, canonicalFinIndices_length,
      productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
  unfold actions alphaSchedule
  rw [scheduleActions_zip_eq_labelled _ _ alphaLength]
  change labelledActions (FiatShamir.alphaLabels productionShape)
        ((layoutProgram interface offset).samples.take 25) ++
      labelledActions [.gamma] [gamma interface offset] = _
  rw [← labelledActions_append]
  · rw [layoutSamples_take_gamma]
    rfl
  · exact alphaLength

theorem actions_shape_eq_layout (interface : Interface) (offset : Nat) :
    (actions interface offset).map Formal.Action.shape =
      layoutActions.map Formal.Action.shape := by
  rw [actions_eq_labelled]
  unfold layoutActions
  apply labelledActions_shape_eq
  · exact layoutProgram_samples_length interface offset
  · simp [challengeLabels_length]

theorem program_shape_eq_layout (interface : Interface) (offset : Nat) :
    (program interface offset).recipes =
        (layoutProgram interface offset).recipes ∧
      (program interface offset).samples =
        (layoutProgram interface offset).samples ∧
      (program interface offset).output =
        (layoutProgram interface offset).output := by
  exact Formal.compile_shape_eq offset (interface.initialState offset)
    (actions interface offset) layoutActions
      (actions_shape_eq_layout interface offset)

/-- Executable final-state projection from the recipe-free challenge wiring. -/
def finalStateFast (interface : Interface) (offset : Nat) : EState :=
  (layoutWiring interface offset).output

theorem finalState_eq_finalStateFast_pointwise (interface : Interface)
    (offset : Nat) :
    finalState interface offset = finalStateFast interface offset := by
  calc
    finalState interface offset = (program interface offset).output := by
      rfl
    _ = (layoutProgram interface offset).output :=
      (program_shape_eq_layout interface offset).2.2
    _ = (Formal.compileWiring offset (interface.initialState offset)
          layoutActions).output :=
      (Formal.compileWiring_matches offset (interface.initialState offset)
        layoutActions).2.symm
    _ = (layoutWiring interface offset).output :=
      congrArg Formal.Wiring.output
        (layoutWiring_eq_compileWiring interface offset).symm
    _ = finalStateFast interface offset := by
      rfl

@[csimp] theorem finalState_eq_finalStateFast : @finalState = @finalStateFast := by
  funext interface offset
  exact finalState_eq_finalStateFast_pointwise interface offset

theorem expectedSamples_eq_samples (interface : Interface) (offset : Nat) :
    Formal.expectedSamples (actions interface offset) =
      (program interface offset).samples := by
  have samplesEq := (program_shape_eq_layout interface offset).2.1
  rw [samplesEq]
  rw [actions_eq_labelled]
  exact labelledActions_expectedSamples challengeLabels
    (layoutProgram interface offset).samples
      (layoutProgram_samples_length interface offset |>.trans
        challengeLabels_length.symm)

private theorem labelActions_zero_below
    (label : FiatShamir.ChallengeLabel productionShape) (offset : Nat) :
    Formal.ActionsBelow offset (labelActions label KExpr.zero) := by
  intro action member
  simp only [labelActions, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · intro expression expressionMember
    simp [constantWords] at expressionMember
    rcases expressionMember with ⟨_, _, rfl⟩
    trivial
  · exact ⟨trivial, trivial⟩

private theorem labelledActions_zero_below
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (offset : Nat) :
    Formal.ActionsBelow offset
      (labelledActions labels (List.replicate labels.length KExpr.zero)) := by
  induction labels with
  | nil => simp [labelledActions, Formal.ActionsBelow]
  | cons label labels inductionHypothesis =>
      intro action member
      rw [List.length_cons, List.replicate_succ, labelledActions,
        List.mem_append] at member
      rcases member with head | tail
      · exact labelActions_zero_below label offset action head
      · exact inductionHypothesis action tail

theorem layoutActions_below (offset : Nat) :
    Formal.ActionsBelow offset layoutActions := by
  unfold layoutActions
  simpa [challengeLabels_length] using
    labelledActions_zero_below challengeLabels offset

/-- External inputs precede this leaf. Samples and final state are owned by
the local recipe interval. -/
def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  ∀ lane, (interface.initialState offset lane).VarsBelow offset

theorem program_causal (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    RecipesCausal offset (program interface offset).recipes := by
  rw [(program_shape_eq_layout interface offset).1]
  exact Formal.compile_causal offset (interface.initialState offset)
    layoutActions assumptions (layoutActions_below offset)

def duplexInterface (interface : Interface) : Formal.Interface where
  initial := interface.initialState
  actions := actions interface
  final := finalState interface

def evalState (env : Env) (state : EState) : State :=
  List.ofFn (Layer.evalState env state)

def evalAlpha (interface : Interface) (offset : Nat) (env : Env) :
    CubePoint K productionShape.cubeVariables where
  coordinates := (alphaSchedule interface offset).map fun item =>
    item.2.eval env
  dimension := by
    unfold alphaSchedule
    rw [List.length_map, List.length_zip]
    simp [FiatShamir.alphaLabels, canonicalFinIndices_length,
      layoutProgram_samples_length, productionShape,
      Phi81MatrixSource.phi81Shape, cubeVariables]

def evalGamma (interface : Interface) (offset : Nat) (env : Env) : K :=
  (gamma interface offset).eval env

/-- Named circuit predicate: the symbolic outputs equal the exact production
Fiat–Shamir pre-SumCheck replay. -/
def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  NightstreamFPrime.Spec.Folding.PiCCS.Transcript.PreSumcheckHolds
    oracle
    (evalState env (interface.initialState offset))
    (evalAlpha interface offset env)
    (evalGamma interface offset env)
    (evalState env (finalState interface offset))

private theorem eval_constantWords (env : Env) (words : List F) :
    Hash.evalList env (constantWords words) = words := by
  simp [Hash.evalList, constantWords, Function.comp_def]

private theorem labelActions_trace_iff
    (env : Env) (state final : State)
    (label : FiatShamir.ChallengeLabel productionShape)
    (expected : KExpr) (tail : List Formal.Action) :
    Formal.TraceHolds state
        ((labelActions label expected ++ tail).map (Formal.Action.eval env))
        final ↔
      expected.eval env = (oracle.squeeze state label).1 ∧
        Formal.TraceHolds (oracle.squeeze state label).2
          (tail.map (Formal.Action.eval env)) final := by
  simp [labelActions, Formal.Action.eval, Formal.TraceHolds,
    eval_constantWords, oracle,
    NightstreamFPrime.Lifecycle.Transcript.piCcsOracle,
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Absorb.reference,
    NightstreamFPrime.Lifecycle.Transcript.absorb,
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Squeeze.referenceSample,
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Squeeze.referenceState,
    NightstreamFPrime.Lifecycle.Transcript.squeezeK,
    NightstreamFPrime.Lifecycle.Transcript.squeezeF, Hash.inputChunks]

private theorem labelActions_trace_terminal_iff
    (env : Env) (state final : State)
    (label : FiatShamir.ChallengeLabel productionShape)
    (expected : KExpr) :
    Formal.TraceHolds state
        ((labelActions label expected).map (Formal.Action.eval env)) final ↔
      expected.eval env = (oracle.squeeze state label).1 ∧
        (oracle.squeeze state label).2 = final := by
  simpa [Formal.TraceHolds] using
    (labelActions_trace_iff env state final label expected [])

private theorem schedule_trace_iff
    (env : Env) (state final : State)
    (schedule : List
      (FiatShamir.ChallengeLabel productionShape × KExpr))
    (tail : List Formal.Action) :
    Formal.TraceHolds state
        ((scheduleActions schedule ++ tail).map (Formal.Action.eval env)) final ↔
      let result := FiatShamir.squeezeMany oracle state (schedule.map Prod.fst)
      schedule.map (fun item => item.2.eval env) = result.1 ∧
        Formal.TraceHolds result.2
          (tail.map (Formal.Action.eval env)) final := by
  induction schedule generalizing state with
  | nil => simp [scheduleActions, FiatShamir.squeezeMany]
  | cons item schedule inductionHypothesis =>
      simp only [scheduleActions] at inductionHypothesis
      rw [scheduleActions, List.flatMap_cons, List.append_assoc,
        labelActions_trace_iff, inductionHypothesis]
      simp only [List.map_cons, FiatShamir.squeezeMany, List.cons.injEq]
      aesop

private theorem alphaSchedule_labels (interface : Interface) (offset : Nat) :
    (alphaSchedule interface offset).map Prod.fst =
      FiatShamir.alphaLabels productionShape := by
  unfold alphaSchedule
  apply List.map_fst_zip
  rw [List.length_take, layoutProgram_samples_length]
  norm_num [FiatShamir.alphaLabels, canonicalFinIndices_length,
    productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

theorem alphaSchedule_values (interface : Interface) (offset : Nat) :
    (alphaSchedule interface offset).map Prod.snd =
      List.ofFn (alpha interface offset) := by
  unfold alphaSchedule
  rw [List.map_snd_zip]
  · let values := (layoutProgram interface offset).samples.take 25
    change values = List.ofFn (fun coordinate =>
      values.get ⟨coordinate.val, by
        have valuesLength : values.length = 25 := by simp [values]
        rw [valuesLength]
        simpa [productionShape, Phi81MatrixSource.phi81Shape,
          cubeVariables] using coordinate.isLt⟩)
    simpa [values, alpha, productionShape, Phi81MatrixSource.phi81Shape,
      cubeVariables] using (List.ofFn_get values).symm
  · rw [List.length_take, layoutProgram_samples_length]
    norm_num [FiatShamir.alphaLabels, canonicalFinIndices_length,
      productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

theorem evalAlpha_coordinates (interface : Interface) (offset : Nat)
    (env : Env) :
    (evalAlpha interface offset env).coordinates =
      List.ofFn fun coordinate => (alpha interface offset coordinate).eval env := by
  have mapped := congrArg (List.map (KExpr.eval env))
    (alphaSchedule_values interface offset)
  simpa [evalAlpha, List.map_map, Function.comp_def] using mapped

@[simp] theorem evalGamma_eq (interface : Interface) (offset : Nat)
    (env : Env) :
    evalGamma interface offset env = (gamma interface offset).eval env := by
  rfl

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- The owned action trace is exactly the named pre-SumCheck predicate. -/
private theorem trace_iff_specHolds
    (interface : Interface) (offset : Nat) (env : Env) :
    Formal.TraceHolds
        (evalState env (interface.initialState offset))
        ((actions interface offset).map (Formal.Action.eval env))
        (evalState env (finalState interface offset)) ↔
      SpecHolds interface offset env := by
  let initial := evalState env (interface.initialState offset)
  let final := evalState env (finalState interface offset)
  let alphaResult := FiatShamir.squeezeMany oracle initial
    (FiatShamir.alphaLabels productionShape)
  let gammaResult := oracle.squeeze alphaResult.2 .gamma
  unfold actions
  rw [schedule_trace_iff]
  rw [alphaSchedule_labels]
  dsimp only
  rw [labelActions_trace_terminal_iff]
  constructor
  · rintro ⟨alphaEq, gammaEq, finalEq⟩
    refine ⟨?_, gammaEq, finalEq.symm⟩
    exact cubePoint_eq_of_coordinates _ _ alphaEq
  · rintro ⟨alphaEq, gammaEq, finalEq⟩
    exact ⟨congrArg CubePoint.coordinates alphaEq, gammaEq, finalEq.symm⟩

theorem trace_implies_specHolds
    (interface : Interface) (offset : Nat) (env : Env)
    (trace : Formal.TraceHolds
      (evalState env (interface.initialState offset))
      ((actions interface offset).map (Formal.Action.eval env))
      (evalState env (finalState interface offset))) :
    SpecHolds interface offset env :=
  (trace_iff_specHolds interface offset env).mp trace

/-- The child Duplex trace is exactly the named pre-SumCheck predicate. -/
theorem duplexSpec_iff_specHolds
    (interface : Interface) (offset : Nat) (env : Env) :
    Formal.SpecHolds (duplexInterface interface) offset env ↔
      SpecHolds interface offset env := by
  simpa only [Formal.SpecHolds, duplexInterface] using
    trace_iff_specHolds interface offset env

/-- The owned leaf emits only the causal recipe batch. Sample and final-state
copy assertions are theorems, not rows. -/
def opsAt (interface : Interface) (offset : Nat) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset
    (program interface offset).recipes)]

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + (program interface offset).recipes.length,
    opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

@[simp] theorem opsAt_localLength (interface : Interface) (offset : Nat) :
    localLength (opsAt interface offset) =
      (program interface offset).recipes.length := by
  simp [opsAt, localLength, Op.localLength]

@[simp] theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (program interface offset).recipes := by
  simp [opsAt, flatConstraints, Op.flatConstraints]

/-- Execute the verifier-owned challenge program for any valid external
initial state. No caller-supplied challenge or output state is required. -/
theorem build (interface : Interface) (env : Env) (offset : Nat)
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
  refine ⟨completed, ?_, ?_⟩
  · exact executeRecipes_agreesOutside env offset compiled.recipes
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact recipeRows

/-- The sole logical circuit for this leaf. -/
def circuit (interface : Interface) : FormalCircuit where
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

@[simp] theorem circuit_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (circuit interface).main offset = opsAt interface offset := by
  rfl

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  (circuit interface).soundness env offset assumptions rows

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  (circuit interface).completeness env offset assumptions specification

private theorem layoutSample_varsBelow (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env)
    (sample : KExpr)
    (member : sample ∈ (layoutProgram interface offset).samples) :
    sample.VarsBelow
      (offset + (program interface offset).recipes.length) := by
  have layoutScope := Formal.compile_samples_scope offset
    (interface.initialState offset) layoutActions assumptions
      (layoutActions_below offset)
  have below := layoutScope sample member
  have recipesEq := (program_shape_eq_layout interface offset).1
  rw [recipesEq]
  exact below

theorem alpha_varsBelow (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env)
    (coordinate : Fin productionShape.cubeVariables) :
    (alpha interface offset coordinate).VarsBelow
      (offset + (program interface offset).recipes.length) := by
  apply layoutSample_varsBelow interface offset assumptions
  exact List.mem_of_mem_take (List.get_mem _ _)

theorem gamma_varsBelow (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    (gamma interface offset).VarsBelow
      (offset + (program interface offset).recipes.length) := by
  apply layoutSample_varsBelow interface offset assumptions
  exact List.get_mem _ _

theorem finalState_varsBelow (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    ∀ lane, (finalState interface offset lane).VarsBelow
      (offset + (program interface offset).recipes.length) := by
  have layoutScope := (Formal.compile_scope offset
    (interface.initialState offset) layoutActions assumptions
      (layoutActions_below offset)).1
  intro lane
  have recipesEq := (program_shape_eq_layout interface offset).1
  have outputEq := (program_shape_eq_layout interface offset).2.2
  change ((program interface offset).output lane).VarsBelow
    (offset + (program interface offset).recipes.length)
  rw [outputEq, recipesEq]
  exact layoutScope lane

theorem specHolds_of_agree_below (interface : Interface) (offset : Nat)
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
      after before (assumptions lane) (fun index below => agrees index (by omega))
  have alphaEq : evalAlpha interface offset after =
      evalAlpha interface offset before := by
    apply cubePoint_eq_of_coordinates
    change
      ((canonicalFinIndices productionShape.cubeVariables).map fun coordinate =>
          (alpha interface offset coordinate).eval after) =
        ((canonicalFinIndices productionShape.cubeVariables).map fun coordinate =>
          (alpha interface offset coordinate).eval before)
    apply List.map_congr_left
    intro coordinate member
    exact (alpha interface offset coordinate).eval_eq_of_agree_below
      (offset + (program interface offset).recipes.length) after before
      (alpha_varsBelow interface offset assumptions coordinate) agrees
  have gammaEq : evalGamma interface offset after =
      evalGamma interface offset before := by
    exact (gamma interface offset).eval_eq_of_agree_below
      (offset + (program interface offset).recipes.length) after before
      (gamma_varsBelow interface offset assumptions) agrees
  have finalEq : evalState after (finalState interface offset) =
      evalState before (finalState interface offset) := by
    apply congrArg List.ofFn
    funext lane
    exact (finalState interface offset lane).eval_eq_of_agree_below
      (offset + (program interface offset).recipes.length) after before
      (finalState_varsBelow interface offset
        assumptions lane) agrees
  unfold SpecHolds at specification ⊢
  rw [initialEq, alphaEq, gammaEq, finalEq]
  exact specification

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  have causal := program_causal interface offset assumptions
  have scope := recipeConstraints_varsBelow_of_causal offset
    (program interface offset).recipes causal
  rw [circuit_ops, flatConstraints_opsAt, opsAt_localLength]
  exact scope

theorem alphaSchedule_length (interface : Interface) (offset : Nat) :
    (alphaSchedule interface offset).length = 25 := by
  rw [alphaSchedule, List.length_zip, List.length_take,
    layoutProgram_samples_length]
  norm_num [FiatShamir.alphaLabels, canonicalFinIndices_length,
    productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

/-- The leaf has 25 label/squeeze pairs for `α` and one for `γ`. -/
theorem actions_length (interface : Interface) (offset : Nat) :
    (actions interface offset).length = 52 := by
  rw [actions_eq_labelled]
  rw [labelledActions_length]
  · norm_num [challengeLabels_length]
  · exact layoutProgram_samples_length interface offset |>.trans
      challengeLabels_length.symm

private theorem labelActions_recipeCount
    (label : FiatShamir.ChallengeLabel productionShape)
    (expected : KExpr) :
    Formal.recipeCount (labelActions label expected) = 1776 := by
  cases label <;>
    norm_num [labelActions, constantWords, Formal.recipeCount,
      Formal.Action.recipeCount, Hash.inputChunks,
      NightstreamFPrime.Lifecycle.Transcript.labelWord, Poseidon2.rate]

private theorem labelledActions_recipeCount
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (samples : List KExpr) (sameLength : samples.length = labels.length) :
    Formal.recipeCount (labelledActions labels samples) =
      labels.length * 1776 := by
  induction labels generalizing samples with
  | nil =>
      have : samples = [] := List.eq_nil_of_length_eq_zero sameLength
      subst samples
      rfl
  | cons label labels inductionHypothesis =>
      cases samples with
      | nil => simp at sameLength
      | cons sample samples =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          rw [labelledActions, Formal.recipeCount_append,
            labelActions_recipeCount,
            inductionHypothesis samples sameLength]
          simp only [List.length_cons, Nat.succ_mul]
          omega

def recipeCount (interface : Interface) (offset : Nat) : Nat :=
  Formal.recipeCount (actions interface offset)

/-- Exact private symbolic footprint: 26 labelled squeezes and their label
absorptions compile to 44,400 recipe variables. -/
theorem recipeCount_eq (interface : Interface) (offset : Nat) :
    recipeCount interface offset = 46176 := by
  unfold recipeCount
  rw [actions_eq_labelled]
  rw [labelledActions_recipeCount]
  · norm_num [challengeLabels_length]
  · exact layoutProgram_samples_length interface offset |>.trans
      challengeLabels_length.symm

@[simp] theorem program_recipes_length (interface : Interface) (offset : Nat) :
    (program interface offset).recipes.length = 46176 := by
  change (Formal.compile offset (interface.initialState offset)
    (actions interface offset)).recipes.length = 46176
  rw [Formal.compile_recipes_length]
  exact recipeCount_eq interface offset

/-- Layout may allocate exactly this private interval and no boundary copy. -/
theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 46176 := by
  rw [circuit_ops, opsAt_localLength, program_recipes_length]

/-- One owned witness operation and no sample or final-state copy operation. -/
theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 := by
  rw [circuit_ops]
  rfl

/-- One row per causal recipe and no boundary-copy row. -/
theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      46176 := by
  rw [circuit_ops, flatConstraints_opsAt, recipeConstraints_length,
    program_recipes_length]

/-- Parent-wiring theorem: once the incoming state is the canonical statement
state, every leaf output is the key-owned Fiat–Shamir pre-SumCheck result. -/
theorem spec_implies_derivePreSumcheck
    (interface : Interface) (offset : Nat) (env : Env)
    (context : Context)
    (initial_eq : evalState env (interface.initialState offset) =
      oracle.initialState context)
    (specification : SpecHolds interface offset env) :
    evalAlpha interface offset env =
        (FiatShamir.derivePreSumcheck oracle context).alpha ∧
      evalGamma interface offset env =
        (FiatShamir.derivePreSumcheck oracle context).gamma ∧
      evalState env (finalState interface offset) =
        (FiatShamir.derivePreSumcheck oracle context).state := by
  rcases specification with ⟨alphaEq, gammaEq, finalEq⟩
  rw [initial_eq,
    NightstreamFPrime.Spec.Folding.PiCCS.Transcript.deriveFromState_initialState]
    at alphaEq gammaEq finalEq
  exact ⟨alphaEq, gammaEq, finalEq⟩

/-- The exact statement context used by the one production NIFS key. -/
noncomputable def productionContext
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : Context :=
  let key := ProductionKey.key relation ajtai
  {
    priorState := key.publicInputState running fresh
    input := (key.statement running fresh).verifierInput key.lift
  }

/-- Concrete parent coverage: satisfying this leaf binds `α` and `γ` to the
coins of the production key's complete `piCcsExecution`. -/
theorem spec_implies_keyExecution_challenges
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
    (interface : Interface) (offset : Nat) (env : Env)
    (initial_eq : evalState env (interface.initialState offset) =
      oracle.initialState
        (productionContext relation ajtai running fresh))
    (specification : SpecHolds interface offset env) :
    evalAlpha interface offset env =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha ∧
      evalGamma interface offset env =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma := by
  let key := ProductionKey.key relation ajtai
  let context := productionContext relation ajtai running fresh
  have derived := spec_implies_derivePreSumcheck interface offset env context
    (by simpa [context] using initial_eq) specification
  have coinsEq := key.piCcsExecution_coins_eq_derive running fresh proof
  let certificate : FiatShamir.Certificate K productionShape :=
    { rounds := fun round => (proof.piCcsRounds round).toMessage }
  have alphaExecution :
      (key.piCcsExecution running fresh proof).coins.alpha =
        (FiatShamir.derivePreSumcheck oracle context).alpha := by
    calc
      (key.piCcsExecution running fresh proof).coins.alpha =
          (FiatShamir.derive key.oracle.transcript context certificate).alpha := by
        simpa [context, certificate, productionContext, key] using
          congrArg (fun coins => coins.alpha) coinsEq
      _ = (FiatShamir.derivePreSumcheck key.oracle.transcript context).alpha :=
        NightstreamFPrime.Spec.Folding.PiCCS.Transcript.derive_alpha_eq_preSumcheck
          key.oracle.transcript context certificate
      _ = (FiatShamir.derivePreSumcheck oracle context).alpha := by
        rfl
  have gammaExecution :
      (key.piCcsExecution running fresh proof).coins.gamma =
        (FiatShamir.derivePreSumcheck oracle context).gamma := by
    calc
      (key.piCcsExecution running fresh proof).coins.gamma =
          (FiatShamir.derive key.oracle.transcript context certificate).gamma := by
        simpa [context, certificate, productionContext, key] using
          congrArg (fun coins => coins.gamma) coinsEq
      _ = (FiatShamir.derivePreSumcheck key.oracle.transcript context).gamma :=
        NightstreamFPrime.Spec.Folding.PiCCS.Transcript.derive_gamma_eq_preSumcheck
          key.oracle.transcript context certificate
      _ = (FiatShamir.derivePreSumcheck oracle context).gamma := by
        rfl
  exact ⟨derived.1.trans alphaExecution.symm,
    derived.2.1.trans gammaExecution.symm⟩

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
