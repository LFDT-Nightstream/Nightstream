import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction

/-!
Deterministic causal execution for the paper `Pi_CCS` strong reduction.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 and Appendix D.4).
Phase: public alpha/gamma, message-before-challenge SumCheck rounds, complete
output, and post-protocol target-witness attachment.
Constraint family: paper semantics only; this file emits no rows.

Owns: a length-indexed revealed history; a strategy that cannot receive its
current or future SumCheck challenge; deterministic prefix execution; explicit
target failure via `none`; executable corrected-ambient membership; and the
fixed-witness pointwise cover consumed by Appendix D.4's fresh second run.

Does not own: probability, rejection sampling, conditioning, Schwartz--Zippel
bounds, SumCheck error bounds, Fiat--Shamir, Rust, R1CS, artifacts, or costs.

Emits constraints: no.

| Owned object | Exact equation or ordering |
|---|---|
| revealed history | `messages.length = challenges.length = rounds` |
| paper degree width | `sumcheckDegreeBound = sumcheckWidth` |
| prefix replay | `execute.history = replayPrefix verifierWord` |
| target witness | attached only after the causal prefix |

The target witness is attached only after the causal prefix is complete. This
prevents later extractor randomness from influencing any `Pi_CCS` message or
the complete prover output.
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open FullOutputCoordinates
open MatrixCoefficientSource

universe uExtension uCommitment uPublicInput uProverTape

/-- All deterministic laws and public inputs needed by the pointwise
extraction theorem. No source assignment or target-validity proposition is a
field. -/
structure Context
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (shape : Shape)
    (columns blockCount : Nat) where
  baseOps : InterpolationOps F
  baseLaws : InterpolationEvaluationLaws baseOps
  baseZero : NormResidualTable.BaseZeroAgreement baseOps
  noZeroDivisors : NormRange.BaseFieldNoZeroDivisors
  extensionOps : InterpolationOps Extension
  extensionLaws : InterpolationEvaluationLaws extensionOps
  extensionZeroLaws : InterpolationZeroLaws extensionOps
  lift : F -> Extension
  liftLaws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift
  openingMaps : OpeningMaps Commitment PublicInput columns
  params : GlobalParams
  freshBound : params.b = 2
  statement : Statement Extension Commitment PublicInput shape
    columns blockCount baseOps
  /-- Verifier-owned computation for the finite corrected-ambient relation.
  This decides membership; it supplies no evidence that membership holds. -/
  ambientDecision : forall probe witness,
    Decidable (AmbientOutputHolds extensionOps lift openingMaps params
      statement probe witness)
  constantLaw : ConstantTermLaw baseOps statement.matrixSource.kernel
  /-- Common coefficient width selected before the interaction. -/
  sumcheckWidth : Nat
  /-- The selected width covers the exact paper-polynomial degree. -/
  sumcheckDegreeBound_le :
    (statement.verifierInput lift).sumcheckDegreeBound <= sumcheckWidth
  challengeSetSize : Nat

/-- Paper-valid selection of the fixed SumCheck storage width. The width is
the verifier-computed syntax ceiling, not merely an arbitrary larger transport
allocation. High zero coefficients within that ceiling remain valid. -/
def PaperDegreeWidthExact
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount) : Prop :=
  (context.statement.verifierInput context.lift).sumcheckDegreeBound =
    context.sumcheckWidth

/-- Exact width selection fits Appendix D.4's conservative degree expression
for the context's frozen strict-`b = 2` semantics. -/
theorem paperDegreeWidthExact_implies_width_le_paperRoundDegreeCeiling
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context) :
    context.sumcheckWidth <=
      (context.statement.verifierInput context.lift).paperRoundDegreeCeiling
        context.params.b := by
  rw [← exact]
  exact
    ProtocolPolynomial.VerifierInput.sumcheckDegreeBound_le_paperRoundDegreeCeiling_of_b_eq_two
      (context.statement.verifierInput context.lift) context.freshBound

/-- Exactly the messages already sent and challenges already revealed before
round `rounds`. The equal-length indices prevent a strategy call for round
`i` from receiving challenge `i` or any later challenge. -/
structure History (Extension : Type uExtension) (rounds : Nat) where
  messages : List (SumCheck.Finite.Message Extension)
  challenges : List Extension
  messages_length : messages.length = rounds
  challenges_length : challenges.length = rounds

namespace History

def empty (Extension : Type uExtension) : History Extension 0 where
  messages := []
  challenges := []
  messages_length := rfl
  challenges_length := rfl

def snoc
    {Extension : Type uExtension}
    {rounds : Nat}
    (history : History Extension rounds)
    (message : SumCheck.Finite.Message Extension)
    (challenge : Extension) :
    History Extension (rounds + 1) where
  messages := history.messages ++ [message]
  challenges := history.challenges ++ [challenge]
  messages_length := by simp [history.messages_length]
  challenges_length := by simp [history.challenges_length]

end History

/-- A causal prover strategy. `roundMessage i` receives only the length-`i`
history. `fullOutput` is produced only after every round challenge has been
revealed. Neither method receives post-protocol target-extractor data. -/
structure Strategy
    (Extension : Type uExtension)
    (shape : Shape)
    (ProverTape : Type uProverTape) where
  roundMessage : forall round : Fin shape.cubeVariables,
    ProverTape ->
    CubePoint Extension shape.cubeVariables ->
    Extension ->
    History Extension round.val ->
    SumCheck.Finite.Message Extension
  fullOutput :
    ProverTape ->
    CubePoint Extension shape.cubeVariables ->
    Extension ->
    History Extension shape.cubeVariables ->
    FullOutput Extension shape

private def challengeAt
    {Extension : Type uExtension}
    {variables : Nat}
    (point : CubePoint Extension variables)
    (round : Fin variables) : Extension :=
  point.coordinates.get (Fin.cast point.dimension.symm round)

private def run
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (roundPoint : CubePoint Extension shape.cubeVariables) :
    (rounds : Nat) -> rounds <= shape.cubeVariables -> History Extension rounds
  | 0, _ => History.empty Extension
  | rounds + 1, within =>
      let prior := run strategy proverTape alpha gamma roundPoint rounds
        (Nat.le_trans (Nat.le_succ rounds) within)
      let round : Fin shape.cubeVariables :=
        ⟨rounds, Nat.lt_of_succ_le within⟩
      prior.snoc
        (strategy.roundMessage round proverTape alpha gamma prior)
        (challengeAt roundPoint round)

/-- Public prefix-only presentation of the same causal replay.  The word has
exactly the already-revealed coordinates; no current or future challenge is
available while the preceding messages are constructed. -/
def replayPrefix
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension) :
    (rounds : Nat) ->
      rounds <= shape.cubeVariables ->
      (Fin rounds -> Extension) ->
      History Extension rounds
  | 0, _, _ => History.empty Extension
  | rounds + 1, within, word =>
      let prior := replayPrefix strategy proverTape alpha gamma rounds
        (Nat.le_trans (Nat.le_succ rounds) within)
        (fun index => word index.castSucc)
      let round : Fin shape.cubeVariables :=
        ⟨rounds, Nat.lt_of_succ_le within⟩
      prior.snoc
        (strategy.roundMessage round proverTape alpha gamma prior)
        (word (Fin.last rounds))

/-- Prefix replay records exactly the supplied challenge word. -/
theorem replayPrefix_challenges
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (rounds : Nat)
    (within : rounds <= shape.cubeVariables)
    (word : Fin rounds -> Extension) :
    (replayPrefix strategy proverTape alpha gamma rounds within word).challenges =
      List.ofFn word := by
  induction rounds with
  | zero => rfl
  | succ rounds inductionHypothesis =>
      simp only [replayPrefix, History.snoc]
      rw [inductionHypothesis]
      exact (List.ofFn_succ_last).symm

/-- Every replayed message is the strategy output computed from exactly the
strictly earlier challenge prefix. -/
theorem replayPrefix_messages
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (rounds : Nat)
    (within : rounds <= shape.cubeVariables)
    (word : Fin rounds -> Extension) :
    (replayPrefix strategy proverTape alpha gamma rounds within word).messages =
      List.ofFn fun index =>
        strategy.roundMessage (Fin.castLE within index) proverTape alpha gamma
          (replayPrefix strategy proverTape alpha gamma index.val
            (Nat.le_trans (Nat.le_of_lt index.isLt) within)
            (fun prior =>
              word (Fin.castLT prior
                (Nat.lt_trans prior.isLt index.isLt)))) := by
  induction rounds with
  | zero => rfl
  | succ rounds inductionHypothesis =>
      simp only [replayPrefix, History.snoc]
      rw [inductionHypothesis]
      rw [List.ofFn_succ_last]
      congr 1

private theorem run_eq_replayPrefix
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (roundPoint : CubePoint Extension shape.cubeVariables) :
    forall (rounds : Nat) (within : rounds <= shape.cubeVariables),
      run strategy proverTape alpha gamma roundPoint rounds within =
        replayPrefix strategy proverTape alpha gamma rounds within
          (fun index =>
            challengeAt roundPoint (Fin.castLE within index)) := by
  intro rounds
  induction rounds with
  | zero =>
      intro within
      rfl
  | succ rounds inductionHypothesis =>
      intro within
      simp only [run, replayPrefix]
      rw [inductionHypothesis]
      have priorWordEqual :
          (fun index : Fin rounds =>
            challengeAt roundPoint
              (Fin.castLE
                (Nat.le_trans (Nat.le_succ rounds) within) index)) =
          (fun index : Fin rounds =>
            challengeAt roundPoint
              (Fin.castLE within index.castSucc)) := by
        funext index
        congr 1
      have lastIndexEqual :
          (show Fin shape.cubeVariables from
            ⟨rounds, Nat.lt_of_succ_le within⟩) =
          Fin.castLE within (Fin.last rounds) := by
        apply Fin.ext
        rfl
      have lastChallengeEqual :
          challengeAt roundPoint
              (show Fin shape.cubeVariables from
                ⟨rounds, Nat.lt_of_succ_le within⟩) =
            challengeAt roundPoint
              (Fin.castLE within (Fin.last rounds)) :=
        congrArg (challengeAt roundPoint) lastIndexEqual
      rw [priorWordEqual, lastChallengeEqual]

/-- The causal `Pi_CCS` prefix. It has no target witness slot. -/
structure PrefixExecution
    (Extension : Type uExtension)
    (shape : Shape) where
  history : History Extension shape.cubeVariables
  probe : Probe Extension shape

/-- Execute every message-before-challenge round, then compute the complete
output from the fully revealed history. -/
def execute
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (coins : PublicCoins Extension shape) :
    PrefixExecution Extension shape :=
  let history := run strategy proverTape coins.alpha coins.gamma
    coins.roundPoint shape.cubeVariables (Nat.le_refl _)
  {
    history := history
    probe := {
      coins := coins
      response := {
        rounds := { rounds := history.messages }
        fullOutput := strategy.fullOutput proverTape coins.alpha coins.gamma
          history
      }
    }
  }

/-- The history consumed by `execute` is exactly the public prefix replay of
the verifier round word, in the same coordinate order. -/
theorem execute_history_eq_replayPrefix
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (coins : PublicCoins Extension shape)
    (word : Fin shape.cubeVariables -> Extension)
    (coordinates : coins.roundPoint.coordinates = List.ofFn word) :
    (execute strategy proverTape coins).history =
      replayPrefix strategy proverTape coins.alpha coins.gamma
        shape.cubeVariables (Nat.le_refl _) word := by
  have replay :=
    run_eq_replayPrefix strategy proverTape coins.alpha coins.gamma
      coins.roundPoint shape.cubeVariables (Nat.le_refl _)
  have wordEqual :
      (fun index =>
        challengeAt coins.roundPoint
          (Fin.castLE (Nat.le_refl shape.cubeVariables) index)) =
        word := by
    funext index
    unfold challengeAt
    rw [List.get_of_eq coordinates]
    simp [List.get_eq_getElem]
  change
    run strategy proverTape coins.alpha coins.gamma coins.roundPoint
        shape.cubeVariables (Nat.le_refl _) =
      replayPrefix strategy proverTape coins.alpha coins.gamma
        shape.cubeVariables (Nat.le_refl _) word
  calc
    _ =
        replayPrefix strategy proverTape coins.alpha coins.gamma
          shape.cubeVariables (Nat.le_refl _)
            (fun index =>
              challengeAt coins.roundPoint
                (Fin.castLE (Nat.le_refl shape.cubeVariables) index)) :=
      replay
    _ = _ := congrArg
      (replayPrefix strategy proverTape coins.alpha coins.gamma
        shape.cubeVariables (Nat.le_refl _)) wordEqual

private theorem run_induction
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (coins : PublicCoins Extension shape)
    (motive : forall rounds, History Extension rounds -> Prop)
    (empty : motive 0 (History.empty Extension))
    (step : forall
      (rounds : Nat)
      (within : rounds + 1 <= shape.cubeVariables)
      (prior : History Extension rounds),
      motive rounds prior ->
      motive (rounds + 1)
        (prior.snoc
          (strategy.roundMessage
            ⟨rounds, Nat.lt_of_succ_le within⟩
            proverTape coins.alpha coins.gamma prior)
          (challengeAt coins.roundPoint
            ⟨rounds, Nat.lt_of_succ_le within⟩))) :
    forall (rounds : Nat) (within : rounds <= shape.cubeVariables),
      motive rounds
        (run strategy proverTape coins.alpha coins.gamma coins.roundPoint
          rounds within) := by
  intro rounds within
  induction rounds with
  | zero => exact empty
  | succ rounds inductionHypothesis =>
      dsimp only [run]
      exact step rounds within _
        (inductionHypothesis
          (Nat.le_trans (Nat.le_succ rounds) within))

/-- Generic provenance eliminator for the exact causal history constructed by
`execute`. A consumer proves a base case and one message-before-challenge
step; no private run representation or future challenge is exposed. -/
theorem execute_history_induction
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (coins : PublicCoins Extension shape)
    (motive : forall rounds, History Extension rounds -> Prop)
    (empty : motive 0 (History.empty Extension))
    (step : forall
      (rounds : Nat)
      (within : rounds + 1 <= shape.cubeVariables)
      (prior : History Extension rounds),
      motive rounds prior ->
      motive (rounds + 1)
        (prior.snoc
          (strategy.roundMessage
            ⟨rounds, Nat.lt_of_succ_le within⟩
            proverTape coins.alpha coins.gamma prior)
          (challengeAt coins.roundPoint
            ⟨rounds, Nat.lt_of_succ_le within⟩))) :
    motive shape.cubeVariables
      (execute strategy proverTape coins).history := by
  exact run_induction strategy proverTape coins motive empty step
    shape.cubeVariables (Nat.le_refl _)

/-- The revealed challenge history is exactly the verifier-owned round point,
in order and with no omitted or future coordinate. -/
theorem execute_history_challenges_eq_roundPoint
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (coins : PublicCoins Extension shape) :
    (execute strategy proverTape coins).history.challenges =
      coins.roundPoint.coordinates := by
  have prefixIdentity := execute_history_induction strategy proverTape coins
    (motive := fun rounds history =>
      history.challenges = coins.roundPoint.coordinates.take rounds)
    (by rfl)
    (by
      intro rounds within prior priorIdentity
      simp only [History.snoc]
      rw [priorIdentity]
      rw [List.take_succ_eq_append_getElem]
      rfl)
  calc
    (execute strategy proverTape coins).history.challenges =
        coins.roundPoint.coordinates.take shape.cubeVariables :=
      prefixIdentity
    _ = coins.roundPoint.coordinates :=
      List.take_of_length_le (Nat.le_of_eq coins.roundPoint.dimension)

@[simp] theorem execute_probe_coins
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (coins : PublicCoins Extension shape) :
    (execute strategy proverTape coins).probe.coins = coins := by
  rfl

@[simp] theorem execute_rounds_eq_history
    {Extension : Type uExtension}
    {shape : Shape}
    {ProverTape : Type uProverTape}
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (coins : PublicCoins Extension shape) :
    (execute strategy proverTape coins).probe.response.rounds.rounds =
      (execute strategy proverTape coins).history.messages := by
  rfl

/-- A post-protocol target result. `none` is explicit extractor/target failure;
it is never identified with an ambient witness. -/
structure Execution
    (Extension : Type uExtension)
    (shape : Shape)
    (columns : Nat) where
  causalRun : PrefixExecution Extension shape
  target : Option (OutputWitness shape columns)

/-- Attach a post-protocol result without changing the causal prefix. -/
def attachWitness
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (causalRun : PrefixExecution Extension shape)
    (target : Option (OutputWitness shape columns)) :
    Execution Extension shape columns where
  causalRun := causalRun
  target := target

/-- Acceptance plus actual corrected-ambient target membership. `none` is
definitionally unsuccessful. -/
def AmbientSuccess
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (execution : Execution Extension shape columns) : Prop :=
  match execution.target with
  | none => False
  | some witness =>
      execution.causalRun.probe.FixedWidthAccepted context.extensionOps
          context.lift context.statement context.sumcheckWidth /\
        AmbientOutputHolds context.extensionOps context.lift
          context.openingMaps context.params context.statement
          execution.causalRun.probe witness

/-- The actual paper-polynomial verifier Boolean for one causal run. -/
def acceptedCheck
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (causalRun : PrefixExecution Extension shape) : Bool :=
  ProtocolPolynomial.FixedWidth.check context.extensionOps
    context.sumcheckWidth
    (context.statement.verifierInput context.lift)
    causalRun.probe.coins.alpha causalRun.probe.coins.gamma
    causalRun.probe.coins.roundPoint
    (context.statement.projectOutput causalRun.probe.response.fullOutput)
    causalRun.probe.response.rounds

theorem acceptedCheck_eq_true_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (causalRun : PrefixExecution Extension shape) :
    acceptedCheck context causalRun = true <->
      causalRun.probe.FixedWidthAccepted context.extensionOps context.lift
        context.statement context.sumcheckWidth := by
  rfl

/-- Executable target-membership filter for the eventual rejection sampler.
It uses the context's computational relation decision procedure on the exact
`AmbientOutputHolds` proposition; no Classical choice, validity evidence, or
caller-supplied acceptance Boolean is used. -/
def ambientCheck
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (execution : Execution Extension shape columns) : Bool :=
  match execution.target with
  | none => false
  | some witness =>
      letI := context.ambientDecision execution.causalRun.probe witness
      acceptedCheck context execution.causalRun &&
        decide (AmbientOutputHolds context.extensionOps context.lift
          context.openingMaps context.params context.statement
          execution.causalRun.probe witness)

theorem ambientCheck_eq_true_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (execution : Execution Extension shape columns) :
    ambientCheck context execution = true <->
      AmbientSuccess context execution := by
  cases targetEqual : execution.target with
  | none => simp [ambientCheck, AmbientSuccess, targetEqual]
  | some witness =>
      letI := context.ambientDecision execution.causalRun.probe witness
      simp [ambientCheck, AmbientSuccess, targetEqual,
        acceptedCheck_eq_true_iff]

/-- Source relation obtained from the attached target witness. -/
def SourceExtracted
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (execution : Execution Extension shape columns) : Prop :=
  match execution.target with
  | none => False
  | some witness =>
      SourceHolds context.extensionOps context.lift context.openingMaps
        context.params context.statement witness

/-- The exact alpha/gamma root event for one fixed witness. The witness is an
argument fixed outside the fresh second probe. -/
def MixingFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (causalRun : PrefixExecution Extension shape)
    (witness : OutputWitness shape columns) : Prop :=
  SignedCoefficientObject.MixingRoot context.extensionOps
    ((context.statement.sourceProtocolData context.lift witness).toJointData
      context.extensionOps)
    causalRun.probe.coins.alpha causalRun.probe.coins.gamma

/-- The exact finite SumCheck bad-challenge event for one fixed witness and
the actual fresh second certificate. -/
def SumCheckFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (causalRun : PrefixExecution Extension shape)
    (witness : OutputWitness shape columns) : Prop :=
  FixedWidthSumCheckFailure context.extensionOps context.lift
    context.statement context.sumcheckWidth context.challengeSetSize
    causalRun.probe witness

/-- The deterministic fresh-second-run theorem. Its witness is fixed by the
first accepted ambient execution before this prefix's coins are sampled. -/
theorem acceptedPrefix_extracts_fixedWitness_or_badEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (causalRun : PrefixExecution Extension shape)
    (witness : OutputWitness shape columns)
    (ambient : AmbientOutputHolds context.extensionOps context.lift
      context.openingMaps context.params context.statement
      causalRun.probe witness)
    (accepted : causalRun.probe.FixedWidthAccepted context.extensionOps
      context.lift context.statement context.sumcheckWidth) :
    SourceHolds context.extensionOps context.lift context.openingMaps
        context.params context.statement witness \/
      MixingFailure context causalRun witness \/
      SumCheckFailure context causalRun witness := by
  exact fixedWidthAcceptedProbe_extracts_source_or_badEvent context.baseLaws
    context.baseZero context.noZeroDivisors context.extensionOps
    context.extensionLaws context.extensionZeroLaws context.lift
    context.liftLaws context.openingMaps context.params context.freshBound
    context.statement context.constantLaw context.sumcheckWidth
    context.sumcheckDegreeBound_le context.challengeSetSize
    causalRun.probe witness ambient accepted

/-- One actual ambient-success execution is pointwise covered by source
extraction or the two exact named failures. No adaptive root probability or
implementation-refinement event appears. -/
theorem ambientSuccess_implies_source_or_badEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (execution : Execution Extension shape columns)
    (success : AmbientSuccess context execution) :
    SourceExtracted context execution \/
      (exists witness,
        execution.target = some witness /\
        MixingFailure context execution.causalRun witness) \/
      (exists witness,
        execution.target = some witness /\
        SumCheckFailure context execution.causalRun witness) := by
  cases targetEqual : execution.target with
  | none =>
      simp [AmbientSuccess, targetEqual] at success
  | some witness =>
      have acceptedAmbient :
          execution.causalRun.probe.FixedWidthAccepted context.extensionOps
              context.lift context.statement context.sumcheckWidth /\
            AmbientOutputHolds context.extensionOps context.lift
              context.openingMaps context.params context.statement
              execution.causalRun.probe witness := by
        simpa [AmbientSuccess, targetEqual] using success
      rcases acceptedPrefix_extracts_fixedWitness_or_badEvent context
          execution.causalRun witness acceptedAmbient.2 acceptedAmbient.1 with
        source | mixing | sumCheck
      · left
        simpa [SourceExtracted, targetEqual] using source
      · exact Or.inr (Or.inl ⟨witness, rfl, mixing⟩)
      · exact Or.inr (Or.inr ⟨witness, rfl, sumCheck⟩)

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
