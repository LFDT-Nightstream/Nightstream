import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts

/-!
Model-level obstruction at the exact paper-joint `Pi_CCS` SumCheck soundness
contract.

Protocol: SuperNeo Definition 6, Section 7.3, and Appendix D.4.
Phase: the actual causal fresh-run finite experiment.
Constraint family: fixed-width SumCheck bad-challenge probability only; this
file emits no rows.

Owns: a concrete `StrongExecution.Context`, a causal strategy whose one round
message is fixed before the current challenge, the actual verifier product
support with six uniformly sampled round challenges, exact decoding into the
fixed-width certificate, transport to the repository Boolean
`sumCheckBadChallengeEvent`, probability one for that event, and negation of
`SumCheckSoundnessContract` at the finite-model instantiation `5 / 6` of
Appendix D.4's degree/cardinality formula.

Does not own: a positive SumCheck theorem, an alternative failure event,
terminal-consistency failure, alpha/gamma mixing, Schwartz--Zippel, binding,
first-success conditioning, Fiat--Shamir, Rust, R1CS, generated dimensions, or
changes to the frozen `PiCcsStrong` loss expression.

Emits constraints: no.

| Property | Kernel-checked owner |
|---|---|
| syntax degree four is below selected width six | `context_syntaxDegree_lt_width` |
| Appendix D.4 round-degree expression is five | `paperRoundDegreeCeiling_eq_five` |
| causal message before challenge | `strategy_roundMessage_eq` |
| exact source datum alignment | `sourceProtocolData_eq_zero` |
| actual support and context cardinality agree | `context_challengeSetSize_eq_alphabet_cardinality` |
| finite-model paper formula is `5 / 6` | `paperSumCheckBudget_eq_five_six` |
| raw decode into exact failure | `sumCheckFailure_execute` |
| actual Boolean event has probability one | `sumCheckFailure_probability_eq_one` |
| every nontrivial budget is false | `not_sumCheckSoundnessContract_of_lt_one` |
| formula-instantiated contract at `5 / 6` is false | `not_sumCheckSoundnessContract_at_paper_budget` |

Authority boundary: Definition 6 charges the actual univariate degree, and
Appendix D.4 permits ceiling `max(u, 2b + 1, 2)`. Here `u = 0`, `b = 2`, and
one SumCheck variable give numerator five; the explicit six-element verifier
support gives the finite-model quotient `5 / 6`. The current context
permits selected width six and accepts a nonzero coefficient in position six.
The counterexample is therefore a necessity result for verifier-checked
zero-padding above the paper ceiling (or an equivalent exact-width invariant),
not a counterexample to the paper protocol with its degree premise enforced.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding

private def cubeLayout : UnifiedSources.ColumnLayout 1 2 where
  toColumn := fun vertex =>
    match vertex with
    | .cons false .nil => 0
    | .cons true .nil => 1
  toVertex := fun column =>
    if column.val = 0 then .cons false .nil else .cons true .nil
  toColumn_toVertex := by
    intro column
    by_cases zero : column.val = 0
    · apply Fin.eq_of_val_eq
      simp [zero]
    · have one : column.val = 1 := by omega
      apply Fin.eq_of_val_eq
      simp [one]
  toVertex_toColumn := by
    intro vertex
    cases vertex with
    | cons coordinate tail =>
        cases tail
        cases coordinate <;> rfl

private def ringLayout :
    MatrixCoefficientSource.RingColumnLayout 1 2 2 where
  decode := fun column => (column, 0)
  encode? := fun block _ => some block
  decode_encode := by
    intro block coefficient column equal
    cases equal
    apply Prod.ext
    · rfl
    · exact Subsingleton.elim _ _
  encode_decode := by
    intro column
    rfl

private def coefficientKernel :
    MatrixCoefficientSource.CoefficientKernel F 1 where
  constant := 0
  weight := fun _ row assignment =>
    if row = assignment then baseOps.one else baseOps.zero

private def constantLaw :
    MatrixCoefficientSource.ConstantTermLaw baseOps coefficientKernel where
  weight := by
    intro row assignment
    rfl

private def baseConstraint :
    CCSResidualTable.ConstraintPolynomial F shape.matrixCount where
  degreeBound := 0
  terms := []
  termsBelowDegree := by simp

private def matrixSource :
    MatrixCoefficientSource.MatrixSource F shape 2 2 where
  columnLayout := ringLayout
  matrices := fun _ vertex column =>
    if column = cubeLayout.toColumn vertex then baseOps.one else baseOps.zero
  constraintPolynomial := baseConstraint
  kernel := coefficientKernel

private def statement :
    StrongReduction.Statement Extension PUnit PUnit shape 2 2 baseOps where
  cubeLayout := cubeLayout
  matrixSource := matrixSource
  commitments := fun _ => PUnit.unit
  publicInputs := fun _ => PUnit.unit
  priorPoint := {
    coordinates := [extensionOps.zero]
    dimension := rfl
  }
  claimedCoefficient := fun coordinate => Fin.elim0 coordinate.running
  matrixCountPositive := by decide
  identityFirstEntry := by
    intro vertex column
    rfl

private def openingMaps : StrongReduction.OpeningMaps PUnit PUnit 2 where
  commit := fun _ => PUnit.unit
  projectPublicInput := fun _ => PUnit.unit

private def params : GlobalParams where
  q := 1
  b := 2
  k := 1
  maxFresh := 1
  expansionT := 0
  rlc_bound := by decide

/-- Exact current-interface context: the syntax-derived degree is four, the
selected fixed width is six, and the only relation between them is the
existing non-strict upper-bound field. -/
noncomputable def context
    (euclid : NormRange.GoldilocksModulusEuclid) :
    StrongExecution.Context Extension PUnit PUnit shape 2 2 where
  baseOps := baseOps
  baseLaws := baseLaws
  baseZero := baseZeroAgreement
  noZeroDivisors := NormRange.baseFieldNoZeroDivisors_of_modulusEuclid euclid
  extensionOps := extensionOps
  extensionLaws := extensionLaws
  extensionZeroLaws := extensionZeroLaws
  lift := K.embed
  liftLaws := protocolLift
  openingMaps := openingMaps
  params := params
  freshBound := rfl
  statement := statement
  ambientDecision := fun _ _ => Classical.propDecidable _
  constantLaw := constantLaw
  sumcheckWidth := 6
  sumcheckDegreeBound_le := by
    change (4 : Nat) <= 6
    decide
  challengeSetSize := 6

/-- The first witness is fixed outside the fresh verifier experiment. Its one
positive fresh-source assignment is identically zero. -/
def witness : StrongReduction.OutputWitness shape 2 where
  assignments := fun _ _ => baseOps.zero

/-- The context's verifier-owned syntax theorem evaluates to four. -/
theorem context_syntaxDegree_eq_four :
    (statement.verifierInput K.embed).sumcheckDegreeBound = 4 := by
  rfl

/-- The selected verifier width is strictly larger than the exact syntax
degree. -/
theorem context_syntaxDegree_lt_width
    (euclid : NormRange.GoldilocksModulusEuclid) :
    ((context euclid).statement.verifierInput (context euclid).lift).sumcheckDegreeBound <
      (context euclid).sumcheckWidth := by
  change (4 : Nat) < 6
  decide

/-- The countermodel is excluded by the repaired paper-family/key boundary:
its selected width is not the verifier-computed syntax ceiling. -/
theorem context_not_paperDegreeWidthExact
    (euclid : NormRange.GoldilocksModulusEuclid) :
    ¬ StrongExecution.PaperDegreeWidthExact (context euclid) := by
  intro exact
  change (4 : Nat) = 6 at exact
  omega

/-- Appendix D.4's permitted per-variable SumCheck degree for this exact
context: `max(u, 2b + 1, 2)`, computed from the same verifier-owned sparse
constraint and global-parameter records assembled into `context`. -/
def paperRoundDegreeCeiling : Nat :=
  Nat.max baseConstraint.degreeBound (Nat.max (2 * params.b + 1) 2)

/-- The Appendix D.4 degree expression evaluates to
`max(0, 2 * 2 + 1, 2) = 5`. -/
theorem paperRoundDegreeCeiling_eq_five :
    paperRoundDegreeCeiling = 5 := by
  rfl

/-- There is one SumCheck variable, so Definition 6's `ell * d` numerator is
also five. -/
theorem paperSumCheckNumerator_eq_five :
    shape.cubeVariables * paperRoundDegreeCeiling = 5 := by
  rfl

/-- A causal strategy whose message is constant in every input it is allowed
to observe. In particular, the current challenge is absent from the method's
type and is sampled only after this message is returned by `execute`. -/
def strategy : StrongExecution.Strategy Extension shape PUnit where
  roundMessage := fun _ _ _ _ _ => rootPolynomial.toMessage
  fullOutput := fun _ _ _ _ => {
    coordinate := fun _ _ _ => extensionOps.zero
  }

/-- The actual round message is fixed independently of the prior history,
alpha, gamma, and prover tape. There is no current-challenge argument. -/
theorem strategy_roundMessage_eq
    (round : Fin shape.cubeVariables)
    (tape : PUnit)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (history : StrongExecution.History Extension round.val) :
    strategy.roundMessage round tape alpha gamma history =
      rootPolynomial.toMessage := by
  rfl

private theorem replicate_snoc {Value : Type} (value : Value) :
    forall rounds,
      List.replicate rounds value ++ [value] =
        List.replicate (rounds + 1) value := by
  intro rounds
  induction rounds with
  | zero => rfl
  | succ rounds inductionHypothesis =>
      simp only [List.replicate_succ, List.cons_append]
      rw [inductionHypothesis]
      rfl

private theorem execute_messages
    (coins : StrongReduction.PublicCoins Extension shape) :
    (StrongExecution.execute strategy PUnit.unit coins).history.messages =
      [rootPolynomial.toMessage] := by
  have history := StrongExecution.execute_history_induction
    strategy PUnit.unit coins
    (motive := fun rounds history =>
      history.messages =
        List.replicate rounds rootPolynomial.toMessage)
    (by rfl)
    (by
      intro rounds within prior priorIdentity
      simp only [StrongExecution.History.snoc, strategy]
      rw [priorIdentity]
      exact replicate_snoc rootPolynomial.toMessage rounds)
  simpa [shape] using history

private theorem finiteCertificate_eq_of_rounds
    {Field : Type}
    {left right : SumCheck.Finite.Certificate Field}
    (equal : left.rounds = right.rounds) :
    left = right := by
  cases left
  cases right
  cases equal
  rfl

private theorem execute_raw_rounds
    (coins : StrongReduction.PublicCoins Extension shape) :
    (StrongExecution.execute strategy PUnit.unit coins).probe.response.rounds =
      SumCheck.Finite.FixedPhase.RawCertificate.encode certificate := by
  apply finiteCertificate_eq_of_rounds
  simpa [SumCheck.Finite.FixedPhase.RawCertificate.encode, certificate]
    using execute_messages coins

/-- The concrete positive-fresh-source statement and fixed witness induce
exactly the zero protocol datum used by the collision theorem. -/
theorem sourceProtocolData_eq_zero :
    statement.sourceProtocolData K.embed witness = zeroProtocolData := by
  rw [ProtocolPolynomial.Data.mk.injEq]
  unfold StrongReduction.Statement.sourceProtocolData
    ProtocolDataRefinement.toProtocolData
    StrongReduction.Statement.sourceConnectedInputs
    MatrixCoefficientSource.ConnectedInputs.toUnifiedInputs
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · funext source matrix
    dsimp only [ProtocolPolynomial.Data.freshMatrixImages, zeroProtocolData]
    change BooleanTable.tabulate _ = BooleanTable.tabulate _
    apply congrArg BooleanTable.tabulate
    funext vertex
    change K.embed
      (PaperLinearAlgebra.matrixVectorAt baseOps
        ((matrixSource.system).matrices matrix)
        (fun _ => baseOps.zero) vertex) = extensionOps.zero
    rw [PaperLinearAlgebra.matrixVectorAt_zero baseOps baseLaws]
    exact ProtocolDataRefinement.ProtocolLift.map_zero protocolLift
  · funext source
    dsimp only [ProtocolPolynomial.Data.sourceAssignments, zeroProtocolData]
    change BooleanTable.tabulate _ = BooleanTable.tabulate _
    apply congrArg BooleanTable.tabulate
    funext vertex
    exact ProtocolDataRefinement.ProtocolLift.map_zero protocolLift
  · rfl
  · funext coordinate
    exact Fin.elim0 coordinate.running
  · funext coordinate
    exact Fin.elim0 coordinate.running

private theorem cubePoint_eq_of_coordinates
    {variables : Nat}
    {left right : CubePoint Extension variables}
    (equal : left.coordinates = right.coordinates) :
    left = right := by
  cases left with
  | mk leftCoordinates leftDimension =>
      cases right with
      | mk rightCoordinates rightDimension =>
          cases equal
          rfl

/-- Exact bridge from one root equality through raw-certificate decoding to
the repository's fixed-width `StrongExecution.SumCheckFailure`. -/
theorem sumCheckFailure_execute
    (euclid : NormRange.GoldilocksModulusEuclid)
    (coins : StrongReduction.PublicCoins Extension shape)
    (challenge : Extension)
    (roundPoint_eq : coins.roundPoint = point challenge)
    (rootH : rootPolynomial.evaluate extensionOps.toOps challenge =
      extensionOps.zero) :
    StrongExecution.SumCheckFailure (context euclid)
      (StrongExecution.execute strategy PUnit.unit coins) witness := by
  unfold StrongExecution.SumCheckFailure StrongReduction.FixedWidthSumCheckFailure
  refine ⟨certificate, ?_, ?_⟩
  · rw [execute_raw_rounds]
    exact SumCheck.Finite.FixedPhase.RawCertificate.decode_encode
      certificate
  · change ProtocolPolynomial.FixedWidth.SumCheckCollision extensionOps
      (statement.sourceProtocolData K.embed witness)
      coins.alpha coins.gamma 6 6 coins.roundPoint certificate
    rw [roundPoint_eq]
    rw [sourceProtocolData_eq_zero]
    exact collision_at coins.alpha coins.gamma challenge rootH

/-- Six pairwise-distinct verifier challenges. `Support` makes the uniform
law and nonemptiness explicit rather than treating cardinality as a
probability distribution. -/
def alphabet : Support Extension where
  values := [K.embed 0, K.embed 1, K.embed 2, K.embed 3, K.embed 4, K.embed 5]
  nodup := by decide
  nonempty := by decide

/-- The actual finite support has cardinality six. -/
theorem alphabet_cardinality_eq_six : alphabet.cardinality = 6 := by
  rfl

/-- The context's numeric challenge-set field matches the actual finite uniform
support used by the verifier experiment; cardinality is not substituted for a
distribution. -/
theorem context_challengeSetSize_eq_alphabet_cardinality
    (euclid : NormRange.GoldilocksModulusEuclid) :
    (context euclid).challengeSetSize = alphabet.cardinality := by
  rfl

/-- Finite-model instantiation of the Appendix D.4/Definition 6 SumCheck budget:
one variable, degree ceiling five, and six uniformly sampled challenges. -/
def paperSumCheckBudget : Rat :=
  ratio (shape.cubeVariables * paperRoundDegreeCeiling)
    alphabet.cardinality

/-- The formula-derived paper budget is exactly `5 / 6`. -/
theorem paperSumCheckBudget_eq_five_six :
    paperSumCheckBudget = ratio 5 6 := by
  unfold paperSumCheckBudget
  rw [paperSumCheckNumerator_eq_five, alphabet_cardinality_eq_six]

private theorem rootPolynomial_zero_of_mem
    {challenge : Extension}
    (member : challenge ∈ alphabet.values) :
    rootPolynomial.evaluate extensionOps.toOps challenge =
      extensionOps.zero := by
  simp [alphabet] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl
  · exact rootPolynomial_zero_at_zero
  · exact rootPolynomial_zero_at_one
  · exact rootPolynomial_zero_at_two
  · exact rootPolynomial_zero_at_three
  · exact rootPolynomial_zero_at_four
  · exact rootPolynomial_zero_at_five

private def singletonSupport : Support PUnit where
  values := [PUnit.unit]
  nodup := by simp
  nonempty := by simp

/-- The operational adversary uses singleton prover/target tapes and the
causal constant-message strategy. The target returns the already-fixed
`witness`; it receives no fresh challenge from which to choose another one. -/
noncomputable def adversary
    (euclid : NormRange.GoldilocksModulusEuclid) :
    OperationalExperiment.Adversary (context euclid) PUnit PUnit PUnit where
  proverSupport := singletonSupport
  targetSupport := singletonSupport
  strategy := strategy
  proverTape := fun _ => PUnit.unit
  target := fun _ _ => some witness

private def onlyRound : Fin shape.cubeVariables :=
  ⟨0, by change 0 < 1; decide⟩

private theorem fin_shape_eq
    (left right : Fin shape.cubeVariables) : left = right := by
  apply Fin.eq_of_val_eq
  have leftLt : left.val < 1 := by
    change left.val < 1
    exact left.isLt
  have rightLt : right.val < 1 := by
    change right.val < 1
    exact right.isLt
  omega

private theorem verifierRoundPoint_eq_point
    (seed : VerifierCoins.Seed Extension shape.cubeVariables) :
    (VerifierCoins.toPublicCoins seed).roundPoint =
      point (VerifierCoins.roundWord seed onlyRound) := by
  apply cubePoint_eq_of_coordinates
  rw [VerifierCoins.toPublicCoins_round_coordinates]
  apply List.ext_get
  · simp [shape, point]
  · intro index leftLt rightLt
    have indexZero : index = 0 := by
      simp [point] at rightLt
      omega
    subst index
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    simp [point]
    exact congrArg (VerifierCoins.roundWord seed)
      (fin_shape_eq _ onlyRound)

private theorem sumCheckFailure_run_of_mem
    (euclid : NormRange.GoldilocksModulusEuclid)
    (seed : OperationalExperiment.RunSeed Extension shape PUnit PUnit)
    (member : seed ∈
      (OperationalExperiment.runSupport (context euclid) alphabet
        (adversary euclid)).values) :
    StrongExecution.SumCheckFailure (context euclid)
      (OperationalExperiment.run (context euclid) (adversary euclid) seed).causalRun
      witness := by
  have verifierMember : seed.2.2 ∈
      (VerifierCoins.support alphabet shape.cubeVariables).values :=
    ((OperationalExperiment.mem_runSupport_iff (context euclid) alphabet
      (adversary euclid) seed).mp member).2.2
  have coordinateMember :
      VerifierCoins.roundWord seed.2.2 onlyRound ∈ alphabet.values :=
    ((VerifierCoins.mem_support_iff alphabet shape.cubeVariables seed.2.2).mp
      verifierMember).2.2 onlyRound
  rw [OperationalExperiment.run_causalRun]
  exact sumCheckFailure_execute euclid
    (VerifierCoins.toPublicCoins seed.2.2)
    (VerifierCoins.roundWord seed.2.2 onlyRound)
    (verifierRoundPoint_eq_point seed.2.2)
    (rootPolynomial_zero_of_mem coordinateMember)

private theorem sumCheckEvent_true_of_mem
    (euclid : NormRange.GoldilocksModulusEuclid)
    (seed : OperationalExperiment.RunSeed Extension shape PUnit PUnit)
    (member : seed ∈
      (OperationalExperiment.experiment (context euclid) alphabet
        (adversary euclid)).support.values) :
    SecurityContracts.sumCheckBadChallengeEvent (context euclid) witness
      ((OperationalExperiment.experiment (context euclid) alphabet
        (adversary euclid)).outcome seed) = true := by
  change (@ite Bool
    (StrongExecution.SumCheckFailure (context euclid)
      (OperationalExperiment.run (context euclid) (adversary euclid) seed).causalRun
      witness)
    (Classical.propDecidable _)
    true false) = true
  rw [if_pos (sumCheckFailure_run_of_mem euclid seed member)]

/-- Exact probability theorem for the repository Boolean event under the
actual fresh product experiment. Every support seed triggers the same decoded
collision, so the event count equals the full support cardinality. -/
theorem sumCheckFailure_probability_eq_one
    (euclid : NormRange.GoldilocksModulusEuclid) :
    (OperationalExperiment.experiment (context euclid) alphabet
      (adversary euclid)).probabilityBool
        (SecurityContracts.sumCheckBadChallengeEvent (context euclid) witness) =
      1 := by
  let experiment := OperationalExperiment.experiment (context euclid) alphabet
    (adversary euclid)
  have countAll :
      experiment.countBool
          (SecurityContracts.sumCheckBadChallengeEvent (context euclid) witness) =
        experiment.support.cardinality := by
    unfold Experiment.countBool Support.cardinality
    apply List.countP_eq_length.mpr
    intro seed member
    exact sumCheckEvent_true_of_mem euclid seed member
  unfold Experiment.probabilityBool
  rw [countAll, Rat.div_def]
  exact Rat.mul_inv_cancel _
    (Rat.ne_of_gt (Rat.natCast_pos.mpr experiment.support.cardinality_pos))

private theorem paperBudget_lt_one :
    paperSumCheckBudget < (1 : Rat) := by
  rw [paperSumCheckBudget_eq_five_six]
  unfold ratio
  apply (Rat.div_lt_iff
    (a := (5 : Rat)) (b := (6 : Rat)) (c := (1 : Rat)) (by decide)).mpr
  rw [Rat.one_mul]
  decide

/-- Exact current-interface obstruction for every nontrivial loss budget. The
actual repository event has probability one, so no contract below one follows
from the existing context, causal execution, and finite fresh support. -/
theorem not_sumCheckSoundnessContract_of_lt_one
    (euclid : NormRange.GoldilocksModulusEuclid)
    {budget : Rat}
    (budget_lt_one : budget < 1) :
    ¬ SecurityContracts.SumCheckSoundnessContract
      (context euclid) alphabet (adversary euclid) budget := by
  intro contract
  have bound := contract witness
  rw [sumCheckFailure_probability_eq_one] at bound
  exact (Rat.not_le.mpr budget_lt_one) bound

/-- Exact repository-contract obstruction at the finite-model Appendix D.4
formula. Its degree expression gives ceiling five and
one round over six challenges, hence budget `5 / 6`; the accepted degree-six
message makes the exact repository bad-challenge event occur with probability
one. -/
theorem not_sumCheckSoundnessContract_at_paper_budget
    (euclid : NormRange.GoldilocksModulusEuclid) :
    ¬ SecurityContracts.SumCheckSoundnessContract
      (context euclid) alphabet (adversary euclid)
        paperSumCheckBudget := by
  exact not_sumCheckSoundnessContract_of_lt_one euclid paperBudget_lt_one

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract
