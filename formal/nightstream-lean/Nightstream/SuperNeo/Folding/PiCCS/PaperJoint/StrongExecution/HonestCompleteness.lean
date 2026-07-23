import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialHonestProver
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution

/-!
Uniform honest completeness for deterministic causal paper `Pi_CCS`.

Owns: one round selector chosen uniformly before verifier coins; the causal
honest strategy; exact fixed-to-raw SumCheck acceptance; construction of the
complete honest output; and corrected-ambient target membership derived from
source membership and an explicit verifier-bound inclusion.

Does not own: probability, rejection sampling, conditioning, root bounds,
Fiat--Shamir, commitment hardness, Rust, R1CS, artifacts, or costs.

The headline quantifier order is `exists strategy, forall coins`. Therefore
the strategy cannot close over the verifier's future round challenges.
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.HonestCompleteness

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open FullOutputCoordinates
open MatrixCoefficientSource
open UnifiedSources

universe uExtension uCommitment uPublicInput

/-- A verifier-coin-independent choice of one representing fixed polynomial
for every reachable alpha/gamma and already-revealed challenge prefix. -/
structure HonestRoundSelector
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns) where
  polynomial : forall
    (_alpha : CubePoint Extension shape.cubeVariables)
    (_gamma : Extension)
    (round : Fin shape.cubeVariables)
    (fixed : List Extension),
    fixed.length = round.val ->
      SumCheck.Finite.FixedPolynomial Extension
        (context.statement.verifierInput context.lift).sumcheckDegreeBound
  represents : forall
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (round : Fin shape.cubeVariables)
    (fixed : List Extension)
    (fixedLength : fixed.length = round.val),
    SumCheck.Finite.FixedPhase.Represents context.extensionOps.toOps
      (polynomial alpha gamma round fixed fixedLength)
      (fun point =>
        SumCheck.Finite.HypercubeTruth.sumCompletions
          context.extensionOps.toOps
          (ProtocolPolynomial.polynomial context.extensionOps
            (context.statement.sourceProtocolData context.lift witness)
            alpha gamma)
          (fixed ++ [point])
          (shape.cubeVariables - (round.val + 1)))

/-- Protocol-polynomial degree exactness supplies one uniform causal selector.
The choice ranges over alpha/gamma and prior prefixes, never future coins. -/
noncomputable def honestRoundSelector
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns) :
    HonestRoundSelector context witness := by
  classical
  let data := context.statement.sourceProtocolData context.lift witness
  let representable := fun
      (alpha : CubePoint Extension shape.cubeVariables)
      (gamma : Extension) =>
    ProtocolPolynomialDegree.sequentialRoundRepresentable
      context.extensionOps context.extensionLaws data alpha gamma
  refine {
    polynomial := fun alpha gamma round fixed fixedLength =>
      Classical.choose
        (representable alpha gamma fixed
          (shape.cubeVariables - (round.val + 1)) (by
            rw [fixedLength]
            omega))
    represents := ?_
  }
  intro alpha gamma round fixed fixedLength
  exact Classical.choose_spec
    (representable alpha gamma fixed
      (shape.cubeVariables - (round.val + 1)) (by
        rw [fixedLength]
        omega))

private def historyPoint
    {Extension : Type uExtension}
    {rounds : Nat}
    (history : History Extension rounds) : CubePoint Extension rounds where
  coordinates := history.challenges
  dimension := history.challenges_length

/-- The uniform honest strategy. A round sees only its typed prior history;
the complete output is evaluated only after the final history exists. -/
noncomputable def honestStrategy
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns) :
    Strategy Extension shape PUnit where
  roundMessage := fun round _ alpha gamma history =>
    SumCheck.Finite.FixedPolynomial.canonicalMessage
      context.extensionOps.toOps
      ((honestRoundSelector context witness).polynomial alpha gamma round
        history.challenges history.challenges_length)
  fullOutput := fun _ _ _ history =>
    FullOutput.honestAt context.baseOps context.extensionOps context.lift
      (context.statement.sourceConnectedInputs witness) (historyPoint history)

/-- Representations for an already-revealed prefix while `future` Boolean
variables remain unrevealed. Decreasing `future` when appending one challenge
keeps every earlier expected polynomial unchanged. -/
private def PrefixRepresentations
    {Extension : Type uExtension}
    {degree : Nat}
    (ops : SumCheck.Finite.Ops Extension)
    (q : List Extension -> Extension) :
    List Extension -> Nat -> List Extension ->
      List (SumCheck.Finite.FixedPolynomial Extension degree) -> Prop
  | _, _, [], [] => True
  | fixed, future, challenge :: challenges, polynomial :: polynomials =>
      SumCheck.Finite.FixedPhase.Represents ops polynomial (fun point =>
        SumCheck.Finite.HypercubeTruth.sumCompletions ops q
          (fixed ++ [point]) (challenges.length + future)) /\
      PrefixRepresentations ops q (fixed ++ [challenge]) future
        challenges polynomials
  | _, _, _, _ => False

private theorem prefixRepresentations_snoc
    {Extension : Type uExtension}
    {degree : Nat}
    (ops : SumCheck.Finite.Ops Extension)
    (q : List Extension -> Extension)
    (fixed challenges : List Extension)
    (future : Nat)
    (polynomials : List
      (SumCheck.Finite.FixedPolynomial Extension degree))
    (challenge : Extension)
    (polynomial : SumCheck.Finite.FixedPolynomial Extension degree)
    (prior : PrefixRepresentations ops q fixed (future + 1)
      challenges polynomials)
    (last : SumCheck.Finite.FixedPhase.Represents ops polynomial (fun point =>
      SumCheck.Finite.HypercubeTruth.sumCompletions ops q
        ((fixed ++ challenges) ++ [point]) future)) :
    PrefixRepresentations ops q fixed future
      (challenges ++ [challenge]) (polynomials ++ [polynomial]) := by
  induction challenges generalizing fixed polynomials with
  | nil =>
      cases polynomials with
      | nil => simpa [PrefixRepresentations] using last
      | cons _ _ => simp [PrefixRepresentations] at prior
  | cons head tail inductionHypothesis =>
      cases polynomials with
      | nil => simp [PrefixRepresentations] at prior
      | cons headPolynomial tailPolynomials =>
          rcases prior with ⟨headRepresents, tailRepresents⟩
          refine ⟨?_, ?_⟩
          · simpa [PrefixRepresentations, Nat.add_assoc, Nat.add_comm,
              Nat.add_left_comm] using headRepresents
          · have last' :
                SumCheck.Finite.FixedPhase.Represents ops polynomial
                  (fun point =>
                    SumCheck.Finite.HypercubeTruth.sumCompletions ops q
                      (((fixed ++ [head]) ++ tail) ++ [point]) future) := by
              simpa [List.append_assoc] using last
            exact inductionHypothesis (fixed := fixed ++ [head])
              (polynomials := tailPolynomials) tailRepresents last'

private theorem prefixRepresentations_zero_eq_representations
    {Extension : Type uExtension}
    {degree : Nat}
    (ops : SumCheck.Finite.Ops Extension)
    (q : List Extension -> Extension)
    (fixed challenges : List Extension)
    (polynomials : List
      (SumCheck.Finite.FixedPolynomial Extension degree))
    (represented : PrefixRepresentations ops q fixed 0 challenges
      polynomials) :
    SumCheck.Finite.FixedPhase.Representations ops polynomials
      (SumCheck.Finite.HypercubeTruth.expectedPolynomialsFrom ops q fixed
        challenges) := by
  induction challenges generalizing fixed polynomials with
  | nil =>
      cases polynomials with
      | nil => trivial
      | cons _ _ => simp [PrefixRepresentations] at represented
  | cons challenge challenges inductionHypothesis =>
      cases polynomials with
      | nil => simp [PrefixRepresentations] at represented
      | cons polynomial polynomials =>
          rcases represented with ⟨head, tail⟩
          refine ⟨?_, ?_⟩
          · simpa [PrefixRepresentations] using head
          · exact inductionHypothesis (fixed := fixed ++ [challenge])
              (polynomials := polynomials) tail

private theorem sourceTableTruth
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns)
    (source : SourceHolds context.extensionOps context.lift
      context.openingMaps context.params context.statement witness) :
    (TableResidualData.toTableObligations context.extensionOps
      (SignedCoefficientObject.toTableResidualData context.extensionOps
        ((context.statement.sourceProtocolData context.lift witness).toJointData
          context.extensionOps))).AllHold := by
  let unifiedData :=
    (context.statement.sourceConnectedInputs witness).toUnifiedInputs
      context.baseOps
  have independentSemantic :
      unifiedData.toIndependentInputs.SemanticTruth context.baseOps
        context.extensionOps context.lift :=
    (unifiedData.toIndependentInputs_semanticTruth_iff context.baseOps
      context.extensionOps context.lift).2 source.2
  have independentTableTruth :=
    (ConcreteJointData.jointTableTruth_iff_semanticTruth context.baseOps
      context.baseZero context.noZeroDivisors context.extensionOps
      context.extensionLaws context.lift
      context.liftLaws.toZeroReflectingLift
      unifiedData.toIndependentInputs).2 independentSemantic
  simpa only [Statement.sourceProtocolData, unifiedData,
    ProtocolDataRefinement.toProtocolData_toJointData_eq
      context.baseOps context.extensionOps context.lift context.liftLaws
      unifiedData] using independentTableTruth

private theorem fixedInitialIsTrue
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (sourceTruth :
      (TableResidualData.toTableObligations context.extensionOps
        (SignedCoefficientObject.toTableResidualData context.extensionOps
          ((context.statement.sourceProtocolData context.lift witness).toJointData
            context.extensionOps))).AllHold) :
    (context.statement.verifierInput context.lift).initial
        context.extensionOps gamma =
      SumCheck.Finite.FixedPhase.semanticInitial context.extensionOps.toOps
        (ProtocolPolynomial.polynomial context.extensionOps
          (context.statement.sourceProtocolData context.lift witness)
          alpha gamma)
        shape.cubeVariables := by
  let data := context.statement.sourceProtocolData context.lift witness
  let q := ProtocolPolynomial.polynomial context.extensionOps data alpha gamma
  let degree := data.toVerifierInput.sumcheckDegreeBound
  have coefficientTruth :
      SignedCoefficientObject.CoefficientTruth context.extensionOps
        (data.toJointData context.extensionOps) :=
    (SignedCoefficientObject.coefficientTruth_iff_tableObligations
      context.extensionOps context.extensionZeroLaws
      (data.toJointData context.extensionOps)).2 sourceTruth
  have sampledZero :
      (SignedCoefficientPolynomial.polynomial context.extensionOps
        (data.toJointData context.extensionOps) alpha).evaluate
          context.extensionOps.toOps gamma = context.extensionOps.zero :=
    SignedCoefficientObject.evaluate_eq_zero_of_coefficientTruth
      context.extensionOps context.extensionLaws
      (data.toJointData context.extensionOps) alpha gamma coefficientTruth
  have jointInitialTrue :
      SumCheckInitial.verifierInitial context.extensionOps
          (data.toJointData context.extensionOps) gamma =
        SumCheckInitial.semanticInitial context.extensionOps
          (data.toJointData context.extensionOps) alpha gamma := by
    have claimTrue :=
      (SumCheckInitial.claimTrue_iff_polynomial_evaluate_eq_zero
        context.extensionOps context.extensionLaws
        (data.toJointData context.extensionOps) alpha gamma degree 0
        [] (q []) { rounds := [] }
        (ProtocolPolynomial.canonicalExpected context.extensionOps data
          alpha gamma [])).2 sampledZero
    simpa [SumCheck.Claim.True, SumCheckInitial.symbolicInstance] using
      claimTrue
  rw [← context.statement.sourceProtocolData_toVerifierInput context.lift
    witness]
  rw [ProtocolPolynomial.verifierInput_initial_eq_joint_initial]
  rw [jointInitialTrue]
  unfold SumCheck.Finite.FixedPhase.semanticInitial
  rw [ProtocolPolynomial.sumCompletions_polynomial_eq_summedQ
    context.extensionOps context.extensionLaws data alpha gamma]
  rfl

private theorem cubePoint_ext
    {Extension : Type uExtension}
    {variables : Nat}
    (left right : CubePoint Extension variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  cases coordinates
  rfl

private theorem honestStrategy_accepted
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns)
    (source : SourceHolds context.extensionOps context.lift
      context.openingMaps context.params context.statement witness)
    (coins : PublicCoins Extension shape) :
    (execute (honestStrategy context witness) PUnit.unit coins).probe.Accepted
      context.extensionOps context.lift context.statement := by
  let data := context.statement.sourceProtocolData context.lift witness
  let q := ProtocolPolynomial.polynomial context.extensionOps data
    coins.alpha coins.gamma
  let degree := (context.statement.verifierInput context.lift).sumcheckDegreeBound
  let strategy := honestStrategy context witness
  let causalRun := execute strategy PUnit.unit coins
  have built := execute_history_induction strategy PUnit.unit coins
    (motive := fun rounds history =>
      exists fixedRounds : List
          (SumCheck.Finite.FixedPolynomial Extension degree),
        history.messages = fixedRounds.map
          (SumCheck.Finite.FixedPolynomial.canonicalMessage
            context.extensionOps.toOps) /\
        PrefixRepresentations context.extensionOps.toOps q []
          (shape.cubeVariables - rounds) history.challenges fixedRounds)
    (by exact ⟨[], rfl, trivial⟩)
    (by
      intro rounds within prior priorBuilt
      rcases priorBuilt with ⟨fixedRounds, messagesEqual, represented⟩
      let round : Fin shape.cubeVariables :=
        ⟨rounds, Nat.lt_of_succ_le within⟩
      let polynomial :=
        (honestRoundSelector context witness).polynomial
          coins.alpha coins.gamma round prior.challenges
            prior.challenges_length
      refine ⟨fixedRounds ++ [polynomial], ?_, ?_⟩
      · change
          prior.messages ++
              [SumCheck.Finite.FixedPolynomial.canonicalMessage
                context.extensionOps.toOps polynomial] =
            (fixedRounds ++ [polynomial]).map
              (SumCheck.Finite.FixedPolynomial.canonicalMessage
                context.extensionOps.toOps)
        rw [messagesEqual, List.map_append]
        rfl
      · have futureIdentity :
            shape.cubeVariables - rounds =
              (shape.cubeVariables - (rounds + 1)) + 1 := by
          omega
        rw [futureIdentity] at represented
        apply prefixRepresentations_snoc context.extensionOps.toOps q []
          prior.challenges (shape.cubeVariables - (rounds + 1)) fixedRounds
          _ polynomial represented
        simpa [polynomial, round, List.nil_append] using
          (honestRoundSelector context witness).represents
            coins.alpha coins.gamma round prior.challenges
              prior.challenges_length)
  rcases built with ⟨fixedRounds, messagesEqual, represented⟩
  have challengeCoordinates : causalRun.history.challenges =
      coins.roundPoint.coordinates := by
    exact execute_history_challenges_eq_roundPoint strategy PUnit.unit coins
  have fixedHonest :
      SumCheck.Finite.FixedPhase.Honest context.extensionOps.toOps q
        coins.roundPoint.coordinates { rounds := fixedRounds } := by
    have finalRepresentations :=
      prefixRepresentations_zero_eq_representations
        context.extensionOps.toOps q [] causalRun.history.challenges
        fixedRounds (by simpa [causalRun] using represented)
    simpa [SumCheck.Finite.FixedPhase.Honest,
      SumCheck.Finite.FixedPhase.expectedRounds, challengeCoordinates] using
      finalRepresentations
  have initialIsTrue := fixedInitialIsTrue context witness coins.alpha
    coins.gamma (sourceTableTruth context witness source)
  have fixedAccepted :
      SumCheck.Finite.FixedPhase.Accepted context.extensionOps.toOps q
        ((context.statement.verifierInput context.lift).initial
          context.extensionOps coins.gamma)
        coins.roundPoint.coordinates { rounds := fixedRounds } :=
    SumCheck.Finite.FixedPhase.complete context.extensionOps.toOps q _
      coins.roundPoint.coordinates { rounds := fixedRounds }
      (by
        calc
          (context.statement.verifierInput context.lift).initial
                context.extensionOps coins.gamma =
              SumCheck.Finite.FixedPhase.semanticInitial
                context.extensionOps.toOps q shape.cubeVariables := by
            simpa [q] using initialIsTrue
          _ = SumCheck.Finite.FixedPhase.semanticInitial
                context.extensionOps.toOps q
                  coins.roundPoint.coordinates.length :=
            congrArg
              (SumCheck.Finite.FixedPhase.semanticInitial
                context.extensionOps.toOps q)
              coins.roundPoint.dimension.symm)
      fixedHonest
  have rawAccepted :
      SumCheck.Finite.Accepted context.extensionOps.toOps degree
        ((context.statement.verifierInput context.lift).initial
          context.extensionOps coins.gamma)
        coins.roundPoint.coordinates (q coins.roundPoint.coordinates)
        (SumCheck.Finite.FixedPhase.Canonical.toFinite
          context.extensionOps.toOps { rounds := fixedRounds }) :=
    SumCheck.Finite.FixedPhase.Canonical.accepted_toFinite
      context.extensionOps.toOps
      (ProtocolPolynomialDegree.Support.polynomialLaws
        context.extensionLaws)
      q _ coins.roundPoint.coordinates { rounds := fixedRounds } fixedAccepted
  have historyPointEqual : historyPoint causalRun.history =
      coins.roundPoint :=
    cubePoint_ext _ _ challengeCoordinates
  have fullOutputEqual : causalRun.probe.response.fullOutput =
      FullOutput.honestAt context.baseOps context.extensionOps context.lift
        (context.statement.sourceConnectedInputs witness)
        coins.roundPoint := by
    change
      FullOutput.honestAt context.baseOps context.extensionOps context.lift
          (context.statement.sourceConnectedInputs witness)
          (historyPoint causalRun.history) = _
    rw [historyPointEqual]
  have outputEqual :
      context.statement.projectOutput causalRun.probe.response.fullOutput =
        ProtocolPolynomial.messageAt context.extensionOps data
          coins.roundPoint := by
    calc
      context.statement.projectOutput causalRun.probe.response.fullOutput =
          causalRun.probe.response.fullOutput.toOutputMessage
            (context.statement.identityFirstMatrix witness) :=
        context.statement.projectOutput_eq_toOutputMessage witness _
      _ = (FullOutput.honestAt context.baseOps context.extensionOps
            context.lift (context.statement.sourceConnectedInputs witness)
            coins.roundPoint).toOutputMessage
              (context.statement.identityFirstMatrix witness) := by
        rw [fullOutputEqual]
      _ = ProtocolPolynomial.messageAt context.extensionOps data
          coins.roundPoint :=
        FullOutput.honestAt_toOutputMessage_eq_messageAt context.baseOps
          context.baseLaws context.extensionOps context.lift
          (context.statement.sourceConnectedInputs witness)
          context.constantLaw (context.statement.identityFirstMatrix witness)
          coins.roundPoint
  unfold Probe.Accepted
  apply (ProtocolPolynomial.check_eq_true_iff_accepted
    context.extensionOps (context.statement.verifierInput context.lift)
    coins.alpha coins.gamma coins.roundPoint
    (context.statement.projectOutput causalRun.probe.response.fullOutput)
    causalRun.probe.response.rounds).2
  rw [outputEqual]
  have terminalExact :
      ProtocolPolynomial.terminalFromMessage context.extensionOps
          (context.statement.verifierInput context.lift)
          coins.alpha coins.gamma coins.roundPoint
          (ProtocolPolynomial.messageAt context.extensionOps data
            coins.roundPoint) = q coins.roundPoint.coordinates := by
    unfold q ProtocolPolynomial.polynomial
    rw [dif_pos coins.roundPoint.dimension]
    rfl
  rw [terminalExact]
  have rawRoundsEqual : causalRun.probe.response.rounds.rounds =
      (SumCheck.Finite.FixedPhase.Canonical.toFinite
        context.extensionOps.toOps { rounds := fixedRounds }).rounds := by
    simpa [causalRun, strategy,
      SumCheck.Finite.FixedPhase.Canonical.toFinite] using messagesEqual
  unfold SumCheck.Finite.Accepted at rawAccepted ⊢
  rw [rawRoundsEqual]
  simpa [degree, q, data,
    context.statement.sourceProtocolData_toVerifierInput context.lift witness]
    using rawAccepted

private theorem honestStrategy_ambient
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (ambientAdmissible : context.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.params)
    (witness : OutputWitness shape columns)
    (source : SourceHolds context.extensionOps context.lift
      context.openingMaps context.params context.statement witness)
    (coins : PublicCoins Extension shape) :
    AmbientOutputHolds context.extensionOps context.lift context.openingMaps
      context.params context.statement
      (execute (honestStrategy context witness) PUnit.unit coins).probe
      witness := by
  let strategy := honestStrategy context witness
  let causalRun := execute strategy PUnit.unit coins
  have challengeCoordinates : causalRun.history.challenges =
      coins.roundPoint.coordinates :=
    execute_history_challenges_eq_roundPoint strategy PUnit.unit coins
  have pointEqual : historyPoint causalRun.history = coins.roundPoint :=
    cubePoint_ext _ _ challengeCoordinates
  have fullOutputEqual : causalRun.probe.response.fullOutput =
      FullOutput.honestAt context.baseOps context.extensionOps context.lift
        (context.statement.sourceConnectedInputs witness)
        coins.roundPoint := by
    change
      FullOutput.honestAt context.baseOps context.extensionOps context.lift
          (context.statement.sourceConnectedInputs witness)
          (historyPoint causalRun.history) = _
    rw [pointEqual]
  intro sourceIndex
  have sourceOpening := source.1 sourceIndex
  refine ⟨?_, trivial, ?_⟩
  · refine ⟨sourceOpening.1, sourceOpening.2.1, ?_⟩
    intro column
    exact Nat.lt_of_lt_of_le (sourceOpening.2.2 column) ambientAdmissible
  · change
      #[fun matrix coefficient =>
        (BooleanTable.tabulate fun vertex =>
          context.lift (PaperLinearAlgebra.matrixVectorAt context.baseOps
            (context.statement.matrixSource.coefficientMatrix context.baseOps
              matrix coefficient)
            (witness.assignments sourceIndex) vertex)).evaluate
              context.extensionOps coins.roundPoint] =
        #[fun matrix coefficient =>
          causalRun.probe.response.fullOutput.coordinate sourceIndex matrix
            coefficient]
    rw [fullOutputEqual]
    rfl

/-- One source-valid witness constructs one causal honest strategy before any
verifier coins are chosen. Every public-coin execution accepts and its attached
witness satisfies the corrected ambient target relation. -/
theorem exists_uniform_honestStrategy
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (ambientAdmissible : context.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.params)
    (witness : OutputWitness shape columns)
    (source : SourceHolds context.extensionOps context.lift
      context.openingMaps context.params context.statement witness) :
    exists strategy : Strategy Extension shape PUnit,
      forall coins : PublicCoins Extension shape,
        AmbientSuccess context
          (attachWitness (execute strategy PUnit.unit coins) (some witness)) := by
  refine ⟨honestStrategy context witness, ?_⟩
  intro coins
  exact ⟨honestStrategy_accepted context witness source coins,
    honestStrategy_ambient context ambientAdmissible witness source coins⟩

/-- Executable perfect-completeness corollary for the same uniform strategy. -/
theorem exists_uniform_honestStrategy_check
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (ambientAdmissible : context.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.params)
    (witness : OutputWitness shape columns)
    (source : SourceHolds context.extensionOps context.lift
      context.openingMaps context.params context.statement witness) :
    exists strategy : Strategy Extension shape PUnit,
      forall coins : PublicCoins Extension shape,
        ambientCheck context
          (attachWitness (execute strategy PUnit.unit coins) (some witness)) =
            true := by
  rcases exists_uniform_honestStrategy context ambientAdmissible witness source
      with ⟨strategy, complete⟩
  exact ⟨strategy, fun coins =>
    (ambientCheck_eq_true_iff context _).2 (complete coins)⟩

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.HonestCompleteness
