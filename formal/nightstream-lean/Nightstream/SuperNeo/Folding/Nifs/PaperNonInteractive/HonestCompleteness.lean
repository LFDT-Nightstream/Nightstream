import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Soundness

/-!
Honest construction for the transcript-bound paper SuperNeo NIFS.

Owns: conversion of independent source relation truth into a causal
Fiat--Shamir `Pi_CCS` certificate with the full honest output, followed by
the algebraic `Pi_RLC` parent and honest operational `Pi_DEC` children.

Does not own: extraction, bad-event probabilities, concrete transcript or
commitment primitives, HyperNova/F-prime, Rust, R1CS, artifacts, minimality,
or costs.

Emits constraints: no.

| Honest phase | Mathematical obligation | Lean owner |
|---|---|---|
| source | turn independent CCS/CE truth into the joint residual-table truth | `sourceTableTruth` |
| `Pi_CCS` | causally construct the exact finite certificate and coefficient-complete output | `exists_honestPiCcsCertificate` |
| `Pi_RLC` | compute the honest combined parent from extracted source assignments | `honestParentAssignment`, `honestParent_holds` |
| `Pi_DEC` | construct exact child messages and all recomposition equations | `honestProof`, `honestProof_piDecAttempt` |
| NIFS | construct both executable acceptance and the independent transition | `sourceValid_exists_verifiedTransition` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uState

/-- Source relation truth implies the unsampled joint residual-table truth
consumed by the causal honest `Pi_CCS` prover. -/
theorem sourceTableTruth
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (source : SourceValid key running fresh witness) :
    (TableResidualData.toTableObligations key.extensionOps
      (SignedCoefficientObject.toTableResidualData key.extensionOps
        (((key.statement running fresh).sourceProtocolData key.lift witness).toJointData
          key.extensionOps))).AllHold := by
  let unifiedData :=
    ((key.statement running fresh).sourceConnectedInputs witness).toUnifiedInputs
      key.baseOps
  have independentSemantic :
      unifiedData.toIndependentInputs.SemanticTruth key.baseOps
        key.extensionOps key.lift :=
    (unifiedData.toIndependentInputs_semanticTruth_iff key.baseOps
      key.extensionOps key.lift).2 source.2
  have independentTableTruth :=
    (ConcreteJointData.jointTableTruth_iff_semanticTruth key.baseOps
      key.baseZero key.noZeroDivisors key.extensionOps key.extensionLaws
      key.lift key.liftLaws.toZeroReflectingLift
      unifiedData.toIndependentInputs).2 independentSemantic
  simpa only [Statement.sourceProtocolData, unifiedData,
    ProtocolDataRefinement.toProtocolData_toJointData_eq key.baseOps
      key.extensionOps key.lift key.liftLaws unifiedData] using
        independentTableTruth

/-- A prefix carrier used only to compute the `Pi_RLC` parent before the
honest `Pi_DEC` messages have been materialized.  Its child fields are never
read by `piCcsExecution`, `piRlcChallenges`, or `parent`. -/
def prefixProof
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {degreeBound : Nat}
    (running : Running Extension Commitment PublicInput shape)
    (rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound)
    (fullOutput : FullOutput Extension shape) :
    Proof Extension Commitment shape degreeBound where
  piCcsRounds := rounds
  piCcsOutput := fullOutput
  piDecCommitments := running.commitments
  piDecEvaluations := running.evaluations

private theorem fixedRun_eq_rawRun
    {Field : Type uExtension}
    {State : Type uState}
    {degree : Nat}
    (step : State -> SumCheck.Finite.Message Field -> Field × State)
    (state : State)
    (rounds : List (SumCheck.Finite.FixedPolynomial Field degree)) :
    SumCheck.Finite.FixedPhase.Sequential.run
        (fun current polynomial => step current polynomial.toMessage)
        state rounds =
      ProtocolPolynomialHonestProver.runRaw step state
        (rounds.map SumCheck.Finite.FixedPolynomial.toMessage) := by
  induction rounds generalizing state with
  | nil => rfl
  | cons polynomial polynomials inductionHypothesis =>
      simp only [SumCheck.Finite.FixedPhase.Sequential.run,
        ProtocolPolynomialHonestProver.runRaw, List.map_cons]
      rw [inductionHypothesis]

private theorem cubePoint_eq_of_coordinates_eq
    {Field : Type uExtension}
    {variables : Nat}
    (left right : CubePoint Field variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  cases coordinates
  rfl

/-- An honest source constructs fixed-width rounds causally: each polynomial
is fixed from the current challenge prefix, absorbed into the transcript, and
only then is the corresponding challenge sampled. -/
theorem exists_honestPiCcsCertificate
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (source : SourceValid key running fresh witness) :
    exists rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound,
    exists fullOutput : FullOutput Extension shape,
      piCcsCheck key running fresh (prefixProof running rounds fullOutput) = true /\
      fullOutput = FullOutput.honestAt key.baseOps key.extensionOps key.lift
        ((key.statement running fresh).sourceConnectedInputs witness)
        (key.piCcsExecution running fresh
          (prefixProof running rounds fullOutput)).coins.roundPoint := by
  let statement := key.statement running fresh
  let data := statement.sourceProtocolData key.lift witness
  let input := statement.verifierInput key.lift
  let transcriptStatement : ProtocolVerifier.Statement Extension State shape := {
    priorState := key.initialTranscriptState
    input := input
  }
  let pre := FiatShamir.derivePreSumcheck key.oracle.transcript
    transcriptStatement
  let indices := canonicalFinIndices shape.cubeVariables
  let rawStep := ProtocolVerifier.HonestCompleteness.indexedChallengeStep
    key.oracle.transcript pre.gamma
  let fixedStep := fun current
      (polynomial : SumCheck.Finite.FixedPolynomial Extension degreeBound) =>
    rawStep current polynomial.toMessage
  let q := ProtocolPolynomial.polynomial key.extensionOps data
    pre.alpha pre.gamma
  have degreeLe : data.toVerifierInput.sumcheckDegreeBound <= degreeBound := by
    simpa [data, statement] using
      key.statement_sumcheckDegreeBound_le running fresh
  have roundRepresentable :
      SumCheck.Finite.FixedPhase.Sequential.RoundRepresentable
        key.extensionOps.toOps q degreeBound shape.cubeVariables := by
    intro fixed remaining length
    rcases
        (ProtocolPolynomialDegree.sequentialRoundRepresentable
          key.extensionOps key.extensionLaws data pre.alpha pre.gamma)
          fixed remaining length with ⟨polynomial, represents⟩
    refine ⟨SumCheck.Finite.FixedPolynomial.widen key.extensionOps.toOps
      degreeLe polynomial, ?_⟩
    intro point
    rw [SumCheck.Finite.FixedPolynomial.evaluate_widen
      key.extensionOps.toOps
      (ProtocolPolynomialDegree.Support.polynomialLaws key.extensionLaws)]
    exact represents point
  rcases SumCheck.Finite.FixedPhase.Sequential.exists_honest_run
      key.extensionOps.toOps q degreeBound shape.cubeVariables fixedStep
      roundRepresentable (indices, pre.state) with
    ⟨fixedCertificate, challenges, finalState, roundsLength,
      challengesLength, fixedReplay, fixedHonest⟩
  let rounds := SumCheck.Finite.FixedPhase.Sequential.functionOfExactList
    fixedCertificate.rounds roundsLength
  let rawRounds : Fin shape.cubeVariables ->
      SumCheck.Finite.Message Extension :=
    fun round => (rounds round).toMessage
  have roundsList : List.ofFn rounds = fixedCertificate.rounds := by
    exact SumCheck.Finite.FixedPhase.Sequential.ofFn_functionOfExactList
      fixedCertificate.rounds roundsLength
  have indexedMessages :
      indices.map rawRounds =
        fixedCertificate.rounds.map
          SumCheck.Finite.FixedPolynomial.toMessage := by
    calc
      indices.map rawRounds = List.ofFn rawRounds := by
        simp [indices, canonicalFinIndices]
      _ = (List.ofFn rounds).map
            SumCheck.Finite.FixedPolynomial.toMessage := by
        apply List.ext_get
        · simp
        · intro index leftBound rightBound
          simp [rawRounds]
      _ = fixedCertificate.rounds.map
            SumCheck.Finite.FixedPolynomial.toMessage := by
        rw [roundsList]
  have rawReplay :
      ProtocolPolynomialHonestProver.runRaw rawStep (indices, pre.state)
          (fixedCertificate.rounds.map
            SumCheck.Finite.FixedPolynomial.toMessage) =
        (challenges, finalState) := by
    rw [← fixedRun_eq_rawRun rawStep]
    simpa [fixedStep] using fixedReplay
  let replay := FiatShamir.deriveRoundsFrom key.oracle.transcript rawRounds
    pre.state indices
  have verifierReplay :
      ProtocolPolynomialHonestProver.runRaw rawStep (indices, pre.state)
          (indices.map rawRounds) =
        (replay.1, ([], replay.2)) := by
    exact ProtocolVerifier.HonestCompleteness.runRaw_indexedChallengeStep
      key.oracle.transcript pre.gamma rawRounds pre.state indices
  have rawReplayIndexed :
      ProtocolPolynomialHonestProver.runRaw rawStep (indices, pre.state)
          (indices.map rawRounds) =
        (challenges, finalState) := by
    rw [indexedMessages]
    exact rawReplay
  have replayCoordinates : replay.1 = challenges := by
    exact congrArg Prod.fst (verifierReplay.symm.trans rawReplayIndexed)
  let roundPoint : CubePoint Extension shape.cubeVariables := {
    coordinates := challenges
    dimension := challengesLength
  }
  let fullOutput := FullOutput.honestAt key.baseOps key.extensionOps key.lift
    (statement.sourceConnectedInputs witness) roundPoint
  let prefixValue := prefixProof running rounds fullOutput
  have derivedAlpha :
      (key.piCcsExecution running fresh prefixValue).coins.alpha = pre.alpha := by
    rfl
  have derivedGamma :
      (key.piCcsExecution running fresh prefixValue).coins.gamma = pre.gamma := by
    rfl
  have derivedPoint :
      (key.piCcsExecution running fresh prefixValue).coins.roundPoint =
        roundPoint := by
    apply cubePoint_eq_of_coordinates_eq
    change replay.1 = challenges
    exact replayCoordinates
  have fixedRounds :
      (key.piCcsFixedCertificate running fresh prefixValue).rounds =
        fixedCertificate.rounds := by
    simpa [Key.piCcsFixedCertificate, prefixValue, prefixProof, rounds] using
      roundsList
  have projected :
      statement.projectOutput fullOutput =
        ProtocolPolynomial.messageAt key.extensionOps data roundPoint := by
    calc
      statement.projectOutput fullOutput =
          fullOutput.toOutputMessage
            (statement.identityFirstMatrix witness) :=
        statement.projectOutput_eq_toOutputMessage witness fullOutput
      _ = ProtocolPolynomial.messageAt key.extensionOps data roundPoint := by
        exact FullOutput.honestAt_toOutputMessage_eq_messageAt key.baseOps
          key.baseLaws key.extensionOps key.lift
          (statement.sourceConnectedInputs witness) key.constantLaw
          (statement.identityFirstMatrix witness) roundPoint
  have coefficientTruth :
      SignedCoefficientObject.CoefficientTruth key.extensionOps
        (data.toJointData key.extensionOps) :=
    (SignedCoefficientObject.coefficientTruth_iff_tableObligations
      key.extensionOps key.extensionZeroLaws
      (data.toJointData key.extensionOps)).2
        (sourceTableTruth key running fresh witness source)
  have sampledZero :
      (SignedCoefficientPolynomial.polynomial key.extensionOps
        (data.toJointData key.extensionOps) pre.alpha).evaluate
          key.extensionOps.toOps pre.gamma = key.extensionOps.zero :=
    SignedCoefficientObject.evaluate_eq_zero_of_coefficientTruth
      key.extensionOps key.extensionLaws (data.toJointData key.extensionOps)
      pre.alpha pre.gamma coefficientTruth
  have jointInitialTrue :
      data.toVerifierInput.initial key.extensionOps pre.gamma =
        SumCheckInitial.semanticInitial key.extensionOps
          (data.toJointData key.extensionOps) pre.alpha pre.gamma := by
    have claimTrue :=
      (SumCheckInitial.claimTrue_iff_polynomial_evaluate_eq_zero
        key.extensionOps key.extensionLaws (data.toJointData key.extensionOps)
        pre.alpha pre.gamma degreeBound 0 challenges (q challenges)
        (key.piCcsCertificate running fresh prefixValue).toFinite []).2 sampledZero
    simpa [SumCheck.Claim.True, SumCheckInitial.symbolicInstance] using
      claimTrue
  have initialIsTrue :
      data.toVerifierInput.initial key.extensionOps pre.gamma =
        SumCheck.Finite.FixedPhase.semanticInitial key.extensionOps.toOps q
          challenges.length := by
    rw [challengesLength]
    unfold SumCheck.Finite.FixedPhase.semanticInitial
    dsimp only [q]
    rw [ProtocolPolynomial.sumCompletions_polynomial_eq_summedQ
      key.extensionOps key.extensionLaws data pre.alpha pre.gamma]
    rw [ProtocolPolynomial.verifierInput_initial_eq_joint_initial]
    exact jointInitialTrue
  have fixedAccepted :
      SumCheck.Finite.FixedPhase.Accepted key.extensionOps.toOps q
        (data.toVerifierInput.initial key.extensionOps pre.gamma) challenges
        fixedCertificate :=
    SumCheck.Finite.FixedPhase.complete key.extensionOps.toOps q
      (data.toVerifierInput.initial key.extensionOps pre.gamma) challenges
      fixedCertificate initialIsTrue fixedHonest
  have terminalExact :
      ProtocolPolynomial.terminalFromMessage key.extensionOps input
          pre.alpha pre.gamma roundPoint
          (statement.projectOutput fullOutput) = q challenges := by
    rw [projected]
    unfold q ProtocolPolynomial.polynomial
    rw [dif_pos challengesLength]
    have rebuiltPoint :
        ({ coordinates := challenges, dimension := challengesLength } :
          CubePoint Extension shape.cubeVariables) = roundPoint := by
      rfl
    rw [rebuiltPoint]
    unfold ProtocolPolynomial.qAtPoint
    rw [statement.sourceProtocolData_toVerifierInput key.lift witness]
  have checked : piCcsCheck key running fresh prefixValue = true := by
    apply (piCcsCheck_eq_true_iff key running fresh prefixValue).2
    unfold SumCheck.Finite.FixedPhase.Accepted at fixedAccepted
    rw [derivedAlpha, derivedGamma, derivedPoint, fixedRounds]
    change SumCheck.Finite.FixedPhase.Chain key.extensionOps.toOps
      (input.initial key.extensionOps pre.gamma)
      fixedCertificate.rounds challenges
      (ProtocolPolynomial.terminalFromMessage key.extensionOps input
        pre.alpha pre.gamma roundPoint (statement.projectOutput fullOutput))
    rw [terminalExact]
    simpa [data, input, statement, q, roundPoint] using fixedAccepted
  refine ⟨rounds, fullOutput, ?_, ?_⟩
  · simpa [prefixValue] using checked
  · rw [derivedPoint]

/-- Honest combined assignment under the exact post-`Pi_CCS` challenge
vector. -/
def honestParentAssignment
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound)
    (fullOutput : FullOutput Extension shape) : Assignment F columns :=
  PiRLC.combinedWitness key.piRlcAlgebra
    (key.piRlcChallenges running fresh
      (prefixProof running rounds fullOutput))
    (sourceAssignments key witness)

/-- Honest private child assignment at one `Pi_DEC` output coordinate. -/
def honestChildAssignment
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound)
    (fullOutput : FullOutput Extension shape)
    (child : Fin key.params.k) : Assignment F columns :=
  key.piDecAlgebra.splitAssignment
    (honestParentAssignment key running fresh witness rounds fullOutput)
    child

/-- Exact paper evaluation family for one honest private child. -/
def honestChildEvaluation
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound)
    (fullOutput : FullOutput Extension shape)
    (runningIndex : Fin shape.runningCount) : EvaluationFamily Extension shape :=
  fun matrix coefficient =>
    (BooleanTable.tabulate fun vertex =>
      key.lift (matrixVectorAt key.baseOps
        (key.matrixSource.coefficientMatrix key.baseOps matrix coefficient)
        (honestChildAssignment key running fresh witness rounds fullOutput
          (Fin.cast key.runningCount_eq_outputCount runningIndex))
        vertex)).evaluate key.extensionOps
          (key.piCcsExecution running fresh
            (prefixProof running rounds fullOutput)).coins.roundPoint

/-- The sole honest NIFS message: causal `Pi_CCS` prefix plus private-split
`Pi_DEC` commitment/evaluation messages. -/
def honestProof
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound)
    (fullOutput : FullOutput Extension shape) :
    Proof Extension Commitment shape degreeBound where
  piCcsRounds := rounds
  piCcsOutput := fullOutput
  piDecCommitments := fun runningIndex =>
    key.semantics.commit
      (honestChildAssignment key running fresh witness rounds fullOutput
        (Fin.cast key.runningCount_eq_outputCount runningIndex))
  piDecEvaluations :=
    honestChildEvaluation key running fresh witness rounds fullOutput

/-- The honest proof and its prefix have exactly the same `Pi_CCS` transcript
and output payload.  The later child fields cannot influence any challenge. -/
theorem honestProof_piCcsCertificate
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound)
    (fullOutput : FullOutput Extension shape) :
    key.piCcsCertificate running fresh
        (honestProof key running fresh witness rounds fullOutput) =
      key.piCcsCertificate running fresh
        (prefixProof running rounds fullOutput) := by
  rfl

/-- Every public `Pi_CCS` output of the honest proof is a fresh `CE(b)`
opening of the same authoritative source assignment. -/
theorem honestPiCcsOutputs_hold
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (source : SourceValid key running fresh witness)
    (rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound)
    (fullOutput : FullOutput Extension shape)
    (fullOutputHonest :
      fullOutput = FullOutput.honestAt key.baseOps key.extensionOps key.lift
        ((key.statement running fresh).sourceConnectedInputs witness)
        (key.piCcsExecution running fresh
          (prefixProof running rounds fullOutput)).coins.roundPoint) :
    forall index,
      CE.Holds key.semantics key.params
        (key.piCcsOutputs running fresh
          (honestProof key running fresh witness rounds fullOutput) index)
        (sourceAssignments key witness index) := by
  intro index
  let sourceIndex : Fin shape.sourceCount :=
    Fin.cast key.total_eq_sourceCount index
  have opening := source.1 sourceIndex
  have executionEq :
      key.piCcsExecution running fresh
          (honestProof key running fresh witness rounds fullOutput) =
        key.piCcsExecution running fresh
          (prefixProof running rounds fullOutput) := by
    unfold Key.piCcsExecution
    rw [honestProof_piCcsCertificate key running fresh witness rounds fullOutput]
  refine ⟨?_, ?_, ?_⟩
  · simpa [Key.piCcsOutputs, Key.piCcsProbe, Key.statement,
      sourceAssignments, sourceIndex, Key.semantics, NormStage.bound] using
        opening
  · trivial
  · change
      key.semantics.evaluations key.matrixSource
          (sourceAssignments key witness index)
          (key.piCcsExecution running fresh
            (honestProof key running fresh witness rounds fullOutput)).coins.roundPoint =
        #[fun matrix coefficient =>
          fullOutput.coordinate sourceIndex matrix coefficient]
    rw [executionEq, fullOutputHonest]
    rfl

/-- The verifier-computed `Pi_RLC` parent has the honest combined opening. -/
theorem honestParent_holds
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (source : SourceValid key running fresh witness)
    (rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound)
    (fullOutput : FullOutput Extension shape)
    (fullOutputHonest :
      fullOutput = FullOutput.honestAt key.baseOps key.extensionOps key.lift
        ((key.statement running fresh).sourceConnectedInputs witness)
        (key.piCcsExecution running fresh
          (prefixProof running rounds fullOutput)).coins.roundPoint) :
    CE.Holds key.semantics key.params
      (key.parent running fresh
        (honestProof key running fresh witness rounds fullOutput))
      (honestParentAssignment key running fresh witness rounds
        fullOutput) := by
  let proof := honestProof key running fresh witness rounds fullOutput
  have inputsValid := honestPiCcsOutputs_hold key running fresh witness source
    rounds fullOutput fullOutputHonest
  have pointValid :
      key.semantics.evaluationPointValid key.matrixSource
        (key.piCcsExecution running fresh proof).coins.roundPoint := by
    trivial
  simpa [Key.parent, honestParentAssignment, proof, honestProof, prefixProof] using
    (PiRLC.combinedOutput_holds key.semantics key.params key.piRlcAlgebra
      key.arity key.matrixSource
      (key.piCcsExecution running fresh proof).coins.roundPoint
      (key.piCcsOutputs running fresh proof)
      (key.piRlcChallenges running fresh proof)
      (sourceAssignments key witness)
      (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
      (key.piRlcResponseValid
        (key.piCcsExecution running fresh proof).outgoingState)
      inputsValid pointValid)

/-- The child fields materialized by `honestProof` are exactly the paper
verifier's private-split honest attempt. -/
theorem honestProof_piDecAttempt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (rounds : Fin shape.cubeVariables ->
      SumCheck.Finite.FixedPolynomial Extension degreeBound)
    (fullOutput : FullOutput Extension shape) :
    key.piDecAttempt running fresh
        (honestProof key running fresh witness rounds fullOutput) =
      PiDEC.PaperVerifier.honestAttempt key.piDecAlgebra
        (key.parent running fresh
          (honestProof key running fresh witness rounds fullOutput))
        (honestParentAssignment key running fresh witness rounds
          fullOutput) := by
  unfold Key.piDecAttempt PiDEC.PaperVerifier.honestAttempt
  congr 1

/-- Every independently valid paper source product constructs a concrete
accepted one-message NIFS proof and an independently witnessed transition. -/
theorem sourceValid_exists_verifiedTransition
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (sourceWitness : OutputWitness shape columns)
    (source : SourceValid key running fresh sourceWitness) :
    exists proof : Proof Extension Commitment shape degreeBound,
    exists result : Running Extension Commitment PublicInput shape,
      verify key running fresh proof = some result /\
      Transition key running fresh result := by
  rcases exists_honestPiCcsCertificate key running fresh sourceWitness source with
    ⟨rounds, fullOutput, prefixChecked, fullOutputHonest⟩
  let proof := honestProof key running fresh sourceWitness rounds fullOutput
  let parentAssignment := honestParentAssignment key running fresh sourceWitness
    rounds fullOutput
  let childAssignments : Fin key.params.k -> Assignment F columns :=
    honestChildAssignment key running fresh sourceWitness rounds fullOutput
  have piCcsChecked : piCcsCheck key running fresh proof = true := by
    simpa [proof, honestProof, prefixProof] using prefixChecked
  have inputsValid := honestPiCcsOutputs_hold key running fresh sourceWitness
    source rounds fullOutput fullOutputHonest
  have parentValid :
      CE.Holds key.semantics key.params (key.parent running fresh proof)
        parentAssignment := by
    exact honestParent_holds key running fresh sourceWitness source rounds
      fullOutput fullOutputHonest
  have parentCombined : (key.parent running fresh proof).stage = .combined := by
    rfl
  have operational := PiDEC.PaperVerifier.complete key.semantics key.params
    key.piDecAlgebra key.piDecPublicInputSplit key.piDecEvaluationArity
    (key.parent running fresh proof) parentAssignment parentCombined parentValid
  have attemptEq :
      key.piDecAttempt running fresh proof =
        PiDEC.PaperVerifier.honestAttempt key.piDecAlgebra
          (key.parent running fresh proof) parentAssignment := by
    exact honestProof_piDecAttempt key running fresh sourceWitness rounds
      fullOutput
  have piDecAccepted : PiDEC.PaperVerifier.Accepted key.piDecAlgebra
      key.piDecEvaluationArity (key.piDecAttempt running fresh proof) := by
    rw [attemptEq]
    exact operational.1
  have childrenValid : forall child,
      CE.Holds key.semantics key.params
        (PiDEC.PaperVerifier.children key.piDecPublicInputSplit
          (key.piDecAttempt running fresh proof) child)
        (childAssignments child) := by
    intro child
    rw [attemptEq]
    exact operational.2 child
  have parentAssignmentEq :
      PiRLC.combinedWitness key.piRlcAlgebra
          (key.piRlcChallenges running fresh proof)
          (sourceAssignments key sourceWitness) =
        key.piDecAlgebra.recomposeAssignment childAssignments := by
    simpa [proof, parentAssignment, childAssignments, honestParentAssignment,
      honestChildAssignment, honestProof, prefixProof] using
        (key.piDecAlgebra.split_recompose parentAssignment).symm
  have piDecChecked : piDecCheck key running fresh proof = true :=
    (piDecCheck_eq_true_iff key running fresh proof).2 piDecAccepted
  let result := key.output running fresh proof
  have verified : verify key running fresh proof = some result :=
    (verify_eq_some_iff key running fresh proof result).2
      ⟨piCcsChecked, piDecChecked, rfl⟩
  have transition : Transition key running fresh result := by
    refine ⟨proof, sourceWitness, childAssignments, ?_⟩
    exact {
      piCcsRoundChain :=
        piCcsRoundChain_of_check key running fresh proof piCcsChecked
      piDecParentCombined := piDecAccepted.parentCombined
      piDecParentEvaluationSize := piDecAccepted.parentEvaluationSize
      piDecMessageEvaluationSize := piDecAccepted.messageEvaluationSize
      piDecCommitmentEquation := piDecAccepted.commitmentEquation
      piDecEvaluationEquation := piDecAccepted.evaluationEquation
      sourceValid := source
      piCcsInputsValid := inputsValid
      childValid := childrenValid
      parentAssignment := parentAssignmentEq
      resultComputed := rfl
    }
  exact ⟨proof, result, verified, transition⟩

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
