import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree

/-!
Soundness and graph completeness of the executable paper SuperNeo NIFS.

Owns: composition of the transcript-bound joint `Pi_CCS` checker, computed
`Pi_RLC` parent, and operational `Pi_DEC` checker into the independent
transition from `Semantics`; and the converse construction of executable
acceptance from that transition's expanded equations.

Does not own: probability bounds for the named events, a concrete transcript
or commitment, HyperNova/F-prime, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

Neither theorem receives source validity, child validity, the desired
transition, or a generic refinement proposition as a soundness premise.

| Direction | Mathematical obligation | Main theorem |
|---|---|---|
| checker exposure | recover the exact fixed-width joint-SumCheck chain from the Boolean check | `piCcsRoundChain_of_check` |
| soundness | executable acceptance gives the independent transition or one closed named event | `verify_sound` |
| completeness | the independently expanded transition equations construct an accepted proof | `verify_complete` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uState

/-- The Boolean joint-`Pi_CCS` check exposes the exact fixed-width claimed-chain
equation stored by the independent transition. -/
theorem piCcsRoundChain_of_check
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
    (proof : Proof Extension Commitment shape degreeBound)
    (checked : piCcsCheck key running fresh proof = true) :
    SumCheck.Finite.FixedPhase.Chain key.extensionOps.toOps
      (((key.statement running fresh).verifierInput key.lift).initial
        key.extensionOps
        (key.piCcsExecution running fresh proof).coins.gamma)
      (key.piCcsFixedCertificate running fresh proof).rounds
      (key.piCcsExecution running fresh proof).coins.roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage key.extensionOps
        ((key.statement running fresh).verifierInput key.lift)
        (key.piCcsExecution running fresh proof).coins.alpha
        (key.piCcsExecution running fresh proof).coins.gamma
        (key.piCcsExecution running fresh proof).coins.roundPoint
        (key.piCcsCertificate running fresh proof).output) := by
  exact (piCcsCheck_eq_true_iff key running fresh proof).1 checked

/-- Fixed-width `Pi_CCS` acceptance reaches source truth or one of the two
paper algebraic failures.  Message width is supplied by the proof type and
the key proves that the independently derived polynomial fits that width. -/
private theorem piCcsSourceValid_or_badEvent
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
    (proof : Proof Extension Commitment shape degreeBound)
    (witness : OutputWitness shape columns)
    (ambient : AmbientTargetOpenings key running fresh proof witness)
    (checked : piCcsCheck key running fresh proof = true) :
    SourceValid key running fresh witness ∨
      PiCcsMixingRoot key running fresh proof witness ∨
      PiCcsSumCheckCollision key running fresh proof witness := by
  let statement := key.statement running fresh
  let data := statement.sourceProtocolData key.lift witness
  let execution := key.piCcsExecution running fresh proof
  let q := ProtocolPolynomial.polynomial key.extensionOps data
    execution.coins.alpha execution.coins.gamma
  let fixedCertificate := key.piCcsFixedCertificate running fresh proof
  have chain := piCcsRoundChain_of_check key running fresh proof checked
  have outputExact :
      statement.projectOutput proof.piCcsOutput =
        ProtocolPolynomial.messageAt key.extensionOps data
          execution.coins.roundPoint := by
    simpa [statement, data, execution, Key.piCcsProbe] using
      (projectedOutput_eq_messageAt_of_ambientOutputHolds
        key.baseLaws key.baseZero key.extensionOps key.lift key.openingMaps key.params
        statement key.constantLaw (key.piCcsProbe running fresh proof)
        witness ambient)
  have terminalExact :
      ProtocolPolynomial.terminalFromMessage key.extensionOps
          (statement.verifierInput key.lift)
          execution.coins.alpha execution.coins.gamma
          execution.coins.roundPoint
          (statement.projectOutput proof.piCcsOutput) =
        q execution.coins.roundPoint.coordinates := by
    rw [outputExact]
    unfold q ProtocolPolynomial.polynomial
    rw [dif_pos execution.coins.roundPoint.dimension]
    have rebuiltPoint :
        ({
          coordinates := execution.coins.roundPoint.coordinates
          dimension := execution.coins.roundPoint.dimension
        } : CubePoint Extension shape.cubeVariables) =
          execution.coins.roundPoint := by
      cases execution.coins.roundPoint
      rfl
    rw [show
      ({
        coordinates := execution.coins.roundPoint.coordinates
        dimension := execution.coins.roundPoint.dimension
      } : CubePoint Extension shape.cubeVariables) =
        execution.coins.roundPoint from rebuiltPoint]
    unfold ProtocolPolynomial.qAtPoint
    rw [statement.sourceProtocolData_toVerifierInput key.lift witness]
  have fixedAccepted :
      SumCheck.Finite.FixedPhase.Accepted key.extensionOps.toOps q
        ((statement.verifierInput key.lift).initial key.extensionOps
          execution.coins.gamma)
        execution.coins.roundPoint.coordinates fixedCertificate := by
    unfold SumCheck.Finite.FixedPhase.Accepted
    rw [← terminalExact]
    simpa [statement, execution, fixedCertificate, Key.piCcsCertificate] using
      chain
  have expectedAtExactDegree :=
    ProtocolPolynomialDegree.expectedRoundsRepresentable key.extensionOps
      key.extensionLaws data execution.coins.alpha execution.coins.gamma
      execution.coins.roundPoint
  have expectedAtBound :
      SumCheck.Finite.FixedPhase.ExpectedRoundsRepresentable
        key.extensionOps.toOps q degreeBound
        execution.coins.roundPoint.coordinates := by
    intro expected expectedIn
    rcases expectedAtExactDegree expected (by simpa [q] using expectedIn) with
      ⟨polynomial, represents⟩
    refine ⟨SumCheck.Finite.FixedPolynomial.widen key.extensionOps.toOps
      (key.statement_sumcheckDegreeBound_le running fresh) polynomial, ?_⟩
    intro point
    rw [SumCheck.Finite.FixedPolynomial.evaluate_widen
      key.extensionOps.toOps
      (ProtocolPolynomialDegree.Support.polynomialLaws key.extensionLaws)]
    exact represents point
  let jointData := data.toJointData key.extensionOps
  by_cases tableTruth :
      (TableResidualData.toTableObligations key.extensionOps
        (SignedCoefficientObject.toTableResidualData key.extensionOps
          jointData)).AllHold
  · left
    let unifiedData :=
      statement.sourceConnectedInputs witness |>.toUnifiedInputs key.baseOps
    have independentTableTruth :
        (TableResidualData.toTableObligations key.extensionOps
          (SignedCoefficientObject.toTableResidualData key.extensionOps
            (unifiedData.toIndependentInputs.toJointData
              key.baseOps key.lift))).AllHold := by
      rw [← ProtocolDataRefinement.toProtocolData_toJointData_eq
        key.baseOps key.extensionOps key.lift key.liftLaws unifiedData]
      simpa [data, jointData, statement, unifiedData] using tableTruth
    have independentSemantic :=
      (ConcreteJointData.jointTableTruth_iff_semanticTruth
        key.baseOps key.baseZero key.noZeroDivisors key.extensionOps
        key.extensionLaws key.lift key.liftLaws.toZeroReflectingLift
        unifiedData.toIndependentInputs).mp independentTableTruth
    have sourceSemantic :
        (statement.sourceConnectedInputs witness).SemanticTruth
          key.baseOps key.extensionOps key.lift := by
      simpa [ConnectedInputs.SemanticTruth, unifiedData] using
        (unifiedData.toIndependentInputs_semanticTruth_iff
          key.baseOps key.extensionOps key.lift).mp independentSemantic
    refine ⟨?_, sourceSemantic⟩
    intro source
    have ambientOpening := (ambient source).1
    refine ⟨ambientOpening.1, ambientOpening.2.1, ?_⟩
    intro column
    change centeredMagnitude (witness.assignments source column) < key.params.b
    rw [key.freshBound]
    exact sourceSemantic.2.1 source column
  · by_cases mixingRoot :
        SignedCoefficientObject.MixingRoot key.extensionOps jointData
          execution.coins.alpha execution.coins.gamma
    · exact Or.inr (Or.inl (by
        simpa [PiCcsMixingRoot, jointData, data, statement, execution,
          Key.piCcsProbe] using mixingRoot))
    · right
      right
      have falseInitial :
          (statement.verifierInput key.lift).initial key.extensionOps
              execution.coins.gamma ≠
            SumCheck.Finite.FixedPhase.semanticInitial key.extensionOps.toOps q
              execution.coins.roundPoint.coordinates.length := by
        intro initialTrue
        have jointInitialTrue :
            (statement.verifierInput key.lift).initial key.extensionOps
                execution.coins.gamma =
              SumCheckInitial.semanticInitial key.extensionOps jointData
                execution.coins.alpha execution.coins.gamma := by
          calc
            _ = SumCheck.Finite.FixedPhase.semanticInitial
                  key.extensionOps.toOps q
                  execution.coins.roundPoint.coordinates.length := initialTrue
            _ = _ := by
              rw [execution.coins.roundPoint.dimension]
              unfold SumCheck.Finite.FixedPhase.semanticInitial q jointData
              rw [ProtocolPolynomial.sumCompletions_polynomial_eq_summedQ
                key.extensionOps key.extensionLaws data
                execution.coins.alpha execution.coins.gamma]
              rfl
        have polynomialZero :
            (SignedCoefficientPolynomial.polynomial key.extensionOps jointData
              execution.coins.alpha).evaluate key.extensionOps.toOps
                execution.coins.gamma = key.extensionOps.zero := by
          apply (SumCheckInitial.claimTrue_iff_polynomial_evaluate_eq_zero
            key.extensionOps key.extensionLaws jointData
            execution.coins.alpha execution.coins.gamma degreeBound
            key.challengeSetSize execution.coins.roundPoint.coordinates
            (q execution.coins.roundPoint.coordinates)
            (key.piCcsCertificate running fresh proof).toFinite []).1
          simpa [SumCheck.Claim.True, SumCheckInitial.symbolicInstance] using
            jointInitialTrue
        rcases
            (SignedCoefficientObject.evaluate_eq_zero_iff_coefficientTruth_or_mixingRoot
              key.extensionOps key.extensionLaws jointData
              execution.coins.alpha execution.coins.gamma).1 polynomialZero with
          coefficientTruth | root
        · exact tableTruth
            ((SignedCoefficientObject.coefficientTruth_iff_tableObligations
              key.extensionOps key.extensionZeroLaws jointData).1
                coefficientTruth)
        · exact mixingRoot root
      have collision :=
        SumCheck.Finite.FixedPhase.false_acceptance_implies_bad_challenge
          key.extensionOps.toOps q key.challengeSetSize
          ((statement.verifierInput key.lift).initial key.extensionOps
            execution.coins.gamma)
          execution.coins.roundPoint.coordinates fixedCertificate
          expectedAtBound fixedAccepted falseInitial
      simpa [PiCcsSumCheckCollision, q, data, statement, execution,
        fixedCertificate] using collision

/-- Exact deterministic NIFS soundness.  The only alternatives to the
independent transition are the five closed failure constructors in
`BadEvent`. -/
theorem verify_sound
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
    (proof : Proof Extension Commitment shape degreeBound)
    (result : Running Extension Commitment PublicInput shape)
    (accepted : verify key running fresh proof = some result) :
    Transition key running fresh result ∨
      BadEvent key running fresh proof result := by
  rcases (verify_eq_some_iff key running fresh proof result).1 accepted with
    ⟨piCcsChecked, piDecChecked, resultComputed⟩
  have piDecAccepted : PiDEC.PaperVerifier.Accepted key.piDecAlgebra
      key.piDecEvaluationArity (key.piDecAttempt running fresh proof) :=
    (piDecCheck_eq_true_iff key running fresh proof).1 piDecChecked
  by_cases ambientExists : exists sourceWitness : OutputWitness shape columns,
      AmbientTargetOpenings key running fresh proof sourceWitness
  · rcases ambientExists with ⟨sourceWitness, ambient⟩
    rcases piCcsSourceValid_or_badEvent key running fresh proof sourceWitness
        ambient piCcsChecked with
      sourceValid | mixingRoot | sumCheckCollision
    · by_cases childExists :
          exists childAssignments : Fin key.params.k -> Assignment F columns,
            forall child,
              CE.Holds key.semantics key.params
                (PiDEC.PaperVerifier.children key.piDecPublicInputSplit
                  (key.piDecAttempt running fresh proof) child)
                (childAssignments child)
      · rcases childExists with ⟨childAssignments, childrenValid⟩
        have inputsValid := piCcsOutputs_hold key running fresh proof
          sourceWitness sourceValid ambient
        let first : Fin key.arity.total := ⟨0, key.arity.totalPositive⟩
        have pointValid :
            key.semantics.evaluationPointValid key.matrixSource
              (key.piCcsExecution running fresh proof).coins.roundPoint := by
          simpa [Key.piCcsOutputs, Key.piCcsProbe, Key.statement, first] using
            (inputsValid first).2.1
        have parentValid :
            CE.Holds key.semantics key.params
              (key.parent running fresh proof)
              (PiRLC.combinedWitness key.piRlcAlgebra
                (key.piRlcChallenges running fresh proof)
                (sourceAssignments key sourceWitness)) := by
          simpa [Key.parent] using
            (PiRLC.combinedOutput_holds key.semantics key.params
              key.piRlcAlgebra key.arity key.matrixSource
              (key.piCcsExecution running fresh proof).coins.roundPoint
              (key.piCcsOutputs running fresh proof)
              (key.piRlcChallenges running fresh proof)
              (sourceAssignments key sourceWitness)
              (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
              (key.piRlcResponseValid
                (key.piCcsExecution running fresh proof).outgoingState)
              inputsValid pointValid)
        have parentValidForAttempt :
            CE.Holds key.semantics key.params
              (key.piDecAttempt running fresh proof).parent
              (PiRLC.combinedWitness key.piRlcAlgebra
                (key.piRlcChallenges running fresh proof)
                (sourceAssignments key sourceWitness)) := by
          simpa [Key.piDecAttempt] using parentValid
        rcases PiDEC.PaperVerifier.parent_eq_recompose_or_bindingCollision
            key.semantics key.params key.piDecAlgebra key.piDecPublicInputSplit
            key.piDecEvaluationArity (key.piDecAttempt running fresh proof)
            (PiRLC.combinedWitness key.piRlcAlgebra
              (key.piRlcChallenges running fresh proof)
              (sourceAssignments key sourceWitness))
            childAssignments piDecAccepted parentValidForAttempt childrenValid with
          parentAssignment | bindingCollision
        · left
          refine ⟨proof, sourceWitness, childAssignments, ?_⟩
          exact {
            piCcsRoundChain :=
              piCcsRoundChain_of_check key running fresh proof piCcsChecked
            piDecParentCombined := piDecAccepted.parentCombined
            piDecParentEvaluationSize := piDecAccepted.parentEvaluationSize
            piDecMessageEvaluationSize := piDecAccepted.messageEvaluationSize
            piDecCommitmentEquation := piDecAccepted.commitmentEquation
            piDecEvaluationEquation := piDecAccepted.evaluationEquation
            sourceValid := sourceValid
            piCcsInputsValid := inputsValid
            childValid := childrenValid
            parentAssignment := parentAssignment
            resultComputed := resultComputed
          }
        · right
          exact .parentBindingCollision bindingCollision
      · right
        exact .piDecChildExtraction childExists
    · right
      exact .piCcsMixingRoot sourceWitness mixingRoot
    · right
      exact .piCcsSumCheckCollision sourceWitness sumCheckCollision
  · right
    exact .piRlcCoordinateForkExtraction ambientExists

/-- Graph completeness from the independently expanded paper equations.  No
checker result or bundled acceptance predicate occurs in `Transition`. -/
theorem verify_complete
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
    (result : Running Extension Commitment PublicInput shape)
    (transition : Transition key running fresh result) :
    exists proof, verify key running fresh proof = some result := by
  rcases transition with
    ⟨proof, sourceWitness, childAssignments, realization⟩
  have piCcsChecked : piCcsCheck key running fresh proof = true := by
    exact (piCcsCheck_eq_true_iff key running fresh proof).2
      realization.piCcsRoundChain
  have piDecAccepted : PiDEC.PaperVerifier.Accepted key.piDecAlgebra
      key.piDecEvaluationArity (key.piDecAttempt running fresh proof) := {
    parentCombined := realization.piDecParentCombined
    parentEvaluationSize := realization.piDecParentEvaluationSize
    messageEvaluationSize := realization.piDecMessageEvaluationSize
    commitmentEquation := realization.piDecCommitmentEquation
    evaluationEquation := realization.piDecEvaluationEquation
  }
  have piDecChecked : piDecCheck key running fresh proof = true :=
    (piDecCheck_eq_true_iff key running fresh proof).2 piDecAccepted
  exact ⟨proof, (verify_eq_some_iff key running fresh proof result).2
    ⟨piCcsChecked, piDecChecked, realization.resultComputed⟩⟩

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
