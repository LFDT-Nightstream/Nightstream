import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteJointData

/-!
Verifier-owned Fiat--Shamir authority for paper-level joint `Pi_CCS`.

Protocol: SuperNeo v1.2 `Pi_CCS` (Section 7.3 / Appendix B.2).
Phase: pre-SumCheck alpha/gamma sampling and interleaved SumCheck rounds.
Constraint family: transcript ownership and challenge schedule only.

Owns: a typed challenge-label schedule, a certificate containing only finite
polynomial messages, deterministic derivation of alpha, gamma, and every
SumCheck round challenge from public context plus those messages, and an
explicit residual-table audit checker retained for differential reasoning.

Does not own: construction of a complete public statement from `Context`, the
nonlinear off-cube paper polynomial, the prover's output evaluation message,
absorption of that message into the outgoing transcript, a concrete transcript
encoding, Poseidon2, domain-tag field values, random-oracle or collision
security, challenge-distribution bounds, root-counting probability, the exact
semantic degree theorem, Rust, R1CS, or constraint counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiatShamir

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uContext uField uState uExtension

inductive ChallengeLabel (shape : Shape) where
  | alpha (coordinate : Fin shape.cubeVariables)
  | gamma
  | sumcheck (round : Fin shape.cubeVariables)
deriving Repr, DecidableEq

structure Oracle
    (Context : Type uContext)
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  initialState : Context -> State
  absorbRound : State -> Fin shape.cubeVariables ->
    SumCheck.Finite.Message Field -> State
  squeeze : State -> ChallengeLabel shape -> Field × State

structure Certificate (Field : Type uField) (shape : Shape) where
  rounds : Fin shape.cubeVariables -> SumCheck.Finite.Message Field

namespace Certificate

def toFinite
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    SumCheck.Finite.Certificate Field where
  rounds := List.ofFn certificate.rounds

theorem toFinite_rounds_length
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    certificate.toFinite.rounds.length = shape.cubeVariables := by
  simp [toFinite]

end Certificate

def alphaLabels (shape : Shape) : List (ChallengeLabel shape) :=
  (canonicalFinIndices shape.cubeVariables).map ChallengeLabel.alpha

theorem alphaLabels_length (shape : Shape) :
    (alphaLabels shape).length = shape.cubeVariables := by
  simp [alphaLabels, canonicalFinIndices_length]

def squeezeMany
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Context Field State shape) :
    State -> List (ChallengeLabel shape) -> List Field × State
  | state, [] => ([], state)
  | state, label :: labels =>
      let sample := oracle.squeeze state label
      let tail := squeezeMany oracle sample.2 labels
      (sample.1 :: tail.1, tail.2)

theorem squeezeMany_values_length
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (state : State)
    (labels : List (ChallengeLabel shape)) :
    (squeezeMany oracle state labels).1.length = labels.length := by
  induction labels generalizing state with
  | nil => rfl
  | cons label labels inductionHypothesis =>
      simp only [squeezeMany, List.length_cons]
      rw [inductionHypothesis]

structure PreSumcheck (Field : Type uField) (State : Type uState)
    (shape : Shape) where
  alpha : CubePoint Field shape.cubeVariables
  gamma : Field
  state : State

/-- Derive alpha and gamma from the public context before absorbing any prover
message, matching step 1 of the paper protocol. -/
def derivePreSumcheck
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context) : PreSumcheck Field State shape :=
  let alphaResult := squeezeMany oracle (oracle.initialState context)
    (alphaLabels shape)
  let gammaResult := oracle.squeeze alphaResult.2 .gamma
  {
    alpha := {
      coordinates := alphaResult.1
      dimension := by
        rw [squeezeMany_values_length oracle, alphaLabels_length]
    }
    gamma := gammaResult.1
    state := gammaResult.2
  }

def deriveRoundsFrom
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (rounds : Fin shape.cubeVariables -> SumCheck.Finite.Message Field) :
    State -> List (Fin shape.cubeVariables) -> List Field × State
  | state, [] => ([], state)
  | state, round :: remaining =>
      let absorbed := oracle.absorbRound state round (rounds round)
      let sample := oracle.squeeze absorbed (.sumcheck round)
      let tail := deriveRoundsFrom oracle rounds sample.2 remaining
      (sample.1 :: tail.1, tail.2)

theorem deriveRoundsFrom_values_length
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (rounds : Fin shape.cubeVariables -> SumCheck.Finite.Message Field)
    (state : State)
    (indices : List (Fin shape.cubeVariables)) :
    (deriveRoundsFrom oracle rounds state indices).1.length = indices.length := by
  induction indices generalizing state with
  | nil => rfl
  | cons round remaining inductionHypothesis =>
      simp only [deriveRoundsFrom, List.length_cons]
      rw [inductionHypothesis]

structure DerivedCoins (Field : Type uField) (State : Type uState)
    (shape : Shape) where
  alpha : CubePoint Field shape.cubeVariables
  gamma : Field
  roundPoint : CubePoint Field shape.cubeVariables
  finalState : State

def derive
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (certificate : Certificate Field shape) :
    DerivedCoins Field State shape :=
  let pre := derivePreSumcheck oracle context
  let roundResult := deriveRoundsFrom oracle certificate.rounds pre.state
    (canonicalFinIndices shape.cubeVariables)
  {
    alpha := pre.alpha
    gamma := pre.gamma
    roundPoint := {
      coordinates := roundResult.1
      dimension := by
        rw [deriveRoundsFrom_values_length oracle certificate.rounds,
          canonicalFinIndices_length]
    }
    finalState := roundResult.2
  }

def checkResidualTableAudit
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (maxDegree : Nat)
    (certificate : Certificate Field shape) : Bool :=
  let coins := derive oracle context certificate
  SumCheckInitial.checkJoint ops data coins.alpha coins.gamma maxDegree
    coins.roundPoint certificate.toFinite

theorem checkResidualTableAudit_eq_true_iff_accepted
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (maxDegree : Nat)
    (certificate : Certificate Field shape) :
    checkResidualTableAudit oracle context ops data maxDegree certificate = true <->
      let coins := derive oracle context certificate
      SumCheck.Finite.Accepted ops.toOps maxDegree
        (SumCheckInitial.verifierInitial ops data coins.gamma)
        coins.roundPoint.coordinates
        (SumCheckTruthPath.verifierTerminal ops data coins.alpha coins.gamma
          coins.roundPoint.coordinates)
        certificate.toFinite := by
  unfold checkResidualTableAudit SumCheckInitial.checkJoint
  exact SumCheck.Finite.check_eq_true_iff_accepted ops.toOps maxDegree
    (SumCheckInitial.verifierInitial ops data
      (derive oracle context certificate).gamma)
    (derive oracle context certificate).roundPoint.coordinates
    (SumCheckTruthPath.verifierTerminal ops data
      (derive oracle context certificate).alpha
      (derive oracle context certificate).gamma
      (derive oracle context certificate).roundPoint.coordinates)
    certificate.toFinite

theorem checkResidualTableAudit_complete_of_accepted
    {Context : Type uContext}
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : Oracle Context Field State shape)
    (context : Context)
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (maxDegree : Nat)
    (certificate : Certificate Field shape)
    (accepted :
      let coins := derive oracle context certificate
      SumCheck.Finite.Accepted ops.toOps maxDegree
        (SumCheckInitial.verifierInitial ops data coins.gamma)
        coins.roundPoint.coordinates
        (SumCheckTruthPath.verifierTerminal ops data coins.alpha coins.gamma
          coins.roundPoint.coordinates)
        certificate.toFinite) :
    checkResidualTableAudit oracle context ops data maxDegree certificate = true :=
  (checkResidualTableAudit_eq_true_iff_accepted oracle context ops data maxDegree
    certificate).2
    accepted

theorem checkResidualTableAudit_implies_semanticTruth_or_badEvent
    {Context : Type uContext}
    {State : Type uState}
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {shape : Shape}
    {columns : Nat}
    (oracle : Oracle Context Extension State shape)
    (context : Context)
    (baseOps : InterpolationOps F)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> Extension)
    (liftLaw : ConcreteJointData.ZeroReflectingLift
      baseOps extensionOps lift)
    (data : ConcreteJointData.IndependentInputs Extension shape columns)
    (maxDegree challengeSetSize : Nat)
    (certificate : Certificate Extension shape)
    (checked : checkResidualTableAudit oracle context extensionOps
      (data.toJointData baseOps lift) maxDegree certificate = true) :
    let coins := derive oracle context certificate
    data.SemanticTruth baseOps extensionOps lift \/
      SignedCoefficientObject.MixingRoot extensionOps
        (data.toJointData baseOps lift) coins.alpha coins.gamma \/
      exists round,
        SumCheck.BadChallenge
          (SumCheckInitial.symbolicInstance extensionOps
            (data.toJointData baseOps lift) coins.alpha coins.gamma maxDegree
            challengeSetSize coins.roundPoint.coordinates
            (SumCheckTruthPath.verifierTerminal extensionOps
              (data.toJointData baseOps lift) coins.alpha coins.gamma
              coins.roundPoint.coordinates)
            certificate.toFinite
            (SumCheckInitial.canonicalExpected extensionOps
              (data.toJointData baseOps lift) coins.alpha coins.gamma
              coins.roundPoint.coordinates))
          round := by
  exact ConcreteJointData.checkJoint_implies_semanticTruth_or_badEvent
    baseOps baseZero noZeroDivisors extensionOps extensionLaws
    extensionZeroLaws lift liftLaw data
    (derive oracle context certificate).alpha
    (derive oracle context certificate).gamma maxDegree challengeSetSize
    (derive oracle context certificate).roundPoint certificate.toFinite checked

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiatShamir
