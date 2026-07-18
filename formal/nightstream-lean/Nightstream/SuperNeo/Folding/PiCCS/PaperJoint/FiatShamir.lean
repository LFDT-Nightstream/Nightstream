import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData

/-!
Verifier-owned Fiat--Shamir authority for paper-level joint `Pi_CCS`.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
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
semantic degree theorem, SplitNc, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: `Certificate` has no alpha, gamma, round-challenge,
terminal, degree, or verifier-parameter field. This generic machine initializes
from an abstract `Context`; it does not prove that the context contains the
statement. `ProtocolVerifier.Statement` is the narrower protocol surface that
passes the prior state and complete public polynomial input together. The
machine squeezes alpha and gamma before any prover message, then absorbs each
finite round message before squeezing that round's challenge. `Oracle` may
still ignore or collide on any transition. The protocol wrapper names
whole-replay challenge and final-state collisions; concrete Poseidon2
refinement and security are still required. The checker at the bottom deliberately evaluates the
residual-table MLE and is named `checkResidualTableAudit`; it is not the
protocol verifier.

| Protocol | Phase | Transcript action | Mathematical owner | Lean owner |
|---|---|---|---|---|
| generic transcript | initialization | caller-supplied abstract context | verifier configuration | `Oracle.initialState`; completeness not asserted here |
| `Pi_CCS` | pre-SumCheck | squeeze one alpha coordinate per cube variable | verifier | `ChallengeLabel.alpha` |
| `Pi_CCS` | pre-SumCheck | squeeze gamma after all alpha coordinates | verifier | `ChallengeLabel.gamma` |
| `Pi_CCS` | round message | absorb canonical coefficient message | prover message, verifier transcript | `Oracle.absorbRound` |
| `Pi_CCS` | round challenge | squeeze after the corresponding message | verifier | `ChallengeLabel.sumcheck` |
| `Pi_CCS` | certificate | exactly one message per cube variable | prover | `Certificate.rounds` |
| audit only | residual-table checker | derived coins feed the algebraic audit path | verifier | `checkResidualTableAudit_eq_true_iff_accepted` |
| audit only | deterministic reduction | acceptance yields semantic truth or named mixing/SumCheck event for that audit path | paper semantics | `checkResidualTableAudit_implies_semanticTruth_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uContext uField uState uExtension

/-- Domain-separated semantic labels for every verifier coin in one joint
`Pi_CCS` execution. Concrete numeric tags are a later refinement. -/
inductive ChallengeLabel (shape : Shape) where
  | alpha (coordinate : Fin shape.cubeVariables)
  | gamma
  | sumcheck (round : Fin shape.cubeVariables)
deriving Repr, DecidableEq

/-- Abstract deterministic transcript state machine.

The interface exposes the paper schedule without choosing a byte encoding or
hash permutation. The state is always verifier-owned. -/
structure Oracle
    (Context : Type uContext)
    (Field : Type uField)
    (State : Type uState)
    (shape : Shape) where
  initialState : Context -> State
  absorbRound : State -> Fin shape.cubeVariables ->
    SumCheck.Finite.Message Field -> State
  squeeze : State -> ChallengeLabel shape -> Field × State

/-- The prover-visible certificate contains exactly one finite polynomial
message per paper SumCheck variable and no challenge carriers. -/
structure Certificate (Field : Type uField) (shape : Shape) where
  rounds : Fin shape.cubeVariables -> SumCheck.Finite.Message Field

namespace Certificate

/-- Projection into the generic finite SumCheck checker. Canonical finite
index order is fixed by the semantic model. -/
def toFinite
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    SumCheck.Finite.Certificate Field where
  rounds := List.ofFn certificate.rounds

/-- The projected certificate has exactly the paper's number of rounds. -/
theorem toFinite_rounds_length
    {Field : Type uField}
    {shape : Shape}
    (certificate : Certificate Field shape) :
    certificate.toFinite.rounds.length = shape.cubeVariables := by
  simp [toFinite]

end Certificate

private def alphaLabels (shape : Shape) : List (ChallengeLabel shape) :=
  (canonicalFinIndices shape.cubeVariables).map ChallengeLabel.alpha

private theorem alphaLabels_length (shape : Shape) :
    (alphaLabels shape).length = shape.cubeVariables := by
  simp [alphaLabels, canonicalFinIndices_length]

/-- Squeeze a fixed typed label list, threading state in program order. -/
private def squeezeMany
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

private theorem squeezeMany_values_length
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

/-- Verifier-owned values available immediately before the first SumCheck
message. Their order is exactly alpha coordinates followed by gamma. -/
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

/-- Interleave each polynomial-message absorb with the corresponding
verifier challenge squeeze. -/
private def deriveRoundsFrom
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

private theorem deriveRoundsFrom_values_length
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

/-- All verifier-derived coins and the final transcript state. -/
structure DerivedCoins (Field : Type uField) (State : Type uState)
    (shape : Shape) where
  alpha : CubePoint Field shape.cubeVariables
  gamma : Field
  roundPoint : CubePoint Field shape.cubeVariables
  finalState : State

/-- Deterministically replay the complete abstract paper transcript. -/
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

/-- The executable verifier uses only transcript-derived coins and the raw
finite polynomial messages. -/
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

/-- Exact executable/logical correspondence under the derived transcript.
No certificate field can override the initial, challenge vector, terminal, or
degree bound supplied to the finite verifier. -/
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

/-- Completeness of transcript-bound checking from an honest finite claimed
chain under the coins the verifier actually derives. -/
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

/-- Conditional reduction for the residual-table audit path.

Every coin in the conclusion is derived from the public context and exact
finite message vector. Acceptance implies the independent CCS/norm/carried
semantics or one of the already named algebraic/SumCheck bad events. Turning
those events into a probability bound still requires concrete transcript and
field-degree assumptions. -/
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

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir
