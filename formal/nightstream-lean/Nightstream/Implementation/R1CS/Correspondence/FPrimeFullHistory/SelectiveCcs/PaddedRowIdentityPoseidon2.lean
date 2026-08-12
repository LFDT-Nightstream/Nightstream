import Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir
import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteAlgebra
import Nightstream.SuperNeo.Folding.Nifs.PaperProfile
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Types
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-!
Contract: exact Poseidon2 transcript and bounded `Pi_RLC` sampler for the
selected `PaddedRowIdentity` protocol.

Owns:
- the corrected HyperNova Construction 3 domain labels and fixed event schedule;
- the versioned public-NIFS-input field order;
- the existing one-joint `Pi_CCS` tags and field order;
- the exact post-SumCheck output absorption;
- the selected width-8 Poseidon2 constants;
- the 15-by-54 indexed full-field `Pi_RLC` sampler;
- the exact three-attempt rejection rule and balanced mod-5 decoder; and
- a total internal response plus the fail-closed sampler-success predicate.

Does not own: collision or random-oracle probability bounds, the external
Phi81 low-norm invertibility theorem, Ajtai/Module-SIS security, Rust, R1CS
rows, or byte encoding.

Assurance tier: model-level. The one-joint round and challenge schedule reuses
the existing canonical Poseidon2 semantics. `SamplerShortfall` is an explicit
proof-rejection condition. The total response exists only to satisfy the
generic paper-key carrier; the selected verifier must check
`samplerSucceeded` before it can use that key.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KPiCcsPaperFiatShamir
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra

abbrev State := Poseidon2Duplex.State
/-- Four Goldilocks output lanes from one Poseidon2 digest. -/
abbrev StatementId := Fin 4 -> F
abbrev PaperShape := Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape
abbrev SelectedCommitment :=
  PaddedRowIdentityConcreteAlgebra.Commitment
abbrev SelectedPublicInput :=
  PaddedRowIdentityConcreteAlgebra.PublicInput
abbrev SelectedEvaluation :=
  PaddedRowIdentityConcreteAlgebra.Evaluation

/-- Selected, Lean-owned width-8 Poseidon2 constants. -/
def constants : Poseidon2Schedule.Constants :=
  Poseidon2CanonicalConstants.selected

/-- Empty state before the statement identifier starts one NIFS transcript. -/
def initialState : State := Poseidon2Duplex.empty

/-! ## Canonical field serialization -/

/-- Numeric words are reduced exactly as the existing one-joint schedule. -/
def word (value : Nat) : Nat := value % goldilocksModulus

/-- Canonical UTF-8 bytes for
`HyperNova/MultiFold/Fiat-Shamir/v2`. -/
def construction3DomainBytes : List Nat :=
  [72, 121, 112, 101, 114, 78, 111, 118, 97, 47, 77, 117, 108, 116,
    105, 70, 111, 108, 100, 47, 70, 105, 97, 116, 45, 83, 104, 97,
    109, 105, 114, 47, 118, 50]

/-- Canonical UTF-8 bytes for Construction 3's `statement-id` label. -/
def statementIdLabelBytes : List Nat :=
  [115, 116, 97, 116, 101, 109, 101, 110, 116, 45, 105, 100]

/-- Canonical UTF-8 bytes for Construction 3's `proof` label. -/
def proofLabelBytes : List Nat := [112, 114, 111, 111, 102]

/-- Canonical UTF-8 bytes for Construction 3's `prover-message` label. -/
def proverMessageLabelBytes : List Nat :=
  [112, 114, 111, 118, 101, 114, 45, 109, 101, 115, 115, 97, 103, 101]

/-- Canonical UTF-8 bytes for Construction 3's `verifier-challenge` label. -/
def verifierChallengeLabelBytes : List Nat :=
  [118, 101, 114, 105, 102, 105, 101, 114, 45, 99, 104, 97, 108,
    108, 101, 110, 103, 101]

/-- Type-and-length frame for one Construction 3 string. -/
def stringFields (bytes : List Nat) : List Nat :=
  [word 32, word bytes.length] ++ bytes.map word

def construction3DomainFields : List Nat :=
  stringFields construction3DomainBytes

def statementIdLabelFields : List Nat :=
  stringFields statementIdLabelBytes

def proofLabelFields : List Nat := stringFields proofLabelBytes

def proverMessageLabelFields : List Nat :=
  stringFields proverMessageLabelBytes

def verifierChallengeLabelFields : List Nat :=
  stringFields verifierChallengeLabelBytes

/-- Construction 3 domain tag for the fixed-length statement identifier. -/
def statementIdentifierTag : Nat := 39

/-- Exact selected event descriptor. Indices are one-based, as in
Construction 3. `fieldCount` fixes the declared message or challenge space. -/
inductive Event where
  | proverMessage
      (eventIndex messageIndex messageType fieldCount : Nat)
  | verifierCoins
      (eventIndex challengeIndex challengeType fieldCount : Nat)
deriving Repr, DecidableEq

def Event.fields : Event -> List Nat
  | .proverMessage eventIndex messageIndex messageType fieldCount =>
      [word 34, word eventIndex, word messageIndex, word messageType,
        word fieldCount]
  | .verifierCoins eventIndex challengeIndex challengeType fieldCount =>
      [word 35, word eventIndex, word challengeIndex, word challengeType,
        word fieldCount]

/-- Production transcript tag for one indexed PiRLC candidate. -/
def piRlcCandidateTag : Nat := 1314062624

/-- The final PiDEC prover message has its own fixed type tag. -/
def piDecOutputTag : Nat := 48

/-- The exact 53-event public schedule for one selected fold:
alpha, gamma, 24 interleaved SumCheck rounds, the complete PiCCS output,
the PiRLC challenge block, and the PiDEC child message. -/
def eventSchedule : List Event :=
  [.verifierCoins 1 1 42 48,
    .verifierCoins 2 2 43 2] ++
  ((List.range rowVariables).flatMap fun round =>
    [.proverMessage (3 + 2 * round) (1 + round) 45 20,
      .verifierCoins (4 + 2 * round) (3 + round) 46 2]) ++
  [.proverMessage 51 25 47 22680,
    .verifierCoins 52 27 piRlcCandidateTag 810,
    .proverMessage 53 26 piDecOutputTag 34776]

@[simp] theorem eventSchedule_length : eventSchedule.length = 53 := by
  decide

/-- Canonical schedule encoding. The two event-kind labels are bound once,
then all fixed descriptors are encoded in event order. -/
def eventScheduleFields : List Nat :=
  proverMessageLabelFields ++ verifierChallengeLabelFields ++
    [word 33, word eventSchedule.length] ++ eventSchedule.flatMap Event.fields

@[simp] theorem eventScheduleFields_length : eventScheduleFields.length = 303 := by
  decide

/-- Exact Construction 3 prefix of the statement-identifier preimage. -/
def statementIdentifierPrefixFields : List Nat :=
  construction3DomainFields ++ statementIdLabelFields ++ eventScheduleFields

@[simp] theorem statementIdentifierPrefixFields_length :
    statementIdentifierPrefixFields.length = 353 := by
  decide

/-- Canonical type-and-length frame for the fixed four-field statement
identifier. -/
def statementIdFields (statementId : StatementId) : List Nat :=
  [word statementIdentifierTag, word 4] ++
    (canonicalFinIndices 4).map fun lane => (statementId lane).val

@[simp] theorem statementIdFields_length (statementId : StatementId) :
    (statementIdFields statementId).length = 6 := by
  simp [statementIdFields, canonicalFinIndices_length]

/-- Exact Construction 3 proof prefix before the public instances. -/
def proofPrefixFields (statementId : StatementId) : List Nat :=
  construction3DomainFields ++ proofLabelFields ++ eventScheduleFields ++
    [word 36, word 2, word 1, word 1] ++ statementIdFields statementId

@[simp] theorem proofPrefixFields_length (statementId : StatementId) :
    (proofPrefixFields statementId).length = 356 := by
  simp [proofPrefixFields, construction3DomainFields, proofLabelFields,
    stringFields, construction3DomainBytes, proofLabelBytes,
    canonicalFinIndices_length]

/-- The fixed-length identifier is absorbed with the Construction 3 domain,
`proof` label, event schedule, and `(mu, nu) = (1, 1)` before all public NIFS
inputs. -/
def statementIdentifierFields (statementId : StatementId) : List Nat :=
  proofPrefixFields statementId

/-- Transcript state after the verifier binds the full public statement. -/
def initialStateForStatement (statementId : StatementId) : State :=
  Poseidon2Duplex.absorbList constants
    (statementIdentifierFields statementId) initialState

/-- A base-field element has one canonical Goldilocks coordinate. -/
def fFields (value : F) : List Nat := [value.val]

/-- A quadratic-extension element is low limb followed by high limb. -/
def kFields (value : K) : List Nat := [value.c0.val, value.c1.val]

/-- Encode a finite function in increasing `Fin` order. -/
def finFields
    {count : Nat} {Value : Type}
    (encode : Value -> List Nat) (values : Fin count -> Value) : List Nat :=
  (canonicalFinIndices count).flatMap fun index => encode (values index)

/-- Ring coefficients use increasing polynomial degree. -/
def ringFFields (value : RingF) : List Nat :=
  finFields fFields value

/-- The selected paper shape, in the same order as `KPiCcsTranscript`. -/
def shapeFields (value : PaperShape) : List Nat :=
  [word value.cubeVariables, word value.freshCount,
    word value.runningCount, word value.matrixCount,
    word value.coefficientCount]

def monomialFields
    {valueShape : PaperShape}
    (monomial : CCSResidualTable.Monomial K valueShape.matrixCount) : List Nat :=
  kFields monomial.coefficient ++
    (canonicalFinIndices valueShape.matrixCount).map fun index =>
      word (monomial.exponents index)

def polynomialFields
    {valueShape : PaperShape}
    (polynomial :
      CCSResidualTable.ConstraintPolynomial K valueShape.matrixCount) : List Nat :=
  word polynomial.degreeBound :: word polynomial.terms.length ::
    polynomial.terms.flatMap monomialFields

def pointFields
    {variableCount : Nat} (point : CubePoint K variableCount) : List Nat :=
  point.coordinates.flatMap kFields

def commitmentFields (commitment : SelectedCommitment) : List Nat :=
  finFields ringFFields commitment

def publicInputFields (input : SelectedPublicInput) : List Nat :=
  finFields fFields input

def evaluationFields (evaluation : SelectedEvaluation) : List Nat :=
  finFields (fun coefficients => finFields kFields coefficients) evaluation

/-- Selected protocol identifier. It is distinct from every one-joint phase
tag and fixes the padded-row profile before public claims are absorbed. -/
def publicInputTag : Nat := 40

/-- Wire/profile version for the selected padded-row NIFS. -/
def protocolVersion : Nat := 1

/-- Static selected profile fields. They make the relation dimensions part of
the public transcript prefix instead of relying on an unnamed build profile. -/
def profileFields : List Nat :=
  shapeFields shape ++
    [word assignmentColumns,
      word (Phi81ColumnLayout.blockCount assignmentColumns),
      word verifierRows,
      word relationShape.publicWidth,
      word 9]

def runningFields
    (running : Running K SelectedCommitment SelectedPublicInput shape) : List Nat :=
  pointFields running.point ++
    finFields commitmentFields running.commitments ++
    finFields publicInputFields running.publicInputs ++
    finFields evaluationFields running.evaluations

def freshFields
    (fresh : Fresh SelectedCommitment SelectedPublicInput shape) : List Nat :=
  finFields commitmentFields fresh.commitments ++
    finFields publicInputFields fresh.publicInputs

/-- Complete public NIFS input, before any `Pi_CCS` challenge. -/
def publicNifsFields
    (running : Running K SelectedCommitment SelectedPublicInput shape)
    (fresh : Fresh SelectedCommitment SelectedPublicInput shape) : List Nat :=
  [word publicInputTag, word protocolVersion] ++ profileFields ++
    runningFields running ++ freshFields fresh

/-- Verifier-owned public-input absorption. -/
def absorbPublicInput
    (state : State)
    (running : Running K SelectedCommitment SelectedPublicInput shape)
    (fresh : Fresh SelectedCommitment SelectedPublicInput shape) : State :=
  Poseidon2Duplex.absorbList constants
    (publicNifsFields running fresh) state

/-! ## Exact one-joint PiCCS schedule -/

/-- Complete verifier statement. Tag `41` and all following fields are the
value-level form of the existing `KPiCcsTranscript.statementFields`. -/
def statementFields
    (statement : ProtocolVerifier.Statement K State shape) : List Nat :=
  [word 41] ++ shapeFields shape ++
    polynomialFields statement.input.constraintPolynomial ++
    [word shape.cubeVariables] ++ pointFields statement.input.priorPoint ++
    [word shape.carriedEvaluationCount] ++
    (canonicalCarriedCoordinates shape).flatMap fun coordinate =>
      kFields (statement.input.claimedCoefficient coordinate)

/-- Canonical Construction 3 frame for one prover message. -/
def proverMessageFields
    (eventIndex messageIndex messageType : Nat)
    (payload : List Nat) : List Nat :=
  proverMessageLabelFields ++
    [word eventIndex, word messageIndex, word messageType,
      word payload.length] ++ payload

/-- One SumCheck round. Its event index, global message index, type, payload
length, and constant-first coefficients are all explicit. -/
def roundFields
    (round : Fin shape.cubeVariables) (message : Message K) : List Nat :=
  proverMessageFields (3 + 2 * round.val) (1 + round.val) 45
    (message.coefficients.flatMap kFields)

/-- Scalar projection used only by the algebraic terminal checker. This is
not the authority-bearing NIFS handoff because it omits fresh nonconstant
ring coefficients. -/
def projectedOutputFields
    (message : ProtocolPolynomial.OutputMessage K shape) : List Nat :=
  proverMessageFields 51 25 47
    (finFields (fun matrices => finFields kFields matrices)
        message.freshMatrixImage ++
      finFields kFields message.sourceAssignment ++
      (canonicalCarriedCoordinates shape).flatMap fun coordinate =>
        kFields (message.carriedImage coordinate))

/-- Complete paper output in source-major, matrix-major, coefficient-major
order. Tag `47` is followed by both extension-field limbs of every `y'`
coordinate sent in Step 3 of Section 7.3. -/
def outputFields
    (message : FullOutputCoordinates.FullOutput K shape) : List Nat :=
  proverMessageFields 51 25 47
    (finFields
      (fun matrices => finFields
        (fun coefficients => finFields kFields coefficients) matrices)
      message.coordinate)

/-- Final prover message in Construction 3's fixed schedule. It contains the
14 ordered PiDEC child commitments and evaluation families. No later challenge
depends on this frame. -/
def piDecOutputFields
    (proof : Proof K SelectedCommitment shape 9) : List Nat :=
  proverMessageFields 53 26 piDecOutputTag
    (finFields commitmentFields proof.piDecCommitments ++
      finFields evaluationFields proof.piDecEvaluations)

/-- Receipt-only final transcript state. The selected verifier can omit this
post-challenge state when it is not returned or used for acceptance. -/
def finalTranscriptState
    (state : State)
    (proof : Proof K SelectedCommitment shape 9) : State :=
  Poseidon2Duplex.absorbList constants (piDecOutputFields proof) state

/-- Exact post-SumCheck handoff used before `Pi_RLC` challenge sampling. -/
def absorbFullOutput
    (state : State)
    (message : FullOutputCoordinates.FullOutput K shape) : State :=
  Poseidon2Duplex.absorbList constants (outputFields message) state

private theorem coefficientOutputFields_length
    (values : Fin shape.coefficientCount -> K) :
    (finFields kFields values).length = shape.coefficientCount * 2 := by
  unfold finFields
  calc
    _ = (canonicalFinIndices shape.coefficientCount).length * 2 := by
      apply Poseidon2Program.length_flatMap_uniform
      intro coefficient
      rfl
    _ = shape.coefficientCount * 2 := by
      rw [canonicalFinIndices_length]

private theorem matrixOutputFields_length
    (values : Fin shape.matrixCount -> Fin shape.coefficientCount -> K) :
    (finFields (fun coefficients => finFields kFields coefficients)
      values).length =
      shape.matrixCount * (shape.coefficientCount * 2) := by
  unfold finFields
  calc
    _ = (canonicalFinIndices shape.matrixCount).length *
        (shape.coefficientCount * 2) := by
      apply Poseidon2Program.length_flatMap_uniform
      intro matrix
      exact coefficientOutputFields_length (values matrix)
    _ = shape.matrixCount * (shape.coefficientCount * 2) := by
      rw [canonicalFinIndices_length]

private theorem sourceOutputFields_length
    (values : Fin shape.sourceCount -> Fin shape.matrixCount ->
      Fin shape.coefficientCount -> K) :
    (finFields
      (fun matrices => finFields
        (fun coefficients => finFields kFields coefficients) matrices)
      values).length =
      shape.sourceCount *
        (shape.matrixCount * (shape.coefficientCount * 2)) := by
  unfold finFields
  calc
    _ = (canonicalFinIndices shape.sourceCount).length *
        (shape.matrixCount * (shape.coefficientCount * 2)) := by
      apply Poseidon2Program.length_flatMap_uniform
      intro source
      exact matrixOutputFields_length (values source)
    _ = shape.sourceCount *
        (shape.matrixCount * (shape.coefficientCount * 2)) := by
      rw [canonicalFinIndices_length]

private theorem ringFFields_length (value : RingF) :
    (ringFFields value).length = ringDegree := by
  unfold ringFFields finFields
  calc
    _ = (canonicalFinIndices ringDegree).length * 1 := by
      apply Poseidon2Program.length_flatMap_uniform
      intro coefficient
      rfl
    _ = ringDegree := by
      rw [canonicalFinIndices_length]
      omega

private theorem commitmentFields_length (value : SelectedCommitment) :
    (commitmentFields value).length = verifierRows * ringDegree := by
  unfold commitmentFields finFields
  calc
    _ = (canonicalFinIndices verifierRows).length * ringDegree := by
      apply Poseidon2Program.length_flatMap_uniform
      intro row
      exact ringFFields_length (value row)
    _ = verifierRows * ringDegree := by
      rw [canonicalFinIndices_length]

private theorem piDecCommitmentFields_length
    (values : Fin shape.runningCount -> SelectedCommitment) :
    (finFields commitmentFields values).length =
      shape.runningCount * (verifierRows * ringDegree) := by
  unfold finFields
  calc
    _ = (canonicalFinIndices shape.runningCount).length *
        (verifierRows * ringDegree) := by
      apply Poseidon2Program.length_flatMap_uniform
      intro child
      exact commitmentFields_length (values child)
    _ = shape.runningCount * (verifierRows * ringDegree) := by
      rw [canonicalFinIndices_length]

private theorem piDecEvaluationFields_length
    (values : Fin shape.runningCount -> SelectedEvaluation) :
    (finFields evaluationFields values).length =
      shape.runningCount *
        (shape.matrixCount * (shape.coefficientCount * 2)) := by
  unfold finFields
  calc
    _ = (canonicalFinIndices shape.runningCount).length *
        (shape.matrixCount * (shape.coefficientCount * 2)) := by
      apply Poseidon2Program.length_flatMap_uniform
      intro child
      exact matrixOutputFields_length (values child)
    _ = shape.runningCount *
        (shape.matrixCount * (shape.coefficientCount * 2)) := by
      rw [canonicalFinIndices_length]

/-- The selected complete output contains the Construction 3 message frame and
all 15x14x54 quadratic-extension coordinates. -/
@[simp] theorem outputFields_length
    (message : FullOutputCoordinates.FullOutput K shape) :
    (outputFields message).length = 22700 := by
  unfold outputFields proverMessageFields
  simp only [List.length_append, List.length_cons, List.length_nil,
    Nat.reduceAdd]
  change 20 + _ = 22700
  rw [sourceOutputFields_length message.coordinate]
  rfl

/-- The final PiDEC message contains its Construction 3 frame, 14 ordered
commitments of 18x54 base-field values, and 14 ordered 14x54 extension-field
evaluation families. -/
@[simp] theorem piDecOutputFields_length
    (proof : Proof K SelectedCommitment shape 9) :
    (piDecOutputFields proof).length = 34796 := by
  unfold piDecOutputFields proverMessageFields
  simp only [List.length_append, List.length_cons, List.length_nil,
    Nat.reduceAdd]
  change 20 + (_ + _) = 34796
  rw [piDecCommitmentFields_length proof.piDecCommitments,
    piDecEvaluationFields_length proof.piDecEvaluations]
  rfl

/-- Interpret the first two freshly permuted lanes as the selected concrete
quadratic extension. -/
def challengeValue (state : State) : K where
  c0 := ⟨state.lanes ⟨0, by decide⟩ % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩
  c1 := ⟨state.lanes ⟨1, by decide⟩ % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

def squeezeK (state : State) : K × State :=
  let next := Poseidon2Duplex.gate constants state
  (challengeValue next, next)

/-- Construction 3 challenge frame. The domain, literal challenge label,
event index, challenge index, declared type, and domain-expansion coordinates
are absorbed before the concrete squeeze. -/
def verifierChallengeFields
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) : List Nat :=
  construction3DomainFields ++ verifierChallengeLabelFields ++
    [word eventIndex, word challengeIndex, word challengeType,
      word coordinates.length] ++ coordinates.map word

def squeezeVerifierChallenge
    (eventIndex challengeIndex challengeType : Nat)
    (coordinates : List Nat) (state : State) : K × State :=
  squeezeK (Poseidon2Duplex.absorbList constants
    (verifierChallengeFields eventIndex challengeIndex challengeType coordinates)
    state)

/-- Exact typed paper transcript, with no caller-provided challenges. -/
def transcript :
    FiatShamir.Oracle (ProtocolVerifier.Statement K State shape) K State shape where
  initialState statement :=
    Poseidon2Duplex.absorbList constants (statementFields statement)
      statement.priorState
  absorbRound state round message :=
    Poseidon2Duplex.absorbList constants (roundFields round message) state
  squeeze state label :=
    match label with
    | .alpha coordinate =>
        squeezeVerifierChallenge 1 1 42 [coordinate.val] state
    | .gamma => squeezeVerifierChallenge 2 2 43 [] state
    | .sumcheck round =>
        squeezeVerifierChallenge (4 + 2 * round.val) (3 + round.val) 46 [] state

/-- Algebraic `Pi_CCS` oracle. Its output operation is retained for the
generic projected terminal checker. The selected NIFS uses
`absorbFullOutput` for its authority-bearing handoff. -/
def oracle : ProtocolVerifier.Oracle K State shape where
  transcript := transcript
  absorbOutput state message :=
    Poseidon2Duplex.absorbList constants (projectedOutputFields message) state

@[simp] theorem transcript_absorbRound_eq_canonical
    (state : State) (round : Fin shape.cubeVariables) (message : Message K) :
    transcript.absorbRound state round message =
      Poseidon2Duplex.absorbList constants
        (proverMessageFields (3 + 2 * round.val) (1 + round.val) 45
          (message.coefficients.flatMap kFields)) state := by
  rfl

@[simp] theorem transcript_alpha_eq_canonical
    (state : State) (coordinate : Fin shape.cubeVariables) :
    transcript.squeeze state (.alpha coordinate) =
      squeezeVerifierChallenge 1 1 42 [coordinate.val] state := by
  rfl

@[simp] theorem transcript_sumcheck_eq_canonical
    (state : State) (round : Fin shape.cubeVariables) :
    transcript.squeeze state (.sumcheck round) =
      squeezeVerifierChallenge (4 + 2 * round.val) (3 + round.val) 46 [] state := by
  rfl

/-! ## Exact bounded full-field PiRLC sampling -/

abbrev Coefficient :=
  Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.Coefficient

abbrev Scalar :=
  Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.Scalar

/-- Every ring coefficient gets at most three full-field candidates. -/
def samplerAttemptCount : Nat := 3

/-- Canonical attempt indices. Named values keep the attempt identity stable
across the sampler and its security proof. -/
def firstAttempt : Fin samplerAttemptCount := ⟨0, by decide⟩

def secondAttempt : Fin samplerAttemptCount := ⟨1, by decide⟩

def thirdAttempt : Fin samplerAttemptCount := ⟨2, by decide⟩

/-- Exact number of coefficients in one Phi81 challenge. -/
def samplerCoefficientCount : Nat := 54

/-- Exact source-major, coefficient-major, attempt-minor candidate index. -/
def candidateFlat
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (attempt : Fin samplerAttemptCount) : Nat :=
  (source.val * samplerCoefficientCount + coefficient.val) *
      samplerAttemptCount + attempt.val

/-- Exact fixed-width domain frame for one PiRLC candidate fork.

The complete post-PiCCS state already binds the statement, profile, event
schedule, proof prefix, and PiCCS output. One unique candidate tag plus the
injective flat candidate index therefore separates every candidate without
reabsorbing the text-form Construction-3 labels on each of the 2,430 forks.
The fixed two-field arity and index order are verifier-key data. -/
def candidateFields
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (attempt : Fin samplerAttemptCount) : List Nat :=
  [word piRlcCandidateTag, word (candidateFlat source coefficient attempt)]

@[simp] theorem candidateFields_length
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (attempt : Fin samplerAttemptCount) :
    (candidateFields source coefficient attempt).length = 2 := by
  rfl

/-- One indexed full-field candidate derived from the fixed post-PiCCS state.
The candidate is a complete canonical Goldilocks element, not a 16-bit chunk. -/
def candidateValue
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount)
    (attempt : Fin samplerAttemptCount) : F :=
  let tagged := Poseidon2Duplex.absorbList constants
    (candidateFields source coefficient attempt) state
  let sampled := Poseidon2Duplex.challengeField constants tagged
  ⟨sampled.1 % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Reject only the final Goldilocks residue `q-1`. -/
def candidateAccepted (candidate : F) : Bool :=
  decide (candidate.val < goldilocksModulus - 1)

/-- Accepted residues map in order to the five centered digits. -/
def candidateDigit (candidate : F) : Coefficient :=
  ⟨candidate.val % 5, Nat.mod_lt _ (by decide)⟩

@[simp] theorem candidateAccepted_eq_true_iff (candidate : F) :
    candidateAccepted candidate = true ↔
      candidate.val < goldilocksModulus - 1 := by
  simp [candidateAccepted]

@[simp] theorem candidateAccepted_eq_false_iff (candidate : F) :
    candidateAccepted candidate = false ↔
      candidate.val = goldilocksModulus - 1 := by
  simp only [candidateAccepted, decide_eq_false_iff_not]
  have upper := candidate.isLt
  simp only [goldilocksModulus] at upper ⊢
  omega

/-- One coefficient uses the first accepted candidate and fails after exactly
three rejections. Unused later attempts have no authority. -/
def sampleCoefficient
    (state : State)
    (source : Fin PaperProfile.arity.total)
    (coefficient : Fin samplerCoefficientCount) : Option Coefficient :=
  let first := candidateValue state source coefficient firstAttempt
  if candidateAccepted first then
    some (candidateDigit first)
  else
    let second := candidateValue state source coefficient secondAttempt
    if candidateAccepted second then
      some (candidateDigit second)
    else
      let third := candidateValue state source coefficient thirdAttempt
      if candidateAccepted third then some (candidateDigit third) else none

/-- Exact proof-rejection event: at least one of the 15x54 coefficients
exhausts all three attempts. -/
def SamplerShortfall (state : State) : Prop :=
  Exists fun source : Fin PaperProfile.arity.total =>
    Exists fun coefficient : Fin samplerCoefficientCount =>
      sampleCoefficient state source coefficient = none

def SamplerAvailable (state : State) : Prop :=
  ¬ SamplerShortfall state

/-- Executable gate used by the selected verifier. -/
noncomputable def samplerSucceeded (state : State) : Bool := by
  classical
  exact decide (SamplerAvailable state)

@[simp] theorem samplerSucceeded_eq_true_iff (state : State) :
    samplerSucceeded state = true ↔ SamplerAvailable state := by
  classical
  simp [samplerSucceeded]

@[simp] theorem samplerSucceeded_eq_false_iff (state : State) :
    samplerSucceeded state = false ↔ SamplerShortfall state := by
  classical
  simp [samplerSucceeded, SamplerAvailable]

theorem available_or_shortfall (state : State) :
    SamplerAvailable state \/ SamplerShortfall state := by
  rcases Classical.em (SamplerShortfall state) with shortfall | available
  · exact Or.inr shortfall
  · exact Or.inl available

theorem available_excludes_shortfall
    {state : State} (available : SamplerAvailable state) :
    ¬ SamplerShortfall state :=
  available

theorem not_available_iff_shortfall (state : State) :
    ¬ SamplerAvailable state ↔ SamplerShortfall state := by
  simp only [SamplerAvailable, Classical.not_not]

/-- Centered zero is symbol `2`, since the semantic value is `symbol - 2`. -/
def zeroCoefficient : Coefficient := ⟨2, by decide⟩

def zeroScalar : Scalar := fun _ => zeroCoefficient

/-- The generic key needs a total scalar function. Failed coordinates use a
fixed internal zero only so the carrier is total. `samplerSucceeded` prevents
the selected verifier from accepting any execution that reaches this case. -/
def scalarResponse
    (state : State) (source : Fin PaperProfile.arity.total) : Scalar :=
  fun coefficient =>
    (sampleCoefficient state source
      (Fin.cast (by rfl) coefficient)).getD zeroCoefficient

/-- Ring-valued response consumed by the generic paper-key carrier. -/
def piRlcResponse
    (state : State) (source : Fin PaperProfile.arity.total) : RingF :=
  Phi81StrongSet.embedScalar (scalarResponse state source)

theorem piRlcResponse_valid (state : State)
    (source : Fin PaperProfile.arity.total) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.challengeValid
      (piRlcResponse state source) := by
  exact
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge.embedScalar_valid _

/-- Pointwise refinement to the exact successful three-attempt sampler. -/
def ResponseRefinesAt
    (response : State -> Fin PaperProfile.arity.total -> Scalar)
    (state : State) : Prop :=
  forall source coefficient,
    sampleCoefficient state source (Fin.cast (by rfl) coefficient) =
      some (response state source coefficient)

theorem piRlcResponse_refines_of_available
    {state : State} (available : SamplerAvailable state) :
    ResponseRefinesAt scalarResponse state := by
  intro source coefficient
  have succeeds :
      sampleCoefficient state source (Fin.cast (by rfl) coefficient) ≠ none := by
    intro failed
    exact available ⟨source, Fin.cast (by rfl) coefficient, failed⟩
  unfold scalarResponse
  cases sampled : sampleCoefficient state source (Fin.cast (by rfl) coefficient) with
  | none => exact False.elim (succeeds sampled)
  | some value => simp [sampled]

theorem piRlcResponse_refines_of_no_shortfall
    {state : State} (noShortfall : ¬ SamplerShortfall state) :
    ResponseRefinesAt scalarResponse state :=
  piRlcResponse_refines_of_available noShortfall

/-! ## Exact balance of the accepted field domain -/

def acceptedQuotientCount : Nat := 3689348813882916864

theorem acceptedDomain_factorization :
    goldilocksModulus - 1 = acceptedQuotientCount * 5 := by
  decide

abbrev AcceptedCandidate :=
  { candidate : F // candidate.val < goldilocksModulus - 1 }

def factorAccepted (candidate : AcceptedCandidate) :
    Fin acceptedQuotientCount × Coefficient :=
  let quotient := candidate.val.val / 5
  have quotientLt : quotient < acceptedQuotientCount := by
    have accepted := candidate.property
    simp only [goldilocksModulus, acceptedQuotientCount] at accepted ⊢
    omega
  ⟨⟨quotient, quotientLt⟩, candidateDigit candidate.val⟩

def combineAccepted
    (coordinates : Fin acceptedQuotientCount × Coefficient) :
    AcceptedCandidate :=
  let value := coordinates.1.val * 5 + coordinates.2.val
  have accepted : value < goldilocksModulus - 1 := by
    have quotientLt := coordinates.1.isLt
    have residueLt : coordinates.2.val < 5 := by
      simpa [Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.alphabetSize]
        using coordinates.2.isLt
    simp only [acceptedQuotientCount] at quotientLt
    change coordinates.1.val * 5 + coordinates.2.val <
      goldilocksModulus - 1
    simp only [goldilocksModulus]
    omega
  have canonical : value < goldilocksModulus := by
    omega
  ⟨⟨value, canonical⟩, accepted⟩

theorem combineAccepted_factorAccepted (candidate : AcceptedCandidate) :
    combineAccepted (factorAccepted candidate) = candidate := by
  apply Subtype.ext
  apply Fin.ext
  change candidate.val.val / 5 * 5 + candidate.val.val % 5 = candidate.val.val
  simpa [Nat.mul_comm] using Nat.div_add_mod candidate.val.val 5

theorem factorAccepted_combineAccepted
    (coordinates : Fin acceptedQuotientCount × Coefficient) :
    factorAccepted (combineAccepted coordinates) = coordinates := by
  rcases coordinates with ⟨quotient, residue⟩
  have residueLt : residue.val < 5 := by
    exact residue.isLt
  apply Prod.ext
  · apply Fin.ext
    change (quotient.val * 5 + residue.val) / 5 = quotient.val
    omega
  · apply Fin.ext
    change (quotient.val * 5 + residue.val) % 5 = residue.val
    omega

/-- The accepted candidate domain is exactly a product with `Fin 5`.
Therefore a uniform full-field candidate, conditioned on acceptance, gives an
exactly uniform centered digit. -/
theorem acceptedCandidate_exactly_balanced :
    (forall candidate, combineAccepted (factorAccepted candidate) = candidate) /\
      (forall coordinates, factorAccepted (combineAccepted coordinates) = coordinates) :=
  ⟨combineAccepted_factorAccepted, factorAccepted_combineAccepted⟩

/-- Concrete transcript-security event added by the bounded sampler. The four
paper transcript collision classes remain those in
`PaperNonInteractive.TranscriptSecurityEvent`. -/
inductive Poseidon2SecurityEvent (state : State) where
  | boundedSamplerShortfall (failure : SamplerShortfall state)

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2
