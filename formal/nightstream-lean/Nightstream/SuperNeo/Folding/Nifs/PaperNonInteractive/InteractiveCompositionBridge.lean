import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableContinuation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec

/-!
Paper-owned context bridge from the non-interactive NIFS to the operational
`Pi_DEC ∘ Pi_RLC ∘ Pi_CCS` composition.

Source: SuperNeo Sections 7.3--7.5 and Appendices D.3--D.6.

Owns: the exact interactive contexts selected by one NIFS key and public
statement; definitional equality with the NIFS `Pi_RLC` and `Pi_DEC`
contexts; and transport of a causally generated `Pi_CCS` prefix into the
coefficient-complete NIFS batch.

Does not own: a distribution on causal prefixes, an ideal-random-oracle
coupling, a proof that a non-interactive prover is causal, a `Pi_DEC` target
witness, event bounds, Poseidon2, Ajtai, Rust, R1CS, artifacts, minimality, or
costs.

Emits constraints: no.

The prefix-alignment premise is intentionally an equality receipt rather than
an oracle assumption.  A later random-oracle coupling must construct it from
independent prover and verifier seeds.  In particular, arbitrary correlated
prefix worlds cannot manufacture this receipt by definition.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uState
  uProverSeed uProverTape

namespace Key

/-- The interactive `Pi_CCS` context selected by the NIFS key and the exact
public statement.  The ambient relation is unchanged; classical decidability
only supplies the finite experiment's executable membership branch. -/
noncomputable def strongExecutionContext
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
    (fresh : Fresh Commitment PublicInput shape) :
    StrongExecution.Context Extension Commitment PublicInput shape
      columns blockCount where
  baseOps := key.baseOps
  baseLaws := key.baseLaws
  baseZero := key.baseZero
  noZeroDivisors := key.noZeroDivisors
  extensionOps := key.extensionOps
  extensionLaws := key.extensionLaws
  extensionZeroLaws := key.extensionZeroLaws
  lift := key.lift
  liftLaws := key.liftLaws
  openingMaps := key.openingMaps
  params := key.params
  freshBound := key.freshBound
  statement := key.statement running fresh
  ambientDecision := fun _probe _witness => Classical.propDecidable _
  constantLaw := key.constantLaw
  sumcheckWidth := degreeBound
  sumcheckDegreeBound_le :=
    key.statement_sumcheckDegreeBound_le running fresh
  challengeSetSize := key.challengeSetSize

/-- The NIFS key selects exactly the paper-owned degree width; the looser
generic causal context cannot introduce unchecked coefficients above it. -/
theorem strongExecutionContext_paperDegreeWidthExact
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
    (fresh : Fresh Commitment PublicInput shape) :
    PaperDegreeWidthExact (key.strongExecutionContext running fresh) := by
  exact key.statement_sumcheckDegreeBound_eq running fresh

/-- The adjacent interactive `Pi_CCS`/`Pi_RLC` context selected by exactly the
same key, arity partition, relation semantics, and combination algebra. -/
noncomputable def compatibleContext
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
    (fresh : Fresh Commitment PublicInput shape) :
    CompatibleContext Extension Commitment PublicInput Scalar shape
      columns blockCount where
  piCcs := key.strongExecutionContext running fresh
  arity := key.arity
  freshCount_eq := key.freshCount_eq
  runningCount_eq := key.runningCount_eq
  piRlcSemantics := key.piRlcSemantics
  ambientAgreement := key.ambientAgreement
  piRlcEvaluationsSize := key.piRlcEvaluationsSize
  piRlcAlgebra := key.piRlcAlgebra

/-- The final interactive `Pi_DEC` context shares the same semantics and
parameters definitionally; no relation equality is supplied by a caller. -/
noncomputable def compatiblePiDecContext
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
    (fresh : Fresh Commitment PublicInput shape) :
    PiRlcComposition.PiDec.CompatiblePiDecContext
      (key.compatibleContext running fresh) where
  algebra := key.piDecAlgebra
  publicSplit := key.piDecPublicInputSplit
  evaluationArity := key.piDecEvaluationArity
  kPositive := key.kPositive

/-- The interactive adapter reaches literally the NIFS `Pi_RLC` context. -/
@[simp] theorem compatibleContext_piRlc
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
    (fresh : Fresh Commitment PublicInput shape) :
    (key.compatibleContext running fresh).piRlc =
      key.nifsPiRlcContext := by
  rfl

/-- The interactive adapter reaches literally the NIFS `Pi_DEC` context. -/
@[simp] theorem compatiblePiDecContext_paper
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
    (fresh : Fresh Commitment PublicInput shape) :
    (key.compatiblePiDecContext running fresh).paper =
      key.nifsPiDecContext := by
  rfl

end Key

/-- Exact receipt connecting one prefix produced by the causal interactive
strategy interface to one verifier-replayed NIFS proof.  Equality includes
all public coins, every SumCheck message, and the complete `Pi_CCS` output. -/
structure CausalPrefixAlignment
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
    (proof : Proof Extension Commitment shape degreeBound)
    (causalRun : PrefixExecution Extension shape) : Prop where
  probe_eq : causalRun.probe = key.piCcsProbe running fresh proof

private theorem piCcsProbe_rounds_eq_fixedEncode
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
    (proof : Proof Extension Commitment shape degreeBound) :
    (key.piCcsProbe running fresh proof).response.rounds =
      SumCheck.Finite.FixedPhase.RawCertificate.encode
        (key.piCcsFixedCertificate running fresh proof) := by
  apply congrArg SumCheck.Finite.Certificate.mk
  simp [Key.piCcsCertificate, Key.piCcsFixedCertificate,
    ProtocolVerifier.Certificate.toTranscript]
  congr

/-- Exact replay alignment now implies literal equality of the interactive
paper PiCCS gate and the frozen NIFS PiCCS gate.  No canonicalization or
encoding-failure branch is involved: both consume the submitted fixed-width
coefficient lists coefficient-for-coefficient. -/
theorem acceptedCheck_eq_piCcsCheck
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {proof : Proof Extension Commitment shape degreeBound}
    {causalRun : PrefixExecution Extension shape}
    (alignment : CausalPrefixAlignment key running fresh proof causalRun) :
    acceptedCheck (key.compatibleContext running fresh).piCcs causalRun =
      piCcsCheck key running fresh proof := by
  unfold acceptedCheck
  rw [alignment.probe_eq]
  unfold ProtocolPolynomial.FixedWidth.check
  rw [piCcsProbe_rounds_eq_fixedEncode key running fresh proof]
  simp only [Key.compatibleContext, Key.strongExecutionContext]
  rw [SumCheck.Finite.FixedPhase.RawCertificate.check_encode]
  rfl

/-- Exact replay alignment identifies the interactive alpha/gamma root with
the frozen NIFS alpha/gamma root for the same source witness. -/
theorem mixingFailure_iff_piCcsMixingRoot
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {proof : Proof Extension Commitment shape degreeBound}
    {causalRun : PrefixExecution Extension shape}
    (alignment : CausalPrefixAlignment key running fresh proof causalRun)
    (witness : OutputWitness shape columns) :
    MixingFailure (key.compatibleContext running fresh).piCcs causalRun
        witness <->
      PiCcsMixingRoot key running fresh proof witness := by
  unfold MixingFailure PiCcsMixingRoot
  rw [alignment.probe_eq]
  rfl

/-- Exact replay alignment identifies the decoded interactive fixed-width
SumCheck collision with the frozen NIFS collision.  The decoder receipt
forces the collision certificate to be the submitted NIFS certificate. -/
theorem sumCheckFailure_iff_piCcsSumCheckCollision
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {proof : Proof Extension Commitment shape degreeBound}
    {causalRun : PrefixExecution Extension shape}
    (alignment : CausalPrefixAlignment key running fresh proof causalRun)
    (witness : OutputWitness shape columns) :
    SumCheckFailure (key.compatibleContext running fresh).piCcs causalRun
        witness <->
      PiCcsSumCheckCollision key running fresh proof witness := by
  unfold SumCheckFailure FixedWidthSumCheckFailure PiCcsSumCheckCollision
  rw [alignment.probe_eq]
  simp only [Key.compatibleContext, Key.strongExecutionContext]
  rw [piCcsProbe_rounds_eq_fixedEncode key running fresh proof]
  simp only [SumCheck.Finite.FixedPhase.RawCertificate.decode_encode,
    Option.some.injEq]
  have inputEqual :=
    (key.statement running fresh).sourceProtocolData_toVerifierInput
      key.lift witness
  constructor
  · rintro ⟨certificate, certificate_eq, collision⟩
    subst certificate
    unfold ProtocolPolynomial.FixedWidth.SumCheckCollision at collision
    rw [inputEqual] at collision
    simpa [Key.piCcsProbe] using collision
  · intro collision
    refine ⟨key.piCcsFixedCertificate running fresh proof, rfl, ?_⟩
    unfold ProtocolPolynomial.FixedWidth.SumCheckCollision
    rw [inputEqual]
    simpa [Key.piCcsProbe] using collision

/-- A replay-aligned NIFS PiCCS acceptance and one independently supplied
ambient opening reduce to source validity or exactly the two NIFS D.4
events.  The source witness is an explicit argument fixed outside this
single-run theorem; no adaptive-witness probability claim is made. -/
theorem piCcsCheck_extracts_sourceValid_or_badEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {proof : Proof Extension Commitment shape degreeBound}
    {causalRun : PrefixExecution Extension shape}
    (alignment : CausalPrefixAlignment key running fresh proof causalRun)
    (witness : OutputWitness shape columns)
    (ambient : AmbientTargetOpenings key running fresh proof witness)
    (accepted : piCcsCheck key running fresh proof = true) :
    SourceValid key running fresh witness \/
      PiCcsMixingRoot key running fresh proof witness \/
      PiCcsSumCheckCollision key running fresh proof witness := by
  let context := (key.compatibleContext running fresh).piCcs
  have interactiveAmbient :
      AmbientOutputHolds context.extensionOps context.lift
        context.openingMaps context.params context.statement
        causalRun.probe witness := by
    rw [alignment.probe_eq]
    exact ambient
  have interactiveAccepted :
      causalRun.probe.FixedWidthAccepted context.extensionOps context.lift
        context.statement context.sumcheckWidth := by
    apply (acceptedCheck_eq_true_iff context causalRun).1
    rw [acceptedCheck_eq_piCcsCheck alignment]
    exact accepted
  rcases acceptedPrefix_extracts_fixedWitness_or_badEvent context causalRun
      witness interactiveAmbient interactiveAccepted with
    source | mixing | sumCheck
  · exact Or.inl source
  · exact Or.inr (Or.inl
      ((mixingFailure_iff_piCcsMixingRoot alignment witness).1 mixing))
  · exact Or.inr (Or.inr
      ((sumCheckFailure_iff_piCcsSumCheckCollision alignment witness).1
        sumCheck))

private theorem inputBatch_ext
    {Structure : Type _}
    {PublicInput : Type _}
    {Point : Type _}
    {Evaluation : Type _}
    {Commitment : Type _}
    {params : GlobalParams}
    {arity : BatchArity params}
    (left right :
      InputBatch Structure PublicInput Point Evaluation Commitment
        params arity)
    (system_eq : left.system = right.system)
    (point_eq : left.point = right.point)
    (inputs_eq : left.inputs = right.inputs)
    (evaluationCount_eq : left.evaluationCount = right.evaluationCount) :
    left = right := by
  cases left with
  | mk leftSystem leftPoint leftInputs leftSameSystem leftSamePoint
      leftEvaluationCount leftEvaluationsSize =>
      cases right with
      | mk rightSystem rightPoint rightInputs rightSameSystem rightSamePoint
          rightEvaluationCount rightEvaluationsSize =>
          cases system_eq
          cases point_eq
          cases inputs_eq
          cases evaluationCount_eq
          rfl

/-- A causally generated and exactly replay-aligned prefix induces the same
coefficient-complete `Pi_RLC` batch as the NIFS verifier. -/
theorem batchOfPrefix_eq_nifsPiRlcBatch
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {proof : Proof Extension Commitment shape degreeBound}
    {causalRun : PrefixExecution Extension shape}
    (alignment : CausalPrefixAlignment key running fresh proof causalRun) :
    (key.compatibleContext running fresh).batchOfPrefix causalRun =
      key.nifsPiRlcBatch running fresh proof := by
  apply inputBatch_ext
  · rfl
  · exact congrArg (fun probe => probe.coins.roundPoint) alignment.probe_eq
  · funext source
    exact congrArg
      (fun probe =>
        (key.statement running fresh).publicOutput probe
          (Fin.cast key.total_eq_sourceCount source))
      alignment.probe_eq
  · rfl

/-- Under the same receipt, the interactive `Pi_RLC` parent computation is
the NIFS parent for the verifier-derived challenge vector. -/
theorem combinedParent_eq_nifsParent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {proof : Proof Extension Commitment shape degreeBound}
    {causalRun : PrefixExecution Extension shape}
    (alignment : CausalPrefixAlignment key running fresh proof causalRun) :
    PiRlcComposition.PiDec.combinedParent
        (key.compatibleContext running fresh) causalRun
        (key.piRlcChallenges running fresh proof) =
      key.parent running fresh proof := by
  rw [PiRlcComposition.PiDec.combinedParent,
    batchOfPrefix_eq_nifsPiRlcBatch alignment]
  rfl

namespace RewindableProver

/-- Present one owned NIFS continuation reply through the exact second-stage
interface of the interactive composition.  The verifier-computed parent is
absent from both message carriers. -/
noncomputable def toInteractivePiDecReply
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    PiRlcComposition.PiDec.Reply
      (key.compatibleContext running fresh) where
  messages := fun child => {
    commitment :=
      (prover.reply challenges).piDecCommitments
        (Fin.cast key.outputCount_eq child)
    evaluations := #[
      (prover.reply challenges).piDecEvaluations
        (Fin.cast key.outputCount_eq child)]
  }
  childAssignments := (prover.reply challenges).childAssignments

/-- Exact `Pi_DEC` execution owned by one NIFS continuation at an explicit
post-prefix challenge vector.  The parent is verifier-computed from the
coefficient-complete NIFS batch; only the child messages and assignments come
from the continuation. -/
noncomputable def continuationPiDecExecution
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    PiDEC.PaperReduction.Execution key.nifsPiDecContext where
  attempt := {
    parent :=
      PiRLC.combinedOutput key.piRlcAlgebra
        (key.nifsPiRlcBatch running fresh
          (prover.baseProof running fresh)).system
        (key.nifsPiRlcBatch running fresh
          (prover.baseProof running fresh)).point
        (key.nifsPiRlcBatch running fresh
          (prover.baseProof running fresh)).inputs
        challenges
    messages :=
      (prover.toInteractivePiDecReply running fresh challenges).messages
  }
  childAssignments :=
    (prover.toInteractivePiDecReply running fresh challenges).childAssignments

@[simp] theorem toInteractivePiDecReply_childAssignments
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    (prover.toInteractivePiDecReply running fresh challenges).childAssignments =
      (prover.reply challenges).childAssignments := by
  rfl

/-- Exact replay and reply alignment identify the complete interactive
`Pi_DEC` execution with the continuation-owned NIFS execution.  This theorem
does not assert that either execution satisfies the target relation. -/
theorem interactivePiDecExecution_eq_continuation
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (adversary : PiRlcComposition.PiDec.Adversary
      (key.compatibleContext running fresh) ProverSeed ProverTape)
    (causalRun : PrefixExecution Extension shape)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext)
    (alignment : CausalPrefixAlignment key running fresh
      (prover.baseProof running fresh) causalRun)
    (replyAligned :
      prover.toInteractivePiDecReply running fresh challenges =
        adversary.reply causalRun challenges) :
    PiRlcComposition.PiDec.piDecExecution
        (key.compatibleContext running fresh)
        (key.compatiblePiDecContext running fresh)
        adversary causalRun challenges =
      prover.continuationPiDecExecution running fresh challenges := by
  unfold PiRlcComposition.PiDec.piDecExecution
    continuationPiDecExecution
  rw [← replyAligned, PiRlcComposition.PiDec.combinedParent,
    batchOfPrefix_eq_nifsPiRlcBatch alignment]
  rfl

/-- At the transcript-derived base vector, the continuation execution's
public attempt is exactly the attempt checked by the one-message NIFS
verifier. -/
@[simp] theorem continuationPiDecExecution_baseChallenges_attempt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    (prover.continuationPiDecExecution running fresh
      (prover.baseChallenges running fresh)).attempt =
        key.piDecAttempt running fresh
          (prover.baseProof running fresh) := by
  rfl

end RewindableProver

namespace RewindableForkOutcome

/-- The existing fork carrier and the continuation-owned carrier induce the
same complete `Pi_DEC` execution at every challenge vector. -/
@[simp] theorem piDecExecutionAt_eq_continuation
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    outcome.piDecExecutionAt challenges =
      outcome.prover.continuationPiDecExecution
        outcome.running outcome.fresh challenges := by
  rfl

/-- The D.6 success premise at the base vector is exactly NIFS `Pi_DEC`
acceptance together with valid target child witnesses.  In particular, the
second conjunct is not derived from public acceptance. -/
theorem continuationSuccessAt_baseChallenges_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key) :
    outcome.ContinuationSuccessAt
        (outcome.prover.baseChallenges outcome.running outcome.fresh) <->
      PiDEC.PaperVerifier.Accepted key.piDecAlgebra
          key.piDecEvaluationArity
          (key.piDecAttempt outcome.running outcome.fresh
            (outcome.prover.baseProof outcome.running outcome.fresh)) /\
        forall child,
          CE.Holds key.semantics key.params
            (PiDEC.PaperVerifier.children key.piDecPublicInputSplit
              (key.piDecAttempt outcome.running outcome.fresh
                (outcome.prover.baseProof outcome.running outcome.fresh))
              child)
            ((outcome.prover.reply
              (outcome.prover.baseChallenges
                outcome.running outcome.fresh)).childAssignments child) := by
  rfl

end RewindableForkOutcome

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
