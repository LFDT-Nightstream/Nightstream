import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiRLC/PaperCompleteness.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Deterministic completeness and public coins for the paper `Pi_RLC` reduction.

Source: SuperNeo Section 7.4 and Appendix D.5, Lemma 9.

Owns: perfect completeness for an arbitrary valid public `CE(b)` batch, the
verifier's public challenge vector, the honest combined witness, and a
successful response for the exact verifier-computed `PiRLC.combinedOutput`.
Canonical fresh inputs made directly from assignments are a separate
corollary, not the primary completeness statement.

Does not own: coordinate-wise rewinding, probability, relaxed-binding
uniqueness, Fiat--Shamir, a concrete commitment or field, Rust, R1CS, or costs.

Emits constraints: no.

The primary theorem assumes only the paper's source relation: each public
input is at stage `fresh` and has an honest opening.  It preserves those
public instances exactly.  The output relation is a conclusion:
`Algebra.norm_growth` is the abstract form of the paper's expansion-factor calculation
`(K + k) * T * (b - 1) < B` stored in `GlobalParams.rlc_bound`.
-/

namespace NightstreamFPrime.Spec.Folding.PiRLC.PaperCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar

/-- The exact deterministic operations and deployment parameters used by one
paper `Pi_RLC` execution. -/
structure Context
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar) where
  semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment
  params : GlobalParams
  arity : BatchArity params
  algebra : Algebra Structure Assignment PublicInput Point Evaluation
    Commitment Scalar semantics params
  evaluationCount : Structure -> Nat
  evaluationsSize : forall system assignment point,
    (semantics.evaluations system assignment point).size =
      evaluationCount system

/-- Canonical source statement made from one honest assignment at the shared
constraint system and evaluation point. -/
def honestInput
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (system : Structure)
    (point : Point)
    (assignment : Assignment) :
    CE.Instance Structure PublicInput Point Evaluation Commitment where
  constraintSystem := system
  commitment := context.semantics.commit assignment
  publicInput := context.semantics.projectPublicInput assignment
  point := point
  evaluations := context.semantics.evaluations system assignment point
  stage := .fresh

/-- The canonical source statement satisfies the exact `CE(b)` relation.  No
source-validity proposition is assumed. -/
theorem honestInput_holds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (system : Structure)
    (point : Point)
    (assignment : Assignment)
    (sourceNorm : context.semantics.normBounded context.params.b assignment)
    (pointValid : context.semantics.evaluationPointValid system point) :
    CE.Holds context.semantics context.params
      (honestInput context system point assignment) assignment := by
  exact ⟨⟨rfl, rfl, sourceNorm⟩, pointValid, rfl⟩

/-- Canonical shared-system/shared-point input batch. -/
def honestBatch
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (system : Structure)
    (point : Point)
    (assignments : Fin context.arity.total -> Assignment) :
    InputBatch Structure PublicInput Point Evaluation Commitment
      context.params context.arity where
  system := system
  point := point
  inputs := fun index => honestInput context system point (assignments index)
  sameSystem := fun _ => rfl
  samePoint := fun _ => rfl
  evaluationCount := context.evaluationCount system
  evaluationsSize := fun index =>
    context.evaluationsSize system (assignments index) point

/-- All verifier randomness for paper `Pi_RLC`.  Membership in the strong
sampling set is attached to the verifier-generated coins, not supplied as a
prover assertion. -/
structure PublicCoins
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar) where
  challenges : Fin context.arity.total -> Scalar
  valid : forall index, context.algebra.challengeValid (challenges index)

/-- The honest prover applies exactly the public challenge vector to the
source assignments. -/
def honestResponse
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (assignments : Fin context.arity.total -> Assignment)
    (coins : PublicCoins context) :
    Response Assignment Scalar context.params context.arity where
  challenges := coins.challenges
  assignment := combinedWitness context.algebra coins.challenges assignments

/-- The canonical honest response opens the verifier-computed `CE(B)` output.
The only semantic premises are the source `b`-norms and validity of the shared
evaluation point. -/
theorem honestResponse_success
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (system : Structure)
    (point : Point)
    (assignments : Fin context.arity.total -> Assignment)
    (coins : PublicCoins context)
    (sourceNorms : forall index,
      context.semantics.normBounded context.params.b (assignments index))
    (pointValid : context.semantics.evaluationPointValid system point) :
    (honestResponse context assignments coins).Success
      context.semantics context.params context.algebra
      (honestBatch context system point assignments) := by
  exact combinedOutput_holds context.semantics context.params context.algebra
    context.arity system point
    (honestBatch context system point assignments).inputs
    coins.challenges assignments
    (fun _ => rfl) (honestBatch context system point assignments).sameSystem
    (honestBatch context system point assignments).samePoint coins.valid
    (fun index => honestInput_holds context system point (assignments index)
      (sourceNorms index) pointValid)
    pointValid

/-- The honest response preserves an arbitrary public input batch and opens
its exact verifier-computed `CE(B)` output.  Shared-point validity is derived
from one source opening; production arity is nonempty by `BatchArity`. -/
theorem honestResponse_success_of_inputHolds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (batch : InputBatch Structure PublicInput Point Evaluation Commitment
      context.params context.arity)
    (assignments : Fin context.arity.total -> Assignment)
    (coins : PublicCoins context)
    (inputFresh : forall index, (batch.inputs index).stage = .fresh)
    (inputHolds : forall index,
      CE.Holds context.semantics context.params (batch.inputs index)
        (assignments index)) :
    (honestResponse context assignments coins).Success
      context.semantics context.params context.algebra batch := by
  let first : Fin context.arity.total :=
    ⟨0, context.arity.totalPositive⟩
  have pointValid :
      context.semantics.evaluationPointValid batch.system batch.point := by
    simpa only [batch.sameSystem first, batch.samePoint first] using
      (inputHolds first).2.1
  exact combinedOutput_holds context.semantics context.params context.algebra
    context.arity batch.system batch.point batch.inputs coins.challenges
    assignments inputFresh batch.sameSystem batch.samePoint coins.valid
    inputHolds pointValid

/-- Pointwise perfect completeness for the actual public source instances. -/
def PerfectComplete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar) : Prop :=
  forall
      (batch : InputBatch Structure PublicInput Point Evaluation Commitment
        context.params context.arity)
      assignments (coins : PublicCoins context),
    (forall index, (batch.inputs index).stage = .fresh) ->
    (forall index,
      CE.Holds context.semantics context.params (batch.inputs index)
        (assignments index)) ->
    exists response : Response Assignment Scalar context.params context.arity,
      response.challenges = coins.challenges /\
      response.assignment =
        combinedWitness context.algebra coins.challenges assignments /\
      response.Success context.semantics context.params context.algebra
        batch

/-- Every valid public `CE(b)` batch constructs a successful response.  Target
membership is not a premise. -/
theorem perfectComplete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar) :
    PerfectComplete context := by
  intro batch assignments coins inputFresh inputHolds
  exact ⟨honestResponse context assignments coins, rfl, rfl,
    honestResponse_success_of_inputHolds context batch assignments coins
      inputFresh inputHolds⟩

/-- Canonical-instance completeness is retained as a useful corollary. -/
def CanonicalPerfectComplete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar) : Prop :=
  forall system point assignments (coins : PublicCoins context),
    (forall index,
      context.semantics.normBounded context.params.b (assignments index)) ->
    context.semantics.evaluationPointValid system point ->
    (honestResponse context assignments coins).Success
      context.semantics context.params context.algebra
      (honestBatch context system point assignments)

/-- Canonical source instances satisfy the primary operational theorem. -/
theorem canonicalPerfectComplete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar) :
    CanonicalPerfectComplete context := by
  intro system point assignments coins sourceNorms pointValid
  exact honestResponse_success context system point assignments coins
    sourceNorms pointValid

/-- Public-coin ownership: the verifier's output is computed from exactly the
public challenge vector and contains no hidden verifier randomness. -/
def PublicCoin
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar) : Prop :=
  forall
      (batch : InputBatch Structure PublicInput Point Evaluation Commitment
        context.params context.arity)
      assignments (coins : PublicCoins context),
    (honestResponse context assignments coins).output context.algebra batch =
      combinedOutput context.algebra batch.system batch.point batch.inputs
        coins.challenges

/-- Paper `Pi_RLC` is public coin: its sole verifier message is the sampled
challenge vector used definitionally by `combinedOutput`. -/
theorem publicCoin
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (context : Context Structure Assignment PublicInput Point Evaluation
      Commitment Scalar) :
    PublicCoin context := by
  intro batch assignments coins
  rfl

end NightstreamFPrime.Spec.Folding.PiRLC.PaperCompleteness
