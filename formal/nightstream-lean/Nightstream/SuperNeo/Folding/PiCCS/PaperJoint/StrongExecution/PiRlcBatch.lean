import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
import Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness

/-!
The exact intermediate public batch shared by paper `Pi_CCS` and `Pi_RLC`.

Source: SuperNeo Theorem 6 and Sections 7.3--7.4.

Owns: fresh/running arity alignment, construction of the `Pi_RLC` input
batch from one verifier-constructed `Pi_CCS` public output, and literal
equality of the two commitment projections.

Does not own: either reduction's probability experiment, an extractor,
commitment security, Fiat--Shamir, Rust, R1CS, or costs.

Emits constraints: no.

The adapter contains no assignments or validity evidence.  In particular,
it cannot smuggle a target witness from the strong reduction into the weak
reduction.  It transports only the public instance produced by the verifier.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar

/-- Verifier-owned data shared by the two adjacent reductions.  The two
partition equalities, rather than total-count equality alone, prevent a
fresh source from being silently reinterpreted as a running source. -/
structure CompatibleContext
    (Extension : Type uExtension)
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (Scalar : Type uScalar)
    (shape : Shape)
    (columns blockCount : Nat) where
  piCcs : StrongExecution.Context Extension Commitment PublicInput shape
    columns blockCount
  arity : BatchArity piCcs.params
  freshCount_eq : arity.freshCount = shape.freshCount
  runningCount_eq : arity.mode.count piCcs.params = shape.runningCount
  /-- The weak reduction may use a canonicalized semantic view of the fixed
  paper matrix source.  It is kept separate from the strong relation so its
  algebra does not have to claim laws for malformed layouts or kernels. -/
  piRlcSemantics : RelationSemantics
    (MatrixSource F shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint Extension shape.cubeVariables)
    (EvaluationFamily Extension shape)
    Commitment
  /-- At the verifier-owned matrix source, the canonical weak relation is
  exactly the corrected ambient relation emitted by `Pi_CCS`. -/
  ambientAgreement : forall
      (statement : CE.Instance
        (MatrixSource F shape columns blockCount)
        PublicInput
        (CubePoint Extension shape.cubeVariables)
        (EvaluationFamily Extension shape)
        Commitment)
      (assignment : Assignment F columns),
    statement.constraintSystem = piCcs.statement.matrixSource ->
      (PiRLC.PaperCorrections.CorrectedAmbientHolds
          (paperRelationSemantics piCcs.baseOps piCcs.extensionOps piCcs.lift
            piCcs.openingMaps)
          piCcs.params statement assignment <->
        PiRLC.PaperCorrections.CorrectedAmbientHolds
          piRlcSemantics piCcs.params statement assignment)
  piRlcEvaluationsSize : forall system assignment point,
    (piRlcSemantics.evaluations system assignment point).size = 1
  piRlcAlgebra : PiRLC.Algebra
    (MatrixSource F shape columns blockCount)
    (Assignment F columns)
    PublicInput
    (CubePoint Extension shape.cubeVariables)
    (EvaluationFamily Extension shape)
    Commitment
    Scalar
    piRlcSemantics
    piCcs.params

namespace CompatibleContext

/-- Partition agreement implies the exact `K+k` total used by both
reductions. -/
theorem total_eq_sourceCount
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount) :
    context.arity.total = shape.sourceCount := by
  simp only [BatchArity.total, Shape.sourceCount]
  rw [context.freshCount_eq, context.runningCount_eq]

/-- Reindex one `Pi_RLC` coordinate into the exact paper output order. -/
def sourceIndex
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (source : Fin context.arity.total) : Fin shape.sourceCount :=
  Fin.cast context.total_eq_sourceCount source

/-- The `Pi_RLC` deterministic context uses exactly the relation semantics,
parameters, arity, and linear-combination algebra of the shared context. -/
def piRlc
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount) :
    PiRLC.PaperCompleteness.Context
      (MatrixSource F shape columns blockCount)
      (Assignment F columns)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment
      Scalar where
  semantics := context.piRlcSemantics
  params := context.piCcs.params
  arity := context.arity
  algebra := context.piRlcAlgebra
  evaluationCount := fun _ => 1
  evaluationsSize := context.piRlcEvaluationsSize

/-- The authoritative intermediate batch is the verifier's complete
`Pi_CCS` public output at its sampled round point. -/
def batchOfPrefix
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (causalRun : PrefixExecution Extension shape) :
    InputBatch
      (MatrixSource F shape columns blockCount)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment
      context.piCcs.params
      context.arity where
  system := context.piCcs.statement.matrixSource
  point := causalRun.probe.coins.roundPoint
  inputs := fun source =>
    context.piCcs.statement.publicOutput causalRun.probe
      (context.sourceIndex source)
  sameSystem := fun _ => rfl
  samePoint := fun _ => rfl
  evaluationCount := 1
  evaluationsSize := fun _ => rfl

@[simp] theorem batchOfPrefix_input
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (causalRun : PrefixExecution Extension shape)
    (source : Fin context.arity.total) :
    (context.batchOfPrefix causalRun).inputs source =
      context.piCcs.statement.publicOutput causalRun.probe
        (context.sourceIndex source) := by
  rfl

/-- Honest target membership and relaxed target membership are two relations
over the same public output.  Its protocol stage is therefore `fresh`; the
corrected ambient extractor relation overrides the norm bound separately. -/
@[simp] theorem batchOfPrefix_inputFresh
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (causalRun : PrefixExecution Extension shape)
    (source : Fin context.arity.total) :
    ((context.batchOfPrefix causalRun).inputs source).stage = .fresh := by
  rfl

/-- The shared projection is literally the statement's commitment vector in
the partition-preserving output order. -/
theorem batchPhi_eq_statementCommitments
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (causalRun : PrefixExecution Extension shape) :
    PiRLC.phi (context.batchOfPrefix causalRun).inputs =
      fun source =>
        context.piCcs.statement.commitments (context.sourceIndex source) := by
  rfl

/-- The `Pi_RLC` projection equals `Pi_CCS.outputPhi` after the sole audited
arity transport. -/
theorem batchPhi_eq_piCcsOutputPhi
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (causalRun : PrefixExecution Extension shape) :
    PiRLC.phi (context.batchOfPrefix causalRun).inputs =
      fun source =>
        outputPhi (context.piCcs.statement.publicOutput causalRun.probe)
          (context.sourceIndex source) := by
  rfl

/-- Two fresh `Pi_CCS` prefixes for one statement induce equal weak-game
inputs under the exact commitment projection, as required by Theorem 6. -/
theorem repeatedBatch_samePhi
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (left right : PrefixExecution Extension shape) :
    PiRLC.phi (context.batchOfPrefix left).inputs =
      PiRLC.phi (context.batchOfPrefix right).inputs := by
  rw [context.batchPhi_eq_statementCommitments,
    context.batchPhi_eq_statementCommitments]

end CompatibleContext

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch
