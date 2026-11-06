import NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding

/-! SuperNeo v1.2 Definition 7 at the norm that a chain extracts. A chain recovers accumulators
at up to `B + cap · T · (b − 1)`, so two forks on one commitment give openings up to twice that
bound, not `2B`. The collision algebra of `PaperForkBinding.collisionAt` reads no norm: it runs
on the relation without norms, and each opening gets its norm from the fork responses. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.ChainBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
open NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding (Compatible)

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint} {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  {algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
    semantics params}

/-- SuperNeo v1.2 Definition 7 with openings bounded by `bound`. -/
structure CollisionAt
    (semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams) (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (bound : Nat) (commitment : Commitment) where
  collision : RelaxedBindingCollision (normFree semantics) params ops commitment
  firstNorm : semantics.normBounded bound collision.opening₁
  secondNorm : semantics.normBounded bound collision.opening₂

theorem normFreeCompatible (laws : ExtractionAlgebra semantics params algebra)
    (ops : RelaxedBindingOps Assignment Commitment Scalar) (compatible : Compatible laws ops) :
    Compatible (normFreeLaws laws) ops :=
  { differenceValid := compatible.differenceValid
    assignmentAction := compatible.assignmentAction
    commitmentAction := compatible.commitmentAction
    differenceBounded := fun _ _ _ _ => trivial }

/-- Every response of the fork opens at norm `bound`. -/
def ResponsesBounded
    {batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity}
    (bound : Nat)
    (fork : CompleteFork (normFree semantics) params (normFreeAlgebra algebra) batch) : Prop :=
  semantics.normBounded bound fork.base.assignment ∧
    ∀ coordinate, semantics.normBounded bound (fork.forks coordinate).assignment

/-- SuperNeo v1.2 Definition 7 for two Π_RLC forks on one input commitment whose responses open
at `bound` and whose extracted openings differ. -/
def forkCollision (laws : ExtractionAlgebra semantics params algebra)
    (ops : RelaxedBindingOps Assignment Commitment Scalar) (compatible : Compatible laws ops)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid) {bound : Nat}
    (differenceBounded : ∀ left right, semantics.normBounded bound left →
      semantics.normBounded bound right →
      semantics.normBounded (2 * bound) (laws.assignmentModule.sub left right))
    {leftBatch rightBatch : InputBatch Structure PublicInput Point Evaluation Commitment params
      arity}
    (left : CompleteFork (normFree semantics) params (normFreeAlgebra algebra) leftBatch)
    (right : CompleteFork (normFree semantics) params (normFreeAlgebra algebra) rightBatch)
    (leftBounded : ResponsesBounded bound left) (rightBounded : ResponsesBounded bound right)
    (samePhi : phi leftBatch.inputs = phi rightBatch.inputs) (coordinate : Fin arity.total)
    (different : extractedAssignment (normFreeLaws laws) strongSet left coordinate ≠
      extractedAssignment (normFreeLaws laws) strongSet right coordinate) :
    CollisionAt semantics params ops (2 * bound) (leftBatch.inputs coordinate).commitment where
  collision := PaperForkBinding.collisionAt (normFreeLaws laws) ops
    (normFreeCompatible laws ops compatible) strongSet leftBatch rightBatch left right samePhi
    coordinate different
  firstNorm := differenceBounded _ _ leftBounded.1 (leftBounded.2 coordinate)
  secondNorm := differenceBounded _ _ rightBounded.1 (rightBounded.2 coordinate)

end NightstreamFPrime.Spec.Folding.PiRLC.ChainBinding
