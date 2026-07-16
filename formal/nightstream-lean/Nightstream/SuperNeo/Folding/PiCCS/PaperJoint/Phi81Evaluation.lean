import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanEvaluation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource

/-!
Batch-invariant Phi81 carried-evaluation leaves.

Protocol: SuperNeo coefficient embedding and `Pi_CCS` output CE claims.
Phase: derived coefficient-matrix image at one verifier-owned row point.
Constraint family: semantic evaluation only; this file emits no rows.

Owns: the single mathematical formula that turns a canonical Phi81 matrix
source, one complete-carrier assignment, one matrix, and one coefficient lane
into a Boolean row table and its multilinear evaluation.

Does not own: source-batch ordering, fresh/running arity, claimed output
products, `yZcol`, public inputs, commitments, transcript derivation, Rust,
R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: coefficient matrices are read only through
`MatrixSource.coefficientMatrix`; callers cannot supply an independent carried
matrix. `freshCount` and `runningCount` occur only in the source's paper shape
and do not participate in either definition below.

| Protocol | Phase | Leaf | Mathematical obligation |
|---|---|---|---|
| coefficient embedding | matrix image | `table` | tabulate `cf(bar(M) z)[lane]` over the Boolean row cube |
| carried CE | row evaluation | `evaluate` | evaluate that sole derived table at the verifier-owned point |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81Evaluation

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

/-- Boolean row table for one coefficient image derived from the sole Phi81
matrix source. Batch counts are phantom shape parameters and do not affect the
formula. -/
def table
    {cubeVariables freshCount runningCount matrixCount columns blockCount : Nat}
    (matrixSource : MatrixCoefficientSource.MatrixSource F
      (Phi81MatrixSource.phi81Shape cubeVariables freshCount runningCount
        matrixCount)
      columns blockCount)
    (assignment : Assignment F columns)
    (matrix : Fin matrixCount)
    (lane : Fin ringDegree) : BooleanTable K cubeVariables :=
  BooleanTable.tabulate fun vertex =>
    K.embed <| matrixVectorAt ConcreteCarrier.baseOps
      (matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix lane)
      assignment vertex

/-- Evaluate one source-derived Phi81 coefficient image at the typed
verifier-owned row point. -/
def evaluate
    {cubeVariables freshCount runningCount matrixCount columns blockCount : Nat}
    (matrixSource : MatrixCoefficientSource.MatrixSource F
      (Phi81MatrixSource.phi81Shape cubeVariables freshCount runningCount
        matrixCount)
      columns blockCount)
    (assignment : Assignment F columns)
    (point : CubePoint K cubeVariables)
    (matrix : Fin matrixCount)
    (lane : Fin ringDegree) : K :=
  (table matrixSource assignment matrix lane).evaluate
    ConcreteCarrier.extensionOps point

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81Evaluation
