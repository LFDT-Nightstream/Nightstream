import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.PayloadRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentity

/-!
Contract: exact artifact-refinement boundary for `PaddedRowIdentity`.

Owns: fail-closed decoding of a complete thirteen-matrix payload; transport
of that payload to the dimensions owned by `PaddedRowIdentity`; and exact
equality with the relation produced by the independent Lean row compiler.

Does not own: a concrete production payload, a selected full source program,
Rust provenance, compiler execution, R1CS circuit correspondence, or a
release claim. No complete payload exists in the repository at present.

Emits constraints: no.

Assurance tier: artifact-ready. The complete matrix bridge is a proposition
that a future generated payload must inhabit. This file does not claim
artifact-checked matrix semantics until such an inhabitant exists.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity

abbrev FixedSnapshot :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.PayloadRefinement.FixedSnapshot

abbrev PayloadBundle :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.PayloadRefinement.PayloadBundle

/-- Transport one decoded payload relation to the selected dimensions. The
two equalities must come from the decoded header, not from repeated metadata. -/
def decodedMatrices
    {rawShape : FixedSnapshot}
    {rawBundle : PayloadBundle}
    (payload :
      Artifact.PayloadRefinement.Refinement rawShape rawBundle)
    (rowsExact : rawShape.materialized.verifier.rows = logicalRows)
    (columnsExact :
      rawShape.materialized.verifier.columns = assignmentColumns) :
    ApplicationMatrices := by
  change RelationProfile.FiniteRelation logicalRows assignmentColumns
  rw [← rowsExact, ← columnsExact]
  exact payload.toRelation

/-- Transport one independently compiled Lean source program to the selected
logical row count. -/
def compiledMatrices
    (one : Fin assignmentColumns)
    (program : List (DirectRows.SourceRow assignmentColumns))
    (rowsExact : program.length = logicalRows) : ApplicationMatrices := by
  change RelationProfile.FiniteRelation logicalRows assignmentColumns
  rw [← rowsExact]
  exact DirectRows.relation one program

/-- Complete matrix-refinement certificate required before a generated
payload can be called the selected production relation.

The certificate connects three authorities: the decoded payload, the selected
dimension census, and the independent Lean row compiler. Equality is over all
thirteen matrices, all logical rows, and all assignment columns. -/
structure ProductionMatrixRefinement where
  rawShape : FixedSnapshot
  rawBundle : PayloadBundle
  payload : Artifact.PayloadRefinement.Refinement rawShape rawBundle
  rowsExact : rawShape.materialized.verifier.rows = logicalRows
  columnsExact :
    rawShape.materialized.verifier.columns = assignmentColumns
  one : Fin assignmentColumns
  sourceProgram : List (DirectRows.SourceRow assignmentColumns)
  sourceRowsExact : sourceProgram.length = logicalRows
  decoded_eq_compiled :
    decodedMatrices payload rowsExact columnsExact =
      compiledMatrices one sourceProgram sourceRowsExact

namespace ProductionMatrixRefinement

/-- The sole selected application matrix family decoded from the certified
payload. -/
def matrices (refinement : ProductionMatrixRefinement) :
    ApplicationMatrices :=
  decodedMatrices refinement.payload refinement.rowsExact
    refinement.columnsExact

/-- Certified payload matrices are exactly the independent Lean compiler
output. -/
theorem matrices_eq_compiled (refinement : ProductionMatrixRefinement) :
    refinement.matrices =
      compiledMatrices refinement.one refinement.sourceProgram
        refinement.sourceRowsExact :=
  refinement.decoded_eq_compiled

/-- The payload decoder has accepted the exact raw bundle used by the
certificate. -/
theorem decoder_accepts (refinement : ProductionMatrixRefinement) :
    Artifact.Decoder.decodeProductionBundle refinement.payload.verifierFuel
        refinement.rawBundle = some refinement.payload.decoded :=
  refinement.payload.decodedFromRaw

/-- The certified payload contains exactly the thirteen selected matrices. -/
theorem matrixCount_exact (refinement : ProductionMatrixRefinement) :
    refinement.payload.decoded.bundle.matrices.length =
      applicationMatrixCount := by
  rw [Artifact.PayloadRefinement.Refinement.decoded_matrixCount_eq_13]
  rfl

end ProductionMatrixRefinement

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityArtifact
