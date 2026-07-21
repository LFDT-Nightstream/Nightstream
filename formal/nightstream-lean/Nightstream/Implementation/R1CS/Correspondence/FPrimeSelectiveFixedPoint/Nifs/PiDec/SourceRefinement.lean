import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.PaperBridge
import Nightstream.Implementation.R1CS.Correspondence.Projection.IndexedRows

/-!
Artifact-checked source-R1CS refinement for the bounded active strict
`PiDEC` invocation.

Owns: transport from satisfaction of the exact generated Rust source rows to
the independent production compiler endpoint, typed carrier, and operational
paper verifier.

Does not own: selective-CCS reconstruction, the production-security profile,
`FixedActive.ResultTransition`, certificate/output-column identity, delayed
`s_col` or `y_zcol`, commitment binding, or row removal.

Emits constraints: no.

Assurance tier: artifact-checked for the bounded `kappa = 4` source fixture;
the exact sparse coefficient comparison uses `Lean.trustCompiler`.

| Refinement step | Input authority | Result |
|---|---|---|
| raw layout | stable artifact facade | typed compiler layout |
| sparse rows | generated source A/B/C | exact production compiler rows |
| source satisfaction | canonical assignment and constant one | compiler acceptance |
| paper bridge | typed active profile | operational paper acceptance |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionIndexedRows

namespace SourceArtifact

abbrev rawLayout :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.rawLayout

abbrev sourceRows :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.sourceRows

abbrev commitmentRows :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.commitmentRows

def commitmentLayout
    (raw : Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.RawCommitment) :
    PiDecStrictCompiler.CommitmentLayout where
  dCol := raw.dCol
  kappaCol := raw.kappaCol
  dataCols := raw.dataCols

def claimLayout
    (raw : Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.RawClaim) :
    PiDecStrictCompiler.ClaimLayout where
  commitment := commitmentLayout raw.commitment
  adv := none
  xActiveCols := raw.xActiveCols
  xInactiveCol := raw.xInactiveCol
  xRows := raw.xRows
  xWidth := raw.xWidth
  xRowsCol := raw.xRowsCol
  xWidthCol := raw.xWidthCol
  mIn := raw.mIn
  mInCol := raw.mInCol
  yRingCols := raw.yRingCols
  ctCols := raw.ctCols
  rCols := raw.rCols
  sColCols := raw.sColCols
  foldDigestCols := raw.foldDigestCols

def baseLayout : PiDecStrictCompiler.Layout where
  radix := rawLayout.radix
  ringDimension := rawLayout.ringDimension
  extensionLimbs := rawLayout.extensionLimbs
  firstAllocatedColumn := rawLayout.firstAllocatedColumn
  parent := claimLayout rawLayout.parent
  children := rawLayout.children.map claimLayout

def layout : PiDecStrictProductionCompiler.Layout where
  base := baseLayout
  xSignTraces := rawLayout.xSignTraces
  childCount := by native_decide

private def rowsPermutationEquivalentListDecidable :
    (source reconstructed : List Row) →
      Decidable (RowsPermutationEquivalentList source reconstructed)
  | [], [] => isTrue True.intro
  | [], _ :: _ => isFalse id
  | _ :: _, [] => isFalse id
  | source :: sources, reconstructed :: reconstructions =>
      match inferInstanceAs
          (Decidable (RowsPermutationEquivalent source reconstructed)),
        rowsPermutationEquivalentListDecidable sources reconstructions with
      | isTrue head, isTrue tail => isTrue ⟨head, tail⟩
      | isFalse head, isTrue _ => isFalse fun equivalent => head equivalent.1
      | isTrue _, isFalse tail => isFalse fun equivalent => tail equivalent.2
      | isFalse head, isFalse _ => isFalse fun equivalent => head equivalent.1

local instance (source reconstructed : List Row) :
    Decidable (RowsPermutationEquivalentList source reconstructed) :=
  rowsPermutationEquivalentListDecidable source reconstructed

set_option maxRecDepth 100000 in
theorem compilerRows_length :
    (PiDecStrictProductionCompiler.rows layout).length =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Metadata.sourceRowCount := by
  native_decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 4000000 in
/-- Exact coefficient-level Rust/source-compiler identity. Row order is exact;
only sparse term order inside A/B/C is quotiented by `List.Perm`. -/
theorem sourceRows_exact : RowsPermutationEquivalentList sourceRows
    (PiDecStrictProductionCompiler.rows layout) := by
  native_decide

private theorem rowsPermutationEquivalentList_symm
    {left right : List Row}
    (equivalent : RowsPermutationEquivalentList left right) :
    RowsPermutationEquivalentList right left := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => trivial
      | cons _ _ => simp [RowsPermutationEquivalentList] at equivalent
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp [RowsPermutationEquivalentList] at equivalent
      | cons rightHead rightTail =>
          exact ⟨⟨equivalent.1.1.symm, equivalent.1.2.1.symm,
            equivalent.1.2.2.symm⟩,
            inductionHypothesis equivalent.2⟩

theorem shapeValid : PiDecStrictProductionCompiler.ShapeValid layout :=
  {
    base := {
      ringPositive := by native_decide
      powersCanonical := by native_decide
      commitmentLengths := by native_decide
      xShapes := by native_decide
      activeXLengths := by native_decide
      yShapes := by native_decide
      rShapes := by native_decide
      sColShapes := by native_decide
      ctShapes := by native_decide
      foldDigestShapes := by native_decide
    }
    radixTwo := by native_decide
    ringDimension := by native_decide
    extensionLimbs := by native_decide
    traceCount := by native_decide
    semanticYFits := by native_decide
  }

theorem parentNoAdv : layout.base.parent.adv = none :=
  rfl

theorem childrenNoAdv :
    ∀ child ∈ layout.base.children, child.adv = none :=
  by native_decide

theorem activeProfile :
    PiDecTypedCarrier.Active.ProfileFor commitmentRows layout.base :=
  {
    childCount := by native_decide
    radixTwo := by native_decide
    ringDimension := by native_decide
    extensionLimbs := by native_decide
    activePublicColumns := by native_decide
    parentCommitmentLength := by native_decide
    childCommitmentLength := by native_decide
    publicWidth := by native_decide
    parentPublicLength := by native_decide
    childPublicLength := by native_decide
    parentEvaluationCount := by native_decide
    childEvaluationCount := by native_decide
    parentEvaluationWidth := by native_decide
    childEvaluationWidth := by native_decide
    parentPointLength := by native_decide
    childPointLength := by native_decide
  }

set_option maxRecDepth 100000 in
theorem compilerRows_exact : RowsPermutationEquivalentList
    (PiDecStrictProductionCompiler.rows layout) sourceRows :=
  rowsPermutationEquivalentList_symm sourceRows_exact

end SourceArtifact

set_option maxRecDepth 100000 in
/-- Exact generated source rows force the independent reduced compiler
endpoint. Canonical residues and constant one remain explicit R1CS boundary
premises. -/
theorem sourceRows_imply_compilerAccepted
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies SourceArtifact.sourceRows assignment) :
    PiDecStrictProductionCompiler.Accepted SourceArtifact.layout assignment := by
  apply PiDecStrictProductionCompiler.sound_noAdv prime
    SourceArtifact.shapeValid SourceArtifact.parentNoAdv
    SourceArtifact.childrenNoAdv canonical constantOne
  exact sourceRows_satisfied_of_permutationEquivalent
    SourceArtifact.compilerRows_exact satisfies

/-- Artifact-backed typed endpoint, still separate from any active lifecycle
certificate or delayed projection carrier. -/
theorem sourceRows_imply_typedAccepted
    (prime : EuclidPrime goldilocksP)
    (key : PiRLCAlgebra.Commitment.Key PiDecTypedCarrier.Active.shape
      SourceArtifact.commitmentRows)
    (system : Phi81Relation.Structure PiDecTypedCarrier.Active.shape)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies SourceArtifact.sourceRows assignment) :
    PiDecTypedCarrier.Accepted SourceArtifact.activeProfile key system
      assignment := by
  exact PiDecStrictProductionCompiler.PaperBridge.accepted_refines_typed
    SourceArtifact.shapeValid SourceArtifact.activeProfile key system
    assignment
    (sourceRows_imply_compilerAccepted prime canonical constantOne satisfies)

/-- Operational SuperNeo Section-7.5 acceptance for the exact decoded active
parent and fourteen verifier-computed children. -/
theorem sourceRows_imply_paperAccepted
    (prime : EuclidPrime goldilocksP)
    (key : PiRLCAlgebra.Commitment.Key PiDecTypedCarrier.Active.shape
      SourceArtifact.commitmentRows)
    (system : Phi81Relation.Structure PiDecTypedCarrier.Active.shape)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies SourceArtifact.sourceRows assignment) :
    PiDEC.PaperVerifier.OutputAccepted
      (PiDECAlgebra.Algebra.concrete key)
      (PiDECAlgebra.PaperVerifier.publicInputSplit key)
      (PiDECAlgebra.PaperVerifier.evaluationArity key)
      (PiDecTypedCarrier.decodedParent SourceArtifact.activeProfile system
        assignment)
      (PiDecTypedCarrier.decodedOutput SourceArtifact.activeProfile system
        assignment) := by
  exact PiDecStrictProductionCompiler.PaperBridge.accepted_refines_paper
    SourceArtifact.shapeValid SourceArtifact.activeProfile key system
    assignment
    (sourceRows_imply_compilerAccepted prime canonical constantOne satisfies)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement
