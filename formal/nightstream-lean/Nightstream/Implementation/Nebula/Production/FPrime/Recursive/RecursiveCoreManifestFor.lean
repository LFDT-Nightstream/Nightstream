import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.FoldCoreManifestFor
import Nightstream.Implementation.Nebula.Production.Artifact.StatementAuthorityRowsFor
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: exact ordered row manifest for the exponent-indexed recursive core.

The common paper fold is one nested object. This module adds the row families
that distinguish a recursive invocation from a terminal invocation: memory
continuation, recursive successor production, the current checked-memory
batch used by the next fresh claim, 20 verifier-owned static statement rows,
and
eight dynamic memory-authority links. The construction prevents the recursive
and terminal NIFS verifier cores from drifting apart.

`RowsIncluded` is actual ordered row inclusion. It is not a row-count test.
Therefore satisfaction of the final relation implies satisfaction of every
row in this manifest.

This module does not claim that the final generated relation exists. It does
not own application rows, terminal rows, compiler refinement, cryptographic
reductions, external bytes, or Rust refinement.

Assurance tier: exponent-indexed row implementation.

Emits constraints: common paper-fold rows, continuation rows, successor rows,
one current checked-memory batch, 20 verifier-owned statement-authority rows,
and eight dynamic authority equality rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.Nebula.ProductionRecursiveCoreManifestFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Verifier-key data for one recursive invocation. The complete verifier
core occurs exactly once in `fold`; this structure cannot replace it with an
independent copy. -/
structure Program (candidate : Id) (rowVariables : Nat) where
  fold : ProductionPaperFoldCoreManifestFor.Program candidate rowVariables
  statementIdentity : Soundness.StatementIdentity Digest.Value
  statementProfileExact : statementIdentity.profile = identity candidate
  continuationLayout : ProductionMemorySegmentContinuationRows.Layout candidate
  continuationValid : continuationLayout.Valid
  continuationIntermediate : continuationLayout.intermediate =
    fold.priorLayout.ccs.core.batch.frame.memory.boundaries
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))
  successorLayout : ProductionRecursiveSuccessorRowsFor.Layout rowVariables
  successorValid : successorLayout.Valid fold.priorLayout continuationLayout
  successorNifsOutputAlias : forall index,
    successorLayout.nifsOutputColumn index =
      fold.nifsOutputLayout.carrierColumn index
  currentMemoryLayout : ProductionMemoryCheckedBatchRows.Layout candidate
  currentMemoryValid : currentMemoryLayout.Valid
  currentMemoryStartsAt : currentMemoryLayout.boundaries 0 =
    continuationLayout.outgoing
  currentMemoryHeadersFromPrior : forall index role lane,
    (currentMemoryLayout.boundaries index).carry.headerColumn role lane =
      fold.priorLayout.carry.carry.headerColumn role lane

def priorStateAuthorityPosition (lane : Fin 4) : Fin 28 :=
  ⟨20 + lane.val, by omega⟩

def preCarryAuthorityPosition (lane : Fin 4) : Fin 28 :=
  ⟨24 + lane.val, by omega⟩

/-- The static challenge authority comes from the verifier artifact and writes
into the opening transcript's authority columns. It does not read a prover
sidecar. -/
def Program.statementAuthorityLayout
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    ProductionStatementAuthorityRowsFor.Layout candidate rowVariables :=
  { statementIdentity := program.statementIdentity
    profileExact := program.statementProfileExact
    authorityColumn :=
      program.continuationLayout.opening.transcript.frame.authorityColumn }

/-- The prior-state lanes come from the state digest already verified in the
fold core. The accumulator lanes come from the dedicated gated digest of the
exact NIFS-output successor prefix. -/
def Program.dynamicAuthorityLinkRows
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) : List Row :=
  (List.ofFn fun lane : Fin 4 =>
    Canonical.KEquality.equalityRow
      [(program.continuationLayout.opening.transcript.frame.authorityColumn
          (priorStateAuthorityPosition lane), 1)]
      [(program.fold.priorLayout.ccs.core.stateDigestColumn lane, 1)]) ++
  (List.ofFn fun lane : Fin 4 =>
    Canonical.KEquality.equalityRow
      [(program.continuationLayout.opening.transcript.frame.authorityColumn
          (preCarryAuthorityPosition lane), 1)]
      (ProductionPreCarryDigestRowsFor.digestExpression candidate
        program.successorLayout.preCarryDigest program.fold.statementId lane))

theorem Program.dynamicAuthorityLinkRows_length
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    program.dynamicAuthorityLinkRows.length = 8 := by
  simp [Program.dynamicAuthorityLinkRows]

/-- Canonical family order for one mandatory recursive core. -/
def Program.rows
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) : List Row :=
  program.fold.rows ++
    ProductionMemorySegmentContinuationRows.rows program.continuationLayout ++
    ProductionRecursiveSuccessorRowsFor.rows program.successorLayout
      program.fold.priorLayout program.fold.statementId ++
    ProductionMemoryCheckedBatchRows.rows program.currentMemoryLayout ++
    ProductionStatementAuthorityRowsFor.rows program.statementAuthorityLayout ++
    program.dynamicAuthorityLinkRows

/-- Satisfaction of the recursive manifest implies satisfaction of the one
nested common fold manifest. -/
theorem Program.fold_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies program.fold.rows assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.continuation_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionMemorySegmentContinuationRows.rows program.continuationLayout)
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.successor_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionRecursiveSuccessorRowsFor.rows program.successorLayout
        program.fold.priorLayout program.fold.statementId) assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

/-- The batch committed into the next fresh claim is part of the fixed
recursive arm. It is not a producer-selected side relation. -/
theorem Program.currentMemory_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionMemoryCheckedBatchRows.rows program.currentMemoryLayout)
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.dynamicAuthorityLinks_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies program.dynamicAuthorityLinkRows assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.statementAuthority_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionStatementAuthorityRowsFor.rows
        program.statementAuthorityLayout) assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.rows_imply_staticAuthorityLane
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies program.rows assignment)
    (role : ProductionStatementAuthorityRowsFor.Role) (lane : Fin 4) :
    assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (ProductionStatementAuthorityRowsFor.authorityPosition role lane)) =
      ((role.digest program.statementIdentity).lanes lane).val := by
  exact ProductionStatementAuthorityRowsFor.rows_imply_lane
    canonical one (program.statementAuthority_satisfied satisfied) role lane

structure Program.DynamicAuthorityExact
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables)
    (assignment : Nat -> Nat) : Prop where
  priorState : forall lane : Fin 4,
    assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (priorStateAuthorityPosition lane)) =
      assignment (program.fold.priorLayout.ccs.core.stateDigestColumn lane)
  preCarry : forall lane : Fin 4,
    assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (preCarryAuthorityPosition lane)) =
      lcEval assignment
        (ProductionPreCarryDigestRowsFor.digestExpression candidate
          program.successorLayout.preCarryDigest program.fold.statementId lane)

theorem Program.rows_imply_dynamicAuthorityExact
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies program.rows assignment) :
    program.DynamicAuthorityExact assignment := by
  have linkRows := program.dynamicAuthorityLinks_satisfied satisfied
  constructor
  · intro lane
    have rowHolds := linkRows _ (by
      apply List.mem_append_left
      exact List.mem_ofFn.mpr ⟨lane, rfl⟩)
    have equality :=
      (Canonical.KEquality.equalityRow_iff assignment _ _ one).1 rowHolds
    simpa [lcEval, Nat.mod_eq_of_lt
      (canonical
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (priorStateAuthorityPosition lane))),
      Nat.mod_eq_of_lt
        (canonical (program.fold.priorLayout.ccs.core.stateDigestColumn lane))]
      using equality
  · intro lane
    have rowHolds := linkRows _ (by
      apply List.mem_append_right
      exact List.mem_ofFn.mpr ⟨lane, rfl⟩)
    have equality :=
      (Canonical.KEquality.equalityRow_iff assignment _ _ one).1 rowHolds
    simpa [lcEval, Nat.mod_eq_of_lt
      (canonical
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (preCarryAuthorityPosition lane)))] using equality

def canonicalDigestValue
    (digest : ProductionSuccessorStateBinding.CanonicalDigest) : Digest.Value :=
  { lanes := digest }

/-- The sole memory-challenge authority permitted by a recursive invocation.
The two dynamic digests are computed from the exact prior state and the exact
NIFS-output successor prefix. -/
noncomputable def Program.openingAuthority
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (program : Program candidate rowVariables)
    (prior successor : ProductionSuccessorStateBinding.Value candidate
      fullShape) : MemoryOpenSegment.Authority :=
  MemoryOpenSegment.Authority.ofIdentityAndState
    program.statementIdentity
    (canonicalDigestValue
      (ProductionSuccessorStateBinding.outputDigest program.fold.statementId
        prior))
    (canonicalDigestValue
      (ProductionSuccessorStateBinding.preCarryDigest program.fold.statementId
        successor.preCarry))

/-- The four accumulator-authority lanes come from the gated digest of the
exact NIFS-output successor prefix.  This lemma is separate from the 28-lane
assembly theorem so Lean does not re-elaborate the large Poseidon2 state
argument while it performs finite authority indexing. -/
theorem Program.rows_imply_preCarryAuthorityLane
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)}
    (successorPlaced : ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) program.successorLayout.successor assignment successor)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies program.rows assignment)
    (lane : Fin 4) :
    assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (preCarryAuthorityPosition lane)) =
      (ProductionSuccessorStateBinding.preCarryDigest
        program.fold.statementId successor.preCarry lane).val := by
  have dynamic := program.rows_imply_dynamicAuthorityExact canonical one
    satisfied
  have digestExact :=
    ProductionRecursiveSuccessorRowsFor.rows_imply_preCarry_digest_lane
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) canonical one program.fold.statementId successor
    successorPlaced (program.successor_satisfied satisfied) lane
  exact (dynamic.preCarry lane).trans digestExact

/-- Full recursive-manifest satisfaction derives the exact 28-field challenge
authority. No authority record, digest equality, or challenge result is an
assumption. -/
theorem Program.rows_imply_openingAuthorityPlaced
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    {prior successor : ProductionSuccessorStateBinding.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)}
    (priorDigestPlaced : ProductionMemoryBatchCcsLinkRowsFor.StateDigestPlaced
      program.fold.priorLayout.ccs assignment
      (ProductionSuccessorStateBinding.outputDigest program.fold.statementId
        prior))
    (successorPlaced : ProductionSuccessorStateBindingRowsFor.Placed
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) program.successorLayout.successor assignment successor)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies program.rows assignment) :
    MemoryOpenSegmentSound.AuthorityPlaced
      program.continuationLayout.opening assignment
      (program.openingAuthority prior successor) := by
  have dynamic := program.rows_imply_dynamicAuthorityExact canonical one
    satisfied
  apply MemoryOpenSegmentSound.authorityPlaced_of_lanes
  intro digest lane
  fin_cases digest
  · change assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (ProductionStatementAuthorityRowsFor.authorityPosition
            .verifierKey lane)) =
        (program.statementIdentity.verifierKey.digest.lanes lane).val
    exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .verifierKey lane
  · change assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (ProductionStatementAuthorityRowsFor.authorityPosition
            .applicationRelation lane)) =
        (program.statementIdentity.applicationRelationDigest.lanes
          lane).val
    exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .applicationRelation lane
  · change assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (ProductionStatementAuthorityRowsFor.authorityPosition .program
            lane)) =
        (program.statementIdentity.programDigest.lanes lane).val
    exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .program lane
  · change assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (ProductionStatementAuthorityRowsFor.authorityPosition .memoryPlan
            lane)) =
        (program.statementIdentity.memoryPlanDigest.lanes lane).val
    exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .memoryPlan lane
  · change assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (ProductionStatementAuthorityRowsFor.authorityPosition .laneLayout
            lane)) =
        (program.statementIdentity.verifierKey.laneLayoutDigest.lanes
          lane).val
    exact program.rows_imply_staticAuthorityLane canonical one satisfied
      .laneLayout lane
  · change assignment
        (program.continuationLayout.opening.transcript.frame.authorityColumn
          (priorStateAuthorityPosition lane)) =
        (ProductionSuccessorStateBinding.outputDigest
          program.fold.statementId prior lane).val
    exact (dynamic.priorState lane).trans (priorDigestPlaced lane)
  · exact program.rows_imply_preCarryAuthorityLane successorPlaced canonical
      one satisfied lane

/-- Static equality between the nested common fold manifest and one typed
recursive-call row instance. It contains no assignment values, verifier
acceptance, transition, or execution result. -/
structure Program.MatchesRecursiveRows
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (program : Program candidate rowVariables)
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (nifsOutputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables) : Prop where
  foldMatches : program.fold.MatchesRows statementId config artifact value
    wires samplerBase algebraLayout piDecLayout nifsOutputLayout priorAuthority

/-- Satisfaction of the recursive manifest derives the verifier and delayed
memory row bundle. The result comes from the shared fold rows, not from a
separate recursive-only premise. -/
theorem Program.rows_imply_recursive_rowsHold
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {program : Program candidate rowVariables}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables}
    {samplerBase : Nat}
    {algebraLayout : ProductPiRlcAlgebraRows.Layout}
    {piDecLayout : ProductPiDecRows.Layout}
    {nifsOutputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat}
    (matched : program.MatchesRecursiveRows statementId config artifact value
      wires samplerBase algebraLayout piDecLayout nifsOutputLayout
      priorAuthority)
    (satisfied : Satisfies program.rows assignment) :
    ProductionPaperRecursiveRelationRowsSoundFor.RowsHold candidate statementId
      config artifact value proof wires samplerBase algebraLayout piDecLayout
      nifsOutputLayout priorAuthority program.fold.seedManifest
      program.fold.compactLayout assignment :=
  program.fold.rows_imply_recursive_rowsHold matched.foldMatches
    (program.fold_satisfied satisfied)

theorem Program.rows_length_exact
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    program.rows.length =
      ProductionRecursiveCoreGeometryFor.knownCoreRows candidate rowVariables := by
  simp only [Program.rows, List.length_append,
    ProductionPaperFoldCoreManifestFor.Program.rows_length_exact,
    ProductionMemorySegmentContinuationRows.rows_length_exact
      program.continuationValid,
    ProductionRecursiveSuccessorRowsFor.rows_length_exact,
    ProductionMemoryCheckedBatchRows.rows_length_exact,
    ProductionStatementAuthorityRowsFor.rows_length,
    Program.dynamicAuthorityLinkRows_length]
  simp [ProductionPaperFoldCoreManifestFor.rowCount,
    ProductionRecursiveCoreGeometryFor.knownCoreRows,
    Nat.add_assoc, Nat.add_comm]
  omega

/-- Actual ordered inclusion of the mandatory rows in a final relation. -/
def Program.RowsIncluded
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) (finalRows : List Row) : Prop :=
  program.rows.Sublist finalRows

theorem Program.length_le_of_rowsIncluded
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {finalRows : List Row}
    (included : program.RowsIncluded finalRows) :
    ProductionRecursiveCoreGeometryFor.knownCoreRows candidate rowVariables <=
      finalRows.length := by
  rw [← program.rows_length_exact]
  exact included.length_le

/-- Final-relation satisfaction transfers to every exact core row. -/
theorem Program.satisfies_of_rowsIncluded
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {finalRows : List Row}
    {assignment : Nat -> Nat}
    (included : program.RowsIncluded finalRows)
    (satisfied : Satisfies finalRows assignment) :
    Satisfies program.rows assignment := by
  intro row member
  exact satisfied row (included.subset member)

/-! ## Necessity countermodel -/

def zeroRow : Row := ⟨[], [], []⟩

def rejectingConstantRow : Row :=
  ⟨[(0, 1)], [(0, 1)], []⟩

/-- Equal row counts do not prove row containment. A length-only gate accepts
replacement of a mandatory rejecting row by a zero row, while exact ordered
inclusion rejects it. -/
theorem length_only_accepts_zero_row_substitution :
    [rejectingConstantRow].length <= [zeroRow].length /\
      ¬ [rejectingConstantRow].Sublist [zeroRow] := by
  simp [rejectingConstantRow, zeroRow]

end Nightstream.Implementation.Nebula.ProductionRecursiveCoreManifestFor
