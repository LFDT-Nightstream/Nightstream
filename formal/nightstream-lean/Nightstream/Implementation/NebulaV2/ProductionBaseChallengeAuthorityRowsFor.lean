import Nightstream.Implementation.NebulaV2.BaseManifestSchema
import Nightstream.Implementation.NebulaV2.ProductionPaperBaseInvocationFor
import Nightstream.Implementation.NebulaV2.ProductionPreCarryDigestRowsFor
import Nightstream.Implementation.NebulaV2.ProductionStatementAuthorityRowsFor
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: exact row program for the base-invocation memory-challenge authority.

The program fixes the five static statement digests, hashes the complete
canonical base input state, hashes the complete base successor, forks the
challenge-independent successor prefix, and links the resulting eight
dynamic digest lanes into the segment-open transcript. A selected base
artifact must contain these rows in order.

`Placed` premises below only state which typed base states occupy the source
columns. They do not assume a digest, challenge, authority equality, row
satisfaction, or F-prime conclusion. The application compiler must discharge
the successor placement against its generated columns.

Does not own application lowering, construction of the selected absolute
layout, Poseidon2 collision resistance, recursive-size closure, external
bytes, or Rust refinement.

Assurance tier: exponent-indexed row implementation.

Emits constraints: two complete successor-state Poseidon2 programs, one
pre-carry digest fork, 20 verifier-owned statement rows, and eight equality
rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.NebulaV2.ProductionBaseChallengeAuthorityRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- One generated base-authority row program. All constants in this value are
verifier-key data. -/
structure Program (candidate : Id) (rowVariables : Nat) where
  statementId : ProductPoseidon2.StatementId
  statementIdentity : Soundness.StatementIdentity Digest.Value
  statementProfileExact : statementIdentity.profile = identity candidate
  openingLayout : MemoryOpenSegmentRows.Layout
  initialLayout : ProductionSuccessorStateBindingRowsFor.Layout rowVariables
  initialHashBase : Nat
  successorLayout : ProductionSuccessorStateBindingRowsFor.Layout rowVariables
  successorHashBase : Nat
  preCarryDigestBase : Nat

def initialStateAuthorityPosition (lane : Fin 4) : Fin 28 :=
  ⟨20 + lane.val, by omega⟩

def preCarryAuthorityPosition (lane : Fin 4) : Fin 28 :=
  ⟨24 + lane.val, by omega⟩

def Program.statementAuthorityLayout
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    ProductionStatementAuthorityRowsFor.Layout candidate rowVariables :=
  { statementIdentity := program.statementIdentity
    profileExact := program.statementProfileExact
    authorityColumn :=
      program.openingLayout.transcript.frame.authorityColumn }

def Program.preCarryLayout
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    ProductionPreCarryDigestRowsFor.Layout rowVariables :=
  { source := program.successorLayout
    sourceBase := program.successorHashBase
    digestBase := program.preCarryDigestBase }

def Program.initialDigestExpression
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) (lane : Fin 4) :=
  (ProductionSuccessorStateBindingRowsFor.builder candidate
      program.initialHashBase program.initialLayout program.statementId).lanes
    (ProductionSuccessorStateBinding.outputLane lane)

/-- The base input-state digest and the base successor-prefix digest occupy
the last eight authority lanes. -/
def Program.dynamicAuthorityLinkRows
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) : List Row :=
  (List.ofFn fun lane : Fin 4 =>
    Canonical.KEquality.equalityRow
      [(program.openingLayout.transcript.frame.authorityColumn
          (initialStateAuthorityPosition lane), 1)]
      (program.initialDigestExpression lane)) ++
  (List.ofFn fun lane : Fin 4 =>
    Canonical.KEquality.equalityRow
      [(program.openingLayout.transcript.frame.authorityColumn
          (preCarryAuthorityPosition lane), 1)]
      (ProductionPreCarryDigestRowsFor.digestExpression candidate
        program.preCarryLayout program.statementId lane))

theorem Program.dynamicAuthorityLinkRows_length
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    program.dynamicAuthorityLinkRows.length = 8 := by
  simp [Program.dynamicAuthorityLinkRows]

/-- Canonical family order. The order is part of generated-artifact identity. -/
def Program.rows
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) : List Row :=
  ProductionStatementAuthorityRowsFor.rows program.statementAuthorityLayout ++
    ProductionSuccessorStateBindingRowsFor.rows candidate
      program.initialHashBase program.initialLayout program.statementId ++
    ProductionSuccessorStateBindingRowsFor.rows candidate
      program.successorHashBase program.successorLayout program.statementId ++
    ProductionPreCarryDigestRowsFor.rows candidate program.preCarryLayout
      program.statementId ++
    program.dynamicAuthorityLinkRows

theorem Program.rows_length_exact
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    program.rows.length =
      28 + 2 *
          (ProductionSuccessorStateBindingRowsFor.successorPermutationCount
            rowVariables * 352) +
        ProductionPreCarryDigestRowsFor.permutationCount rowVariables * 352 := by
  simp only [Program.rows, List.length_append,
    ProductionStatementAuthorityRowsFor.rows_length,
    ProductionSuccessorStateBindingRowsFor.rows_length_exact,
    ProductionPreCarryDigestRowsFor.rows_length_exact,
    Program.dynamicAuthorityLinkRows_length]
  omega

theorem Program.rows_length_25
    {candidate : Id} (program : Program candidate 25) :
    program.rows.length = 14698492 := by
  rw [program.rows_length_exact,
    ProductionSuccessorStateBindingRowsFor.successorPermutationCount_25,
    ProductionPreCarryDigestRowsFor.permutationCount_25]

theorem Program.rows_length_26
    {candidate : Id} (program : Program candidate 26) :
    program.rows.length = 14698844 := by
  rw [program.rows_length_exact,
    ProductionSuccessorStateBindingRowsFor.successorPermutationCount_26,
    ProductionPreCarryDigestRowsFor.permutationCount_26]

theorem Program.statementAuthority_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionStatementAuthorityRowsFor.rows
        program.statementAuthorityLayout) assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.initialHash_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionSuccessorStateBindingRowsFor.rows candidate
        program.initialHashBase program.initialLayout program.statementId)
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.successorHash_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionSuccessorStateBindingRowsFor.rows candidate
        program.successorHashBase program.successorLayout program.statementId)
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.preCarryDigest_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionPreCarryDigestRowsFor.rows candidate program.preCarryLayout
        program.statementId) assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.dynamicAuthorityLinks_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies program.dynamicAuthorityLinkRows assignment := by
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
        (program.openingLayout.transcript.frame.authorityColumn
          (ProductionStatementAuthorityRowsFor.authorityPosition role lane)) =
      ((role.digest program.statementIdentity).lanes lane).val := by
  exact ProductionStatementAuthorityRowsFor.rows_imply_lane canonical one
    (program.statementAuthority_satisfied satisfied) role lane

structure Program.DynamicAuthorityExact
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables)
    (assignment : Nat -> Nat) : Prop where
  initialState : forall lane : Fin 4,
    assignment
        (program.openingLayout.transcript.frame.authorityColumn
          (initialStateAuthorityPosition lane)) =
      lcEval assignment (program.initialDigestExpression lane)
  preCarry : forall lane : Fin 4,
    assignment
        (program.openingLayout.transcript.frame.authorityColumn
          (preCarryAuthorityPosition lane)) =
      lcEval assignment
        (ProductionPreCarryDigestRowsFor.digestExpression candidate
          program.preCarryLayout program.statementId lane)

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
        (program.openingLayout.transcript.frame.authorityColumn
          (initialStateAuthorityPosition lane)))] using equality
  · intro lane
    have rowHolds := linkRows _ (by
      apply List.mem_append_right
      exact List.mem_ofFn.mpr ⟨lane, rfl⟩)
    have equality :=
      (Canonical.KEquality.equalityRow_iff assignment _ _ one).1 rowHolds
    simpa [lcEval, Nat.mod_eq_of_lt
      (canonical
        (program.openingLayout.transcript.frame.authorityColumn
          (preCarryAuthorityPosition lane)))] using equality

def canonicalDigestValue
    (digest : ProductionSuccessorStateBinding.CanonicalDigest) : Digest.Value :=
  { lanes := digest }

noncomputable def Program.openingAuthority
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (program : Program candidate rowVariables)
    (initial successor : ProductionSuccessorStateBinding.Value candidate
      fullShape) : MemoryOpenSegment.Authority :=
  MemoryOpenSegment.Authority.ofIdentityAndState
    program.statementIdentity
    (canonicalDigestValue
      (ProductionSuccessorStateBinding.outputDigest program.statementId
        initial))
    (canonicalDigestValue
      (ProductionSuccessorStateBinding.preCarryDigest program.statementId
        successor.preCarry))

/-- The selected base artifact contains this exact authority row program and
uses the same segment-open authority columns. -/
structure Program.MatchesArtifact
    {candidate : Id} {rowVariables : Nat}
    {widths : FullClaimEnvelope.CompilerWidths}
    (program : Program candidate rowVariables)
    (artifact : BaseManifestSchema.Artifact widths) : Prop where
  openingExact : program.openingLayout = artifact.layouts.opening
  rowsIncluded : program.rows.Sublist artifact.other.challengeAuthority

theorem Program.satisfies_of_matchesArtifact
    {candidate : Id} {rowVariables : Nat}
    {widths : FullClaimEnvelope.CompilerWidths}
    {program : Program candidate rowVariables}
    {artifact : BaseManifestSchema.Artifact widths}
    (matched : program.MatchesArtifact artifact)
    {assignment : Nat -> Nat}
    (satisfied : Satisfies artifact.programRows assignment) :
    Satisfies program.rows assignment := by
  have authorityRows := artifact.challengeAuthority_satisfied satisfied
  intro row member
  exact authorityRows row (matched.rowsIncluded.subset member)

end Nightstream.Implementation.NebulaV2.ProductionBaseChallengeAuthorityRowsFor
