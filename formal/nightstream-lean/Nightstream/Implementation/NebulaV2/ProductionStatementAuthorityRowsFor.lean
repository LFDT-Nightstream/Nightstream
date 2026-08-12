import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
import Nightstream.Protocol.NebulaV2.Soundness

/-!
Contract: verifier-key-owned source rows for the five static memory-challenge
digests.

The field-native fresh claim does not carry an application-statement sidecar.
Each output lane is equal to one constant lane in the verifier artifact's
exact `StatementIdentity`. Thus a prover cannot change the challenge identity
by changing witness columns and recomputing a digest chain.

`profileExact` is artifact configuration. It is not a proof-witness fact. A
candidate-specific generated artifact must construct this layout once.

Assurance tier: exponent-indexed row implementation.

Does not own recomputation of the identity digests from deployed artifacts,
external statement parsing, Poseidon2 security, or verifier-key generation.

Emits constraints: 20 rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionStatementAuthorityRowsFor

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Digest
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.Soundness

inductive Role where
  | verifierKey
  | applicationRelation
  | program
  | memoryPlan
  | laneLayout
deriving DecidableEq, Repr

def allRoles : List Role :=
  [ .verifierKey, .applicationRelation, .program, .memoryPlan, .laneLayout ]

theorem allRoles_length : allRoles.length = 5 := rfl

def Role.authorityIndex : Role -> Nat
  | .verifierKey => 0
  | .applicationRelation => 1
  | .program => 2
  | .memoryPlan => 3
  | .laneLayout => 4

def Role.digest
    (role : Role) (statementIdentity : StatementIdentity Digest.Value) :
    Digest.Value :=
  match role with
  | .verifierKey => statementIdentity.verifierKey.digest
  | .applicationRelation => statementIdentity.applicationRelationDigest
  | .program => statementIdentity.programDigest
  | .memoryPlan => statementIdentity.memoryPlanDigest
  | .laneLayout => statementIdentity.verifierKey.laneLayoutDigest

def authorityPosition (role : Role) (lane : Fin 4) : Fin 28 :=
  ⟨role.authorityIndex * 4 + lane.val, by
    cases role <;> simp [Role.authorityIndex] <;> omega⟩

/-- One verifier-artifact-owned static identity and its output columns. The
`rowVariables` index prevents accidental reuse across generated exponents. -/
structure Layout (candidate : Id) (_rowVariables : Nat) where
  statementIdentity : StatementIdentity Digest.Value
  profileExact : statementIdentity.profile = identity candidate
  authorityColumn : Fin 28 -> Nat

def Layout.row
    {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables) (role : Role) (lane : Fin 4) :
    Row :=
  KEquality.equalityRow
    [(layout.authorityColumn (authorityPosition role lane), 1)]
    [(0, ((role.digest layout.statementIdentity).lanes lane).val)]

def rows
    {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables) : List Row :=
  allRoles.flatMap fun role =>
    List.ofFn fun lane : Fin 4 => layout.row role lane

theorem rows_length
    {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables) :
    (rows layout).length = 20 := by
  simp [rows, allRoles]

theorem row_mem
    {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables) (role : Role) (lane : Fin 4) :
    layout.row role lane ∈ rows layout := by
  apply List.mem_flatMap.mpr
  refine ⟨role, ?_, List.mem_ofFn.mpr ⟨lane, rfl⟩⟩
  cases role <;> simp [allRoles]

/-- Satisfying rows force one authority column to the exact verifier-owned
digest lane. No claim placement or prover digest is a premise. -/
theorem rows_imply_lane
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (role : Role) (lane : Fin 4) :
    assignment (layout.authorityColumn (authorityPosition role lane)) =
      ((role.digest layout.statementIdentity).lanes lane).val := by
  have equal :=
    (KEquality.equalityRow_iff assignment _ _ one).1
      (holds _ (row_mem layout role lane))
  have digestBound :
      ((role.digest layout.statementIdentity).lanes lane).val < goldilocksP := by
    simpa [goldilocksP,
      Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.modulus] using
      ((role.digest layout.statementIdentity).lanes lane).property
  simp [Layout.row, lcEval,
    Nat.mod_eq_of_lt
      (canonical (layout.authorityColumn (authorityPosition role lane))),
    one] at equal
  rw [Nat.mod_eq_of_lt digestBound] at equal
  exact equal

/-- Honest placement of the verifier-owned constants satisfies every static
authority row. -/
theorem rows_complete
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : forall role lane,
      assignment (layout.authorityColumn (authorityPosition role lane)) =
        ((role.digest layout.statementIdentity).lanes lane).val) :
    Satisfies (rows layout) assignment := by
  intro row member
  rcases List.mem_flatMap.mp member with ⟨role, roleMember, rowMember⟩
  rcases List.mem_ofFn.mp rowMember with ⟨lane, rowExact⟩
  subst row
  apply (KEquality.equalityRow_iff assignment _ _ one).2
  have digestBound :
      ((role.digest layout.statementIdentity).lanes lane).val < goldilocksP := by
    simpa [goldilocksP,
      Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.modulus] using
      ((role.digest layout.statementIdentity).lanes lane).property
  simp [Layout.row, lcEval,
    Nat.mod_eq_of_lt
      (canonical (layout.authorityColumn (authorityPosition role lane))),
    one]
  rw [Nat.mod_eq_of_lt digestBound]
  exact placed role lane

end Nightstream.Implementation.NebulaV2.ProductionStatementAuthorityRowsFor
