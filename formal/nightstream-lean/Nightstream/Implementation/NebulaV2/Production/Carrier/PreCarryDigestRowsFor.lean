import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.SuccessorStateBindingRowsFor

/-!
Contract: domain-gated accumulator authority for one production F-prime
successor.

The complete successor hash already absorbs the exact NIFS output running
state.  This row program forks from that validated symbolic prefix before
either memory carry is absorbed, applies the Poseidon2 gate in a separate
physical window, and exposes four canonical digest lanes.  The memory
challenge can therefore bind the exact Construction-2 accumulator without a
second absorption of its roughly 83,000 fields and without a cycle through
the outgoing memory carry.

Does not own the equality from these four output expressions to the memory
transcript authority columns, statement-identity authority, Poseidon2
security, generated-window separation, Rust refinement, or a verifier key.

Emits constraints: one or two Poseidon2 permutations, depending on the
generated relation exponent's prefix cursor.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionPreCarryDigestRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

structure Layout (rowVariables : Nat) where
  source : ProductionSuccessorStateBindingRowsFor.Layout rowVariables
  sourceBase : Nat
  digestBase : Nat
deriving Repr

/-- Row-free symbolic fork.  It reuses the exact prefix lane expressions and
cursor but deliberately drops the source builder's already-emitted entries.
The source rows remain the authority for those expressions. -/
def forkStart
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) : SymbolicDuplex.Builder :=
  let source := ProductionSuccessorStateBindingRowsFor.preCarryBuilder
    candidate layout.sourceBase layout.source statementId
  SymbolicDuplex.start source.lanes source.absorbed

/-- Separate gate builder.  If the prefix cursor is full, the gate emits the
guard permutation and the terminal permutation.  Otherwise it emits only the
terminal permutation. -/
def builder
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) : SymbolicDuplex.Builder :=
  SymbolicDuplex.gate layout.digestBase
    (forkStart candidate layout statementId)

def rows
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) : List Row :=
  SymbolicDuplex.rows layout.digestBase ProductPoseidon2.constants
    (builder candidate layout statementId)

def digestExpression
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) (lane : Fin 4) :=
  (builder candidate layout statementId).lanes
    (ProductionSuccessorStateBinding.outputLane lane)

def forkControl (rowVariables : Nat) : SymbolicDuplexCount.Control :=
  SymbolicDuplexCount.gate
    { entries := 0
      absorbed :=
        (ProductionSuccessorStateBindingRowsFor.preCarryControl
          rowVariables).absorbed }

def permutationCount (rowVariables : Nat) : Nat :=
  (forkControl rowVariables).entries

theorem forkStart_control
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    SymbolicDuplexCount.ofBuilder (forkStart candidate layout statementId) =
      { entries := 0
        absorbed :=
          (ProductionSuccessorStateBindingRowsFor.preCarryControl
            rowVariables).absorbed } := by
  have sourceControl :=
    ProductionSuccessorStateBindingRowsFor.preCarryBuilder_control candidate
      layout.sourceBase layout.source statementId
  have absorbedExact :
      (ProductionSuccessorStateBindingRowsFor.preCarryBuilder candidate
        layout.sourceBase layout.source statementId).absorbed =
        (ProductionSuccessorStateBindingRowsFor.preCarryControl
          rowVariables).absorbed := by
    simpa [SymbolicDuplexCount.ofBuilder] using
      congrArg SymbolicDuplexCount.Control.absorbed sourceControl
  simp [forkStart, SymbolicDuplexCount.ofBuilder, SymbolicDuplex.start,
    absorbedExact]

theorem builder_control
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    SymbolicDuplexCount.ofBuilder (builder candidate layout statementId) =
      forkControl rowVariables := by
  rw [builder, SymbolicDuplexCount.ofBuilder_gate,
    forkStart_control]
  rfl

theorem builder_entries_length
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    (builder candidate layout statementId).entries.length =
      permutationCount rowVariables := by
  have control := builder_control candidate layout statementId
  have entries := congrArg SymbolicDuplexCount.Control.entries control
  simpa [SymbolicDuplexCount.ofBuilder, permutationCount] using entries

theorem rows_length_exact
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId) :
    (rows candidate layout statementId).length =
      permutationCount rowVariables * 352 := by
  rw [rows, SymbolicDuplex.rows_length,
    builder_entries_length]

theorem permutationCount_25 : permutationCount 25 = 1 := by
  have preCarryExact :
      ProductionSuccessorStateBindingRowsFor.preCarryControl 25 =
        { entries := 20847, absorbed := 2 } := by
    rw [ProductionSuccessorStateBindingRowsFor.preCarryControl,
      SymbolicDuplexCount.absorbMany_eq_fast]
    decide
  rw [permutationCount, forkControl, preCarryExact]
  decide

theorem permutationCount_26 : permutationCount 26 = 2 := by
  have preCarryExact :
      ProductionSuccessorStateBindingRowsFor.preCarryControl 26 =
        { entries := 20847, absorbed := 4 } := by
    rw [ProductionSuccessorStateBindingRowsFor.preCarryControl,
      SymbolicDuplexCount.absorbMany_eq_fast]
    decide
  rw [permutationCount, forkControl, preCarryExact]
  decide

theorem rows_length_25
    (candidate : Id) (layout : Layout 25)
    (statementId : ProductPoseidon2.StatementId) :
    (rows candidate layout statementId).length = 352 := by
  rw [rows_length_exact, permutationCount_25]

theorem decoded_forkStart_eq_source
    (candidate : Id) {rowVariables : Nat} (layout : Layout rowVariables)
    (statementId : ProductPoseidon2.StatementId)
    (assignment : Nat -> Nat) :
    decodedBuilder assignment (forkStart candidate layout statementId) =
      decodedBuilder assignment
        (ProductionSuccessorStateBindingRowsFor.preCarryBuilder candidate
          layout.sourceBase layout.source statementId) := by
  apply decodedBuilder_eq_of_lanes_absorbed <;> rfl

/-- The fork rows and the already-required full successor rows recover the
exact gated accumulator authority.  No claimed digest is an input. -/
theorem rows_imply_preCarryState
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate fullShape)
    (successorPlaced : ProductionSuccessorStateBindingRowsFor.Placed contract
      layout.source assignment successor)
    (sourceRows : Satisfies
      (ProductionSuccessorStateBindingRowsFor.rows candidate
        layout.sourceBase layout.source statementId) assignment)
    (digestRows : Satisfies (rows candidate layout statementId) assignment) :
    decodedBuilder assignment (builder candidate layout statementId) =
      ProductionSuccessorStateBinding.preCarryState statementId
        successor.preCarry := by
  have sourceExact :=
    ProductionSuccessorStateBindingRowsFor.rows_imply_preCarryAbsorbedState
      contract canonical one statementId successor successorPlaced
      layout.sourceBase sourceRows
  have digestValid : Valid layout.digestBase ProductPoseidon2.constants
      assignment (builder candidate layout statementId) :=
    valid_of_satisfied layout.digestBase ProductPoseidon2.constants
      (builder candidate layout statementId) assignment canonical one
      digestRows
  have gateExact := decodedBuilder_gate layout.digestBase
    ProductPoseidon2.constants assignment
    (forkStart candidate layout statementId) one digestValid
  rw [decoded_forkStart_eq_source candidate layout statementId assignment,
    sourceExact] at gateExact
  simpa [builder, ProductionSuccessorStateBinding.preCarryState] using gateExact

theorem rows_imply_digest_lane
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    {layout : Layout rowVariables} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate fullShape)
    (successorPlaced : ProductionSuccessorStateBindingRowsFor.Placed contract
      layout.source assignment successor)
    (sourceRows : Satisfies
      (ProductionSuccessorStateBindingRowsFor.rows candidate
        layout.sourceBase layout.source statementId) assignment)
    (digestRows : Satisfies (rows candidate layout statementId) assignment)
    (lane : Fin 4) :
    lcEval assignment
        (digestExpression candidate layout statementId lane) =
      (ProductionSuccessorStateBinding.preCarryDigest statementId
        successor.preCarry lane).val := by
  have stateExact := rows_imply_preCarryState contract canonical one
    statementId successor successorPlaced sourceRows digestRows
  have laneExact := congrArg
    (fun state => state.lanes
      (ProductionSuccessorStateBinding.outputLane lane)) stateExact
  simpa [digestExpression, decodedBuilder,
    ProductionSuccessorStateBinding.preCarryDigest] using laneExact

end Nightstream.Implementation.NebulaV2.ProductionPreCarryDigestRowsFor
