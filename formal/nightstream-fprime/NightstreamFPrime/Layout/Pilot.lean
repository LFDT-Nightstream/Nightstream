import NightstreamFPrime.Layout.Poseidon2
import NightstreamFPrime.Lifecycle.Pilot

/-!
Owns the physical pilot layout. Logical prior-state rows come first, followed
by logical output-hash rows. R1CS multiplication variables start after both
logical witness ranges. Every physical row has one phase and logical-row
owner.
-/

namespace NightstreamFPrime.Layout.Pilot

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle

abbrev Interface := Lifecycle.Pilot.Interface

def priorConstraints (interface : Interface) (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops (Lifecycle.Pilot.priorCircuit interface).main offset)

def outputOffset (interface : Interface) (offset : Nat) : Nat :=
  Lifecycle.Pilot.outputOffset interface offset

def outputConstraints (interface : Interface) (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops (Lifecycle.Pilot.outputCircuit interface).main
    (outputOffset interface offset))

def logicalConstraints (interface : Interface) (offset : Nat) : List Expr :=
  priorConstraints interface offset ++ outputConstraints interface offset

theorem priorConstraints_eq (interface : Interface) (offset : Nat) :
    priorConstraints interface offset =
      flatConstraints [PriorStateHash.hashOp interface.prior offset] ++
      flatConstraints (PriorStateHash.wordOps interface.prior offset) ++
      flatConstraints
        (PriorStateHash.bindingAssertions interface.prior offset) := by
  rw [priorConstraints]
  change flatConstraints
      (PriorStateHash.hashOp interface.prior offset ::
        (PriorStateHash.wordOps interface.prior offset ++
          PriorStateHash.bindingAssertions interface.prior offset)) = _
  simp [flatConstraints]

theorem outputConstraints_eq (interface : Interface) (offset : Nat) :
    outputConstraints interface offset =
      Poseidon2.hashConstraints
        (OutputHash.hashInterface interface.output)
        (outputOffset interface offset) := by
  rfl

def logicalColumnCount (interface : Interface) (offset : Nat) : Nat :=
  outputOffset interface offset +
    OutputHash.hashLength interface.output (outputOffset interface offset)

theorem outputOffset_eq_add (interface : Interface) (offset : Nat) :
    outputOffset interface offset = offset +
      localLength (Circuit.ops (Lifecycle.Pilot.priorCircuit interface).main
        offset) := by
  unfold outputOffset Lifecycle.Pilot.outputOffset
  rw [← Lifecycle.Pilot.priorCircuit_localLength]

theorem logicalColumnCount_eq_add (interface : Interface) (offset : Nat) :
    logicalColumnCount interface offset = outputOffset interface offset +
      localLength (Circuit.ops (Lifecycle.Pilot.outputCircuit interface).main
        (outputOffset interface offset)) := by
  unfold logicalColumnCount
  rw [← Lifecycle.Pilot.outputCircuit_localLength]

theorem logicalConstraints_varsBelow
    (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Lifecycle.Pilot.Assumptions interface offset env) :
    ∀ expression ∈ logicalConstraints interface offset,
      expression.VarsBelow (logicalColumnCount interface offset) := by
  intro expression member
  rcases List.mem_append.mp member with priorMember | outputMember
  · have below := PriorStateHash.flatConstraints_varsBelow interface.prior
      offset assumptions.1 expression priorMember
    rw [PriorStateHash.localLength_eq] at below
    exact Expr.VarsBelow.mono expression below (by
      unfold logicalColumnCount outputOffset Lifecycle.Pilot.outputOffset
      omega)
  · have below := OutputHash.flatConstraints_varsBelow interface.output
      (outputOffset interface offset) assumptions.2 expression outputMember
    rw [OutputHash.circuit_localLength] at below
    simpa [logicalColumnCount] using below

def lowering (interface : Interface) (offset : Nat) : R1CS.LoweredConstraints :=
  R1CS.lowerConstraints (logicalConstraints interface offset)
    (logicalColumnCount interface offset)

def physicalRows (interface : Interface) (offset : Nat) : List R1CS.Row :=
  (lowering interface offset).rows

def physicalRowCount (interface : Interface) (offset : Nat) : Nat :=
  (physicalRows interface offset).length

def physicalColumnCount (interface : Interface) (offset : Nat) : Nat :=
  (lowering interface offset).next

inductive RowRole where
  | directRecipe
  | multiplication (ordinal : Nat)
  | assertion
deriving Repr, DecidableEq

structure RowOwner where
  phase : Lifecycle.Phase
  logicalRow : Nat
  role : RowRole
deriving Repr, DecidableEq

def ownersFor (phase : Lifecycle.Phase) : Nat → List Expr → List RowOwner
  | _, [] => []
  | logicalRow, expression :: rest =>
      let current :=
        match R1CS.directConstraint expression with
        | some _ => [⟨phase, logicalRow, .directRecipe⟩]
        | none =>
            (List.range (R1CS.mulCount expression)).map (fun ordinal =>
              ⟨phase, logicalRow, .multiplication ordinal⟩) ++
            [⟨phase, logicalRow, .assertion⟩]
      current ++ ownersFor phase (logicalRow + 1) rest

def rowOwners (interface : Interface) (offset : Nat) : List RowOwner :=
  ownersFor .priorStateHash 0 (priorConstraints interface offset) ++
    ownersFor .outputHash 0 (outputConstraints interface offset)

def ownedRows (interface : Interface) (offset : Nat) :
    List (RowOwner × R1CS.Row) :=
  (rowOwners interface offset).zip (physicalRows interface offset)

inductive ColumnOwner where
  | external (index : Nat)
  | priorWitness (index : Nat)
  | outputWitness (index : Nat)
  | multiplication (index : Nat)
deriving Repr, DecidableEq

/-- Every physical column has exactly one owner. -/
def columnOwner (interface : Interface) (offset : Nat)
    (column : Fin (physicalColumnCount interface offset)) : ColumnOwner :=
  if beforePrior : column.val < offset then
    .external column.val
  else if beforeOutput : column.val < outputOffset interface offset then
    .priorWitness (column.val - offset)
  else if beforeMultiplication : column.val < logicalColumnCount interface offset then
    .outputWitness (column.val - outputOffset interface offset)
  else
    .multiplication (column.val - logicalColumnCount interface offset)

theorem ownersFor_length (phase : Lifecycle.Phase) (start : Nat)
    (constraints : List Expr) :
    (ownersFor phase start constraints).length =
      R1CS.totalRowCount constraints := by
  induction constraints generalizing start with
  | nil => rfl
  | cons expression rest ih =>
      cases result : R1CS.directConstraint expression with
      | none =>
          simp [ownersFor, R1CS.totalRowCount, R1CS.constraintRowCount,
            result, ih]
          omega
      | some direct =>
          simp [ownersFor, R1CS.totalRowCount, R1CS.constraintRowCount,
            result, ih]
          omega

theorem rowOwners_length (interface : Interface) (offset : Nat) :
    (rowOwners interface offset).length = physicalRowCount interface offset := by
  unfold rowOwners physicalRowCount physicalRows lowering logicalConstraints
  rw [List.length_append, ownersFor_length, ownersFor_length]
  rw [R1CS.lowerConstraints_rows_length]
  simp [R1CS.totalRowCount, List.map_append, List.sum_append]

theorem ownedRows_length (interface : Interface) (offset : Nat) :
    (ownedRows interface offset).length = physicalRowCount interface offset := by
  rw [ownedRows, List.length_zip, rowOwners_length]
  simp [physicalRowCount]

theorem physicalColumnCount_eq (interface : Interface) (offset : Nat) :
    physicalColumnCount interface offset =
      logicalColumnCount interface offset +
        R1CS.totalFreshCount (logicalConstraints interface offset) := by
  exact R1CS.lowerConstraints_next _ _

theorem physicalRowCount_eq (interface : Interface) (offset : Nat) :
    physicalRowCount interface offset =
      R1CS.totalRowCount (logicalConstraints interface offset) := by
  exact R1CS.lowerConstraints_rows_length _ _

def PhysicalHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows interface offset)

/-- Physical R1CS satisfaction preserves both logical pilot phase
specifications. -/
theorem physical_implies_spec (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Lifecycle.Pilot.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Lifecycle.Pilot.SpecHolds interface offset env := by
  have logical : ConstraintsHold env (logicalConstraints interface offset) :=
    R1CS.lowerConstraints_sound env (logicalConstraints interface offset)
      (logicalColumnCount interface offset) physical
  have priorFlat : holdsFlat env
      (Circuit.ops (Lifecycle.Pilot.priorCircuit interface).main offset) := by
    intro expression member
    exact logical expression (List.mem_append_left _ member)
  have outputFlat : holdsFlat env
      (Circuit.ops (Lifecycle.Pilot.outputCircuit interface).main
        (outputOffset interface offset)) := by
    intro expression member
    exact logical expression (List.mem_append_right _ member)
  exact Lifecycle.Pilot.phase_soundness interface offset env assumptions
    (holdsFlat_implies_holds env _ priorFlat)
    (holdsFlat_implies_holds env _ outputFlat)

end NightstreamFPrime.Layout.Pilot
