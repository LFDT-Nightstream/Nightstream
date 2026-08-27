import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.TranscriptAbsorption
import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestWindow
import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.First54
import NightstreamFPrime.Layout.R1CS.Segments

/-!
Owns physical composition for one complete PiRLC scalar sampler.

The parent order is one scalar-domain entry, eight digest windows, and the
exact first-54 selector. Child constraints remain opaque. This module sums
their certified physical footprints and lowers the unchanged sampler rows.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Sampler

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.Interface
abbrev circuit :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.circuit
abbrev main :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.main
abbrev opsAt :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.opsAt
abbrev entryInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.entryInterface
abbrev entryOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.entryOffset
abbrev entryCircuit :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.entryCircuit
abbrev windowInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowInterface
abbrev windowInitialState :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowInitialState
abbrev windowOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOffset
abbrev windowCircuit :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowCircuit
abbrev windowOps :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOps
abbrev windowOp :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
abbrev selectorInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.selectorInterface
abbrev selectorOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.selectorOffset
abbrev selectorCircuit :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.selectorCircuit
abbrev Assumptions :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.Assumptions
abbrev SpecHolds :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.SpecHolds
abbrev RelationHolds :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.RelationHolds
abbrev soundness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.soundness
abbrev completeness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.completeness
abbrev localLength_eq :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.localLength_eq
abbrev flatConstraints_varsBelow :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.flatConstraints_varsBelow
abbrev logicalPrivateCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.logicalPrivateCount
abbrev digestRoundCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.digestRoundCount

end Logical

/-- The sampler's only external expressions are its incoming transcript
state. Later child inputs are compiler-owned output variables. -/
structure InputsAffine (interface : Logical.Interface) (offset : Nat) : Prop where
  initialState : StateAffine (interface.initialState offset)

private def entryInputs (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    Leaves.TranscriptAbsorption.InputsAffine
      (Logical.entryInterface interface) (Logical.entryOffset offset) where
  initialState := by
    simpa [Logical.entryInterface, Logical.entryOffset] using inputs.initialState

private theorem entryOutput_fresh (interface : Logical.Interface)
    (coordinate offset : Nat) :
    StateFresh
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.output
        (Logical.entryInterface interface) coordinate
          (Logical.entryOffset offset)) := by
  unfold NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.output
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.ownedInterface
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.output
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.program
  change StateFresh
    (NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.compile
      (Logical.entryOffset offset) (interface.initialState offset)
      [.absorb
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.constantWords
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.frameWords
            coordinate))]).output
  apply compile_output_fresh_of_head_absorb
  intro empty
  have lengthZero := congrArg List.length empty
  simp [NightstreamFPrime.Gadgets.Poseidon2.Hash.inputChunks,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.constantWords,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.frameWords,
    NightstreamFPrime.Spec.Poseidon2.rate] at lengthZero

private theorem digestWindowOutput_fresh
    (interface : Logical.Interface) (coordinate offset round : Nat) :
    StateFresh
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.output
        (Logical.windowInterface interface coordinate offset round)
          (Logical.windowOffset offset round)) := by
  unfold NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.output
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.permutationInterface
    NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.output
    NightstreamFPrime.Gadgets.Poseidon2.Permutation.scheduleOutput
  exact ⟨_, rfl⟩

/-- The scalar sampler's outgoing transcript state is the fresh output of
its fixed last digest window. -/
theorem outputState_fresh (interface : Logical.Interface)
    (coordinate offset : Nat) :
    StateFresh
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.outputState
        interface coordinate offset) := by
  unfold NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.outputState
  exact digestWindowOutput_fresh interface coordinate offset
    (Logical.digestRoundCount - 1)

def windowInputs (interface : Logical.Interface)
    (coordinate offset round : Nat) :
    Leaves.DigestWindow.InputsAffine
      (Logical.windowInterface interface coordinate offset round)
        (Logical.windowOffset offset round) where
  initialState := by
    cases round with
    | zero =>
        simpa [Logical.windowInterface, Logical.windowInitialState] using
          (entryOutput_fresh interface coordinate offset).affine
    | succ previous =>
        simpa [Logical.windowInterface, Logical.windowInitialState] using
          (digestWindowOutput_fresh interface coordinate offset previous).affine

def childConstraints (child : FormalCircuit) (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops child.main offset)

private theorem childOp_flatConstraints (name : String)
    (child : FormalCircuit) (offset : Nat) :
    (NightstreamFPrime.Circuit.Sequence.childOp name child offset).flatConstraints =
      childConstraints child offset := by
  rfl

def appendAll : List (List Expr) → List Expr
  | [] => []
  | [constraints] => constraints
  | constraints :: next :: rest =>
      constraints ++ appendAll (next :: rest)

theorem appendAll_eq_flatten (lists : List (List Expr)) :
    appendAll lists = lists.flatten := by
  induction lists with
  | nil => rfl
  | cons first rest inductionHypothesis =>
      cases rest with
      | nil => simp [appendAll]
      | cons second tail =>
          simp only [appendAll, List.flatten_cons]
          simp only [inductionHypothesis, List.flatten_cons]

def childConstraintLists (interface : Logical.Interface)
    (coordinate offset : Nat) : List (List Expr) :=
  [childConstraints (Logical.entryCircuit interface coordinate)
      (Logical.entryOffset offset),
   childConstraints (Logical.windowCircuit interface coordinate offset 0)
      (Logical.windowOffset offset 0),
   childConstraints (Logical.windowCircuit interface coordinate offset 1)
      (Logical.windowOffset offset 1),
   childConstraints (Logical.windowCircuit interface coordinate offset 2)
      (Logical.windowOffset offset 2),
   childConstraints (Logical.windowCircuit interface coordinate offset 3)
      (Logical.windowOffset offset 3),
   childConstraints (Logical.windowCircuit interface coordinate offset 4)
      (Logical.windowOffset offset 4),
   childConstraints (Logical.windowCircuit interface coordinate offset 5)
      (Logical.windowOffset offset 5),
   childConstraints (Logical.windowCircuit interface coordinate offset 6)
      (Logical.windowOffset offset 6),
   childConstraints (Logical.windowCircuit interface coordinate offset 7)
      (Logical.windowOffset offset 7),
   childConstraints (Logical.selectorCircuit interface coordinate offset)
      (Logical.selectorOffset offset)]

private def physicalFreshDeltas (interface : Logical.Interface)
    (coordinate offset : Nat) : List Nat :=
  (childConstraintLists interface coordinate offset).map R1CS.totalFreshCount

private def physicalRowDeltas (interface : Logical.Interface)
    (coordinate offset : Nat) : List Nat :=
  (childConstraintLists interface coordinate offset).map R1CS.totalRowCount

def orderedConstraints (interface : Logical.Interface)
    (coordinate offset : Nat) : List Expr :=
  appendAll (childConstraintLists interface coordinate offset)

private theorem windowOps_eq (interface : Logical.Interface)
    (coordinate offset : Nat) :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOps
      interface coordinate offset =
      [NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
        interface coordinate offset 0,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
        interface coordinate offset 1,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
        interface coordinate offset 2,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
        interface coordinate offset 3,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
        interface coordinate offset 4,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
        interface coordinate offset 5,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
        interface coordinate offset 6,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
        interface coordinate offset 7] := by
  rfl

private theorem opsAt_eq (interface : Logical.Interface)
    (coordinate offset : Nat) :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.opsAt
        interface coordinate offset =
      [NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.entryOp
          interface coordinate offset,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
          interface coordinate offset 0,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
          interface coordinate offset 1,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
          interface coordinate offset 2,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
          interface coordinate offset 3,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
          interface coordinate offset 4,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
          interface coordinate offset 5,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
          interface coordinate offset 6,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp
          interface coordinate offset 7,
       NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.selectorOp
          interface coordinate offset] := by
  unfold NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.opsAt
  rw [windowOps_eq]
  rfl

def logicalConstraints (interface : Logical.Interface)
    (coordinate offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops (Logical.main interface coordinate) offset)

theorem logicalConstraints_eq_ordered (interface : Logical.Interface)
    (coordinate offset : Nat) :
    logicalConstraints interface coordinate offset =
      orderedConstraints interface coordinate offset := by
  unfold logicalConstraints
  change flatConstraints
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.opsAt
      interface coordinate offset) = _
  rw [opsAt_eq]
  unfold orderedConstraints childConstraintLists appendAll
  simp only [appendAll, flatConstraints, List.flatMap_cons, List.flatMap_nil,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.entryOp,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.windowOp,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.selectorOp,
    childOp_flatConstraints, List.append_nil]

theorem logicalConstraints_eq_flatten (interface : Logical.Interface)
    (coordinate offset : Nat) :
    logicalConstraints interface coordinate offset =
      (childConstraintLists interface coordinate offset).flatten := by
  rw [logicalConstraints_eq_ordered]
  exact appendAll_eq_flatten _

private theorem totalFreshCount_appendAll (lists : List (List Expr)) :
    R1CS.totalFreshCount (appendAll lists) =
      (lists.map R1CS.totalFreshCount).sum := by
  induction lists with
  | nil => rfl
  | cons first rest inductionHypothesis =>
      cases rest with
      | nil =>
          simp only [appendAll, List.map_cons, List.map_nil, List.sum_cons,
            List.sum_nil, Nat.add_zero]
      | cons second tail =>
          simp only [appendAll, R1CS.totalFreshCount_append, List.map_cons,
            List.sum_cons, inductionHypothesis]

private theorem totalRowCount_appendAll (lists : List (List Expr)) :
    R1CS.totalRowCount (appendAll lists) =
      (lists.map R1CS.totalRowCount).sum := by
  induction lists with
  | nil => rfl
  | cons first rest inductionHypothesis =>
      cases rest with
      | nil =>
          simp only [appendAll, List.map_cons, List.map_nil, List.sum_cons,
            List.sum_nil, Nat.add_zero]
      | cons second tail =>
          simp only [appendAll, R1CS.totalRowCount_append, List.map_cons,
            List.sum_cons, inductionHypothesis]

private theorem totalFreshCount_eq_deltas (interface : Logical.Interface)
    (coordinate offset : Nat) :
    R1CS.totalFreshCount (logicalConstraints interface coordinate offset) =
      (physicalFreshDeltas interface coordinate offset).sum := by
  rw [logicalConstraints_eq_ordered]
  change R1CS.totalFreshCount
      (appendAll (childConstraintLists interface coordinate offset)) =
    ((childConstraintLists interface coordinate offset).map
      R1CS.totalFreshCount).sum
  exact totalFreshCount_appendAll _

private theorem totalRowCount_eq_deltas (interface : Logical.Interface)
    (coordinate offset : Nat) :
    R1CS.totalRowCount (logicalConstraints interface coordinate offset) =
      (physicalRowDeltas interface coordinate offset).sum := by
  rw [logicalConstraints_eq_ordered]
  change R1CS.totalRowCount
      (appendAll (childConstraintLists interface coordinate offset)) =
    ((childConstraintLists interface coordinate offset).map
      R1CS.totalRowCount).sum
  exact totalRowCount_appendAll _

private theorem entryFreshCount_eq (interface : Logical.Interface)
    (coordinate offset : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    R1CS.totalFreshCount
      (childConstraints (Logical.entryCircuit interface coordinate)
        (Logical.entryOffset offset)) = 0 := by
  change R1CS.totalFreshCount (flatConstraints (Circuit.ops
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.circuit
      (Logical.entryInterface interface) coordinate).main
        (Logical.entryOffset offset))) = 0
  exact Leaves.TranscriptAbsorption.freshColumnCount_eq
    (Logical.entryInterface interface) coordinate
    (fun current => entryInputs interface current (inputs current))
    (Logical.entryOffset offset)

private theorem entryRowCount_eq (interface : Logical.Interface)
    (coordinate offset : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    R1CS.totalRowCount
      (childConstraints (Logical.entryCircuit interface coordinate)
        (Logical.entryOffset offset)) = 592 := by
  change R1CS.totalRowCount (flatConstraints (Circuit.ops
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.circuit
      (Logical.entryInterface interface) coordinate).main
        (Logical.entryOffset offset))) = 592
  exact Leaves.TranscriptAbsorption.physicalRowCount_eq
    (Logical.entryInterface interface) coordinate
    (fun current => entryInputs interface current (inputs current))
    (Logical.entryOffset offset)

private theorem windowFreshCount_eq (interface : Logical.Interface)
    (coordinate offset round : Nat) :
    R1CS.totalFreshCount
      (childConstraints
        (Logical.windowCircuit interface coordinate offset round)
          (Logical.windowOffset offset round)) = 1212 := by
  change R1CS.totalFreshCount
    (Leaves.DigestWindow.logicalConstraints
      (Logical.windowInterface interface coordinate offset round)
        (Logical.windowOffset offset round)) = 1212
  exact Leaves.DigestWindow.totalFreshCount_eq _ _
    (windowInputs interface coordinate offset round)

private theorem windowRowCount_eq (interface : Logical.Interface)
    (coordinate offset round : Nat) :
    R1CS.totalRowCount
      (childConstraints
        (Logical.windowCircuit interface coordinate offset round)
          (Logical.windowOffset offset round)) = 2216 := by
  change R1CS.totalRowCount
    (Leaves.DigestWindow.logicalConstraints
      (Logical.windowInterface interface coordinate offset round)
        (Logical.windowOffset offset round)) = 2216
  exact Leaves.DigestWindow.totalRowCount_eq _ _
    (windowInputs interface coordinate offset round)

private theorem selectorFreshCount_eq (interface : Logical.Interface)
    (coordinate offset : Nat) :
    R1CS.totalFreshCount
      (childConstraints (Logical.selectorCircuit interface coordinate offset)
        (Logical.selectorOffset offset)) = 34047 := by
  unfold childConstraints
  exact Leaves.First54.selectorCircuit_totalFreshCount_eq
    interface coordinate offset
    (Logical.selectorOffset offset)

private theorem selectorRowCount_eq (interface : Logical.Interface)
    (coordinate offset : Nat) :
    R1CS.totalRowCount
      (childConstraints (Logical.selectorCircuit interface coordinate offset)
        (Logical.selectorOffset offset)) = 41024 := by
  unfold childConstraints
  exact Leaves.First54.selectorCircuit_totalRowCount_eq
    interface coordinate offset
    (Logical.selectorOffset offset)

private theorem physicalFreshDeltas_eq (interface : Logical.Interface)
    (coordinate offset : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    physicalFreshDeltas interface coordinate offset =
      [0, 1212, 1212, 1212, 1212, 1212, 1212, 1212, 1212, 34047] := by
  unfold physicalFreshDeltas childConstraintLists
  simp only [List.map_cons, List.map_nil]
  rw [entryFreshCount_eq interface coordinate offset inputs,
    windowFreshCount_eq interface coordinate offset 0,
    windowFreshCount_eq interface coordinate offset 1,
    windowFreshCount_eq interface coordinate offset 2,
    windowFreshCount_eq interface coordinate offset 3,
    windowFreshCount_eq interface coordinate offset 4,
    windowFreshCount_eq interface coordinate offset 5,
    windowFreshCount_eq interface coordinate offset 6,
    windowFreshCount_eq interface coordinate offset 7,
    selectorFreshCount_eq interface coordinate offset]

private theorem physicalRowDeltas_eq (interface : Logical.Interface)
    (coordinate offset : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    physicalRowDeltas interface coordinate offset =
      [592, 2216, 2216, 2216, 2216, 2216, 2216, 2216, 2216, 41024] := by
  unfold physicalRowDeltas childConstraintLists
  simp only [List.map_cons, List.map_nil]
  rw [entryRowCount_eq interface coordinate offset inputs,
    windowRowCount_eq interface coordinate offset 0,
    windowRowCount_eq interface coordinate offset 1,
    windowRowCount_eq interface coordinate offset 2,
    windowRowCount_eq interface coordinate offset 3,
    windowRowCount_eq interface coordinate offset 4,
    windowRowCount_eq interface coordinate offset 5,
    windowRowCount_eq interface coordinate offset 6,
    windowRowCount_eq interface coordinate offset 7,
    selectorRowCount_eq interface coordinate offset]

private theorem physicalFreshDeltaSum_eq :
    [0, 1212, 1212, 1212, 1212, 1212, 1212, 1212, 1212, 34047].sum =
      43743 := by
  norm_num

private theorem physicalRowDeltaSum_eq :
    [592, 2216, 2216, 2216, 2216, 2216, 2216, 2216, 2216, 41024].sum =
      59344 := by
  norm_num

theorem totalFreshCount_eq (interface : Logical.Interface)
    (coordinate offset : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    R1CS.totalFreshCount (logicalConstraints interface coordinate offset) =
      43743 := by
  calc
    _ = (physicalFreshDeltas interface coordinate offset).sum :=
      totalFreshCount_eq_deltas interface coordinate offset
    _ = [0, 1212, 1212, 1212, 1212, 1212, 1212, 1212, 1212,
          34047].sum :=
      congrArg List.sum
        (physicalFreshDeltas_eq interface coordinate offset inputs)
    _ = 43743 := physicalFreshDeltaSum_eq

theorem totalRowCount_eq (interface : Logical.Interface)
    (coordinate offset : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    R1CS.totalRowCount (logicalConstraints interface coordinate offset) =
      59344 := by
  calc
    _ = (physicalRowDeltas interface coordinate offset).sum :=
      totalRowCount_eq_deltas interface coordinate offset
    _ = [592, 2216, 2216, 2216, 2216, 2216, 2216, 2216, 2216,
          41024].sum :=
      congrArg List.sum
        (physicalRowDeltas_eq interface coordinate offset inputs)
    _ = 59344 := physicalRowDeltaSum_eq

def footprint (interface : Logical.Interface) (coordinate : Nat)
    (inputs : ∀ offset, InputsAffine interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface coordinate) where
  freshColumnCount := fun _ => 43743
  physicalRowCount := fun _ => 59344
  freshColumnCount_eq := fun offset =>
    totalFreshCount_eq interface coordinate offset inputs
  physicalRowCount_eq := fun offset =>
    totalRowCount_eq interface coordinate offset inputs

theorem physicalPrivateColumnCount_eq (interface : Logical.Interface)
    (coordinate offset : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    localLength (Circuit.ops (Logical.circuit interface coordinate).main offset) +
      R1CS.totalFreshCount
        (logicalConstraints interface coordinate offset) = 59247 := by
  change localLength (Circuit.ops (Logical.main interface coordinate) offset) +
      R1CS.totalFreshCount
        (logicalConstraints interface coordinate offset) = 59247
  rw [Logical.localLength_eq, totalFreshCount_eq interface coordinate offset inputs]
  rfl

def plan (interface : Logical.Interface) (coordinate offset : Nat) :
    R1CS.LoweringPlan where
  constraints := logicalConstraints interface coordinate offset
  firstFresh := offset + Logical.logicalPrivateCount

@[simp] theorem plan_constraints_eq_flatten (interface : Logical.Interface)
    (coordinate offset : Nat) :
    (plan interface coordinate offset).constraints =
      (childConstraintLists interface coordinate offset).flatten := by
  exact logicalConstraints_eq_flatten interface coordinate offset

@[simp] theorem plan_firstFresh (interface : Logical.Interface)
    (coordinate offset : Nat) :
    (plan interface coordinate offset).firstFresh =
      offset + Logical.logicalPrivateCount := by
  rfl

def PhysicalHolds (interface : Logical.Interface) (coordinate offset : Nat)
    (env : Env) : Prop :=
  R1CS.SegmentsHold env (childConstraintLists interface coordinate offset)
    (offset + Logical.logicalPrivateCount)

/-- The segmented sampler predicate is exactly the canonical monolithic
lowering used by the package and backend. -/
theorem physicalHolds_iff_rowsHold (interface : Logical.Interface)
    (coordinate offset : Nat) (env : Env) :
    PhysicalHolds interface coordinate offset env ↔
      R1CS.RowsHold env (plan interface coordinate offset).rows := by
  have exactRows :=
    R1CS.LoweringPlan.rowsHold_iff_segments_of_constraints
      (plan interface coordinate offset) env
      (childConstraintLists interface coordinate offset)
      (plan_constraints_eq_flatten interface coordinate offset)
  simpa only [PhysicalHolds, plan_firstFresh] using exactRows.symm

theorem physical_implies_relation (interface : Logical.Interface)
    (coordinate offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface coordinate offset env) :
    Logical.RelationHolds interface coordinate offset env := by
  apply NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.rows_imply_relation
    interface coordinate offset env assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints interface coordinate offset)
  exact R1CS.LoweringPlan.sound (plan interface coordinate offset) env
    ((physicalHolds_iff_rowsHold interface coordinate offset env).mp physical)

end NightstreamFPrime.Layout.PiRLC.v1_1.Sampler
