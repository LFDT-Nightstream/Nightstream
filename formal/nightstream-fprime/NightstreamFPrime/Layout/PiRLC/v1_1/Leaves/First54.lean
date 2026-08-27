import NightstreamFPrime.Layout.R1CS.Segments
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler

/-!
Owns physical lowering for the exact first-54 selector used by one PiRLC
scalar sampler. Round zero has constant prior position/output vectors; later
rounds read the preceding selector variables. The separate cost theorems keep
that physical distinction explicit.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.First54

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.Interface
abbrev selectorInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.selectorInterface
abbrev selectorOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.selectorOffset
abbrev positionCircuit :=
  NightstreamFPrime.Gadgets.Sampling.First54.positionCircuit
abbrev positionInterface :=
  NightstreamFPrime.Gadgets.Sampling.First54.positionInterface
abbrev positionOffset :=
  NightstreamFPrime.Gadgets.Sampling.First54.positionOffset
abbrev valueCircuit :=
  NightstreamFPrime.Gadgets.Sampling.First54.valueCircuit
abbrev valueInterface :=
  NightstreamFPrime.Gadgets.Sampling.First54.valueInterface
abbrev valueOffset :=
  NightstreamFPrime.Gadgets.Sampling.First54.valueOffset
abbrev selectorCircuit :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.selectorCircuit
abbrev Assumptions :=
  NightstreamFPrime.Gadgets.Sampling.First54.Assumptions
abbrev RelationHolds :=
  NightstreamFPrime.Gadgets.Sampling.First54.RelationHolds
abbrev logicalPrivateCount :=
  NightstreamFPrime.Gadgets.Sampling.First54.logicalPrivateCount

end Logical

def selectorInterface (interface : Logical.Interface)
    (coordinate parentOffset : Nat) :=
  Logical.selectorInterface interface coordinate parentOffset

def selectorOffset (parentOffset : Nat) : Nat :=
  Logical.selectorOffset parentOffset

def positionConstraints (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) : List Expr :=
  flatConstraints (Circuit.ops
    (Logical.positionCircuit
      (selectorInterface interface coordinate parentOffset)
        offset round).main
    (Logical.positionOffset offset round))

def valueConstraints (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) : List Expr :=
  flatConstraints (Circuit.ops
    (Logical.valueCircuit
      (selectorInterface interface coordinate parentOffset)
        offset round).main
    (Logical.valueOffset offset round))

def positionConstraint (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat)
    (slot : Fin NightstreamFPrime.Gadgets.Sampling.First54Step.slotCount) :
    Expr :=
  let start := Logical.positionOffset offset round
  Expr.var (start + slot.val) -
    NightstreamFPrime.Gadgets.Sampling.First54Step.recipe
      (Logical.positionInterface
        (selectorInterface interface coordinate parentOffset)
          offset round) start slot

def valueConstraint (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat)
    (slot : Fin NightstreamFPrime.Gadgets.Sampling.First54ValueStep.outputCount) :
    Expr :=
  let start := Logical.valueOffset offset round
  Expr.var (start + slot.val) -
    NightstreamFPrime.Gadgets.Sampling.First54ValueStep.recipe
      (Logical.valueInterface
        (selectorInterface interface coordinate parentOffset)
          offset round) start slot

private theorem recipeConstraints_ofFn {count : Nat} (start : Nat)
    (recipes : Fin count → Expr) :
    recipeConstraints start (List.ofFn recipes) =
      List.ofFn fun slot => Expr.var (start + slot.val) - recipes slot := by
  induction count generalizing start with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [List.ofFn_succ, recipeConstraints]
      apply congrArg₂ List.cons
      · simp
      · rw [inductionHypothesis]
        apply congrArg List.ofFn
        funext slot
        congr 2
        simp only [Fin.val_succ]
        omega

theorem positionConstraints_eq (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) :
    positionConstraints interface coordinate parentOffset offset round =
      List.ofFn
        (positionConstraint interface coordinate parentOffset offset round) := by
  unfold positionConstraints positionConstraint
  change flatConstraints
      (NightstreamFPrime.Gadgets.Sampling.First54Step.operations
        (Logical.positionInterface
          (selectorInterface interface coordinate parentOffset)
            offset round)
        (Logical.positionOffset offset round)) = _
  rw [NightstreamFPrime.Gadgets.Sampling.First54Step.flatConstraints_operations]
  unfold NightstreamFPrime.Gadgets.Sampling.First54Step.recipes
  exact recipeConstraints_ofFn _ _

theorem valueConstraints_eq (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) :
    valueConstraints interface coordinate parentOffset offset round =
      List.ofFn
        (valueConstraint interface coordinate parentOffset offset round) := by
  unfold valueConstraints valueConstraint
  change flatConstraints
      (NightstreamFPrime.Gadgets.Sampling.First54ValueStep.operations
        (Logical.valueInterface
          (selectorInterface interface coordinate parentOffset)
            offset round)
        (Logical.valueOffset offset round)) = _
  rw [NightstreamFPrime.Gadgets.Sampling.First54ValueStep.flatConstraints_operations]
  unfold NightstreamFPrime.Gadgets.Sampling.First54ValueStep.recipes
  exact recipeConstraints_ofFn _ _

def runningPositionFresh
    (slot : Fin NightstreamFPrime.Gadgets.Sampling.First54Step.slotCount) :
    Nat :=
  if slot.val = 0 then 0 else if slot.val = 54 then 3 else 6

def runningPositionRows
    (slot : Fin NightstreamFPrime.Gadgets.Sampling.First54Step.slotCount) :
    Nat :=
  if slot.val = 0 then 1 else if slot.val = 54 then 4 else 7

theorem positionZero_cost (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat)
    (slot : Fin NightstreamFPrime.Gadgets.Sampling.First54Step.slotCount) :
    R1CS.constraintFreshCount
        (positionConstraint interface coordinate parentOffset offset 0 slot) = 0 ∧
      R1CS.constraintRowCount
        (positionConstraint interface coordinate parentOffset offset 0 slot) = 1 := by
  fin_cases slot <;> constructor <;> rfl

theorem positionSucc_cost (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat)
    (slot : Fin NightstreamFPrime.Gadgets.Sampling.First54Step.slotCount) :
    R1CS.constraintFreshCount
        (positionConstraint interface coordinate parentOffset offset (round + 1)
          slot) = runningPositionFresh slot ∧
      R1CS.constraintRowCount
        (positionConstraint interface coordinate parentOffset offset (round + 1)
          slot) = runningPositionRows slot := by
  fin_cases slot <;> constructor <;> rfl

theorem valueZero_cost (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat)
    (slot : Fin NightstreamFPrime.Gadgets.Sampling.First54ValueStep.outputCount) :
    R1CS.constraintFreshCount
        (valueConstraint interface coordinate parentOffset offset 0 slot) = 4 ∧
      R1CS.constraintRowCount
        (valueConstraint interface coordinate parentOffset offset 0 slot) = 5 := by
  fin_cases slot <;> constructor <;> rfl

theorem valueSucc_cost (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat)
    (slot : Fin NightstreamFPrime.Gadgets.Sampling.First54ValueStep.outputCount) :
    R1CS.constraintFreshCount
        (valueConstraint interface coordinate parentOffset offset (round + 1)
          slot) = 4 ∧
      R1CS.constraintRowCount
        (valueConstraint interface coordinate parentOffset offset (round + 1)
          slot) = 5 := by
  fin_cases slot <;> constructor <;> rfl

private theorem runningPositionFresh_sum :
    (List.ofFn runningPositionFresh).sum = 321 := by
  rfl

private theorem runningPositionRows_sum :
    (List.ofFn runningPositionRows).sum = 376 := by
  rfl

private theorem totalFreshCount_ofFn {count : Nat}
    (constraints : Fin count → Expr) (cost : Fin count → Nat)
    (pointwise : ∀ slot,
      R1CS.constraintFreshCount (constraints slot) = cost slot) :
    R1CS.totalFreshCount (List.ofFn constraints) =
      (List.ofFn cost).sum := by
  unfold R1CS.totalFreshCount
  rw [List.map_ofFn]
  apply congrArg List.sum
  apply congrArg List.ofFn
  funext slot
  exact pointwise slot

private theorem totalRowCount_ofFn {count : Nat}
    (constraints : Fin count → Expr) (cost : Fin count → Nat)
    (pointwise : ∀ slot,
      R1CS.constraintRowCount (constraints slot) = cost slot) :
    R1CS.totalRowCount (List.ofFn constraints) =
      (List.ofFn cost).sum := by
  unfold R1CS.totalRowCount
  rw [List.map_ofFn]
  apply congrArg List.sum
  apply congrArg List.ofFn
  funext slot
  exact pointwise slot

theorem positionZero_totalFreshCount (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalFreshCount
      (positionConstraints interface coordinate parentOffset offset 0) = 0 := by
  rw [positionConstraints_eq]
  rw [totalFreshCount_ofFn _ (fun _ => 0)
    (fun slot =>
      (positionZero_cost interface coordinate parentOffset offset slot).1)]
  rfl

private theorem positionZero_totalRowCount (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalRowCount
      (positionConstraints interface coordinate parentOffset offset 0) = 55 := by
  rw [positionConstraints_eq]
  rw [totalRowCount_ofFn _ (fun _ => 1)
    (fun slot =>
      (positionZero_cost interface coordinate parentOffset offset slot).2)]
  rfl

theorem positionSucc_totalFreshCount (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) :
    R1CS.totalFreshCount
      (positionConstraints interface coordinate parentOffset offset (round + 1)) =
        321 := by
  rw [positionConstraints_eq]
  rw [totalFreshCount_ofFn _ runningPositionFresh
    (fun slot =>
      (positionSucc_cost interface coordinate parentOffset offset round slot).1)]
  exact runningPositionFresh_sum

private theorem positionSucc_totalRowCount (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) :
    R1CS.totalRowCount
      (positionConstraints interface coordinate parentOffset offset (round + 1)) =
        376 := by
  rw [positionConstraints_eq]
  rw [totalRowCount_ofFn _ runningPositionRows
    (fun slot =>
      (positionSucc_cost interface coordinate parentOffset offset round slot).2)]
  exact runningPositionRows_sum

theorem valueZero_totalFreshCount (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalFreshCount
      (valueConstraints interface coordinate parentOffset offset 0) = 216 := by
  rw [valueConstraints_eq]
  rw [totalFreshCount_ofFn _ (fun _ => 4)
    (fun slot =>
      (valueZero_cost interface coordinate parentOffset offset slot).1)]
  rfl

private theorem valueZero_totalRowCount (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalRowCount
      (valueConstraints interface coordinate parentOffset offset 0) = 270 := by
  rw [valueConstraints_eq]
  rw [totalRowCount_ofFn _ (fun _ => 5)
    (fun slot =>
      (valueZero_cost interface coordinate parentOffset offset slot).2)]
  rfl

theorem valueSucc_totalFreshCount (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) :
    R1CS.totalFreshCount
      (valueConstraints interface coordinate parentOffset offset (round + 1)) =
        216 := by
  rw [valueConstraints_eq]
  rw [totalFreshCount_ofFn _ (fun _ => 4)
    (fun slot =>
      (valueSucc_cost interface coordinate parentOffset offset round slot).1)]
  rfl

private theorem valueSucc_totalRowCount (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) :
    R1CS.totalRowCount
      (valueConstraints interface coordinate parentOffset offset (round + 1)) =
        270 := by
  rw [valueConstraints_eq]
  rw [totalRowCount_ofFn _ (fun _ => 5)
    (fun slot =>
      (valueSucc_cost interface coordinate parentOffset offset round slot).2)]
  rfl

def roundConstraints (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) : List Expr :=
  positionConstraints interface coordinate parentOffset offset round ++
    valueConstraints interface coordinate parentOffset offset round

theorem flatConstraints_roundOps (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) :
    flatConstraints
      (NightstreamFPrime.Gadgets.Sampling.First54.roundOps
        (selectorInterface interface coordinate parentOffset)
          offset round) =
      roundConstraints interface coordinate parentOffset offset round := by
  unfold NightstreamFPrime.Gadgets.Sampling.First54.roundOps
    NightstreamFPrime.Gadgets.Sampling.First54.positionOp
    NightstreamFPrime.Gadgets.Sampling.First54.valueOp roundConstraints
  simp only [flatConstraints, List.flatMap_cons, List.flatMap_nil,
    NightstreamFPrime.Circuit.Sequence.childOp, Op.flatConstraints,
    FormalCircuit.asSubcircuit_constraints, List.append_nil]
  rfl

theorem roundZero_totalFreshCount (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalFreshCount
      (flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOps
          (selectorInterface interface coordinate parentOffset)
            offset 0)) = 216 := by
  rw [flatConstraints_roundOps]
  unfold roundConstraints
  rw [R1CS.totalFreshCount_append, positionZero_totalFreshCount,
    valueZero_totalFreshCount]

private theorem roundZero_totalRowCount (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalRowCount
      (flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOps
          (selectorInterface interface coordinate parentOffset)
            offset 0)) = 325 := by
  rw [flatConstraints_roundOps]
  unfold roundConstraints
  rw [R1CS.totalRowCount_append, positionZero_totalRowCount,
    valueZero_totalRowCount]

theorem roundSucc_totalFreshCount (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) :
    R1CS.totalFreshCount
      (flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOps
          (selectorInterface interface coordinate parentOffset)
            offset (round + 1))) = 537 := by
  rw [flatConstraints_roundOps]
  unfold roundConstraints
  rw [R1CS.totalFreshCount_append, positionSucc_totalFreshCount,
    valueSucc_totalFreshCount]

private theorem roundSucc_totalRowCount (interface : Logical.Interface)
    (coordinate parentOffset offset round : Nat) :
    R1CS.totalRowCount
      (flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOps
          (selectorInterface interface coordinate parentOffset)
            offset (round + 1))) = 646 := by
  rw [flatConstraints_roundOps]
  unfold roundConstraints
  rw [R1CS.totalRowCount_append, positionSucc_totalRowCount,
    valueSucc_totalRowCount]

theorem roundOpsPrefix_succ (interface : Logical.Interface)
    (coordinate parentOffset offset count : Nat) :
    NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
        (selectorInterface interface coordinate parentOffset)
          offset (count + 1) =
      NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
          (selectorInterface interface coordinate parentOffset)
            offset count ++
        NightstreamFPrime.Gadgets.Sampling.First54.roundOps
          (selectorInterface interface coordinate parentOffset)
            offset count := by
  simp [NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix,
    List.range_succ]

def prefixFreshCount : Nat → Nat
  | 0 => 0
  | previous + 1 => 216 + previous * 537

private def prefixRowCount : Nat → Nat
  | 0 => 0
  | previous + 1 => 325 + previous * 646

theorem roundOpsPrefix_totalFreshCount
    (interface : Logical.Interface)
    (coordinate parentOffset offset count : Nat) :
    R1CS.totalFreshCount
      (flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
          (selectorInterface interface coordinate parentOffset)
            offset count)) =
      prefixFreshCount count := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [roundOpsPrefix_succ, flatConstraints_append,
        R1CS.totalFreshCount_append, inductionHypothesis]
      cases count with
      | zero =>
          simpa [prefixFreshCount] using
            roundZero_totalFreshCount interface coordinate parentOffset offset
      | succ previous =>
          rw [roundSucc_totalFreshCount]
          simp only [prefixFreshCount]
          omega

private theorem roundOpsPrefix_totalRowCount
    (interface : Logical.Interface)
    (coordinate parentOffset offset count : Nat) :
    R1CS.totalRowCount
      (flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
          (selectorInterface interface coordinate parentOffset)
            offset count)) =
      prefixRowCount count := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [roundOpsPrefix_succ, flatConstraints_append,
        R1CS.totalRowCount_append, inductionHypothesis]
      cases count with
      | zero =>
          simpa [prefixRowCount] using
            roundZero_totalRowCount interface coordinate parentOffset offset
      | succ previous =>
          rw [roundSucc_totalRowCount]
          simp only [prefixRowCount]
          omega

def logicalConstraints (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops
    (NightstreamFPrime.Gadgets.Sampling.First54.main
      (selectorInterface interface coordinate parentOffset)) offset)

private theorem logicalConstraints_eq (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    logicalConstraints interface coordinate parentOffset offset =
      flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
          (selectorInterface interface coordinate parentOffset)
            offset
              NightstreamFPrime.Gadgets.Sampling.First54.candidateCount) ++
      [NightstreamFPrime.Gadgets.Sampling.First54.finalFull
          offset - 1] := by
  unfold logicalConstraints
  change flatConstraints
      (NightstreamFPrime.Gadgets.Sampling.First54.opsAt
        (selectorInterface interface coordinate parentOffset)
          offset) = _
  rw [NightstreamFPrime.Gadgets.Sampling.First54.opsAt,
    flatConstraints_append, flatConstraints_singleton]
  rfl

/-- The selector constraint list is its exact 64-round prefix followed by
the fail-closed final-full assertion. -/
theorem logicalConstraints_eq_rounds_append_final
    (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    logicalConstraints interface coordinate parentOffset offset =
      flatConstraints
        (NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
          (selectorInterface interface coordinate parentOffset)
            offset NightstreamFPrime.Gadgets.Sampling.First54.candidateCount) ++
      [NightstreamFPrime.Gadgets.Sampling.First54.finalFull offset - 1] :=
  logicalConstraints_eq interface coordinate parentOffset offset

def constraintSegments (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) : List (List Expr) :=
  [flatConstraints
      (NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
        (selectorInterface interface coordinate parentOffset)
          offset NightstreamFPrime.Gadgets.Sampling.First54.candidateCount),
   [NightstreamFPrime.Gadgets.Sampling.First54.finalFull offset - 1]]

/-- The two opaque selector segments flatten to the unchanged canonical
64-round prefix followed by the final fail-closed assertion. -/
theorem logicalConstraints_eq_segments_flatten
    (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    logicalConstraints interface coordinate parentOffset offset =
      (constraintSegments interface coordinate parentOffset offset).flatten := by
  rw [logicalConstraints_eq]
  simp only [constraintSegments, List.flatten_cons, List.flatten_nil,
    List.append_nil]

theorem totalFreshCount_eq (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalFreshCount
      (logicalConstraints interface coordinate parentOffset offset) = 34047 := by
  rw [logicalConstraints_eq, R1CS.totalFreshCount_append,
    roundOpsPrefix_totalFreshCount]
  rfl

theorem totalRowCount_eq (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalRowCount
      (logicalConstraints interface coordinate parentOffset offset) = 41024 := by
  rw [logicalConstraints_eq, R1CS.totalRowCount_append,
    roundOpsPrefix_totalRowCount]
  rfl

set_option maxRecDepth 100000 in -- fixed-size: one 64-candidate selector
def footprint (interface : Logical.Interface) (coordinate parentOffset : Nat) :
    R1CS.CircuitFootprint
      (Logical.selectorCircuit interface coordinate parentOffset) where
  freshColumnCount := fun _ => 34047
  physicalRowCount := fun _ => 41024
  freshColumnCount_eq := by
    intro offset
    change R1CS.totalFreshCount
      (logicalConstraints interface coordinate parentOffset offset) = 34047
    exact totalFreshCount_eq interface coordinate parentOffset offset
  physicalRowCount_eq := by
    intro offset
    change R1CS.totalRowCount
      (logicalConstraints interface coordinate parentOffset offset) = 41024
    exact totalRowCount_eq interface coordinate parentOffset offset

/-- Opaque exact-target bridge for parents that compose this child circuit.
It prevents a parent count proof from reducing the complete selector list. -/
theorem selectorCircuit_totalFreshCount_eq (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalFreshCount
      (flatConstraints (Circuit.ops
        (Logical.selectorCircuit interface coordinate parentOffset).main
          offset)) = 34047 := by
  exact (footprint interface coordinate parentOffset).freshColumnCount_eq offset

/-- Opaque exact-target row-count bridge for parent composition. -/
theorem selectorCircuit_totalRowCount_eq (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.totalRowCount
      (flatConstraints (Circuit.ops
        (Logical.selectorCircuit interface coordinate parentOffset).main
          offset)) = 41024 := by
  exact (footprint interface coordinate parentOffset).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    localLength (Circuit.ops
        (Logical.selectorCircuit interface coordinate parentOffset).main
          offset) +
      R1CS.totalFreshCount
        (logicalConstraints interface coordinate parentOffset offset) =
      41023 := by
  change localLength
      (NightstreamFPrime.Gadgets.Sampling.First54.opsAt
        (selectorInterface interface coordinate parentOffset)
          offset) +
      R1CS.totalFreshCount
        (logicalConstraints interface coordinate parentOffset offset) = 41023
  have lengthEq : localLength
      (NightstreamFPrime.Gadgets.Sampling.First54.opsAt
        (selectorInterface interface coordinate parentOffset) offset) =
      Logical.logicalPrivateCount := by
    simpa using NightstreamFPrime.Gadgets.Sampling.First54.localLength_eq
      (selectorInterface interface coordinate parentOffset) offset
  rw [lengthEq, totalFreshCount_eq]
  rfl

def plan (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    R1CS.LoweringPlan where
  constraints := logicalConstraints interface coordinate parentOffset offset
  firstFresh := offset + Logical.logicalPrivateCount

@[simp] theorem plan_constraints_eq_segments_flatten
    (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    (plan interface coordinate parentOffset offset).constraints =
      (constraintSegments interface coordinate parentOffset offset).flatten := by
  exact logicalConstraints_eq_segments_flatten interface coordinate
    parentOffset offset

@[simp] theorem plan_firstFresh (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) :
    (plan interface coordinate parentOffset offset).firstFresh =
      offset + Logical.logicalPrivateCount := by
  rfl

def PhysicalHolds (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) (env : Env) : Prop :=
  R1CS.SegmentsHold env
    (constraintSegments interface coordinate parentOffset offset)
    (offset + Logical.logicalPrivateCount)

/-- Segment satisfaction is exactly satisfaction of the canonical selector
lowering used by the sampler and package. -/
theorem physicalHolds_iff_rowsHold (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) (env : Env) :
    PhysicalHolds interface coordinate parentOffset offset env ↔
      R1CS.RowsHold env
        (plan interface coordinate parentOffset offset).rows := by
  have exactRows :=
    R1CS.LoweringPlan.rowsHold_iff_segments_of_constraints
      (plan interface coordinate parentOffset offset) env
      (constraintSegments interface coordinate parentOffset offset)
      (plan_constraints_eq_segments_flatten interface coordinate
        parentOffset offset)
  simpa only [PhysicalHolds, plan_firstFresh] using exactRows.symm

theorem physical_implies_relation (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions
      (selectorInterface interface coordinate parentOffset)
        offset env)
    (physical : PhysicalHolds interface coordinate parentOffset offset env) :
    Logical.RelationHolds
      (selectorInterface interface coordinate parentOffset)
        offset env := by
  apply NightstreamFPrime.Gadgets.Sampling.First54.rows_imply_boundedSample
    (selectorInterface interface coordinate parentOffset)
      offset env assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env
    (logicalConstraints interface coordinate parentOffset offset)
  exact R1CS.LoweringPlan.sound
    (plan interface coordinate parentOffset offset) env
      ((physicalHolds_iff_rowsHold interface coordinate parentOffset offset
        env).mp physical)

theorem physical_complete (interface : Logical.Interface)
    (coordinate parentOffset offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions
      (selectorInterface interface coordinate parentOffset)
        offset env)
    (relation : Logical.RelationHolds
      (selectorInterface interface coordinate parentOffset)
        offset env) :
    ∃ completed,
      AgreesOutside env completed offset 41023 ∧
      PhysicalHolds interface coordinate parentOffset offset completed := by
  rcases (Logical.selectorCircuit interface coordinate parentOffset).completeness
      env offset assumptions relation with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed :
      AgreesOutside env logicalEnv offset
        Logical.logicalPrivateCount := by
    change AgreesOutside env logicalEnv offset
        (localLength (Circuit.ops
        (NightstreamFPrime.Gadgets.Sampling.First54.main
          (selectorInterface interface coordinate parentOffset)) offset)) at logicalAgrees
    rw [NightstreamFPrime.Gadgets.Sampling.First54.localLength_eq] at logicalAgrees
    exact logicalAgrees
  have logicalAssumptions : Logical.Assumptions
      (selectorInterface interface coordinate parentOffset)
        offset logicalEnv := by
    constructor
    · exact assumptions.acceptedBelow
    · exact assumptions.symbolBelow
    · intro candidate
      have evaluation := Expr.eval_eq_of_agree_below
        ((selectorInterface interface coordinate parentOffset).accepted
          offset candidate)
        offset logicalEnv env
        (assumptions.acceptedBelow candidate) (by
          intro index below
          exact logicalAgreesFixed index (Or.inl below))
      rw [evaluation]
      exact assumptions.acceptedBoolean candidate
  have scope : ∀ expression ∈
      logicalConstraints interface coordinate parentOffset offset,
      expression.VarsBelow
        (offset + Logical.logicalPrivateCount) := by
    have logicalScope :=
      NightstreamFPrime.Gadgets.Sampling.First54.flatConstraints_varsBelow
      (selectorInterface interface coordinate parentOffset)
        offset logicalEnv logicalAssumptions
    rw [NightstreamFPrime.Gadgets.Sampling.First54.localLength_eq] at logicalScope
    exact logicalScope
  have logicalConstraintsHold : ConstraintsHold logicalEnv
      (logicalConstraints interface coordinate parentOffset offset) := by
    change ConstraintsHold logicalEnv
      (flatConstraints (Circuit.ops
        (NightstreamFPrime.Gadgets.Sampling.First54.main
          (selectorInterface interface coordinate parentOffset)) offset))
    exact logicalRows
  have segmentScope : ∀ expression ∈
      (constraintSegments interface coordinate parentOffset offset).flatten,
      expression.VarsBelow (offset + Logical.logicalPrivateCount) := by
    rw [← logicalConstraints_eq_segments_flatten]
    exact scope
  have segmentLogical : ConstraintsHold logicalEnv
      (constraintSegments interface coordinate parentOffset offset).flatten := by
    rw [← logicalConstraints_eq_segments_flatten]
    exact logicalConstraintsHold
  rcases R1CS.lowerSegments_complete logicalEnv
      (constraintSegments interface coordinate parentOffset offset)
      (offset + Logical.logicalPrivateCount) segmentScope segmentLogical with
    ⟨completed, physicalAgrees, segmentRows⟩
  refine ⟨completed, ?_, segmentRows⟩
  have combined := logicalAgreesFixed.append physicalAgrees
  rw [← logicalConstraints_eq_segments_flatten interface coordinate
      parentOffset offset,
    totalFreshCount_eq interface coordinate parentOffset offset] at combined
  simpa [Logical.logicalPrivateCount] using combined

end NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.First54
