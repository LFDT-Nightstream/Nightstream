import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCommon

/-! Coordinate order of the canonical sampler and product suffix. The
proof traverses compact block schedules and never expands coordinates. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportTail

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation CanonicalBlockAssignment
open PiRlcWideSampler
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def rangeEnv (env : Circuit.Env) : Schedule :=
  [ofBlock Retained.canonicalBits (fun column => env column.val),
   ofBlock Retained.canonicalFields (fun column => env column.val),
   ofBlock Retained.resultBits (fun column => env column.val)]

def range (initial : Witness.FieldState) (source : Fin 17) : Schedule :=
  rangeEnv (Witness.rangeSources initial source)

theorem range_count (initial : Witness.FieldState) (source : Fin 17) : coordinateCount (range initial source) = 937 := by
  simp only [range, rangeEnv, CanonicalBlockAssignment.coordinateCount, ofBlock, BlockValue.coordinateCount,
    Retained.coordinateCounts.1, Retained.coordinateCounts.2.1, Retained.coordinateCounts.2.2]

private theorem rangeEnv_lookup (env : Circuit.Env) (index : Fin 937) :
    coordinateAt (rangeEnv env) index.val = Witness.rangeCoordinate env index := by
  have bound := index.isLt
  simp only [rangeEnv, coordinateAt, BlockValue.coordinateAt, BlockValue.coordinateCount, ofBlock,
    LowNormBlock.Block.coordinateCount, Retained.canonicalBits, Retained.canonicalFields, Retained.resultBits,
    LowNormSlot.Kind.width, BalancedTernary.width, Witness.rangeCoordinate, Nat.mul_one,
    Nat.div_one, Nat.mod_one]
  norm_num only
  by_cases bits : index.val < 256
  · simp only [dif_pos bits]
  · rw [dif_neg bits]
    by_cases fields : index.val < 584
    · have inside : index.val - 256 < 328 := by omega
      simp only [dif_neg bits, dif_pos fields, dif_pos inside]
    · have outside : ¬index.val - 256 < 328 := by omega
      have inside : index.val - 584 < 353 := by omega
      simp only [dif_neg bits, dif_neg fields, dif_neg outside, Nat.sub_sub]
      norm_num only
      simp only [dif_pos inside]

theorem range_lookup (initial : Witness.FieldState) (source : Fin 17) (index : Fin 937) :
    coordinateAt (range initial source) index.val = Witness.rangeCoordinate (Witness.rangeSources initial source) index :=
  rangeEnv_lookup _ index

private theorem uniform_count {count : Nat} (blocks : Fin count → Schedule) (width : Nat)
    (counts : ∀ source, coordinateCount (blocks source) = width) :
    coordinateCount (List.ofFn blocks).flatten = count * width := by
  induction count with
  | zero => simp only [List.ofFn_zero, List.flatten_nil, CanonicalBlockAssignment.coordinateCount, Nat.zero_mul]
  | succ count ih =>
    rw [List.ofFn_succ, List.flatten_cons, coordinateCount_append, counts 0,
      ih (fun source => blocks source.succ) (fun source => counts source.succ)]
    rw [Nat.succ_mul, Nat.add_comm]

private theorem uniform_lookup {count : Nat} (blocks : Fin count → Schedule) (width : Nat)
    (counts : ∀ source, coordinateCount (blocks source) = width)
    (source : Fin count) (position : Nat) (bounded : position < width) :
    coordinateAt (List.ofFn blocks).flatten (source.val * width + position) = coordinateAt (blocks source) position := by
  induction count with
  | zero => exact Fin.elim0 source
  | succ count ih =>
    rw [List.ofFn_succ, List.flatten_cons]
    refine Fin.cases ?_ (fun source => ?_) source
    · simp only [Fin.val_zero, Nat.zero_mul, Nat.zero_add]
      exact AssignmentTransportCommon.lookup_before _ _ position (by rwa [counts 0])
    · simp only [Fin.val_succ, Nat.succ_mul]
      have shifted : source.val * width + width + position = coordinateCount (blocks 0) + (source.val * width + position) := by
        rw [counts 0]
        omega
      rw [shifted, coordinateAt_append_offset]
      exact ih (fun current => blocks current.succ) (fun current => counts current.succ) source

def sampler (initial : Witness.FieldState) : Schedule :=
  ofBlock BatchPlan.poseidonBlock (Witness.sboxValues initial) :: (List.ofFn (range initial)).flatten

theorem sampler_count (initial : Witness.FieldState) : coordinateCount (sampler initial) = 135813 := by
  rw [sampler, CanonicalBlockAssignment.coordinateCount, uniform_count (range initial) 937 (range_count initial)]
  change (34 * 86) * 41 + 17 * 937 = 135813
  norm_num

theorem sampler_lookup (initial : Witness.FieldState) (index : Fin 135813) :
    coordinateAt (sampler initial) index.val = Witness.coordinate initial index := by
  by_cases poseidon : index.val < 119884
  · change (if index.val < 119884 then _ else _) = _
    rw [if_pos poseidon]
    unfold BlockValue.coordinateAt Witness.coordinate
    rw [dif_pos poseidon, dif_pos (show index.val <
      (ofBlock BatchPlan.poseidonBlock (Witness.sboxValues initial)).coordinateCount from poseidon)]
    simp only [ofBlock, BatchPlan.poseidonBlock, LowNormSlot.Kind.width, BalancedTernary.width, id_eq]
  · change (if index.val < 119884 then _ else _) = _
    rw [if_neg poseidon]
    unfold Witness.coordinate
    rw [dif_neg poseidon]
    let source : Fin 17 := ⟨(index.val - 119884) / 937, by omega⟩
    let position : Fin 937 := ⟨(index.val - 119884) % 937, Nat.mod_lt _ (by decide)⟩
    have location : index.val - 119884 = source.val * 937 + position.val := by
      dsimp only [source, position]
      omega
    change coordinateAt (List.ofFn (range initial)).flatten (index.val - 119884) = _
    conv_lhs => rw [location]
    rw [uniform_lookup (range initial) 937 (range_count initial) source position.val position.isLt]
    exact range_lookup initial source position

def tail (initial : Witness.FieldState) (values : PiRLCValues.Values) : Schedule :=
  sampler initial ++ [ofBlock PiRLCGeometry.fieldBlock (PiRLCValues.fieldValue initial values)]

theorem tail_count (initial : Witness.FieldState) (values : PiRLCValues.Values) :
    coordinateCount (tail initial values) = PiRLCGeometry.coordinateCount := by
  rw [tail, coordinateCount_append, sampler_count]
  change 135813 + (104652 * 41 + 0) = PiRLCGeometry.coordinateCount
  rw [PiRLCGeometry.coordinateCount_eq]

private theorem field_lookup {columns count : Nat} (start : Nat)
    (base : Assignment F columns) (values : Fin count → F) (column : Fin columns)
    (owned : start ≤ column.val ∧ column.val < start + count * 41) :
    coordinateAt [ofBlock (FieldAssignment.block count) values] (column.val - start) =
      FieldAssignment.write start base values column := by
  have inside : column.val - start < count * 41 := by omega
  simp only [coordinateAt, BlockValue.coordinateAt, ofBlock, BlockValue.coordinateCount,
    FieldAssignment.block, LowNormBlock.Block.coordinateCount, LowNormSlot.Kind.width,
    BalancedTernary.width, if_pos inside, dif_pos inside, FieldAssignment.write, dif_pos owned, id_eq]

/-- The sampler and field suffix gives the same direct coordinate as the
existing phase constructor, for every owned column. -/
theorem tail_lookup {columns : Nat} (interface : PiRLCGeometry.Interface columns)
    (base : Assignment F columns) (column : Fin columns)
    (owned : interface.start ≤ column.val ∧ column.val < interface.start + PiRLCGeometry.coordinateCount) :
    coordinateAt (tail (PiRLCWitness.initial interface base) (PiRLCWitness.inputValues interface base))
        (column.val - interface.start) = PiRLCWitness.assignment interface base column := by
  let initial := PiRLCWitness.initial interface base
  let values := PiRLCWitness.inputValues interface base
  by_cases sample : column.val < interface.start + 135813
  · have inside : column.val - interface.start < 135813 := by omega
    rw [tail, AssignmentTransportCommon.lookup_before _ _ _ (by rw [sampler_count]; exact inside)]
    rw [sampler_lookup initial ⟨column.val - interface.start, inside⟩]
    unfold PiRLCWitness.assignment Witness.assignment
    rw [dif_pos (by exact ⟨owned.1, sample⟩)]
    rfl
  · have outside : interface.start + 135813 ≤ column.val := by omega
    have location : column.val - interface.start = coordinateCount (sampler initial) +
        (column.val - (interface.start + 135813)) := by rw [sampler_count]; omega
    rw [tail, location, coordinateAt_append_offset]
    have inside : column.val - (interface.start + 135813) < PiRLCGeometry.fieldBlock.coordinateCount := by
      rw [PiRLCGeometry.coordinateCount_eq] at owned
      rw [PiRLCGeometry.fieldBlock_coordinateCount]
      omega
    rw [PiRLCWitness.assignment, Witness.assignment_outside _ _ _ column (Or.inr outside)]
    exact field_lookup _ _ _ _ ⟨outside, by
      rw [PiRLCGeometry.coordinateCount_eq] at owned
      change column.val < (interface.start + 135813) + 104652 * 41
      omega⟩

theorem canonical_assignment (program : RetainedLayout.Program)
    (raw : PerApplicationCanonicalAssignment.RawValues program) :
    CanonicalBlockAssignment.assignment
        (Lifecycle.encodedHashCells raw.outputDigest)
        (AssignmentTransportCommon.common raw ++
          tail
            (PiRLCWitness.initial (Stage1Plan.piRlcInterface program) (AssignmentProjection.seed program raw.assignment))
            (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) (AssignmentProjection.seed program raw.assignment))) =
      AssignmentProjection.assignment program raw.assignment := by
  funext column
  let base := AssignmentProjection.seed program raw.assignment
  let interface := Stage1Plan.piRlcInterface program
  by_cases before : column.val < RetainedLayout.commonCount program
  · have same := AssignmentTransportCommon.assignment_before raw column before
    have unchanged := PiRLCWitness.assignment_before interface base column before
    change CanonicalBlockAssignment.assignment _ _ column = PiRLCWitness.assignment interface base column
    rw [unchanged]
    change _ = AssignmentProjection.seed program raw.assignment column
    rw [← same]
    unfold CanonicalBlockAssignment.assignment
    split
    · rfl
    · apply AssignmentTransportCommon.lookup_before
      have counted := AssignmentTransportCommon.common_count raw
      rw [RetainedLayout.commonCount_eq] at before counted
      change column.val - 270 < coordinateCount (AssignmentTransportCommon.common raw)
      omega
  · change CanonicalBlockAssignment.assignment _ _ column = PiRLCWitness.assignment interface base column
    unfold CanonicalBlockAssignment.assignment
    rw [dif_neg (by
      rw [RetainedLayout.commonCount_eq] at before
      change ¬ column.val < 270
      omega)]
    have location : column.val - ProductionAssignment.publicWidth =
        coordinateCount (AssignmentTransportCommon.common raw) + (column.val - interface.start) := by
      have counted := AssignmentTransportCommon.common_count raw
      change column.val - 270 = coordinateCount (AssignmentTransportCommon.common raw) +
        (column.val - RetainedLayout.commonCount program)
      omega
    rw [location, coordinateAt_append_offset]
    apply tail_lookup interface base column
    exact ⟨by exact Nat.le_of_not_gt before, column.isLt⟩

end NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportTail
