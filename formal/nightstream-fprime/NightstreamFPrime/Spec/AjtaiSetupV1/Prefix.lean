import NightstreamFPrime.Spec.AjtaiSetupV1
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Binding

/-!
Same-seed Ajtai key prefixes and suffix zero-extension of complete carriers.
The smaller integer kernel becomes a kernel of the larger key at the same
strict norm bound. Padding starts after the smaller full carrier; it does
not restore deleted interior positions or preserve setup authority words.
No key evaluation, hardness, probability, or runtime claim is added.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.AjtaiSetupV1.Prefix

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra

/-- The retained key coefficient has the same seed, row, block, and lane.
The number of message columns is not part of the PRG address. -/
theorem verifierKey_prefix {rows smallColumns largeColumns : Nat}
    (small : Setup rows smallColumns) (large : Setup rows largeColumns)
    (sameSeed : small.seed = large.seed) (fits : smallColumns ≤ largeColumns)
    (row : Fin rows) (block : Fin smallColumns) :
    small.verifierKey row block = large.verifierKey row (Fin.castLE fits block) := by
  funext lane
  apply Fin.ext
  change wideCoefficientNat small.seed.bytes row.val block.val lane.val =
    wideCoefficientNat large.seed.bytes row.val block.val lane.val
  rw [sameSeed]

/-- Preserve the full smaller vector as a prefix and append zeros. The size
proof excludes truncation; no coordinate is inserted in the retained prefix. -/
def zeroExtend {Value : Type*} [Zero Value] {smallWidth largeWidth : Nat}
    (_fits : smallWidth ≤ largeWidth) (value : Fin smallWidth → Value) :
    Fin largeWidth → Value :=
  fun column => if inside : column.val < smallWidth then value ⟨column.val, inside⟩ else 0

private theorem zeroExtend_prefix {Value : Type*} [Zero Value]
    {smallWidth largeWidth : Nat} (fits : smallWidth ≤ largeWidth)
    (value : Fin smallWidth → Value) (column : Fin smallWidth) :
    zeroExtend fits value (Fin.castLE fits column) = value column := by
  change (if inside : column.val < smallWidth then value ⟨column.val, inside⟩ else 0) = value column
  rw [dif_pos column.isLt]

private theorem zeroExtend_outside {Value : Type*} [Zero Value]
    {smallWidth largeWidth : Nat} (fits : smallWidth ≤ largeWidth)
    (value : Fin smallWidth → Value) (column : Fin largeWidth)
    (outside : smallWidth ≤ column.val) : zeroExtend fits value column = 0 := by
  exact dif_neg (Nat.not_lt_of_ge outside)

private theorem blockCount_mono {smallWidth largeWidth : Nat}
    (fits : smallWidth ≤ largeWidth) :
    Phi81ColumnLayout.blockCount smallWidth ≤ Phi81ColumnLayout.blockCount largeWidth := by
  simp only [Phi81ColumnLayout.blockCount, ringDegree]
  omega

private theorem carrierWidth_blocks (shape : Phi81Relation.Shape) :
    shape.carrierWidth = Phi81ColumnLayout.blockCount shape.carrierWidth * ringDegree := by
  change Phi81CarrierLayout.carrierWidth shape.logicalWidth =
    Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth shape.logicalWidth) * ringDegree
  rw [Phi81CarrierLayout.blockCount_carrierWidth]
  rfl

private theorem assignmentBlock_prefix {smallShape largeShape : Phi81Relation.Shape}
    (fits : smallShape.carrierWidth ≤ largeShape.carrierWidth)
    (value : Assignment smallShape)
    (block : Fin (Phi81ColumnLayout.blockCount smallShape.carrierWidth)) :
    CarrierAction.assignmentBlock (logicalWidth := largeShape.logicalWidth)
        (zeroExtend fits value) (Fin.castLE (blockCount_mono fits) block) =
      CarrierAction.assignmentBlock (logicalWidth := smallShape.logicalWidth) value block := by
  funext lane
  have sameColumn :
      Phi81CarrierLayout.carrierColumn (logicalWidth := largeShape.logicalWidth)
          (Fin.castLE (blockCount_mono fits) block) lane =
        Fin.castLE fits
          (Phi81CarrierLayout.carrierColumn (logicalWidth := smallShape.logicalWidth) block lane) := by
    apply Fin.ext
    rfl
  change zeroExtend fits value
      (Phi81CarrierLayout.carrierColumn (logicalWidth := largeShape.logicalWidth)
        (Fin.castLE (blockCount_mono fits) block) lane) =
    value (Phi81CarrierLayout.carrierColumn (logicalWidth := smallShape.logicalWidth) block lane)
  rw [sameColumn, zeroExtend_prefix]

private theorem assignmentBlock_outside {smallShape largeShape : Phi81Relation.Shape}
    (fits : smallShape.carrierWidth ≤ largeShape.carrierWidth)
    (value : Assignment smallShape)
    (block : Fin (Phi81ColumnLayout.blockCount largeShape.carrierWidth))
    (outside : Phi81ColumnLayout.blockCount smallShape.carrierWidth ≤ block.val) :
    CarrierAction.assignmentBlock (logicalWidth := largeShape.logicalWidth)
      (zeroExtend fits value) block = ringFZero := by
  funext lane
  apply zeroExtend_outside fits value
  change smallShape.carrierWidth ≤ block.val * ringDegree + lane.val
  calc
    _ = Phi81ColumnLayout.blockCount smallShape.carrierWidth * ringDegree :=
      carrierWidth_blocks smallShape
    _ ≤ block.val * ringDegree := Nat.mul_le_mul_right ringDegree outside
    _ ≤ block.val * ringDegree + lane.val := Nat.le_add_right _ _

private theorem ringFSum_prefix {smallCount largeCount : Nat}
    (fits : smallCount ≤ largeCount)
    (small : Fin smallCount → RingF) (large : Fin largeCount → RingF)
    (prefixMatches : ∀ index, large (Fin.castLE fits index) = small index)
    (outside : ∀ index, smallCount ≤ index.val → large index = ringFZero) :
    Commitment.ringFSum large = Commitment.ringFSum small := by
  induction largeCount generalizing smallCount with
  | zero =>
      have zero : smallCount = 0 := Nat.eq_zero_of_le_zero fits
      subst smallCount
      rfl
  | succ largeCount induction =>
      cases smallCount with
      | zero =>
          have tail := induction (smallCount := 0) (Nat.zero_le largeCount) small
            (fun index => large index.succ)
            (fun index => Fin.elim0 index)
            (fun index _ => outside index.succ (Nat.zero_le _))
          change ringFAdd (large 0) (Commitment.ringFSum (fun index => large index.succ)) = ringFZero
          rw [outside 0 (Nat.zero_le _), tail]
          funext lane
          exact ConcreteCarrier.baseLaws.zero_add 0
      | succ smallCount =>
          have bound : smallCount ≤ largeCount := Nat.le_of_succ_le_succ fits
          have head : large 0 = small 0 := prefixMatches 0
          have tails : ∀ index : Fin smallCount,
              large (Fin.castLE bound index).succ = small index.succ := by
            intro index
            exact prefixMatches index.succ
          have tailOutside : ∀ index : Fin largeCount,
              smallCount ≤ index.val → large index.succ = ringFZero := by
            intro index absent
            exact outside index.succ (Nat.succ_le_succ absent)
          change ringFAdd (large 0) (Commitment.ringFSum (fun index => large index.succ)) =
            ringFAdd (small 0) (Commitment.ringFSum (fun index => small index.succ))
          rw [head, induction bound _ _ tails tailOutside]

/-- Commit the smaller complete carrier against the same-seed key prefix.
Only complete zero blocks are appended, so the exact commitment is preserved. -/
theorem commit_zeroExtend {rows : Nat} {smallShape largeShape : Phi81Relation.Shape}
    (fits : smallShape.carrierWidth ≤ largeShape.carrierWidth)
    (small : Setup rows (Phi81ColumnLayout.blockCount smallShape.carrierWidth))
    (large : Setup rows (Phi81ColumnLayout.blockCount largeShape.carrierWidth))
    (sameSeed : small.seed = large.seed) (value : Assignment smallShape) :
    Commitment.commit (shape := largeShape) large.verifierKey (zeroExtend fits value) =
      Commitment.commit (shape := smallShape) small.verifierKey value := by
  funext row
  apply ringFSum_prefix (blockCount_mono fits)
  · intro block
    exact congrArg₂ ringFMul
      (verifierKey_prefix small large sameSeed (blockCount_mono fits) row block).symm
      (assignmentBlock_prefix (smallShape := smallShape) (largeShape := largeShape) fits value block)
  · intro block outside
    exact (congrArg (ringFMul (large.verifierKey row block))
      (assignmentBlock_outside (smallShape := smallShape) (largeShape := largeShape)
        fits value block outside)).trans (CarrierAction.ringFMul_zero_right _)

private theorem zeroExtend_intCast {smallWidth largeWidth : Nat}
    (fits : smallWidth ≤ largeWidth) (value : Fin smallWidth → Int) :
    (fun column => ((zeroExtend fits value column : Int) : ZMod goldilocksModulus)) =
      zeroExtend fits (fun column => (value column : ZMod goldilocksModulus)) := by
  funext column
  by_cases inside : column.val < smallWidth
  · simp only [zeroExtend, dif_pos inside]
  · simp only [zeroExtend, dif_neg inside, Int.cast_zero]

private theorem kernel_bound_positive {shape : Phi81Relation.Shape} {rows bound : Nat}
    {key : Commitment.Key shape rows} (solution : Binding.ShortKernelVector key bound) :
    0 < bound := by
  by_contra notPositive
  have zero : bound = 0 := Nat.eq_zero_of_not_pos notPositive
  apply solution.nonzero
  funext column
  have impossible : (solution.vector column).natAbs < 0 := by
    simpa only [zero] using solution.bounded column
  exact False.elim (Nat.not_lt_zero _ impossible)

/-- A smaller same-seed MSIS solution gives a larger-key solution by appending
integer zeros after the complete smaller carrier. The strict bound is unchanged;
its positivity follows from the supplied nonzero solution. -/
def extendShortKernel {rows bound : Nat} {smallShape largeShape : Phi81Relation.Shape}
    (fits : smallShape.carrierWidth ≤ largeShape.carrierWidth)
    (small : Setup rows (Phi81ColumnLayout.blockCount smallShape.carrierWidth))
    (large : Setup rows (Phi81ColumnLayout.blockCount largeShape.carrierWidth))
    (sameSeed : small.seed = large.seed)
    (solution : Binding.ShortKernelVector (shape := smallShape) small.verifierKey bound) :
    Binding.ShortKernelVector (shape := largeShape) large.verifierKey bound where
  vector := zeroExtend fits solution.vector
  nonzero := by
    intro zero
    apply solution.nonzero
    funext column
    have same := congrFun zero (Fin.castLE fits column)
    simpa only [zeroExtend_prefix] using same
  bounded := by
    intro column
    by_cases inside : column.val < smallShape.carrierWidth
    · simpa only [zeroExtend, dif_pos inside] using solution.bounded ⟨column.val, inside⟩
    · rw [zeroExtend, dif_neg inside, Int.natAbs_zero]
      exact kernel_bound_positive solution
  kernel := by
    rw [zeroExtend_intCast]
    exact (commit_zeroExtend (smallShape := smallShape) (largeShape := largeShape)
      fits small large sameSeed (fun column => (solution.vector column : ZMod goldilocksModulus))).trans
      solution.kernel

end NightstreamFPrime.Spec.AjtaiSetupV1.Prefix
