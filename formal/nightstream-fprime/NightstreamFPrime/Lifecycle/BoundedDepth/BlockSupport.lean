import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Spec.Folding.Refresh.Bound
import NightstreamFPrime.Spec.RingRelation.PiDECAlgebra.Radix

/-! Block support of a witness (CW-10.6). A witness packs into ring elements of `ringDegree`
coordinates, block `j` holding coordinates `64j` to `64j + 63` (SPEC.md section 2.3). A witness is
supported below `width` when it is zero on every block that starts at or past `width`
(`Supported`). A circuit's witness is zero past its package's logical width (SPEC.md section
2.4), so it is supported below that width (`supported_of_tail`).

The production folds keep support. Π_RLC multiplies each block by a ring element and adds the
products block by block (`CarrierAction.act`, `PiRLCFinite.combineAssignments`), so a zero block
stays zero (`combineAssignments_supported`). Π_DEC splits each coordinate into digits, and the
digits of 0 are 0 (`splitAssignment_supported`). Together they are the production support law
(`supportLaw`), the support counterpart of `FoldCompleteness.countNormLaw`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.BlockSupport

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.RingRelation
open NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism
open NightstreamFPrime.Lifecycle

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-- `witness` is zero on every ring block that starts at or past `width`. Its coordinates are
those of `PaperAlgebra.Assignment` and `Terminal.Witness`. -/
def Supported (width : Nat)
    (witness : PaperLinearAlgebra.Assignment F (RingCarrierLayout.carrierWidth logicalWidth)) :
    Prop :=
  ∀ column, width ≤ ringDegree * (column.val / ringDegree) → witness column = 0

/-- A witness that is zero past `width` is supported below `width`. -/
theorem supported_of_tail {width : Nat}
    {witness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (tail : ∀ column, width ≤ column.val → witness column = 0) : Supported width witness :=
  fun column above => tail column (le_trans above (Nat.mul_div_le column.val ringDegree))

theorem supported_zero (width : Nat) :
    Supported width
      (BaseLinear.assignmentZero (shape := PaperAlgebra.FullShape logicalWidth publicFits)) :=
  fun _ _ => rfl

theorem add_supported {width : Nat}
    {left right : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftSupported : Supported width left) (rightSupported : Supported width right) :
    Supported width (BaseLinear.assignmentAdd left right) := by
  intro column above
  show left column + right column = 0
  rw [leftSupported column above, rightSupported column above]
  rfl

/-- A ring multiple of a supported witness is supported: `CarrierAction.act` multiplies each
block by the challenge, and the product of a zero block is zero. -/
theorem act_supported {width : Nat} (challenge : RingF)
    {witness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (supported : Supported width witness) :
    Supported width (CarrierAction.act challenge witness) := by
  intro column above
  have zeroBlock : CarrierAction.assignmentBlock witness (ConcreteRingColumnLayout.decode column).1 =
      ringFZero := by
    funext lane
    apply supported
    have laneBound := lane.isLt
    show width ≤ ringDegree *
      ((column.val / ringDegree * ringDegree + lane.val) / ringDegree)
    simp only [ringDegree] at above laneBound ⊢
    omega
  show ringFMul challenge (CarrierAction.assignmentBlock witness
    (ConcreteRingColumnLayout.decode column).1) (ConcreteRingColumnLayout.decode column).2 = 0
  rw [zeroBlock, CarrierAction.ringFMul_zero_right]
  rfl

/-- The production Π_RLC combination of supported witnesses is supported. -/
theorem combineAssignments_supported {width : Nat} :
    ∀ {count : Nat} (challenges : Fin count → RingF)
      (assignments : Fin count →
        PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)),
      (∀ index, Supported width (assignments index)) →
        Supported width (PiRLCFinite.combineAssignments challenges assignments)
  | 0, _, _, _ => supported_zero width
  | _ + 1, challenges, assignments, supported =>
      add_supported (act_supported (challenges 0) (supported 0))
        (combineAssignments_supported (fun index => challenges index.succ)
          (fun index => assignments index.succ) fun index => supported index.succ)

theorem splitScalar_zero (child : PiDECAlgebra.Radix.ChildIndex) :
    PiDECAlgebra.Radix.splitScalar 0 child = 0 := by
  have below : centeredMagnitude (0 : F) < PiDECAlgebra.Radix.combinedBound := by
    rw [PiRLCAlgebra.Norm.Centered.centeredMagnitude_zero]
    decide
  have digit : PiDECAlgebra.Radix.magnitudeDigit 0 child = 0 := by
    simp [PiDECAlgebra.Radix.magnitudeDigit, PiDECAlgebra.Radix.natBit]
  unfold PiDECAlgebra.Radix.splitScalar PiDECAlgebra.Radix.boundedDigit
  rw [ite_eq_left below, digit, PiDECAlgebra.Radix.fieldOfNat_zero]
  split <;> rfl

/-- Each production Π_DEC digit of a supported witness is supported. -/
theorem splitAssignment_supported {width : Nat}
    {witness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (child : PiDECAlgebra.Radix.ChildIndex) (supported : Supported width witness) :
    Supported width (PiDECAlgebra.Radix.splitAssignment witness child) := by
  intro column above
  show PiDECAlgebra.Radix.splitScalar (witness column) child = 0
  rw [supported column above]
  exact splitScalar_zero child

/-- The production Π_RLC and Π_DEC keep support below every width. -/
theorem supportLaw (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits)) (width : Nat) :
    Refresh.SupportLaw (PaperAlgebra.piRlcAlgebra ajtai) (PaperAlgebra.piDecAlgebra ajtai)
      (Supported width) :=
  ⟨fun challenges assignments supported =>
      combineAssignments_supported challenges assignments supported,
    fun _ child supported => splitAssignment_supported child supported⟩

end NightstreamFPrime.Lifecycle.BoundedDepth.BlockSupport
