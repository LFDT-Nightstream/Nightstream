import NightstreamFPrime.Lifecycle.BoundedDepth.Terminal
import NightstreamFPrime.Spec.Compression.Claim.Slot

/-! A slot claim of the compression argument holds under `Spec.CE.Holds` (SPEC.md, Algorithm 10).

`claimValues` gives the values of a slot claim to the argument: the commitment by row and lane,
the public input, the point, `Eval_K` and each `Eval_A` by lane. When `Relation` holds at the
production parameters with the Ajtai key `ajtai` and dense rows that represent the slot circuit's
relation (`Spec.Compression.Claim.Represents`), the witness slice of the slot, with zeros past its
blocks, opens the slot claim under the circuit's relation as a combined claim (`slot_holds`), and
it has the norm `accumulatorBound(depth) + 1` (`slot_normBounded_slice`). -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.CompressionSlot

open Finset NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Compression.Algebra NightstreamFPrime.Spec.Compression.Verifier
open NightstreamFPrime.Spec.Compression.Production NightstreamFPrime.Spec.Compression.Claim
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Schedule (productionLightFoldCap)

local notation "ι" => algebraMap F K

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

/-- The values of a slot claim in the compression argument. -/
noncomputable def claimValues (claim : SlotClaim logicalWidth publicFits) : ClaimValues F K where
  commitment e := if h : e < productionProfile.commitmentWidth then cell (claim.commitment ⟨e, h⟩)
    else fun _ => 0
  publicInput := cell claim.publicInput
  point := claim.point.coordinates
  evalK := cell claim.evaluations.pad
  evalA m := if h : m < productionProfile.ccsMatrices then cell (claim.evaluations.matrix ⟨m, h⟩)
    else fun _ => 0

section Slot

variable {key : ℕ → ℕ → ℕ → F} {matrices : Fin 3 → Matrices F} {depths : Fin 2 → ℕ}
  {layout : Layout} {st : Statement (params key matrices depths layout)} {W : ℕ → F}

/-- The witness of claim `c` with zeros past its blocks. -/
def slotWitness (matrices : Fin 3 → Matrices F) (layout : Layout) (W : ℕ → F) (c : Fin 3) :
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  sliceAssignment W (layout.witness c).offset (ringDegree * ((matrices c).columns / ringDegree))

/-- Ring equation `e` of claim `c` on its witness slice. -/
theorem ring_equation (holds : Relation (params key matrices depths layout) st W) (c : Fin 3)
    {e l : ℕ} (e_lt : e < 23) (l_lt : l < ringDegree) :
    ∑ j ∈ range ((matrices c).columns / ringDegree),
        mul ringDegree ((params key matrices depths layout).form c (st c).point e j)
          (fun h => ι (W ((layout.witness c).offset + (ringDegree * j + h)))) l =
      (params key matrices depths layout).right (st c) e l :=
  (holds c).1 e e_lt l l_lt

/-- The slot witness has the norm `accumulatorBound(depth) + 1`, which is at most `B`. -/
theorem slot_normBounded_slice (holds : Relation (params key matrices depths layout) st W)
    (wf : WellFormed (params key matrices depths layout)) (s : Fin 2)
    (depth_le : depths s ≤ productionLightFoldCap) :
    RingRelation.assignmentNormBounded (productionGlobalParams.accumulatorBound (depths s) + 1)
      (slotWitness (logicalWidth := logicalWidth) (publicFits := publicFits) matrices layout W
        s.castSucc) := by
  intro column
  have norm := slot_norm key matrices depths layout holds s depth_le
  have fits : ringDegree * ((matrices s.castSucc).columns / ringDegree) ≤
      2 ^ (layout.witness s.castSucc).vars := wf.blocks_fit s.castSucc
  unfold slotWitness sliceAssignment
  split_ifs with within
  · exact norm.1 column (by omega)
  · show min 0 (fieldModulus - 0) < _
    omega

variable (holds : Relation (params key matrices depths layout) st W)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuit : CircuitKey logicalWidth publicFits) (claim : SlotClaim logicalWidth publicFits)

include holds

/-- The commitment: the Ajtai rows of `Relation` are `Σ_j ringFMul(A_ij, z_j)` through `ι`. -/
theorem slot_commitment (c : Fin 3) (keyEq : key = keyOf ajtai)
    (cover : ringDegree * ((matrices c).columns / ringDegree) ≤
      RingCarrierLayout.carrierWidth logicalWidth)
    (values : st c = claimValues claim) :
    (semantics ajtai).commit (slotWitness matrices layout W c) = claim.commitment := by
  funext row
  funext lane
  apply (algebraMap F K).injective
  have equation := ring_equation (key := key) (depths := depths) (st := st) holds c
    (e := row) (by have := row.isLt; change (row : ℕ) < 18 at this; omega) lane.isLt
  have row_lt : (row : ℕ) < productionProfile.commitmentWidth := row.isLt
  simp only [Params.form, Params.right, params, row_lt, ite_true, values, claimValues,
    dite_true, cell, lane.isLt, keyEq] at equation
  exact (ajtaiRow_slice ajtai W _ cover row lane).trans equation

/-- The public input: the first `320` cells of the slice. -/
theorem slot_publicInput (wf : WellFormed (params key matrices depths layout)) (c : Fin 3)
    (values : st c = claimValues claim) :
    RingRelation.projectPublicInput (slotWitness (logicalWidth := logicalWidth)
      (publicFits := publicFits) matrices layout W c) = claim.publicInput := by
  have publicFit : ringDegree * publicRingColumns ≤
      ringDegree * ((matrices c).columns / ringDegree) := wf.public_fit c
  funext y
  have y_lt : (y : ℕ) < ringDegree * publicRingColumns := y.isLt
  have equation := (holds c).2.1 y (by change (y : ℕ) < ringDegree * 5; exact y_lt)
  rw [values] at equation
  change (if (y : ℕ) < ringDegree * ((matrices c).columns / ringDegree)
    then W ((layout.witness c).offset + y) else 0) = claim.publicInput y
  rw [ite_eq_left (by omega)]
  change W ((layout.witness c).offset + y) = cell claim.publicInput y at equation
  simp only [cell, y.isLt, dite_true] at equation
  exact equation

/-- The evaluations: `Eval_K` and each `Eval_A` of `Relation` are the evaluations of the circuit's
relation at the claim's point. -/
theorem slot_evaluations (c : Fin 3)
    (represents : Represents (matrices c) circuit.relation.matrices)
    (values : st c = claimValues claim) :
    (semantics ajtai).evaluations (circuit.system ajtai) (slotWitness matrices layout W c)
      claim.point = #[claim.evaluations] := by
  have widths : productionProfile.commitmentWidth = 18 := rfl
  have matrixCount : productionProfile.ccsMatrices = 4 := rfl
  show #[evaluationFamily (relationSource (Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
      (RingCarrierLayout.carrierWidth logicalWidth) circuit.relation.cubeFits)
      circuit.relation.system) _ claim.point] = #[claim.evaluations]
  rw [evaluationFamily_eq_paper]
  apply congrArg (fun family => #[family])
  show (StrongReduction.EvaluationFamily.mk _ _ :
      StrongReduction.EvaluationFamily K productionShape) =
    ⟨claim.evaluations.pad, claim.evaluations.matrix⟩
  congr 1
  · funext o
    have equation := ring_equation (key := key) (depths := depths) (st := st) holds c
      (e := 18) (by omega) o.isLt
    simp only [Params.form, Params.right, Params.lanes, params, widths, lt_self_iff_false,
      ↓reduceIte, values, claimValues, cell, o.isLt, ↓reduceDIte] at equation
    exact (evaluate_pad _ _ circuit.relation.cubeFits represents.cover W _ claim.point o).trans
      equation
  · funext m o
    have m_lt : (m : ℕ) < 4 := m.isLt
    have equation := ring_equation (key := key) (depths := depths) (st := st) holds c
      (e := 19 + m) (by omega) o.isLt
    simp only [Params.form, Params.right, params, widths, matrixCount,
      show ¬ (19 + (m : ℕ) < 18) by omega, show 19 + (m : ℕ) ≠ 18 by omega, ↓reduceIte,
      show 19 + (m : ℕ) - 18 - 1 = m by omega, values, claimValues, m_lt, ↓reduceDIte, cell,
      o.isLt] at equation
    exact (evaluate_matrix _ _ represents W _ claim.point m o).trans equation

/-- A slot claim of the argument holds under its circuit's relation as a combined claim. -/
theorem slot_holds (wf : WellFormed (params key matrices depths layout)) (s : Fin 2)
    (depth_le : depths s ≤ productionLightFoldCap) (keyEq : key = keyOf ajtai)
    (represents : Represents (matrices s.castSucc) circuit.relation.matrices)
    (values : st s.castSucc = claimValues claim) :
    CE.Holds (semantics ajtai) productionGlobalParams (toClaim (circuit.system ajtai) claim)
      (slotWitness matrices layout W s.castSucc) :=
  ⟨⟨slot_commitment holds ajtai claim s.castSucc keyEq represents.cover values,
      slot_publicInput holds claim wf s.castSucc values,
      fun column => lt_of_lt_of_le (slot_normBounded_slice holds wf s depth_le column)
        (slot_norm key matrices depths layout holds s depth_le).2⟩,
    trivial, slot_evaluations holds ajtai circuit claim s.castSucc represents values⟩

end Slot

end NightstreamFPrime.Lifecycle.BoundedDepth.CompressionSlot
