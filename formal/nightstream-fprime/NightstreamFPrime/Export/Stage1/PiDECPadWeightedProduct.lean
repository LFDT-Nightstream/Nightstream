import NightstreamFPrime.Export.Stage1.PiDECCommitmentFold
import NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericCompletionSum

/-!
Combine all 54 Pad basis weights before multiplying the same complete child
blocks. Two stored base-field bar keys encode the two K components. Each key
is shared by all sixteen children through the existing product kernel.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECPadWeightedProduct

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)
open NumericCompletionSum (numericSum)

private def addBasis (scalar : F) (lane : Fin ringDegree)
    (initial : StoredRing) : StoredRing :=
  PiDECCommitmentFold.add initial (Vector.ofFn fun output =>
    scalar * Phi81CoefficientKernel.nativeBarEntry output lane)

private theorem addBasis_product (scalar : F) (lane : Fin ringDegree)
    (initial : StoredRing) (right : RingF) (output : Fin ringDegree) :
    ringFMul (addBasis scalar lane initial).get right output =
      ringFMul initial.get right output +
        scalar * CarrierAction.kernelImage lane right output := by
  have basis : (Vector.ofFn (fun coefficient : Fin ringDegree =>
      scalar * Phi81CoefficientKernel.nativeBarEntry coefficient lane)).get =
      CarrierAction.ringFScale scalar (Phi81CoefficientKernel.barBasis lane) := by
    funext coefficient
    change (Vector.ofFn (fun index : Fin ringDegree =>
      scalar * Phi81CoefficientKernel.nativeBarEntry index lane))[coefficient.val] = _
    rw [Vector.getElem_ofFn]
    rfl
  rw [addBasis, PiDECCommitmentFold.add_value, basis,
    CarrierAction.ringFMul_add_left, CarrierAction.ringFMul_scale_left,
    ← CarrierAction.kernelImage_eq_ringFMul]
  rfl

private def keyPrefix (weights : Vector F ringDegree) (count : Nat) : StoredRing :=
  Nat.fold count (fun index _ initial =>
    if live : index < ringDegree then
      addBasis (weights.get ⟨index, live⟩) ⟨index, live⟩ initial
    else initial) PiDECCommitmentFold.zero

private theorem keyPrefix_succ (weights : Vector F ringDegree) (count : Nat) :
    keyPrefix weights (count + 1) =
      if live : count < ringDegree then
        addBasis (weights.get ⟨count, live⟩) ⟨count, live⟩ (keyPrefix weights count)
      else keyPrefix weights count := by
  simp only [keyPrefix, Nat.fold_succ]

private theorem keyPrefix_product (weights : Vector F ringDegree) (count : Nat)
    (right : RingF) (output : Fin ringDegree) :
    ringFMul (keyPrefix weights count).get right output =
      numericSum baseOps count (fun index =>
        if live : index < ringDegree then
          weights.get ⟨index, live⟩ *
            CarrierAction.kernelImage ⟨index, live⟩ right output
        else 0) := by
  induction count with
  | zero =>
      change ringFMul PiDECCommitmentFold.zero.get right output = (0 : F)
      rw [PiDECCommitmentFold.zero_value, RingFLaws.ringFMul_zero_left]
      rfl
  | succ count inductionHypothesis =>
      rw [keyPrefix_succ]
      change ringFMul
          (if live : count < ringDegree then
            addBasis (weights.get ⟨count, live⟩) ⟨count, live⟩ (keyPrefix weights count)
          else keyPrefix weights count).get right output =
        numericSum baseOps count (fun index =>
          if live : index < ringDegree then
            weights.get ⟨index, live⟩ *
              CarrierAction.kernelImage ⟨index, live⟩ right output
          else 0) +
        (if live : count < ringDegree then
          weights.get ⟨count, live⟩ * CarrierAction.kernelImage ⟨count, live⟩ right output
        else 0)
      by_cases live : count < ringDegree
      · rw [dif_pos live, dif_pos live, addBasis_product, inductionHypothesis]
      · rw [dif_neg live, dif_neg live, Fin.add_zero, inductionHypothesis]

/-- Materialize the weighted sum of the existing 54 bar bases. -/
def weightedKey (weights : Vector F ringDegree) : StoredRing :=
  keyPrefix weights ringDegree

private theorem weightedKey_product (weights : Vector F ringDegree)
    (right : RingF) (output : Fin ringDegree) :
    ringFMul (weightedKey weights).get right output =
      numericSum baseOps ringDegree (fun index =>
        if live : index < ringDegree then
          weights.get ⟨index, live⟩ *
            CarrierAction.kernelImage ⟨index, live⟩ right output
        else 0) :=
  keyPrefix_product weights ringDegree right output

private def pack (first second : Vector StoredRing productionGlobalParams.k) :
    Vector MaterializedRingK productionGlobalParams.k :=
  Vector.ofFn fun child => MaterializedRingK.ofRing fun output =>
    ⟨(first.get child).get output, (second.get child).get output⟩

private theorem pack_value
    (first second : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree) :
    ((pack first second).get child).toRing output =
      ⟨(first.get child).get output, (second.get child).get output⟩ := by
  change ((Vector.ofFn (fun selected : Fin productionGlobalParams.k =>
    MaterializedRingK.ofRing (fun lane =>
      ⟨(first.get selected).get lane, (second.get selected).get lane⟩)))[child.val]).toRing
      output = _
  rw [Vector.getElem_ofFn, MaterializedRingK.toRing_ofRing]

/-- Two base-field products per child replace the 54 separately weighted
basis products. Both keys and both product vectors are computed once. -/
def products (weights : Vector K ringDegree)
    (children : Vector StoredRing productionGlobalParams.k) :
    Vector MaterializedRingK productionGlobalParams.k :=
  let first := PiDECCommitmentBlock.products (weightedKey (weights.map K.c0)) children
  let second := PiDECCommitmentBlock.products (weightedKey (weights.map K.c1)) children
  pack first second

private theorem map_get (weights : Vector K ringDegree) (component : K → F)
    (lane : Fin ringDegree) :
    (weights.map component).get lane = component (weights.get lane) := by
  change (weights.map component)[lane.val] = component (weights[lane.val])
  rw [Vector.getElem_map]

private theorem numericSum_pair (count : Nat) (first second : Nat → F) :
    numericSum extensionOps count (fun index => (⟨first index, second index⟩ : K)) =
      ⟨numericSum baseOps count first, numericSum baseOps count second⟩ := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      change K.add
          (numericSum extensionOps count (fun index => (⟨first index, second index⟩ : K)))
          ⟨first count, second count⟩ =
        ⟨numericSum baseOps count first + first count,
          numericSum baseOps count second + second count⟩
      rw [inductionHypothesis]
      rfl

/-- Every returned K coefficient is exactly the original 54-term weighted
Pad basis sum. Arbitrary weights and complete child blocks are allowed; no
norm, expected-output, zero-lane, point or runtime premise is needed. -/
theorem products_value (weights : Vector K ringDegree)
    (children : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree) :
    ((products weights children).get child).toRing output =
      numericSum extensionOps ringDegree (fun index =>
        if live : index < ringDegree then
          K.mul (weights.get ⟨index, live⟩)
            (K.embed (CarrierAction.kernelImage ⟨index, live⟩
              (children.get child).get output))
        else K.zero) := by
  simp only [products, pack_value, PiDECCommitmentBlock.products_value,
    weightedKey_product]
  symm
  calc
    _ = numericSum extensionOps ringDegree (fun index =>
        (⟨if live : index < ringDegree then
            (weights.map K.c0).get ⟨index, live⟩ *
              CarrierAction.kernelImage ⟨index, live⟩ (children.get child).get output
          else 0,
          if live : index < ringDegree then
            (weights.map K.c1).get ⟨index, live⟩ *
              CarrierAction.kernelImage ⟨index, live⟩ (children.get child).get output
          else 0⟩ : K)) := by
      apply congrArg (numericSum extensionOps ringDegree)
      funext index
      by_cases live : index < ringDegree
      · simp only [dif_pos live, map_get, K.mul, K.embed,
          Fin.mul_zero, Fin.add_zero, Fin.zero_add]
      · simp only [dif_neg live, K.zero]
    _ = _ := numericSum_pair ringDegree _ _

end NightstreamFPrime.Export.Stage1.PiDECPadWeightedProduct
