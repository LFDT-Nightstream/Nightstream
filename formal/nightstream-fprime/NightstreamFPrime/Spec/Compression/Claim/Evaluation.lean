import NightstreamFPrime.Spec.Compression.Claim.Ring
import NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout
import NightstreamFPrime.Spec.RingRelation.Types

/-! The evaluations of `Spec.CE.Holds` as ring equations of the compression argument (SPEC.md,
Algorithm 9, step 7).

An evaluation of SuperNeo v1.2 Section 5 tabulates a coefficient matrix of the ring source over the
Boolean cube and evaluates the table at the claim's point. Vertex `v` of the cube is cell `i` with
`NumericBooleanDomain.index v = i`, and the first coordinate of the point binds bit 0 of `i`, as
`eqAt` does (`evaluate_tabulate`). Lane `o` of the evaluation of the coefficient matrix of `A` is
lane `o` of `Σ_j conj(w_j) · z_j`, where lane `l` of `w_j` is `Σ_i eq(r, i) · A[i, 64j + l]`
(`evaluate_coefficientMatrixOf`): the ring kernel `RingCoefficientKernel.ringKernel` is the product
with `conj(X^row)`. -/

namespace NightstreamFPrime.Spec.Compression.Claim

open Finset NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

local notation "ι" => algebraMap F K

section Linear

variable {R : Type} [CommRing R]

/-- `δ_r`, the ring element `X^r`. -/
def unit (r : ℕ) : ℕ → R := fun x => if x = r then 1 else 0

theorem mulX_congr {D h l : ℕ} (lane : l < D) (shift : h < D) {a a' : ℕ → R}
    (same : ∀ i < D, a i = a' i) : mulX D h a l = mulX D h a' l := by
  simp only [mulX]
  split_ifs <;> rw [same _ (by omega)]

theorem mulX_sum (D h l : ℕ) (s : Finset ℕ) (c : ℕ → R) (g : ℕ → ℕ → R) :
    mulX D h (fun x => ∑ r ∈ s, c r * g r x) l = ∑ r ∈ s, c r * mulX D h (g r) l := by
  simp only [mulX]
  split_ifs
  · rfl
  · rw [← sum_neg_distrib]
    exact sum_congr rfl fun r _ => by ring

/-- `conj(w)` is `Σ_r w_r · conj(X^r)` on its lanes. -/
theorem conjugate_eq_sum {D l : ℕ} (lane : l < D) (w : ℕ → R) :
    conjugate D w l = ∑ r ∈ range D, w r * conjugate D (unit r) l := by
  simp only [Algebra.conjugate, unit]
  split_ifs with zero
  · simp only [mul_ite, mul_one, mul_zero]
    rw [sum_ite_eq (range D) 0 w, ite_eq_left (mem_range.mpr (by omega))]
  · simp only [mul_neg, mul_ite, mul_one, mul_zero, sum_neg_distrib]
    rw [sum_ite_eq (range D) (D - l) w, ite_eq_left (mem_range.mpr (by omega))]

theorem mulX_conjugate {D h l : ℕ} (lane : l < D) (shift : h < D) (w : ℕ → R) :
    mulX D h (conjugate D w) l = ∑ r ∈ range D, w r * mulX D h (conjugate D (unit r)) l := by
  rw [mulX_congr lane shift (a' := fun x => ∑ r ∈ range D, w r * conjugate D (unit r) x)
    fun i lt => conjugate_eq_sum lt w, mulX_sum]

theorem conjugate_smul (D : ℕ) (c : R) (w : ℕ → R) :
    conjugate D (fun l => c * w l) = fun l => c * conjugate D w l := by
  funext l
  simp only [Algebra.conjugate]
  split_ifs <;> ring

end Linear

section Cube

/-- The value at cell `i` of a function on the vertices: its value at the vertex with index `i`. -/
def atIndex {R : Type} [Zero R] {v : ℕ} (f : BooleanVertex v → R) (i : ℕ) : R :=
  if h : i < 2 ^ v then f (NumericBooleanDomain.vertex v ⟨i, h⟩) else 0

theorem vertex_even (v i : ℕ) (h : 2 * i < 2 ^ (v + 1)) :
    NumericBooleanDomain.vertex (v + 1) ⟨2 * i, h⟩ =
      .cons false (NumericBooleanDomain.vertex v ⟨i, by rw [pow_succ] at h; omega⟩) := by
  simp only [NumericBooleanDomain.vertex]
  congr 1
  · show (2 * i % 2 == 1) = false
    simp
  · exact congrArg _ (Fin.ext (by simp))

theorem vertex_odd (v i : ℕ) (h : 2 * i + 1 < 2 ^ (v + 1)) :
    NumericBooleanDomain.vertex (v + 1) ⟨2 * i + 1, h⟩ =
      .cons true (NumericBooleanDomain.vertex v ⟨i, by rw [pow_succ] at h; omega⟩) := by
  simp only [NumericBooleanDomain.vertex]
  congr 1
  · show ((2 * i + 1) % 2 == 1) = true
    simp
  · exact congrArg _ (Fin.ext (by simp only; omega))

theorem evaluateCoordinates_tabulate : ∀ {v : ℕ} (f : BooleanVertex v → K) (coordinates : List K),
    coordinates.length = v →
    (BooleanTable.tabulate f).evaluateCoordinates extensionOps coordinates =
      mle coordinates (atIndex f)
  | 0, f, [], _ => by
      simp [Lib.mle, Lib.fold, atIndex, BooleanTable.tabulate, BooleanTable.evaluateCoordinates,
        NumericBooleanDomain.vertex]
  | v + 1, f, c :: cs, length => by
      have tail : cs.length = v := by simpa using length
      have bound : mle cs (bindLow c (atIndex f)) =
          (1 - c) * mle cs (atIndex fun t => f (.cons false t)) +
            c * mle cs (atIndex fun t => f (.cons true t)) := by
        rw [← mle_mul_left, ← mle_mul_left, ← mle_add]
        refine mle_congr cs fun i lt => ?_
        rw [tail] at lt
        have even : 2 * i < 2 ^ (v + 1) := by rw [pow_succ]; omega
        have odd : 2 * i + 1 < 2 ^ (v + 1) := by rw [pow_succ]; omega
        simp only [bindLow, atIndex, lt, even, odd, dite_true, vertex_even, vertex_odd]
      rw [BooleanTable.tabulate, BooleanTable.evaluateCoordinates,
        evaluateCoordinates_tabulate _ cs tail, evaluateCoordinates_tabulate _ cs tail, mle_cons,
        bound]
      show K.add _ (K.mul c (K.add _ (K.sub K.zero _))) = _
      rw [← sub_eq, ← zero_eq, ← add_eq, ← mul_eq, ← add_eq]
      ring

/-- The evaluation of a tabulated function at a point is `Σ_i eq(point, i) · f(vertex i)`. -/
theorem evaluate_tabulate {v : ℕ} (f : BooleanVertex v → K) (point : CubePoint K v) :
    (BooleanTable.tabulate f).evaluate extensionOps point =
      ∑ i ∈ range (2 ^ v), eqAt point.coordinates i * atIndex f i := by
  rw [BooleanTable.evaluate, evaluateCoordinates_tabulate f _ point.dimension, mle_eq_cubeSum,
    cubeSum_eq_sum_range, point.dimension]

end Cube

section Kernel

theorem foldl_add_eq {R B : Type} [AddCommMonoid R] (g : B → R) : ∀ (l : List B) (init : R),
    l.foldl (fun acc x => acc + g x) init = init + (l.map g).sum
  | [], init => by simp
  | x :: l, init => by rw [List.foldl_cons, foldl_add_eq g l, List.map_cons, List.sum_cons, add_assoc]

theorem matrixVectorAt_eq {v C : ℕ} (A : BooleanMatrix F v C) (z : Assignment F C)
    (vertex : BooleanVertex v) :
    matrixVectorAt baseOps A z vertex = ∑ x ∈ range C, cell (A vertex) x * cell z x := by
  refine (foldl_add_eq (fun col => A vertex col * z col) (canonicalFinIndices C) 0).trans ?_
  rw [zero_add, canonicalFinIndices, List.map_ofFn, List.sum_ofFn,
    ← Fin.sum_univ_eq_sum_range (fun x => cell (A vertex) x * cell z x)]
  simp [cell]

theorem sumRange_eq (n : ℕ) (f : ℕ → F) : sumRange baseOps n f = ∑ i ∈ range n, f i := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [sumRange, ih, sum_range_succ]
      rfl

/-- Lane `o` of `conj(X^r) · X^h`. -/
def kernelK (o r h : ℕ) : K := mulX ringDegree h (conjugate ringDegree (unit r)) o

/-- The ring kernel of SuperNeo v1.2 Section 5 is the product with `conj(X^row)`. -/
theorem weight_eq (o r h : Fin ringDegree) :
    ι (RingCoefficientKernel.ringKernel.weight o r h) = kernelK o r h := by
  change ι (ringFMul (RingCoefficientKernel.barBasis r) (ringFMonomial h 1) o) = _
  rw [ringFMul_apply, Claim.map_mul, mul, sum_eq_single (h : ℕ)
    (fun k member other => by
      have := mem_range.mp member
      simp [ringFCoeff, ringFMonomial, this, other])
    (fun out => absurd (mem_range.mpr h.isLt) out)]
  simp only [ringFCoeff, h.isLt, dite_true, ringFMonomial, ite_true, map_one, one_mul, kernelK]
  refine mulX_congr o.isLt h.isLt fun l lt => ?_
  simp only [lt, dite_true, RingCoefficientKernel.barBasis,
    RingCoefficientKernel.nativeBarEntry, Algebra.conjugate, unit]
  by_cases zero : l = 0
  · subst zero
    by_cases rz : (r : ℕ) = 0 <;> simp [rz, eq_comm]
  · simp only [zero, ite_false]
    by_cases rl : (r : ℕ) = ringDegree - l
    · simp [rl]
    · simp [rl, Ne.symm rl]

variable {v fc rc mc lw : ℕ} (mats : Fin mc → BooleanMatrix F v lw)
  (poly : CCSResidualTable.ConstraintPolynomial F mc)

theorem paddedEntry_apply (A : BooleanMatrix F v (RingCarrierLayout.carrierWidth lw))
    (vertex : BooleanVertex v)
    (b : Fin (ConcreteRingColumnLayout.blockCount (RingCarrierLayout.carrierWidth lw)))
    (r : Fin ringDegree) :
    (RingMatrixSource.source v fc rc mc lw mats poly).paddedEntry baseOps A vertex b r =
      cell (A vertex) (ringDegree * b + r) := by
  have lt := (RingCarrierLayout.carrierColumn b r).isLt
  change (b : ℕ) * ringDegree + r < _ at lt
  have lt' : ringDegree * b + r < RingCarrierLayout.carrierWidth lw := by rw [Nat.mul_comm]; exact lt
  have encoded : (RingMatrixSource.source v fc rc mc lw mats poly).columnLayout.encode? b r =
      some ⟨ringDegree * b + r, lt'⟩ := by
    simp only [RingMatrixSource.source, RingCarrierLayout.layout, ConcreteRingColumnLayout.layout,
      ConcreteRingColumnLayout.encode?, ConcreteRingColumnLayout.flatIndex, lt, dite_true,
      Option.some.injEq, Fin.mk.injEq]
    ring
  unfold MatrixSource.paddedEntry
  rw [encoded]
  simp only [cell, lt', dite_true]

/-- Lane `o` of the evaluation of `matrixVectorAt` of the coefficient matrix of `A` at a vertex:
`Σ_j Σ_h z_j[h] · Σ_r A[vertex, 64j + r] · (conj(X^r) · X^h)[o]`. -/
theorem embed_matrixVectorAt (A : BooleanMatrix F v (RingCarrierLayout.carrierWidth lw))
    (z : Assignment F (RingCarrierLayout.carrierWidth lw)) (o : Fin ringDegree)
    (vertex : BooleanVertex v) :
    ι (matrixVectorAt baseOps
        ((RingMatrixSource.source v fc rc mc lw mats poly).coefficientMatrixOf baseOps A o) z
        vertex) =
      ∑ j ∈ range (ConcreteRingColumnLayout.blockCount lw), ∑ h ∈ range ringDegree,
        ι (cell z (ringDegree * j + h)) *
          ∑ r ∈ range ringDegree, ι (cell (A vertex) (ringDegree * j + r)) * kernelK o r h := by
  rw [matrixVectorAt_eq, show range (RingCarrierLayout.carrierWidth lw) =
      range (ringDegree * ConcreteRingColumnLayout.blockCount lw) by
        rw [RingCarrierLayout.carrierWidth, Nat.mul_comm],
    Algebra.sum_range_mul, map_sum]
  refine sum_congr rfl fun j jm => ?_
  rw [map_sum]
  refine sum_congr rfl fun h hm => ?_
  have j_lt := mem_range.mp jm
  have h_lt := mem_range.mp hm
  have D : ringDegree = 64 := rfl
  have inside : ringDegree * j + h < RingCarrierLayout.carrierWidth lw := by
    rw [RingCarrierLayout.carrierWidth, Nat.mul_comm]
    nlinarith
  rw [_root_.map_mul]
  refine (mul_comm _ _).trans (congrArg (fun t => ι (cell z (ringDegree * j + h)) * t) ?_)
  simp only [cell, inside, dite_true, MatrixSource.coefficientMatrixOf]
  rw [sumRange_eq, map_sum]
  refine sum_congr rfl fun r rm => ?_
  have r_lt := mem_range.mp rm
  have div : (ringDegree * j + h) / ringDegree = j := by
    rw [Nat.mul_add_div (by omega), Nat.div_eq_of_lt h_lt, add_zero]
  have mod : (ringDegree * j + h) % ringDegree = h := by
    rw [Nat.mul_add_mod, Nat.mod_eq_of_lt h_lt]
  have block : (RingMatrixSource.source v fc rc mc lw mats poly).columnLayout.decode
      ⟨ringDegree * j + h, inside⟩ =
      (⟨j, by rw [RingCarrierLayout.blockCount_carrierWidth]; exact j_lt⟩,
        ⟨h, h_lt⟩) := by
    simp only [RingMatrixSource.source, RingCarrierLayout.layout, ConcreteRingColumnLayout.layout,
      ConcreteRingColumnLayout.decode, div, mod]
  have r_lt' : r < (RingMatrixSource.ringShape v fc rc mc).coefficientCount := r_lt
  simp only [block, r_lt', dite_true]
  rw [paddedEntry_apply]
  exact (_root_.map_mul _ _ _).trans (congrArg₂ (· * ·) rfl (weight_eq o ⟨r, r_lt⟩ ⟨h, h_lt⟩))

/-- Lane `o` of the evaluation at `point` of the table of the coefficient matrix of `A` is lane `o`
of `Σ_j conj(w_j) · z_j`, where lane `l` of `w_j` is `Σ_i eq(point, i) · A[vertex i, 64j + l]`. -/
theorem evaluate_coefficientMatrixOf (A : BooleanMatrix F v (RingCarrierLayout.carrierWidth lw))
    (z : Assignment F (RingCarrierLayout.carrierWidth lw)) (point : CubePoint K v)
    (o : Fin ringDegree) :
    (BooleanTable.tabulate fun vertex => K.embed (matrixVectorAt baseOps
        ((RingMatrixSource.source v fc rc mc lw mats poly).coefficientMatrixOf baseOps A o) z
        vertex)).evaluate extensionOps point =
      ∑ j ∈ range (ConcreteRingColumnLayout.blockCount lw),
        mul ringDegree (conjugate ringDegree fun l => ∑ i ∈ range (2 ^ v),
            eqAt point.coordinates i * ι (atIndex (fun vertex => cell (A vertex) (ringDegree * j + l)) i))
          (fun h => ι (cell z (ringDegree * j + h))) o := by
  show (BooleanTable.tabulate fun vertex : BooleanVertex v => K.embed (matrixVectorAt baseOps
      ((RingMatrixSource.source v fc rc mc lw mats poly).coefficientMatrixOf baseOps A o) z
      vertex)).evaluate extensionOps point = _
  rw [evaluate_tabulate]
  have lhs : ∀ i ∈ range (2 ^ v), eqAt point.coordinates i * atIndex (fun vertex : BooleanVertex v =>
      K.embed (matrixVectorAt baseOps
        ((RingMatrixSource.source v fc rc mc lw mats poly).coefficientMatrixOf baseOps A o) z
        vertex)) i =
      ∑ j ∈ range (ConcreteRingColumnLayout.blockCount lw), ∑ h ∈ range ringDegree,
        ∑ r ∈ range ringDegree, ι (cell z (ringDegree * j + h)) * kernelK o r h *
          (eqAt point.coordinates i *
            ι (atIndex (fun vertex => cell (A vertex) (ringDegree * j + r)) i)) := by
    intro i member
    have lt := mem_range.mp member
    simp only [atIndex, lt, dite_true]
    simp only [show K.embed = fun x => ι x from rfl]
    rw [embed_matrixVectorAt, mul_sum]
    refine sum_congr rfl fun j _ => ?_
    rw [mul_sum]
    refine sum_congr rfl fun h _ => ?_
    rw [mul_sum, mul_sum]
    exact sum_congr rfl fun r _ => by ring
  rw [sum_congr rfl lhs, sum_comm]
  refine sum_congr rfl fun j _ => ?_
  rw [sum_comm]
  simp only [mul]
  refine sum_congr rfl fun h hm => ?_
  rw [mulX_conjugate o.isLt (mem_range.mp hm), sum_comm, mul_sum]
  refine sum_congr rfl fun r _ => ?_
  rw [sum_mul, mul_sum]
  exact sum_congr rfl fun i _ => by simp only [kernelK]; ring

end Kernel

end NightstreamFPrime.Spec.Compression.Claim
