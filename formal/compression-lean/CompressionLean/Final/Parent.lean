import CompressionLean.Final.Bridge
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.OutputBinding

/-!
Owns the link from layer 1's relation to F′'s. Under the honest setup of the
F′ key, a layer-1 CE(B) witness of the statement that an F′ PiRLC parent
names is a witness of F′'s CE(B) relation for that parent (`parent_holds`,
SuperNeo v1.2 Definition 21).

- `HonestSetup`: the setup's key rows are the Ajtai key's rows, and its runs
  give every entry of every completed CCS matrix (`runEntry`), row `i` in
  F′'s little-endian order. Rust builds the setup in `lib.rs` (`key`,
  `oracle`) from `Tables::from_rows` (`matrix.rs`).
- `ParentShape`: the layer-1 sizes that `CompressionKey::relation` passes to
  `Relation::new`, and `H < B`.
- `statementOf`: the layer-1 statement that an F′ instance names.
  `assignmentOf`: the F′ assignment of a layer-1 witness.
- `evaluate_coefficient`: F′'s evaluation of a coefficient-expanded matrix
  is the Φ81-kernel sum of `Mᵀχ_r` (SuperNeo v1.2 Theorem 10).

Does not own the terminal output columns (`ParentClaim.lean`) or the games.
-/

namespace CompressionLean.Final

open CompressionLean.Game
open Polynomial NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial

section Evaluation

variable {lw : ℕ}

/-- A carrier column as its block and lane. -/
def columnEquiv :
    Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)) × Fin ringDegree ≃
      Fin (Phi81CarrierLayout.carrierWidth lw) where
  toFun x := Phi81CarrierLayout.carrierColumn x.1 x.2
  invFun c := Phi81ColumnLayout.decode c
  left_inv x := Phi81CarrierLayout.decode_carrierColumn x.1 x.2
  right_inv c := Fin.ext (Phi81ColumnLayout.flatIndex_decode c)

theorem sum_columns {M : Type} [AddCommMonoid M] (g : Fin (Phi81CarrierLayout.carrierWidth lw) → M) :
    ∑ c, g c = ∑ b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)),
      ∑ e : Fin ringDegree, g (Phi81CarrierLayout.carrierColumn b e) := by
  rw [← Fintype.sum_prod_type' (fun b e => g (Phi81CarrierLayout.carrierColumn b e))]
  exact (Fintype.sum_equiv columnEquiv _ _ (fun _ => rfl)).symm

/-- Entry `(i, c)` of an F′ matrix on numbers: row `i` in F′'s little-endian
order, zero past the carrier. -/
def entryAt (M : PaperLinearAlgebra.BooleanMatrix F cubeVariables (Phi81CarrierLayout.carrierWidth lw))
    (i c : ℕ) : Gl :=
  if h : c < Phi81CarrierLayout.carrierWidth lw then gl (M (vertexAt cubeVariables i) ⟨c, h⟩) else 0

/-- The real part of a layer-1 `K` value. -/
def reHom : Layer1.Quad Gl →+ Gl where
  toFun q := q.re
  map_zero' := rfl
  map_add' _ _ := rfl

/-- The imaginary part of a layer-1 `K` value. -/
def imHom : Layer1.Quad Gl →+ Gl where
  toFun q := q.im
  map_zero' := rfl
  map_add' _ _ := rfl

theorem reHom_scale (q : Layer1.Quad Gl) (x : Gl) : reHom (q * ⟨x, 0⟩) = reHom q * x := by
  simp [reHom]

theorem imHom_scale (q : Layer1.Quad Gl) (x : Gl) : imHom (q * ⟨x, 0⟩) = imHom q * x := by
  simp [imHom]

variable {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

theorem paddedEntry_eq (relation : ProductionKey.LogicalRelation lw pf)
    (M : PaperLinearAlgebra.BooleanMatrix F cubeVariables (Phi81CarrierLayout.carrierWidth lw))
    (v : BooleanVertex cubeVariables) (b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)))
    (l : Fin ringDegree) :
    (PaperAlgebra.matrixSource relation.system).paddedEntry ConcreteCarrier.baseOps M v b l =
      M v (Phi81CarrierLayout.carrierColumn b l) := by
  have encoded : (PaperAlgebra.matrixSource relation.system).columnLayout.encode? b l =
      some (Phi81CarrierLayout.carrierColumn b l) := Phi81CarrierLayout.layout_encode?_isSome b l
  unfold MatrixCoefficientSource.MatrixSource.paddedEntry
  rw [encoded]

/-- One coefficient of F′'s evaluation of a coefficient-expanded matrix
(`PaperAlgebra.evaluationFamily`), under a part `π` of `K`: the sum over
blocks `b`, lanes `l, e` of `π(w_b(l))·z(b, e)·W[p][l][e]`, where
`w = Mᵀχ_r` (SuperNeo v1.2 Definition 21, Theorem 10). -/
theorem evaluate_coefficient (relation : ProductionKey.LogicalRelation lw pf)
    (M : PaperLinearAlgebra.BooleanMatrix F cubeVariables (Phi81CarrierLayout.carrierWidth lw))
    (a : Fin (Phi81CarrierLayout.carrierWidth lw) → F)
    (point : PaperAlgebra.Point) (p : Fin ringDegree) (π : Layer1.Quad Gl →+ Gl)
    (scale : ∀ q x, π (q * ⟨x, 0⟩) = π q * x) :
    π (toQuad ((BooleanTable.tabulate fun v => K.embed (PaperLinearAlgebra.matrixVectorAt
        ConcreteCarrier.baseOps ((PaperAlgebra.matrixSource relation.system).coefficientMatrixOf
          ConcreteCarrier.baseOps M p) a v)).evaluate ConcreteCarrier.extensionOps point)) =
      ∑ b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)),
        ∑ l : Fin ringDegree, ∑ e : Fin ringDegree,
          (∑ i ∈ Finset.range (2 ^ cubeVariables),
              π (Lib.eqAt (point.coordinates.map toQuad) i) * entryAt M i (ringDegree * b + l)) *
            gl (a (Phi81CarrierLayout.carrierColumn b e)) * gl (phi81Kernel.weight p l e) := by
  have entry (i : ℕ) (b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)))
      (l : Fin ringDegree) :
      entryAt M i (ringDegree * b + l) = gl (M (vertexAt cubeVariables i) (Phi81CarrierLayout.carrierColumn b l)) := by
    have val : (Phi81CarrierLayout.carrierColumn b l).val = ringDegree * b + l := by
      simp only [Phi81CarrierLayout.carrierColumn, Phi81ColumnLayout.flatIndex]
      ring
    have lt := (Phi81CarrierLayout.carrierColumn b l).isLt
    rw [val] at lt
    unfold entryAt
    rw [dif_pos lt]
    congr 2
    exact Fin.ext val.symm
  have coefficient (v : BooleanVertex cubeVariables) (c : Fin (Phi81CarrierLayout.carrierWidth lw)) :
      gl ((PaperAlgebra.matrixSource relation.system).coefficientMatrixOf ConcreteCarrier.baseOps M p v c) =
        ∑ l : Fin ringDegree, gl (M v (Phi81CarrierLayout.carrierColumn (Phi81ColumnLayout.decode c).1 l)) *
          gl (phi81Kernel.weight p l (Phi81ColumnLayout.decode c).2) := by
    unfold MatrixCoefficientSource.MatrixSource.coefficientMatrixOf
    rw [sumRange_eq, ← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro l _
    rw [dif_pos l.isLt]
    exact congrArg (fun x => gl x * gl (phi81Kernel.weight p l (Phi81ColumnLayout.decode c).2))
      (paddedEntry_eq relation M v _ l)
  unfold BooleanTable.evaluate
  rw [evaluate_tabulate _ _ _ point.dimension, map_sum]
  calc _ = ∑ x ∈ Finset.range (2 ^ cubeVariables), π (Lib.eqAt (point.coordinates.map toQuad) x) *
        ∑ b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)), ∑ e : Fin ringDegree,
          (∑ l : Fin ringDegree, gl (M (vertexAt cubeVariables x) (Phi81CarrierLayout.carrierColumn b l)) *
            gl (phi81Kernel.weight p l e)) * gl (a (Phi81CarrierLayout.carrierColumn b e)) := by
        apply Finset.sum_congr rfl
        intro x _
        rw [show toQuad (K.embed (PaperLinearAlgebra.matrixVectorAt ConcreteCarrier.baseOps
            ((PaperAlgebra.matrixSource relation.system).coefficientMatrixOf ConcreteCarrier.baseOps M p) a
            (vertexAt productionShape.cubeVariables x))) = ⟨gl (PaperLinearAlgebra.matrixVectorAt
            ConcreteCarrier.baseOps ((PaperAlgebra.matrixSource relation.system).coefficientMatrixOf
              ConcreteCarrier.baseOps M p) a (vertexAt productionShape.cubeVariables x)), 0⟩ from rfl,
          scale, matrixVectorAt_eq, sum_columns]
        simp only [coefficient, Phi81CarrierLayout.decode_carrierColumn]
        rfl
    _ = _ := by
        simp only [Finset.mul_sum, Finset.sum_mul]
        conv_lhs => rw [Finset.sum_comm]
        apply Finset.sum_congr rfl
        intro b _
        rw [Finset.sum_comm]
        rw [Finset.sum_congr rfl fun e _ => Finset.sum_comm]
        rw [Finset.sum_comm]
        apply Finset.sum_congr rfl
        intro l _
        apply Finset.sum_congr rfl
        intro e _
        apply Finset.sum_congr rfl
        intro x _
        rw [entry]
        ring

end Evaluation

section Rows

variable {R : Type} [Field R] (s : Layer1.Shape) (bar : ℕ → ℕ → R) (S : Layer1.Setup R)
  (st : Layer1.Statement R)

theorem weight_key {ρ : ℕ} (inside : ρ < s.kappa) (b : ℕ) :
    (Layer1.relationRows s bar S st).weight ρ b =
      Layer1.coeffPoly s.D fun l => S.key ρ (2 ^ s.laneVars * b + l) := by
  simp [Layer1.relationRows, inside]

theorem target_key {ρ : ℕ} (inside : ρ < s.kappa) :
    (Layer1.relationRows s bar S st).target ρ = Layer1.coeffPoly s.D (st.commitment ρ) := by
  simp [Layer1.relationRows, inside]

theorem weight_evalK (part : ℕ) (small : part < 2) (b : ℕ) :
    (Layer1.relationRows s bar S st).weight (s.evalKRow part) b =
      Layer1.barPoly s.D bar fun l =>
        if part = 0 then (st.chi (s.D * b + l)).1 else (st.chi (s.D * b + l)).2 := by
  simp only [Layer1.relationRows, Layer1.Shape.evalKRow]
  rw [if_neg (by omega), if_pos (by omega)]
  simp

theorem target_evalK (part : ℕ) (small : part < 2) :
    (Layer1.relationRows s bar S st).target (s.evalKRow part) =
      Layer1.coeffPoly s.D fun l => if part = 0 then (st.evalK l).1 else (st.evalK l).2 := by
  simp only [Layer1.relationRows, Layer1.Shape.evalKRow]
  rw [if_neg (by omega), if_pos (by omega)]
  simp

theorem weight_evalA (j part : ℕ) (inside : j < s.matrices) (small : part < 2) (b : ℕ) :
    (Layer1.relationRows s bar S st).weight (s.evalARow j part) b =
      Layer1.barPoly s.D bar fun l => if part = 0 then (Layer1.evalAWeight S st s.D j (s.D * b + l)).1
        else (Layer1.evalAWeight S st s.D j (s.D * b + l)).2 := by
  simp only [Layer1.relationRows, Layer1.Shape.evalARow]
  rw [if_neg (by omega), if_neg (by omega), if_pos (by omega),
    show s.kappa + 2 + 2 * j + part - s.kappa - 2 = 2 * j + part by omega,
    show (2 * j + part) / 2 = j by omega]
  rcases (show part = 0 ∨ part = 1 by omega) with rfl | rfl <;> simp

theorem target_evalA (j part : ℕ) (inside : j < s.matrices) (small : part < 2) :
    (Layer1.relationRows s bar S st).target (s.evalARow j part) =
      Layer1.coeffPoly s.D fun l => if part = 0 then (st.evalA j l).1 else (st.evalA j l).2 := by
  simp only [Layer1.relationRows, Layer1.Shape.evalARow]
  rw [if_neg (by omega), if_neg (by omega), if_pos (by omega),
    show s.kappa + 2 + 2 * j + part - s.kappa - 2 = 2 * j + part by omega,
    show (2 * j + part) / 2 = j by omega]
  rcases (show part = 0 ∨ part = 1 by omega) with rfl | rfl <;> simp

theorem weight_public (j : ℕ) (b : ℕ) :
    (Layer1.relationRows s bar S st).weight (s.publicRow j) b = if b = j then 1 else 0 := by
  simp only [Layer1.relationRows, Layer1.Shape.publicRow]
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
    show s.kappa + 2 + 2 * s.matrices + j - (s.kappa + 2 + 2 * s.matrices) = j by omega]

theorem target_public (j : ℕ) :
    (Layer1.relationRows s bar S st).target (s.publicRow j) = Layer1.coeffPoly s.D (st.publicInput j) := by
  simp only [Layer1.relationRows, Layer1.Shape.publicRow]
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
    show s.kappa + 2 + 2 * s.matrices + j - (s.kappa + 2 + 2 * s.matrices) = j by omega]

end Rows

section Link

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

/-- Entry `(i, c)` of matrix `j` as the runs of `S` give it: the sum of the
values `m_k·ratio^(c - start)` of the runs of matrix `j` and row `i` that
cover column `c` (`matrix.rs`, one `GeometricRowRun` per run). -/
noncomputable def runEntry (S : Layer1.Setup Gl) (D j i c : ℕ) : Gl :=
  ∑ k ∈ ((Finset.range S.runs).filter (fun k => S.runMatrix k = j ∧
      D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) ≤ c ∧
      c < D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) + S.slotLength (S.runSlot k))).filter
      (fun k => S.runRow k = i),
    S.runCoef k * S.slotRatio (S.runSlot k) ^ (c - (D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k)))

/-- When every run names a row below `2^n`, the Eval_A weight of column `c`
is `Σ_i M_j(i, c)·χ_r(i)` over the rows, with the entries the runs give. -/
theorem evalAWeight_eq (S : Layer1.Setup Gl) (st : Layer1.Statement Gl) (D j c n : ℕ)
    (row_lt : ∀ k, k < S.runs → S.runRow k < 2 ^ n) :
    Layer1.evalAWeight S st D j c = ∑ i ∈ Finset.range (2 ^ n), runEntry S D j i c • st.chi i := by
  unfold Layer1.evalAWeight runEntry
  rw [← Finset.sum_fiberwise_of_maps_to (g := S.runRow) (t := Finset.range (2 ^ n))
    (fun k member => Finset.mem_range.mpr (row_lt k (Finset.mem_range.mp (Finset.mem_filter.mp member).1)))]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.sum_smul]
  apply Finset.sum_congr rfl
  intro k member
  rw [(Finset.mem_filter.mp member).2]

/-- The layer-1 shape of the CE(B) relation of F′'s PiRLC parent at logical
width `lw`: what `CompressionKey::relation` (`finish.rs`) passes to
`Relation::new`. -/
structure ParentShape (s : Layer1.Shape) (lw : ℕ) : Prop where
  D : s.D = ringDegree
  laneVars : s.laneVars = 6
  kappa : s.kappa = productionProfile.commitmentWidth
  matrices : s.matrices = productionProfile.ccsMatrices
  publicBlocks : s.publicBlocks = PaperAlgebra.publicRingColumns
  blocks : s.blocks = Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)
  /-- The checked bound `H` is below `B`. -/
  norm : s.normBound < productionGlobalParams.bigB

/-- **The honest setup.** `S` is the layer-1 setup of the F′ circuit with the
logical relation `relation` and the Ajtai key `ajtai`. Rust builds it in
`neo_spartan::Setup::build` and `Key::derive` (`lib.rs`, `key` and `oracle`)
from `Tables::from_rows` (`matrix.rs`) of the circuit's matrix rows. -/
structure HonestSetup (s : Layer1.Shape) (S : Layer1.Setup Gl) (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf)) : Prop where
  /-- Key row `i` holds Ajtai row `i`: block `b`, lane `l` at cube index
  `2^laneVars·b + l` (`oracle`). -/
  key : ∀ (i : Fin productionProfile.commitmentWidth)
    (b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw))) (l : Fin ringDegree),
    S.key i (2 ^ s.laneVars * b + l) = gl (ajtai i b l)
  /-- Every run names a row of the cube. -/
  row_lt : ∀ k, k < S.runs → S.runRow k < 2 ^ cubeVariables
  /-- The runs give every entry of every completed CCS matrix, row `i` in
  F′'s little-endian order (`Tables::from_rows`). -/
  entry : ∀ (j : Fin productionProfile.ccsMatrices) (i : ℕ) (c : Fin (Phi81CarrierLayout.carrierWidth lw)),
    i < 2 ^ cubeVariables →
      runEntry S s.D j i c = gl ((PaperAlgebra.matrixSource relation.system).matrices j (vertexAt cubeVariables i) c)

/-- The single evaluation family of an F′ CE instance. -/
def family (P : PiRLC.v1_2.InputBinding.InputInstance lw pf) : PaperAlgebra.Evaluation :=
  P.evaluations.getD 0 PaperAlgebra.evaluationZero

/-- The layer-1 statement that an F′ CE instance names: its words, as
`Package::apply` reads them (`terminal/package.rs`). -/
def statementOf (P : PiRLC.v1_2.InputBinding.InputInstance lw pf) : Layer1.Statement Gl where
  commitment ρ l :=
    if h : ρ < productionProfile.commitmentWidth ∧ l < ringDegree then gl (P.commitment ⟨ρ, h.1⟩ ⟨l, h.2⟩) else 0
  publicInput j l :=
    if h : l < ringDegree ∧ ringDegree * j + l < ringDegree * PaperAlgebra.publicRingColumns then
      gl (P.publicInput ⟨ringDegree * j + l, h.2⟩) else 0
  r := P.point.coordinates.map fun c => (gl c.c0, gl c.c1)
  evalK l := if h : l < ringDegree then (gl ((family P).pad ⟨l, h⟩).c0, gl ((family P).pad ⟨l, h⟩).c1) else 0
  evalA j l := if h : j < productionProfile.ccsMatrices ∧ l < ringDegree then
      (gl ((family P).matrix ⟨j, h.1⟩ ⟨l, h.2⟩).c0, gl ((family P).matrix ⟨j, h.1⟩ ⟨l, h.2⟩).c1) else 0

theorem statementOf_point (P : PiRLC.v1_2.InputBinding.InputInstance lw pf) :
    (statementOf P).point = P.point.coordinates.map toQuad := by
  simp only [Layer1.Statement.point, statementOf, List.map_map]
  rfl

/-- The F′ assignment of a layer-1 witness: carrier column `54·b + l` is the
cell `2^laneVars·b + l` of the witness cube. -/
def assignmentOf (s : Layer1.Shape) (z : ℕ → Gl) : PaperAlgebra.Assignment (logicalWidth := lw) (publicFits := pf) :=
  fun c => z (2 ^ s.laneVars * (c.val / ringDegree) + c.val % ringDegree)

theorem coeffPoly_eq (v : ℕ → Gl) (w : RingF) (same : ∀ l : Fin ringDegree, v l = gl (w l)) :
    Layer1.coeffPoly ringDegree v = toPolynomial w := by
  unfold Layer1.coeffPoly toPolynomial
  rw [← Fin.sum_univ_eq_sum_range (fun p => C (v p) * X ^ p) ringDegree]
  apply Finset.sum_congr rfl
  intro l _
  rw [same]

theorem barPoly_eq (bar : ℕ → ℕ → Gl) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (w : ℕ → Gl) (u : RingF) (same : ∀ l : Fin ringDegree, w l = gl (u l)) :
    Layer1.barPoly ringDegree bar w = toPolynomial (barRing u) := by
  apply coeffPoly_eq
  intro p
  rw [← Fin.sum_univ_eq_sum_range (fun l => bar p l * w l) ringDegree]
  apply Finset.sum_congr rfl
  intro l _
  rw [barEq, same]

theorem assignmentBlock_apply (s : Layer1.Shape) (z : ℕ → Gl)
    (b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw))) (l : Fin ringDegree) :
    gl (Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock
      (assignmentOf (lw := lw) (pf := pf) s z) b l) = z (2 ^ s.laneVars * b + l) := by
  have div : (b.val * ringDegree + l.val) / ringDegree = b.val := by
    have := l.isLt
    simp only [ringDegree] at this ⊢
    omega
  have mod : (b.val * ringDegree + l.val) % ringDegree = l.val := by
    have := l.isLt
    simp only [ringDegree] at this ⊢
    omega
  simp only [Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock, assignmentOf,
    Phi81CarrierLayout.carrierColumn, Phi81ColumnLayout.flatIndex, div, mod]

theorem witnessBlock_eq (s : Layer1.Shape) (z : ℕ → Gl) (shape : ParentShape s lw)
    (b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw))) :
    Layer1.witnessBlock s z b = toPolynomial (Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock
      (assignmentOf (lw := lw) (pf := pf) s z) b) := by
  unfold Layer1.witnessBlock
  rw [shape.D]
  exact coeffPoly_eq _ _ fun l => (assignmentBlock_apply s z b l).symm

/-- A layer-1 ring row in F′'s terms: weights `W_b`, the witness blocks of
the assignment, and the target `y`. -/
theorem row_dvd {s : Layer1.Shape} {bar : ℕ → ℕ → Gl} {S : Layer1.Setup Gl} {st : Layer1.Statement Gl}
    {z : ℕ → Gl} (shape : ParentShape s lw)
    (rows : (Layer1.relationRows s bar S st).Holds modulus (Layer1.witnessBlock s z))
    {ρ : ℕ} (inside : ρ < s.rows)
    (W : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)) → RingF) (y : RingF)
    (weight : ∀ b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)),
      (Layer1.relationRows s bar S st).weight ρ b = toPolynomial (W b))
    (target : (Layer1.relationRows s bar S st).target ρ = toPolynomial y) :
    modulus ∣ ∑ b, toPolynomial (W b) * toPolynomial
      (Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock (assignmentOf (lw := lw) (pf := pf) s z) b) -
        toPolynomial y := by
  have holds := rows ρ inside
  unfold Layer1.RingRows.residual at holds
  rw [show (Layer1.relationRows s bar S st).blocks = s.blocks from rfl, shape.blocks,
    ← Fin.sum_univ_eq_sum_range (fun b => (Layer1.relationRows s bar S st).weight ρ b * Layer1.witnessBlock s z b),
    target] at holds
  have same : ∑ b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)),
      (Layer1.relationRows s bar S st).weight ρ b * Layer1.witnessBlock s z b =
        ∑ b, toPolynomial (W b) * toPolynomial (Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock
          (assignmentOf (lw := lw) (pf := pf) s z) b) :=
    Finset.sum_congr rfl fun b _ => by rw [weight b, witnessBlock_eq s z shape b]
  rwa [same] at holds

/-- An Eval row of layer 1 in F′'s terms: the target's coefficient `p` is the
Φ81-kernel sum of the weights `u_b` and the assignment blocks. -/
theorem evalRow_coeff {s : Layer1.Shape} {bar : ℕ → ℕ → Gl} {S : Layer1.Setup Gl} {st : Layer1.Statement Gl}
    {z : ℕ → Gl} (shape : ParentShape s lw)
    (rows : (Layer1.relationRows s bar S st).Holds modulus (Layer1.witnessBlock s z))
    {ρ : ℕ} (inside : ρ < s.rows)
    (u : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)) → RingF) (y : RingF)
    (weight : ∀ b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)),
      (Layer1.relationRows s bar S st).weight ρ b = toPolynomial (barRing (u b)))
    (target : (Layer1.relationRows s bar S st).target ρ = toPolynomial y) (p : Fin ringDegree) :
    gl (y p) = ∑ b, ∑ l, ∑ e, gl (u b l) * gl (Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock
      (assignmentOf (lw := lw) (pf := pf) s z) b e) * gl (phi81Kernel.weight p l e) := by
  rw [← ringSum_eq_of_dvd _ _ _ (row_dvd (pf := pf) shape rows inside _ _ weight target), ringFSum_apply]
  exact Finset.sum_congr rfl fun b _ => ringFMul_barRing _ _ p

/-- One part of a pair: the first for part `0`, the second otherwise. -/
def sel (part : ℕ) (x : Gl × Gl) : Gl := if part = 0 then x.1 else x.2

theorem sel_sum {ι : Type} (part : ℕ) (t : Finset ι) (f : ι → Gl × Gl) :
    sel part (∑ i ∈ t, f i) = ∑ i ∈ t, sel part (f i) := by
  unfold sel
  split_ifs
  · exact Prod.fst_sum
  · exact Prod.snd_sum

theorem sel_smul (part : ℕ) (a : Gl) (x : Gl × Gl) : sel part (a • x) = a * sel part x := by
  unfold sel
  split_ifs <;> rfl

theorem centeredMagnitude_slotValue (H t : ℕ) (small : t < 2 * H + 1) (half : 2 * H < goldilocksModulus) :
    centeredMagnitude (Layer1.slotValue H t : Gl) ≤ H := by
  unfold Layer1.slotValue centeredMagnitude
  change min (ZMod.val ((t : Gl) - (H : Gl))) (goldilocksModulus - ZMod.val ((t : Gl) - (H : Gl))) ≤ H
  rcases le_or_gt H t with le | lt
  · rw [← Nat.cast_sub le, ZMod.val_natCast_of_lt (by omega)]
    exact (min_le_left _ _).trans (by omega)
  · have nonzero : ((H - t : ℕ) : Gl) ≠ 0 := by
      intro zero
      have := congrArg ZMod.val zero
      rw [ZMod.val_natCast_of_lt (by omega), ZMod.val_zero] at this
      omega
    rw [show (t : Gl) - (H : Gl) = -((H - t : ℕ) : Gl) by rw [Nat.cast_sub lt.le]; ring, ZMod.neg_val,
      if_neg nonzero, ZMod.val_natCast_of_lt (by omega)]
    exact (min_le_right _ _).trans (by omega)

theorem family_ext {x y : PaperAlgebra.Evaluation} (pad : ∀ p, x.pad p = y.pad p)
    (matrix : ∀ j p, x.matrix j p = y.matrix j p) : x = y := by
  cases x
  cases y
  congr 1
  · exact funext pad
  · exact funext fun j => funext (matrix j)

theorem K_ext {x y : K} (c0 : gl x.c0 = gl y.c0) (c1 : gl x.c1 = gl y.c1) : x = y := by
  cases x
  cases y
  congr

/-- The padded identity `Pad = [I; 0]` of F′'s canonical row layout, as a
matrix. -/
def padMatrix (relation : ProductionKey.LogicalRelation lw pf) :
    PaperLinearAlgebra.BooleanMatrix F cubeVariables (Phi81CarrierLayout.carrierWidth lw) :=
  fun row column => (Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
    (Phi81CarrierLayout.carrierWidth lw) relation.cubeFits).paddedIdentityEntry
      ConcreteCarrier.baseOps.zero ConcreteCarrier.baseOps.one row column

theorem entryAt_pad (relation : ProductionKey.LogicalRelation lw pf) {i c : ℕ} (hi : i < 2 ^ cubeVariables)
    (hc : c < Phi81CarrierLayout.carrierWidth lw) :
    entryAt (padMatrix relation) i c = if i = c then 1 else 0 := by
  unfold entryAt padMatrix
  rw [dif_pos hc]
  simp only [UnifiedSources.ColumnLayout.paddedIdentityEntry, Spec.Folding.PiCCS.CanonicalRowLayout.layout,
    index_vertexAt, Nat.mod_eq_of_lt hi]
  by_cases same : i = c
  · subst same
    rw [dif_pos hc, if_pos rfl]
    dsimp only
    rw [if_pos rfl]
    rfl
  · rw [if_neg same]
    by_cases live : i < Phi81CarrierLayout.carrierWidth lw
    · rw [dif_pos live]
      dsimp only
      rw [if_neg (fun equal => same (congrArg Fin.val equal).symm)]
      rfl
    · rw [dif_neg live]
      rfl

/-- Pad's weight of column `c`: `Σ_i π(χ_r(i))·Pad(i, c) = π(χ_r(c))`. -/
theorem pad_weight (relation : ProductionKey.LogicalRelation lw pf) (π : Layer1.Quad Gl →+ Gl)
    (point : List (Layer1.Quad Gl)) {c : ℕ} (hc : c < Phi81CarrierLayout.carrierWidth lw) :
    ∑ i ∈ Finset.range (2 ^ cubeVariables), π (Lib.eqAt point i) * entryAt (padMatrix relation) i c =
      π (Lib.eqAt point c) := by
  rw [Finset.sum_eq_single c]
  · rw [entryAt_pad relation (hc.trans_le relation.cubeFits) hc, if_pos rfl, mul_one]
  · intro i member different
    rw [entryAt_pad relation (Finset.mem_range.mp member) hc, if_neg different, mul_zero]
  · intro outside
    exact absurd (Finset.mem_range.mpr (hc.trans_le relation.cubeFits)) outside

theorem column_lt (b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)))
    (l : Fin ringDegree) : ringDegree * b + l < Phi81CarrierLayout.carrierWidth lw := by
  have lt := (Phi81CarrierLayout.carrierColumn b l).isLt
  simp only [Phi81CarrierLayout.carrierColumn, Phi81ColumnLayout.flatIndex] at lt
  linarith

theorem public_lt {j l : ℕ} (hj : j < PaperAlgebra.publicRingColumns) (hl : l < ringDegree) :
    ringDegree * j + l < ringDegree * PaperAlgebra.publicRingColumns := by
  change j < 5 at hj
  change l < 54 at hl
  change 54 * j + l < 54 * 5
  omega

theorem rows_eq {s : Layer1.Shape} (shape : ParentShape s lw) : s.rows = 37 := by
  simp only [Layer1.Shape.rows, shape.kappa, shape.matrices, shape.publicBlocks]
  rfl

theorem chi_part (P : PiRLC.v1_2.InputBinding.InputInstance lw pf) (part : ℕ) (π : Layer1.Quad Gl →+ Gl)
    (hπ : ∀ q, π q = sel part (q.re, q.im)) (i : ℕ) :
    sel part ((statementOf P).chi i) = π (Lib.eqAt (P.point.coordinates.map toQuad) i) := by
  rw [hπ, ← statementOf_point]
  rfl

/-- The Eval_A weight of column `c` under a part `π`: the column of
`π(M_jᵀχ_r)` with the honest setup's entries. -/
theorem evalA_weight {s : Layer1.Shape} {S : Layer1.Setup Gl} {relation : ProductionKey.LogicalRelation lw pf}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf)} (honest : HonestSetup s S relation ajtai)
    (P : PiRLC.v1_2.InputBinding.InputInstance lw pf) (part : ℕ) (π : Layer1.Quad Gl →+ Gl)
    (hπ : ∀ q, π q = sel part (q.re, q.im)) (j : Fin productionProfile.ccsMatrices) {c : ℕ}
    (hc : c < Phi81CarrierLayout.carrierWidth lw) :
    sel part (Layer1.evalAWeight S (statementOf P) s.D j c) =
      ∑ i ∈ Finset.range (2 ^ cubeVariables), π (Lib.eqAt (P.point.coordinates.map toQuad) i) *
        entryAt ((PaperAlgebra.matrixSource relation.system).matrices j) i c := by
  rw [evalAWeight_eq S _ s.D j c cubeVariables honest.row_lt, sel_sum]
  apply Finset.sum_congr rfl
  intro i member
  rw [sel_smul, honest.entry j i ⟨c, hc⟩ (Finset.mem_range.mp member), chi_part P part π hπ, mul_comm]
  unfold entryAt
  rw [dif_pos hc]

/-- One part of one Pad coefficient: F′'s evaluation is layer 1's `Eval_K`. -/
theorem pad_part {s : Layer1.Shape} {bar : ℕ → ℕ → Gl} {S : Layer1.Setup Gl}
    (relation : ProductionKey.LogicalRelation lw pf) (P : PiRLC.v1_2.InputBinding.InputInstance lw pf)
    (shape : ParentShape s lw) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    {z : ℕ → Gl} (rows : (Layer1.relationRows s bar S (statementOf P)).Holds modulus (Layer1.witnessBlock s z))
    (part : ℕ) (small : part < 2) (π : Layer1.Quad Gl →+ Gl) (hπ : ∀ q, π q = sel part (q.re, q.im))
    (scale : ∀ q x, π (q * ⟨x, 0⟩) = π q * x) (p : Fin ringDegree) :
    π (toQuad ((BooleanTable.tabulate fun v => K.embed (PaperLinearAlgebra.matrixVectorAt
        ConcreteCarrier.baseOps ((PaperAlgebra.matrixSource relation.system).coefficientMatrixOf
          ConcreteCarrier.baseOps (padMatrix relation) p) (assignmentOf (lw := lw) (pf := pf) s z) v)).evaluate
            ConcreteCarrier.extensionOps P.point)) =
      sel part ((statementOf P).evalK p) := by
  have row := evalRow_coeff (pf := pf) shape rows (ρ := s.evalKRow part)
    (by rw [rows_eq shape, Layer1.Shape.evalKRow, shape.kappa]; change 22 + part < 37; omega)
    (fun b l => π (Lib.eqAt (P.point.coordinates.map toQuad) (ringDegree * b + l)))
    (fun l => sel part ((statementOf P).evalK l))
    (fun b => by
      rw [weight_evalK s bar S _ part small b, shape.D]
      exact barPoly_eq bar barEq _ _ fun l => chi_part P part π hπ _)
    (by
      rw [target_evalK s bar S _ part small, shape.D]
      exact coeffPoly_eq _ _ fun l => rfl) p
  rw [evaluate_coefficient relation (padMatrix relation) _ P.point p π scale]
  change _ = gl (sel part ((statementOf P).evalK p))
  rw [row]
  apply Finset.sum_congr rfl
  intro b _
  apply Finset.sum_congr rfl
  intro l _
  apply Finset.sum_congr rfl
  intro e _
  rw [pad_weight relation π _ (column_lt b l)]
  rfl

/-- One part of one Eval_A coefficient: F′'s evaluation of matrix `j` is
layer 1's `Eval_A_j`. -/
theorem matrix_part {s : Layer1.Shape} {bar : ℕ → ℕ → Gl} {S : Layer1.Setup Gl}
    {relation : ProductionKey.LogicalRelation lw pf}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf)} (P : PiRLC.v1_2.InputBinding.InputInstance lw pf)
    (shape : ParentShape s lw) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (honest : HonestSetup s S relation ajtai)
    {z : ℕ → Gl} (rows : (Layer1.relationRows s bar S (statementOf P)).Holds modulus (Layer1.witnessBlock s z))
    (part : ℕ) (small : part < 2) (π : Layer1.Quad Gl →+ Gl) (hπ : ∀ q, π q = sel part (q.re, q.im))
    (scale : ∀ q x, π (q * ⟨x, 0⟩) = π q * x) (j : Fin productionProfile.ccsMatrices) (p : Fin ringDegree) :
    π (toQuad ((BooleanTable.tabulate fun v => K.embed (PaperLinearAlgebra.matrixVectorAt
        ConcreteCarrier.baseOps ((PaperAlgebra.matrixSource relation.system).coefficientMatrix
          ConcreteCarrier.baseOps j p) (assignmentOf (lw := lw) (pf := pf) s z) v)).evaluate
            ConcreteCarrier.extensionOps P.point)) =
      sel part ((statementOf P).evalA j p) := by
  have inside : j.val < s.matrices := by rw [shape.matrices]; exact j.isLt
  have row := evalRow_coeff (pf := pf) shape rows (ρ := s.evalARow j part)
    (by
      have := j.isLt
      rw [rows_eq shape, Layer1.Shape.evalARow, shape.kappa]
      change 22 + 2 + 2 * j.val + part < 37
      change j.val < 4 at this
      omega)
    (fun b l => ∑ i ∈ Finset.range (2 ^ cubeVariables), π (Lib.eqAt (P.point.coordinates.map toQuad) i) *
      entryAt ((PaperAlgebra.matrixSource relation.system).matrices j) i (ringDegree * b + l))
    (fun l => sel part ((statementOf P).evalA j l))
    (fun b => by
      rw [weight_evalA s bar S _ j part inside small b, shape.D]
      exact barPoly_eq bar barEq _ _ fun l => by
        have weight := evalA_weight honest P part π hπ j (column_lt b l)
        rw [shape.D] at weight
        exact weight)
    (by
      rw [target_evalA s bar S _ j part inside small, shape.D]
      exact coeffPoly_eq _ _ fun l => rfl) p
  rw [show (PaperAlgebra.matrixSource relation.system).coefficientMatrix ConcreteCarrier.baseOps j p =
    (PaperAlgebra.matrixSource relation.system).coefficientMatrixOf ConcreteCarrier.baseOps
      ((PaperAlgebra.matrixSource relation.system).matrices j) p from rfl,
    evaluate_coefficient relation _ _ P.point p π scale]
  change _ = gl (sel part ((statementOf P).evalA j p))
  rw [row]
  rfl

theorem array_singleton (values : Array PaperAlgebra.Evaluation) (single : values.size = 1) :
    #[values.getD 0 PaperAlgebra.evaluationZero] = values := by
  apply Array.ext
  · simp [single]
  · intro i h₁ _
    have zero : i = 0 := by simp at h₁; omega
    subst zero
    simp [Array.getD, single]

/-- **The F′ link.** Under the honest setup of the F′ circuit's key, a
layer-1 CE(B) witness `z` of the statement that an F′ PiRLC parent `P`
names gives a witness of F′'s CE(B) relation for `P` (SuperNeo v1.2
Definition 21): the Ajtai commitment, the public projection, the norm below
`B`, and the Pad and matrix evaluations at `P`'s point. -/
theorem parent_holds (s : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (P : PiRLC.v1_2.InputBinding.InputInstance lw pf)
    (shape : ParentShape s lw) (blocks_le : s.blocks ≤ 2 ^ s.blockVars)
    (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (honest : HonestSetup s S relation ajtai)
    (source : P.constraintSystem = PiRLC.v1_2.InputBinding.relationSource relation)
    (stage : P.stage = .combined) (single : P.evaluations.size = 1)
    {z : ℕ → Gl} (holds : Layer1.Relation s modulus bar S (statementOf P) z) :
    Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams P (assignmentOf s z) := by
  obtain ⟨norm, rows⟩ := holds
  refine ⟨⟨?_, ?_, ?_⟩, trivial, ?_⟩
  · funext ρ
    change Phi81Relation.PiRLCAlgebra.Commitment.ringFSum (fun b => ringFMul (ajtai ρ b)
      (Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock (assignmentOf s z) b)) = P.commitment ρ
    have key : ρ.val < s.kappa := by rw [shape.kappa]; exact ρ.isLt
    apply ringSum_eq_of_dvd
    apply row_dvd (pf := pf) shape rows (ρ := ρ)
      (by have := ρ.isLt; rw [rows_eq shape]; change ρ.val < 22 at this; omega)
    · intro b
      rw [weight_key s bar S _ key, shape.D]
      exact coeffPoly_eq _ _ fun l => honest.key ρ b l
    · rw [target_key s bar S _ key, shape.D]
      exact coeffPoly_eq _ _ fun l => by simp [statementOf, ρ.isLt, l.isLt]
  · funext column
    have width : column.val < 270 := column.isLt
    have jlt : column.val / ringDegree < Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw) := by
      have fits := pf
      rw [Phi81CarrierLayout.blockCount_carrierWidth]
      change 54 * 5 ≤ Phi81ColumnLayout.blockCount lw * 54 at fits
      change column.val / 54 < Phi81ColumnLayout.blockCount lw
      omega
    have lane : column.val % ringDegree < ringDegree := Nat.mod_lt _ (by decide)
    let block : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)) := ⟨column.val / ringDegree, jlt⟩
    have jPublic : column.val / ringDegree < PaperAlgebra.publicRingColumns := by
      change column.val / 54 < 5
      omega
    let y : RingF := fun l => P.publicInput ⟨ringDegree * (column.val / ringDegree) + l.val,
      public_lt jPublic l.isLt⟩
    have dvd := row_dvd (pf := pf) shape rows (ρ := s.publicRow (column.val / ringDegree))
      (by
        rw [rows_eq shape, Layer1.Shape.publicRow, shape.kappa, shape.matrices]
        change 22 + 2 + 2 * 4 + column.val / 54 < 37
        omega)
      (fun b => if b = block then ringFOne else ringFZero) y
      (fun b => by
        rw [weight_public]
        by_cases same : b = block
        · rw [if_pos (congrArg Fin.val same), if_pos same, toPolynomial_one]
        · rw [if_neg (fun equal => same (Fin.ext equal)), if_neg same, toPolynomial_zero])
      (by
        rw [target_public, shape.D]
        exact coeffPoly_eq _ _ fun l => by
          simp only [statementOf]
          rw [dif_pos ⟨l.isLt, public_lt jPublic l.isLt⟩])
    rw [Finset.sum_eq_single block (fun b _ other => by rw [if_neg other, toPolynomial_zero, zero_mul])
      (fun outside => absurd (Finset.mem_univ block) outside), if_pos rfl, toPolynomial_one, one_mul] at dvd
    have atLane := congrFun (eq_of_dvd dvd) ⟨column.val % ringDegree, lane⟩
    calc _ = Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock (assignmentOf s z) block
          ⟨column.val % ringDegree, lane⟩ :=
          congrArg (assignmentOf (lw := lw) (pf := pf) s z) (Fin.ext (Nat.div_add_mod' column.val ringDegree).symm)
      _ = y ⟨column.val % ringDegree, lane⟩ := atLane
      _ = P.publicInput column := congrArg P.publicInput (Fin.ext (Nat.div_add_mod column.val ringDegree))
  · rw [stage]
    intro column
    change centeredMagnitude (assignmentOf (lw := lw) (pf := pf) s z column) < productionGlobalParams.bigB
    have index : 2 ^ s.laneVars * (column.val / ringDegree) + column.val % ringDegree < 2 ^ s.cubeVars := by
      have hc := column.isLt
      have hb := blocks_le
      rw [shape.blocks, Phi81CarrierLayout.blockCount_carrierWidth] at hb
      change column.val < Phi81ColumnLayout.blockCount lw * 54 at hc
      rw [Layer1.Shape.cubeVars, shape.laneVars, pow_add]
      change 64 * (column.val / 54) + column.val % 54 < 64 * 2 ^ s.blockVars
      omega
    obtain ⟨t, small, value⟩ := norm _ index
    have bound := shape.norm
    change s.normBound < 65536 at bound
    unfold assignmentOf
    rw [value]
    exact (centeredMagnitude_slotValue s.normBound t small (by unfold goldilocksModulus; omega)).trans_lt
      shape.norm
  · change #[PaperAlgebra.evaluationFamily P.constraintSystem (assignmentOf s z) P.point] = P.evaluations
    rw [source, show PiRLC.v1_2.InputBinding.relationSource relation = PaperAlgebra.relationSource
      (Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables (Phi81CarrierLayout.carrierWidth lw)
        relation.cubeFits) relation.system from rfl, PaperAlgebra.evaluationFamily_eq_paper,
      ← array_singleton P.evaluations single]
    refine congrArg (fun family => #[family]) ?_
    apply family_ext
    · intro p
      have evalK (part : ℕ) (part_lt : part < 2) : sel part ((statementOf P).evalK p) =
          if part = 0 then gl ((family P).pad p).c0 else gl ((family P).pad p).c1 := by
        simp only [sel, statementOf]
        rw [dif_pos (show p.val < ringDegree from p.isLt)]
      apply K_ext
      · exact (pad_part relation P shape barEq rows 0 (by decide) reHom (fun _ => rfl) reHom_scale p).trans
          ((evalK 0 (by decide)).trans (if_pos rfl))
      · exact (pad_part relation P shape barEq rows 1 (by decide) imHom (fun _ => rfl) imHom_scale p).trans
          ((evalK 1 (by decide)).trans (if_neg (by decide)))
    · intro j p
      have evalA (part : ℕ) (part_lt : part < 2) : sel part ((statementOf P).evalA j p) =
          if part = 0 then gl ((family P).matrix j p).c0 else gl ((family P).matrix j p).c1 := by
        simp only [sel, statementOf]
        rw [dif_pos (show j.val < productionProfile.ccsMatrices ∧ p.val < ringDegree from ⟨j.isLt, p.isLt⟩)]
      apply K_ext
      · exact (matrix_part P shape barEq honest rows 0 (by decide) reHom (fun _ => rfl) reHom_scale j p).trans
          ((evalA 0 (by decide)).trans (if_pos rfl))
      · exact (matrix_part P shape barEq honest rows 1 (by decide) imHom (fun _ => rfl) imHom_scale j p).trans
          ((evalA 1 (by decide)).trans (if_neg (by decide)))

end Link

end CompressionLean.Final
