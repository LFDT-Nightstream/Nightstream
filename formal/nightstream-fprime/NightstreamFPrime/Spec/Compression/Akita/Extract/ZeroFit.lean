import NightstreamFPrime.Spec.Compression.Akita.Extract.RootComplete

/-! The fit conditions of the honest prover of an opening of Akita hold for the zero tables
(ePrint 2026/1983, revision of 2026-10-15, Definition 10.5 and Theorem 10.9). The theorems
`tower_complete` and `Key20.tower_complete` have the fit conditions `BatchFit`, `LiftFit` and
`TraceFit` as hypotheses. This file shows that these hypotheses hold together for one input.

- For a zero source, each digit vector of the honest prover is 0: `decompose_zero`,
  `binaryDecompose_zero`, `pathCommit_zero`, `honestPartials_zero` and `tracePartials_zero`. So
  its flat witness is the zero table (`honest_flat_zero`, `honestTrace_flat_zero`,
  `Batch.honest_flat_zero`), and each response `z = Σ_i c_i·s_i` is 0.
- `FlatRelation.completion_zero`: the quotients of the ring rows of the zero table are 0, so the
  table with the digits of the quotients is the zero table.
- `batchFit_zero`, `liftFit_zero` and `traceFit_zero` are the fit conditions of a fold with
  opening groups, of a packing level and of a trace level for the zero tables, when the
  condition `after` holds for the zero table. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators
open Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

section Digits

variable {d : ℕ} [NeZero d] {ι : Type}

theorem ofCoeffs_zero : (ofCoeffs (fun _ => 0) : Rq d) = 0 :=
  ofCoeffs_eq fun j _ => (coeff_zero j).symm

theorem balancedDigit_zero {b : ℕ} (positive : 0 < b) : ∀ u, balancedDigit b 0 u = 0
  | 0 => by
      have half : ((b / 2 : ℕ) : ℤ) < b := by exact_mod_cast Nat.div_lt_self positive (by decide)
      show ((0 : ℤ) + (b / 2 : ℕ)) % b - (b / 2 : ℕ) = 0
      rw [zero_add, Int.emod_eq_of_lt (Int.natCast_nonneg _) half, sub_self]
  | u + 1 => by
      have half : ((b / 2 : ℕ) : ℤ) < b := by exact_mod_cast Nat.div_lt_self positive (by decide)
      show balancedDigit b (((0 : ℤ) + (b / 2 : ℕ)) / b) u = 0
      rw [zero_add, Int.ediv_eq_zero_of_lt (Int.natCast_nonneg _) half]
      exact balancedDigit_zero positive u

theorem fieldDigit_zero {b : ℕ} (two : 2 ≤ b) (δ u : ℕ) : fieldDigit b δ 0 u = 0 := by
  have half : (1 : ℤ) ≤ ((b / 2 : ℕ) : ℤ) := by exact_mod_cast Nat.div_pos two (by decide)
  have lifted : lift b δ 0 = 0 := by
    have small : ((0 : ℕ) : ℤ) ≤ min ((((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range δ, (b : ℤ) ^ u)
        (fieldModulus / 2 : ℕ) :=
      le_min (mul_nonneg (sub_nonneg.mpr half)
        (Finset.sum_nonneg fun u _ => pow_nonneg (Int.natCast_nonneg b) u)) (Int.natCast_nonneg _)
    unfold lift
    rw [ZMod.val_zero, ite_eq_left small]
    rfl
  unfold fieldDigit
  rw [lifted, balancedDigit_zero (by omega), Int.cast_zero]

/-- The digits of the zero vector are 0. -/
theorem decompose_zero {b : ℕ} (two : 2 ≤ b) {δ : ℕ} :
    (decompose b (0 : ι → Rq d) : ι × Fin δ → Rq d) = 0 :=
  funext fun entry => ofCoeffs_eq fun i _ => by
    rw [Pi.zero_apply, coeff_zero, Pi.zero_apply, coeff_zero, fieldDigit_zero two]

omit [NeZero d] in
theorem gadget_zero (b : ℕ) {δ : ℕ} : gadget b (0 : ι × Fin δ → Rq d) = 0 :=
  funext fun j => by simp [gadget]

/-- The zero vector satisfies `Enc` of Akita (147). -/
theorem encodable_zero {b : ℕ} (two : 2 ≤ b) (δ : ℕ) : Encodable b δ (0 : ι → Rq d) := by
  have half : (1 : ℤ) ≤ ((b / 2 : ℕ) : ℤ) := by exact_mod_cast Nat.div_pos two (by decide)
  have total : (0 : ℤ) ≤ ∑ u ∈ Finset.range δ, (b : ℤ) ^ u :=
    Finset.sum_nonneg fun u _ => pow_nonneg (Int.natCast_nonneg b) u
  intro j i _
  rw [Pi.zero_apply, coeff_zero, ZMod.valMinAbs_zero]
  exact ⟨mul_nonpos_of_nonpos_of_nonneg (by omega) total, mul_nonneg (by omega) total⟩

theorem vecNormSq_zero [Fintype ι] : vecNormSq (0 : ι → Rq d) = 0 := by
  simp [vecNormSq, normSq]

theorem sequence_zero {m : ℕ} : sequence (0 : Fin m → Rq d) = 0 :=
  funext fun position => by
    unfold sequence
    split
    · exact coeff_zero _
    · rfl

theorem cellsOf_zero {m : ℕ} (code : ι ≃ Fin m) : cellsOf code (0 : ι → Rq d) = 0 :=
  sequence_zero

theorem concat_zero {n : ℕ} (size : Fin n → ℕ) : concat size (fun _ => 0) = 0 :=
  funext fun x => Finset.sum_eq_zero fun g _ => by split <;> rfl

theorem binaryDigit_zero (u : ℕ) : binaryDigit 0 u = 0 := by
  simp [binaryDigit]

theorem binaryDecompose_zero {n : ℕ} (width inputDegree : ℕ) [NeZero inputDegree] :
    binaryDecompose width inputDegree (0 : Fin n → Rq d) = 0 :=
  funext fun j => ofCoeffs_eq fun i _ => by
    rw [Pi.zero_apply, coeff_zero, sequence_zero, Pi.zero_apply, binaryDigit_zero, ite_self]

omit [NeZero d] in
theorem stack_zero {n slices : ℕ} {cols : Type} [Fintype cols] (M : Matrix (Fin n) cols (Rq d)) :
    stack M (0 : Fin slices → cols → Rq d) = 0 :=
  funext fun j => by
    show M.mulVec ((0 : Fin slices → cols → Rq d) _) _ = 0
    rw [Pi.zero_apply, Matrix.mulVec_zero]
    rfl

/-- The digits of the compression maps for a zero image are 0 (Akita Section 4.5). -/
theorem pathCommit_zero {n : ℕ} : (chain : Option ChainShape) → (key : PathKey chain) →
    PathDigits.firstTable chain (pathCommit chain key (0 : Fin n → Rq d)).1 = 0 ∧
      PathDigits.secondTable chain (pathCommit chain key (0 : Fin n → Rq d)).1 = 0
  | none, _ => ⟨rfl, rfl⟩
  | some c, key => by
      constructor
      · show sequence (binaryDecompose c.firstWidth c.firstDegree (0 : Fin n → Rq d)) = 0
        rw [binaryDecompose_zero, sequence_zero]
      · show sequence (binaryDecompose c.secondWidth c.secondDegree ((ChainKey.first key).mulVec
          (binaryDecompose c.firstWidth c.firstDegree (0 : Fin n → Rq d)))) = 0
        rw [binaryDecompose_zero, Matrix.mulVec_zero, binaryDecompose_zero, sequence_zero]

end Digits

section Rows

variable {E : Type} [Field E] [Algebra Zq E] (rel : FlatRelation E) {base : ℕ} (G : Gaps)
  (digits : ℕ)

/-- A ring row that holds for the zero table has the left side 0 of Akita (111). -/
theorem RingRow.lifted_zero {row : RingRow} (holds : row.Holds 0) : row.lifted 0 = 0 := by
  have reads : ∀ c, row.read 0 c = 0 := fun c =>
    (congrArg ofCoeffs (funext fun j => by cases row.address c j <;> rfl)).trans ofCoeffs_zero
  have target : row.target = 0 := by
    have residual : row.residual 0 = 0 := holds
    simpa [RingRow.residual, reads] using residual
  simp [RingRow.lifted, reads, target, rep]

/-- The table with the digits of the quotients of the zero table is the zero table, when each
ring row holds for the zero table. -/
theorem FlatRelation.completion_zero (two : 2 ≤ base) (holds : ∀ r, (rel.row r).Holds 0) :
    rel.completion G base digits 0 = 0 := by
  have quotients : ∀ r, rel.quotients 0 r = 0 := fun r =>
    (mul_eq_zero.mp (((rel.quotients_spec (holds r)).2.symm.trans
      (RingRow.lifted_zero (holds r))))).resolve_left (modulus_monic _).ne_zero
  have table : rel.quotientTable base digits (rel.quotients 0) = 0 := by
    unfold FlatRelation.quotientTable
    simp only [quotients, Polynomial.coeff_zero, fieldDigit_zero two]
    exact concat_zero _
  show G.merge 0 (rel.quotientTable base digits (rel.quotients 0)) = 0
  rw [table, G.merge_zero]

end Rows

section Shapes

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

theorem Shape.outerView_zero : S.outerView (0 : S.InnerDigits) = 0 :=
  funext fun column => ofCoeffs_eq fun k _ => by
    rw [Pi.zero_apply, coeff_zero, Pi.zero_apply, coeff_zero]

theorem Shape.sliceInput_zero : S.sliceInput (0 : Fin S.blocks → S.InnerDigits) = 0 :=
  funext fun s => funext fun position => by
    unfold Shape.sliceInput
    split
    · rw [Pi.zero_apply, Shape.outerView_zero]
      rfl
    · rfl

/-- The bounded source blocks of the zero table are 0. -/
theorem Shape.embed_pack_zero : (fun i : Fin S.blocks => S.embed (S.pack 0 i)) = 0 :=
  funext fun i => funext fun entry => by
    show (if (entry.2 : ℕ) = 0 then S.pack 0 i entry.1 else 0) = 0
    rw [show S.pack 0 i entry.1 = 0 from ofCoeffs_zero, ite_self]

/-- The digits of the compression maps of the commitment for zero digits `t̂` are 0. -/
theorem Shape.path_zero (key : Key S) {inner : Fin S.blocks → S.InnerDigits} (zero : inner = 0) :
    PathDigits.firstTable S.chain
        (pathCommit S.chain key.chain (stack key.outer (S.sliceInput inner))).1 = 0 ∧
      PathDigits.secondTable S.chain
        (pathCommit S.chain key.chain (stack key.outer (S.sliceInput inner))).1 = 0 := by
  rw [zero, Shape.sliceInput_zero, stack_zero]
  exact pathCommit_zero _ _

/-- The flat witness of zero digit vectors is the zero table. -/
theorem Witness.flat_eq_zero {w : Witness O fold} (response : w.response = 0)
    (partials : w.partials = 0) (inner : w.inner = 0)
    (first : PathDigits.firstTable S.chain w.chain = 0)
    (second : PathDigits.secondTable S.chain w.chain = 0)
    (openFirst : PathDigits.firstTable O.chain w.openChain = 0)
    (openSecond : PathDigits.secondTable O.chain w.openChain = 0) : w.flat = 0 := by
  have view : (fun entry : Fin S.blocks × S.OuterColumn => S.outerView (w.inner entry.1) entry.2) =
      0 := funext fun entry => by
    rw [inner, Pi.zero_apply, Shape.outerView_zero]
    rfl
  funext x
  unfold Witness.flat
  rw [response, partials, view, first, second, openFirst, openSecond, cellsOf_zero, cellsOf_zero,
    cellsOf_zero]
  simp only [join]
  split_ifs <;> rfl

end Shapes

section Honest

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E]
  [Algebra Zq E]

theorem response_zero (c : Fin S.blocks → Rq O.carrierDegree) :
    response O c (0 : Fin S.blocks → S.Source) = 0 := by
  simp [response]

theorem traceResponse_zero (c : Fin S.blocks → Rq S.innerDegree) :
    traceResponse c (0 : Fin S.blocks → S.Source) = 0 := by
  simp [traceResponse]

theorem OpeningShape.native_zero : O.native (0 : Fin O.planes → Rq O.carrierDegree) = 0 :=
  funext fun y => ofCoeffs_eq fun k _ => by
    rw [sequence_zero, Pi.zero_apply, Pi.zero_apply, coeff_zero]

theorem OpeningShape.traceNative_zero : O.traceNative 0 = 0 :=
  funext fun y => ofCoeffs_eq fun k _ => by rw [coeff_zero, Pi.zero_apply, coeff_zero]

/-- The digits `ê` of a packing fold for zero source blocks are 0. -/
theorem honestPartials_zero (two : 2 ≤ S.openingBase) (β : Module.Basis (Fin O.planes) Zq E)
    (point : Point O E) : honestPartials β point (0 : Fin S.blocks → S.Source) = 0 :=
  funext fun column => by
    have planes : (fun t => contract β point t
        (gadget S.sourceBase ((0 : Fin S.blocks → S.Source) column.1))) = 0 :=
      funext fun t => by
        rw [Pi.zero_apply, gadget_zero, map_zero]
        rfl
    show decompose S.openingBase (O.native fun t => contract β point t
      (gadget S.sourceBase ((0 : Fin S.blocks → S.Source) column.1))) column.2 = 0
    rw [planes, OpeningShape.native_zero, decompose_zero two]
    rfl

omit [Algebra Zq E] in
/-- The digits `ê` of a trace fold for zero source blocks are 0. -/
theorem tracePartials_zero (two : 2 ≤ S.openingBase) (subfield : E →+* Rq S.innerDegree)
    (position : Fin S.blockLength → E) :
    tracePartials O subfield position (0 : Fin S.blocks → S.Source) = 0 :=
  funext fun column => by
    have contracted : traceContract subfield position
        (gadget S.sourceBase ((0 : Fin S.blocks → S.Source) column.1)) = 0 := by
      rw [Pi.zero_apply, gadget_zero]
      simp [traceContract]
    show decompose S.openingBase (O.traceNative (traceContract subfield position
      (gadget S.sourceBase ((0 : Fin S.blocks → S.Source) column.1)))) column.2 = 0
    rw [contracted, OpeningShape.traceNative_zero, decompose_zero two]
    rfl

/-- The digits of `H_1`, `H_2` for zero digits `ê` are 0. -/
theorem honestOpening_zero (key : OpeningKey O) :
    PathDigits.firstTable O.chain (honestOpening key 0).1 = 0 ∧
      PathDigits.secondTable O.chain (honestOpening key 0).1 = 0 := by
  have image : stack key.matrix (fun _ : Fin 1 => (0 : O.Partials)) = 0 := stack_zero key.matrix
  unfold honestOpening
  rw [image]
  exact pathCommit_zero _ _

/-- The state of `commitSource` for zero source blocks has zero digit vectors. -/
theorem commitSource_zero (two : 2 ≤ S.openingBase) (key : Key S) :
    (commitSource key (0 : Fin S.blocks → S.Source)).2.inner = 0 ∧
      PathDigits.firstTable S.chain (commitSource key (0 : Fin S.blocks → S.Source)).2.chain = 0 ∧
      PathDigits.secondTable S.chain (commitSource key (0 : Fin S.blocks → S.Source)).2.chain =
        0 := by
  have inner : (commitSource key (0 : Fin S.blocks → S.Source)).2.inner = 0 := funext fun i => by
    show decompose S.openingBase (key.inner.mulVec ((0 : Fin S.blocks → S.Source) i)) = 0
    rw [Pi.zero_apply, Matrix.mulVec_zero, decompose_zero two]
  exact ⟨inner, Shape.path_zero key inner⟩

/-- The state of `commit` for the zero table has zero digit vectors. -/
theorem commit_zero (source : 2 ≤ S.sourceBase) (opening : 2 ≤ S.openingBase) (key : Key S) :
    (commit key 0).2.source = 0 ∧ (commit key 0).2.inner = 0 ∧
      PathDigits.firstTable S.chain (commit key 0).2.chain = 0 ∧
      PathDigits.secondTable S.chain (commit key 0).2.chain = 0 := by
  have blocks : (commit key 0).2.source = 0 := funext fun i => by
    show decompose S.sourceBase (S.pack 0 i) = 0
    rw [show S.pack 0 i = 0 from funext fun x => ofCoeffs_zero, decompose_zero source]
  have inner : (commit key 0).2.inner = 0 := funext fun i => by
    show decompose S.openingBase (key.inner.mulVec ((commit key 0).2.source i)) = 0
    rw [blocks, Pi.zero_apply, Matrix.mulVec_zero, decompose_zero opening]
  exact ⟨blocks, inner, Shape.path_zero key inner⟩

/-- The flat witness of the honest prover of a packing fold for zero source blocks is the zero
table. -/
theorem honest_flat_zero (opening : 2 ≤ S.openingBase) (radix : 2 ≤ fold.responseBase)
    (β : Module.Basis (Fin O.planes) Zq E) (key : Key S) (openKey : OpeningKey O)
    (point : Point O E) (c : Fin S.blocks → Rq O.carrierDegree) :
    (honest β openKey point fold c (commitSource key 0).2).flat = 0 := by
  have state := commitSource_zero opening key
  have partials := honestPartials_zero (O := O) opening β point
  have opened := honestOpening_zero openKey
  refine Witness.flat_eq_zero ?_ partials state.1 state.2.1 state.2.2 ?_ ?_
  · show decompose fold.responseBase (response O c (0 : Fin S.blocks → S.Source)) = 0
    rw [response_zero, decompose_zero radix]
  · show PathDigits.firstTable O.chain (honestOpening openKey
      (honestPartials β point (0 : Fin S.blocks → S.Source))).1 = 0
    rw [partials]
    exact opened.1
  · show PathDigits.secondTable O.chain (honestOpening openKey
      (honestPartials β point (0 : Fin S.blocks → S.Source))).1 = 0
    rw [partials]
    exact opened.2

omit [Algebra Zq E] in
/-- The flat witness of the honest prover of a trace fold for zero source blocks is the zero
table. -/
theorem honestTrace_flat_zero (opening : 2 ≤ S.openingBase) (radix : 2 ≤ fold.responseBase)
    (key : Key S) (openKey : OpeningKey O) (subfield : E →+* Rq S.innerDegree)
    (position : Fin S.blockLength → E) (c : Fin S.blocks → Rq S.innerDegree) :
    (honestTrace openKey subfield position fold c (commitSource key 0).2).flat = 0 := by
  have state := commitSource_zero opening key
  have partials := tracePartials_zero (O := O) opening subfield position
  have opened := honestOpening_zero openKey
  refine Witness.flat_eq_zero ?_ partials state.1 state.2.1 state.2.2 ?_ ?_
  · show decompose fold.responseBase (traceResponse c (0 : Fin S.blocks → S.Source)) = 0
    rw [traceResponse_zero, decompose_zero radix]
  · show PathDigits.firstTable O.chain (honestOpening openKey
      (tracePartials O subfield position (0 : Fin S.blocks → S.Source))).1 = 0
    rw [partials]
    exact opened.1
  · show PathDigits.secondTable O.chain (honestOpening openKey
      (tracePartials O subfield position (0 : Fin S.blocks → S.Source))).1 = 0
    rw [partials]
    exact opened.2

end Honest

namespace Batch

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Algebra Zq E]

/-- The flat witness of the honest prover of a fold with opening groups for the zero tables is
the zero table. -/
theorem honest_flat_zero (sources : ∀ g, 2 ≤ (B.shape g).sourceBase)
    (openings : ∀ g, 2 ≤ (B.shape g).openingBase) (radix : ∀ g, 2 ≤ (B.fold g).responseBase)
    (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (key : (g : Fin n) → Key (B.shape g))
    (openKey : B.OpenKey) (point : (g : Fin n) → Point (B.opening g) E) (c : B.Challenge) :
    (honest β openKey point c fun g => (commit (key g) 0).2).flat = 0 := by
  have state := fun g => commit_zero (sources g) (openings g) (key g)
  have partials : honestPartials β point (fun g => (commit (key g) 0).2.source) = 0 :=
    funext fun column => by
      show Akita.honestPartials (O := B.opening column.1) (β column.1) (point column.1)
        (commit (key column.1) 0).2.source column.2 = 0
      rw [(state column.1).1, honestPartials_zero (openings column.1)]
      rfl
  have opened : PathDigits.firstTable B.chain (honestOpening openKey
        (honestPartials β point fun g => (commit (key g) 0).2.source)).1 = 0 ∧
      PathDigits.secondTable B.chain (honestOpening openKey
        (honestPartials β point fun g => (commit (key g) 0).2.source)).1 = 0 := by
    have image : stack openKey.matrix (fun _ : Fin 1 => (0 : B.Partials)) = 0 :=
      stack_zero openKey.matrix
    unfold honestOpening
    rw [partials, image]
    exact pathCommit_zero _ _
  have group : (fun g => ((honest β openKey point c fun g => (commit (key g) 0).2).group g).flat) =
      fun _ => 0 := funext fun g => Akita.Witness.flat_eq_zero
    (by
      show decompose (B.fold g).responseBase
        (response (B.opening g) (c g) (commit (key g) 0).2.source) = 0
      rw [(state g).1, response_zero, decompose_zero (radix g)])
    (funext fun column => congrFun partials ⟨g, column⟩) (state g).2.1 (state g).2.2.1
    (state g).2.2.2 opened.1 opened.2
  have first : (fun g => PathDigits.firstTable (B.shape g).chain (commit (key g) 0).2.chain) =
      fun _ => 0 := funext fun g => (state g).2.2.1
  have second : (fun g => PathDigits.secondTable (B.shape g).chain (commit (key g) 0).2.chain) =
      fun _ => 0 := funext fun g => (state g).2.2.2
  funext x
  unfold Witness.flat
  rw [group]
  show join _ _ (join _ (concat _ fun g => PathDigits.firstTable (B.shape g).chain
    (commit (key g) 0).2.chain) (join _ (PathDigits.firstTable B.chain (honestOpening openKey
      (honestPartials β point fun g => (commit (key g) 0).2.source)).1) (join _ (concat _ fun g =>
        PathDigits.secondTable (B.shape g).chain (commit (key g) 0).2.chain) (join _
          (PathDigits.secondTable B.chain (honestOpening openKey
            (honestPartials β point fun g => (commit (key g) 0).2.source)).1) _)))) x = _
  rw [first, second, opened.1, opened.2, concat_zero, concat_zero, concat_zero]
  simp only [join]
  split_ifs <;> rfl

end Batch

namespace Extract

open NightstreamFPrime.Spec.Compression.Lib (eqAt)

attribute [local instance low] Classical.propDecidable

section Fits

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E]
  [Algebra Zq E] {β : Module.Basis (Fin O.planes) Zq E} {key : Key S} {openKey : OpeningKey O}

/-- The table `liftNext` of a packing level for zero source blocks is the zero table. `shape`,
`radix`, `certified`, `path`: the conditions of `sourceClaim_satisfies`. `two`: `2 ≤ b⋆`. -/
theorem liftNext_zero (shape : S.Bounded) (radix : Radix fold.responseBase)
    (certified : fold.responseBase ≤ S.certifiedBase)
    (path : PathFits O.openDegree (1 * O.openRank) O.chain) (two : 2 ≤ S.certifiedBase) (μ : ℕ)
    (cells : O.cells fold ≤ 2 ^ μ) (digits : ℕ) (point : Point O E)
    (c : Fin S.blocks → Rq O.carrierDegree) :
    liftNext β key openKey fold μ cells digits 0 point c = 0 := by
  have satisfies := sourceClaim_satisfies (β := β) (key := key) (openKey := openKey)
    (fold := fold) 0 point c shape radix certified path
    (by rw [response_zero]; exact encodable_zero radix.two _)
  have holds := satisfies.flat μ cells
  have flat := honest_flat_zero (fold := fold) shape.opening.two radix.two β key openKey point c
  unfold liftNext
  rw [flat] at holds ⊢
  exact FlatRelation.completion_zero _ _ _ two holds.1

/-- The fit conditions `LiftFit` of a packing level hold for the bounded source of the zero
table, when `after` holds for the zero table. -/
theorem liftFit_zero (shape : S.Bounded) (radix : Radix fold.responseBase)
    (certified : fold.responseBase ≤ S.certifiedBase)
    (path : PathFits O.openDegree (1 * O.openRank) O.chain) (two : 2 ≤ S.certifiedBase) {μ : ℕ}
    {cells : O.cells fold ≤ 2 ^ μ} {digits : ℕ} {decode : E → Option (Rq O.carrierDegree)}
    {after : Table → Prop} (holds : after 0) :
    LiftFit β key openKey fold μ cells digits decode (fun i => S.embed (S.pack 0 i)) after := by
  rw [Shape.embed_pack_zero]
  exact fun point c _ => ⟨by rw [response_zero]; exact encodable_zero radix.two _, by
    rw [liftNext_zero shape radix certified path two]
    exact holds⟩

variable [Degree2 E] {m : ℕ}

theorem packed_zero : (packed (0 : Table) : ℕ → E) = 0 :=
  funext fun _ => packed_of_le (bound := 0) (fun _ _ => rfl) (Nat.zero_le _)

/-- The bounded source blocks of the zero packed polynomial are 0. -/
theorem traceSource_zero (quarter : 4 * m = S.innerDegree) :
    traceSource (S := S) quarter (0 : ℕ → E) = 0 := by
  have table : packTable quarter (0 : ℕ → E) = 0 := funext fun cell => by
    show coeff (psi quarter fun _ => (0 : E)) (cell % S.innerDegree) = 0
    rw [show psi quarter (fun _ => (0 : E)) = 0 by simp [psi], coeff_zero]
  funext i
  show S.embed (S.pack (packTable quarter (0 : ℕ → E)) i) = 0
  rw [table]
  exact congrFun Shape.embed_pack_zero i

/-- The fit conditions `TraceFit` of a trace level hold for the packed polynomial of the zero
table, when the zero response is admissible and `after` holds for the zero table. -/
theorem traceFit_zero (quarter : 4 * m = S.innerDegree) (opening : 2 ≤ S.openingBase)
    (radix : 2 ≤ fold.responseBase) {decode : Fin S.blocks → E → Option (Rq S.innerDegree)}
    {admissible : S.Source → Prop} {after : Table → Prop} (admitted : admissible 0)
    (holds : after 0) :
    TraceFit quarter key openKey fold decode admissible (packed 0) after := by
  intro position c _
  rw [packed_zero, traceSource_zero, traceResponse_zero]
  refine ⟨encodable_zero radix _, admitted, ?_⟩
  show after (honestTrace openKey (subfield quarter).toRingHom
    (fun x : Fin S.blockLength => eqAt position x) fold c
    (commitSource key (traceSource quarter (0 : ℕ → E))).2).flat
  rw [traceSource_zero, honestTrace_flat_zero opening radix]
  exact holds

end Fits

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Algebra Zq E]
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey}

/-- The fit conditions `BatchFit` of a fold with opening groups hold for the zero tables, when
`after` holds for the zero table. `shape`, `radix`, `certified`, `path`: the conditions of
`Batch.honestClaim_satisfies`. `two`: `2 ≤ base`. -/
theorem batchFit_zero (shape : ∀ g, (B.shape g).Honest)
    (radix : ∀ g, Radix (B.fold g).responseBase)
    (certified : ∀ g, (B.fold g).responseBase ≤ (B.shape g).certifiedBase)
    (path : PathFits B.openDegree (1 * B.openRank) B.chain) {base : ℕ} (two : 2 ≤ base) {μ : ℕ}
    {cells : B.cells ≤ 2 ^ μ} {digits : ℕ}
    {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))} {after : Table → Prop}
    (holds : after 0) :
    BatchFit β key openKey μ cells base digits decode (fun _ => 0) after := by
  intro point weight c _
  have encodable : ∀ g, Encodable (B.fold g).responseBase (B.fold g).responseDigits
      (response (B.opening g) (c g) (commit (key g) 0).2.source) := fun g => by
    rw [(commit_zero (shape g).source.two (shape g).opening.two (key g)).1, response_zero]
    exact encodable_zero (radix g).two _
  refine ⟨encodable, ?_⟩
  have satisfies := Batch.honestClaim_satisfies (β := β) (key := key) (openKey := openKey)
    (fun _ => 0) point weight c shape radix certified path encodable
  have rows := satisfies.flat μ cells
  have flat := Batch.honest_flat_zero (fun g => (shape g).source.two)
    (fun g => (shape g).opening.two) (fun g => (radix g).two) β key openKey point c
  unfold batchNext
  rw [flat] at rows ⊢
  rw [FlatRelation.completion_zero _ _ _ two rows.1]
  exact holds

end Groups

end Extract

end NightstreamFPrime.Spec.Compression.Akita
