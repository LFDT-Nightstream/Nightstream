import NightstreamFPrime.Spec.Compression.Akita.Instance
import NightstreamFPrime.Spec.Compression.Akita.SetupView

/-! The matrices of a key as the views of one shared setup vector, and the solutions of Module-SIS
for these views (Akita, ePrint 2026/1983, revision of 2026-10-15, Section 4.3 with (35),
Definition 3.2, Table 3 and Lemma 10.20).

- `Setup.Solved list S` states that an instance `I` of `list` has a solution of Module-SIS for
  the view `M_I(S)` of the flat setup vector `S`: the output of the algorithm `B` of Akita
  Lemma 10.20 (`Setup.solved_iff`).
- `Setup.shared_eq` states that `Setup.shared S d n m code` is the view of each instance with the
  degree `d`, `n` rows and `m` columns, with the columns in the order `code`. `Setup.kernel_view`
  moves a solution for this matrix to a solution for the view, with zeros in the other columns.
- `PathShared`, `Key.Shared`, `OpeningKey.Shared` and `Batch.OpenKey.Shared` state that each
  matrix of a key is the view of `S` with the degree, the rank and the width of its instance in
  `Shape.instances`, `OpeningShape.instances` or `Batch.openInstances`. The Akita code reads each
  matrix of each level from one stream from the offset 0
  (`akita-params/src/layout/flat_matrix.rs`). `Key.ofSetup`, `OpeningKey.ofSetup` and
  `Batch.OpenKey.ofSetup` are keys with these views.
- `Key.solved_inner`, `Key.solved_outer`, `OpeningKey.solved`, `level_solved` and
  `Batch.solved` are `Key.reduces_inner`, `Key.reduces_outer`, `OpeningKey.reduces`,
  `level_reduces` and `Batch.reduces` for these keys: each solution that a witness of `R_bind`
  holds gives `Setup.Solved` for each list that has the instances of the level. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open Setup (Instance)

variable {N : ℕ}

namespace Setup

/-- An instance `I` of `list` has a solution of Module-SIS for the view `M_I(S)` of the flat
setup vector `S` (Akita Definition 3.2 and (35)). -/
def Solved (list : List Instance) (S : Fin N → Zq) : Prop :=
  ∃ I ∈ list, Nonempty (I.Kernel (view I S))

/-- `Solved` is the event of the algorithm `B` of Akita Lemma 10.20 for the inventory
`list.get`: an index of the list and a solution for the view of its instance. -/
theorem solved_iff (list : List Instance) (S : Fin N → Zq) :
    Solved list S ↔ ∃ i : Fin list.length, Nonempty ((list.get i).Kernel (view (list.get i) S)) :=
  ⟨fun ⟨_, member, z⟩ => by
      obtain ⟨i, rfl⟩ := List.mem_iff_get.mp member
      exact ⟨i, z⟩,
    fun ⟨i, z⟩ => ⟨_, List.get_mem _ _, z⟩⟩

/-- `shared S d n width code` is the view of each instance with the degree `d`, `n` rows and
`width` columns, with the column `c` at the column `code c`. -/
theorem shared_eq (S : Fin N → Zq) (d n width : ℕ) [NeZero d] {cols : Type}
    (code : cols → Fin width) (euclidean : Bool) (bound : ℕ) :
    shared S d n width code = (view ⟨d, n, width, euclidean, bound⟩ S).submatrix id code :=
  rfl

/-- A solution for `shared S d n width code` at the bound `bound` is a solution for the view of
the instance `(d, n, width, p, bound)`, with zeros in the columns outside `code`
(`ShortKernel.carry`). -/
theorem kernel_view {S : Fin N → Zq} {d n width : ℕ} [NeZero d] {cols : Type} [Fintype cols]
    {code : cols → Fin width} (injective : Function.Injective code) (euclidean : Bool)
    (bound : ℕ)
    (z : cond euclidean (ShortKernelSq (shared S d n width code) bound)
      (ShortKernel (shared S d n width code) bound)) :
    Nonempty (Instance.Kernel ⟨d, n, width, euclidean, bound⟩
      (view ⟨d, n, width, euclidean, bound⟩ S)) := by
  cases euclidean
  · exact ⟨ShortKernel.carry id ⟨code, injective⟩ (fun _ _ => rfl) le_rfl z⟩
  · exact ⟨ShortKernelSq.carry id ⟨code, injective⟩ (fun _ _ => rfl) le_rfl z⟩

/-- A solution for `shared S d n width code` gives `Solved list S` when `list` has the instance
`(d, n, width, p, bound)`. -/
theorem Solved.of_kernel {list : List Instance} {S : Fin N → Zq} {d n width : ℕ} [NeZero d]
    {cols : Type} [Fintype cols] {code : cols → Fin width} {euclidean : Bool} {bound : ℕ}
    (member : (⟨d, n, width, euclidean, bound⟩ : Instance) ∈ list)
    (injective : Function.Injective code)
    (z : cond euclidean (ShortKernelSq (shared S d n width code) bound)
      (ShortKernel (shared S d n width code) bound)) : Solved list S :=
  ⟨_, member, kernel_view injective euclidean bound z⟩

end Setup

open Setup (Solved)

variable (S : Fin N → Zq)

/-- The compression maps of a path are the views of `S` with the degree, the rank and the width
of their instances in `pathInstances`. -/
def PathShared : (chain : Option ChainShape) → PathKey chain → Prop
  | none, _ => True
  | some c, key =>
      ChainKey.first key = Setup.view ⟨c.firstDegree, c.firstRank, c.firstWidth, false, 1⟩ S ∧
        ChainKey.second key = Setup.view ⟨c.secondDegree, c.secondRank, c.secondWidth, false, 1⟩ S

/-- The compression maps of a path as the views of `S`. -/
noncomputable def PathKey.ofSetup : (chain : Option ChainShape) → PathKey chain
  | none => PUnit.unit
  | some c => (⟨Setup.view ⟨c.firstDegree, c.firstRank, c.firstWidth, false, 1⟩ S,
      Setup.view ⟨c.secondDegree, c.secondRank, c.secondWidth, false, 1⟩ S⟩ : ChainKey c)

theorem PathKey.ofSetup_shared : (chain : Option ChainShape) →
    PathShared S chain (PathKey.ofSetup S chain)
  | none => trivial
  | some _ => ⟨rfl, rfl⟩

variable {S}

/-- A solution for a compression map that is a view of `S` gives `Solved list S`, when `list` has
the instances of the path. -/
theorem PathShared.solved {list : List Instance} {chain : Option ChainShape} {key : PathKey chain}
    (shared : PathShared S chain key) (within : pathInstances chain ⊆ list)
    (z : PathKernel chain key) : Solved list S := by
  cases chain with
  | none => exact z.elim
  | some c =>
      obtain ⟨first, second⟩ := shared
      rcases (z : ShortKernel (ChainKey.first key) 1 ⊕ ShortKernel (ChainKey.second key) 1) with
        z | z
      · rw [first] at z
        exact ⟨_, within List.mem_cons_self, ⟨z⟩⟩
      · rw [second] at z
        exact ⟨_, within (List.mem_cons_of_mem _ List.mem_cons_self), ⟨z⟩⟩

section Level

variable {T : Shape} {O : OpeningShape T} {list : List Instance}

variable (S) in
/-- The matrices of the commitment key are the views of `S`: `A` and `B` in the orders
`innerSetupCode` and `outerSetupCode` of the witness layout, with the degrees, the ranks and the
widths of `Shape.innerInstance` and `Shape.outerInstance`, and the compression maps. -/
def Key.Shared (key : Key T) : Prop :=
  key.inner = Setup.shared S T.innerDegree T.innerRank _ T.innerSetupCode ∧
    key.outer = Setup.shared S T.outerDegree T.outerRank _ T.outerSetupCode ∧
      PathShared S T.chain key.chain

variable (S T) in
/-- The commitment key whose matrices are the views of `S`. -/
noncomputable def Key.ofSetup : Key T :=
  ⟨Setup.shared S T.innerDegree T.innerRank _ T.innerSetupCode,
    Setup.shared S T.outerDegree T.outerRank _ T.outerSetupCode, PathKey.ofSetup S T.chain⟩

variable (S T) in
theorem Key.ofSetup_shared : (Key.ofSetup S T).Shared S :=
  ⟨rfl, rfl, PathKey.ofSetup_shared S T.chain⟩

/-- The order of the columns of `D` of one group in the witness layout: the block, the subcolumn
and the digit, in this order (Akita (69)). Its width is the width of `OpeningShape.openInstance`. -/
def OpeningShape.openSetupCode (O : OpeningShape T) :
    O.OpenColumn ≃ Fin (T.blocks * (O.subcolumns * T.openingDigits)) :=
  (Equiv.prodCongr (Equiv.refl _) finProdFinEquiv).trans finProdFinEquiv

variable (S) in
/-- The matrices of the opening key are the views of `S`: `D` in the order `openSetupCode`, with
the degree, the rank and the width of `OpeningShape.openInstance`, and the compression maps. -/
def OpeningKey.Shared (key : OpeningKey O) : Prop :=
  key.matrix = Setup.shared S O.openDegree O.openRank _ O.openSetupCode ∧
    PathShared S O.chain key.chain

variable (S O) in
/-- The opening key whose matrices are the views of `S`. -/
noncomputable def OpeningKey.ofSetup : OpeningKey O :=
  ⟨Setup.shared S O.openDegree O.openRank _ O.openSetupCode, PathKey.ofSetup S O.chain⟩

variable (S O) in
theorem OpeningKey.ofSetup_shared : (OpeningKey.ofSetup S O).Shared S :=
  ⟨rfl, PathKey.ofSetup_shared S O.chain⟩

/-- `Key.reduces_inner` for a key of views of `S`: a solution for `A` at the bound `bound` gives
`Solved list S`. -/
theorem Key.solved_inner {key : Key T} {euclidean : Bool} {bound : ℕ} (shared : key.Shared S)
    (within : T.instances euclidean bound ⊆ list)
    (z : cond euclidean (ShortKernelSq key.inner bound) (ShortKernel key.inner bound)) :
    Solved list S := by
  rw [shared.1] at z
  exact Setup.Solved.of_kernel (within List.mem_cons_self) T.innerSetupCode.injective z

/-- `Key.reduces_outer` for a key of views of `S`. -/
theorem Key.solved_outer {key : Key T} {euclidean : Bool} {bound : ℕ} (shared : key.Shared S)
    (within : T.instances euclidean bound ⊆ list) (z : OuterBreak key) : Solved list S := by
  rcases z with z | z
  · rw [shared.2.1] at z
    exact Setup.Solved.of_kernel (euclidean := false)
      (within (List.mem_cons_of_mem _ List.mem_cons_self)) T.outerSetupCode.injective z
  · exact PathShared.solved shared.2.2
      (fun v member => within (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ member))) z

/-- `OpeningKey.reduces` for a key of views of `S`. -/
theorem OpeningKey.solved {key : OpeningKey O} (shared : key.Shared S)
    (within : O.instances ⊆ list) (z : OpenBreak key) : Solved list S := by
  rcases z with z | z
  · rw [shared.1] at z
    exact Setup.Solved.of_kernel (euclidean := false) (within List.mem_cons_self)
      O.openSetupCode.injective z
  · exact PathShared.solved shared.2 (fun v member => within (List.mem_cons_of_mem _ member)) z

/-- `level_reduces` for keys of views of `S`: each solution that a witness of `R_bind` of one
level holds gives `Solved list S`, when `list` has the instances of the shapes of the level. -/
theorem level_solved {key : Key T} {openKey : OpeningKey O} {euclidean : Bool} {bound : ℕ}
    (shared : key.Shared S) (sharedOpen : openKey.Shared S)
    (within : T.instances euclidean bound ⊆ list) (withinOpen : O.instances ⊆ list)
    (z : cond euclidean (ShortKernelSq key.inner bound) (ShortKernel key.inner bound) ⊕
      (OuterBreak key ⊕ OpenBreak openKey)) : Solved list S := by
  rcases z with z | z | z
  · exact Key.solved_inner shared within z
  · exact Key.solved_outer shared within z
  · exact OpeningKey.solved sharedOpen withinOpen z

end Level

namespace Batch

variable {n : ℕ} {B : Batch (Fin n)} {list : List Instance}

variable (S) in
/-- The matrices of the shared opening key are the views of `S`: `D` in the order
`openSetupCode`, with the degree, the rank and the width of `Batch.openInstance`, and the
compression maps. -/
def OpenKey.Shared (key : B.OpenKey) : Prop :=
  key.matrix = Setup.shared S B.openDegree B.openRank _ B.openSetupCode ∧
    PathShared S B.chain key.chain

variable (S B) in
/-- The shared opening key whose matrices are the views of `S`. -/
noncomputable def OpenKey.ofSetup : B.OpenKey :=
  ⟨Setup.shared S B.openDegree B.openRank _ B.openSetupCode, PathKey.ofSetup S B.chain⟩

variable (S B) in
theorem OpenKey.ofSetup_shared : (OpenKey.ofSetup S B).Shared S :=
  ⟨rfl, PathKey.ofSetup_shared S B.chain⟩

/-- `Batch.reduces` for keys of views of `S`: each solution that a witness of `R_bind` of a level
with opening groups holds gives `Solved list S`, when `list` has the instances of the shapes of
the level. -/
theorem solved {key : (g : Fin n) → Key (B.shape g)} {openKey : B.OpenKey}
    {innerBound : Fin n → ℕ} (shared : ∀ g, (key g).Shared S) (sharedOpen : openKey.Shared S)
    (within : ∀ g, (B.shape g).instances false (innerBound g) ⊆ list)
    (withinOpen : B.openInstances ⊆ list)
    (z : (Σ g, ShortKernel (key g).inner (innerBound g)) ⊕
      ((Σ g, OuterBreak (key g)) ⊕ B.OpenBreak openKey)) : Solved list S := by
  rcases z with ⟨g, z⟩ | ⟨g, z⟩ | z | z
  · exact Key.solved_inner (euclidean := false) (shared g) (within g) z
  · exact Key.solved_outer (shared g) (within g) z
  · rw [sharedOpen.1] at z
    exact Setup.Solved.of_kernel (euclidean := false) (withinOpen List.mem_cons_self)
      B.openSetupCode.injective z
  · exact PathShared.solved sharedOpen.2
      (fun v member => withinOpen (List.mem_cons_of_mem _ member)) z

end Batch

end NightstreamFPrime.Spec.Compression.Akita
