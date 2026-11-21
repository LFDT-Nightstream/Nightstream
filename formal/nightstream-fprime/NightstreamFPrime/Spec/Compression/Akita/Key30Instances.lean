import NightstreamFPrime.Spec.Compression.Akita.Key20Instances
import NightstreamFPrime.Spec.Compression.Akita.Key30Opening

/-! The instances of Module-SIS of the opening of the schedule file
`crates/ns-compress/artifacts/key30.aks` (Akita, ePrint 2026/1983, revision of 2026-10-15,
Definition 3.2, Table 3, Section 9 with Lemma 9.1, Section 10.3 and Appendix G).

- `views r` has the instance of each matrix of the tower of `Extract/OffTower.lean` for the
  numbers of the row `r` of `Key30.lean`, at the bound of the witness of `R_bind` of its level:
  15 of the root, 11 of the level 1, 7 of each of the levels 2 to 4, 3 of each of the levels 5
  and 6, and 1 of the terminal. The group of the setup prefix is the group 0 of the level 1: the
  witness `Batch.pinBreak` of Akita Lemma 9.1 is a solution for a matrix of this group at the
  bound of the level.
- `instances` has the 18 instances that are not below another instance, for the two rows. A
  coefficient instance has the bound of its view. A Euclidean instance has the power of two that
  the file has in `collision_l2_sq`, which is the bound that the Akita code prices.
  `tests/AkitaSchedules.lean` compares the list with the file.
- `views_covered` states that each view of each row is below an instance of the list, so each
  bound of a witness of `R_bind` is at most the bound that the file prices. `instances_exact`
  gives the numbers of the list, and that the list has no instance without a view and no two
  instances in the order `Instance.Below`.
- `reduced` and `reduced_key` state that the list is complete: each solution of a witness of
  `R_bind` of the tower, and each solution of the reduction `tableCollision` for the key
  commitment at a bound of at most `η_A`, gives a solution of an instance of the list
  (`Reduces`).
- `Shared S` states that each matrix of the opening is the view of the flat setup vector `S` with
  the degree, the rank and the width of its instance in `views r`, and `shared_ofSetup` gives
  such keys. `reduced_shared` and `reduced_key_shared` are `reduced` and `reduced_key` for these
  keys: each solution gives a solution of Module-SIS for the view of `S` of an instance of
  `views r` (`Setup.Solved`). `Tower.shared_of_views` gives `Tower.Shared` from these keys at the
  setup prefix.
- `views_fit` states that each view reads at most the `2^23` cells of the setup prefix, and that
  `A` of the key group reads all of them. `shared_setup` is Akita Lemma 10.20 (i) for the 54 views
  of a row.

Lean has no statement on the hardness of an instance. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key30

open Setup (Instance)
open Extract
open Key20 (maps levelViews)

/-- The instances of the matrices of the opening of the row `r` of `key30.aks`, each at the bound
of the witness of `R_bind` of its level (`Extract.offRoot`): `η_A = 2·κ̄_1·Δ^cert_f` of Akita
(176) on the coefficient route, `64·Γ²·S_max` of Akita (182) on the Euclidean route, `b⋆ − 1` for
`B` and `D`, and 1 for a compression map. -/
def views (r : Row) : List Instance :=
  ((List.finRange 3).flatMap fun g => ((root r).shape g).instances false
      (2 * (2 * mass g) * ((root r).fold g).sourceBound ((root r).shape g))) ++
    ((root r).openInstances ++
      (((List.finRange 2).flatMap fun g => ((level₁ r).shape g).instances false
          (2 * (2 * mass₁ g) * ((level₁ r).fold g).sourceBound ((level₁ r).shape g))) ++
        ((level₁ r).openInstances ++
          (levelViews (level₂ r) false
              (2 * (2 * 51) * (level₂ r).fold.sourceBound (level₂ r).shape) ++
            (levelViews level₃ false (2 * (2 * 51) * level₃.fold.sourceBound level₃.shape) ++
              (levelViews level₄ false (2 * (2 * 31) * level₄.fold.sourceBound level₄.shape) ++
                (levelViews level₅ true (64 * norm₅.operator ^ 2 * norm₅.limit) ++
                  (levelViews level₆ true (64 * norm₆.operator ^ 2 * norm₆.limit) ++
                    [terminal.innerInstance true
                      (64 * normTerminal.operator ^ 2 * normTerminal.limit)]))))))))

/-- The instances of Module-SIS of the opening of the two rows of `key30.aks`, at the bounds that
the file has: the second and the first compression map at their largest width, `D` and `B` of the
levels 5 and 6, `A` of the level 6 and of the terminal, `A` of the level 5, `B` of the level 4,
`B` of the level 2 for `W` of 27 variables, `B` of `W`, `A` of the levels 4, 3 and 2, `B` of the
setup prefix, `D` of the root, `A` of the setup prefix, of the witness of the level 1 and of `V`,
and `A` of the key polynomial. -/
def instances : List Instance :=
  [⟨(maps 1024).secondDegree, (maps 1024).secondRank, (maps 1024).secondWidth, false, 1⟩,
    ⟨(maps 1024).firstDegree, (maps 1024).firstRank, (maps 1024).firstWidth, false, 1⟩,
    level₆.opening.openInstance,
    level₆.shape.outerInstance,
    level₆.shape.innerInstance true (2 ^ 46),
    level₅.shape.innerInstance true (2 ^ 45),
    level₄.shape.outerInstance,
    (level₂ w27).shape.outerInstance,
    ((root w27).shape 1).outerInstance,
    level₄.shape.innerInstance false (level₄.fold.innerBound level₄.shape),
    level₃.shape.innerInstance false (level₃.fold.innerBound level₃.shape),
    (level₂ w27).shape.innerInstance false ((level₂ w27).fold.innerBound (level₂ w27).shape),
    shapeS.outerInstance,
    (root w27).openInstance,
    shapeS.innerInstance false (((level₁ w27).fold 0).innerBound shapeS),
    (shapeW₁ w27).innerInstance false (((level₁ w27).fold 1).innerBound (shapeW₁ w27)),
    shapeV.innerInstance false (((root w27).fold 2).innerBound shapeV),
    Shape.key30.innerInstance false (Descriptor.key30.innerBound Shape.key30)]

/-- Each view of each row is below an instance of the list: an instance of the list has its
degree, its rank and its norm, a width that is not smaller and a bound that is not smaller. So
the bound of each witness of `R_bind` is at most the bound that the file prices. -/
theorem views_covered : ∀ r ∈ rows, Covered instances (views r) := by
  unfold Covered
  decide

/-- The list `instances` and the views. Each view of each row is below an instance of the list.
The list has 18 instances `(d, n, m, p, η)` for the 54 views of each row. Each instance has the
width of a view of a row below it, with the bound of this view on the coefficient route and with
a bound below two times the bound of this view on the Euclidean route. No instance is below
another instance. -/
theorem instances_exact :
    (∀ r ∈ rows, Covered instances (views r)) ∧
      instances.map Instance.key =
        [(32, 1, 128, false, 1), (64, 1, 1024, false, 1), (64, 2, 77, false, 63),
          (64, 3, 616, false, 63), (64, 8, 256, true, 2 ^ 46), (64, 8, 512, true, 2 ^ 45),
          (128, 1, 416, false, 31), (128, 1, 4224, false, 15), (128, 1, 11264, false, 7),
          (128, 4, 512, false, 126852), (128, 5, 512, false, 835380),
          (128, 6, 2048, false, 835380), (256, 1, 8192, false, 15), (256, 1, 28160, false, 7),
          (512, 2, 1024, false, 8126340), (512, 2, 2048, false, 507780),
          (512, 3, 5120, false, 2080374660), (2048, 1, 4096, false, 16642998148)] ∧
      (∀ r ∈ rows, (views r).length = 54) ∧
      (∀ I ∈ instances, ∃ r ∈ rows, ∃ v ∈ views r, v.Below I ∧ v.cols = I.cols ∧
        cond I.euclidean (I.bound < 2 * v.bound) (v.bound = I.bound)) ∧
      instances.Pairwise fun I J => ¬ I.Below J ∧ ¬ J.Below I :=
  ⟨views_covered, by decide⟩

variable {r : Row} {key : (g : Fin 3) → Key ((root r).shape g)} {openKey : (root r).OpenKey}
  {key₁ : (g : Fin 2) → Key ((level₁ r).shape g)} {openKey₁ : (level₁ r).OpenKey}
  {key₂ : Key (level₂ r).shape} {openKey₂ : OpeningKey (level₂ r).opening}
  {key₃ : Key level₃.shape} {openKey₃ : OpeningKey level₃.opening}
  {key₄ : Key level₄.shape} {openKey₄ : OpeningKey level₄.opening}
  {key₅ : Key level₅.shape} {openKey₅ : OpeningKey level₅.opening}
  {key₆ : Key level₆.shape} {openKey₆ : OpeningKey level₆.opening}
  {matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
    (Rq terminal.innerDegree)}

/-- A matrix of the opening of the row `r` of `key30.aks` has a solution of Module-SIS that gives
a solution of an instance of `list` (`Reduces`): a matrix of a group of the root, of the shared
opening path of the root, of a group of the level 1, of the shared opening path of the level 1,
of one of the levels 2 to 6, or `A_term`. The levels 1 to 4 have the norm `‖·‖_∞` for `A`, and
the levels 5 and 6 and the terminal have the norm `‖·‖_2`. -/
def Reduced (list : List Instance) (key : (g : Fin 3) → Key ((root r).shape g))
    (openKey : (root r).OpenKey) (key₁ : (g : Fin 2) → Key ((level₁ r).shape g))
    (openKey₁ : (level₁ r).OpenKey) (key₂ : Key (level₂ r).shape)
    (openKey₂ : OpeningKey (level₂ r).opening) (key₃ : Key level₃.shape)
    (openKey₃ : OpeningKey level₃.opening) (key₄ : Key level₄.shape)
    (openKey₄ : OpeningKey level₄.opening) (key₅ : Key level₅.shape)
    (openKey₅ : OpeningKey level₅.opening) (key₆ : Key level₆.shape)
    (openKey₆ : OpeningKey level₆.opening)
    (matrix : Matrix (Fin terminal.innerRank)
      (Fin terminal.blockLength × Fin terminal.sourceDigits) (Rq terminal.innerDegree)) : Prop :=
  ((∃ g, (key g).Reduces list false) ∨ openKey.Reduces list) ∨
    ((∃ g, (key₁ g).Reduces list false) ∨ openKey₁.Reduces list) ∨
    (key₂.Reduces list false ∨ openKey₂.Reduces list) ∨
    (key₃.Reduces list false ∨ openKey₃.Reduces list) ∨
    (key₄.Reduces list false ∨ openKey₄.Reduces list) ∨
    (key₅.Reduces list true ∨ openKey₅.Reduces list) ∨
    (key₆.Reduces list true ∨ openKey₆.Reduces list) ∨ Akita.Reduces list matrix true

/-- The list `instances` is complete for the witnesses of `R_bind` of the opening of a row of
`key30.aks` (Akita Table 3, Lemma 10.20 and Lemma 9.1). `brk` is a witness of `R_bind` of the
tower: for the root, for one of the levels 1 to 6 or for the terminal. A witness of the level 1
is also the witness `Batch.pinBreak` of Akita Lemma 9.1 for the group of the setup prefix, and
the collision of two tables of the witness group. `held`: it holds a solution of Module-SIS. Then
a matrix of the opening has a solution that gives a solution of an instance of the list, for each
matrix of this instance with the columns of that matrix, at the bound that the file prices. -/
theorem reduced (member : r ∈ rows) {weight₅ weight₆ weightL boundL : ℕ}
    (brk : (root r).LevelBreak key openKey
        (fun g => 2 * (2 * mass g) * ((root r).fold g).sourceBound ((root r).shape g)) ⊕
      (level₁ r).LevelBreak key₁ openKey₁
        (fun g => 2 * (2 * mass₁ g) * ((level₁ r).fold g).sourceBound ((level₁ r).shape g)) ⊕
      LevelBreak key₂ openKey₂ (2 * (2 * 51) * (level₂ r).fold.sourceBound (level₂ r).shape) ⊕
      LevelBreak key₃ openKey₃ (2 * (2 * 51) * level₃.fold.sourceBound level₃.shape) ⊕
      Route.Break key₄ openKey₄ 31 (level₄.fold.sourceBound level₄.shape) .coefficient ⊕
      Route.Break key₅ openKey₅ weight₅ (level₅.fold.sourceBound level₅.shape)
        (.euclidean norm₅.operator norm₅.limit) ⊕
      Route.Break key₆ openKey₆ weight₆ (level₆.fold.sourceBound level₆.shape)
        (.euclidean norm₆.operator norm₆.limit) ⊕
      Option ((Route.euclidean normTerminal.operator normTerminal.limit).Kernel matrix weightL
        boundL))
    (held : Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
      (Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
        (Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
          (Sum.elim (fun b => Option.isSome b = true)
            (fun b => Option.isSome b = true))))))) brk) :
    Reduced instances key openKey key₁ openKey₁ key₂ openKey₂ key₃ openKey₃ key₄ openKey₄ key₅
      openKey₅ key₆ openKey₆ matrix := by
  have covered := views_covered r member
  have one := covered.right.right
  have rest := one.right.right
  rcases brk with b | b | b | b | b | b | b | b
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inl (Batch.reduces
      (fun g v inside => covered.left v
        (List.mem_flatMap.mpr ⟨g, List.mem_finRange g, inside⟩))
      covered.right.left z)
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inl (Batch.reduces
      (fun g v inside => one.left v (List.mem_flatMap.mpr ⟨g, List.mem_finRange g, inside⟩))
      one.right.left z))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inl (level_reduces (euclidean := false) rest.left.left
      rest.left.right z)))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inr (Or.inl (level_reduces (euclidean := false)
      rest.right.left.left rest.right.left.right z))))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (level_reduces (euclidean := false)
      rest.right.right.left.left rest.right.right.left.right z)))))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (level_reduces (euclidean := true)
      rest.right.right.right.left.left rest.right.right.right.left.right z))))))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (level_reduces
      (euclidean := true) rest.right.right.right.right.left.left
      rest.right.right.right.right.left.right z)))))))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Reduces.of_kernel
      (euclidean := true) (by simp only [Fintype.card_prod, Fintype.card_fin])
      (rest.right.right.right.right.right _ List.mem_cons_self) z)))))))

/-- The list `instances` is complete for the key commitment. A solution of the reduction
`tableCollision` for the matrices of the key group at a bound `innerBound ≤ η_A` gives a solution
of an instance of the list: `η_A` is the bound of `A` of the key group in the file. -/
theorem reduced_key (member : r ∈ rows) {innerBound : ℕ}
    (inner : innerBound ≤ ((root r).fold 0).innerBound ((root r).shape 0))
    (z : ShortKernel (key 0).inner innerBound ⊕ OuterBreak (key 0)) :
    (key 0).Reduces instances false := by
  have covered : Covered instances (((root r).shape 0).instances false
      (2 * (2 * mass 0) * ((root r).fold 0).sourceBound ((root r).shape 0))) :=
    fun v inside => (views_covered r member).left v
      (List.mem_flatMap.mpr ⟨0, List.mem_finRange 0, inside⟩)
  rcases z with z | z
  · exact Key.reduces_inner (euclidean := false) covered
      (ShortKernel.carry id (Function.Embedding.refl _) (fun _ _ => rfl)
        (inner.trans_eq rfl) z)
  · exact Key.reduces_outer covered z

variable {N : ℕ}

/-- The matrices of the opening of the row `r` of `key30.aks` are the views of one flat setup
vector `S` (Akita Section 4.3 and (35)): the keys of the root, of the shared opening path of the
root, of the level 1, of the shared opening path of the level 1, of the levels 2 to 6 and `A_term`.
Each matrix is the view with the degree, the rank and the width of its instance in `views r`. -/
def Shared (S : Fin N → Zq) (key : (g : Fin 3) → Key ((root r).shape g))
    (openKey : (root r).OpenKey) (key₁ : (g : Fin 2) → Key ((level₁ r).shape g))
    (openKey₁ : (level₁ r).OpenKey) (key₂ : Key (level₂ r).shape)
    (openKey₂ : OpeningKey (level₂ r).opening) (key₃ : Key level₃.shape)
    (openKey₃ : OpeningKey level₃.opening) (key₄ : Key level₄.shape)
    (openKey₄ : OpeningKey level₄.opening) (key₅ : Key level₅.shape)
    (openKey₅ : OpeningKey level₅.opening) (key₆ : Key level₆.shape)
    (openKey₆ : OpeningKey level₆.opening)
    (matrix : Matrix (Fin terminal.innerRank)
      (Fin terminal.blockLength × Fin terminal.sourceDigits) (Rq terminal.innerDegree)) : Prop :=
  ((∀ g, (key g).Shared S) ∧ openKey.Shared S) ∧ ((∀ g, (key₁ g).Shared S) ∧ openKey₁.Shared S) ∧
    (key₂.Shared S ∧ openKey₂.Shared S) ∧ (key₃.Shared S ∧ openKey₃.Shared S) ∧
    (key₄.Shared S ∧ openKey₄.Shared S) ∧ (key₅.Shared S ∧ openKey₅.Shared S) ∧
    (key₆.Shared S ∧ openKey₆.Shared S) ∧
    matrix = Setup.shared S terminal.innerDegree terminal.innerRank _ terminal.innerSetupCode

variable (r) in
/-- The keys whose matrices are the views of `S` satisfy `Shared`. -/
theorem shared_ofSetup (S : Fin N → Zq) :
    Shared S (fun g => Key.ofSetup S ((root r).shape g)) (Batch.OpenKey.ofSetup S (root r))
      (fun g => Key.ofSetup S ((level₁ r).shape g)) (Batch.OpenKey.ofSetup S (level₁ r))
      (Key.ofSetup S _) (OpeningKey.ofSetup S _) (Key.ofSetup S _) (OpeningKey.ofSetup S _)
      (Key.ofSetup S _) (OpeningKey.ofSetup S _) (Key.ofSetup S _) (OpeningKey.ofSetup S _)
      (Key.ofSetup S _) (OpeningKey.ofSetup S _)
      (Setup.shared S terminal.innerDegree terminal.innerRank _ terminal.innerSetupCode) :=
  ⟨⟨fun _ => Key.ofSetup_shared S _, Batch.OpenKey.ofSetup_shared S _⟩,
    ⟨fun _ => Key.ofSetup_shared S _, Batch.OpenKey.ofSetup_shared S _⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩, rfl⟩

/-- `reduced` for the views of one setup vector `S`: each solution of a witness of `R_bind` of
the tower of the row `r` of `key30.aks` gives a solution of Module-SIS for the view `M_I(S)` of
an instance `I` of `views r` itself, with the width of its matrix and the bound of the witness of
its level (`Setup.Solved`). `shared`: each matrix of the opening is the view of `S` with the
numbers of its instance. -/
theorem reduced_shared {weight₅ weight₆ weightL boundL : ℕ} {S : Fin N → Zq}
    (shared : Shared S key openKey key₁ openKey₁ key₂ openKey₂ key₃ openKey₃ key₄ openKey₄ key₅
      openKey₅ key₆ openKey₆ matrix)
    (brk : (root r).LevelBreak key openKey
        (fun g => 2 * (2 * mass g) * ((root r).fold g).sourceBound ((root r).shape g)) ⊕
      (level₁ r).LevelBreak key₁ openKey₁
        (fun g => 2 * (2 * mass₁ g) * ((level₁ r).fold g).sourceBound ((level₁ r).shape g)) ⊕
      LevelBreak key₂ openKey₂ (2 * (2 * 51) * (level₂ r).fold.sourceBound (level₂ r).shape) ⊕
      LevelBreak key₃ openKey₃ (2 * (2 * 51) * level₃.fold.sourceBound level₃.shape) ⊕
      Route.Break key₄ openKey₄ 31 (level₄.fold.sourceBound level₄.shape) .coefficient ⊕
      Route.Break key₅ openKey₅ weight₅ (level₅.fold.sourceBound level₅.shape)
        (.euclidean norm₅.operator norm₅.limit) ⊕
      Route.Break key₆ openKey₆ weight₆ (level₆.fold.sourceBound level₆.shape)
        (.euclidean norm₆.operator norm₆.limit) ⊕
      Option ((Route.euclidean normTerminal.operator normTerminal.limit).Kernel matrix weightL
        boundL))
    (held : Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
      (Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
        (Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
          (Sum.elim (fun b => Option.isSome b = true)
            (fun b => Option.isSome b = true))))))) brk) :
    Setup.Solved (views r) S := by
  obtain ⟨⟨sharedRoot, sharedOpen⟩, ⟨shared₁, sharedOpen₁⟩, ⟨shared₂, sharedOpen₂⟩,
    ⟨shared₃, sharedOpen₃⟩, ⟨shared₄, sharedOpen₄⟩, ⟨shared₅, sharedOpen₅⟩,
    ⟨shared₆, sharedOpen₆⟩, terminalEq⟩ := shared
  obtain ⟨roots, rest⟩ := List.append_subset.mp (List.Subset.refl (views r))
  obtain ⟨rootOpen, rest⟩ := List.append_subset.mp rest
  obtain ⟨groups₁, rest⟩ := List.append_subset.mp rest
  obtain ⟨open₁, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₂Views, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₃Views, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₄Views, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₅Views, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₆Views, last⟩ := List.append_subset.mp rest
  rcases brk with b | b | b | b | b | b | b | b
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Batch.solved sharedRoot sharedOpen
      (fun g v member => roots (List.mem_flatMap.mpr ⟨g, List.mem_finRange g, member⟩)) rootOpen z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Batch.solved shared₁ sharedOpen₁
      (fun g v member => groups₁ (List.mem_flatMap.mpr ⟨g, List.mem_finRange g, member⟩)) open₁ z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := false) shared₂ sharedOpen₂
      (List.append_subset.mp level₂Views).1 (List.append_subset.mp level₂Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := false) shared₃ sharedOpen₃
      (List.append_subset.mp level₃Views).1 (List.append_subset.mp level₃Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := false) shared₄ sharedOpen₄
      (List.append_subset.mp level₄Views).1 (List.append_subset.mp level₄Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := true) shared₅ sharedOpen₅
      (List.append_subset.mp level₅Views).1 (List.append_subset.mp level₅Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := true) shared₆ sharedOpen₆
      (List.append_subset.mp level₆Views).1 (List.append_subset.mp level₆Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    subst terminalEq
    exact Setup.Solved.of_kernel (euclidean := true) (last List.mem_cons_self)
      terminal.innerSetupCode.injective z

/-- `reduced_key` for the views of one setup vector `S`: a solution of the reduction
`tableCollision` for the matrices of the key group at a bound `innerBound ≤ η_A` gives a solution
of Module-SIS for the view of an instance of `views r`. `shared`: the matrices of the key group
are the views of `S`. -/
theorem reduced_key_shared {S : Fin N → Zq} (shared : (key 0).Shared S) {innerBound : ℕ}
    (inner : innerBound ≤ ((root r).fold 0).innerBound ((root r).shape 0))
    (z : ShortKernel (key 0).inner innerBound ⊕ OuterBreak (key 0)) :
    Setup.Solved (views r) S := by
  have within : ((root r).shape 0).instances false
      (2 * (2 * mass 0) * ((root r).fold 0).sourceBound ((root r).shape 0)) ⊆ views r :=
    fun v member => (List.append_subset.mp (List.Subset.refl (views r))).1
      (List.mem_flatMap.mpr ⟨0, List.mem_finRange 0, member⟩)
  rcases z with z | z
  · exact Key.solved_inner (euclidean := false) shared within
      (ShortKernel.carry id (Function.Embedding.refl _) (fun _ _ => rfl)
        (inner.trans_eq rfl) z)
  · exact Key.solved_outer shared within z

/-- Each view of each row of `key30.aks` reads at most `2^23` cells of the flat setup vector, the
cells of the setup prefix (`setupVars = 23`), and `A` of the key group reads `2^23` cells. -/
theorem views_fit : ∀ r ∈ rows,
    (∀ I ∈ views r, I.size ≤ 2 ^ setupVars) ∧ ∃ I ∈ views r, I.size = 2 ^ setupVars := by
  decide

theorem views_fits (member : r ∈ rows) {N : ℕ} (large : 2 ^ setupVars ≤ N)
    (i : Fin (views r).length) : ((views r).get i).size ≤ N :=
  ((views_fit r member).1 _ (List.get_mem _ _)).trans large

open scoped BigOperators in
/-- Akita Lemma 10.20 (i), (168), for the 54 views of the row `r` of `key30.aks` and a flat setup
vector of at least `2^23` cells: the probability that `finder` returns an index of `views r` and a
solution of Module-SIS for the view of its instance (`Setup.Solved`, `Setup.solved_iff`) is the
sum over the 54 views of the probability of `Setup.embedded` for a uniform matrix of the view.
With `views_covered` each view is below an instance of `instances`. -/
theorem shared_setup (member : r ∈ rows) {N : ℕ} {Aux : Type} [Fintype Aux]
    (large : 2 ^ setupVars ≤ N) (finder : Setup.Finder (views r).get N Aux) :
    𝔼 S : Fin N → Zq, 𝔼 aux : Aux, (if (finder S aux).isSome then (1 : ℝ) else 0) =
      ∑ i, 𝔼 M : Matrix (Fin ((views r).get i).rows) (Fin ((views r).get i).cols)
          (Rq ((views r).get i).degree),
        𝔼 coins : (Fin N → Zq) × Aux,
          if (Setup.embedded (views_fits member large) finder i M coins).isSome then (1 : ℝ)
          else 0 :=
  Setup.shared_setup_eq (views_fits member large) finder

/-- The join with `Tower.Shared`: when the matrices of the root are the views of the setup prefix
`flat` (`Key.Shared`, `Batch.OpenKey.Shared`), the data `o` satisfies `Tower.Shared`. So
`Shared` at the setup prefix and `Tower.Shared` hold together. -/
theorem Tower.shared_of_views {member : r ∈ rows} {E : Type} [Field E] [Fintype E]
    [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} {total : ℕ}
    (o : Tower r member E Wire Coord total)
    (roots : (∀ g, (o.key g).Shared fun p : Fin (2 ^ setupVars) => o.flat p) ∧
      o.openKey.Shared fun p : Fin (2 ^ setupVars) => o.flat p) : o.Shared :=
  ⟨fun g => ⟨(roots.1 g).1, (roots.1 g).2.1⟩, roots.2.1⟩

end NightstreamFPrime.Spec.Compression.Akita.Key30
