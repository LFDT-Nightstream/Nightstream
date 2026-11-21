import NightstreamFPrime.Spec.Compression.Akita.SetupInstances
import NightstreamFPrime.Spec.Compression.Akita.Key20Challenge

/-! The instances of Module-SIS of the opening of the schedule file
`crates/ns-compress/artifacts/key20.aks` (Akita, ePrint 2026/1983, revision of 2026-10-15,
Definition 3.2, Table 3, Section 10.3 and Appendix G).

- `views` has the instance of each matrix of the tower of `Extract/Tower.lean` for the numbers of
  `Key20.lean`, at the bound of the witness of `R_bind` of its level: 15 of the root, 7 of each
  of the levels 1 to 3, 3 of each of the levels 4 and 5, and 1 of the terminal.
- `instances` has the 13 instances that are not below another instance. A coefficient instance
  has the bound of its view. A Euclidean instance has the power of two that the file has in
  `collision_l2_sq`, which is the bound that the Akita code prices.
  `tests/AkitaSchedules.lean` compares the list with the file.
- `views_covered` states that each view is below an instance of the list, so each bound of a
  witness of `R_bind` is at most the bound that the file prices. `instances_exact` gives the
  numbers of the list, and that the list has no instance without a view and no two instances in
  the order `Instance.Below`. `maps_tower` is the condition of ring descent for the first
  compression map.
- `reduced` and `reduced_key` state that the list is complete: each solution of a witness of
  `R_bind` of the tower, and each solution of the reduction `tableCollision` for the key
  commitment at a bound of at most `η_A`, gives a solution of an instance of the list
  (`Reduces`).
- `Shared S` states that each matrix of the opening is the view of the flat setup vector `S` with
  the degree, the rank and the width of its instance in `views`, and `shared_ofSetup` gives such
  keys. `reduced_shared` and `reduced_key_shared` are `reduced` and `reduced_key` for these keys:
  each solution gives a solution of Module-SIS for the view of `S` of an instance of `views`
  (`Setup.Solved`). So the list `views` is the list of the matrices that the verifier uses.
- `views_fit` states that each view reads at most `2^18` cells of `S`, and that `A` of the levels
  2 and 4 read `2^18` cells. `shared_setup` is Akita Lemma 10.20 (i) for the 43 views.

Lean has no statement on the hardness of an instance. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open Setup (Instance)

theorem Covered.left {list first second : List Instance} (covered : Covered list (first ++ second)) :
    Covered list first :=
  fun v member => covered v (List.mem_append_left _ member)

theorem Covered.right {list first second : List Instance}
    (covered : Covered list (first ++ second)) : Covered list second :=
  fun v member => covered v (List.mem_append_right _ member)

namespace Key20

open Extract

/-- The instances of the matrices of a level after the root, with the norm and the bound of
`A`. -/
def levelViews (L : Level) (euclidean : Bool) (bound : ℕ) : List Instance :=
  L.shape.instances euclidean bound ++ L.opening.instances

/-- The instances of the matrices of the opening of `key20.aks`, each at the bound of the witness
of `R_bind` of its level (`Extract.towerRoot`): `η_A = 2·κ̄_1·Δ^cert_f` of Akita (176) on the
coefficient route, `64·Γ²·S_max` of Akita (182) on the Euclidean route, `b⋆ − 1` for `B` and `D`,
and 1 for a compression map. -/
def views : List Instance :=
  ((List.finRange 3).flatMap fun g => (root.shape g).instances false
      (2 * (2 * rootMass g) * (root.fold g).sourceBound (root.shape g))) ++
    (root.openInstances ++
      (levelViews level₁ false (2 * (2 * 51) * level₁.fold.sourceBound level₁.shape) ++
        (levelViews level₂ false (2 * (2 * 31) * level₂.fold.sourceBound level₂.shape) ++
          (levelViews level₃ true (64 * norm₃.operator ^ 2 * norm₃.limit) ++
            (levelViews level₄ true (64 * norm₄.operator ^ 2 * norm₄.limit) ++
              (levelViews level₅ true (64 * norm₅.operator ^ 2 * norm₅.limit) ++
                [terminal.innerInstance true
                  (64 * normTerminal.operator ^ 2 * normTerminal.limit)]))))))

/-- The instances of Module-SIS of the opening of `key20.aks`, at the bounds that the file has:
the second and the first compression map at their largest width, `D` and `B` of the levels 4 and
5, `A` of the level 5 and of the terminal, `A` of the level 4, `D` of the root, `B` of the level
2, `A` of the level 2, `A` of the level 3, and `A` of `W`, of the key polynomial and of `V`. -/
def instances : List Instance :=
  [⟨(maps 1024).secondDegree, (maps 1024).secondRank, (maps 1024).secondWidth, false, 1⟩,
    ⟨(maps 1024).firstDegree, (maps 1024).firstRank, (maps 1024).firstWidth, false, 1⟩,
    level₅.opening.openInstance,
    level₅.shape.outerInstance,
    level₅.shape.innerInstance true (2 ^ 46),
    level₄.shape.innerInstance true (2 ^ 45),
    root.openInstance,
    level₂.shape.outerInstance,
    level₂.shape.innerInstance false (level₂.fold.innerBound level₂.shape),
    level₃.shape.innerInstance true (2 ^ 44),
    (root.shape 1).innerInstance false ((root.fold 1).innerBound (root.shape 1)),
    (root.shape 0).innerInstance false ((root.fold 0).innerBound (root.shape 0)),
    (root.shape 2).innerInstance false ((root.fold 2).innerBound (root.shape 2))]

/-- Each view is below an instance of the list: an instance of the list has its degree, its rank
and its norm, a width that is not smaller and a bound that is not smaller. So the bound of each
witness of `R_bind` is at most the bound that the file prices. -/
theorem views_covered : Covered instances views := by
  unfold Covered
  decide

/-- The list `instances` and the views. Each view is below an instance of the list. The list has
13 instances `(d, n, m, p, η)` for the 43 views. Each instance has the width of a view below it,
with the bound of this view on the coefficient route and with a bound below two times the bound
of this view on the Euclidean route. No instance is below another instance. -/
theorem instances_exact :
    Covered instances views ∧
      instances.map Instance.key =
        [(32, 1, 128, false, 1), (64, 1, 1024, false, 1), (64, 2, 77, false, 63),
          (64, 3, 616, false, 63), (64, 8, 256, true, 2 ^ 46), (64, 8, 512, true, 2 ^ 45),
          (128, 1, 1936, false, 7), (128, 1, 1092, false, 31), (128, 4, 512, false, 126852),
          (128, 4, 256, true, 2 ^ 44), (512, 2, 56, false, 53477172),
          (1024, 1, 224, false, 32505732), (1024, 1, 56, false, 53477172)] ∧
      views.length = 43 ∧
      (∀ I ∈ instances, ∃ v ∈ views, v.Below I ∧ v.cols = I.cols ∧
        cond I.euclidean (I.bound < 2 * v.bound) (v.bound = I.bound)) ∧
      instances.Pairwise fun I J => ¬ I.Below J ∧ ¬ J.Below I :=
  ⟨views_covered, by decide⟩

/-- The degree of the first compression map is two times the degree of the second: the condition
`stride · d_car = d` of ring descent (`ShortKernel.lower_chance`) from the degree 64 to the degree
32. -/
theorem maps_tower (width : ℕ) : 2 * (maps width).secondDegree = (maps width).firstDegree := rfl

variable {key : (g : Fin 3) → Key (root.shape g)} {openKey : root.OpenKey}
  {key₁ : Key level₁.shape} {openKey₁ : OpeningKey level₁.opening}
  {key₂ : Key level₂.shape} {openKey₂ : OpeningKey level₂.opening}
  {key₃ : Key level₃.shape} {openKey₃ : OpeningKey level₃.opening}
  {key₄ : Key level₄.shape} {openKey₄ : OpeningKey level₄.opening}
  {key₅ : Key level₅.shape} {openKey₅ : OpeningKey level₅.opening}
  {matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
    (Rq terminal.innerDegree)}

/-- A matrix of the opening of `key20.aks` has a solution of Module-SIS that gives a solution of
an instance of `list` (`Reduces`): a matrix of a group of the root, of the shared opening path of
the root, of one of the levels 1 to 5, or `A_term`. The levels 1 and 2 have the norm `‖·‖_∞` for
`A`, and the levels 3 to 5 and the terminal have the norm `‖·‖_2`. -/
def Reduced (list : List Instance) (key : (g : Fin 3) → Key (root.shape g))
    (openKey : root.OpenKey) (key₁ : Key level₁.shape) (openKey₁ : OpeningKey level₁.opening)
    (key₂ : Key level₂.shape) (openKey₂ : OpeningKey level₂.opening) (key₃ : Key level₃.shape)
    (openKey₃ : OpeningKey level₃.opening) (key₄ : Key level₄.shape)
    (openKey₄ : OpeningKey level₄.opening) (key₅ : Key level₅.shape)
    (openKey₅ : OpeningKey level₅.opening)
    (matrix : Matrix (Fin terminal.innerRank)
      (Fin terminal.blockLength × Fin terminal.sourceDigits) (Rq terminal.innerDegree)) : Prop :=
  ((∃ g, (key g).Reduces list false) ∨ openKey.Reduces list) ∨
    (key₁.Reduces list false ∨ openKey₁.Reduces list) ∨
    (key₂.Reduces list false ∨ openKey₂.Reduces list) ∨
    (key₃.Reduces list true ∨ openKey₃.Reduces list) ∨
    (key₄.Reduces list true ∨ openKey₄.Reduces list) ∨
    (key₅.Reduces list true ∨ openKey₅.Reduces list) ∨ Akita.Reduces list matrix true

/-- The list `instances` is complete for the witnesses of `R_bind` of the opening of `key20.aks`
(Akita Table 3 and Lemma 10.20). `brk` is a witness of `R_bind` of the tower: for the root, for
one of the levels 1 to 5 or for the terminal. `held`: it holds a solution of Module-SIS. Then a
matrix of the opening has a solution that gives a solution of an instance of the list, for each
matrix of this instance with the columns of that matrix, at the bound that the file prices.
`mass`, `first` and `second`: the norms `‖c‖_1` of the fold challenges of the root and of the
levels 1 and 2 are those of the families of `Key20Challenge.lean`. -/
theorem reduced {mass : Fin 3 → ℕ}
    {weight₁ weight₂ weight₃ weight₄ weight₅ weightL boundL : ℕ} (massEq : mass = rootMass)
    (first : weight₁ = 51) (second : weight₂ = 31)
    (brk : root.LevelBreak key openKey
        (fun g => 2 * (2 * mass g) * (root.fold g).sourceBound (root.shape g)) ⊕
      LevelBreak key₁ openKey₁ (2 * (2 * weight₁) * level₁.fold.sourceBound level₁.shape) ⊕
      Route.Break key₂ openKey₂ weight₂ (level₂.fold.sourceBound level₂.shape) .coefficient ⊕
      Route.Break key₃ openKey₃ weight₃ (level₃.fold.sourceBound level₃.shape)
        (.euclidean norm₃.operator norm₃.limit) ⊕
      Route.Break key₄ openKey₄ weight₄ (level₄.fold.sourceBound level₄.shape)
        (.euclidean norm₄.operator norm₄.limit) ⊕
      Route.Break key₅ openKey₅ weight₅ (level₅.fold.sourceBound level₅.shape)
        (.euclidean norm₅.operator norm₅.limit) ⊕
      Option ((Route.euclidean normTerminal.operator normTerminal.limit).Kernel matrix weightL
        boundL))
    (held : Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
      (Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
        (Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
          (fun b => Option.isSome b = true)))))) brk) :
    Reduced instances key openKey key₁ openKey₁ key₂ openKey₂ key₃ openKey₃ key₄ openKey₄ key₅
      openKey₅ matrix := by
  subst massEq first second
  have rest := views_covered.right.right
  rcases brk with b | b | b | b | b | b | b
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inl (Batch.reduces
      (fun g v member => views_covered.left v
        (List.mem_flatMap.mpr ⟨g, List.mem_finRange g, member⟩))
      views_covered.right.left z)
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inl (level_reduces (euclidean := false) rest.left.left rest.left.right z))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inl (level_reduces (euclidean := false) rest.right.left.left
      rest.right.left.right z)))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inr (Or.inl (level_reduces (euclidean := true)
      rest.right.right.left.left rest.right.right.left.right z))))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (level_reduces (euclidean := true)
      rest.right.right.right.left.left rest.right.right.right.left.right z)))))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (level_reduces (euclidean := true)
      rest.right.right.right.right.left.left rest.right.right.right.right.left.right z))))))
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Reduces.of_kernel (euclidean := true)
      (by simp only [Fintype.card_prod, Fintype.card_fin])
      (rest.right.right.right.right.right _ List.mem_cons_self) z))))))

/-- The list `instances` is complete for the key commitment. A solution of the reduction
`tableCollision` for the matrices of the key group at a bound `innerBound ≤ η_A` gives a solution
of an instance of the list: `η_A` is the bound of `A` of the key group in the file. -/
theorem reduced_key {innerBound : ℕ}
    (inner : innerBound ≤ (root.fold 0).innerBound (root.shape 0))
    (z : ShortKernel (key 0).inner innerBound ⊕ OuterBreak (key 0)) :
    (key 0).Reduces instances false := by
  have covered : Covered instances ((root.shape 0).instances false
      (2 * (2 * rootMass 0) * (root.fold 0).sourceBound (root.shape 0))) :=
    fun v member => views_covered.left v (List.mem_flatMap.mpr ⟨0, List.mem_finRange 0, member⟩)
  rcases z with z | z
  · exact Key.reduces_inner (euclidean := false) covered
      (ShortKernel.carry id (Function.Embedding.refl _) (fun _ _ => rfl)
        (inner.trans_eq (by decide)) z)
  · exact Key.reduces_outer covered z

variable {N : ℕ}

/-- The matrices of the opening of `key20.aks` are the views of one flat setup vector `S` (Akita
Section 4.3 and (35)): the keys of the root, of the shared opening path of the root, of the levels
1 to 5 and `A_term`. Each matrix is the view with the degree, the rank and the width of its
instance in `views`. -/
def Shared (S : Fin N → Zq) (key : (g : Fin 3) → Key (root.shape g)) (openKey : root.OpenKey)
    (key₁ : Key level₁.shape) (openKey₁ : OpeningKey level₁.opening) (key₂ : Key level₂.shape)
    (openKey₂ : OpeningKey level₂.opening) (key₃ : Key level₃.shape)
    (openKey₃ : OpeningKey level₃.opening) (key₄ : Key level₄.shape)
    (openKey₄ : OpeningKey level₄.opening) (key₅ : Key level₅.shape)
    (openKey₅ : OpeningKey level₅.opening)
    (matrix : Matrix (Fin terminal.innerRank)
      (Fin terminal.blockLength × Fin terminal.sourceDigits) (Rq terminal.innerDegree)) : Prop :=
  ((∀ g, (key g).Shared S) ∧ openKey.Shared S) ∧ (key₁.Shared S ∧ openKey₁.Shared S) ∧
    (key₂.Shared S ∧ openKey₂.Shared S) ∧ (key₃.Shared S ∧ openKey₃.Shared S) ∧
    (key₄.Shared S ∧ openKey₄.Shared S) ∧ (key₅.Shared S ∧ openKey₅.Shared S) ∧
    matrix = Setup.shared S terminal.innerDegree terminal.innerRank _ terminal.innerSetupCode

/-- The keys whose matrices are the views of `S` satisfy `Shared`. -/
theorem shared_ofSetup (S : Fin N → Zq) :
    Shared S (fun g => Key.ofSetup S (root.shape g)) (Batch.OpenKey.ofSetup S root)
      (Key.ofSetup S _) (OpeningKey.ofSetup S _) (Key.ofSetup S _) (OpeningKey.ofSetup S _)
      (Key.ofSetup S _) (OpeningKey.ofSetup S _) (Key.ofSetup S _) (OpeningKey.ofSetup S _)
      (Key.ofSetup S _) (OpeningKey.ofSetup S _)
      (Setup.shared S terminal.innerDegree terminal.innerRank _ terminal.innerSetupCode) :=
  ⟨⟨fun _ => Key.ofSetup_shared S _, Batch.OpenKey.ofSetup_shared S root⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩,
    ⟨Key.ofSetup_shared S _, OpeningKey.ofSetup_shared S _⟩, rfl⟩

/-- `reduced` for the views of one setup vector `S`: each solution of a witness of `R_bind` of
the tower of `key20.aks` gives a solution of Module-SIS for the view `M_I(S)` of an instance `I`
of `views` itself, with the width of its matrix and the bound of the witness of its level
(`Setup.Solved`). `shared`: each matrix of the opening is the view of `S` with the numbers of its
instance. -/
theorem reduced_shared {mass : Fin 3 → ℕ}
    {weight₁ weight₂ weight₃ weight₄ weight₅ weightL boundL : ℕ} (massEq : mass = rootMass)
    (first : weight₁ = 51) (second : weight₂ = 31) {S : Fin N → Zq}
    (shared : Shared S key openKey key₁ openKey₁ key₂ openKey₂ key₃ openKey₃ key₄ openKey₄ key₅
      openKey₅ matrix)
    (brk : root.LevelBreak key openKey
        (fun g => 2 * (2 * mass g) * (root.fold g).sourceBound (root.shape g)) ⊕
      LevelBreak key₁ openKey₁ (2 * (2 * weight₁) * level₁.fold.sourceBound level₁.shape) ⊕
      Route.Break key₂ openKey₂ weight₂ (level₂.fold.sourceBound level₂.shape) .coefficient ⊕
      Route.Break key₃ openKey₃ weight₃ (level₃.fold.sourceBound level₃.shape)
        (.euclidean norm₃.operator norm₃.limit) ⊕
      Route.Break key₄ openKey₄ weight₄ (level₄.fold.sourceBound level₄.shape)
        (.euclidean norm₄.operator norm₄.limit) ⊕
      Route.Break key₅ openKey₅ weight₅ (level₅.fold.sourceBound level₅.shape)
        (.euclidean norm₅.operator norm₅.limit) ⊕
      Option ((Route.euclidean normTerminal.operator normTerminal.limit).Kernel matrix weightL
        boundL))
    (held : Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
      (Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
        (Sum.elim (fun b => Option.isSome b = true) (Sum.elim (fun b => Option.isSome b = true)
          (fun b => Option.isSome b = true)))))) brk) :
    Setup.Solved views S := by
  subst massEq first second
  obtain ⟨⟨sharedRoot, sharedOpen⟩, ⟨shared₁, sharedOpen₁⟩, ⟨shared₂, sharedOpen₂⟩,
    ⟨shared₃, sharedOpen₃⟩, ⟨shared₄, sharedOpen₄⟩, ⟨shared₅, sharedOpen₅⟩, terminalEq⟩ := shared
  obtain ⟨roots, rest⟩ := List.append_subset.mp (List.Subset.refl views)
  obtain ⟨rootOpen, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₁Views, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₂Views, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₃Views, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₄Views, rest⟩ := List.append_subset.mp rest
  obtain ⟨level₅Views, last⟩ := List.append_subset.mp rest
  rcases brk with b | b | b | b | b | b | b
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact Batch.solved sharedRoot sharedOpen
      (fun g v member => roots (List.mem_flatMap.mpr ⟨g, List.mem_finRange g, member⟩)) rootOpen z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := false) shared₁ sharedOpen₁
      (List.append_subset.mp level₁Views).1 (List.append_subset.mp level₁Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := false) shared₂ sharedOpen₂
      (List.append_subset.mp level₂Views).1 (List.append_subset.mp level₂Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := true) shared₃ sharedOpen₃
      (List.append_subset.mp level₃Views).1 (List.append_subset.mp level₃Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := true) shared₄ sharedOpen₄
      (List.append_subset.mp level₄Views).1 (List.append_subset.mp level₄Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    exact level_solved (euclidean := true) shared₅ sharedOpen₅
      (List.append_subset.mp level₅Views).1 (List.append_subset.mp level₅Views).2 z
  · obtain ⟨z, rfl⟩ := Option.isSome_iff_exists.mp held
    subst terminalEq
    exact Setup.Solved.of_kernel (euclidean := true) (last List.mem_cons_self)
      terminal.innerSetupCode.injective z

/-- `reduced_key` for the views of one setup vector `S`: a solution of the reduction
`tableCollision` for the matrices of the key group at a bound `innerBound ≤ η_A` gives a solution
of Module-SIS for the view of an instance of `views`. `shared`: the matrices of the key group are
the views of `S`. -/
theorem reduced_key_shared {S : Fin N → Zq} (shared : (key 0).Shared S) {innerBound : ℕ}
    (inner : innerBound ≤ (root.fold 0).innerBound (root.shape 0))
    (z : ShortKernel (key 0).inner innerBound ⊕ OuterBreak (key 0)) : Setup.Solved views S := by
  have within : (root.shape 0).instances false
      (2 * (2 * rootMass 0) * (root.fold 0).sourceBound (root.shape 0)) ⊆ views :=
    fun v member => (List.append_subset.mp (List.Subset.refl views)).1
      (List.mem_flatMap.mpr ⟨0, List.mem_finRange 0, member⟩)
  rcases z with z | z
  · exact Key.solved_inner (euclidean := false) shared within
      (ShortKernel.carry id (Function.Embedding.refl _) (fun _ _ => rfl)
        (inner.trans_eq (by decide)) z)
  · exact Key.solved_outer shared within z

/-- Each view of `key20.aks` reads at most `2^18` cells of the flat setup vector, and `A` of the
levels 2 and 4 read `2^18` cells. -/
theorem views_fit : (∀ I ∈ views, I.size ≤ 2 ^ 18) ∧ ∃ I ∈ views, I.size = 2 ^ 18 := by
  decide

theorem views_fits {N : ℕ} (large : 2 ^ 18 ≤ N) (i : Fin views.length) :
    (views.get i).size ≤ N :=
  (views_fit.1 _ (List.get_mem _ _)).trans large

open scoped BigOperators in
/-- Akita Lemma 10.20 (i), (168), for the 43 views of `key20.aks` and a flat setup vector of at
least `2^18` cells: the probability that `finder` returns an index of `views` and a solution of
Module-SIS for the view of its instance (`Setup.Solved`, `Setup.solved_iff`) is the sum over the
43 views of the probability of `Setup.embedded` for a uniform matrix of the view. With
`views_covered` each view is below an instance of `instances`. -/
theorem shared_setup {N : ℕ} {Aux : Type} [Fintype Aux] (large : 2 ^ 18 ≤ N)
    (finder : Setup.Finder views.get N Aux) :
    𝔼 S : Fin N → Zq, 𝔼 aux : Aux, (if (finder S aux).isSome then (1 : ℝ) else 0) =
      ∑ i, 𝔼 M : Matrix (Fin (views.get i).rows) (Fin (views.get i).cols)
          (Rq (views.get i).degree),
        𝔼 coins : (Fin N → Zq) × Aux,
          if (Setup.embedded (views_fits large) finder i M coins).isSome then (1 : ℝ) else 0 :=
  Setup.shared_setup_eq (views_fits large) finder

end Key20

end NightstreamFPrime.Spec.Compression.Akita
