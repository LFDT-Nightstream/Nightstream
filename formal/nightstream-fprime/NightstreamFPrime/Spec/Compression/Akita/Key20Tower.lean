import NightstreamFPrime.Spec.Compression.Akita.Key20
import NightstreamFPrime.Spec.Compression.Akita.Extract.Credit

/-! The opening suffix of Akita for the schedule file `crates/ns-compress/artifacts/key20.aks`
(Akita, ePrint 2026/1983, revision of 2026-11-23, Proposition 10.44, Theorem 10.46, Corollary
10.35 (ii) and Corollary 10.49 with (206)).

- `tower_suffix` is `mixedTower_suffix` for the numbers of `Key20.lean`, under the verifier
  `towerLinks`. The retry loops are 142 and the tree size is 21449539584. Its hypotheses are: an
  answer of the oracle has 22 slots, the fold coordinates of each level have different slots, each
  challenge family has units as differences and its norm bound, and the stage of the terminal.
- `towerError_le` bounds the sum of the stage errors by `317479/(1024·|E|) + 135·p` when each
  challenge family has the slot probability at most `p`, with the work bits of the stage checks of
  the Akita code (Remark 10.18), no work credit on the fold term, and `|E| = q²` (`card_field`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key20

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The conditions `LevelSized` of a level from the conditions on its coordinates and on its
challenge family, with the numbers `base` and `small` of the schedule. -/
theorem levelSized {E Coord : Type} [Fintype Coord] {d blocks : ℕ} [NeZero d]
    {block : Fin blocks → Coord} {decode : E → Option (Rq d)}
    {weight responseBase certifiedBase μ batch : ℕ}
    (slots : 22 ≤ Fintype.card Coord) (vars : μ ≤ 22) (rows : batch ≤ 22)
    (base : 2 ≤ responseBase) (small : certifiedBase < fieldModulus)
    (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∃ peak, ∀ challenge, (∃ slot, decode slot = some challenge) →
      Sparse weight peak challenge) :
    LevelSized block decode weight responseBase certifiedBase μ batch :=
  ⟨injective, units, sparse, base, small, vars.trans slots, rows.trans slots⟩

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {total : ℕ}
  {read : Reader Wire} {start : ℕ}
  {β : ∀ g, Module.Basis (Fin (root.planes g)) Zq E} {key : (g : Fin 3) → Key (root.shape g)}
  {openKey : root.OpenKey} {claim : Wire → root.Claim E} {block : root.Block → Coord}
  {decode : (g : Fin 3) → E → Option (Rq (root.carrierDegree g))} {mass : Fin 3 → ℕ}
  {β₁ : Module.Basis (Fin level₁.opening.planes) Zq E} {key₁ : Key level₁.shape}
  {openKey₁ : OpeningKey level₁.opening} {claim₁ : Wire → Claim level₁.opening E}
  {block₁ : Fin level₁.shape.blocks → Coord}
  {decode₁ : E → Option (Rq level₁.opening.carrierDegree)} {weight₁ : ℕ}
  {key₂ : Key level₂.shape} {openKey₂ : OpeningKey level₂.opening}
  {claim₂ : Wire → TraceClaim level₂.opening E} {block₂ : Fin level₂.shape.blocks → Coord}
  {decode₂ : E → Option (Rq level₂.shape.innerDegree)} {weight₂ : ℕ}
  {key₃ : Key level₃.shape} {openKey₃ : OpeningKey level₃.opening}
  {claim₃ : Wire → TraceClaim level₃.opening E} {block₃ : Fin level₃.shape.blocks → Coord}
  {decode₃ : E → Option (Rq level₃.shape.innerDegree)} {weight₃ : ℕ}
  {key₄ : Key level₄.shape} {openKey₄ : OpeningKey level₄.opening}
  {claim₄ : Wire → TraceClaim level₄.opening E} {block₄ : Fin level₄.shape.blocks → Coord}
  {decode₄ : E → Option (Rq level₄.shape.innerDegree)} {weight₄ : ℕ}
  {key₅ : Key level₅.shape} {openKey₅ : OpeningKey level₅.opening}
  {claim₅ : Wire → TraceClaim level₅.opening E} {block₅ : Fin level₅.shape.blocks → Coord}
  {decode₅ : E → Option (Rq level₅.shape.innerDegree)} {weight₅ : ℕ}
  {L : Terminal terminal E Wire Coord E total
    (.euclidean normTerminal.operator normTerminal.limit)}
  {commit : Wire → (g : Fin 3) → (root.shape g).Payload} {value : Wire → Fin 3 → E}
  {place : Wire → Fin 3 → List E} {commit₁ : Wire → level₁.shape.Payload}
  {commit₂ : Wire → level₂.shape.Payload} {commit₃ : Wire → level₃.shape.Payload}
  {commit₄ : Wire → level₄.shape.Payload} {commit₅ : Wire → level₅.shape.Payload}
  {stateOf : Wire → Fin terminal.blocks → Fin terminal.innerRank → Rq terminal.innerDegree}

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] [DecidableEq E] in
/-- The extension field of degree 2 has `q²` elements. -/
theorem card_field : Fintype.card E = fieldModulus ^ 2 := by
  rw [← Nat.card_eq_fintype_card,
    Nat.card_congr ((Degree2.coords (E := E)).toEquiv.trans (QuadraticAlgebra.equivProd 2 0)),
    Nat.card_prod, sq]
  exact congrArg₂ _ (Nat.card_zmod _) (Nat.card_zmod _)

omit [Field E] [DecidableEq E] [Algebra Zq E] [Degree2 E] [Fintype Coord] [DecidableEq Coord]
  [Nonempty Coord] in
/-- The sum of the bounds `foldCosts` of the draws of the checks of one fold. -/
theorem foldCosts_total (badBound vars batch base : ℕ) (bin : Bool) :
    (foldCosts E badBound vars batch base bin).sum =
      ((badBound + vars + batch + rangeNumerator vars base + binDraws bin + 1 +
        vars * 3 : ℕ) : ℝ) / Fintype.card E := by
  simp only [foldCosts, List.sum_cons, List.sum_append, List.sum_replicate, nsmul_eq_mul,
    rangeCosts_sum, binCosts_sum]
  push_cast
  ring

omit [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E] [Fintype Coord]
  [DecidableEq Coord] [Nonempty Coord] in
/-- The claims do not count the work credit `2^{−g}` on the fold term
(`decisions/akita-compression.md` (d)). -/
theorem uncredited {x y : ℝ} {g : ℕ} (nonnegative : 0 ≤ x) (bound : x ≤ y) :
    1 / 2 ^ g * x ≤ y :=
  (mul_le_of_le_one_left nonnegative
    (div_le_one_of_le₀ (one_le_pow₀ one_le_two) (by positivity))).trans bound

omit [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E] [Fintype Coord]
  [DecidableEq Coord] [Nonempty Coord] in
/-- A sum of `count` numbers in `[0, p]` without the work credit is at most `count·p`. -/
theorem sum_le_mul {ι : Type} [Fintype ι] {f : ι → ℝ} {p : ℝ} {count g : ℕ}
    (card : Fintype.card ι = count) (nonnegative : ∀ b, 0 ≤ f b) (each : ∀ b, f b ≤ p) :
    1 / 2 ^ g * ∑ b, f b ≤ count * p := by
  have total := Finset.sum_le_card_nsmul Finset.univ f p fun b _ => each b
  rw [Finset.card_univ, card, nsmul_eq_mul] at total
  exact uncredited (Finset.sum_nonneg fun b _ => nonnegative b) total

theorem sized₂ : level₂.opening.cells level₂.fold ≤ 2 ^ level₂.vars ∧
    level₂.opening.traceRowCount + 1 ≤ 2 ^ level₂.batch ∧ 2 ≤ level₂.fold.responseBase ∧
    level₂.shape.certifiedBase < fieldModulus := traces_sized level₂ (by simp [traces])

theorem sized₃ : level₃.opening.cells level₃.fold ≤ 2 ^ level₃.vars ∧
    level₃.opening.traceRowCount + 1 ≤ 2 ^ level₃.batch ∧ 2 ≤ level₃.fold.responseBase ∧
    level₃.shape.certifiedBase < fieldModulus := traces_sized level₃ (by simp [traces])

theorem sized₄ : level₄.opening.cells level₄.fold ≤ 2 ^ level₄.vars ∧
    level₄.opening.traceRowCount + 1 ≤ 2 ^ level₄.batch ∧ 2 ≤ level₄.fold.responseBase ∧
    level₄.shape.certifiedBase < fieldModulus := traces_sized level₄ (by simp [traces])

theorem sized₅ : level₅.opening.cells level₅.fold ≤ 2 ^ level₅.vars ∧
    level₅.opening.traceRowCount + 1 ≤ 2 ^ level₅.batch ∧ 2 ≤ level₅.fold.responseBase ∧
    level₅.shape.certifiedBase < fieldModulus := traces_sized level₅ (by simp [traces])

local notation "zeroth" =>
  liftBatch β key openKey claim rootVars (And.left root_sized) 8 quotientDigits root_room
    rootBatch (And.left (And.right root_sized)) block decode read mass (start + 1)

local notation "first" =>
  Tier.tensored (liftFold β₁ key₁ openKey₁ claim₁ level₁.vars (And.left level₁_sized)
    quotientDigits (And.left (And.right level₁_sized)) level₁.batch
    (And.left (And.right (And.right level₁_sized))) block₁ decode₁ read weight₁
    (Tier.finish zeroth)) level₁.tail

local notation "second" =>
  Tier.tensored (committedTrace (fold := level₂.fold) key₂ openKey₂ claim₂ level₂.vars
    (And.left sized₂) level₂.batch (And.left (And.right sized₂)) block₂ decode₂ read weight₂
    (Tier.finish first)) level₂.tail

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := level₃.fold) key₃ openKey₃ claim₃ level₃.vars
    (And.left sized₃) level₃.batch (And.left (And.right sized₃)) block₃ decode₃ read
    weight₃ norm₃.operator norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) key₄ openKey₄ claim₄ level₄.vars
    (And.left sized₄) level₄.batch (And.left (And.right sized₄)) block₄ decode₄ read
    weight₄ norm₄.operator norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅ claim₅ level₅.vars
    (And.left sized₅) level₅.batch (And.left (And.right sized₅)) block₅ decode₅ read
    weight₅ norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail

local notation "accepts" =>
  towerLinks (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅)
    Route.coefficient (Route.euclidean norm₃.operator norm₃.limit)
    (Route.euclidean norm₄.operator norm₄.limit) (Route.euclidean norm₅.operator norm₅.limit)
    zeroth first second third fourth fifth L start read commit value place (And.left quarters)
    (And.left (And.right quarters)) (And.left (And.right (And.right quarters)))
    (And.left (And.right (And.right (And.right quarters))))
    (And.right (And.right (And.right (And.right quarters)))) commit₁ commit₂ commit₃ commit₄
    commit₅ stateOf

local notation "towerError" =>
  (1 / (Fintype.card E : ℝ) / 2 ^ Root.rootBits 3 + (Tier.levelError zeroth +
    (Tier.levelError first +
      (Tier.levelError second +
        (Tier.levelError third +
          (Tier.levelError fourth +
            (Tier.levelError fifth + Terminal.termError L)))))))

/-- Akita Proposition 10.44 and Theorem 10.46 for the schedule `key20.aks`: the batching root
over the fold of the key polynomial, `W` and `V`, the packing level 1, the trace levels 2 to 5
and the terminal is a suffix from the stage `start` under the verifier `towerLinks`
(`mixedTower_suffix`). The levels 1 and 2 have the coefficient route, and the levels 3 to 5 and
the terminal have the Euclidean route. The suffix has 142 retry loops and the tree size
21449539584. `slots`: an answer of the oracle has 22 slots. `injective`, `units`, `sparse` and
`bounded`: the fold coordinates of each level have different slots, and the challenge family of
each level has units as differences (Akita Definition 9.2, E1), members with `‖c‖_1 = ω`, and
at the levels 3 to 5 and at the terminal `‖c‖_mul,2 ≤ Γ`. `next` and `inside`: the fold
challenge of the terminal is the stage after the stages of the level 5, and it is a stage of the
transcript. -/
theorem tower_suffix {peak : Fin 3 → ℕ} (slots : 22 ≤ Fintype.card Coord)
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (injective₁ : Function.Injective block₁)
    (units₁ : UnitDifferences {challenge | ∃ slot, decode₁ slot = some challenge})
    (sparse₁ : ∃ peak, ∀ challenge, (∃ slot, decode₁ slot = some challenge) →
      Sparse weight₁ peak challenge)
    (injective₂ : Function.Injective block₂)
    (units₂ : UnitDifferences {challenge | ∃ slot, decode₂ slot = some challenge})
    (sparse₂ : ∃ peak, ∀ challenge, (∃ slot, decode₂ slot = some challenge) →
      Sparse weight₂ peak challenge)
    (injective₃ : Function.Injective block₃)
    (units₃ : UnitDifferences {challenge | ∃ slot, decode₃ slot = some challenge})
    (sparse₃ : ∃ peak, ∀ challenge, (∃ slot, decode₃ slot = some challenge) →
      Sparse weight₃ peak challenge)
    (bounded₃ : ∀ challenge, (∃ slot, decode₃ slot = some challenge) →
      MulBound norm₃.operator challenge)
    (injective₄ : Function.Injective block₄)
    (units₄ : UnitDifferences {challenge | ∃ slot, decode₄ slot = some challenge})
    (sparse₄ : ∃ peak, ∀ challenge, (∃ slot, decode₄ slot = some challenge) →
      Sparse weight₄ peak challenge)
    (bounded₄ : ∀ challenge, (∃ slot, decode₄ slot = some challenge) →
      MulBound norm₄.operator challenge)
    (injective₅ : Function.Injective block₅)
    (units₅ : UnitDifferences {challenge | ∃ slot, decode₅ slot = some challenge})
    (sparse₅ : ∃ peak, ∀ challenge, (∃ slot, decode₅ slot = some challenge) →
      Sparse weight₅ peak challenge)
    (bounded₅ : ∀ challenge, (∃ slot, decode₅ slot = some challenge) →
      MulBound norm₅.operator challenge)
    (next : L.start = Tier.finish fifth) (inside : L.start < total)
    (injectiveL : Function.Injective L.block) (unitsL : UnitDifferences L.family)
    (boundedL : ∀ challenge ∈ L.family, MulBound normTerminal.operator challenge) :
    Tier.Suffix accepts start (OpensAt start commit value place key mass)
      (towerRoot (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) .coefficient
        (.euclidean norm₃.operator norm₃.limit) (.euclidean norm₄.operator norm₄.limit)
        (.euclidean norm₅.operator norm₅.limit) accepts zeroth first second third fourth fifth L
        32 32 16 16 16 start read commit value place)
      towerError 142 21449539584 := by
  have loops : 1 + (∑ g, (root.shape g).blocks + 1 + (level₁.shape.blocks + 1 +
      (level₂.shape.blocks + 1 + (level₃.shape.blocks + 1 + (level₄.shape.blocks + 1 +
        (level₅.shape.blocks + 1 + terminal.blocks)))))) = 142 := by decide
  have tree : 2 * (2 * (∑ g, (root.shape g).blocks + 1) * (2 * (level₁.shape.blocks + 1) *
      (2 * (level₂.shape.blocks + 1) * (2 * (level₃.shape.blocks + 1) *
        (2 * (level₄.shape.blocks + 1) * (2 * (level₅.shape.blocks + 1) *
          (terminal.blocks + 1))))))) = 21449539584 := by decide
  have suffix := mixedTower_suffix (accept := accepts) (L' := L) injective units sparse
    root_sized.2.2.1 root_sized.2.2.2.1 root_sized.2.2.2.2
    ((by decide : rootVars ≤ 22).trans slots) ((by decide : rootBatch ≤ 22).trans slots)
    ((by decide : 3 ≤ 22).trans slots)
    (levelSized slots (by decide) (by decide) level₁_sized.2.2.2.1 level₁_sized.2.2.2.2
      injective₁ units₁ sparse₁)
    (levelSized slots (by decide) (by decide) sized₂.2.2.1 sized₂.2.2.2 injective₂ units₂ sparse₂)
    (levelSized slots (by decide) (by decide) sized₃.2.2.1 sized₃.2.2.2 injective₃ units₃ sparse₃)
    (levelSized slots (by decide) (by decide) sized₄.2.2.1 sized₄.2.2.2 injective₄ units₄ sparse₄)
    (levelSized slots (by decide) (by decide) sized₅.2.2.1 sized₅.2.2.2 injective₅ units₅ sparse₅)
    ⟨bounded₃, (norms_sized (level₃, norm₃) (by simp)).1,
      (norms_sized (level₃, norm₃) (by simp)).2⟩
    ⟨bounded₄, (norms_sized (level₄, norm₄) (by simp)).1,
      (norms_sized (level₄, norm₄) (by simp)).2⟩
    ⟨bounded₅, (norms_sized (level₅, norm₅) (by simp)).1,
      (norms_sized (level₅, norm₅) (by simp)).2⟩
    next inside quarters.1 quarters.2.1 quarters.2.2.1 quarters.2.2.2.1 quarters.2.2.2.2
    injectiveL unitsL boundedL commit₁ commit₂ commit₃ commit₄ commit₅ stateOf
    (fun _ accepted => ⟨accepted.1, accepted.2.1⟩) (fun _ accepted => accepted.2.2.1)
    (fun _ accepted => accepted.2.2.2.1) (fun _ accepted => accepted.2.2.2.2.1)
    (fun _ accepted => accepted.2.2.2.2.2.1) (fun _ accepted => accepted.2.2.2.2.2.2.1)
    (fun _ accepted => accepted.2.2.2.2.2.2.2.1) (fun _ accepted => accepted.2.2.2.2.2.2.2.2)
  rw [loops, tree] at suffix
  exact suffix

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- The sum of the stage errors of `tower_suffix` (Akita (206)) is at most
`317479/(1024·|E|) + 135·p`, when each challenge family has the slot probability at most `p`.
Each error of a check has the factor `2^{−g}` for the work bits `g` of its site in the Akita code
(Remark 10.18, `workBits`); the fold term has no work credit (`uncredited`). The fold coordinates
are 56 of the root, 28, 21, 11, 7 and 7 of the levels 1 to 5, and 5 of the terminal:
`56 + 28 + 21 + 11 + 7 + 7 + 5 = 135`.
The draws of the checks of the root and of the levels 1 to 5, with the tensor reduction and the
direct norm check, give `74367/2048`, `92479/2048`, `13623/256`, `13199/256`, `7907/128` and
`7907/128`, and the weights of the batching root give `1/4`. -/
theorem towerError_le {p : ℝ} (chance : ∀ g, slotChance (decode g) ≤ p)
    (chance₁ : slotChance decode₁ ≤ p) (chance₂ : slotChance decode₂ ≤ p)
    (chance₃ : slotChance decode₃ ≤ p) (chance₄ : slotChance decode₄ ≤ p)
    (chance₅ : slotChance decode₅ ≤ p) (chanceL : slotChance L.decode ≤ p) :
    towerError ≤ 317479 / (1024 * Fintype.card E) + 135 * p := by
  have sum₀ : 1 / 2 ^ Tier.foldBits zeroth * ∑ b, Tier.chance zeroth b ≤ (56 : ℕ) * p :=
    sum_le_mul (root.card_block.trans (by decide)) (fun _ => slotChance_nonnegative _)
      fun b => chance b.1
  have sum₁ : 1 / 2 ^ Tier.foldBits first * ∑ b, Tier.chance first b ≤ (28 : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) (fun _ => slotChance_nonnegative _) fun _ => chance₁
  have sum₂ : 1 / 2 ^ Tier.foldBits second * ∑ b, Tier.chance second b ≤ (21 : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) (fun _ => slotChance_nonnegative _) fun _ => chance₂
  have sum₃ : 1 / 2 ^ Tier.foldBits third * ∑ b, Tier.chance third b ≤ (11 : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) (fun _ => slotChance_nonnegative _) fun _ => chance₃
  have sum₄ : 1 / 2 ^ Tier.foldBits fourth * ∑ b, Tier.chance fourth b ≤ (7 : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) (fun _ => slotChance_nonnegative _) fun _ => chance₄
  have sum₅ : 1 / 2 ^ Tier.foldBits fifth * ∑ b, Tier.chance fifth b ≤ (7 : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) (fun _ => slotChance_nonnegative _) fun _ => chance₅
  have last : 1 / 2 ^ Terminal.foldBits L * ((terminal.blocks : ℝ) * slotChance L.decode) ≤
      (5 : ℕ) * p :=
    uncredited (mul_nonneg (Nat.cast_nonneg _) (slotChance_nonnegative _))
      (mul_le_mul_of_nonneg_left chanceL (Nat.cast_nonneg _))
  have rooted : Root.rootBits 3 = 2 := by decide
  have positive : (0 : ℝ) < Fintype.card E := by exact_mod_cast Fintype.card_pos
  have w1 : workBits 1 = 1 := by decide
  have w2 : workBits 2 = 2 := by decide
  have w3 : workBits 3 = 2 := by decide
  have w4 : workBits 4 = 3 := by decide
  have w5 : workBits 5 = 3 := by decide
  have w7 : workBits 7 = 3 := by decide
  have w17 : workBits 17 = 5 := by decide
  have w18 : workBits 18 = 5 := by decide
  have w19 : workBits 19 = 5 := by decide
  have w21 : workBits 21 = 5 := by decide
  have w22 : workBits 22 = 5 := by decide
  have w127 : workBits 127 = 7 := by decide
  have w255 : workBits 255 = 8 := by decide
  have w2047 : workBits 2047 = 11 := by decide
  have costs₀ : (credit (Tier.costs zeroth) (Tier.bits zeroth)).sum =
      74367 / 2048 / (Fintype.card E : ℝ) := by
    dsimp only [liftBatch]
    rw [foldCredit_eq]
    rw [show 2 * root.largestDegree - 1 = 2047 by decide, show 2 * root.innerMax - 1 = 2047 by
      decide,
      show root.compressed = true by decide]
    simp only [foldCredit, treeCredit, show rangeArities 8 = [] by decide,
      show rangeLeaf 8 = 4 by decide, show leafDegree 8 = 4 by decide, rootVars, rootBatch,
      w1, w3, w5, w22, w2047, ite_true]
    push_cast
    field_simp
    ring
  have costs₁ : (credit (Tier.costs first) (Tier.bits first)).sum =
      92479 / 2048 / (Fintype.card E : ℝ) := by
    dsimp only [Tier.tensored, liftFold]
    rw [credit_append (by rw [List.length_map]; exact foldLosses_length ..), List.sum_append,
      foldCredit_eq, tensorCredit_eq]
    rw [show 2 * level₁.opening.largestDegree - 1 = 2047 by decide,
      show 2 * level₁.shape.innerDegree - 1 = 2047 by decide,
      show level₁.opening.compressed level₁.fold = true by decide]
    simp only [foldCredit, tensorCredit, treeCredit, show level₁.shape.certifiedBase = 8 by rfl,
      show rangeArities 8 = [] by decide, show rangeLeaf 8 = 4 by decide,
      show leafDegree 8 = 4 by decide, show level₁.vars = 21 by rfl, show level₁.batch = 4 by rfl,
      show level₁.tail = 20 by rfl, w1, w2, w3, w4, w5, w21, w2047, ite_true]
    push_cast
    field_simp
    ring
  have costs₂ : (credit (Tier.costs second) (Tier.bits second)).sum =
      13623 / 256 / (Fintype.card E : ℝ) := by
    dsimp only [Tier.tensored, committedTrace]
    rw [credit_append (by rw [List.length_map]; exact foldLosses_length ..), List.sum_append,
      foldCredit_eq, tensorCredit_eq]
    rw [show level₂.opening.largestDegree - 1 = 127 by decide,
      show 2 * level₂.shape.innerDegree - 1 = 255 by decide,
      show level₂.opening.compressed level₂.fold = true by decide]
    simp only [foldCredit, tensorCredit, treeCredit, show level₂.shape.certifiedBase = 32 by rfl,
      show rangeArities 32 = [4] by decide, show rangeLeaf 32 = 4 by decide,
      show leafDegree 32 = 4 by decide, show level₂.vars = 19 by rfl, show level₂.batch = 4 by rfl,
      show level₂.tail = 18 by rfl, show (4 + 1 : ℕ) = 5 by rfl, show (1 * 4 - 1 : ℕ) = 3 by rfl,
      w1, w2, w3, w4, w5, w19, w255, ite_true]
    push_cast
    field_simp
    ring
  have costs₃ : (credit (Tier.costs third) (Tier.bits third)).sum =
      13199 / 256 / (Fintype.card E : ℝ) := by
    dsimp only [Tier.tensored, committedTraceSq]
    rw [credit_append (by rw [List.length_map]; exact foldLossesSq_length ..), List.sum_append,
      foldCreditSq_eq, tensorCredit_eq]
    rw [show level₃.opening.largestDegree - 1 = 127 by decide,
      show 2 * level₃.shape.innerDegree - 1 = 255 by decide,
      show level₃.opening.compressed level₃.fold = true by decide]
    simp only [foldCreditSq, tensorCredit, treeCredit, show level₃.shape.certifiedBase = 32 by rfl,
      show rangeArities 32 = [4] by decide, show rangeLeaf 32 = 4 by decide,
      show leafDegree 32 = 4 by decide, show level₃.vars = 18 by rfl, show level₃.batch = 4 by rfl,
      show level₃.tail = 17 by rfl, show (4 + 1 : ℕ) = 5 by rfl, show (1 * 4 - 1 : ℕ) = 3 by rfl,
      show max (4 + 1) 2 + 1 = 6 by rfl, w1, w2, w3, w4, w5, w18, w255, ite_true]
    push_cast
    field_simp
    ring
  have costs₄ : (credit (Tier.costs fourth) (Tier.bits fourth)).sum =
      7907 / 128 / (Fintype.card E : ℝ) := by
    dsimp only [Tier.tensored, committedTraceSq]
    rw [credit_append (by rw [List.length_map]; exact foldLossesSq_length ..), List.sum_append,
      foldCreditSq_eq, tensorCredit_eq]
    rw [show level₄.opening.largestDegree - 1 = 63 by decide,
      show 2 * level₄.shape.innerDegree - 1 = 127 by decide,
      show level₄.opening.compressed level₄.fold = false by decide]
    simp only [foldCreditSq, tensorCredit, treeCredit, show level₄.shape.certifiedBase = 64 by rfl,
      show rangeArities 64 = [2, 4] by decide, show rangeLeaf 64 = 4 by decide,
      show leafDegree 64 = 4 by decide, show level₄.vars = 17 by rfl, show level₄.batch = 4 by rfl,
      show level₄.tail = 16 by rfl, show (2 + 1 : ℕ) = 3 by rfl, show (1 * 2 - 1 : ℕ) = 1 by rfl,
      show (4 + 1 : ℕ) = 5 by rfl, show (1 * 2 * 4 - 1 : ℕ) = 7 by rfl,
      show max (4 + 1) 2 + 1 = 6 by rfl, w1, w2, w3, w4, w5, w7, w17, w127]
    push_cast
    field_simp
    ring
  have costs₅ : (credit (Tier.costs fifth) (Tier.bits fifth)).sum =
      7907 / 128 / (Fintype.card E : ℝ) := by
    dsimp only [Tier.tensored, committedTraceSq]
    rw [credit_append (by rw [List.length_map]; exact foldLossesSq_length ..), List.sum_append,
      foldCreditSq_eq, tensorCredit_eq]
    rw [show level₅.opening.largestDegree - 1 = 63 by decide,
      show 2 * level₅.shape.innerDegree - 1 = 127 by decide,
      show level₅.opening.compressed level₅.fold = false by decide]
    simp only [foldCreditSq, tensorCredit, treeCredit, show level₅.shape.certifiedBase = 64 by rfl,
      show rangeArities 64 = [2, 4] by decide, show rangeLeaf 64 = 4 by decide,
      show leafDegree 64 = 4 by decide, show level₅.vars = 17 by rfl, show level₅.batch = 4 by rfl,
      show level₅.tail = 16 by rfl, show (2 + 1 : ℕ) = 3 by rfl, show (1 * 2 - 1 : ℕ) = 1 by rfl,
      show (4 + 1 : ℕ) = 5 by rfl, show (1 * 2 * 4 - 1 : ℕ) = 7 by rfl,
      show max (4 + 1) 2 + 1 = 6 by rfl, w1, w2, w3, w4, w5, w7, w17, w127]
    push_cast
    field_simp
    ring
  unfold Tier.levelError Terminal.termError
  rw [costs₀, costs₁, costs₂, costs₃, costs₄, costs₅, rooted]
  push_cast at sum₀ sum₁ sum₂ sum₃ sum₄ sum₅ last ⊢
  rw [show (1024 : ℝ) * Fintype.card E = 1024 * Fintype.card E from rfl]
  have split : 317479 / (1024 * (Fintype.card E : ℝ)) =
      1 / (Fintype.card E : ℝ) / 2 ^ 2 + (74367 / 2048 / (Fintype.card E : ℝ) +
        (92479 / 2048 / (Fintype.card E : ℝ) + (13623 / 256 / (Fintype.card E : ℝ) +
          (13199 / 256 / (Fintype.card E : ℝ) + (7907 / 128 / (Fintype.card E : ℝ) +
            7907 / 128 / (Fintype.card E : ℝ)))))) := by
    field_simp
    ring
  rw [split]
  nlinarith [sum₀, sum₁, sum₂, sum₃, sum₄, sum₅, last]

end NightstreamFPrime.Spec.Compression.Akita.Key20
