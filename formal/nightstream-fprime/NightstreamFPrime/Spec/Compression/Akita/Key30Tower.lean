import NightstreamFPrime.Spec.Compression.Akita.Key20Tower
import NightstreamFPrime.Spec.Compression.Akita.Key30
import NightstreamFPrime.Spec.Compression.Akita.Extract.OffTower

/-! The opening suffix of Akita for the schedule file `crates/ns-compress/artifacts/key30.aks`
(Akita, ePrint 2026/1983, revision of 2026-10-15, Section 9 with (136) and Lemma 9.1,
Proposition 10.44, Theorem 10.46, Corollary 10.35 (ii) and Corollary 10.49 with (206)).

- `tower_suffix` is `offTower_suffix` for the numbers of a row of `Key30.lean`, under a verifier
  that accepts only with `offLinks`. The retry loops are `1467 + B_1 + B_2` and the tree size is
  `54400647168·(129 + B_1)·(B_2 + 1)`, for the live blocks `B_1` of the witness group of level 1
  and `B_2` of level 2: 1678 and 724616620277760 for `W` of 27 variables, and 1668 and
  686971372437504 for `W` of 26 variables (`tower_counts`).
- Its hypotheses are: an answer of the oracle has 28 slots, the fold coordinates of each level
  have different slots, each challenge family has units as differences and its norm bound, the
  premise `Batch.Registered` of Akita Lemma 9.1 for the commitment `registry` of the setup prefix,
  and the stage of the terminal. The split `scan` of the public weight of the root (Akita (132))
  is a parameter.
- `towerError_le` bounds the sum of the stage errors by `8132/|E| + (1458 + B_1 + B_2)·p` when
  each challenge family has the slot probability at most `p`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key30

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Checkpoint
open Key20 (card_field foldCosts_total sum_le_mul)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The retry loops and the tree size of the suffix of each row. -/
theorem tower_counts : ∀ r ∈ rows,
    1 + (∑ g, ((root r).shape g).blocks + 1 + (1 + (∑ g, ((level₁ r).shape g).blocks + 1 +
      ((level₂ r).shape.blocks + 1 + (level₃.shape.blocks + 1 + (level₄.shape.blocks + 1 +
        (level₅.shape.blocks + 1 + (level₆.shape.blocks + 1 + terminal.blocks)))))))) =
      1467 + r.blocks₁ + r.blocks₂ ∧
    2 * (2 * (∑ g, ((root r).shape g).blocks + 1) * (2 * (2 *
      (∑ g, ((level₁ r).shape g).blocks + 1) * (2 * ((level₂ r).shape.blocks + 1) *
        (2 * (level₃.shape.blocks + 1) * (2 * (level₄.shape.blocks + 1) *
          (2 * (level₅.shape.blocks + 1) * (2 * (level₆.shape.blocks + 1) *
            (terminal.blocks + 1))))))))) =
      54400647168 * ((129 + r.blocks₁) * (r.blocks₂ + 1)) ∧
    [1467 + w27.blocks₁ + w27.blocks₂, 54400647168 * ((129 + w27.blocks₁) * (w27.blocks₂ + 1)),
      1467 + w26.blocks₁ + w26.blocks₂, 54400647168 * ((129 + w26.blocks₁) * (w26.blocks₂ + 1))] =
      [1678, 724616620277760, 1668, 686971372437504] := by decide

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {total : ℕ}
  {read : Reader Wire} {start : ℕ} {r : Row} {member : r ∈ rows}
  {accept : Prefix Wire (Coord → E) total → Prop}
  {β : ∀ g, Module.Basis (Fin ((root r).planes g)) Zq E}
  {key : (g : Fin 3) → Key ((root r).shape g)} {openKey : (root r).OpenKey}
  {claim : Wire → (root r).Claim E} {block : (root r).Block → Coord}
  {decode : (g : Fin 3) → E → Option (Rq ((root r).carrierDegree g))} {mass : Fin 3 → ℕ}
  {flat : Table}
  {scan : ∀ (st : (root r).Statement E) (challenge : (root r).Challenge),
    SetupScan (Batch.liftedReduction β st challenge rootVars (root_sized r member).1 8
      quotientDigits₀ (root_sized r member).2.1 rootBatch (root_sized r member).2.2.1) setupVars
      flat}
  {β₁ : ∀ g, Module.Basis (Fin ((level₁ r).planes g)) Zq E}
  {key₁ : (g : Fin 2) → Key ((level₁ r).shape g)} {openKey₁ : (level₁ r).OpenKey}
  {claim₁ : Wire → (level₁ r).Claim E} {block₁ : (level₁ r).Block → Coord}
  {decode₁ : (g : Fin 2) → E → Option (Rq ((level₁ r).carrierDegree g))} {mass₁ : Fin 2 → ℕ}
  {registry : ((level₁ r).shape 0).Payload} {reference : Decommitment ((level₁ r).shape 0)}
  {β₂ : Module.Basis (Fin (level₂ r).opening.planes) Zq E} {key₂ : Key (level₂ r).shape}
  {openKey₂ : OpeningKey (level₂ r).opening} {claim₂ : Wire → Claim (level₂ r).opening E}
  {block₂ : Fin (level₂ r).shape.blocks → Coord}
  {decode₂ : E → Option (Rq (level₂ r).opening.carrierDegree)} {weight₂ : ℕ}
  {β₃ : Module.Basis (Fin level₃.opening.planes) Zq E} {key₃ : Key level₃.shape}
  {openKey₃ : OpeningKey level₃.opening} {claim₃ : Wire → Claim level₃.opening E}
  {block₃ : Fin level₃.shape.blocks → Coord}
  {decode₃ : E → Option (Rq level₃.opening.carrierDegree)} {weight₃ : ℕ}
  {key₄ : Key level₄.shape} {openKey₄ : OpeningKey level₄.opening}
  {claim₄ : Wire → TraceClaim level₄.opening E} {block₄ : Fin level₄.shape.blocks → Coord}
  {decode₄ : E → Option (Rq level₄.shape.innerDegree)} {weight₄ : ℕ}
  {key₅ : Key level₅.shape} {openKey₅ : OpeningKey level₅.opening}
  {claim₅ : Wire → TraceClaim level₅.opening E} {block₅ : Fin level₅.shape.blocks → Coord}
  {decode₅ : E → Option (Rq level₅.shape.innerDegree)} {weight₅ : ℕ}
  {key₆ : Key level₆.shape} {openKey₆ : OpeningKey level₆.opening}
  {claim₆ : Wire → TraceClaim level₆.opening E} {block₆ : Fin level₆.shape.blocks → Coord}
  {decode₆ : E → Option (Rq level₆.shape.innerDegree)} {weight₆ : ℕ}
  {L : Terminal terminal E Wire Coord E total
    (.euclidean normTerminal.operator normTerminal.limit)}
  {commit : Wire → (g : Fin 3) → ((root r).shape g).Payload} {value : Wire → Fin 3 → E}
  {place : Wire → Fin 3 → List E}
  {commits : Wire → (g : Fin 2) → ((level₁ r).shape g).Payload} {values : Wire → Fin 2 → E}
  {places : Wire → Fin 2 → List E} {commit₁ : Wire → ((level₁ r).shape 1).Payload}
  {commit₂ : Wire → (level₂ r).shape.Payload} {commit₃ : Wire → level₃.shape.Payload}
  {commit₄ : Wire → level₄.shape.Payload} {commit₅ : Wire → level₅.shape.Payload}
  {commit₆ : Wire → level₆.shape.Payload}
  {stateOf : Wire → Fin terminal.blocks → Fin terminal.innerRank → Rq terminal.innerDegree}

local notation "zeroth" =>
  offBatch β key openKey claim rootVars (And.left (root_sized r member)) 8 quotientDigits₀
    (And.left (And.right (root_sized r member))) rootBatch
    (And.left (And.right (And.right (root_sized r member)))) block decode read mass (start + 1)
    setupVars flat scan

local notation "first" =>
  Tier.pinned (liftBatch β₁ key₁ openKey₁ claim₁ vars₁ (And.left (level₁_sized r member)) 16
    quotientDigits (And.left (And.right (level₁_sized r member))) batch₁
    (And.left (And.right (And.right (level₁_sized r member)))) block₁ decode₁ read mass₁
    (Tier.finish zeroth + 1)) (Batch.Pinned 0 registry flat)
    (fun ω _ => Batch.pinBreak key₁ openKey₁ _ 0 reference ω)

local notation "second" =>
  liftFold β₂ key₂ openKey₂ claim₂ (Key20.Level.vars (level₂ r))
    (And.left (level₂_sized r member)) quotientDigits
    (And.left (And.right (level₂_sized r member))) (Key20.Level.batch (level₂ r))
    (And.left (And.right (And.right (level₂_sized r member)))) block₂ decode₂ read weight₂
    (Tier.finish first)

local notation "third" =>
  Tier.tensored (slimFold β₃ key₃ openKey₃ claim₃ level₃.vars (And.left level₃_sized)
    quotientDigits (And.left (And.right level₃_sized)) level₃.batch
    (And.left (And.right (And.right level₃_sized))) block₃ decode₃ read weight₃
    (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := level₄.fold) key₄ openKey₄ claim₄ level₄.vars
    (And.left level₄_sized) level₄.batch (And.left (And.right level₄_sized)) block₄ decode₄ read
    weight₄ (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅ claim₅ level₅.vars
    (And.left (tails_sized level₅ (List.mem_cons_self ..))) level₅.batch
    (And.left (And.right (tails_sized level₅ (List.mem_cons_self ..)))) block₅ decode₅
    read weight₅ norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail

local notation "sixth" =>
  Tier.tensored (committedTraceSq (fold := level₆.fold) key₆ openKey₆ claim₆ level₆.vars
    (And.left (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))) level₆.batch
    (And.left (And.right (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))))
    block₆ decode₆ read weight₆ norm₆.operator norm₆.limit (Tier.finish fifth))
    level₆.tail

local notation "accepts" =>
  offLinks (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
    (route₅ := Route.euclidean norm₅.operator norm₅.limit)
    (route₆ := Route.euclidean norm₆.operator norm₆.limit) zeroth first second third fourth fifth
    sixth L start read commit value place commits values places registry commit₁ commit₂ commit₃
    (And.left quarters) (And.left (And.right quarters))
    (And.left (And.right (And.right quarters))) (And.right (And.right (And.right quarters)))
    commit₄ commit₅ commit₆ stateOf

local notation "towerError" =>
  (1 / (Fintype.card E : ℝ) + (∑ b, Tier.chance zeroth b + List.sum (Tier.costs zeroth) +
    (1 / (Fintype.card E : ℝ) + (∑ b, Tier.chance first b + List.sum (Tier.costs first) +
      (∑ b, Tier.chance second b + List.sum (Tier.costs second) +
        (∑ b, Tier.chance third b + List.sum (Tier.costs third) +
          (∑ b, Tier.chance fourth b + List.sum (Tier.costs fourth) +
            (∑ b, Tier.chance fifth b + List.sum (Tier.costs fifth) +
              (∑ b, Tier.chance sixth b + List.sum (Tier.costs sixth) +
                terminal.blocks * slotChance (Terminal.decode L))))))))))

/-- Akita Proposition 10.44, Theorem 10.46 and Lemma 9.1 for a row of the schedule `key30.aks`:
the batching root over the offloaded fold of the key polynomial, `W` and `V`, the level 1 with
the group of the setup prefix and the group of the witness, the packing levels 2 and 3, the trace
levels 4 to 6 and the terminal is a suffix from the stage `start` under each verifier `accept`
that accepts only with `offLinks` (`linked`, `offTower_suffix`). The levels 1 to 4 have the coefficient route, and the levels 5 and 6 and the
terminal have the Euclidean route. The suffix has `1467 + B_1 + B_2` retry loops and the tree size
`54400647168·(129 + B_1)·(B_2 + 1)` (`tower_counts`). `slots`: an answer of the oracle has 28
slots. `injective`, `units`, `sparse` and `bounded`: the fold coordinates of each level have
different slots, and the challenge family of each level has units as differences (Akita
Definition 9.2, E1), members with `‖c‖_1 = ω`, and at the levels 5 and 6 and at the terminal
`‖c‖_mul,2 ≤ Γ`. `registered` and `positive₁`: the premise of Akita Lemma 9.1 for the commitment
`registry` of the setup prefix `flat`, and `ω ≥ 1` for the group of the setup prefix. `next` and
`inside`: the fold challenge of the terminal is the stage after the stages of the level 6, and it
is a stage of the transcript. -/
theorem tower_suffix {peak : Fin 3 → ℕ} {peak₁ : Fin 2 → ℕ} (slots : 28 ≤ Fintype.card Coord)
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (injective₁ : Function.Injective block₁)
    (units₁ : ∀ g, UnitDifferences {challenge | ∃ slot, decode₁ g slot = some challenge})
    (sparse₁ : ∀ g challenge, (∃ slot, decode₁ g slot = some challenge) →
      Sparse (mass₁ g) (peak₁ g) challenge)
    (registered : Batch.Registered (key₁ 0) registry flat reference) (positive₁ : 0 < mass₁ 0)
    (injective₂ : Function.Injective block₂)
    (units₂ : UnitDifferences {challenge | ∃ slot, decode₂ slot = some challenge})
    (sparse₂ : ∃ peak, ∀ challenge, (∃ slot, decode₂ slot = some challenge) →
      Sparse weight₂ peak challenge)
    (injective₃ : Function.Injective block₃)
    (units₃ : UnitDifferences {challenge | ∃ slot, decode₃ slot = some challenge})
    (sparse₃ : ∃ peak, ∀ challenge, (∃ slot, decode₃ slot = some challenge) →
      Sparse weight₃ peak challenge)
    (injective₄ : Function.Injective block₄)
    (units₄ : UnitDifferences {challenge | ∃ slot, decode₄ slot = some challenge})
    (sparse₄ : ∃ peak, ∀ challenge, (∃ slot, decode₄ slot = some challenge) →
      Sparse weight₄ peak challenge)
    (injective₅ : Function.Injective block₅)
    (units₅ : UnitDifferences {challenge | ∃ slot, decode₅ slot = some challenge})
    (sparse₅ : ∃ peak, ∀ challenge, (∃ slot, decode₅ slot = some challenge) →
      Sparse weight₅ peak challenge)
    (bounded₅ : ∀ challenge, (∃ slot, decode₅ slot = some challenge) →
      MulBound norm₅.operator challenge)
    (injective₆ : Function.Injective block₆)
    (units₆ : UnitDifferences {challenge | ∃ slot, decode₆ slot = some challenge})
    (sparse₆ : ∃ peak, ∀ challenge, (∃ slot, decode₆ slot = some challenge) →
      Sparse weight₆ peak challenge)
    (bounded₆ : ∀ challenge, (∃ slot, decode₆ slot = some challenge) →
      MulBound norm₆.operator challenge)
    (next : L.start = Tier.finish sixth) (inside : L.start < total)
    (injectiveL : Function.Injective L.block) (unitsL : UnitDifferences L.family)
    (boundedL : ∀ challenge ∈ L.family, MulBound normTerminal.operator challenge)
    (linked : ∀ point, accept point → accepts point) :
    Tier.Suffix accept start (OpensAt start commit value place key mass)
      (offRoot (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
        (route₅ := Route.euclidean norm₅.operator norm₅.limit)
        (route₆ := Route.euclidean norm₆.operator norm₆.limit) accept zeroth first second third
        fourth fifth sixth L 32 16 16 16 start read commit value place commits values places flat)
      towerError (1467 + r.blocks₁ + r.blocks₂)
      (54400647168 * ((129 + r.blocks₁) * (r.blocks₂ + 1))) := by
  have sized₀ := root_sized r member
  have sized₁ := level₁_sized r member
  have sized₂ := level₂_sized r member
  have sized₅ := tails_sized level₅ (List.mem_cons_self ..)
  have sized₆ := tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..))
  have normed₅ := norms_sized (level₅, norm₅) (List.mem_cons_self ..)
  have normed₆ := norms_sized (level₆, norm₆) (List.mem_cons_of_mem _ (List.mem_cons_self ..))
  have covers : ((level₁ r).fold 0).sourceBound ((level₁ r).shape 0) +
      2 * mass₁ 0 * (((level₁ r).shape 0).sourceBase / 2) ≤
        2 * (2 * mass₁ 0) * ((level₁ r).fold 0).sourceBound ((level₁ r).shape 0) := by
    have source : ((level₁ r).fold 0).sourceBound ((level₁ r).shape 0) = 65535 :=
      show Descriptor.sourceBound ⟨31, 16, 4⟩ shapeS = 65535 by decide
    have digit : ((level₁ r).shape 0).sourceBase / 2 = 128 :=
      show shapeS.sourceBase / 2 = 128 by decide
    rw [source, digit]
    nlinarith
  have suffix := offTower_suffix (route₄ := Route.coefficient)
    (route₅ := Route.euclidean norm₅.operator norm₅.limit)
    (route₆ := Route.euclidean norm₆.operator norm₆.limit) (accept := accept) (T₀ := zeroth) (T₁ := first) (T₂ := second)
    (T₃ := third) (T₄ := fourth) (T₅ := fifth) (T₆ := sixth) (L := L) (flat := flat)
    (registry := registry) (start := start) (read := read) (commit := commit) (value := value)
    (place := place) (commits := commits) (values := values) (places := places)
    (root r).card_block (level₁ r).card_block
    (offBatch_valid injective units sparse sized₀.2.2.2.1 sized₀.2.2.2.2.1 sized₀.2.2.2.2.2
      ((by decide : rootVars ≤ 28).trans slots) ((by decide : rootBatch ≤ 28).trans slots))
    (Tier.pinned_valid
      (liftBatch_valid injective₁ units₁ sparse₁ sized₁.2.2.2.1 sized₁.2.2.2.2.1
        sized₁.2.2.2.2.2 ((by decide : vars₁ ≤ 28).trans slots)
        ((by decide : batch₁ ≤ 28).trans slots))
      fun ω claim opens unpinned =>
        Batch.pin_fixed registered.opens registered.short covers ω claim opens unpinned)
    (by
      obtain ⟨peak₂, sparse₂⟩ := sparse₂
      exact liftFold_valid injective₂ units₂ sparse₂ sized₂.2.2.2.1 sized₂.2.2.2.2
        ((le_of_eq_of_le (rfl : (level₂ r).vars = 21) (by decide : 21 ≤ 28)).trans slots)
        ((le_of_eq_of_le (rfl : (level₂ r).batch = 4) (by decide : 4 ≤ 28)).trans slots))
    (LevelSized.slim ⟨injective₃, units₃, sparse₃, level₃_sized.2.2.2.1, level₃_sized.2.2.2.2,
      (by decide : level₃.vars ≤ 28).trans slots, (by decide : level₃.batch ≤ 28).trans slots⟩)
    (LevelSized.trace ⟨injective₄, units₄, sparse₄, level₄_sized.2.2.1, level₄_sized.2.2.2,
      (by decide : level₄.vars ≤ 28).trans slots, (by decide : level₄.batch ≤ 28).trans slots⟩)
    (LevelSized.traceSq ⟨injective₅, units₅, sparse₅, sized₅.2.2.1, sized₅.2.2.2,
      (by decide : level₅.vars ≤ 28).trans slots, (by decide : level₅.batch ≤ 28).trans slots⟩
      ⟨bounded₅, normed₅.1, normed₅.2⟩)
    (LevelSized.traceSq ⟨injective₆, units₆, sparse₆, sized₆.2.2.1, sized₆.2.2.2,
      (by decide : level₆.vars ≤ 28).trans slots, (by decide : level₆.batch ≤ 28).trans slots⟩
      ⟨bounded₆, normed₆.1, normed₆.2⟩)
    (exists_fair_slots ((by decide : 3 ≤ 28).trans slots))
    (exists_fair_slots ((by decide : 2 ≤ 28).trans slots)) rfl rfl rfl rfl rfl rfl rfl rfl rfl
    rfl rfl rfl rfl rfl next inside quarters.1 quarters.2.1 quarters.2.2.1 quarters.2.2.2
    offBatch_uniform (fun _ r2 v ρ s opened setup => ⟨r2, v, ρ, s, rfl, opened, setup⟩)
    (fun _ _ table other => foldCheck_bare _ _ _ table other)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)
    (fun _ _ table other => foldCheck_bare _ _ _ table other)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)
    (Tier.tensored_uniform fun _ _ table other => foldCheck_bare _ _ _ table other)
    (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    (Tier.tensored_uniform committedTrace_uniform) (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    (Tier.tensored_uniform committedTraceSq_uniform)
    (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    (Tier.tensored_uniform committedTraceSq_uniform)
    (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    injectiveL unitsL boundedL commit₁ commit₂ commit₃ commit₄ commit₅ commit₆ stateOf linked
  rw [(tower_counts r member).1, (tower_counts r member).2.1] at suffix
  exact suffix

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- The sum of the stage errors of `tower_suffix` (Akita (206)) is at most
`8132/|E| + (1458 + B_1 + B_2)·p`, when each challenge family has the slot probability at most
`p`. The fold coordinates are 1280 of the root, `128 + B_1` of the level 1, `B_2`, 23, 8, 7 and 7
of the levels 2 to 6, and 5 of the terminal. The 8132 are 1 for the weights of each of the two
batching stages, 4429 for the draws of the checks of the root with the 23 rounds of Stage 3
(Akita (134)), and 1343, 535, 546, 441, 418 and 418 for the draws of the checks of the levels 1
to 6, with the tensor reduction and the direct norm check. -/
theorem towerError_le {p : ℝ} (chance : ∀ g, slotChance (decode g) ≤ p)
    (chance₁ : ∀ g, slotChance (decode₁ g) ≤ p) (chance₂ : slotChance decode₂ ≤ p)
    (chance₃ : slotChance decode₃ ≤ p) (chance₄ : slotChance decode₄ ≤ p)
    (chance₅ : slotChance decode₅ ≤ p) (chance₆ : slotChance decode₆ ≤ p)
    (chanceL : slotChance L.decode ≤ p) :
    towerError ≤ 8132 / (Fintype.card E : ℝ) + ((1458 + r.blocks₁ + r.blocks₂ : ℕ) : ℝ) * p := by
  have row : r = w27 ∨ r = w26 := by simpa [rows] using member
  have sum₀ : ∑ b, Tier.chance zeroth b ≤ (1280 : ℕ) * p :=
    sum_le_mul ((root r).card_block.trans (by rcases row with rfl | rfl <;> decide))
      fun b => chance b.1
  have sum₁ : ∑ b, Tier.chance first b ≤ ((128 + r.blocks₁ : ℕ) : ℝ) * p :=
    sum_le_mul ((level₁ r).card_block.trans (by rcases row with rfl | rfl <;> decide))
      fun b => chance₁ b.1
  have sum₂ : ∑ b, Tier.chance second b ≤ (r.blocks₂ : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) fun _ => chance₂
  have sum₃ : ∑ b, Tier.chance third b ≤ (23 : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) fun _ => chance₃
  have sum₄ : ∑ b, Tier.chance fourth b ≤ (8 : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) fun _ => chance₄
  have sum₅ : ∑ b, Tier.chance fifth b ≤ (7 : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) fun _ => chance₅
  have sum₆ : ∑ b, Tier.chance sixth b ≤ (7 : ℕ) * p :=
    sum_le_mul (Fintype.card_fin _) fun _ => chance₆
  have last : (terminal.blocks : ℝ) * slotChance L.decode ≤ (5 : ℕ) * p :=
    mul_le_mul_of_nonneg_left chanceL (Nat.cast_nonneg _)
  have costs₀ : (Tier.costs zeroth).sum = (4429 : ℕ) / (Fintype.card E : ℝ) := by
    show (foldCosts E (2 * (root r).largestDegree - 1) rootVars rootBatch 8 ++
      setupCosts E setupVars).sum = _
    rw [List.sum_append, foldCosts_total, setupCosts_sum, ← add_div, ← Nat.cast_add]
    exact congrArg (fun count : ℕ => (count : ℝ) / Fintype.card E)
      (by rcases row with rfl | rfl <;> decide)
  have costs₁ : (Tier.costs first).sum = (1343 : ℕ) / (Fintype.card E : ℝ) := by
    show (foldCosts E (2 * (level₁ r).largestDegree - 1) vars₁ batch₁ 16).sum = _
    rw [foldCosts_total]
    exact congrArg (fun count : ℕ => (count : ℝ) / Fintype.card E)
      (by rcases row with rfl | rfl <;> decide)
  have costs₂ : (Tier.costs second).sum = (535 : ℕ) / (Fintype.card E : ℝ) := by
    show (foldCosts E (2 * (level₂ r).opening.largestDegree - 1) (level₂ r).vars
      (level₂ r).batch (level₂ r).shape.certifiedBase).sum = _
    rw [foldCosts_total]
    exact congrArg (fun count : ℕ => (count : ℝ) / Fintype.card E)
      (by rcases row with rfl | rfl <;> decide)
  have costs₃ : (Tier.costs third).sum = (546 : ℕ) / (Fintype.card E : ℝ) := by
    show (foldCosts E (2 * level₃.opening.largestDegree - 1) level₃.vars level₃.batch
      level₃.shape.certifiedBase ++ tensorCosts E level₃.tail).sum = _
    rw [List.sum_append, foldCosts_total, tensorCosts_sum, ← add_div, ← Nat.cast_add]
    exact congrArg (fun count : ℕ => (count : ℝ) / Fintype.card E) (by decide)
  have costs₄ : (Tier.costs fourth).sum = (441 : ℕ) / (Fintype.card E : ℝ) := by
    show (foldCosts E (level₄.opening.largestDegree - 1) level₄.vars level₄.batch
      level₄.shape.certifiedBase ++ tensorCosts E level₄.tail).sum = _
    rw [List.sum_append, foldCosts_total, tensorCosts_sum, ← add_div, ← Nat.cast_add]
    exact congrArg (fun count : ℕ => (count : ℝ) / Fintype.card E) (by decide)
  have costs₅ : (Tier.costs fifth).sum = (418 : ℕ) / (Fintype.card E : ℝ) := by
    show ((foldCostsSq E (level₅.opening.largestDegree - 1) level₅.vars level₅.batch
      level₅.shape.certifiedBase) ++
        tensorCosts E level₅.tail).sum = _
    rw [List.sum_append, foldCostsSq_sum, tensorCosts_sum, ← add_div, ← Nat.cast_add]
    exact congrArg (fun count : ℕ => (count : ℝ) / Fintype.card E) (by decide)
  have costs₆ : (Tier.costs sixth).sum = (418 : ℕ) / (Fintype.card E : ℝ) := by
    show ((foldCostsSq E (level₆.opening.largestDegree - 1) level₆.vars level₆.batch
      level₆.shape.certifiedBase) ++
        tensorCosts E level₆.tail).sum = _
    rw [List.sum_append, foldCostsSq_sum, tensorCosts_sum, ← add_div, ← Nat.cast_add]
    exact congrArg (fun count : ℕ => (count : ℝ) / Fintype.card E) (by decide)
  have whole : 8132 / (Fintype.card E : ℝ) = 1 / (Fintype.card E : ℝ) +
      (4429 : ℕ) / (Fintype.card E : ℝ) + 1 / (Fintype.card E : ℝ) +
      (1343 : ℕ) / (Fintype.card E : ℝ) + (535 : ℕ) / (Fintype.card E : ℝ) +
      (546 : ℕ) / (Fintype.card E : ℝ) + (441 : ℕ) / (Fintype.card E : ℝ) +
      (418 : ℕ) / (Fintype.card E : ℝ) + (418 : ℕ) / (Fintype.card E : ℝ) := by
    push_cast
    ring
  rw [costs₀, costs₁, costs₂, costs₃, costs₄, costs₅, costs₆, whole]
  push_cast at sum₀ sum₁ sum₂ sum₃ sum₄ sum₅ sum₆ last ⊢
  linarith

end NightstreamFPrime.Spec.Compression.Akita.Key30
