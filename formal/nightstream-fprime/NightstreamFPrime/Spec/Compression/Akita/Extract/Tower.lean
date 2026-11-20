import NightstreamFPrime.Spec.Compression.Akita.Extract.Compose
import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink

/-! The opening suffix of Akita for a batching root, a fold with opening groups, a packing level,
four trace levels and the terminal (ePrint 2026/1983, revision of 2026-10-15, (127),
Proposition 10.44, Theorem 10.46, "Batching root", and Corollary 10.49 with (206)).

The levels have the order of a schedule with the root, level 1 with coefficient packing, and
levels 2 to 5 with evaluation trace before the terminal. Each weak opening has the coefficient
bound `β̄_∞ = Δ^cert_f`, and the terminal has the bound `Z_{∞,term}` of its clear response.

- `tower₅` to `tower₀` are the checkpoint structures of the levels 5 to 0 over the later stages,
  and `towerRoot` is the batching root over them.
- `tower_suffix` states that `towerRoot` is a suffix from the stage of the batching root whose
  first certificates satisfy `OpensAt`: the hypothesis `later` of `Compose.suffix`. Its links are
  `root_suffix`, `Tier.suffix_push` from the fold with groups to the packing level,
  `Tier.suffix_push_trace` from each level to the next trace level, and `Terminal.suffix_linked`.
- `liftTower_suffix` is `tower_suffix` for `liftBatch`, `liftFold` with the tensor reduction and
  four levels `committedTrace` with the tensor reduction. The hypotheses are conditions on the
  parameters and the links of the verifier.
- `towerCompose_suffix` puts the two checkpoints of the compression argument before `towerRoot`
  (`Compose.suffix`). The field of the argument is `Quadratic` here.

The squared-norm certificate of a level and of the terminal is not in this file. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Tower

variable {count : ℕ} {B : Batch (Fin count)} {S₁ : Shape} {O₁ : OpeningShape S₁}
  {fold₁ : Descriptor} {S₂ : Shape} {O₂ : OpeningShape S₂} {fold₂ : Descriptor} {S₃ : Shape}
  {O₃ : OpeningShape S₃} {fold₃ : Descriptor} {S₄ : Shape} {O₄ : OpeningShape S₄}
  {fold₄ : Descriptor} {S₅ : Shape} {O₅ : OpeningShape S₅} {fold₅ : Descriptor} {S₆ : Shape}
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq Quadratic}
  {key : (g : Fin count) → Key (B.shape g)} {openKey : B.OpenKey} {mass : Fin count → ℕ}
  {β₁ : Module.Basis (Fin O₁.planes) Zq Quadratic} {key₁ : Key S₁} {openKey₁ : OpeningKey O₁}
  {weight₁ : ℕ} {key₂ : Key S₂} {openKey₂ : OpeningKey O₂} {weight₂ : ℕ} {key₃ : Key S₃}
  {openKey₃ : OpeningKey O₃} {weight₃ : ℕ} {key₄ : Key S₄} {openKey₄ : OpeningKey O₄}
  {weight₄ : ℕ} {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {weight₅ : ℕ}
  (accept : Prefix Wire (Coord → Quadratic) total → Prop)
  (T₀ : Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim Quadratic)
    (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    B.Witness B.Weak Wire Coord Quadratic (Option (List Quadratic × Quadratic)) Table)
  (T₁ : Tier (Fin S₁.blocks) (fun _ => Rq O₁.carrierDegree) (Claim O₁ Quadratic)
    (LevelBreak key₁ openKey₁ (2 * (2 * weight₁) * fold₁.sourceBound S₁)) (Witness O₁ fold₁)
    (Weak O₁) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table)
  (T₂ : Tier (Fin S₂.blocks) (fun _ => Rq S₂.innerDegree) (TraceClaim O₂ Quadratic)
    (LevelBreak key₂ openKey₂ (2 * (2 * weight₂) * fold₂.sourceBound S₂)) (Witness O₂ fold₂)
    (Weak O₂) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table)
  (T₃ : Tier (Fin S₃.blocks) (fun _ => Rq S₃.innerDegree) (TraceClaim O₃ Quadratic)
    (LevelBreak key₃ openKey₃ (2 * (2 * weight₃) * fold₃.sourceBound S₃)) (Witness O₃ fold₃)
    (Weak O₃) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table)
  (T₄ : Tier (Fin S₄.blocks) (fun _ => Rq S₄.innerDegree) (TraceClaim O₄ Quadratic)
    (LevelBreak key₄ openKey₄ (2 * (2 * weight₄) * fold₄.sourceBound S₄)) (Witness O₄ fold₄)
    (Weak O₄) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table)
  (T₅ : Tier (Fin S₅.blocks) (fun _ => Rq S₅.innerDegree) (TraceClaim O₅ Quadratic)
    (LevelBreak key₅ openKey₅ (2 * (2 * weight₅) * fold₅.sourceBound S₅)) (Witness O₅ fold₅)
    (Weak O₅) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table)
  (L : Terminal S₆ Quadratic Wire Coord Quadratic total) (m₂ m₃ m₄ m₅ m₆ start : ℕ)
  (read : Reader Wire) (commit : Wire → (g : Fin count) → (B.shape g).Payload)
  (value : Wire → Fin count → Quadratic) (place : Wire → Fin count → List Quadratic)

/-- The checkpoint structure of the trace level 5 over the terminal. Its successor source is the
table of `Z_q` of the packed polynomial of the source blocks of a certificate of the terminal. -/
noncomputable def tower₅ :=
  T₅.over accept (L.body accept)
    (fun inner => flatten (unpackTable S₆.innerDegree m₆
      (S₆.decode (Terminal.blocksOf ((L.body accept).witness inner)))))
    (fun inner other => L.collide ((L.body accept).witness inner) ((L.body accept).witness other))

/-- The checkpoint structure of the trace level 4 over the level 5. -/
noncomputable def tower₄ :=
  T₄.over accept (tower₅ accept T₅ L m₆)
    (fun inner => flatten (unpackTable S₅.innerDegree m₅
      (tableOf ((tower₅ accept T₅ L m₆).witness inner))))
    (fun inner other => Sum.inl (Tier.collide key₅ openKey₅
      (2 * (2 * weight₅) * fold₅.sourceBound S₅) ((tower₅ accept T₅ L m₆).witness inner)
      ((tower₅ accept T₅ L m₆).witness other)))

/-- The checkpoint structure of the trace level 3 over the level 4. -/
noncomputable def tower₃ :=
  T₃.over accept (tower₄ accept T₄ T₅ L m₅ m₆)
    (fun inner => flatten (unpackTable S₄.innerDegree m₄
      (tableOf ((tower₄ accept T₄ T₅ L m₅ m₆).witness inner))))
    (fun inner other => Sum.inl (Tier.collide key₄ openKey₄
      (2 * (2 * weight₄) * fold₄.sourceBound S₄) ((tower₄ accept T₄ T₅ L m₅ m₆).witness inner)
      ((tower₄ accept T₄ T₅ L m₅ m₆).witness other)))

/-- The checkpoint structure of the trace level 2 over the level 3. -/
noncomputable def tower₂ :=
  T₂.over accept (tower₃ accept T₃ T₄ T₅ L m₄ m₅ m₆)
    (fun inner => flatten (unpackTable S₃.innerDegree m₃
      (tableOf ((tower₃ accept T₃ T₄ T₅ L m₄ m₅ m₆).witness inner))))
    (fun inner other => Sum.inl (Tier.collide key₃ openKey₃
      (2 * (2 * weight₃) * fold₃.sourceBound S₃)
      ((tower₃ accept T₃ T₄ T₅ L m₄ m₅ m₆).witness inner)
      ((tower₃ accept T₃ T₄ T₅ L m₄ m₅ m₆).witness other)))

/-- The checkpoint structure of the packing level 1 over the trace level 2. -/
noncomputable def tower₁ :=
  T₁.over accept (tower₂ accept T₂ T₃ T₄ T₅ L m₃ m₄ m₅ m₆)
    (fun inner => flatten (unpackTable S₂.innerDegree m₂
      (tableOf ((tower₂ accept T₂ T₃ T₄ T₅ L m₃ m₄ m₅ m₆).witness inner))))
    (fun inner other => Sum.inl (Tier.collide key₂ openKey₂
      (2 * (2 * weight₂) * fold₂.sourceBound S₂)
      ((tower₂ accept T₂ T₃ T₄ T₅ L m₃ m₄ m₅ m₆).witness inner)
      ((tower₂ accept T₂ T₃ T₄ T₅ L m₃ m₄ m₅ m₆).witness other)))

/-- The checkpoint structure of the fold with opening groups over the packing level 1, which
commits to its flat witness. -/
noncomputable def tower₀ :=
  T₀.over accept (tower₁ accept T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆)
    (fun inner => tableOf ((tower₁ accept T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆).witness inner))
    (fun inner other => Sum.inl (Tier.collide key₁ openKey₁
      (2 * (2 * weight₁) * fold₁.sourceBound S₁)
      ((tower₁ accept T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆).witness inner)
      ((tower₁ accept T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆).witness other)))

/-- The checkpoint structure of the opening: the batching root at the stage `start` over the fold
with opening groups, the packing level, the four trace levels and the terminal. A witness of
`R_bind` is a solution of Module-SIS for a matrix of a group, of a level, or for `A_term`. -/
noncomputable def towerRoot :=
  (claimsRoot (Coord := Coord) (Slot := Quadratic) (total := total) start read value
      place).segment.over accept (tower₀ accept T₀ T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆)
    (fun inner =>
      Batch.tablesOf ((tower₀ accept T₀ T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆).witness inner))
    (fun inner other => Sum.inl (Batch.collide key openKey
      (fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      ((tower₀ accept T₀ T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆).witness inner)
      ((tower₀ accept T₀ T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆).witness other)))
    (OpensAt start commit value place key mass)

variable {accept T₀ T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆ start read commit value place}

/-- The batching root over a fold with opening groups, a packing level, four trace levels and
the terminal is a suffix from the stage `start` (Akita Proposition 10.44 and Theorem 10.46,
"Batching root"). The stage errors are `1/|E|` for the weights of the root, `Σ_b p_b + Σ_j ε_j`
for each level and `B_term·p_term` for the terminal. `valid`, `packed`, `traced`, `uniform` and
`passes` are the conditions of the step lemmas on each level. `rooted`, `linked₀` to `linked₅`
and `passed`: the verifier accepts only with the link of the root, the link from each level to
the next level or to the terminal, and the checks of the terminal. The hypotheses on the terminal
are those of `Terminal.suffix_linked`. -/
theorem tower_suffix {blocks peak : ℕ} (card : Fintype.card B.Block = blocks)
    (valid₀ : T₀.Valid) (valid₁ : T₁.Valid) (valid₂ : T₂.Valid) (valid₃ : T₃.Valid)
    (valid₄ : T₄.Valid) (valid₅ : T₅.Valid)
    (fair : ∃ decode : (Coord → Quadratic) → Fin count → Quadratic, Fair decode)
    (packed₀ : T₀.method = Batch.method β key openKey mass)
    (packed₁ : T₁.method = packing β₁ key₁ openKey₁ weight₁)
    (traced₂ : T₂.method = trace key₂ openKey₂ weight₂)
    (traced₃ : T₃.method = trace key₃ openKey₃ weight₃)
    (traced₄ : T₄.method = trace key₄ openKey₄ weight₄)
    (traced₅ : T₅.method = trace key₅ openKey₅ weight₅)
    (first : T₀.start = start + 1) (next₁ : T₁.start = T₀.finish) (next₂ : T₂.start = T₁.finish)
    (next₃ : T₃.start = T₂.finish) (next₄ : T₄.start = T₃.finish) (next₅ : T₅.start = T₄.finish)
    (next₆ : L.start = T₅.finish) (fits : L.start < total)
    (quarter₂ : 4 * m₂ = S₂.innerDegree) (quarter₃ : 4 * m₃ = S₃.innerDegree)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree)
    (uniform₀ : ∀ wire challenge table other,
      bare (T₀.check wire challenge table) = bare (T₀.check wire challenge other))
    (passes₀ : ∀ (table : Table) r2 v, v = mle r2 (lifted Quadratic table) →
      T₀.passes table (some (r2, v)))
    (uniform₁ : ∀ wire challenge table other,
      bare (T₁.check wire challenge table) = bare (T₁.check wire challenge other))
    (passes₁ : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T₁.passes table (some (ρ, θ, v)))
    (uniform₂ : ∀ wire challenge table other,
      bare (T₂.check wire challenge table) = bare (T₂.check wire challenge other))
    (passes₂ : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T₂.passes table (some (ρ, θ, v)))
    (uniform₃ : ∀ wire challenge table other,
      bare (T₃.check wire challenge table) = bare (T₃.check wire challenge other))
    (passes₃ : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T₃.passes table (some (ρ, θ, v)))
    (uniform₄ : ∀ wire challenge table other,
      bare (T₄.check wire challenge table) = bare (T₄.check wire challenge other))
    (passes₄ : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T₄.passes table (some (ρ, θ, v)))
    (uniform₅ : ∀ wire challenge table other,
      bare (T₅.check wire challenge table) = bare (T₅.check wire challenge other))
    (passes₅ : ∀ (table : Table) ρ θ v, v = θ * mle ρ (packed table) →
      T₅.passes table (some (ρ, θ, v)))
    (injective : Function.Injective L.block) (units : UnitDifferences L.family)
    (sparse : ∀ challenge ∈ L.family, Sparse L.weight peak challenge)
    (commit₁ : Wire → S₁.Payload) (commit₂ : Wire → S₂.Payload) (commit₃ : Wire → S₃.Payload)
    (commit₄ : Wire → S₄.Payload) (commit₅ : Wire → S₅.Payload)
    (stateOf : Wire → Fin S₆.blocks → Fin S₆.innerRank → Rq S₆.innerDegree)
    (rooted : ∀ point, accept point → start < point.rest.length ∧
      RootLinked T₀ start commit value place
        ((claimsRoot (Coord := Coord) (Slot := Quadratic) (total := total) start read value
          place).weights point) point)
    (linked₀ : ∀ point, accept point → T₀.Linked T₁ commit₁ point)
    (linked₁ : ∀ point, accept point → T₁.TraceLinked T₂ quarter₂ commit₂ point)
    (linked₂ : ∀ point, accept point → T₂.TraceLinked T₃ quarter₃ commit₃ point)
    (linked₃ : ∀ point, accept point → T₃.TraceLinked T₄ quarter₄ commit₄ point)
    (linked₄ : ∀ point, accept point → T₄.TraceLinked T₅ quarter₅ commit₅ point)
    (linked₅ : ∀ point, accept point → Terminal.Linked T₅ L quarter₆ stateOf point)
    (passed : ∀ point, accept point → L.Passed point) :
    Tier.Suffix accept start (OpensAt start commit value place key mass)
      (towerRoot accept T₀ T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆ start read commit value place)
      (1 / Fintype.card Quadratic + (∑ b, T₀.chance b + T₀.costs.sum +
        (∑ b, T₁.chance b + T₁.costs.sum + (∑ b, T₂.chance b + T₂.costs.sum +
          (∑ b, T₃.chance b + T₃.costs.sum + (∑ b, T₄.chance b + T₄.costs.sum +
            (∑ b, T₅.chance b + T₅.costs.sum + S₆.blocks * slotChance L.decode)))))))
      (1 + (blocks + 1 + (S₁.blocks + 1 + (S₂.blocks + 1 + (S₃.blocks + 1 + (S₄.blocks + 1 +
        (S₅.blocks + 1 + S₆.blocks)))))))
      (2 * (2 * (blocks + 1) * (2 * (S₁.blocks + 1) * (2 * (S₂.blocks + 1) *
        (2 * (S₃.blocks + 1) * (2 * (S₄.blocks + 1) *
          (2 * (S₅.blocks + 1) * (S₆.blocks + 1)))))))) := by
  have ends₀ := T₀.finish_eq
  have ends₁ := T₁.finish_eq
  have ends₂ := T₂.finish_eq
  have ends₃ := T₃.finish_eq
  have ends₄ := T₄.finish_eq
  have ends₅ := T₅.finish_eq
  have level₅ := Terminal.suffix_linked valid₅ ⟨injective, fits, passed⟩ next₆ units sparse
    quarter₆ stateOf uniform₅ passes₅ linked₅
  have level₄ := Tier.suffix_push_trace valid₄ traced₅ next₅ quarter₅ commit₅ (by omega)
    uniform₄ passes₄ linked₄ Sum.inl
    (fun solution found => by
      show T₅.method.broken solution
      rw [traced₅]
      exact found)
    level₅
  have level₃ := Tier.suffix_push_trace valid₃ traced₄ next₄ quarter₄ commit₄ (by omega)
    uniform₃ passes₃ linked₃ Sum.inl
    (fun solution found => by
      show T₄.method.broken solution
      rw [traced₄]
      exact found)
    level₄
  have level₂ := Tier.suffix_push_trace valid₂ traced₃ next₃ quarter₃ commit₃ (by omega)
    uniform₂ passes₂ linked₂ Sum.inl
    (fun solution found => by
      show T₃.method.broken solution
      rw [traced₃]
      exact found)
    level₃
  have level₁ := Tier.suffix_push_trace valid₁ traced₂ next₂ quarter₂ commit₂ (by omega)
    uniform₁ passes₁ linked₁ Sum.inl
    (fun solution found => by
      show T₂.method.broken solution
      rw [traced₂]
      exact found)
    level₂
  have level₀ := Tier.suffix_push card valid₀ packed₁ next₁ commit₁ (by omega) uniform₀ passes₀
    linked₀ Sum.inl
    (fun solution found => by
      show T₁.method.broken solution
      rw [packed₁]
      exact found)
    level₁
  exact root_suffix start read value place fair packed₀ first (by omega) commit rooted Sum.inl
    (fun solution found => by
      show T₀.method.broken solution
      rw [packed₀]
      exact found)
    level₀

end Tower

section Sized

variable {E Coord Wire : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  [Fintype Coord] [DecidableEq Coord]

/-- The conditions on the parameters of a level with one group. `injective`: the fold
coordinates have different coordinates of the answer. `units` and `sparse`: the challenge family
has units as differences (Akita Definition 9.2, E1) and members with `‖c‖_1 = ω`. `base`:
`b_z ≥ 2`. `small`: `b⋆ < q`. `varsFit` and `batchFit`: an answer of the oracle has `μ'` slots or
more and `s_rel` slots or more. -/
structure LevelSized {d blocks : ℕ} [NeZero d] (block : Fin blocks → Coord)
    (decode : E → Option (Rq d)) (weight responseBase certifiedBase μ batch : ℕ) : Prop where
  injective : Function.Injective block
  units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge}
  sparse : ∃ peak, ∀ challenge, (∃ slot, decode slot = some challenge) →
    Sparse weight peak challenge
  base : 2 ≤ responseBase
  small : certifiedBase < fieldModulus
  varsFit : μ ≤ Fintype.card Coord
  batchFit : batch ≤ Fintype.card Coord

variable [Nonempty Coord] {S : Shape} {O : OpeningShape S} {fold : Descriptor} {μ : ℕ}
  {fits : O.cells fold < 2 ^ μ} {batch : ℕ} {read : Reader Wire} {weight start ν : ℕ}

/-- A packing level on the quotient-lift route with the tensor reduction satisfies the conditions
of `Tier.Valid`. -/
theorem LevelSized.lift {β : Module.Basis (Fin O.planes) Zq Quadratic} {key : Key S}
    {openKey : OpeningKey O} {claim : Wire → Claim O Quadratic} {digits : ℕ}
    {room : O.quotientRoom fold digits ≤ 2 ^ μ} {rowsFit : O.rowCount + 1 + 1 ≤ 2 ^ batch}
    {block : Fin S.blocks → Coord} {decode : Quadratic → Option (Rq O.carrierDegree)}
    (sized : LevelSized block decode weight fold.responseBase S.certifiedBase μ batch) :
    (Tier.tensored (liftFold β key openKey claim μ fits digits room batch rowsFit block decode
      read weight start) ν).Valid := by
  obtain ⟨peak, sparse⟩ := sized.sparse
  exact Tier.tensored_valid
    (liftFold_valid sized.injective sized.units sparse sized.base sized.small sized.varsFit
      sized.batchFit)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)
    fun _ _ table rest tail rejected ended =>
      foldCheck_stagedTo _ _ _ table sized.varsFit sized.batchFit rest tail rejected ended

/-- A trace level with the tensor reduction satisfies the conditions of `Tier.Valid`. -/
theorem LevelSized.trace {key : Key S} {openKey : OpeningKey O}
    {claim : Wire → TraceClaim O Quadratic} {rowsFit : O.traceRowCount + 1 + 1 ≤ 2 ^ batch}
    {block : Fin S.blocks → Coord} {decode : Quadratic → Option (Rq S.innerDegree)}
    (sized : LevelSized block decode weight fold.responseBase S.certifiedBase μ batch) :
    (Tier.tensored (committedTrace (fold := fold) key openKey claim μ fits batch rowsFit block
      decode read weight start) ν).Valid := by
  obtain ⟨peak, sparse⟩ := sized.sparse
  exact Tier.tensored_valid
    (committedTrace_valid sized.injective sized.units sparse sized.base sized.small sized.varsFit
      sized.batchFit)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)
    fun _ _ table rest tail rejected ended =>
      foldCheck_stagedTo _ _ _ table sized.varsFit sized.batchFit rest tail rejected ended

end Sized

section Lift

variable {count : ℕ} {B : Batch (Fin count)} {S₁ : Shape} {O₁ : OpeningShape S₁}
  {fold₁ : Descriptor} {S₂ : Shape} {O₂ : OpeningShape S₂} {fold₂ : Descriptor} {S₃ : Shape}
  {O₃ : OpeningShape S₃} {fold₃ : Descriptor} {S₄ : Shape} {O₄ : OpeningShape S₄}
  {fold₄ : Descriptor} {S₅ : Shape} {O₅ : OpeningShape S₅} {fold₅ : Descriptor} {S₆ : Shape}
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {total : ℕ}
  {accept : Prefix Wire (Coord → Quadratic) total → Prop} {read : Reader Wire} {start : ℕ}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq Quadratic}
  {key : (g : Fin count) → Key (B.shape g)} {openKey : B.OpenKey}
  {claim : Wire → B.Claim Quadratic} {μ : ℕ} {fits : B.cells < 2 ^ μ} {base digits : ℕ}
  {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ} {rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch}
  {block : B.Block → Coord} {decode : (g : Fin count) → Quadratic → Option (Rq (B.carrierDegree g))}
  {mass : Fin count → ℕ}
  {β₁ : Module.Basis (Fin O₁.planes) Zq Quadratic} {key₁ : Key S₁} {openKey₁ : OpeningKey O₁}
  {claim₁ : Wire → Claim O₁ Quadratic} {μ₁ : ℕ} {fits₁ : O₁.cells fold₁ < 2 ^ μ₁} {digits₁ : ℕ}
  {room₁ : O₁.quotientRoom fold₁ digits₁ ≤ 2 ^ μ₁} {batch₁ : ℕ}
  {rowsFit₁ : O₁.rowCount + 1 + 1 ≤ 2 ^ batch₁} {block₁ : Fin S₁.blocks → Coord}
  {decode₁ : Quadratic → Option (Rq O₁.carrierDegree)} {weight₁ ν₁ : ℕ}
  {key₂ : Key S₂} {openKey₂ : OpeningKey O₂} {claim₂ : Wire → TraceClaim O₂ Quadratic}
  {μ₂ : ℕ} {fits₂ : O₂.cells fold₂ < 2 ^ μ₂} {batch₂ : ℕ}
  {rowsFit₂ : O₂.traceRowCount + 1 + 1 ≤ 2 ^ batch₂} {block₂ : Fin S₂.blocks → Coord}
  {decode₂ : Quadratic → Option (Rq S₂.innerDegree)} {weight₂ ν₂ m₂ : ℕ}
  {key₃ : Key S₃} {openKey₃ : OpeningKey O₃} {claim₃ : Wire → TraceClaim O₃ Quadratic}
  {μ₃ : ℕ} {fits₃ : O₃.cells fold₃ < 2 ^ μ₃} {batch₃ : ℕ}
  {rowsFit₃ : O₃.traceRowCount + 1 + 1 ≤ 2 ^ batch₃} {block₃ : Fin S₃.blocks → Coord}
  {decode₃ : Quadratic → Option (Rq S₃.innerDegree)} {weight₃ ν₃ m₃ : ℕ}
  {key₄ : Key S₄} {openKey₄ : OpeningKey O₄} {claim₄ : Wire → TraceClaim O₄ Quadratic}
  {μ₄ : ℕ} {fits₄ : O₄.cells fold₄ < 2 ^ μ₄} {batch₄ : ℕ}
  {rowsFit₄ : O₄.traceRowCount + 1 + 1 ≤ 2 ^ batch₄} {block₄ : Fin S₄.blocks → Coord}
  {decode₄ : Quadratic → Option (Rq S₄.innerDegree)} {weight₄ ν₄ m₄ : ℕ}
  {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {claim₅ : Wire → TraceClaim O₅ Quadratic}
  {μ₅ : ℕ} {fits₅ : O₅.cells fold₅ < 2 ^ μ₅} {batch₅ : ℕ}
  {rowsFit₅ : O₅.traceRowCount + 1 + 1 ≤ 2 ^ batch₅} {block₅ : Fin S₅.blocks → Coord}
  {decode₅ : Quadratic → Option (Rq S₅.innerDegree)} {weight₅ ν₅ m₅ : ℕ}
  {L : Terminal S₆ Quadratic Wire Coord Quadratic total} {m₆ : ℕ}
  {commit : Wire → (g : Fin count) → (B.shape g).Payload} {value : Wire → Fin count → Quadratic}
  {place : Wire → Fin count → List Quadratic}

local notation "zeroth" =>
  liftBatch β key openKey claim μ fits base digits room batch rowsFit block decode read mass
    (start + 1)

local notation "first" =>
  Tier.tensored (liftFold β₁ key₁ openKey₁ claim₁ μ₁ fits₁ digits₁ room₁ batch₁ rowsFit₁ block₁
    decode₁ read weight₁ (Tier.finish zeroth)) ν₁

local notation "second" =>
  Tier.tensored (committedTrace (fold := fold₂) key₂ openKey₂ claim₂ μ₂ fits₂ batch₂ rowsFit₂
    block₂ decode₂ read weight₂ (Tier.finish first)) ν₂

local notation "third" =>
  Tier.tensored (committedTrace (fold := fold₃) key₃ openKey₃ claim₃ μ₃ fits₃ batch₃ rowsFit₃
    block₃ decode₃ read weight₃ (Tier.finish second)) ν₃

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := fold₄) key₄ openKey₄ claim₄ μ₄ fits₄ batch₄ rowsFit₄
    block₄ decode₄ read weight₄ (Tier.finish third)) ν₄

local notation "fifth" =>
  Tier.tensored (committedTrace (fold := fold₅) key₅ openKey₅ claim₅ μ₅ fits₅ batch₅ rowsFit₅
    block₅ decode₅ read weight₅ (Tier.finish fourth)) ν₅

/-- The batching root over `liftBatch`, `liftFold` with the tensor reduction, four levels
`committedTrace` with the tensor reduction and the terminal is a suffix from the stage `start`:
`tower_suffix` for these levels. The first group of hypotheses is that of `liftBatch_valid`, with
`groupsFit` for the weights of the root. `sized₁` to `sized₅`: the conditions `LevelSized` of the
levels 1 to 5. `next` and `inside`: the fold challenge of the terminal is the stage after the
stages of the level 5, and it is a stage of the transcript. The other hypotheses are those of
`tower_suffix`. -/
theorem liftTower_suffix {peak : Fin count → ℕ} {peakL : ℕ}
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (groupsFit : count ≤ Fintype.card Coord)
    (sized₁ : LevelSized block₁ decode₁ weight₁ fold₁.responseBase S₁.certifiedBase μ₁ batch₁)
    (sized₂ : LevelSized block₂ decode₂ weight₂ fold₂.responseBase S₂.certifiedBase μ₂ batch₂)
    (sized₃ : LevelSized block₃ decode₃ weight₃ fold₃.responseBase S₃.certifiedBase μ₃ batch₃)
    (sized₄ : LevelSized block₄ decode₄ weight₄ fold₄.responseBase S₄.certifiedBase μ₄ batch₄)
    (sized₅ : LevelSized block₅ decode₅ weight₅ fold₅.responseBase S₅.certifiedBase μ₅ batch₅)
    (next : L.start = Tier.finish fifth) (inside : L.start < total)
    (quarter₂ : 4 * m₂ = S₂.innerDegree) (quarter₃ : 4 * m₃ = S₃.innerDegree)
    (quarter₄ : 4 * m₄ = S₄.innerDegree) (quarter₅ : 4 * m₅ = S₅.innerDegree)
    (quarter₆ : 4 * m₆ = S₆.innerDegree)
    (injectiveL : Function.Injective L.block) (unitsL : UnitDifferences L.family)
    (sparseL : ∀ challenge ∈ L.family, Sparse L.weight peakL challenge)
    (commit₁ : Wire → S₁.Payload) (commit₂ : Wire → S₂.Payload) (commit₃ : Wire → S₃.Payload)
    (commit₄ : Wire → S₄.Payload) (commit₅ : Wire → S₅.Payload)
    (stateOf : Wire → Fin S₆.blocks → Fin S₆.innerRank → Rq S₆.innerDegree)
    (rooted : ∀ point, accept point → start < point.rest.length ∧
      RootLinked zeroth start commit value place
        ((claimsRoot (Coord := Coord) (Slot := Quadratic) (total := total) start read value
          place).weights point) point)
    (linked₀ : ∀ point, accept point → Tier.Linked zeroth first commit₁ point)
    (linked₁ : ∀ point, accept point → Tier.TraceLinked first second quarter₂ commit₂ point)
    (linked₂ : ∀ point, accept point → Tier.TraceLinked second third quarter₃ commit₃ point)
    (linked₃ : ∀ point, accept point → Tier.TraceLinked third fourth quarter₄ commit₄ point)
    (linked₄ : ∀ point, accept point → Tier.TraceLinked fourth fifth quarter₅ commit₅ point)
    (linked₅ : ∀ point, accept point → Terminal.Linked fifth L quarter₆ stateOf point)
    (passed : ∀ point, accept point → L.Passed point) :
    Tier.Suffix accept start (OpensAt start commit value place key mass)
      (towerRoot accept zeroth first second third fourth fifth L m₂ m₃ m₄ m₅ m₆ start read commit
        value place)
      (1 / Fintype.card Quadratic + (∑ b, Tier.chance zeroth b + (Tier.costs zeroth).sum +
        (∑ b, Tier.chance first b + (Tier.costs first).sum +
          (∑ b, Tier.chance second b + (Tier.costs second).sum +
            (∑ b, Tier.chance third b + (Tier.costs third).sum +
              (∑ b, Tier.chance fourth b + (Tier.costs fourth).sum +
                (∑ b, Tier.chance fifth b + (Tier.costs fifth).sum +
                  S₆.blocks * slotChance L.decode)))))))
      (1 + (∑ g, (B.shape g).blocks + 1 + (S₁.blocks + 1 + (S₂.blocks + 1 + (S₃.blocks + 1 +
        (S₄.blocks + 1 + (S₅.blocks + 1 + S₆.blocks)))))))
      (2 * (2 * (∑ g, (B.shape g).blocks + 1) * (2 * (S₁.blocks + 1) * (2 * (S₂.blocks + 1) *
        (2 * (S₃.blocks + 1) * (2 * (S₄.blocks + 1) *
          (2 * (S₅.blocks + 1) * (S₆.blocks + 1)))))))) :=
  tower_suffix B.card_block
    (liftBatch_valid injective units sparse response uniform small varsFit batchFit) sized₁.lift
    sized₂.trace sized₃.trace sized₄.trace sized₅.trace (exists_fair_slots groupsFit) rfl rfl rfl
    rfl rfl rfl rfl rfl rfl rfl rfl rfl next inside quarter₂ quarter₃ quarter₄ quarter₅ quarter₆
    (fun _ _ table other => foldCheck_bare _ _ _ table other)
    (fun _ r2 v opened => ⟨r2, v, rfl, opened⟩)
    (Tier.tensored_uniform fun _ _ table other => foldCheck_bare _ _ _ table other)
    (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    (Tier.tensored_uniform committedTrace_uniform) (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    (Tier.tensored_uniform committedTrace_uniform) (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    (Tier.tensored_uniform committedTrace_uniform) (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    (Tier.tensored_uniform committedTrace_uniform) (fun _ ρ θ v opened => ⟨ρ, θ, v, rfl, opened⟩)
    injectiveL unitsL sparseL commit₁ commit₂ commit₃ commit₄ commit₅ stateOf rooted linked₀
    linked₁ linked₂ linked₃ linked₄ linked₅ passed

end Lift

section Composed

variable {B : Batch (Fin 3)} {S₁ : Shape} {O₁ : OpeningShape S₁}
  {fold₁ : Descriptor} {S₂ : Shape} {O₂ : OpeningShape S₂} {fold₂ : Descriptor} {S₃ : Shape}
  {O₃ : OpeningShape S₃} {fold₃ : Descriptor} {S₄ : Shape} {O₄ : OpeningShape S₄}
  {fold₄ : Descriptor} {S₅ : Shape} {O₅ : OpeningShape S₅} {fold₅ : Descriptor} {S₆ : Shape}
  {Wire Coord Lead : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {total : ℕ}
  {P : Verifier.Params Zq Quadratic} {C : Compose P B Wire Coord Lead total}
  {vf : Verifier.VerifierData P}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq Quadratic}
  {key : (g : Fin 3) → Key (B.shape g)} {openKey : B.OpenKey} {mass : Fin 3 → ℕ}
  {β₁ : Module.Basis (Fin O₁.planes) Zq Quadratic} {key₁ : Key S₁} {openKey₁ : OpeningKey O₁}
  {weight₁ : ℕ} {key₂ : Key S₂} {openKey₂ : OpeningKey O₂} {weight₂ : ℕ} {key₃ : Key S₃}
  {openKey₃ : OpeningKey O₃} {weight₃ : ℕ} {key₄ : Key S₄} {openKey₄ : OpeningKey O₄}
  {weight₄ : ℕ} {key₅ : Key S₅} {openKey₅ : OpeningKey O₅} {weight₅ : ℕ}
  {accept : Prefix Wire (Coord → Quadratic) total → Prop}
  {T₀ : Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim Quadratic)
    (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
    B.Witness B.Weak Wire Coord Quadratic (Option (List Quadratic × Quadratic)) Table}
  {T₁ : Tier (Fin S₁.blocks) (fun _ => Rq O₁.carrierDegree) (Claim O₁ Quadratic)
    (LevelBreak key₁ openKey₁ (2 * (2 * weight₁) * fold₁.sourceBound S₁)) (Witness O₁ fold₁)
    (Weak O₁) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table}
  {T₂ : Tier (Fin S₂.blocks) (fun _ => Rq S₂.innerDegree) (TraceClaim O₂ Quadratic)
    (LevelBreak key₂ openKey₂ (2 * (2 * weight₂) * fold₂.sourceBound S₂)) (Witness O₂ fold₂)
    (Weak O₂) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table}
  {T₃ : Tier (Fin S₃.blocks) (fun _ => Rq S₃.innerDegree) (TraceClaim O₃ Quadratic)
    (LevelBreak key₃ openKey₃ (2 * (2 * weight₃) * fold₃.sourceBound S₃)) (Witness O₃ fold₃)
    (Weak O₃) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table}
  {T₄ : Tier (Fin S₄.blocks) (fun _ => Rq S₄.innerDegree) (TraceClaim O₄ Quadratic)
    (LevelBreak key₄ openKey₄ (2 * (2 * weight₄) * fold₄.sourceBound S₄)) (Witness O₄ fold₄)
    (Weak O₄) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table}
  {T₅ : Tier (Fin S₅.blocks) (fun _ => Rq S₅.innerDegree) (TraceClaim O₅ Quadratic)
    (LevelBreak key₅ openKey₅ (2 * (2 * weight₅) * fold₅.sourceBound S₅)) (Witness O₅ fold₅)
    (Weak O₅) Wire Coord Quadratic (Option (List Quadratic × Quadratic × Quadratic)) Table}
  {L : Terminal S₆ Quadratic Wire Coord Quadratic total} {m₂ m₃ m₄ m₅ m₆ : ℕ}
  {commit : Wire → (g : Fin 3) → (B.shape g).Payload} {value : Wire → Fin 3 → Quadratic}
  {place : Wire → Fin 3 → List Quadratic} {error : ℝ} {loops tree : ℕ}


local notation "opening" =>
  towerRoot accept T₀ T₁ T₂ T₃ T₄ T₅ L m₂ m₃ m₄ m₅ m₆ (Compose.root C vf) (Compose.read C)
    commit value place

/-- The compression argument over the opening `towerRoot` with three groups is a suffix from the
stage of the lookup challenges: `Compose.suffix` for a suffix `later` of `towerRoot` at the stage
`Compose.root`, which `tower_suffix` gives. A solution of Module-SIS for a matrix of a group is a
witness of `R_bind` of `towerRoot`. `sized` and `ran` are the hypotheses of `Compose.suffix`. -/
theorem towerCompose_suffix (sized : C.Sized vf)
    (ran : ∀ point, accept point → C.Ran vf commit value place point)
    (packed₀ : T₀.method = Batch.method β key openKey mass)
    (later : Tier.Suffix accept (C.root vf) (OpensAt (C.root vf) commit value place key mass)
      opening error loops tree) :
    Tier.Suffix accept C.start (C.Extracted key mass)
      (C.chain vf accept opening key openKey mass Sum.inl)
      (C.leadCosts.sum + Verifier.runCost vf + error) (2 + loops) (4 * tree) :=
  Compose.suffix sized ran Sum.inl
    (fun solution found => by
      show T₀.method.broken solution
      rw [packed₀]
      exact found)
    later

end Composed

end NightstreamFPrime.Spec.Compression.Akita.Extract
