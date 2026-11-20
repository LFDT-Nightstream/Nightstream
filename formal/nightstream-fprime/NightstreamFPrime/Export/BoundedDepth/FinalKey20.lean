import NightstreamFPrime.Export.BoundedDepth.FinalAkita
import NightstreamFPrime.Spec.Compression.Akita.Key20Tower

/-! The final verifier over the Akita opening of the schedule file
`crates/ns-compress/artifacts/key20.aks` (SPEC.md, Algorithm 10; Akita, ePrint 2026/1983, revision
of 2026-10-15, Theorem 10.17, Proposition 10.44 and Theorem 10.46).

`sound` is `FinalAkita.mixed_sound` for the numbers of `Spec/Compression/Akita/Key20.lean`, with
the bound `Key20.towerError_le` of the stage errors of the opening and its 142 retry loops.

`sound` holds for each `setup`. `key20_points` gives the numbers of the three opening claims for
a setup with the key polynomial of 512 blocks, which is the key of `key20.aks`: the point of the
key polynomial has 20 coordinates, the point of `W` has 16 coordinates, and the point of `V` has
`spanVars + 2 + setup.zeros` coordinates with `spanVars + 2 ≤ 16`. The link `RootLinked` of the
root needs 20, 16 and 16 coordinates or more (`Key20.root_points`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalKey20

open scoped BigOperators
open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier (Mode Claims Premises)
open NightstreamFPrime.Export.BoundedDepth.FinalAkita
open Key20 (root level₁ level₂ level₃ level₄ level₅ norm₃ norm₄ norm₅ normTerminal rootVars
  rootBatch quotientDigits root_sized root_room level₁_sized sized₂ sized₃ sized₄ sized₅ quarters)

attribute [local instance low] Classical.propDecidable

noncomputable local instance : Algebra Zq K := inferInstanceAs (Algebra F K)

noncomputable local instance : Degree2 K :=
  ⟨AlgEquiv.ofRingEquiv (f := carrierEquiv.ringEquiv) fun _ => rfl⟩

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode} (setup : FinalVerifier.Setup mode logicalWidth publicFits)
  (packageWidths : Fin 2 → ℕ) (step : AppState → AppWitness → AppState) (cap : ℕ)
  (control : AppState → AppWitness → Fin 2) (shape : Claims logicalWidth publicFits)
  {Wire Coord : Type} {total : ℕ} (read : Reader Wire)
  (head : Wire → First logicalWidth publicFits)
  (bound : Prefix Wire (Coord → K) total → (g : Fin 3) → (root.shape g).Payload)
  [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]
  {β : ∀ g, Module.Basis (Fin (root.planes g)) Zq K} {key : (g : Fin 3) → Key (root.shape g)}
  {openKey : root.OpenKey} {claim : Wire → root.Claim K} {block : root.Block → Coord}
  {decode : (g : Fin 3) → K → Option (Rq (root.carrierDegree g))} {mass : Fin 3 → ℕ}
  {β₁ : Module.Basis (Fin level₁.opening.planes) Zq K} {key₁ : Key level₁.shape}
  {openKey₁ : OpeningKey level₁.opening} {claim₁ : Wire → Claim level₁.opening K}
  {block₁ : Fin level₁.shape.blocks → Coord}
  {decode₁ : K → Option (Rq level₁.opening.carrierDegree)} {weight₁ : ℕ}
  {key₂ : Key level₂.shape} {openKey₂ : OpeningKey level₂.opening}
  {claim₂ : Wire → TraceClaim level₂.opening K} {block₂ : Fin level₂.shape.blocks → Coord}
  {decode₂ : K → Option (Rq level₂.shape.innerDegree)} {weight₂ : ℕ}
  {key₃ : Key level₃.shape} {openKey₃ : OpeningKey level₃.opening}
  {claim₃ : Wire → TraceClaim level₃.opening K} {block₃ : Fin level₃.shape.blocks → Coord}
  {decode₃ : K → Option (Rq level₃.shape.innerDegree)} {weight₃ : ℕ}
  {key₄ : Key level₄.shape} {openKey₄ : OpeningKey level₄.opening}
  {claim₄ : Wire → TraceClaim level₄.opening K} {block₄ : Fin level₄.shape.blocks → Coord}
  {decode₄ : K → Option (Rq level₄.shape.innerDegree)} {weight₄ : ℕ}
  {key₅ : Key level₅.shape} {openKey₅ : OpeningKey level₅.opening}
  {claim₅ : Wire → TraceClaim level₅.opening K} {block₅ : Fin level₅.shape.blocks → Coord}
  {decode₅ : K → Option (Rq level₅.shape.innerDegree)} {weight₅ : ℕ}
  {L : Terminal Key20.terminal K Wire Coord K total
    (.euclidean normTerminal.operator normTerminal.limit)}
  {commit : Wire → (g : Fin 3) → (root.shape g).Payload} {value : Wire → Fin 3 → K}
  {place : Wire → Fin 3 → List K} {commit₁ : Wire → level₁.shape.Payload}
  {commit₂ : Wire → level₂.shape.Payload} {commit₃ : Wire → level₃.shape.Payload}
  {commit₄ : Wire → level₄.shape.Payload} {commit₅ : Wire → level₅.shape.Payload}
  {stateOf : Wire → Fin Key20.terminal.blocks → Fin Key20.terminal.innerRank →
    Rq Key20.terminal.innerDegree}

local notation "C" => compose setup packageWidths step cap control shape read head bound

local notation "vf" => FinalVerifier.Setup.verifier setup shape

local notation "zeroth" =>
  liftBatch β key openKey claim rootVars (And.left root_sized) 8 quotientDigits root_room
    rootBatch (And.left (And.right root_sized)) block decode (Compose.read C) mass
    (Compose.root C vf + 1)

local notation "first" =>
  Tier.tensored (liftFold β₁ key₁ openKey₁ claim₁ level₁.vars (And.left level₁_sized)
    quotientDigits (And.left (And.right level₁_sized)) level₁.batch
    (And.left (And.right (And.right level₁_sized))) block₁ decode₁ (Compose.read C) weight₁
    (Tier.finish zeroth)) level₁.tail

local notation "second" =>
  Tier.tensored (committedTrace (fold := level₂.fold) key₂ openKey₂ claim₂ level₂.vars
    (And.left sized₂) level₂.batch (And.left (And.right sized₂)) block₂ decode₂ (Compose.read C)
    weight₂ (Tier.finish first)) level₂.tail

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := level₃.fold) key₃ openKey₃ claim₃ level₃.vars
    (And.left sized₃) level₃.batch (And.left (And.right sized₃)) norm₃.vars block₃ decode₃
    (Compose.read C) weight₃ norm₃.operator norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) key₄ openKey₄ claim₄ level₄.vars
    (And.left sized₄) level₄.batch (And.left (And.right sized₄)) norm₄.vars block₄ decode₄
    (Compose.read C) weight₄ norm₄.operator norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅ claim₅ level₅.vars
    (And.left sized₅) level₅.batch (And.left (And.right sized₅)) norm₅.vars block₅ decode₅
    (Compose.read C) weight₅ norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail

local notation "accepts" =>
  towerAccepts (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) C vf
    Route.coefficient (Route.euclidean norm₃.operator norm₃.limit)
    (Route.euclidean norm₄.operator norm₄.limit) (Route.euclidean norm₅.operator norm₅.limit)
    zeroth first second third fourth fifth L commit value place (And.left quarters)
    (And.left (And.right quarters)) (And.left (And.right (And.right quarters)))
    (And.left (And.right (And.right (And.right quarters))))
    (And.right (And.right (And.right (And.right quarters)))) commit₁ commit₂ commit₃ commit₄
    commit₅ stateOf

local notation "opening" =>
  towerRoot (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) Route.coefficient
    (Route.euclidean norm₃.operator norm₃.limit) (Route.euclidean norm₄.operator norm₄.limit)
    (Route.euclidean norm₅.operator norm₅.limit) accepts zeroth first second third fourth fifth L
    32 32 16 16 16 (Compose.root C vf) (Compose.read C) commit value place

local notation "whole" => Compose.chain C vf accepts opening key openKey mass Sum.inl

/-- The numbers of the opening claims of the argument for the key polynomial of `key20.aks`
(Rust `Layout::new` with `akita::SCHEDULES`, and SPEC.md Algorithm 10, the Akita claims). For laid
claims and a key polynomial of 512 blocks, the point `(kp, rowPoint)` of the key polynomial has
`6 + 9 + 5 = 20` coordinates, the point `ρ` of `W` has 16 coordinates, and the point of `V` has
`spanVars + 2` coordinates before its `setup.zeros` zeros, with `spanVars + 2 ≤ 16`. These are
the sizes `2^20`, `2^16` and `2^16` of the tables of the groups of `Key20.root`. The point of `V`
has 16 coordinates when `setup.zeros = 16 − 2 − spanVars`, which is the value of Rust
`inverse_point`. -/
theorem key20_points (laid : setup.Laid shape) (blocks : setup.keyBlocks = 512) :
    (setup.params shape).laneVars + (setup.params shape).keyBlockVars +
        (setup.params shape).rowVars = 20 ∧
      (setup.params shape).witnessVars = 16 ∧
      (setup.params shape).spanVars + (setup.verifier shape).columnVars ≤ 16 := by
  obtain ⟨⟨l, ok⟩, -⟩ := laid
  have same : setup.layout shape = l.production := by simp only [FinalVerifier.Setup.layout, ok]
  unfold FinalVerifier.Setup.placed at ok
  rw [blocks] at ok
  obtain ⟨schedule, member, held, -, row, inRow, fits, rfl⟩ := Compression.Rust.layoutNew_ok ok
  have all : ∀ s ∈ Compression.Rust.schedules, s.blocks = 512 →
      s.keyVars = 20 ∧ ∀ r ∈ s.rows, r = (16, 16) := by decide
  obtain ⟨keyVars, rows⟩ := all schedule member held
  obtain rfl := rows row inRow
  simp only [Compression.Rust.rowFits, Bool.and_eq_true, decide_eq_true_eq] at fits
  show 6 + (setup.layout shape).keyBlockVars + 5 = 20 ∧ (setup.layout shape).witnessVars = 16 ∧
    (setup.layout shape).spanVars + 2 ≤ 16
  rw [same]
  show 6 + (schedule.keyVars - 6 - 5) + 5 = 20 ∧ (16 : ℕ) = 16 ∧ _ + 2 ≤ 16
  rw [keyVars]
  exact ⟨rfl, rfl, fits.2⟩

/-- Soundness of the final verifier over the Akita opening of the schedule `key20.aks`
(`FinalAkita.mixed_sound` for the numbers of `Key20.lean`). For each prover with at most `Q`
queries to the oracle, the probability that the verifier `towerAccepts` accepts and no proof with
witnesses satisfies `Verifies` for the statement of the first message is at most
`(Q + 1)·(runCost + 7424/|K| + 135·p) + (Q + 1)·144/budget + breakChance`. `p` is a bound of the
slot probability of each challenge family (`Key20.towerError_le`), and 144 is the count of the
retry loops: 2 for the compression argument and 142 for the opening. `breakChance` is the
probability that the extractor returns a solution of Module-SIS for the key commitment, for a
matrix of a group, of a level, or for `A_term`. `premises`, `laid`, `lookupsFit`,
`challengesFit`, `cube`, `early`, `registered` and `covers` are the hypotheses of
`FinalAkita.mixed_sound` on the final verifier. The other hypotheses are those of
`Key20.tower_suffix` and `Key20.towerError_le`. -/
theorem sound [Finite Wire] {peak : Fin 3 → ℕ} (premises : Premises setup packageWidths)
    (laid : setup.Laid shape)
    (lookupsFit : (setup.params shape).lookups ≤ Fintype.card Coord)
    (challengesFit : 4 + ((setup.params shape).lookups + ((setup.params shape).claims +
      (setup.params shape).claims + (setup.params shape).spanVars)) ≤ Fintype.card Coord)
    (cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord)
    (early : ∀ anchor g, g ≠ 2 → bound anchor g = bound (anchor.take 0) g)
    (slots : 22 ≤ Fintype.card Coord) (injective : Function.Injective block)
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
    (next : Terminal.start L = Tier.finish fifth) (inside : Terminal.start L < total)
    (injectiveL : Function.Injective (Terminal.block L))
    (unitsL : UnitDifferences (Terminal.family L))
    (boundedL : ∀ challenge ∈ Terminal.family L, MulBound normTerminal.operator challenge)
    {p : ℝ} (chance : ∀ g, slotChance (decode g) ≤ p) (chance₁ : slotChance decode₁ ≤ p)
    (chance₂ : slotChance decode₂ ≤ p) (chance₃ : slotChance decode₃ ≤ p)
    (chance₄ : slotChance decode₄ ≤ p) (chance₅ : slotChance decode₅ ≤ p)
    (chanceL : slotChance (Terminal.decode L) ≤ p)
    (prover : OracleComp (Prefix Wire (Coord → K) total) (Coord → K) (ℕ → Wire))
    {honest : Certified (root.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (root.fold 0).sourceBound (root.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 _tape : Tape (Prefix Wire (Coord → K) total) Coord K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
            ¬ Proved setup packageWidths step cap control head
              (((Structure.protocol whole).input prover table 0).wire 0) then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + (7424 / (Fintype.card K : ℝ) + 135 * p)) +
        (queries + 1) * (144 : ℕ) / budget +
        breakChance setup packageWidths step cap control shape read head bound accepts opening
          key openKey mass Sum.inl prover honest innerBound budget := by
  have loops : 2 + (1 + (∑ g, (root.shape g).blocks + 1 + (level₁.shape.blocks + 1 +
      (level₂.shape.blocks + 1 + (level₃.shape.blocks + 1 + (level₄.shape.blocks + 1 +
        (level₅.shape.blocks + 1 + Key20.terminal.blocks))))))) = 144 := by decide
  have error := Key20.towerError_le (β := β) (key := key) (openKey := openKey) (claim := claim)
    (block := block) (mass := mass) (β₁ := β₁) (key₁ := key₁) (openKey₁ := openKey₁)
    (claim₁ := claim₁) (block₁ := block₁) (weight₁ := weight₁) (key₂ := key₂)
    (openKey₂ := openKey₂) (claim₂ := claim₂) (block₂ := block₂) (weight₂ := weight₂)
    (key₃ := key₃) (openKey₃ := openKey₃) (claim₃ := claim₃) (block₃ := block₃)
    (weight₃ := weight₃) (key₄ := key₄) (openKey₄ := openKey₄) (claim₄ := claim₄)
    (block₄ := block₄) (weight₄ := weight₄) (key₅ := key₅) (openKey₅ := openKey₅)
    (claim₅ := claim₅) (block₅ := block₅) (weight₅ := weight₅) (read := Compose.read C)
    (start := Compose.root C vf) chance chance₁ chance₂ chance₃ chance₄ chance₅ chanceL
  refine (mixed_sound setup packageWidths step cap control shape read head bound premises laid
    lookupsFit challengesFit cube early injective units sparse root_sized.2.2.1
    root_sized.2.2.2.1 root_sized.2.2.2.2 ((by decide : rootVars ≤ 22).trans slots)
    ((by decide : rootBatch ≤ 22).trans slots) ((by decide : 3 ≤ 22).trans slots)
    (Key20.levelSized slots (by decide) (by decide) level₁_sized.2.2.2.1 level₁_sized.2.2.2.2
      injective₁ units₁ sparse₁)
    (Key20.levelSized slots (by decide) (by decide) sized₂.2.2.1 sized₂.2.2.2 injective₂ units₂
      sparse₂)
    (Key20.levelSized slots (by decide) (by decide) sized₃.2.2.1 sized₃.2.2.2 injective₃ units₃
      sparse₃)
    (Key20.levelSized slots (by decide) (by decide) sized₄.2.2.1 sized₄.2.2.2 injective₄ units₄
      sparse₄)
    (Key20.levelSized slots (by decide) (by decide) sized₅.2.2.1 sized₅.2.2.2 injective₅ units₅
      sparse₅)
    (⟨bounded₃, (Key20.norms_sized (level₃, norm₃) (by simp)).1,
      (Key20.norms_sized (level₃, norm₃) (by simp)).2⟩ :
        NormSized level₃.shape level₃.fold decode₃ norm₃.operator norm₃.limit norm₃.vars)
    (⟨bounded₄, (Key20.norms_sized (level₄, norm₄) (by simp)).1,
      (Key20.norms_sized (level₄, norm₄) (by simp)).2⟩ :
        NormSized level₄.shape level₄.fold decode₄ norm₄.operator norm₄.limit norm₄.vars)
    (⟨bounded₅, (Key20.norms_sized (level₅, norm₅) (by simp)).1,
      (Key20.norms_sized (level₅, norm₅) (by simp)).2⟩ :
        NormSized level₅.shape level₅.fold decode₅ norm₅.operator norm₅.limit norm₅.vars)
    inside injectiveL unitsL boundedL prover next registered covers bounded positive).trans ?_
  rw [loops]
  exact add_le_add (add_le_add (mul_le_mul_of_nonneg_left
    (add_le_add (le_refl (setup.runCost shape)) error) (by positivity : (0 : ℝ) ≤ queries + 1))
    le_rfl) le_rfl

end NightstreamFPrime.Export.BoundedDepth.FinalKey20
