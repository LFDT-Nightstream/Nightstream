import NightstreamFPrime.Export.BoundedDepth.FinalAkita
import NightstreamFPrime.Spec.Compression.Akita.Extract.ComposeComplete
import NightstreamFPrime.Spec.Compression.Akita.Key20Root
import NightstreamFPrime.Spec.Compression.Akita.Key20Tower

/-! The final verifier over the Akita opening of the schedule file
`crates/ns-compress/artifacts/key20.aks` (SPEC.md, Algorithm 10; Akita, ePrint 2026/1983, revision
of 2026-10-15, Theorem 10.17, Proposition 10.44 and Theorem 10.46).

`sound` is `FinalAkita.mixed_sound` for the numbers of `Spec/Compression/Akita/Key20.lean`, with
the bound `Key20.towerError_le` of the stage errors of the opening and its 142 retry loops.

`Use` has the conditions of the use of this verifier with the key of `key20.aks`: the claims
pass the layout of step 5, the key polynomial has 512 blocks, and the opening point of `V` has
the zeros `laidZeros` of the layout. `sound` is for a setup with `Use`. `key20_points` gives the
numbers of the three opening claims: the points of the key polynomial, `W` and `V` have 20, 16
and 16 coordinates. `Use.points` gives these lengths for the points of the Akita opening on each
transcript with the conditions `Compose.Ran`. The link `RootLinked` of the root needs 20, 16 and
16 coordinates or more (`Key20.root_points`).

`complete` states that the verifier `towerAccepts` of `sound` accepts the honest prover: the
event of `sound` is not empty when the hypotheses of `complete` hold. The honest prover of Π_CCS
on the fresh instance and the fit conditions of the Akita opening are hypotheses of
`complete`. -/

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
  (bound : Prefix Wire (Option Coord → K) total → (g : Fin 3) → (root.shape g).Payload)
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
    (And.left sized₃) level₃.batch (And.left (And.right sized₃)) block₃ decode₃
    (Compose.read C) weight₃ norm₃.operator norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) key₄ openKey₄ claim₄ level₄.vars
    (And.left sized₄) level₄.batch (And.left (And.right sized₄)) block₄ decode₄
    (Compose.read C) weight₄ norm₄.operator norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅ claim₅ level₅.vars
    (And.left sized₅) level₅.batch (And.left (And.right sized₅)) block₅ decode₅
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

local notation "links" =>
  towerLinks (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) Route.coefficient
    (Route.euclidean norm₃.operator norm₃.limit) (Route.euclidean norm₄.operator norm₄.limit)
    (Route.euclidean norm₅.operator norm₅.limit) zeroth first second third fourth fifth L
    (Compose.root C vf) (Compose.read C) commit value place (And.left quarters)
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

/-- The zeros of the opening point of `V` for the layout of the claims: `inverse_vars −
table_witness_vars − LOOKUP_VARS` (Rust `inverse_point`, `crates/ns-compress/src/lib.rs`), from
the Lean model `Compression.Rust.layoutNew` of Rust `Layout::new`. No Lean result compares this
number with the Rust code. -/
def laidZeros (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (shape : Claims logicalWidth publicFits) : ℕ :=
  match setup.placed shape with
  | .ok l => l.inverseVars - l.spanVars - Compression.Rust.lookupVars
  | .error _ => 0

/-- The use of the final verifier with the key of `key20.aks`. `laid`: the claims pass the layout
of step 5 (SPEC.md Algorithm 10). `blocks`: the key polynomial has 512 blocks. `zeros`: the
opening point of `V` has the zeros of the layout. -/
structure Use (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (shape : Claims logicalWidth publicFits) : Prop where
  laid : setup.Laid shape
  blocks : setup.keyBlocks = 512
  zeros : setup.zeros = laidZeros setup shape

/-- The numbers of the opening claims of the argument for the key polynomial of `key20.aks`
(Rust `Layout::new` with `akita::SCHEDULES`, and SPEC.md Algorithm 10, the Akita claims). For a
setup with `Use`, the point `(kp, rowPoint)` of the key polynomial has `6 + 9 + 5 = 20`
coordinates, the point `ρ` of `W` has 16 coordinates, and the point of `V` has
`spanVars + 2 + zeros = 16` coordinates. These are the sizes `2^20`, `2^16` and `2^16` of the
tables of the groups of `Key20.root`. The span has at most 14 variables. -/
theorem key20_points (use : Use setup shape) :
    (setup.params shape).laneVars + (setup.params shape).keyBlockVars +
        (setup.params shape).rowVars = 20 ∧
      (setup.params shape).witnessVars = 16 ∧
      (setup.params shape).spanVars + (setup.verifier shape).columnVars +
        (setup.verifier shape).zeros = 16 ∧
      (setup.params shape).spanVars ≤ 14 := by
  obtain ⟨⟨l, ok⟩, -⟩ := use.laid
  have same : setup.layout shape = l.production := by simp only [FinalVerifier.Setup.layout, ok]
  have zeros : setup.zeros = l.inverseVars - l.spanVars - Compression.Rust.lookupVars := by
    rw [use.zeros]
    simp only [laidZeros, ok]
  unfold FinalVerifier.Setup.placed at ok
  rw [use.blocks] at ok
  obtain ⟨schedule, member, held, -, row, inRow, fits, rfl⟩ := Compression.Rust.layoutNew_ok ok
  have all : ∀ s ∈ Compression.Rust.schedules, s.blocks = 512 →
      s.keyVars = 20 ∧ ∀ r ∈ s.rows, r = (16, 16) := by decide
  obtain ⟨keyVars, rows⟩ := all schedule member held
  obtain rfl := rows row inRow
  simp only [Compression.Rust.rowFits, Bool.and_eq_true, decide_eq_true_eq] at fits
  obtain ⟨span, known⟩ : ∃ span, span = (setup.layout shape).spanVars := ⟨_, rfl⟩
  have small : span + 2 ≤ 16 := by
    rw [known, same]
    exact fits.2
  have padded : setup.zeros = 16 - span - 2 := by
    rw [zeros, known, same]
    rfl
  show 6 + (setup.layout shape).keyBlockVars + 5 = 20 ∧ (setup.layout shape).witnessVars = 16 ∧
    (setup.layout shape).spanVars + 2 + setup.zeros = 16 ∧ (setup.layout shape).spanVars ≤ 14
  rw [← known, padded, same]
  show 6 + (schedule.keyVars - 6 - 5) + 5 = 20 ∧ (16 : ℕ) = 16 ∧ _
  rw [keyVars]
  exact ⟨rfl, rfl, by omega, by omega⟩

omit [Nonempty Coord] in
/-- On a transcript with the conditions `Compose.Ran` of the final verifier with `Use`, the points
of the three claims of the Akita opening have 20, 16 and 16 coordinates
(`Compose.Ran.lengths`, `key20_points`). -/
theorem Use.points (use : Use setup shape) {point : Prefix Wire (Option Coord → K) total}
    (ran : Compose.Ran C vf commit value place point) :
    ∀ g, (place (point.wire (Compose.root C vf)) g).length = ![20, 16, 16] g := by
  obtain ⟨key, w, v, -⟩ := key20_points setup shape use
  obtain ⟨keyLength, wLength, vLength⟩ :=
    ran.lengths (FinalVerifier.laid_facts setup use.laid).1
  intro g
  fin_cases g
  · exact keyLength.trans key
  · exact wLength.trans w
  · exact vLength.trans v

/-- Soundness of the final verifier over the Akita opening of the schedule `key20.aks`
(`FinalAkita.mixed_sound` for the numbers of `Key20.lean`). For each prover with at most `Q`
queries to the oracle, the probability that the verifier `towerAccepts` accepts and no proof with
witnesses satisfies `Verifies` for the statement of the first message is at most
`(Q + 1)·(runCost + 317479/(1024·|K|) + 135·p) + (Q + 1)·144/budget + breakChance`. `p` is a bound
of the
slot probability of each challenge family (`Key20.towerError_le`), and 144 is the count of the
retry loops: 2 for the compression argument and 142 for the opening. `breakChance` is the
probability that the extractor returns a solution of Module-SIS for the key commitment, for a
matrix of a group, of a level, or for `A_term`. `use`: the conditions `Use` of the key of
`key20.aks`, with which the points of the three opening claims have 20, 16 and 16 coordinates
(`Use.points`). `coords`: an answer of the oracle has 28 slots, which is a slot for each
coordinate of `α`, for each lookup challenge, for each challenge after `V` and for each fold
challenge of a level. `premises`, `early`, `registered` and `covers` are the hypotheses of
`FinalAkita.mixed_sound` on the final verifier. The other hypotheses are those of
`Key20.tower_suffix` and `Key20.towerError_le`. -/
theorem sound [Finite Wire] {peak : Fin 3 → ℕ} (premises : Premises setup packageWidths)
    (use : Use setup shape) (coords : 28 ≤ Fintype.card Coord)
    (early : ∀ anchor g, g ≠ 2 → bound anchor g = bound (anchor.take 0) g)
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
    (next : Terminal.start L = Tier.finish fifth) (inside : Terminal.start L < total)
    (injectiveL : Function.Injective (Terminal.block L))
    (unitsL : UnitDifferences (Terminal.family L))
    (boundedL : ∀ challenge ∈ Terminal.family L, MulBound normTerminal.operator challenge)
    {p : ℝ} (chance : ∀ g, slotChance (decode g) ≤ p) (chance₁ : slotChance decode₁ ≤ p)
    (chance₂ : slotChance decode₂ ≤ p) (chance₃ : slotChance decode₃ ≤ p)
    (chance₄ : slotChance decode₄ ≤ p) (chance₅ : slotChance decode₅ ≤ p)
    (chanceL : slotChance (Terminal.decode L) ≤ p)
    (prover : OracleComp (Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))
    {honest : Certified (root.shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * (root.fold 0).sourceBound (root.shape 0) + 2 * mass 0 * sourceBound ≤
      innerBound)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 _tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
            ¬ Proved setup packageWidths step cap control head
              (((Structure.protocol whole).input prover table 0).wire 0) then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + (317479 / (1024 * Fintype.card K) + 135 * p)) +
        (queries + 1) * (144 : ℕ) / budget +
        breakChance setup packageWidths step cap control shape read head bound accepts opening
          key openKey mass Sum.inl prover honest innerBound budget := by
  obtain ⟨-, -, -, span⟩ := key20_points setup shape use
  have laid := use.laid
  have slots : 22 ≤ Fintype.card Coord := (by decide : 22 ≤ 28).trans coords
  have lookupsFit : (setup.params shape).lookups ≤ Fintype.card Coord :=
    (by decide : 4 ≤ 28).trans coords
  have cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord := coords
  have challengesFit : 4 + ((setup.params shape).lookups + ((setup.params shape).claims +
      (setup.params shape).claims + (setup.params shape).spanVars)) ≤ Fintype.card Coord := by
    have claims : (setup.params shape).claims ≤ 3 := Nat.succ_le_succ mode.slots_le
    have lookups : (setup.params shape).lookups = 4 := rfl
    omega
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
        NormSized level₃.shape level₃.fold decode₃ norm₃.operator norm₃.limit level₃.vars)
    (⟨bounded₄, (Key20.norms_sized (level₄, norm₄) (by simp)).1,
      (Key20.norms_sized (level₄, norm₄) (by simp)).2⟩ :
        NormSized level₄.shape level₄.fold decode₄ norm₄.operator norm₄.limit level₄.vars)
    (⟨bounded₅, (Key20.norms_sized (level₅, norm₅) (by simp)).1,
      (Key20.norms_sized (level₅, norm₅) (by simp)).2⟩ :
        NormSized level₅.shape level₅.fold decode₅ norm₅.operator norm₅.limit level₅.vars)
    inside injectiveL unitsL boundedL prover next registered covers bounded positive).trans ?_
  rw [loops]
  exact add_le_add (add_le_add (mul_le_mul_of_nonneg_left
    (add_le_add (le_refl (setup.runCost shape)) error) (by positivity : (0 : ℝ) ≤ queries + 1))
    le_rfl) le_rfl

/-- The verifier `towerAccepts` of `sound` accepts the honest prover of a final proof, without
the probability that a response of the Akita opening does not fit (SPEC.md Algorithm 9 and
Algorithm 10; Akita Theorem 10.9; `Compose.accepts` with `Key20.tower_complete`). For a setup with
`Use` and each table of the oracle with `calm`, `regular` and `decodes` to `decodesL`, a proof
with the first message `given` exists whose complete transcript satisfies `towerAccepts`.

`given` is the first message: `checks` and `gate` state that its statement and its claims pass
steps 1 to 5 of Algorithm 10. `W` is the table of the witnesses and the counts, with honest
counts and 0 from the cell `2^16`. `leads`: the prover of Π_CCS on the fresh instance has a
message for each draw sequence such that the run ends in a probe whose claims satisfy `Shaped`
and `Relation` for `W`. No Lean result gives `leads`.

`calm` and `regular`: an answer of the table decodes to lookup challenges that are no table
values and to challenges after `V` whose `ω` has a row point: the two failures of Algorithm 9,
Rejects. `decodes` to `decodesL`: the slots of an answer decode to fold challenges.

`sends`, `wide`, `capped`, `room`, `short` and `carriesM` to `carriesL`: a wire message can have
the data of the prover. `bound₀`, `bound₁`, `bound₂`: the commitments that a stage input fixes
are the commitments of the key polynomial and of `W`, and the commitment of `V` in the message
after the lookup challenges. `next`, `inside`, `reads`, `clears`: the stage of the terminal and
its messages. `nonces`: the terminal reads the nonce of the response search with `read`. `fit`:
the fit conditions of `Key20.tower_complete` for the tables of the key polynomial, `W` and the
honest inverses of each lookup challenges that are no table values. -/
theorem complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    {accept : Prefix Wire (Option Coord → K) total → Prop}
    {statementOf : Wire → TerminalStatement Key20.terminal K}
    {partialsOf : Wire → Fin Key20.terminal.blocks → Rq Key20.terminal.innerDegree}
    (use : Use setup shape) (sends : Sends sendable K width cap) (wide : 8 ≤ width)
    (capped : 3029804899 ≤ cap) (room : 3 < slots)
    (short : ∀ values : Compression.Verifier.Values K, (∀ k, 4 ≤ k → values.2.1 k = 0) →
      sendable (Compression.Verifier.Values K) values)
    (carriesM : Carries sendable slots read commit)
    (carriesC : Carries sendable slots read fun wire => (commit wire, value wire, place wire))
    (carriesR : Carries sendable slots read claim)
    (carries₀ : Carries sendable slots read fun wire => (commit₁ wire, claim₁ wire))
    (carries₁ : Carries sendable slots read fun wire => (commit₂ wire, claim₂ wire))
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire => (commit₅ wire, claim₅ wire))
    (carries₅ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read (Terminal.response L))
    (next : Terminal.start L = Tier.finish fifth) (inside : Terminal.start L < total)
    (reads : ∀ point, Terminal.read L point = statementOf (point.wire (Terminal.start L)))
    (clears : ∀ point, Terminal.partials L point = partialsOf (point.wire (Terminal.start L)))
    (nonces : ∀ wire, Terminal.nonce L wire = read.nonce wire)
    (table : Prefix Wire (Option Coord → K) total → Option Coord → K)
    (calm : ∀ input, sample (Coord → K) (Fin (setup.params shape).lookups → Zq)
      (view (table input)) ∉ Compression.Verifier.tableHit (setup.params shape))
    (regular : ∀ input,
      sample (Coord → K) (Compression.Verifier.Challenges (setup.params shape))
        (view (table input)) ∉ Compression.Verifier.omegaBad (setup.params shape))
    (decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input i, (decode₁ (table input (some (block₁ i)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodesL : ∀ input i, (Terminal.decode L (table input (some (Terminal.block L i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros K 57)
    (given : Wire)
    (checks : FinalVerifier.Checks mode (head given).statement ((head given).claims shape))
    (gate : setup.Gate ((head given).claims shape)) (W : Table)
    (counts : Compression.Verifier.HonestCounts (setup.params shape) W)
    (tail : ∀ cell, 2 ^ 16 ≤ cell → W cell = 0)
    (leads : WinsOff (Compression.Verifier.Rejects (setup.params shape)) sendable slots
      (setup.freshRun (head given).statement ((head given).claims shape) W)
      (Compression.Fresh.costs CompressionFresh.oneShape 8).length 0 fun result _ _ =>
        ∃ probe, result = some probe ∧
          Compression.Verifier.Shaped (setup.params shape)
            (setup.claimsOf (head given).statement ((head given).claims shape) probe) ∧
          Compression.Verifier.Relation (setup.params shape)
            (setup.claimsOf (head given).statement ((head given).claims shape) probe) W)
    (bound₀ : ∀ anchor : Prefix Wire (Option Coord → K) total, anchor.first = given →
      bound anchor 0 = (Compression.Akita.commit (key 0) (keyBase (setup.params shape))).1)
    (bound₁ : ∀ anchor : Prefix Wire (Option Coord → K) total, anchor.first = given →
      bound anchor 1 = (Compression.Akita.commit (key 1) W).1)
    (bound₂ : ∀ anchor : Prefix Wire (Option Coord → K) total,
      bound anchor 2 = commit (anchor.wire (Compose.front C).finish) 2)
    (fit : ∀ τ, Compression.Verifier.Apart (setup.params shape) τ →
      BatchFit β key openKey rootVars (And.left root_sized) 8 quotientDigits decode
        ![keyBase (setup.params shape), W,
          Compression.Verifier.honestInverses (setup.params shape) W τ] fun w₀ =>
      LiftFit β₁ key₁ openKey₁ level₁.fold level₁.vars (And.left level₁_sized) quotientDigits
        decode₁ (fun i => level₁.shape.embed (level₁.shape.pack w₀ i)) fun w₁ =>
      TraceFit (And.left quarters) key₂ openKey₂ level₂.fold (fun _ => decode₂) (fun _ => True)
        (packed w₁) fun w₂ =>
      TraceFit (And.left (And.right quarters)) key₃ openKey₃ level₃.fold (fun _ => decode₃)
        (fun z => vecNormSq z ≤ norm₃.limit) (packed w₂) fun w₃ =>
      TraceFit (And.left (And.right (And.right quarters))) key₄ openKey₄ level₄.fold
        (fun _ => decode₄) (fun z => vecNormSq z ≤ norm₄.limit) (packed w₃) fun w₄ =>
      TraceFit (And.left (And.right (And.right (And.right quarters)))) key₅ openKey₅ level₅.fold
        (fun _ => decode₅) (fun z => vecNormSq z ≤ norm₅.limit) (packed w₄) fun w₅ =>
      ∀ c : Fin Key20.terminal.blocks → Rq Key20.terminal.innerDegree,
        (∀ i, ∃ slot, Terminal.decode L slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource
          (And.right (And.right (And.right (And.right quarters)))) (packed w₅ : ℕ → K))) ≤
          normTerminal.limit) :
    ∃ proof : ℕ → Wire, proof 0 = given ∧
      accepts ((transcript total accept).final (.done proof) table) := by
  obtain ⟨wf, -, matrix, columns, rows, key_fits⟩ := FinalVerifier.laid_facts setup use.laid
  obtain ⟨keyLength, wLength, vLength, -⟩ := key20_points setup shape use
  have fits : Compose.root C vf ≤ total := by
    have ends₀ := Tier.finish_eq zeroth
    have ends₁ := Tier.finish_eq first
    have ends₂ := Tier.finish_eq second
    have ends₃ := Tier.finish_eq third
    have ends₄ := Tier.finish_eq fourth
    have ends₅ := Tier.finish_eq fifth
    have starts₀ : Tier.start zeroth = Compose.root C vf + 1 := rfl
    have starts₁ : Tier.start first = Tier.finish zeroth := rfl
    have starts₂ : Tier.start second = Tier.finish first := rfl
    have starts₃ : Tier.start third = Tier.finish second := rfl
    have starts₄ : Tier.start fourth = Tier.finish third := rfl
    have starts₅ : Tier.start fifth = Tier.finish fourth := rfl
    omega
  have lead : Compose.lead C given W =
      setup.freshRun (head given).statement ((head given).claims shape) W := by
    show setup.lead _ _ W = _
    unfold FinalVerifier.Setup.lead
    exact ite_eq_left ⟨checks, gate⟩
  obtain ⟨proof, zero, ran, linked⟩ := Compose.accepts (commit := commit) (value := value)
    (place := place) (accept := accept) wf matrix columns rows key_fits
    (fun _ W other => FinalVerifier.lead_bare setup _ _ W other) fits sends (by omega) (by omega)
    (fun values zero => short values zero) carriesM carriesC table
    (fun input T _ _ => Compression.Verifier.rejects_sample (calm input) (regular input) T)
    (onPath_nil accept table given) rfl W counts
    (params_tablesDistinct _ _ _ _ fun _ => checks.2.2.2.1 _)
    (by
      rw [lead]
      unfold FinalVerifier.Setup.freshRun
      exact trivial)
    (by
      rw [lead]
      exact leads)
    key bound₀ bound₁ bound₂ links
    (fun τ apart moves on length _ committed lengths valued =>
      Key20.tower_complete sends wide capped room carriesR carries₀ carries₁ carries₂ carries₃
        carries₄ carries₅ carriesL next inside reads clears nonces table decodes decodes₁ decodes₂
        decodes₃ decodes₄ decodes₅ decodesL worked on length
        (tables := ![keyBase (setup.params shape), W,
          Compression.Verifier.honestInverses (setup.params shape) W τ])
        (fun g cell large => by
          fin_cases g
          · exact Compose.keyBase_zero rows (by rw [keyLength]; exact large)
          · exact tail cell large
          · have small : (setup.params shape).spanVars + (setup.verifier shape).columnVars ≤
                16 := by omega
            exact Compression.Verifier.honestInverses_zero W τ columns
              ((Nat.pow_le_pow_right (by decide) small).trans large))
        committed
        (fun g => by
          fin_cases g
          · exact lengths.1.trans keyLength
          · exact lengths.2.1.trans wLength
          · exact lengths.2.2.trans vLength)
        valued (fit τ apart))
  exact ⟨proof, zero, ran, linked⟩

end NightstreamFPrime.Export.BoundedDepth.FinalKey20
