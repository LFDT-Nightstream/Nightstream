import NightstreamFPrime.Export.BoundedDepth.FinalKey20
import NightstreamFPrime.Spec.Compression.Akita.Key30Tower

/-! The final verifier over the Akita opening of a row of the schedule file
`crates/ns-compress/artifacts/key30.aks` (SPEC.md, Algorithm 10; Akita, ePrint 2026/1983, revision
of 2026-10-15, Section 9 with Lemma 9.1, Theorem 10.17, Proposition 10.44 and Theorem 10.46).

`sound` is `FinalAkita.final_sound` for the suffix `Key30.tower_suffix` of the opening with an
offloaded root, with the bound `Key30.towerError_le` of its stage errors and its retry loops.

`Use` has the conditions of the use of this verifier with the key of `key30.aks` and the row `r`
of the file: the claims pass the layout of step 5, the key polynomial has `2^19` blocks, the
opening point of `V` has the zeros `FinalKey20.laidZeros` of the layout, and `W` has the count of
variables of the row. `key30_points` gives the numbers of the three opening claims: the points of
the key polynomial, `W` and `V` have 30, 27 or 26, and 28 coordinates. `Use.points` gives these
lengths for the points of the Akita opening on each transcript with the conditions `Compose.Ran`.
The link `RootLinked` of the root needs these coordinates (`Key30.root_points`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalKey30

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
open Key30 (rows root level₁ level₂ level₃ level₄ level₅ level₆ norm₅ norm₆ normTerminal
  rootVars rootBatch quotientDigits₀ quotientDigits vars₁ batch₁ setupVars root_sized
  level₁_sized level₂_sized level₃_sized level₄_sized tails_sized quarters)

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
  (head : Wire → First logicalWidth publicFits) {r : Key30.Row} {member : r ∈ rows}
  (bound : Prefix Wire (Coord → K) total → (g : Fin 3) → ((root r).shape g).Payload)
  [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]
  {β : ∀ g, Module.Basis (Fin ((root r).planes g)) Zq K}
  {key : (g : Fin 3) → Key ((root r).shape g)} {openKey : (root r).OpenKey}
  {claim : Wire → (root r).Claim K} {block : (root r).Block → Coord}
  {decode : (g : Fin 3) → K → Option (Rq ((root r).carrierDegree g))} {mass : Fin 3 → ℕ}
  {flat : Table}
  {scan : ∀ (st : (root r).Statement K) (challenge : (root r).Challenge),
    SetupScan (Batch.liftedReduction β st challenge rootVars (root_sized r member).1 8
      quotientDigits₀ (root_sized r member).2.1 rootBatch (root_sized r member).2.2.1) setupVars
      flat}
  {β₁ : ∀ g, Module.Basis (Fin ((level₁ r).planes g)) Zq K}
  {key₁ : (g : Fin 2) → Key ((level₁ r).shape g)} {openKey₁ : (level₁ r).OpenKey}
  {claim₁ : Wire → (level₁ r).Claim K} {block₁ : (level₁ r).Block → Coord}
  {decode₁ : (g : Fin 2) → K → Option (Rq ((level₁ r).carrierDegree g))} {mass₁ : Fin 2 → ℕ}
  {registry : ((level₁ r).shape 0).Payload} {reference : Decommitment ((level₁ r).shape 0)}
  {β₂ : Module.Basis (Fin (level₂ r).opening.planes) Zq K} {key₂ : Key (level₂ r).shape}
  {openKey₂ : OpeningKey (level₂ r).opening} {claim₂ : Wire → Claim (level₂ r).opening K}
  {block₂ : Fin (level₂ r).shape.blocks → Coord}
  {decode₂ : K → Option (Rq (level₂ r).opening.carrierDegree)} {weight₂ : ℕ}
  {β₃ : Module.Basis (Fin level₃.opening.planes) Zq K} {key₃ : Key level₃.shape}
  {openKey₃ : OpeningKey level₃.opening} {claim₃ : Wire → Claim level₃.opening K}
  {block₃ : Fin level₃.shape.blocks → Coord}
  {decode₃ : K → Option (Rq level₃.opening.carrierDegree)} {weight₃ : ℕ}
  {key₄ : Key level₄.shape} {openKey₄ : OpeningKey level₄.opening}
  {claim₄ : Wire → TraceClaim level₄.opening K} {block₄ : Fin level₄.shape.blocks → Coord}
  {decode₄ : K → Option (Rq level₄.shape.innerDegree)} {weight₄ : ℕ}
  {key₅ : Key level₅.shape} {openKey₅ : OpeningKey level₅.opening}
  {claim₅ : Wire → TraceClaim level₅.opening K} {block₅ : Fin level₅.shape.blocks → Coord}
  {decode₅ : K → Option (Rq level₅.shape.innerDegree)} {weight₅ : ℕ}
  {key₆ : Key level₆.shape} {openKey₆ : OpeningKey level₆.opening}
  {claim₆ : Wire → TraceClaim level₆.opening K} {block₆ : Fin level₆.shape.blocks → Coord}
  {decode₆ : K → Option (Rq level₆.shape.innerDegree)} {weight₆ : ℕ}
  {L : Terminal Key30.terminal K Wire Coord K total
    (.euclidean normTerminal.operator normTerminal.limit)}
  {commit : Wire → (g : Fin 3) → ((root r).shape g).Payload} {value : Wire → Fin 3 → K}
  {place : Wire → Fin 3 → List K}
  {commits : Wire → (g : Fin 2) → ((level₁ r).shape g).Payload} {values : Wire → Fin 2 → K}
  {places : Wire → Fin 2 → List K} {commit₁ : Wire → ((level₁ r).shape 1).Payload}
  {commit₂ : Wire → (level₂ r).shape.Payload} {commit₃ : Wire → level₃.shape.Payload}
  {commit₄ : Wire → level₄.shape.Payload} {commit₅ : Wire → level₅.shape.Payload}
  {commit₆ : Wire → level₆.shape.Payload}
  {stateOf : Wire → Fin Key30.terminal.blocks → Fin Key30.terminal.innerRank →
    Rq Key30.terminal.innerDegree}

local notation "C" => compose setup packageWidths step cap control shape read head bound

local notation "vf" => FinalVerifier.Setup.verifier setup shape

local notation "zeroth" =>
  offBatch β key openKey claim rootVars (And.left (root_sized r member)) 8 quotientDigits₀
    (And.left (And.right (root_sized r member))) rootBatch
    (And.left (And.right (And.right (root_sized r member)))) block decode (Compose.read C) mass
    (Compose.root C vf + 1) setupVars flat scan

local notation "first" =>
  Tier.pinned (liftBatch β₁ key₁ openKey₁ claim₁ vars₁ (And.left (level₁_sized r member)) 16
    quotientDigits (And.left (And.right (level₁_sized r member))) batch₁
    (And.left (And.right (And.right (level₁_sized r member)))) block₁ decode₁ (Compose.read C)
    mass₁ (Tier.finish zeroth + 1)) (Batch.Pinned 0 registry flat)
    (fun ω _ => Batch.pinBreak key₁ openKey₁ _ 0 reference ω)

local notation "second" =>
  liftFold β₂ key₂ openKey₂ claim₂ (Key20.Level.vars (level₂ r))
    (And.left (level₂_sized r member)) quotientDigits
    (And.left (And.right (level₂_sized r member))) (Key20.Level.batch (level₂ r))
    (And.left (And.right (And.right (level₂_sized r member)))) block₂ decode₂ (Compose.read C)
    weight₂ (Tier.finish first)

local notation "third" =>
  Tier.tensored (slimFold β₃ key₃ openKey₃ claim₃ level₃.vars (And.left level₃_sized)
    quotientDigits (And.left (And.right level₃_sized)) level₃.batch
    (And.left (And.right (And.right level₃_sized))) block₃ decode₃ (Compose.read C) weight₃
    (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := level₄.fold) key₄ openKey₄ claim₄ level₄.vars
    (And.left level₄_sized) level₄.batch (And.left (And.right level₄_sized)) block₄ decode₄
    (Compose.read C) weight₄ (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅ claim₅ level₅.vars
    (And.left (tails_sized level₅ (List.mem_cons_self ..))) level₅.batch
    (And.left (And.right (tails_sized level₅ (List.mem_cons_self ..)))) block₅ decode₅
    (Compose.read C) weight₅ norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail

local notation "sixth" =>
  Tier.tensored (committedTraceSq (fold := level₆.fold) key₆ openKey₆ claim₆ level₆.vars
    (And.left (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))) level₆.batch
    (And.left (And.right (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))))
    block₆ decode₆ (Compose.read C) weight₆ norm₆.operator norm₆.limit
    (Tier.finish fifth)) level₆.tail

local notation "accepts" =>
  offAccepts (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
    (route₅ := Route.euclidean norm₅.operator norm₅.limit)
    (route₆ := Route.euclidean norm₆.operator norm₆.limit) first second third fourth fifth sixth L
    commits values places C vf zeroth commit value place registry commit₁ commit₂ commit₃
    (And.left quarters) (And.left (And.right quarters))
    (And.left (And.right (And.right quarters))) (And.right (And.right (And.right quarters)))
    commit₄ commit₅ commit₆ stateOf

local notation "opening" =>
  offRoot (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
    (route₅ := Route.euclidean norm₅.operator norm₅.limit)
    (route₆ := Route.euclidean norm₆.operator norm₆.limit) accepts zeroth first second third
    fourth fifth sixth L 32 16 16 16 (Compose.root C vf) (Compose.read C) commit value place
    commits values places flat

local notation "whole" => Compose.chain C vf accepts opening key openKey mass Sum.inl

/-- The use of the final verifier with the key of `key30.aks` and the row `r` of the file.
`laid`: the claims pass the layout of step 5 (SPEC.md Algorithm 10). `blocks`: the key polynomial
has `2^19` blocks. `zeros`: the opening point of `V` has the zeros of the layout. `row`: `W` has
the count of variables of the row, 27 or 26. -/
structure Use (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (shape : Claims logicalWidth publicFits) (r : Key30.Row) : Prop where
  laid : setup.Laid shape
  blocks : setup.keyBlocks = 524288
  zeros : setup.zeros = FinalKey20.laidZeros setup shape
  row : (setup.params shape).witnessVars = r.vars

/-- The numbers of the opening claims of the argument for the key polynomial of `key30.aks`
(Rust `Layout::new` with `akita::SCHEDULES`, and SPEC.md Algorithm 10, the Akita claims). For a
setup with `Use`, the point `(kp, rowPoint)` of the key polynomial has `6 + 19 + 5 = 30`
coordinates, the point of `V` has `spanVars + 2 + zeros = 28` coordinates, the span has at most 26
variables, and the point `ρ` of `W` has the count of variables of the row. These are the sizes of
the tables of the groups of `Key30.root`. -/
theorem key30_points (use : Use setup shape r) :
    (setup.params shape).laneVars + (setup.params shape).keyBlockVars +
        (setup.params shape).rowVars = 30 ∧
      (setup.params shape).spanVars + (setup.verifier shape).columnVars +
        (setup.verifier shape).zeros = 28 ∧
      (setup.params shape).spanVars ≤ 26 ∧ (setup.params shape).witnessVars = r.vars := by
  refine and_assoc.mp (and_assoc.mp ⟨?_, use.row⟩)
  obtain ⟨⟨l, ok⟩, -⟩ := use.laid
  have same : setup.layout shape = l.production := by simp only [FinalVerifier.Setup.layout, ok]
  have zeros : setup.zeros = l.inverseVars - l.spanVars - Compression.Rust.lookupVars := by
    rw [use.zeros]
    simp only [FinalKey20.laidZeros, ok]
  unfold FinalVerifier.Setup.placed at ok
  rw [use.blocks] at ok
  obtain ⟨schedule, inList, held, -, row, inRow, fits, rfl⟩ := Compression.Rust.layoutNew_ok ok
  have all : ∀ s ∈ Compression.Rust.schedules, s.blocks = 524288 →
      s.keyVars = 30 ∧ ∀ row ∈ s.rows, row = (26, 28) ∨ row = (27, 28) := by decide
  obtain ⟨keyVars, both⟩ := all schedule inList held
  simp only [Compression.Rust.rowFits, Bool.and_eq_true, decide_eq_true_eq] at fits
  obtain ⟨span, known⟩ : ∃ span, span = (setup.layout shape).spanVars := ⟨_, rfl⟩
  rcases both row inRow with rfl | rfl
  all_goals
    have small : span + 2 ≤ 28 := by
      rw [known, same]
      exact fits.2
    have padded : setup.zeros = 28 - span - 2 := by
      rw [zeros, known, same]
      rfl
    show (6 + (setup.layout shape).keyBlockVars + 5 = 30 ∧
      (setup.layout shape).spanVars + 2 + setup.zeros = 28) ∧
        (setup.layout shape).spanVars ≤ 26
    rw [← known, padded, same]
    show (6 + (schedule.keyVars - 6 - 5) + 5 = 30 ∧ _) ∧ _
    rw [keyVars]
    exact ⟨⟨rfl, by omega⟩, by omega⟩

omit [Nonempty Coord] in
/-- On a transcript with the conditions `Compose.Ran` of the final verifier with `Use`, the points
of the three claims of the Akita opening have 30, 27 or 26, and 28 coordinates
(`Compose.Ran.lengths`, `key30_points`). -/
theorem Use.points (use : Use setup shape r) {point : Prefix Wire (Coord → K) total}
    (ran : Compose.Ran C vf commit value place point) :
    ∀ g, (place (point.wire (Compose.root C vf)) g).length = ![30, r.vars, 28] g := by
  obtain ⟨key, v, -, w⟩ := key30_points setup shape use
  obtain ⟨keyLength, wLength, vLength⟩ :=
    ran.lengths (FinalVerifier.laid_facts setup use.laid).1
  intro g
  fin_cases g
  · exact keyLength.trans key
  · exact wLength.trans w
  · exact vLength.trans v

/-- Soundness of the final verifier over the Akita opening of a row of the schedule `key30.aks`
(`FinalAkita.final_sound` with `Key30.tower_suffix`). For each prover with at most `Q` queries to
the oracle, the probability that the verifier accepts and no proof with witnesses satisfies
`Verifies` for the statement of the first message is at most
`(Q + 1)·(runCost + 8132/|K| + (1458 + B_1 + B_2)·p) + (Q + 1)·(1469 + B_1 + B_2)/budget +
breakChance`. The verifier `offAccepts` accepts with the conditions `Compose.Ran` and the links
`offLinks` of the opening with an offloaded root. `p` is a bound of the slot probability of each challenge
family (`Key30.towerError_le`), and `1469 + B_1 + B_2` is the count of the retry loops: 2 for the
compression argument and `1467 + B_1 + B_2` for the opening, which is 1680 for `W` of 27 variables
and 1670 for `W` of 26 variables. `breakChance` is the probability that the extractor returns a
solution of Module-SIS for the key commitment, for a matrix of a group, of a level, or for
`A_term`. `use`: the conditions `Use` of the key of `key30.aks` and the row, with which the points
of the three opening claims have 30, 27 or 26, and 28 coordinates (`Use.points`). `coords`: an
answer of the oracle has 40 slots, which is a slot for each coordinate of `α`, for each lookup
challenge, for each challenge after `V` and for each fold challenge of a level. `registeredS` and
`positive₁`: the premise of Akita Lemma 9.1 for the commitment `registry` of the setup prefix
`flat`. `premises`, `early`, `registered` and `covers` are the hypotheses of
`FinalAkita.final_sound` on the final verifier. The other hypotheses are those of
`Key30.tower_suffix` and `Key30.towerError_le`. -/
theorem sound [Finite Wire] {peak : Fin 3 → ℕ} {peak₁ : Fin 2 → ℕ}
    (premises : Premises setup packageWidths) (use : Use setup shape r)
    (coords : 40 ≤ Fintype.card Coord)
    (early : ∀ anchor g, g ≠ 2 → bound anchor g = bound (anchor.take 0) g)
    (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (injective₁ : Function.Injective block₁)
    (units₁ : ∀ g, UnitDifferences {challenge | ∃ slot, decode₁ g slot = some challenge})
    (sparse₁ : ∀ g challenge, (∃ slot, decode₁ g slot = some challenge) →
      Sparse (mass₁ g) (peak₁ g) challenge)
    (registeredS : Batch.Registered (key₁ 0) registry flat reference) (positive₁ : 0 < mass₁ 0)
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
    (next : Terminal.start L = Tier.finish sixth) (inside : Terminal.start L < total)
    (injectiveL : Function.Injective (Terminal.block L))
    (unitsL : UnitDifferences (Terminal.family L))
    (boundedL : ∀ challenge ∈ Terminal.family L, MulBound normTerminal.operator challenge)
    {p : ℝ} (chance : ∀ g, slotChance (decode g) ≤ p)
    (chance₁ : ∀ g, slotChance (decode₁ g) ≤ p) (chance₂ : slotChance decode₂ ≤ p)
    (chance₃ : slotChance decode₃ ≤ p) (chance₄ : slotChance decode₄ ≤ p)
    (chance₅ : slotChance decode₅ ≤ p) (chance₆ : slotChance decode₆ ≤ p)
    (chanceL : slotChance (Terminal.decode L) ≤ p)
    (prover : OracleComp (Prefix Wire (Coord → K) total) (Coord → K) (ℕ → Wire))
    {honest : Certified ((root r).shape 0)} {factorBound sourceBound innerBound : ℕ}
    (registered : ∀ anchor, honest.Opens factorBound sourceBound (key 0) (bound anchor 0)
      (keyBase (setup.params shape)))
    (covers : factorBound * ((root r).fold 0).sourceBound ((root r).shape 0) +
      2 * mass 0 * sourceBound ≤ innerBound)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 _tape : Tape (Prefix Wire (Coord → K) total) Coord K budget
          (Structure.protocol whole).stages,
        (if accepts ((Structure.protocol whole).final prover table) ∧
            ¬ Proved setup packageWidths step cap control head
              (((Structure.protocol whole).input prover table 0).wire 0) then (1 : ℝ) else 0) ≤
      (queries + 1) * (setup.runCost shape + (8132 / (Fintype.card K : ℝ) +
          ((1458 + r.blocks₁ + r.blocks₂ : ℕ) : ℝ) * p)) +
        (queries + 1) * (2 + (1467 + r.blocks₁ + r.blocks₂) : ℕ) / budget +
        breakChance setup packageWidths step cap control shape read head bound accepts opening
          key openKey mass Sum.inl prover honest innerBound budget := by
  obtain ⟨-, -, span, -⟩ := key30_points setup shape use
  have slots : 28 ≤ Fintype.card Coord := (by decide : 28 ≤ 40).trans coords
  have lookupsFit : (setup.params shape).lookups ≤ Fintype.card Coord :=
    (by decide : 4 ≤ 40).trans coords
  have cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord := slots
  have challengesFit : 4 + ((setup.params shape).lookups + ((setup.params shape).claims +
      (setup.params shape).claims + (setup.params shape).spanVars)) ≤ Fintype.card Coord := by
    have claims : (setup.params shape).claims ≤ 3 := Nat.succ_le_succ mode.slots_le
    have lookups : (setup.params shape).lookups = 4 := rfl
    omega
  have fits : Compose.root C vf ≤ total := by
    have ends₀ := Tier.finish_eq zeroth
    have ends₁ := Tier.finish_eq first
    have ends₂ := Tier.finish_eq second
    have ends₃ := Tier.finish_eq third
    have ends₄ := Tier.finish_eq fourth
    have ends₅ := Tier.finish_eq fifth
    have ends₆ := Tier.finish_eq sixth
    have starts₀ : Tier.start zeroth = Compose.root C vf + 1 := rfl
    have starts₁ : Tier.start first = Tier.finish zeroth + 1 := rfl
    have starts₂ : Tier.start second = Tier.finish first := rfl
    have starts₃ : Tier.start third = Tier.finish second := rfl
    have starts₄ : Tier.start fourth = Tier.finish third := rfl
    have starts₅ : Tier.start fifth = Tier.finish fourth := rfl
    have starts₆ : Tier.start sixth = Tier.finish fifth := rfl
    omega
  have sized := compose_sized setup packageWidths step cap control shape read head bound premises
    use.laid lookupsFit challengesFit cube ((by decide : 3 ≤ 40).trans coords) early fits
  have later := Key30.tower_suffix (member := member) (scan := scan) (reference := reference)
    (accept := accepts)
    (commit := commit) (value := value) (place := place)
    (commits := commits) (values := values) (places := places) (commit₁ := commit₁)
    (commit₂ := commit₂) (commit₃ := commit₃) (commit₄ := commit₄) (commit₅ := commit₅)
    (commit₆ := commit₆) (stateOf := stateOf) slots injective units sparse injective₁ units₁
    sparse₁ registeredS positive₁ injective₂ units₂ sparse₂ injective₃ units₃ sparse₃ injective₄
    units₄ sparse₄ injective₅ units₅ sparse₅ bounded₅ injective₆ units₆ sparse₆ bounded₆ next
    inside injectiveL unitsL boundedL (fun _ accepted => accepted.2)
  have error := Key30.towerError_le (member := member) (scan := scan) (reference := reference)
    (registry := registry) (β := β) (key := key) (openKey := openKey) (claim := claim)
    (block := block) (mass := mass) (β₁ := β₁) (key₁ := key₁) (openKey₁ := openKey₁)
    (claim₁ := claim₁) (block₁ := block₁) (mass₁ := mass₁) (β₂ := β₂) (key₂ := key₂)
    (openKey₂ := openKey₂) (claim₂ := claim₂) (block₂ := block₂) (weight₂ := weight₂)
    (β₃ := β₃) (key₃ := key₃) (openKey₃ := openKey₃) (claim₃ := claim₃) (block₃ := block₃)
    (weight₃ := weight₃) (key₄ := key₄) (openKey₄ := openKey₄) (claim₄ := claim₄)
    (block₄ := block₄) (weight₄ := weight₄) (key₅ := key₅) (openKey₅ := openKey₅)
    (claim₅ := claim₅) (block₅ := block₅) (weight₅ := weight₅) (key₆ := key₆)
    (openKey₆ := openKey₆) (claim₆ := claim₆) (block₆ := block₆) (weight₆ := weight₆)
    (read := Compose.read C) (start := Compose.root C vf) (flat := flat) chance chance₁ chance₂
    chance₃ chance₄ chance₅ chance₆ chanceL
  refine (final_sound setup packageWidths step cap control shape read head bound Sum.inl prover
    sized (fun _ accepted => accepted.1) (fun _ found => found) later registered covers bounded
    positive).trans ?_
  exact add_le_add (add_le_add (mul_le_mul_of_nonneg_left
    (add_le_add (le_refl (setup.runCost shape)) error) (by positivity : (0 : ℝ) ≤ queries + 1))
    le_rfl) le_rfl

end NightstreamFPrime.Export.BoundedDepth.FinalKey30
