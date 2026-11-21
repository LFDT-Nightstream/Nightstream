import NightstreamFPrime.Spec.Compression.Akita.Key20Complete
import NightstreamFPrime.Spec.Compression.Akita.Extract.RootComplete

/-! The honest prover of the opening of Akita for the schedule file
`crates/ns-compress/artifacts/key20.aks`: the batching root, the root fold of the key polynomial,
`W` and `V`, the packing level 1, the trace levels 2 to 5 and the terminal (Akita, ePrint
2026/1983, revision of 2026-10-15, Theorem 10.9 with Lemma 10.7, Figure 4 and Section 8.2).

- `root_folds`, `level₁_bounded`, `level₁_fold`, `root_cells`, `root_segments` and `lifts` are
  the conditions of `Extract.tower_complete` on the numbers of the root and of the level 1: the
  response base 8 of each group, the bounded source of the level 1, the tables of `2^20`, `2^16`
  and `2^16` cells, the segments `(r_pack, r_tail, r_pos, r_blk)` of the lengths (3, 7, 5, 5),
  (3, 6, 3, 4) and (4, 6, 3, 3) of the points of the groups, and the link from the level 1 to the
  level 2.
- `tower_complete` is `Extract.tower_complete` for these numbers: the verifier `towerLinks` of
  `Key20.tower_suffix` accepts the honest prover. The fit conditions `BatchFit`, `LiftFit` and
  `TraceFit` stay hypotheses on the responses of the honest prover. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key20

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- Each group of the root has the response base `b_z = 8` with the conditions `Radix`, and
`b_z ≤ b⋆`. -/
theorem root_folds : ∀ g, Radix (root.fold g).responseBase ∧
    (root.fold g).responseBase ≤ (root.shape g).certifiedBase := by
  have facts : ∀ g, (root.fold g).responseBase % 2 = 0 ∧ 2 ≤ (root.fold g).responseBase ∧
      (root.fold g).responseBase < fieldModulus ∧
      (root.fold g).responseBase ≤ (root.shape g).certifiedBase := by decide
  exact fun g => ⟨⟨(facts g).1, (facts g).2.1, (facts g).2.2.1⟩, (facts g).2.2.2⟩

/-- The shape of the level 1 satisfies the conditions `Shape.Bounded` of the commitment algorithm
with a bounded source. -/
theorem level₁_bounded : level₁.shape.Bounded :=
  ⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide,
    pathFits_of (by decide) (by decide)⟩

/-- The level 1 has `b_z = 8 ≤ b⋆`, the compression maps of its matrix `D` have a position for
each binary digit, and a message of its range tree has at most 6 field elements. -/
theorem level₁_fold : level₁.fold.responseBase ≤ level₁.shape.certifiedBase ∧
    PathFits level₁.opening.openDegree (1 * level₁.opening.openRank) level₁.opening.chain ∧
    rangeWidth level₁.shape.certifiedBase = 6 :=
  ⟨by decide, pathFits_of (by decide) (by decide), by decide⟩

/-- The blocks of the groups of the root have `2^20`, `2^16` and `2^16` cells. -/
theorem root_cells : ∀ g, (root.shape g).blockLength * (root.shape g).innerDegree *
    (root.shape g).blocks = 2 ^ ![20, 16, 16] g := by decide

/-- The segments `(r_pack, r_tail, r_pos, r_blk)` of the opening point of each group of the root
(`root_points`) have the sums 20, 16 and 16 of their lengths. -/
theorem root_segments : ∀ g : Fin 3,
    ![3, 3, 4] g + (![7, 6, 6] g + (![5, 3, 3] g + ![5, 4, 3] g)) = ![20, 16, 16] g := by decide

/-- The conditions of the link from the level 1 to the level 2: `μ_1 = ν_1 + 1`, and the live
blocks of the level 2 have each cell before the end of the quotient digits of the level 1, as
`2·B·M·(d_A/k)` coordinates of `Z_q` (Akita Section 4.5, (40) to (42)). -/
theorem lifts : level₁.vars = level₁.tail + 1 ∧
    level₁.opening.quotientRoom level₁.fold quotientDigits ≤
      2 * (level₂.shape.blocks * (level₂.shape.blockLength * (2 * 32))) := by decide

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {accept : Prefix Wire (Option Coord → E) total → Prop} {read : Reader Wire} {start : ℕ}
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
  {statementOf : Wire → TerminalStatement terminal E}
  {partialsOf : Wire → Fin terminal.blocks → Rq terminal.innerDegree}

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

local notation "links" =>
  towerLinks (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅)
    Route.coefficient (Route.euclidean norm₃.operator norm₃.limit)
    (Route.euclidean norm₄.operator norm₄.limit) (Route.euclidean norm₅.operator norm₅.limit)
    zeroth first second third fourth fifth L start read commit value place (And.left quarters)
    (And.left (And.right quarters)) (And.left (And.right (And.right quarters)))
    (And.left (And.right (And.right (And.right quarters))))
    (And.right (And.right (And.right (And.right quarters)))) commit₁ commit₂ commit₃ commit₄
    commit₅ stateOf

/-- Akita Theorem 10.9 for the opening of the schedule `key20.aks`, without the probability that
a response does not fit (`Extract.tower_complete`): the verifier `towerLinks` of
`Key20.tower_suffix` accepts the honest prover. For each table of the oracle whose slots decode to
fold challenges, and each transcript prefix of `start` moves on its path whose last message has
the commitment of a table of each group, a point of 20, 16 and 16 coordinates and the value of
the table at this point, later messages exist such that the complete transcript satisfies
`towerLinks`: the link `RootLinked` of the batching root, the link `Tier.Linked` from the root to
the level 1, the links `Tier.TraceLinked` of the levels 1 to 5, the link `Terminal.Linked` and
the checks `Terminal.Passed`. The conditions on the shapes are theorems of this file and of
`Key20Complete.lean`.

`sends`, `wide`, `capped`, `room`, `carriesR` to `carriesL`: a wire message can have the data of
the prover, with messages of 8 field elements, the energy `3029804899` and 4 messages of
the checks. `next`, `inside`, `reads`, `clears`: the stage of the terminal and its messages.
`counts`: the terminal reads the nonce of the response search with `read`. `live`: the table of
the key polynomial is 0 after `2^20` cells, and the tables of `W` and of `V` are 0 after `2^16`
cells. `committed`, `lengths`, `valued`: the last message of `early` has the
commitments `commit` of the tables, points of 20, 16 and 16 coordinates, and the values of the
tables at these points. `fit`: the fit conditions `BatchFit` of the root, `LiftFit` of the
level 1, `TraceFit` of the levels 2 to 5 with `S_max` of `norm₃` to `norm₅`, and
`‖z‖_2² ≤ S_max,term` at the terminal, for each fold challenge vector of the families. -/
theorem tower_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (wide : 8 ≤ width) (capped : 3029804899 ≤ cap)
    (room : 3 < slots) (carriesR : Carries sendable slots read claim)
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
    (counts : ∀ wire, Terminal.nonce L wire = read.nonce wire)
    (table : Prefix Wire (Option Coord → E) total → Option Coord → E)
    (decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input i, (decode₁ (table input (some (block₁ i)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodesL : ∀ input i, (Terminal.decode L (table input (some (Terminal.block L i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57)
    {wire₀ : Wire} {early : List ((Option Coord → E) × Wire)} (on : OnPath accept table wire₀ early)
    (starts : early.length = start) {tables : Fin 3 → Table}
    (live : ∀ g cell, 2 ^ ![20, 16, 16] g ≤ cell → tables g cell = 0)
    (committed : commit (lastWire wire₀ early) = fun g => (Akita.commit (key g) (tables g)).1)
    (lengths : ∀ g, (place (lastWire wire₀ early) g).length = ![20, 16, 16] g)
    (valued : value (lastWire wire₀ early) =
      fun g => mle (place (lastWire wire₀ early) g) (lifted E (tables g)))
    (fit : BatchFit β key openKey rootVars (And.left root_sized) 8 quotientDigits decode tables
        fun w₀ =>
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
      ∀ c : Fin terminal.blocks → Rq terminal.innerDegree,
        (∀ i, ∃ slot, Terminal.decode L slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource
          (And.right (And.right (And.right (And.right quarters)))) (packed w₅ : ℕ → E))) ≤
          normTerminal.limit) :
    ∃ later, OnPath accept table wire₀ (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Option Coord → E) total, stage.first = wire₀ →
        stage.rest = early ++ later → links stage := by
  have sums := root_segments
  have segment : ∀ g, ![3, 3, 4] g ≤ (place (lastWire wire₀ early) g).length ∧
      ![7, 6, 6] g ≤ (place (lastWire wire₀ early) g).length - ![3, 3, 4] g ∧
      ![5, 3, 3] g ≤ (place (lastWire wire₀ early) g).length - ![3, 3, 4] g - ![7, 6, 6] g ∧
      (place (lastWire wire₀ early) g).length - ![3, 3, 4] g - ![7, 6, 6] g - ![5, 3, 3] g =
        ![5, 4, 3] g := fun g => by
    have sum := sums g
    have length := lengths g
    omega
  exact Extract.tower_complete sends (by omega) room carriesR carries₀ carries₁ carries₂
    carries₃ carries₄ carries₅ carriesL (fun g => root_honest g) (fun g => (root_folds g).1)
    (fun g => (root_folds g).2) paths.1 root_sized.2.2.2.1 covers.1 covers.2
    ((by decide : rangeWidth 8 ≤ 6).trans (by omega))
    level₁_bounded covers.1 level₁_fold.1 level₁_fold.2.1 covers.1 covers.2
    (by rw [level₁_fold.2.2]; omega) quarters.1 quarters.2.1
    quarters.2.2.1 quarters.2.2.2.1 quarters.2.2.2.2 source₂ source₃ source₄ source₅ source₆
    (fold₂.mono wide) (fold₃.mono wide) (fold₄.mono wide) (fold₅.mono wide) points.1 points.2.1
    points.2.2.1 blocks_live.1.1 blocks_live.1.2.symm lifts.1 Key20.links.1.1 Key20.links.1.2.1
    Key20.links.1.2.2.1 Key20.links.1.2.2.2 blocks_live.2.1.2.symm Key20.links.2.1.1
    Key20.links.2.1.2.1 Key20.links.2.1.2.2.1 Key20.links.2.1.2.2.2 live_cells.1 lifts.2
    Key20.links.2.2.1 Key20.links.2.2.2.1 Key20.links.2.2.2.2.1 Key20.links.2.2.2.2.2
    (norms_sized (level₃, norm₃) (by simp)).1 (norms_sized (level₄, norm₄) (by simp)).1
    (norms_sized (level₅, norm₅) (by simp)).1
    ((by decide : rangeWidthSq level₃.shape.certifiedBase ≤ 8).trans wide)
    ((by decide : rangeWidthSq level₄.shape.certifiedBase ≤ 8).trans wide)
    ((by decide : rangeWidthSq level₅.shape.certifiedBase ≤ 8).trans wide)
    ((by decide : norm₃.limit ≤ 3029804899).trans capped)
    ((by decide : norm₄.limit ≤ 3029804899).trans capped)
    ((by decide : norm₅.limit ≤ 3029804899).trans capped) next inside reads clears counts table
    decodes decodes₁ decodes₂ decodes₃ decodes₄ decodes₅ decodesL worked on starts
    (blockOf := fun g => (((place (lastWire wire₀ early) g).drop (![3, 3, 4] g)).drop
      (![7, 6, 6] g)).drop (![5, 3, 3] g))
    (positionOf := fun g => (((place (lastWire wire₀ early) g).drop (![3, 3, 4] g)).drop
      (![7, 6, 6] g)).take (![5, 3, 3] g))
    (packOf := fun g => (place (lastWire wire₀ early) g).take (![3, 3, 4] g))
    (tailOf := fun g => ((place (lastWire wire₀ early) g).drop (![3, 3, 4] g)).take
      (![7, 6, 6] g))
    (fun g => (root_points g).1.le.trans_eq (congrArg (2 ^ ·) (by
      simp only [List.length_drop]
      exact (segment g).2.2.2.symm)))
    (fun g => (root_points g).2.1.trans (congrArg (2 ^ ·) (by
      simp only [List.length_take, List.length_drop]
      exact (Nat.min_eq_left (segment g).2.2.1).symm)))
    (fun g => (root_points g).2.2.1.trans (congrArg (2 ^ ·) (by
      simp only [List.length_take]
      exact (Nat.min_eq_left (segment g).1).symm)))
    (fun g => (root_points g).2.2.2.trans (congrArg (2 ^ ·) (by
      simp only [List.length_take, List.length_drop]
      exact (Nat.min_eq_left (segment g).2.1).symm)))
    (fun g cell large => live g cell ((root_cells g).symm.trans_le large)) committed
    (funext fun g => by simp only [List.take_append_drop]) valued fit

end NightstreamFPrime.Spec.Compression.Akita.Key20
