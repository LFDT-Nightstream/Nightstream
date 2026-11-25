import NightstreamFPrime.Spec.Compression.Akita.Key30Root
import NightstreamFPrime.Spec.Compression.Akita.Key30Opening
import NightstreamFPrime.Spec.Compression.Akita.Extract.Page
import NightstreamFPrime.Spec.Compression.Akita.Extract.ZeroFit
import NightstreamFPrime.Spec.Compression.Akita.Extract.GroupComplete

/-! The verifier `offLinks` of the opening of Akita for a row of the schedule file
`crates/ns-compress/artifacts/key30.aks` accepts a proof (Akita, ePrint 2026/1983, revision of
2026-11-23, Theorem 10.9, Section 9 with (136), Figure 4 and Section 8.2). The theorem
`Key30.tower_suffix` bounds an event of this verifier, and `Key30.tower_complete` has hypotheses
on the wire messages, on the setup prefix and on the responses of the honest prover. This file
has no such hypothesis.

- `Note` has the data of a wire message other than the messages of the checks: the commitments,
  the points and the values of the three claims of the root, the claim of the root fold, the
  commitment of the witness of level 1 with the commitments, the values and the points of the
  two claims of level 1, the claim of level 1, the commitment and the claim of each level from
  level 2, and the state, the statement, the clear partials and the clear response of the
  terminal. The points are vectors of 30, `vars` and 26 coordinates for the root and of 23 and 28
  coordinates for level 1, and `Note.place` and `Note.places` read them as lists. `Wire` is
  `Page` for this data, with 4 cells of 8 elements and a number up to `3029804899`.
  `Note.finite`: a note is finite, so `Wire r E` is finite for a finite `E`.
- `terminalOf` is the terminal that reads its statement and its clear partials from the message
  before its fold challenges.
- `Fit` has the fit conditions of the honest prover (Akita Definition 10.5) of each level for a
  setup prefix and the tables of the key polynomial, `W` and `V`: for each opening point and each
  fold challenge vector that the decoders give, the response of each level satisfies `Enc` and
  the norm limit of its route.
- `accepts`: for each setup prefix, each family of tables with `Fit` and each table of the oracle
  whose slots decode to fold challenges, a proof exists whose transcript satisfies `offLinks`.
  It is the proof of the honest prover. The matrices, the decoders of the fold challenges, the
  coordinates of their slots and the split `scan` of the public weight of the root are free.
- `fit_zero` and `fit_quiet` give `Fit` in two cases: for the zero setup prefix and the zero
  tables with each decoder, and for each setup prefix and each family of tables with decoders
  that give the fold challenge 0. For a fold challenge vector of the families of the schedule,
  the response of a table that is not 0 can be larger than the limit of `Enc`, so `Fit` for
  these families is a hypothesis: the honest prover of the Akita code has a search over a nonce
  for a response that fits (`crates/akita-prover/src/protocol/fold_grind.rs`).
- `noteTower` is the data `Key30.Tower` for the wire messages `Wire`: the readers of `Note`, the
  setup prefix with the commitment and the decommitment of `commit`, and the terminal
  `terminalOf` with the decoder of `Tower.Drawn`. `noteTower_accepts` is `accepts` for the
  decoders of the code of `Tower.accepts` and the split `Tower.scan`: the proof of the honest prover
  is a prover with no queries of the game of `Tower.extraction_le`, and `Tower.accepts` accepts
  its transcript. `noteTower_zero` and `noteTower_played` give such a prover for the zero setup
  prefix and the zero tables, with no hypothesis on the fit. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key30

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The data of a wire message of the opening of the row `r` other than the messages of the
checks (Akita Figure 4 and (136): the messages of the prover). The points of the three claims of
the root are vectors of 30, `vars` and 26 coordinates, and the points of the two claims of level 1
are vectors of 23 and 28 coordinates. -/
structure Note (r : Row) (E : Type) [CommRing E] [Algebra Zq E] where
  commit : (g : Fin 3) → ((root r).shape g).Payload
  value : Fin 3 → E
  point : (g : Fin 3) → Fin (![30, r.vars, 26] g) → E
  claim : (root r).Claim E
  commit₁ : ((level₁ r).shape 1).Payload
  commits : (g : Fin 2) → ((level₁ r).shape g).Payload
  values : Fin 2 → E
  points : (g : Fin 2) → Fin (![setupVars, rootVars] g) → E
  claim₁ : (level₁ r).Claim E
  commit₂ : (level₂ r).shape.Payload
  claim₂ : Claim (level₂ r).opening E
  commit₃ : (level₃ r).shape.Payload
  claim₃ : Claim (level₃ r).opening E
  commit₄ : level₄.shape.Payload
  claim₄ : TraceClaim level₄.opening E
  commit₅ : level₅.shape.Payload
  claim₅ : TraceClaim level₅.opening E
  commit₆ : level₆.shape.Payload
  claim₆ : TraceClaim level₆.opening E
  state : Fin terminal.blocks → Fin terminal.innerRank → Rq terminal.innerDegree
  statement : TerminalStatement terminal E
  partials : Fin terminal.blocks → Rq terminal.innerDegree
  response : terminal.Source

/-- The points of the three claims of the root of a note as lists. -/
def Note.place {r : Row} {E : Type} [CommRing E] [Algebra Zq E] (note : Note r E) (g : Fin 3) :
    List E :=
  List.ofFn (note.point g)

/-- The points of the two claims of level 1 of a note as lists. -/
def Note.places {r : Row} {E : Type} [CommRing E] [Algebra Zq E] (note : Note r E) (g : Fin 2) :
    List E :=
  List.ofFn (note.points g)

instance Note.finite {r : Row} {E : Type} [CommRing E] [Algebra Zq E] [Finite E] :
    Finite (Note r E) :=
  Finite.of_injective (fun note : Note r E => (note.commit, note.value, note.point, note.claim,
    note.commit₁, note.commits, note.values, note.points, note.claim₁, note.commit₂, note.claim₂,
    note.commit₃, note.claim₃, note.commit₄, note.claim₄, note.commit₅, note.claim₅,
    note.commit₆, note.claim₆, note.state, note.statement, note.partials, note.response))
    fun left right same => by
      cases left
      cases right
      simp_all

/-- A wire message of the opening of the row `r`: the data `Note`, 4 cells of 8 elements of `E`
and a number up to `3029804899` for the messages of the checks, and the nonce of the response
search. -/
abbrev Wire (r : Row) (E : Type) [CommRing E] [Algebra Zq E] : Type :=
  Page (Note r E) E 4 8 3029804899

variable {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E]
  [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The terminal with the matrix `A_term`, the fold challenges at the stage `start`, and the
statement, the clear partials and the nonce of the response search of the message before this
stage (Akita Section 8.2). -/
def terminalOf (r : Row) (total : ℕ)
    (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
      (Rq terminal.innerDegree))
    (block : Fin terminal.blocks → Coord) (decode : Slot → Option (Rq terminal.innerDegree))
    (start : ℕ) :
    Terminal terminal E (Wire r E) Coord Slot total
      (.euclidean normTerminal.operator normTerminal.limit) where
  matrix := matrix
  read point := (point.wire start).data.statement
  partials point := (point.wire start).data.partials
  response wire := wire.data.response
  nonce wire := wire.nonce
  block := block
  decode := decode
  weight := 0
  bound := 0
  start := start

section Generic

variable {r : Row} (member : r ∈ rows) (flat : Table)
  (β : ∀ g, Module.Basis (Fin ((root r).planes g)) Zq E)
  (key : (g : Fin 3) → Key ((root r).shape g)) (openKey : (root r).OpenKey)
  (block : (root r).Block → Coord)
  (decode : (g : Fin 3) → Slot → Option (Rq ((root r).carrierDegree g))) (mass : Fin 3 → ℕ)
  (scan : ∀ (wire : Wire r E) (challenge : (root r).Challenge),
    SetupScan (Batch.liftedReduction β ((wire.data.claim).statement key openKey) challenge
      rootVars (root_sized r member).1 8 quotientDigits₀ (root_sized r member).2.1 rootBatch
      (root_sized r member).2.2.1) setupVars (Shape.pad (Batch.shape (level₁ r) 0) flat))
  (β₁ : ∀ g, Module.Basis (Fin ((level₁ r).planes g)) Zq E)
  (key₁ : (g : Fin 2) → Key ((level₁ r).shape g)) (openKey₁ : (level₁ r).OpenKey)
  (block₁ : (level₁ r).Block → Coord)
  (decode₁ : (g : Fin 2) → Slot → Option (Rq ((level₁ r).carrierDegree g))) (mass₁ : Fin 2 → ℕ)
  (β₂ : Module.Basis (Fin (level₂ r).opening.planes) Zq E) (key₂ : Key (level₂ r).shape)
  (openKey₂ : OpeningKey (level₂ r).opening) (block₂ : Fin (level₂ r).shape.blocks → Coord)
  (decode₂ : Slot → Option (Rq (level₂ r).opening.carrierDegree)) (weight₂ : ℕ)
  (β₃ : Module.Basis (Fin (level₃ r).opening.planes) Zq E) (key₃ : Key (level₃ r).shape)
  (openKey₃ : OpeningKey (level₃ r).opening) (block₃ : Fin (level₃ r).shape.blocks → Coord)
  (decode₃ : Slot → Option (Rq (level₃ r).opening.carrierDegree)) (weight₃ : ℕ)
  (key₄ : Key level₄.shape) (openKey₄ : OpeningKey level₄.opening)
  (block₄ : Fin level₄.shape.blocks → Coord)
  (decode₄ : Slot → Option (Rq level₄.shape.innerDegree)) (weight₄ : ℕ)
  (key₅ : Key level₅.shape) (openKey₅ : OpeningKey level₅.opening)
  (block₅ : Fin level₅.shape.blocks → Coord)
  (decode₅ : Slot → Option (Rq level₅.shape.innerDegree)) (weight₅ : ℕ)
  (key₆ : Key level₆.shape) (openKey₆ : OpeningKey level₆.opening)
  (block₆ : Fin level₆.shape.blocks → Coord)
  (decode₆ : Slot → Option (Rq level₆.shape.innerDegree)) (weight₆ : ℕ)
  (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
    (Rq terminal.innerDegree))
  (blockL : Fin terminal.blocks → Coord) (decodeL : Slot → Option (Rq terminal.innerDegree))

local notation "zeroth" =>
  offBatch β key openKey (Note.claim ∘ Page.data : Wire r E → _) rootVars
    (And.left (root_sized r member)) 8 quotientDigits₀
    (And.left (And.right (root_sized r member))) rootBatch
    (And.left (And.right (And.right (root_sized r member)))) block decode Page.read mass (0 + 1)
    setupVars (Shape.pad (Batch.shape (level₁ r) 0) flat) scan

local notation "first" =>
  Tier.pinned (liftBatch β₁ key₁ openKey₁ (Note.claim₁ ∘ Page.data : Wire r E → _) vars₁
    (And.left (level₁_sized r member)) 16 quotientDigits
    (And.left (And.right (level₁_sized r member))) batch₁
    (And.left (And.right (And.right (level₁_sized r member)))) block₁ decode₁ Page.read mass₁
    (Tier.finish zeroth + 1)) (Batch.Pinned 0 (Prod.fst (commit (key₁ 0) flat))
      (Shape.pad (Batch.shape (level₁ r) 0) flat))
    (fun ω _ => Batch.pinBreak key₁ openKey₁ _ 0 (Prod.snd (commit (key₁ 0) flat)) ω)

local notation "second" =>
  liftFold β₂ key₂ openKey₂ (Note.claim₂ ∘ Page.data : Wire r E → _) (Key20.Level.vars (level₂ r))
    (And.left (level₂_sized r member)) quotientDigits
    (And.left (And.right (level₂_sized r member))) (Key20.Level.batch (level₂ r))
    (And.left (And.right (And.right (level₂_sized r member)))) block₂ decode₂ Page.read weight₂
    (Tier.finish first)

local notation "third" =>
  Tier.tensored (liftFold β₃ key₃ openKey₃ (Note.claim₃ ∘ Page.data : Wire r E → _) (Key20.Level.vars (level₃ r))
    (And.left (level₃_sized r member)) quotientDigits (And.left (And.right (level₃_sized r member))) (Key20.Level.batch (level₃ r))
    (And.left (And.right (And.right (level₃_sized r member)))) block₃ decode₃ Page.read weight₃
    (Tier.finish second)) (Key20.Level.tail (level₃ r))

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := level₄.fold) key₄ openKey₄
    (Note.claim₄ ∘ Page.data : Wire r E → _) level₄.vars (And.left level₄_sized) level₄.batch
    (And.left (And.right level₄_sized)) block₄ decode₄ Page.read weight₄ (Tier.finish third))
    level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅
    (Note.claim₅ ∘ Page.data : Wire r E → _) level₅.vars
    (And.left (tails_sized level₅ (List.mem_cons_self ..))) level₅.batch
    (And.left (And.right (tails_sized level₅ (List.mem_cons_self ..)))) block₅ decode₅
    Page.read weight₅ (Key20.Squared.operator (norm₅ r)) (Key20.Squared.limit (norm₅ r)) (Tier.finish fourth)) level₅.tail

local notation "sixth" =>
  Tier.tensored (committedTraceSq (fold := level₆.fold) key₆ openKey₆
    (Note.claim₆ ∘ Page.data : Wire r E → _) level₆.vars
    (And.left (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))) level₆.batch
    (And.left (And.right (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))))
    block₆ decode₆ Page.read weight₆ norm₆.operator norm₆.limit (Tier.finish fifth))
    level₆.tail

/-- The fit conditions of the honest prover of the opening of the row `r` (Akita Definition 10.5
and Lemma 10.7) for the setup prefix `flat` and the tables `tables` of the key polynomial, `W` and
`V`: the conditions `BatchFit` of the root, `StateFit` of level 1 for the decommitment of the
setup prefix of the Lean commitment algorithm and the bounded source of the witness, `LiftFit` of
the levels 2 and 3, `TraceFit` of the levels 4 to 6 with `S_max` of `(norm₅ r)` and `norm₆`, and
`‖z‖_2² ≤ S_max,term` at the terminal. Each condition is for each opening point and each fold
challenge vector that the decoders give. -/
def Fit (tables : Fin 3 → Table) : Prop :=
  BatchFit β key openKey rootVars (And.left (root_sized r member)) 8 quotientDigits₀ decode tables
    fun w₀ =>
  StateFit β₁ key₁ openKey₁ vars₁ (And.left (level₁_sized r member)) 16 quotientDigits decode₁
    (offState (key₁ 1) (commit (key₁ 0) flat).1 (commit (key₁ 0) flat).2 w₀) fun w₁ =>
  LiftFit β₂ key₂ openKey₂ (level₂ r).fold (level₂ r).vars (And.left (level₂_sized r member))
    quotientDigits decode₂ (fun i => (level₂ r).shape.embed ((level₂ r).shape.pack w₁ i))
    fun w₂ =>
  LiftFit β₃ key₃ openKey₃ (level₃ r).fold (level₃ r).vars (And.left (level₃_sized r member)) quotientDigits
    decode₃ (fun i => (level₃ r).shape.embed ((level₃ r).shape.pack w₂ i)) fun w₃ =>
  TraceFit (E := E) (And.left quarters) key₄ openKey₄ level₄.fold (fun _ => decode₄) (fun _ => True)
    (packed w₃) fun w₄ =>
  TraceFit (E := E) (And.left (And.right quarters)) key₅ openKey₅ level₅.fold (fun _ => decode₅)
    (fun z => vecNormSq z ≤ (norm₅ r).limit) (packed w₄) fun w₅ =>
  TraceFit (E := E) (And.left (And.right (And.right quarters))) key₆ openKey₆ level₆.fold
    (fun _ => decode₆) (fun z => vecNormSq z ≤ norm₆.limit) (packed w₅) fun w₆ =>
  ∀ c : Fin terminal.blocks → Rq terminal.innerDegree,
    (∀ i, ∃ slot, decodeL slot = some (c i)) →
    vecNormSq (traceResponse c (traceSource (And.right (And.right (And.right quarters)))
      (packed w₆ : ℕ → E))) ≤ normTerminal.limit

omit [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
  [Nonempty Slot] in
/-- The fit conditions `Fit` hold for the zero setup prefix and the zero tables, with each
decoder: each response is 0 (`Extract/ZeroFit.lean`, `stateFit_zero`). -/
theorem fit_zero :
    Fit member 0 β key openKey decode β₁ key₁ openKey₁ decode₁ β₂ key₂ openKey₂ decode₂ β₃ key₃
      openKey₃ decode₃ key₄ openKey₄ decode₄ key₅ openKey₅ decode₅ key₆ openKey₆ decode₆ decodeL
      fun _ => 0 := by
  have openings := level₁_openings r member
  have opens : ∀ g, ∃ table, (offState (key₁ 1) (commit (key₁ 0) 0).1 (commit (key₁ 0) 0).2 0
      g).2.Opens (key₁ g) (offState (key₁ 1) (commit (key₁ 0) 0).1 (commit (key₁ 0) 0).2 0 g).1
      table := Fin.forall_fin_two.mpr
    ⟨⟨_, commit_opens honest.2 (key₁ 0) 0⟩,
      ⟨_, commitSource_opens (level₁_bounded r member) (key₁ 1) _⟩⟩
  have zero : ∀ g, (offState (key₁ 1) (commit (key₁ 0) 0).1 (commit (key₁ 0) 0).2 0
        g).2.source = 0 ∧
      (offState (key₁ 1) (commit (key₁ 0) 0).1 (commit (key₁ 0) 0).2 0 g).2.inner = 0 ∧
      PathDigits.firstTable ((level₁ r).shape g).chain
        (offState (key₁ 1) (commit (key₁ 0) 0).1 (commit (key₁ 0) 0).2 0 g).2.chain = 0 ∧
      PathDigits.secondTable ((level₁ r).shape g).chain
        (offState (key₁ 1) (commit (key₁ 0) 0).1 (commit (key₁ 0) 0).2 0 g).2.chain = 0 := by
    refine Fin.forall_fin_two.mpr ⟨commit_zero (by decide : 2 ≤ 2 ^ 8) (by decide : 2 ≤ 16)
      (key₁ 0), ?_⟩
    dsimp only [offState, Batch.pairOf_one]
    rw [Shape.embed_pack_zero]
    exact ⟨rfl, commitSource_zero (by decide : 2 ≤ 16) (key₁ 1)⟩
  exact batchFit_zero (honest.1 r member) (fun g => (root_folds r member g).1)
    (fun g => (root_folds r member g).2) (root_path r member) (by decide)
    (stateFit_zero (fun g => (openings g).1) (fun g => (openings g).2)
      (fun g => (level₁_folds r member g).1) (fun g => (level₁_folds r member g).2)
      (level₁_path r member) opens zero (by decide)
      (liftFit_zero (lifts r member).1.1 (lifts r member).1.2.1 (lifts r member).1.2.2.1
        (lifts r member).1.2.2.2 (by decide : 2 ≤ 16)
        (liftFit_zero (lifts r member).2.1 (lifts r member).2.2.1 (lifts r member).2.2.2.1 (lifts r member).2.2.2.2 (by decide : 2 ≤ 16)
          (traceFit_zero quarters.1 (by decide) (by decide) trivial
            (traceFit_zero quarters.2.1 (by decide) (by decide)
              (by rw [vecNormSq_zero]; exact Nat.zero_le _)
              (traceFit_zero quarters.2.2.1 (by decide) (by decide)
                (by rw [vecNormSq_zero]; exact Nat.zero_le _) fun c _ => by
                  rw [packed_zero, traceSource_zero, traceResponse_zero, vecNormSq_zero]
                  exact Nat.zero_le _))))))

omit [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
  [Nonempty Slot] in
/-- The fit conditions `Fit` hold for each setup prefix and each family of tables when each
decoder gives only the fold challenge 0: each response is 0. So the hypothesis `fit` of `accepts`
holds for a setup prefix and tables that are not 0. -/
theorem fit_quiet (tables : Fin 3 → Table) :
    Fit (E := E) (Slot := Slot) member flat β key openKey (fun _ _ => some 0) β₁ key₁ openKey₁
      (fun _ _ => some 0) β₂ key₂ openKey₂ (fun _ => some 0) β₃ key₃ openKey₃ (fun _ => some 0)
      key₄ openKey₄ (fun _ => some 0) key₅ openKey₅ (fun _ => some 0) key₆ openKey₆
      (fun _ => some 0) (fun _ => some 0) tables :=
  batchFit_quiet (fun g => (root_folds r member g).1.two)
    (fun _ _ _ decoded => (Option.some.inj decoded).symm) fun _ _ =>
  stateFit_quiet (fun g => (level₁_folds r member g).1.two)
    (fun _ _ _ decoded => (Option.some.inj decoded).symm) fun _ _ =>
  liftFit_quiet (lifts r member).1.2.1.two (fun _ _ decoded => (Option.some.inj decoded).symm)
    fun _ =>
  liftFit_quiet (lifts r member).2.2.1.two (fun _ _ decoded => (Option.some.inj decoded).symm) fun _ =>
  traceFit_quiet quarters.1 (by decide) (fun _ _ _ decoded => (Option.some.inj decoded).symm)
    trivial fun _ =>
  traceFit_quiet quarters.2.1 (by decide) (fun _ _ _ decoded => (Option.some.inj decoded).symm)
    (by rw [vecNormSq_zero]; exact Nat.zero_le _) fun _ =>
  traceFit_quiet quarters.2.2.1 (by decide)
    (fun _ _ _ decoded => (Option.some.inj decoded).symm)
    (by rw [vecNormSq_zero]; exact Nat.zero_le _) fun _ c decodable => by
      obtain rfl : c = fun _ => 0 := funext fun i => by
        obtain ⟨slot, decoded⟩ := decodable i
        exact (Option.some.inj decoded).symm
      rw [traceResponse_quiet, vecNormSq_zero]
      exact Nat.zero_le _

omit [FieldSlot Slot E] in
/-- The verifier `offLinks` of the opening of a row of the schedule `key30.aks` accepts the proof
of the honest prover (Akita Theorem 10.9, Section 9 with (136)) for each setup prefix and each
family of tables with the fit conditions `Fit`. The wire messages are `Wire`, the batching root
is at the stage 0, the setup prefix `S^♭` is the table `flat` on its `2^23` cells with the
commitment and the reference decommitment of `commit`, and the terminal is `terminalOf` at the
stage after the level 6. `tables`: the tables of the key polynomial, `W` and `V`, with opening
points `places`. For each matrix of each level, each decoder of the fold challenges, each split
`scan` of the public weight of the root (Akita (132)), each table of the oracle whose slots
decode to fold challenges and each acceptance condition of the transcript protocol, a proof
exists whose first message has the commitments of the tables, the points and the values of the
tables at the points, and whose complete transcript satisfies `offLinks`: the link `RootLinked`
of the batching root, the link `Tier.Offloaded` from the root fold to the two claims of level 1,
the link `RootLinked` of their batching stage, the links `Tier.Linked` of the levels 1 to 3, the
links `Tier.TraceLinked` of the levels 3 to 6, the link `Terminal.Linked` and the checks
`Terminal.Passed` (`Key30.tower_complete`). `inside`: the fold challenge of the terminal is a
stage of the transcript. `worked`: each work slot of the table has 57 zero work bits. `live`: the
tables are 0 after `2^30`, `2^vars` and `2^26` cells. `lengths`: the points have 30, `vars` and 26
coordinates. `fit`: the fit conditions `Fit`, which `fit_zero` and `fit_quiet` give in two
cases. -/
theorem accepts {total : ℕ} (inside : Tier.finish sixth < total)
    (accept : Prefix (Wire r E) (Option Coord → Slot) total → Prop)
    (table : Address (Wire r E) total → Option Coord → Slot)
    (decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input b, (decode₁ b.1 (table input (some (block₁ b)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodes₆ : ∀ input i, (decode₆ (table input (some (block₆ i)))).isSome)
    (decodesL : ∀ input i, (decodeL (table input (some (blockL i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros Slot 57) {tables : Fin 3 → Table}
    (live : ∀ g cell, 2 ^ ![30, r.vars, 26] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E} (lengths : ∀ g, (places g).length = ![30, r.vars, 26] g)
    (fit : Fit member flat β key openKey decode β₁ key₁ openKey₁ decode₁ β₂ key₂ openKey₂ decode₂ β₃ key₃
      openKey₃ decode₃ key₄ openKey₄ decode₄ key₅ openKey₅ decode₅ key₆ openKey₆ decode₆ decodeL tables) :
    ∃ proof : ℕ → Wire r E,
      (proof 0).data.commit = (fun g => (commit (key g) (tables g)).1) ∧
      (proof 0).data.place = places ∧
      (proof 0).data.value = (fun g => mle (places g) (lifted E (tables g))) ∧
      offLinks (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
        (route₅ := Route.euclidean (norm₅ r).operator (norm₅ r).limit)
        (route₆ := Route.euclidean norm₆.operator norm₆.limit) zeroth first second third fourth
        fifth sixth (terminalOf r total matrix blockL decodeL (Tier.finish sixth)) 0 Page.read
        (Note.commit ∘ Page.data : Wire r E → _) (Note.value ∘ Page.data : Wire r E → _)
        (Note.place ∘ Page.data : Wire r E → _) (Note.commits ∘ Page.data : Wire r E → _)
        (Note.values ∘ Page.data : Wire r E → _) (Note.places ∘ Page.data : Wire r E → _)
        (commit (key₁ 0) flat).1 (Note.commit₁ ∘ Page.data : Wire r E → _)
        (Note.commit₂ ∘ Page.data : Wire r E → _) (Note.commit₃ ∘ Page.data : Wire r E → _)
        (And.left quarters) (And.left (And.right quarters))
        (And.left (And.right (And.right quarters))) (And.right (And.right (And.right quarters)))
        (Note.commit₄ ∘ Page.data : Wire r E → _) (Note.commit₅ ∘ Page.data : Wire r E → _)
        (Note.commit₆ ∘ Page.data : Wire r E → _) (Note.state ∘ Page.data : Wire r E → _)
        ((transcript total accept).final (.done proof) table) := by
  simp only [final_done]
  set table := fun input : Prefix (Wire r E) (Option Coord → Slot) total => table input.address
  replace decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome :=
    fun input => decodes input.address
  replace decodes₁ : ∀ input b, (decode₁ b.1 (table input (some (block₁ b)))).isSome :=
    fun input => decodes₁ input.address
  replace decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome :=
    fun input => decodes₂ input.address
  replace decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome :=
    fun input => decodes₃ input.address
  replace decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome :=
    fun input => decodes₄ input.address
  replace decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome :=
    fun input => decodes₅ input.address
  replace decodes₆ : ∀ input i, (decode₆ (table input (some (block₆ i)))).isSome :=
    fun input => decodes₆ input.address
  replace decodesL : ∀ input i, (decodeL (table input (some (blockL i)))).isSome :=
    fun input => decodesL input.address
  replace worked : ∀ input, table input none ∈ workZeros Slot 57 := fun input => worked input.address
  clear_value table
  obtain ⟨start, started⟩ : ∃ start, start = Tier.finish sixth := ⟨_, rfl⟩
  rw [← started] at inside ⊢
  obtain ⟨base, based⟩ : ∃ base : Note r E, base = {
      commit := fun g => (commit (key g) (tables g)).1
      value := fun g => mle (places g) (lifted E (tables g))
      point := fun g index => (places g).getD index 0
      claim := Batch.honestClaim β key openKey (fun _ => 0)
        (fun g => Point.ofSegments ((root r).opening g) [] [] [] []) 0
      commit₁ := (commitSource (key₁ 1) 0).1
      commits := fun g => (commit (key₁ g) 0).1
      values := 0
      points := fun _ _ => 0
      claim₁ := Batch.honestClaim β₁ key₁ openKey₁ (fun _ => 0)
        (fun g => Point.ofSegments ((level₁ r).opening g) [] [] [] []) 0
      commit₂ := (commitSource key₂ 0).1
      claim₂ := sourceClaim β₂ key₂ openKey₂ 0 (Point.ofSegments (level₂ r).opening [] [] [] [])
      commit₃ := (commitSource key₃ 0).1
      claim₃ := sourceClaim β₃ key₃ openKey₃ 0 (Point.ofSegments (level₃ r).opening [] [] [] [])
      commit₄ := (commitSource key₄ 0).1
      claim₄ := honestTraceClaim quarters.1 key₄ openKey₄ 0 [] [] [] 0
      commit₅ := (commitSource key₅ 0).1
      claim₅ := honestTraceClaim quarters.2.1 key₅ openKey₅ 0 [] [] [] 0
      commit₆ := (commitSource key₆ 0).1
      claim₆ := honestTraceClaim quarters.2.2.1 key₆ openKey₆ 0 [] [] [] 0
      state := 0
      statement := TerminalStatement.ofPoint quarters.2.2.2 matrix 0 [] [] [] 0 0
      partials := 0
      response := 0 } := ⟨_, rfl⟩
  have placed : base.place = places := funext fun g => by
    rw [based]
    exact ofFn_getD (lengths g) 0
  have registered : Batch.Registered (key₁ 0) (commit (key₁ 0) flat).1
      (Shape.pad (Batch.shape (level₁ r) 0) flat) (commit (key₁ 0) flat).2 :=
    Batch.registered_commit (S := (level₁ r).shape 0) honest.2 (key₁ 0) flat
  obtain ⟨later, on, whole, each⟩ := tower_complete (member := member) (β := β) (key := key)
    (openKey := openKey) (block := block) (decode := decode) (mass := mass) (scan := scan)
    (β₁ := β₁) (key₁ := key₁) (openKey₁ := openKey₁) (block₁ := block₁) (decode₁ := decode₁)
    (mass₁ := mass₁) (β₂ := β₂) (key₂ := key₂) (openKey₂ := openKey₂) (block₂ := block₂)
    (decode₂ := decode₂) (weight₂ := weight₂) (β₃ := β₃) (key₃ := key₃) (openKey₃ := openKey₃)
    (block₃ := block₃) (decode₃ := decode₃) (weight₃ := weight₃) (key₄ := key₄)
    (openKey₄ := openKey₄) (block₄ := block₄) (decode₄ := decode₄) (weight₄ := weight₄)
    (key₅ := key₅) (openKey₅ := openKey₅) (block₅ := block₅) (decode₅ := decode₅)
    (weight₅ := weight₅) (key₆ := key₆) (openKey₆ := openKey₆) (block₆ := block₆)
    (decode₆ := decode₆) (weight₆ := weight₆)
    (L := terminalOf r total matrix blockL decodeL start) (accept := accept)
    (read := Page.read) (claim := Note.claim ∘ Page.data) (claim₁ := Note.claim₁ ∘ Page.data)
    (claim₂ := Note.claim₂ ∘ Page.data) (claim₃ := Note.claim₃ ∘ Page.data)
    (claim₄ := Note.claim₄ ∘ Page.data) (claim₅ := Note.claim₅ ∘ Page.data)
    (claim₆ := Note.claim₆ ∘ Page.data)
    (commit := Note.commit ∘ Page.data) (value := Note.value ∘ Page.data)
    (place := Note.place ∘ Page.data) (commits := Note.commits ∘ Page.data)
    (values := Note.values ∘ Page.data) (places := Note.places ∘ Page.data)
    (spots := Note.points ∘ Page.data)
    (commit₁ := Note.commit₁ ∘ Page.data) (commit₂ := Note.commit₂ ∘ Page.data)
    (commit₃ := Note.commit₃ ∘ Page.data) (commit₄ := Note.commit₄ ∘ Page.data)
    (commit₅ := Note.commit₅ ∘ Page.data) (commit₆ := Note.commit₆ ∘ Page.data)
    (stateOf := Note.state ∘ Page.data) (statementOf := Note.statement ∘ Page.data)
    (partialsOf := Note.partials ∘ Page.data)
    (Page.sends (by decide)) le_rfl le_rfl (by decide)
    (Page.carries Note.claim fun target => ⟨{ base with claim := target }, rfl⟩)
    (fun _ => rfl)
    (Page.carries (fun note : Note r E => (note.commit₁, note.commits, note.values, note.points))
      fun target => ⟨{ base with
        commit₁ := target.1, commits := target.2.1, values := target.2.2.1,
        points := target.2.2.2 }, rfl⟩)
    (Page.carries Note.claim₁ fun target => ⟨{ base with claim₁ := target }, rfl⟩)
    (Page.carries (fun note : Note r E => (note.commit₂, note.claim₂)) fun target =>
      ⟨{ base with commit₂ := target.1, claim₂ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note r E => (note.commit₃, note.claim₃)) fun target =>
      ⟨{ base with commit₃ := target.1, claim₃ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note r E => (note.commit₄, note.claim₄)) fun target =>
      ⟨{ base with commit₄ := target.1, claim₄ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note r E => (note.commit₅, note.claim₅)) fun target =>
      ⟨{ base with commit₅ := target.1, claim₅ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note r E => (note.commit₆, note.claim₆)) fun target =>
      ⟨{ base with commit₆ := target.1, claim₆ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note r E => (note.state, note.statement, note.partials)) fun target =>
      ⟨{ base with state := target.1, statement := target.2.1, partials := target.2.2 }, rfl⟩)
    (Page.carries Note.response fun target => ⟨{ base with response := target }, rfl⟩)
    registered started inside (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) table decodes decodes₁
    decodes₂ decodes₃ decodes₄ decodes₅ decodes₆ decodesL worked
    (onPath_nil accept table (⟨base, fun _ => (fun _ => 0, 0), 0⟩ : Wire r E)) rfl
    (tables := tables) live (by rw [based]; rfl)
    (fun g => (congrArg List.length (congrFun placed g)).trans (lengths g))
    (by
      show base.value = fun g => mle (base.place g) (lifted E (tables g))
      rw [placed, based])
    fit
  obtain ⟨proof, head, holds⟩ := OnPath.accepts on whole each
  exact ⟨proof, by rw [head, based], by rw [head]; exact placed, by rw [head, based], holds⟩

omit [FieldSlot Slot E] in
/-- `accepts` with the terminal at a stage `stage` that is the stage after the level 6 (`next`)
and a stage of the transcript (`inside`). -/
theorem accepts_at {total stage : ℕ} (next : stage = Tier.finish sixth) (inside : stage < total)
    (accept : Prefix (Wire r E) (Option Coord → Slot) total → Prop)
    (table : Address (Wire r E) total → Option Coord → Slot)
    (decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input b, (decode₁ b.1 (table input (some (block₁ b)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodes₆ : ∀ input i, (decode₆ (table input (some (block₆ i)))).isSome)
    (decodesL : ∀ input i, (decodeL (table input (some (blockL i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros Slot 57) {tables : Fin 3 → Table}
    (live : ∀ g cell, 2 ^ ![30, r.vars, 26] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E} (lengths : ∀ g, (places g).length = ![30, r.vars, 26] g)
    (fit : Fit member flat β key openKey decode β₁ key₁ openKey₁ decode₁ β₂ key₂ openKey₂ decode₂
      β₃ key₃ openKey₃ decode₃ key₄ openKey₄ decode₄ key₅ openKey₅ decode₅ key₆ openKey₆ decode₆
      decodeL tables) :
    ∃ proof : ℕ → Wire r E,
      (proof 0).data.commit = (fun g => (commit (key g) (tables g)).1) ∧
      (proof 0).data.place = places ∧
      (proof 0).data.value = (fun g => mle (places g) (lifted E (tables g))) ∧
      offLinks (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
        (route₅ := Route.euclidean (norm₅ r).operator (norm₅ r).limit)
        (route₆ := Route.euclidean norm₆.operator norm₆.limit) zeroth first second third fourth
        fifth sixth (terminalOf r total matrix blockL decodeL stage) 0 Page.read
        (Note.commit ∘ Page.data : Wire r E → _) (Note.value ∘ Page.data : Wire r E → _)
        (Note.place ∘ Page.data : Wire r E → _) (Note.commits ∘ Page.data : Wire r E → _)
        (Note.values ∘ Page.data : Wire r E → _) (Note.places ∘ Page.data : Wire r E → _)
        (commit (key₁ 0) flat).1 (Note.commit₁ ∘ Page.data : Wire r E → _)
        (Note.commit₂ ∘ Page.data : Wire r E → _) (Note.commit₃ ∘ Page.data : Wire r E → _)
        (And.left quarters) (And.left (And.right quarters))
        (And.left (And.right (And.right quarters))) (And.right (And.right (And.right quarters)))
        (Note.commit₄ ∘ Page.data : Wire r E → _) (Note.commit₅ ∘ Page.data : Wire r E → _)
        (Note.commit₆ ∘ Page.data : Wire r E → _) (Note.state ∘ Page.data : Wire r E → _)
        ((transcript total accept).final (.done proof) table) := by
  subst next
  exact accepts member flat β key openKey block decode mass scan β₁ key₁ openKey₁ block₁ decode₁
    mass₁ β₂ key₂ openKey₂ block₂ decode₂ weight₂ β₃ key₃ openKey₃ block₃ decode₃ weight₃ key₄
    openKey₄ block₄ decode₄ weight₄ key₅ openKey₅ block₅ decode₅ weight₅ key₆ openKey₆ block₆
    decode₆ weight₆ matrix blockL decodeL inside accept table decodes decodes₁ decodes₂ decodes₃
    decodes₄ decodes₅ decodes₆ decodesL worked live lengths fit

end Generic

section Code

variable {r : Row} (member : r ∈ rows) (flat : Table)
  (β : ∀ g, Module.Basis (Fin ((root r).planes g)) Zq E)
  (key : (g : Fin 3) → Key ((root r).shape g)) (openKey : (root r).OpenKey)
  (block : (root r).Block → Coord)
  (decode : (g : Fin 3) → Replay.CodeSlot E → Option (Rq ((root r).carrierDegree g))) (mass : Fin 3 → ℕ)
  (scan : ∀ (wire : Wire r E) (challenge : (root r).Challenge),
    SetupScan (Batch.liftedReduction β ((wire.data.claim).statement key openKey) challenge
      rootVars (root_sized r member).1 8 quotientDigits₀ (root_sized r member).2.1 rootBatch
      (root_sized r member).2.2.1) setupVars (Shape.pad (Batch.shape (level₁ r) 0) flat))
  (β₁ : ∀ g, Module.Basis (Fin ((level₁ r).planes g)) Zq E)
  (key₁ : (g : Fin 2) → Key ((level₁ r).shape g)) (openKey₁ : (level₁ r).OpenKey)
  (block₁ : (level₁ r).Block → Coord)
  (decode₁ : (g : Fin 2) → Replay.CodeSlot E → Option (Rq ((level₁ r).carrierDegree g))) (mass₁ : Fin 2 → ℕ)
  (β₂ : Module.Basis (Fin (level₂ r).opening.planes) Zq E) (key₂ : Key (level₂ r).shape)
  (openKey₂ : OpeningKey (level₂ r).opening) (block₂ : Fin (level₂ r).shape.blocks → Coord)
  (decode₂ : Replay.CodeSlot E → Option (Rq (level₂ r).opening.carrierDegree)) (weight₂ : ℕ)
  (β₃ : Module.Basis (Fin (level₃ r).opening.planes) Zq E) (key₃ : Key (level₃ r).shape)
  (openKey₃ : OpeningKey (level₃ r).opening) (block₃ : Fin (level₃ r).shape.blocks → Coord)
  (decode₃ : Replay.CodeSlot E → Option (Rq (level₃ r).opening.carrierDegree)) (weight₃ : ℕ)
  (key₄ : Key level₄.shape) (openKey₄ : OpeningKey level₄.opening)
  (block₄ : Fin level₄.shape.blocks → Coord)
  (decode₄ : Replay.CodeSlot E → Option (Rq level₄.shape.innerDegree)) (weight₄ : ℕ)
  (key₅ : Key level₅.shape) (openKey₅ : OpeningKey level₅.opening)
  (block₅ : Fin level₅.shape.blocks → Coord)
  (decode₅ : Replay.CodeSlot E → Option (Rq level₅.shape.innerDegree)) (weight₅ : ℕ)
  (key₆ : Key level₆.shape) (openKey₆ : OpeningKey level₆.opening)
  (block₆ : Fin level₆.shape.blocks → Coord)
  (decode₆ : Replay.CodeSlot E → Option (Rq level₆.shape.innerDegree)) (weight₆ : ℕ)
  (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
    (Rq terminal.innerDegree))
  (blockL : Fin terminal.blocks → Coord) (decodeL : Replay.CodeSlot E → Option (Rq terminal.innerDegree))

local notation "zeroth" =>
  offBatch β key openKey (Note.claim ∘ Page.data : Wire r E → _) rootVars
    (And.left (root_sized r member)) 8 quotientDigits₀
    (And.left (And.right (root_sized r member))) rootBatch
    (And.left (And.right (And.right (root_sized r member)))) block decode Page.read mass (0 + 1)
    setupVars (Shape.pad (Batch.shape (level₁ r) 0) flat) scan

local notation "first" =>
  Tier.pinned (liftBatch β₁ key₁ openKey₁ (Note.claim₁ ∘ Page.data : Wire r E → _) vars₁
    (And.left (level₁_sized r member)) 16 quotientDigits
    (And.left (And.right (level₁_sized r member))) batch₁
    (And.left (And.right (And.right (level₁_sized r member)))) block₁ decode₁ Page.read mass₁
    (Tier.finish zeroth + 1)) (Batch.Pinned 0 (Prod.fst (commit (key₁ 0) flat))
      (Shape.pad (Batch.shape (level₁ r) 0) flat))
    (fun ω _ => Batch.pinBreak key₁ openKey₁ _ 0 (Prod.snd (commit (key₁ 0) flat)) ω)

local notation "second" =>
  liftFold β₂ key₂ openKey₂ (Note.claim₂ ∘ Page.data : Wire r E → _) (Key20.Level.vars (level₂ r))
    (And.left (level₂_sized r member)) quotientDigits
    (And.left (And.right (level₂_sized r member))) (Key20.Level.batch (level₂ r))
    (And.left (And.right (And.right (level₂_sized r member)))) block₂ decode₂ Page.read weight₂
    (Tier.finish first)

local notation "third" =>
  Tier.tensored (liftFold β₃ key₃ openKey₃ (Note.claim₃ ∘ Page.data : Wire r E → _) (Key20.Level.vars (level₃ r))
    (And.left (level₃_sized r member)) quotientDigits (And.left (And.right (level₃_sized r member))) (Key20.Level.batch (level₃ r))
    (And.left (And.right (And.right (level₃_sized r member)))) block₃ decode₃ Page.read weight₃
    (Tier.finish second)) (Key20.Level.tail (level₃ r))

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := level₄.fold) key₄ openKey₄
    (Note.claim₄ ∘ Page.data : Wire r E → _) level₄.vars (And.left level₄_sized) level₄.batch
    (And.left (And.right level₄_sized)) block₄ decode₄ Page.read weight₄ (Tier.finish third))
    level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅
    (Note.claim₅ ∘ Page.data : Wire r E → _) level₅.vars
    (And.left (tails_sized level₅ (List.mem_cons_self ..))) level₅.batch
    (And.left (And.right (tails_sized level₅ (List.mem_cons_self ..)))) block₅ decode₅
    Page.read weight₅ (Key20.Squared.operator (norm₅ r)) (Key20.Squared.limit (norm₅ r)) (Tier.finish fourth)) level₅.tail

local notation "sixth" =>
  Tier.tensored (committedTraceSq (fold := level₆.fold) key₆ openKey₆
    (Note.claim₆ ∘ Page.data : Wire r E → _) level₆.vars
    (And.left (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))) level₆.batch
    (And.left (And.right (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))))
    block₆ decode₆ Page.read weight₆ norm₆.operator norm₆.limit (Tier.finish fifth))
    level₆.tail

/-- `accepts` for the slots `Replay.CodeSlot E`, whose fold decoders are the decoders of the
pinned Akita code. -/
theorem accepts_code {total : ℕ} (inside : Tier.finish sixth < total)
    (accept : Prefix (Wire r E) (Option Coord → Replay.CodeSlot E) total → Prop)
    (table : Address (Wire r E) total → Option Coord → Replay.CodeSlot E)
    (decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input b, (decode₁ b.1 (table input (some (block₁ b)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodes₆ : ∀ input i, (decode₆ (table input (some (block₆ i)))).isSome)
    (decodesL : ∀ input i, (decodeL (table input (some (blockL i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros (Replay.CodeSlot E) 57) {tables : Fin 3 → Table}
    (live : ∀ g cell, 2 ^ ![30, r.vars, 26] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E} (lengths : ∀ g, (places g).length = ![30, r.vars, 26] g)
    (fit : Fit member flat β key openKey decode β₁ key₁ openKey₁ decode₁ β₂ key₂ openKey₂ decode₂ β₃ key₃
      openKey₃ decode₃ key₄ openKey₄ decode₄ key₅ openKey₅ decode₅ key₆ openKey₆ decode₆ decodeL tables) :
    ∃ proof : ℕ → Wire r E,
      (proof 0).data.commit = (fun g => (commit (key g) (tables g)).1) ∧
      (proof 0).data.place = places ∧
      (proof 0).data.value = (fun g => mle (places g) (lifted E (tables g))) ∧
      offLinks (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
        (route₅ := Route.euclidean (norm₅ r).operator (norm₅ r).limit)
        (route₆ := Route.euclidean norm₆.operator norm₆.limit) zeroth first second third fourth
        fifth sixth (terminalOf r total matrix blockL decodeL (Tier.finish sixth)) 0 Page.read
        (Note.commit ∘ Page.data : Wire r E → _) (Note.value ∘ Page.data : Wire r E → _)
        (Note.place ∘ Page.data : Wire r E → _) (Note.commits ∘ Page.data : Wire r E → _)
        (Note.values ∘ Page.data : Wire r E → _) (Note.places ∘ Page.data : Wire r E → _)
        (commit (key₁ 0) flat).1 (Note.commit₁ ∘ Page.data : Wire r E → _)
        (Note.commit₂ ∘ Page.data : Wire r E → _) (Note.commit₃ ∘ Page.data : Wire r E → _)
        (And.left quarters) (And.left (And.right quarters))
        (And.left (And.right (And.right quarters))) (And.right (And.right (And.right quarters)))
        (Note.commit₄ ∘ Page.data : Wire r E → _) (Note.commit₅ ∘ Page.data : Wire r E → _)
        (Note.commit₆ ∘ Page.data : Wire r E → _) (Note.state ∘ Page.data : Wire r E → _)
        ((transcript total accept).final (.done proof) table) :=
  accepts member flat β key openKey block decode mass scan β₁ key₁ openKey₁ block₁ decode₁
    mass₁ β₂ key₂ openKey₂ block₂ decode₂ weight₂ β₃ key₃ openKey₃ block₃ decode₃ weight₃ key₄
    openKey₄ block₄ decode₄ weight₄ key₅ openKey₅ block₅ decode₅ weight₅ key₆ openKey₆ block₆
    decode₆ weight₆ matrix blockL decodeL inside accept table decodes decodes₁ decodes₂ decodes₃
    decodes₄ decodes₅ decodes₆ decodesL worked live lengths fit

local notation "code₂" =>
  (Replay.CodeSlot.open Key20.open64 Key20.open64_fits :
    Replay.CodeSlot E → Option (Rq (OpeningShape.carrierDegree (Key20.Level.opening (level₂ r)))))

local notation "code₃" =>
  (Replay.CodeSlot.open Key20.open64 Key20.open64_fits :
    Replay.CodeSlot E → Option (Rq (OpeningShape.carrierDegree (Key20.Level.opening (level₃ r)))))

local notation "code₄" =>
  (Replay.CodeSlot.open Key20.open128 Key20.open128_fits :
    Replay.CodeSlot E → Option (Rq level₄.shape.innerDegree))

local notation "code₅" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq level₅.shape.innerDegree))

local notation "code₆" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq level₆.shape.innerDegree))

local notation "codeL" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq terminal.innerDegree))

/-- The data `Tower` of the opening of the row `r` for the wire messages `Wire`: the readers of
`Note`, the setup prefix `flat` with zeros after its cells, the commitment and the reference
decommitment of `commit` for `flat`, and the terminal `terminalOf` with the matrix `matrix` and
the decoder `Replay.CodeSlot.kept64` of `Tower.Drawn` at the stage `stage`. The bases, the matrices and the
coordinates of the slots are free, and `Tower.Drawn` fixes `stage` (`Tower.Drawn.next`). -/
noncomputable def noteTower (total stage : ℕ) : Tower r member E (Wire r E) Coord total where
  β := β
  key := key
  openKey := openKey
  claim := Note.claim ∘ Page.data
  block := block
  flat := Shape.pad (Batch.shape (level₁ r) 0) flat
  β₁ := β₁
  key₁ := key₁
  openKey₁ := openKey₁
  claim₁ := Note.claim₁ ∘ Page.data
  block₁ := block₁
  registry := (commit (key₁ 0) flat).1
  reference := (commit (key₁ 0) flat).2
  β₂ := β₂
  key₂ := key₂
  openKey₂ := openKey₂
  claim₂ := Note.claim₂ ∘ Page.data
  block₂ := block₂
  β₃ := β₃
  key₃ := key₃
  openKey₃ := openKey₃
  claim₃ := Note.claim₃ ∘ Page.data
  block₃ := block₃
  key₄ := key₄
  openKey₄ := openKey₄
  claim₄ := Note.claim₄ ∘ Page.data
  block₄ := block₄
  key₅ := key₅
  openKey₅ := openKey₅
  claim₅ := Note.claim₅ ∘ Page.data
  block₅ := block₅
  key₆ := key₆
  openKey₆ := openKey₆
  claim₆ := Note.claim₆ ∘ Page.data
  block₆ := block₆
  terminal := terminalOf r total matrix blockL codeL stage
  commit := Note.commit ∘ Page.data
  value := Note.value ∘ Page.data
  place := Note.place ∘ Page.data
  commits := Note.commits ∘ Page.data
  values := Note.values ∘ Page.data
  places := Note.places ∘ Page.data
  commit₁ := Note.commit₁ ∘ Page.data
  commit₂ := Note.commit₂ ∘ Page.data
  commit₃ := Note.commit₃ ∘ Page.data
  commit₄ := Note.commit₄ ∘ Page.data
  commit₅ := Note.commit₅ ∘ Page.data
  commit₆ := Note.commit₆ ∘ Page.data
  stateOf := Note.state ∘ Page.data

variable {total stage : ℕ}

local notation "notes" =>
  noteTower member flat β key openKey block β₁ key₁ openKey₁ block₁ β₂ key₂ openKey₂ block₂ β₃
    key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ key₆ openKey₆ block₆ matrix
    blockL total stage

local notation "notes₀" =>
  noteTower member 0 β key openKey block β₁ key₁ openKey₁ block₁ β₂ key₂ openKey₂ block₂ β₃ key₃
    openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ key₆ openKey₆ block₆ matrix blockL
    total stage

/-- `accepts` for the data `noteTower` with `Tower.Shared` and `Tower.Drawn` (Akita Theorem 10.9,
Section 9 with (136)): the wire type `Wire r E` is finite, and for each table of the oracle whose
work slots have 57 zero work bits and each family of tables with the fit conditions `Fit` for
the decoders of the code of `Tower.accepts`, a proof exists whose first message has the commitments
of the tables, the points and the values, and the verifier `Tower.accepts` accepts the transcript
of the prover `.done proof`, which has no queries. So the proof of the honest prover is a prover
of the game of `Tower.extraction_le`. The split of the public weight is `Tower.scan`, and the
hypotheses `decodes` of `accepts` hold by `Tower.Drawn.decodes`. -/
theorem noteTower_accepts [Nonempty Coord] (shared : (notes).Shared)
    (drawn : (notes).Drawn shared Page.read 0) (table : Address (Wire r E) total → Option Coord → Replay.CodeSlot E)
    (worked : ∀ input, table input none ∈ workZeros (Replay.CodeSlot E) 57) {tables : Fin 3 → Table}
    (live : ∀ g cell, 2 ^ ![30, r.vars, 26] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E} (lengths : ∀ g, (places g).length = ![30, r.vars, 26] g)
    (fit : Fit member flat β key openKey (rootDecode E r) β₁ key₁ openKey₁ (Key30.decode₁ E r) β₂
      key₂ openKey₂ code₂ β₃ key₃ openKey₃ code₃ key₄ openKey₄ code₄ key₅ openKey₅ code₅ key₆
      openKey₆ code₆ codeL tables) :
    Finite (Wire r E) ∧ ∃ proof : ℕ → Wire r E,
      (OracleComp.done proof :
        OracleComp (Address (Wire r E) total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire r E)).QueryBound 0 ∧
      (proof 0).data.commit = (fun g => (commit (key g) (tables g)).1) ∧
      (proof 0).data.place = places ∧
      (proof 0).data.value = (fun g => mle (places g) (lifted E (tables g))) ∧
      (notes).accepts shared Page.read (((notes).body shared Page.read 0
        ((notes).accepts shared Page.read)).protocol.final (.done proof) table) := by
  have decoded := drawn.decodes
  obtain ⟨proof, committed, placed, valued, holds⟩ := accepts_at member flat β key openKey block
    (rootDecode E r) Key30.mass (Tower.scan (o := notes) shared) β₁ key₁ openKey₁ block₁
    (Key30.decode₁ E r) Key30.mass₁ β₂ key₂ openKey₂ block₂ code₂ 51 β₃ key₃ openKey₃ block₃ code₃
    51 key₄ openKey₄ block₄ code₄ 31 key₅ openKey₅ block₅ code₅ 53 key₆ openKey₆ block₆ code₆ 53
    matrix blockL codeL drawn.next drawn.inside ((notes).accepts shared Page.read) table
    (fun _ b => (decoded _).1 b.1) (fun _ b => (decoded _).2.1 b.1)
    (fun _ _ => (decoded _).2.2.1) (fun _ _ => (decoded _).2.2.2.1)
    (fun _ _ => (decoded _).2.2.2.2.1) (fun _ _ => (decoded _).2.2.2.2.2.1)
    (fun _ _ => (decoded _).2.2.2.2.2.2.1) (fun _ _ => (decoded _).2.2.2.2.2.2.2) worked live
    lengths fit
  exact ⟨inferInstance, proof, .done proof 0, committed, placed, valued, holds⟩

/-- `noteTower_accepts` for the zero setup prefix and the zero tables, with no hypothesis on the
fit (`fit_zero`): for each table of the oracle whose work slots have 57 zero work bits, a prover
with no queries exists whose transcript `Tower.accepts` accepts. -/
theorem noteTower_zero [Nonempty Coord] (shared : (notes₀).Shared)
    (drawn : (notes₀).Drawn shared Page.read 0)
    (table : Address (Wire r E) total → Option Coord → Replay.CodeSlot E)
    (worked : ∀ input, table input none ∈ workZeros (Replay.CodeSlot E) 57) :
    ∃ prover : OracleComp (Address (Wire r E) total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire r E),
      prover.QueryBound 0 ∧ (notes₀).accepts shared Page.read (((notes₀).body shared Page.read 0
        ((notes₀).accepts shared Page.read)).protocol.final prover table) := by
  obtain ⟨-, proof, bounded, -, -, -, holds⟩ := noteTower_accepts member 0 β key openKey block β₁
    key₁ openKey₁ block₁ β₂ key₂ openKey₂ block₂ β₃ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅
    openKey₅ block₅ key₆ openKey₆ block₆ matrix blockL shared drawn table worked
    (tables := fun _ => 0) (fun _ _ _ => rfl)
    (places := fun g => List.replicate (![30, r.vars, 26] g) 0) (fun _ => List.length_replicate)
    (fit_zero member β key openKey (rootDecode E r) β₁ key₁ openKey₁ (Key30.decode₁ E r) β₂ key₂
      openKey₂ code₂ β₃ key₃ openKey₃ code₃ key₄ openKey₄ code₄ key₅ openKey₅ code₅ key₆ openKey₆
      code₆ codeL)
  exact ⟨_, bounded, holds⟩

/-- The accept event of `Tower.extraction_le` for the data `noteTower` of the zero setup prefix
with `Tower.Shared` and `Tower.Drawn` is not empty: a table of the oracle and a prover with no
queries exist whose transcript `Tower.accepts` accepts (`noteTower_zero` for a table whose work
slots are a slot with 57 zero work bits, which exists since a slot has `2^131` values or more,
`Key20.card_codeSlot`). -/
theorem noteTower_played [Nonempty Coord] (shared : (notes₀).Shared)
    (drawn : (notes₀).Drawn shared Page.read 0) :
    ∃ (table : Address (Wire r E) total → Option Coord → Replay.CodeSlot E)
      (prover : OracleComp (Address (Wire r E) total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire r E)),
      prover.QueryBound 0 ∧ (notes₀).accepts shared Page.read (((notes₀).body shared Page.read 0
        ((notes₀).accepts shared Page.read)).protocol.final prover table) := by
  obtain ⟨slot, worked⟩ : (workZeros (Replay.CodeSlot E) 57).Nonempty := Finset.card_pos.mp (by
    rw [workZeros_card]
    exact Nat.div_pos ((Nat.pow_le_pow_right (by decide) (by decide)).trans Key20.card_codeSlot)
      (Nat.two_pow_pos _))
  exact ⟨fun _ _ => slot, noteTower_zero member β key openKey block β₁
    key₁ openKey₁ block₁ β₂ key₂ openKey₂ block₂ β₃ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅
    openKey₅ block₅ key₆ openKey₆ block₆ matrix blockL shared drawn _ fun _ => worked⟩

end Code

end NightstreamFPrime.Spec.Compression.Akita.Key30
