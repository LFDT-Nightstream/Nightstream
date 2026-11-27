import NightstreamFPrime.Spec.Compression.Akita.Key20Root
import NightstreamFPrime.Spec.Compression.Akita.Key20Opening
import NightstreamFPrime.Spec.Compression.Akita.Extract.Page
import NightstreamFPrime.Spec.Compression.Akita.Extract.ZeroFit

/-! The verifier `towerLinks` of the opening of Akita for the schedule file
`crates/ns-compress/artifacts/key20.aks` accepts a proof (Akita, ePrint 2026/1983, revision of
2026-11-23, Theorem 10.9, Figure 4 and Section 8.2). The theorem `Key20.tower_suffix` bounds an
event of this verifier, and `Key20.tower_complete` has hypotheses on the wire messages and on the
responses of the honest prover. This file has the hypotheses on the responses in `Fit`.

- `Note` has the data of a wire message other than the messages of the checks: the commitments,
  the points and the values of the three claims, the claim of the root fold, the commitment and
  the claim of each level, and the state, the statement, the clear partials and the clear
  response of the terminal. The points are vectors of 20, 16 and 16 coordinates, and
  `Note.place` reads them as lists. `Wire` is `Page` for this data, with 4 cells of 8 elements
  and a number up to `3029804899`. `Note.finite`: a note is finite, so `Wire E` is finite for a
  finite `E`.
- `terminalOf` is the terminal that reads its statement and its clear partials from the message
  before its fold challenges.
- `Fit` has the fit conditions of the honest prover (Akita Definition 10.5) of each level for the
  tables of the key polynomial, `W` and `V`: for each opening point and each fold challenge
  vector that the decoders give, the response of each level satisfies `Enc` and the norm limit
  of its route.
- `accepts`: for each family of tables with `Fit` and each table of the oracle whose slots decode
  to fold challenges, a proof exists whose transcript satisfies `towerLinks`. It is the proof of
  the honest prover. The matrices, the decoders of the fold challenges and the coordinates of
  their slots are free.
- `fit_zero` and `fit_quiet` give `Fit` in two cases: for the zero tables with each decoder, and
  for each family of tables with decoders that give the fold challenge 0. For a fold challenge
  vector of the families of the schedule, the response of a table that is not 0 can be larger
  than the limit of `Enc`, so `Fit` for these families is a hypothesis: the honest prover of the
  Akita code has a search over a nonce for a response that fits
  (`crates/akita-prover/src/protocol/fold_grind.rs`).
- `noteTower` is the data `Key20.Tower` for the wire messages `Wire`: the readers of `Note`, and
  the terminal `terminalOf` with the decoder of `Tower.Drawn`. `noteTower_accepts` is `accepts`
  for the decoders of the code of `Tower.accepts`: the proof of the honest prover is a prover with no
  queries of the game of `Tower.extraction_le`, and `Tower.accepts` accepts its transcript.
  `noteTower_zero` and `noteTower_played` give such a prover for the zero tables, with no
  hypothesis on the fit. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key20

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The data of a wire message of the opening other than the messages of the checks (Akita
Figure 4: the messages of the prover). The points of the three claims are vectors of 20, 16 and
16 coordinates. `trace` has the byte strings that the prover sends for the stage of the wire
message: at most 5 strings of at most 16384 bytes for `key20.aks`. -/
structure Note (E : Type) [CommRing E] [Algebra Zq E] where
  commit : (g : Fin 3) → (root.shape g).Payload
  value : Fin 3 → E
  point : (g : Fin 3) → Fin (![20, 16, 16] g) → E
  claim : root.Claim E
  commit₁ : level₁.shape.Payload
  claim₁ : Claim level₁.opening E
  commit₂ : level₂.shape.Payload
  claim₂ : TraceClaim level₂.opening E
  commit₃ : level₃.shape.Payload
  claim₃ : TraceClaim level₃.opening E
  commit₄ : level₄.shape.Payload
  claim₄ : TraceClaim level₄.opening E
  commit₅ : level₅.shape.Payload
  claim₅ : TraceClaim level₅.opening E
  state : Fin terminal.blocks → Fin terminal.innerRank → Rq terminal.innerDegree
  statement : TerminalStatement terminal E
  partials : Fin terminal.blocks → Rq terminal.innerDegree
  response : terminal.Source
  trace : Trace 5 16384

/-- The points of the three claims of a note as lists. -/
def Note.place {E : Type} [CommRing E] [Algebra Zq E] (note : Note E) (g : Fin 3) : List E :=
  List.ofFn (note.point g)

instance Note.finite {E : Type} [CommRing E] [Algebra Zq E] [Finite E] : Finite (Note E) :=
  Finite.of_injective (fun note : Note E => (note.commit, note.value, note.point, note.claim,
    note.commit₁, note.claim₁, note.commit₂, note.claim₂, note.commit₃, note.claim₃,
    note.commit₄, note.claim₄, note.commit₅, note.claim₅, note.state, note.statement,
    note.partials, note.response, note.trace)) fun left right same => by
      cases left
      cases right
      simp_all

/-- A wire message of the opening: the data `Note`, 4 cells of 8 elements of `E` and a number up
to `3029804899` for the messages of the checks, and the nonce of the response search. -/
abbrev Wire (E : Type) [CommRing E] [Algebra Zq E] : Type := Page (Note E) E 4 8 3029804899

variable {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E]
  [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The terminal with the matrix `A_term`, the fold challenges at the stage `start`, and the
statement, the clear partials and the nonce of the response search of the message before this
stage (Akita Section 8.2). -/
def terminalOf (total : ℕ)
    (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
      (Rq terminal.innerDegree))
    (block : Fin terminal.blocks → Coord) (decode : Slot → Option (Rq terminal.innerDegree))
    (start : ℕ) :
    Terminal terminal E (Wire E) Coord Slot total
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

variable (β : ∀ g, Module.Basis (Fin (root.planes g)) Zq E)
  (key : (g : Fin 3) → Key (root.shape g)) (openKey : root.OpenKey) (block : root.Block → Coord)
  (decode : (g : Fin 3) → Slot → Option (Rq (root.carrierDegree g))) (mass : Fin 3 → ℕ)
  (β₁ : Module.Basis (Fin level₁.opening.planes) Zq E) (key₁ : Key level₁.shape)
  (openKey₁ : OpeningKey level₁.opening) (block₁ : Fin level₁.shape.blocks → Coord)
  (decode₁ : Slot → Option (Rq level₁.opening.carrierDegree)) (weight₁ : ℕ)
  (key₂ : Key level₂.shape) (openKey₂ : OpeningKey level₂.opening)
  (block₂ : Fin level₂.shape.blocks → Coord)
  (decode₂ : Slot → Option (Rq level₂.shape.innerDegree)) (weight₂ : ℕ)
  (key₃ : Key level₃.shape) (openKey₃ : OpeningKey level₃.opening)
  (block₃ : Fin level₃.shape.blocks → Coord)
  (decode₃ : Slot → Option (Rq level₃.shape.innerDegree)) (weight₃ : ℕ)
  (key₄ : Key level₄.shape) (openKey₄ : OpeningKey level₄.opening)
  (block₄ : Fin level₄.shape.blocks → Coord)
  (decode₄ : Slot → Option (Rq level₄.shape.innerDegree)) (weight₄ : ℕ)
  (key₅ : Key level₅.shape) (openKey₅ : OpeningKey level₅.opening)
  (block₅ : Fin level₅.shape.blocks → Coord)
  (decode₅ : Slot → Option (Rq level₅.shape.innerDegree)) (weight₅ : ℕ)
  (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
    (Rq terminal.innerDegree))
  (blockL : Fin terminal.blocks → Coord) (decodeL : Slot → Option (Rq terminal.innerDegree))

local notation "zeroth" =>
  liftBatch β key openKey (Note.claim ∘ Page.data : Wire E → _) rootVars (And.left root_sized) 8
    quotientDigits root_room rootBatch (And.left (And.right root_sized)) block decode Page.read
    mass (0 + 1)

local notation "first" =>
  Tier.tensored (liftFold β₁ key₁ openKey₁ (Note.claim₁ ∘ Page.data : Wire E → _) level₁.vars
    (And.left level₁_sized) quotientDigits (And.left (And.right level₁_sized)) level₁.batch
    (And.left (And.right (And.right level₁_sized))) block₁ decode₁ Page.read weight₁
    (Tier.finish zeroth)) level₁.tail

local notation "second" =>
  Tier.tensored (committedTrace (fold := level₂.fold) key₂ openKey₂
    (Note.claim₂ ∘ Page.data : Wire E → _) level₂.vars (And.left sized₂) level₂.batch
    (And.left (And.right sized₂)) block₂ decode₂ Page.read weight₂ (Tier.finish first))
    level₂.tail

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := level₃.fold) key₃ openKey₃
    (Note.claim₃ ∘ Page.data : Wire E → _) level₃.vars (And.left sized₃) level₃.batch
    (And.left (And.right sized₃)) block₃ decode₃ Page.read weight₃ norm₃.operator
    norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) key₄ openKey₄
    (Note.claim₄ ∘ Page.data : Wire E → _) level₄.vars (And.left sized₄) level₄.batch
    (And.left (And.right sized₄)) block₄ decode₄ Page.read weight₄ norm₄.operator
    norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅
    (Note.claim₅ ∘ Page.data : Wire E → _) level₅.vars (And.left sized₅) level₅.batch
    (And.left (And.right sized₅)) block₅ decode₅ Page.read weight₅ norm₅.operator
    norm₅.limit (Tier.finish fourth)) level₅.tail

/-- The fit conditions of the honest prover of the opening (Akita Definition 10.5 and
Lemma 10.7) for the tables `tables` of the key polynomial, `W` and `V`: the conditions `BatchFit`
of the root, `LiftFit` of level 1, `TraceFit` of the levels 2 to 5 with `S_max` of `norm₃`,
`norm₄` and `norm₅`, and `‖z‖_2² ≤ S_max,term` at the terminal. Each condition is for each opening
point and each fold challenge vector that the decoders give. -/
def Fit (tables : Fin 3 → Table) : Prop :=
  BatchFit β key openKey rootVars (And.left root_sized) 8 quotientDigits decode tables fun w₀ =>
  LiftFit β₁ key₁ openKey₁ level₁.fold level₁.vars (And.left level₁_sized) quotientDigits
    decode₁ (fun i => level₁.shape.embed (level₁.shape.pack w₀ i)) fun w₁ =>
  TraceFit (E := E) (And.left quarters) key₂ openKey₂ level₂.fold (fun _ => decode₂) (fun _ => True)
    (packed w₁) fun w₂ =>
  TraceFit (E := E) (And.left (And.right quarters)) key₃ openKey₃ level₃.fold (fun _ => decode₃)
    (fun z => vecNormSq z ≤ norm₃.limit) (packed w₂) fun w₃ =>
  TraceFit (E := E) (And.left (And.right (And.right quarters))) key₄ openKey₄ level₄.fold
    (fun _ => decode₄) (fun z => vecNormSq z ≤ norm₄.limit) (packed w₃) fun w₄ =>
  TraceFit (E := E) (And.left (And.right (And.right (And.right quarters)))) key₅ openKey₅ level₅.fold
    (fun _ => decode₅) (fun z => vecNormSq z ≤ norm₅.limit) (packed w₄) fun w₅ =>
  ∀ c : Fin terminal.blocks → Rq terminal.innerDegree,
    (∀ i, ∃ slot, decodeL slot = some (c i)) →
    vecNormSq (traceResponse c (traceSource
      (And.right (And.right (And.right (And.right quarters)))) (packed w₅ : ℕ → E))) ≤
      normTerminal.limit

omit [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
  [Nonempty Slot] in
/-- The fit conditions `Fit` hold for the zero tables, with each decoder: each response is 0
(`Extract/ZeroFit.lean`). -/
theorem fit_zero :
    Fit β key openKey decode β₁ key₁ openKey₁ decode₁ key₂ openKey₂ decode₂ key₃ openKey₃ decode₃
      key₄ openKey₄ decode₄ key₅ openKey₅ decode₅ decodeL fun _ => 0 :=
  batchFit_zero root_honest (fun g => (root_folds g).1) (fun g => (root_folds g).2) paths.1
    (by decide)
    (liftFit_zero level₁_bounded covers.1 level₁_fold.1 level₁_fold.2.1 (by decide)
      (traceFit_zero quarters.1 (by decide) (by decide) trivial
        (traceFit_zero quarters.2.1 (by decide) (by decide)
          (by rw [vecNormSq_zero]; exact Nat.zero_le _)
          (traceFit_zero quarters.2.2.1 (by decide) (by decide)
            (by rw [vecNormSq_zero]; exact Nat.zero_le _)
            (traceFit_zero quarters.2.2.2.1 (by decide) (by decide)
              (by rw [vecNormSq_zero]; exact Nat.zero_le _) fun c _ => by
                rw [packed_zero, traceSource_zero, traceResponse_zero, vecNormSq_zero]
                exact Nat.zero_le _)))))

omit [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
  [Nonempty Slot] in
/-- The fit conditions `Fit` hold for each family of tables when each decoder gives only the fold
challenge 0: each response is 0. So the hypothesis `fit` of `accepts` holds for tables that are
not 0. -/
theorem fit_quiet (tables : Fin 3 → Table) :
    Fit (E := E) (Slot := Slot) β key openKey (fun _ _ => some 0) β₁ key₁ openKey₁
      (fun _ => some 0) key₂ openKey₂ (fun _ => some 0) key₃ openKey₃ (fun _ => some 0) key₄ openKey₄ (fun _ => some 0)
      key₅ openKey₅ (fun _ => some 0) (fun _ => some 0) tables :=
  batchFit_quiet (fun g => (root_folds g).1.two)
    (fun _ _ _ decoded => (Option.some.inj decoded).symm) fun _ _ =>
  liftFit_quiet (by decide) (fun _ _ decoded => (Option.some.inj decoded).symm) fun _ =>
  traceFit_quiet quarters.1 (by decide) (fun _ _ _ decoded => (Option.some.inj decoded).symm)
    trivial fun _ =>
  traceFit_quiet quarters.2.1 (by decide) (fun _ _ _ decoded => (Option.some.inj decoded).symm)
    (by rw [vecNormSq_zero]; exact Nat.zero_le _) fun _ =>
  traceFit_quiet quarters.2.2.1 (by decide)
    (fun _ _ _ decoded => (Option.some.inj decoded).symm)
    (by rw [vecNormSq_zero]; exact Nat.zero_le _) fun _ =>
  traceFit_quiet quarters.2.2.2.1 (by decide)
    (fun _ _ _ decoded => (Option.some.inj decoded).symm)
    (by rw [vecNormSq_zero]; exact Nat.zero_le _) fun _ c decodable => by
      obtain rfl : c = fun _ => 0 := funext fun i => by
        obtain ⟨slot, decoded⟩ := decodable i
        exact (Option.some.inj decoded).symm
      rw [traceResponse_quiet, vecNormSq_zero]
      exact Nat.zero_le _

omit [FieldSlot Slot E] in
/-- The verifier `towerLinks` of the opening of the schedule `key20.aks` accepts the proof of the
honest prover (Akita Theorem 10.9) for each family of tables with the fit conditions `Fit`. The
wire messages are `Wire`, the batching root is at the stage 0, and the terminal is `terminalOf`
at the stage after the level 5. `tables`: the tables of the key polynomial, `W` and `V`, with
opening points `places`. For each matrix of each level, each decoder of the fold challenges, each
table of the oracle whose slots decode to fold challenges and each acceptance condition of the
transcript protocol, a proof exists whose first message has the commitments of the tables, the
points and the values of the tables at the points, and whose complete transcript satisfies
`towerLinks`: the link `RootLinked` of the batching root, the link `Tier.Linked` from the root to
the level 1, the links `Tier.TraceLinked` of the levels 1 to 5, the link `Terminal.Linked` and the
checks `Terminal.Passed` (`Key20.tower_complete`). `inside`: the fold challenge of the terminal is
a stage of the transcript. `worked`: each work slot of the table has 57 zero work bits. `live`:
the tables are 0 after `2^20`, `2^16` and `2^16` cells. `lengths`: the points have 20, 16 and 16
coordinates. `fit`: the fit conditions `Fit`, which `fit_zero` and `fit_quiet` give in two
cases. -/
theorem accepts {total : ℕ} (inside : Tier.finish fifth < total)
    (accept : Prefix (Wire E) (Option Coord → Slot) total → Prop)
    (table : Address (Wire E) total → Option Coord → Slot)
    (decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input i, (decode₁ (table input (some (block₁ i)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodesL : ∀ input i, (decodeL (table input (some (blockL i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros Slot 57) {tables : Fin 3 → Table}
    (live : ∀ g cell, 2 ^ ![20, 16, 16] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E} (lengths : ∀ g, (places g).length = ![20, 16, 16] g)
    (fit : Fit β key openKey decode β₁ key₁ openKey₁ decode₁ key₂ openKey₂ decode₂ key₃ openKey₃ decode₃ key₄
      openKey₄ decode₄ key₅ openKey₅ decode₅ decodeL tables) :
    ∃ proof : ℕ → Wire E,
      (proof 0).data.commit = (fun g => (commit (key g) (tables g)).1) ∧
      (proof 0).data.place = places ∧
      (proof 0).data.value = (fun g => mle (places g) (lifted E (tables g))) ∧
      towerLinks (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅)
        Route.coefficient (Route.euclidean norm₃.operator norm₃.limit)
        (Route.euclidean norm₄.operator norm₄.limit) (Route.euclidean norm₅.operator norm₅.limit)
        zeroth first second third fourth fifth
        (terminalOf total matrix blockL decodeL (Tier.finish fifth)) 0 Page.read
        (Note.commit ∘ Page.data : Wire E → _) (Note.value ∘ Page.data : Wire E → _)
        (Note.place ∘ Page.data : Wire E → _) (And.left quarters) (And.left (And.right quarters))
        (And.left (And.right (And.right quarters)))
        (And.left (And.right (And.right (And.right quarters))))
        (And.right (And.right (And.right (And.right quarters))))
        (Note.commit₁ ∘ Page.data : Wire E → _) (Note.commit₂ ∘ Page.data : Wire E → _)
        (Note.commit₃ ∘ Page.data : Wire E → _) (Note.commit₄ ∘ Page.data : Wire E → _)
        (Note.commit₅ ∘ Page.data : Wire E → _) (Note.state ∘ Page.data : Wire E → _)
        ((transcript total accept).final (.done proof) table) := by
  simp only [final_done]
  set table := fun input : Prefix (Wire E) (Option Coord → Slot) total => table input.address
  replace decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome :=
    fun input => decodes input.address
  replace decodes₁ : ∀ input i, (decode₁ (table input (some (block₁ i)))).isSome :=
    fun input => decodes₁ input.address
  replace decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome :=
    fun input => decodes₂ input.address
  replace decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome :=
    fun input => decodes₃ input.address
  replace decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome :=
    fun input => decodes₄ input.address
  replace decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome :=
    fun input => decodes₅ input.address
  replace decodesL : ∀ input i, (decodeL (table input (some (blockL i)))).isSome :=
    fun input => decodesL input.address
  replace worked : ∀ input, table input none ∈ workZeros Slot 57 := fun input => worked input.address
  clear_value table
  obtain ⟨base, based⟩ : ∃ base : Note E, base = {
      commit := fun g => (commit (key g) (tables g)).1
      value := fun g => mle (places g) (lifted E (tables g))
      point := fun g index => (places g).getD index 0
      claim := Batch.honestClaim β key openKey (fun _ => 0)
        (fun g => Point.ofSegments (root.opening g) [] [] [] []) 0
      commit₁ := (commitSource key₁ 0).1
      claim₁ := sourceClaim β₁ key₁ openKey₁ 0 (Point.ofSegments level₁.opening [] [] [] [])
      commit₂ := (commitSource key₂ 0).1
      claim₂ := honestTraceClaim quarters.1 key₂ openKey₂ 0 [] [] [] 0
      commit₃ := (commitSource key₃ 0).1
      claim₃ := honestTraceClaim quarters.2.1 key₃ openKey₃ 0 [] [] [] 0
      commit₄ := (commitSource key₄ 0).1
      claim₄ := honestTraceClaim quarters.2.2.1 key₄ openKey₄ 0 [] [] [] 0
      commit₅ := (commitSource key₅ 0).1
      claim₅ := honestTraceClaim quarters.2.2.2.1 key₅ openKey₅ 0 [] [] [] 0
      state := 0
      statement := TerminalStatement.ofPoint quarters.2.2.2.2 matrix 0 [] [] [] 0 0
      partials := 0
      response := 0
      trace := Trace.empty 5 16384 } := ⟨_, rfl⟩
  have placed : base.place = places := funext fun g => by
    rw [based]
    exact ofFn_getD (lengths g) 0
  obtain ⟨later, on, whole, each⟩ := tower_complete (β := β) (key := key) (openKey := openKey)
    (block := block) (decode := decode) (mass := mass) (β₁ := β₁) (key₁ := key₁)
    (openKey₁ := openKey₁) (block₁ := block₁) (decode₁ := decode₁) (weight₁ := weight₁)
    (key₂ := key₂) (openKey₂ := openKey₂) (block₂ := block₂) (decode₂ := decode₂)
    (weight₂ := weight₂) (key₃ := key₃) (openKey₃ := openKey₃) (block₃ := block₃)
    (decode₃ := decode₃) (weight₃ := weight₃) (key₄ := key₄) (openKey₄ := openKey₄)
    (block₄ := block₄) (decode₄ := decode₄) (weight₄ := weight₄) (key₅ := key₅)
    (openKey₅ := openKey₅) (block₅ := block₅) (decode₅ := decode₅) (weight₅ := weight₅)
    (L := terminalOf total matrix blockL decodeL (Tier.finish fifth)) (accept := accept)
    (read := Page.read) (claim := Note.claim ∘ Page.data) (claim₁ := Note.claim₁ ∘ Page.data)
    (claim₂ := Note.claim₂ ∘ Page.data) (claim₃ := Note.claim₃ ∘ Page.data)
    (claim₄ := Note.claim₄ ∘ Page.data) (claim₅ := Note.claim₅ ∘ Page.data)
    (commit := Note.commit ∘ Page.data) (value := Note.value ∘ Page.data)
    (place := Note.place ∘ Page.data) (commit₁ := Note.commit₁ ∘ Page.data)
    (commit₂ := Note.commit₂ ∘ Page.data) (commit₃ := Note.commit₃ ∘ Page.data)
    (commit₄ := Note.commit₄ ∘ Page.data) (commit₅ := Note.commit₅ ∘ Page.data)
    (stateOf := Note.state ∘ Page.data) (statementOf := Note.statement ∘ Page.data)
    (partialsOf := Note.partials ∘ Page.data)
    (Page.sends (by decide)) le_rfl le_rfl (by decide)
    (Page.carries Note.claim fun target => ⟨{ base with claim := target }, rfl⟩)
    (Page.carries (fun note : Note E => (note.commit₁, note.claim₁)) fun target =>
      ⟨{ base with commit₁ := target.1, claim₁ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note E => (note.commit₂, note.claim₂)) fun target =>
      ⟨{ base with commit₂ := target.1, claim₂ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note E => (note.commit₃, note.claim₃)) fun target =>
      ⟨{ base with commit₃ := target.1, claim₃ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note E => (note.commit₄, note.claim₄)) fun target =>
      ⟨{ base with commit₄ := target.1, claim₄ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note E => (note.commit₅, note.claim₅)) fun target =>
      ⟨{ base with commit₅ := target.1, claim₅ := target.2 }, rfl⟩)
    (Page.carries (fun note : Note E => (note.state, note.statement, note.partials)) fun target =>
      ⟨{ base with state := target.1, statement := target.2.1, partials := target.2.2 }, rfl⟩)
    (Page.carries Note.response fun target => ⟨{ base with response := target }, rfl⟩)
    rfl inside (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) table decodes decodes₁ decodes₂ decodes₃
    decodes₄ decodes₅ decodesL worked
    (onPath_nil accept table (⟨base, fun _ => (fun _ => 0, 0), 0⟩ : Wire E)) rfl
    (tables := tables) live (by rw [based]; rfl)
    (fun g => (congrArg List.length (congrFun placed g)).trans (lengths g))
    (by
      show base.value = fun g => mle (base.place g) (lifted E (tables g))
      rw [placed, based])
    fit
  obtain ⟨proof, head, holds⟩ := OnPath.accepts on whole each
  exact ⟨proof, by rw [head, based], by rw [head]; exact placed, by rw [head, based], holds⟩

end Generic

section Code

variable (β : ∀ g, Module.Basis (Fin (root.planes g)) Zq E)
  (key : (g : Fin 3) → Key (root.shape g)) (openKey : root.OpenKey) (block : root.Block → Coord)
  (decode : (g : Fin 3) → Replay.CodeSlot E → Option (Rq (root.carrierDegree g))) (mass : Fin 3 → ℕ)
  (β₁ : Module.Basis (Fin level₁.opening.planes) Zq E) (key₁ : Key level₁.shape)
  (openKey₁ : OpeningKey level₁.opening) (block₁ : Fin level₁.shape.blocks → Coord)
  (decode₁ : Replay.CodeSlot E → Option (Rq level₁.opening.carrierDegree)) (weight₁ : ℕ)
  (key₂ : Key level₂.shape) (openKey₂ : OpeningKey level₂.opening)
  (block₂ : Fin level₂.shape.blocks → Coord)
  (decode₂ : Replay.CodeSlot E → Option (Rq level₂.shape.innerDegree)) (weight₂ : ℕ)
  (key₃ : Key level₃.shape) (openKey₃ : OpeningKey level₃.opening)
  (block₃ : Fin level₃.shape.blocks → Coord)
  (decode₃ : Replay.CodeSlot E → Option (Rq level₃.shape.innerDegree)) (weight₃ : ℕ)
  (key₄ : Key level₄.shape) (openKey₄ : OpeningKey level₄.opening)
  (block₄ : Fin level₄.shape.blocks → Coord)
  (decode₄ : Replay.CodeSlot E → Option (Rq level₄.shape.innerDegree)) (weight₄ : ℕ)
  (key₅ : Key level₅.shape) (openKey₅ : OpeningKey level₅.opening)
  (block₅ : Fin level₅.shape.blocks → Coord)
  (decode₅ : Replay.CodeSlot E → Option (Rq level₅.shape.innerDegree)) (weight₅ : ℕ)
  (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
    (Rq terminal.innerDegree))
  (blockL : Fin terminal.blocks → Coord) (decodeL : Replay.CodeSlot E → Option (Rq terminal.innerDegree))

local notation "zeroth" =>
  liftBatch β key openKey (Note.claim ∘ Page.data : Wire E → _) rootVars (And.left root_sized) 8
    quotientDigits root_room rootBatch (And.left (And.right root_sized)) block decode Page.read
    mass (0 + 1)

local notation "first" =>
  Tier.tensored (liftFold β₁ key₁ openKey₁ (Note.claim₁ ∘ Page.data : Wire E → _) level₁.vars
    (And.left level₁_sized) quotientDigits (And.left (And.right level₁_sized)) level₁.batch
    (And.left (And.right (And.right level₁_sized))) block₁ decode₁ Page.read weight₁
    (Tier.finish zeroth)) level₁.tail

local notation "second" =>
  Tier.tensored (committedTrace (fold := level₂.fold) key₂ openKey₂
    (Note.claim₂ ∘ Page.data : Wire E → _) level₂.vars (And.left sized₂) level₂.batch
    (And.left (And.right sized₂)) block₂ decode₂ Page.read weight₂ (Tier.finish first))
    level₂.tail

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := level₃.fold) key₃ openKey₃
    (Note.claim₃ ∘ Page.data : Wire E → _) level₃.vars (And.left sized₃) level₃.batch
    (And.left (And.right sized₃)) block₃ decode₃ Page.read weight₃ norm₃.operator
    norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) key₄ openKey₄
    (Note.claim₄ ∘ Page.data : Wire E → _) level₄.vars (And.left sized₄) level₄.batch
    (And.left (And.right sized₄)) block₄ decode₄ Page.read weight₄ norm₄.operator
    norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅
    (Note.claim₅ ∘ Page.data : Wire E → _) level₅.vars (And.left sized₅) level₅.batch
    (And.left (And.right sized₅)) block₅ decode₅ Page.read weight₅ norm₅.operator
    norm₅.limit (Tier.finish fourth)) level₅.tail

/-- `accepts` for the slots `Replay.CodeSlot E`, whose fold decoders are the decoders of the
pinned Akita code. -/
theorem accepts_code {total : ℕ} (inside : Tier.finish fifth < total)
    (accept : Prefix (Wire E) (Option Coord → Replay.CodeSlot E) total → Prop)
    (table : Address (Wire E) total → Option Coord → Replay.CodeSlot E)
    (decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input i, (decode₁ (table input (some (block₁ i)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodesL : ∀ input i, (decodeL (table input (some (blockL i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros (Replay.CodeSlot E) 57) {tables : Fin 3 → Table}
    (live : ∀ g cell, 2 ^ ![20, 16, 16] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E} (lengths : ∀ g, (places g).length = ![20, 16, 16] g)
    (fit : Fit β key openKey decode β₁ key₁ openKey₁ decode₁ key₂ openKey₂ decode₂ key₃ openKey₃ decode₃ key₄
      openKey₄ decode₄ key₅ openKey₅ decode₅ decodeL tables) :
    ∃ proof : ℕ → Wire E,
      (proof 0).data.commit = (fun g => (commit (key g) (tables g)).1) ∧
      (proof 0).data.place = places ∧
      (proof 0).data.value = (fun g => mle (places g) (lifted E (tables g))) ∧
      towerLinks (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅)
        Route.coefficient (Route.euclidean norm₃.operator norm₃.limit)
        (Route.euclidean norm₄.operator norm₄.limit) (Route.euclidean norm₅.operator norm₅.limit)
        zeroth first second third fourth fifth
        (terminalOf total matrix blockL decodeL (Tier.finish fifth)) 0 Page.read
        (Note.commit ∘ Page.data : Wire E → _) (Note.value ∘ Page.data : Wire E → _)
        (Note.place ∘ Page.data : Wire E → _) (And.left quarters) (And.left (And.right quarters))
        (And.left (And.right (And.right quarters)))
        (And.left (And.right (And.right (And.right quarters))))
        (And.right (And.right (And.right (And.right quarters))))
        (Note.commit₁ ∘ Page.data : Wire E → _) (Note.commit₂ ∘ Page.data : Wire E → _)
        (Note.commit₃ ∘ Page.data : Wire E → _) (Note.commit₄ ∘ Page.data : Wire E → _)
        (Note.commit₅ ∘ Page.data : Wire E → _) (Note.state ∘ Page.data : Wire E → _)
        ((transcript total accept).final (.done proof) table) :=
  accepts β key openKey block decode mass β₁ key₁ openKey₁ block₁ decode₁ weight₁ key₂
    openKey₂ block₂ decode₂ weight₂ key₃ openKey₃ block₃ decode₃ weight₃ key₄ openKey₄ block₄
    decode₄ weight₄ key₅ openKey₅ block₅ decode₅ weight₅ matrix blockL decodeL inside accept
    table decodes decodes₁ decodes₂ decodes₃ decodes₄ decodes₅ decodesL worked live lengths fit

local notation "code₁" =>
  (Replay.CodeSlot.open open64 open64_fits :
    Replay.CodeSlot E → Option (Rq level₁.opening.carrierDegree))

local notation "code₂" =>
  (Replay.CodeSlot.open open128 open128_fits :
    Replay.CodeSlot E → Option (Rq level₂.shape.innerDegree))

local notation "code₃" =>
  (Replay.CodeSlot.kept128 : Replay.CodeSlot E → Option (Rq level₃.shape.innerDegree))

local notation "code₄" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq level₄.shape.innerDegree))

local notation "code₅" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq level₅.shape.innerDegree))

local notation "codeL" =>
  (Replay.CodeSlot.kept64 : Replay.CodeSlot E → Option (Rq terminal.innerDegree))

/-- The data `Tower` of the opening for the wire messages `Wire`: the reader `Page.read`, the
readers of `Note`, and the terminal `terminalOf` with the matrix `matrix` and the decoder
`Replay.CodeSlot.kept64` of `Tower.Drawn` at the stage `stage`. The bases, the matrices and the coordinates of the slots
are free, and `Tower.Drawn` fixes `stage` (`Tower.Drawn.next`). -/
noncomputable def noteTower (total stage : ℕ) : Tower E (Wire E) Coord total where
  read := Page.read
  β := β
  key := key
  openKey := openKey
  claim := Note.claim ∘ Page.data
  block := block
  β₁ := β₁
  key₁ := key₁
  openKey₁ := openKey₁
  claim₁ := Note.claim₁ ∘ Page.data
  block₁ := block₁
  key₂ := key₂
  openKey₂ := openKey₂
  claim₂ := Note.claim₂ ∘ Page.data
  block₂ := block₂
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
  terminal := terminalOf total matrix blockL codeL stage
  commit := Note.commit ∘ Page.data
  value := Note.value ∘ Page.data
  place := Note.place ∘ Page.data
  commit₁ := Note.commit₁ ∘ Page.data
  commit₂ := Note.commit₂ ∘ Page.data
  commit₃ := Note.commit₃ ∘ Page.data
  commit₄ := Note.commit₄ ∘ Page.data
  commit₅ := Note.commit₅ ∘ Page.data
  stateOf := Note.state ∘ Page.data

variable {total stage : ℕ}

local notation "notes" =>
  noteTower β key openKey block β₁ key₁ openKey₁ block₁ key₂ openKey₂ block₂ key₃ openKey₃ block₃
    key₄ openKey₄ block₄ key₅ openKey₅ block₅ matrix blockL total stage

/-- `accepts` for the data `noteTower` with `Tower.Drawn` (Akita Theorem 10.9): the wire type
`Wire E` is finite, and for each table of the oracle whose work slots have 57 zero work bits and
each family of tables with the fit conditions `Fit` for the decoders of `Tower.accepts`, a
proof exists whose first message has the commitments of the tables, the points and the values,
and the verifier `Tower.accepts` accepts the transcript of the prover `.done proof`, which has no
queries. So the proof of the honest prover is a prover of the game of `Tower.extraction_le`. The
hypotheses `decodes` of `accepts` hold by `Tower.Drawn.decodes`. -/
theorem noteTower_accepts [Nonempty Coord] (drawn : (notes).Drawn)
    (table : Address (Wire E) total → Option Coord → Replay.CodeSlot E)
    (worked : ∀ input, table input none ∈ workZeros (Replay.CodeSlot E) 57) {tables : Fin 3 → Table}
    (live : ∀ g cell, 2 ^ ![20, 16, 16] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E} (lengths : ∀ g, (places g).length = ![20, 16, 16] g)
    (fit : Fit β key openKey (rootDecode E) β₁ key₁ openKey₁ code₁ key₂ openKey₂ code₂ key₃
      openKey₃ code₃ key₄ openKey₄ code₄ key₅ openKey₅ code₅ codeL tables) :
    Finite (Wire E) ∧ ∃ proof : ℕ → Wire E,
      (OracleComp.done proof :
        OracleComp (Address (Wire E) total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire E)).QueryBound 0 ∧
      (proof 0).data.commit = (fun g => (commit (key g) (tables g)).1) ∧
      (proof 0).data.place = places ∧
      (proof 0).data.value = (fun g => mle (places g) (lifted E (tables g))) ∧
      (notes).accepts ((notes).body.protocol.final (.done proof) table) := by
  have decoded := drawn.decodes
  have next := drawn.next
  have inside := drawn.inside
  dsimp only [noteTower, terminalOf] at next inside
  obtain ⟨proof, committed, placed, valued, holds⟩ := accepts β key openKey block (rootDecode E)
    rootMass β₁ key₁ openKey₁ block₁ code₁ 51 key₂ openKey₂ block₂ code₂ 31 key₃ openKey₃ block₃
    code₃ 31 key₄ openKey₄ block₄ code₄ 53 key₅ openKey₅ block₅ code₅ 53 matrix blockL codeL
    (next.symm.trans_lt inside) (notes).accepts table (fun _ b => (decoded _).1 b.1)
    (fun _ _ => (decoded _).2.1) (fun _ _ => (decoded _).2.2.1) (fun _ _ => (decoded _).2.2.2.1)
    (fun _ _ => (decoded _).2.2.2.2.1) (fun _ _ => (decoded _).2.2.2.2.2.1)
    (fun _ _ => (decoded _).2.2.2.2.2.2) worked live lengths fit
  refine ⟨inferInstance, proof, .done proof 0, committed, placed, valued, ?_⟩
  subst next
  exact holds

/-- `noteTower_accepts` for the zero tables, with no hypothesis on the fit (`fit_zero`): for
each table of the oracle whose work slots have 57 zero work bits, a prover with no queries exists
whose transcript `Tower.accepts` accepts. -/
theorem noteTower_zero [Nonempty Coord] (drawn : (notes).Drawn)
    (table : Address (Wire E) total → Option Coord → Replay.CodeSlot E)
    (worked : ∀ input, table input none ∈ workZeros (Replay.CodeSlot E) 57) :
    ∃ prover : OracleComp (Address (Wire E) total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire E),
      prover.QueryBound 0 ∧ (notes).accepts ((notes).body.protocol.final prover table) := by
  obtain ⟨-, proof, bounded, -, -, -, holds⟩ := noteTower_accepts β key openKey block β₁ key₁
    openKey₁ block₁ key₂ openKey₂ block₂ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅
    block₅ matrix blockL drawn table worked (tables := fun _ => 0) (fun _ _ _ => rfl)
    (places := fun g => List.replicate (![20, 16, 16] g) 0) (fun _ => List.length_replicate)
    (fit_zero β key openKey (rootDecode E) β₁ key₁ openKey₁ code₁ key₂ openKey₂ code₂ key₃
      openKey₃ code₃ key₄ openKey₄ code₄ key₅ openKey₅ code₅ codeL)
  exact ⟨_, bounded, holds⟩

/-- The accept event of `Tower.extraction_le` for the data `noteTower` with `Tower.Drawn` is not
empty: a table of the oracle and a prover with no queries exist whose transcript `Tower.accepts`
accepts (`noteTower_zero` for a table whose slots are a slot with 57 zero work bits, which exists
since a slot has `2^131` values or more, `card_codeSlot`). -/
theorem noteTower_played [Nonempty Coord] (drawn : (notes).Drawn) :
    ∃ (table : Address (Wire E) total → Option Coord → Replay.CodeSlot E)
      (prover : OracleComp (Address (Wire E) total) (Option Coord → Replay.CodeSlot E) (ℕ → Wire E)),
      prover.QueryBound 0 ∧ (notes).accepts ((notes).body.protocol.final prover table) := by
  obtain ⟨slot, worked⟩ : (workZeros (Replay.CodeSlot E) 57).Nonempty := Finset.card_pos.mp (by
    rw [workZeros_card]
    exact Nat.div_pos ((Nat.pow_le_pow_right (by decide) (by decide)).trans card_codeSlot)
      (Nat.two_pow_pos _))
  exact ⟨fun _ _ => slot, noteTower_zero β key openKey block β₁ key₁
    openKey₁ block₁ key₂ openKey₂ block₂ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅
    block₅ matrix blockL drawn _ fun _ => worked⟩

end Code

end NightstreamFPrime.Spec.Compression.Akita.Key20
