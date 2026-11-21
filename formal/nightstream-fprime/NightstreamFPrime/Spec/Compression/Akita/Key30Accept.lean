import NightstreamFPrime.Spec.Compression.Akita.Key30Root
import NightstreamFPrime.Spec.Compression.Akita.Extract.Page
import NightstreamFPrime.Spec.Compression.Akita.Extract.ZeroFit
import NightstreamFPrime.Spec.Compression.Akita.Extract.GroupComplete

/-! The verifier `offLinks` of the opening of Akita for a row of the schedule file
`crates/ns-compress/artifacts/key30.aks` accepts a proof (Akita, ePrint 2026/1983, revision of
2026-10-15, Theorem 10.9, Section 9 with (136), Figure 4 and Section 8.2). The theorem
`Key30.tower_suffix` bounds an event of this verifier, and `Key30.tower_complete` has hypotheses
on the wire messages, on the setup prefix and on the responses of the honest prover. This file
has no such hypothesis.

- `Note` has the data of a wire message other than the messages of the checks: the commitments,
  the points and the values of the three claims of the root, the claim of the root fold, the
  commitment of the witness of level 1 with the commitments, the values and the points of the
  two claims of level 1, the claim of level 1, the commitment and the claim of each level from
  level 2, and the state, the statement, the clear partials and the clear response of the
  terminal. `Wire` is `Page` for this data, with 4 cells of 8 elements and a number up to
  `3029804899`.
- `terminalOf` is the terminal that reads its statement and its clear partials from the message
  before its fold challenges.
- `zero_accepts`: for each table of the oracle whose slots decode to fold challenges, a proof
  exists whose transcript satisfies `offLinks`. It is the proof of the honest prover for the
  zero tables of the key polynomial, `W` and `V` and the zero setup prefix, for which each
  response is 0 (`Extract/ZeroFit.lean`, `stateFit_zero`). The response of the group of the setup
  prefix at level 1 is a sum of the blocks of the setup prefix, and a non-zero setup prefix makes
  each later witness non-zero, so for another setup prefix the fit conditions are not theorems:
  `Key30.tower_complete` has them as hypotheses. The matrices, the decoders of the fold
  challenges, the coordinates of their slots and the split `scan` of the public weight of the root
  are free. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key30

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The data of a wire message of the opening of the row `r` other than the messages of the
checks (Akita Figure 4 and (136): the messages of the prover). -/
structure Note (r : Row) (E : Type) [CommRing E] [Algebra Zq E] where
  commit : (g : Fin 3) → ((root r).shape g).Payload
  value : Fin 3 → E
  place : Fin 3 → List E
  claim : (root r).Claim E
  commit₁ : ((level₁ r).shape 1).Payload
  commits : (g : Fin 2) → ((level₁ r).shape g).Payload
  values : Fin 2 → E
  places : Fin 2 → List E
  claim₁ : (level₁ r).Claim E
  commit₂ : (level₂ r).shape.Payload
  claim₂ : Claim (level₂ r).opening E
  commit₃ : level₃.shape.Payload
  claim₃ : Claim level₃.opening E
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

/-- A wire message of the opening of the row `r`: the data `Note`, and 4 cells of 8 elements of
`E` and a number up to `3029804899` for the messages of the checks. -/
abbrev Wire (r : Row) (E : Type) [CommRing E] [Algebra Zq E] : Type :=
  Page (Note r E) E 4 8 3029804899

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The terminal with the matrix `A_term`, the fold challenges at the stage `start`, and the
statement and the clear partials of the message before this stage (Akita Section 8.2). -/
def terminalOf (r : Row) (total : ℕ)
    (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
      (Rq terminal.innerDegree))
    (block : Fin terminal.blocks → Coord) (decode : E → Option (Rq terminal.innerDegree))
    (start : ℕ) :
    Terminal terminal E (Wire r E) Coord E total
      (.euclidean normTerminal.operator normTerminal.limit) where
  matrix := matrix
  read point := (point.wire start).data.statement
  partials point := (point.wire start).data.partials
  response wire := wire.data.response
  block := block
  decode := decode
  weight := 0
  bound := 0
  start := start

variable {r : Row} (member : r ∈ rows)
  (β : ∀ g, Module.Basis (Fin ((root r).planes g)) Zq E)
  (key : (g : Fin 3) → Key ((root r).shape g)) (openKey : (root r).OpenKey)
  (block : (root r).Block → Coord)
  (decode : (g : Fin 3) → E → Option (Rq ((root r).carrierDegree g))) (mass : Fin 3 → ℕ)
  (scan : ∀ (wire : Wire r E) (challenge : (root r).Challenge),
    SetupScan (Batch.liftedReduction β ((wire.data.claim).statement key openKey) challenge
      rootVars (root_sized r member).1 8 quotientDigits₀ (root_sized r member).2.1 rootBatch
      (root_sized r member).2.2.1) setupVars 0)
  (β₁ : ∀ g, Module.Basis (Fin ((level₁ r).planes g)) Zq E)
  (key₁ : (g : Fin 2) → Key ((level₁ r).shape g)) (openKey₁ : (level₁ r).OpenKey)
  (block₁ : (level₁ r).Block → Coord)
  (decode₁ : (g : Fin 2) → E → Option (Rq ((level₁ r).carrierDegree g))) (mass₁ : Fin 2 → ℕ)
  (β₂ : Module.Basis (Fin (level₂ r).opening.planes) Zq E) (key₂ : Key (level₂ r).shape)
  (openKey₂ : OpeningKey (level₂ r).opening) (block₂ : Fin (level₂ r).shape.blocks → Coord)
  (decode₂ : E → Option (Rq (level₂ r).opening.carrierDegree)) (weight₂ : ℕ)
  (β₃ : Module.Basis (Fin level₃.opening.planes) Zq E) (key₃ : Key level₃.shape)
  (openKey₃ : OpeningKey level₃.opening) (block₃ : Fin level₃.shape.blocks → Coord)
  (decode₃ : E → Option (Rq level₃.opening.carrierDegree)) (weight₃ : ℕ)
  (key₄ : Key level₄.shape) (openKey₄ : OpeningKey level₄.opening)
  (block₄ : Fin level₄.shape.blocks → Coord)
  (decode₄ : E → Option (Rq level₄.shape.innerDegree)) (weight₄ : ℕ)
  (key₅ : Key level₅.shape) (openKey₅ : OpeningKey level₅.opening)
  (block₅ : Fin level₅.shape.blocks → Coord)
  (decode₅ : E → Option (Rq level₅.shape.innerDegree)) (weight₅ : ℕ)
  (key₆ : Key level₆.shape) (openKey₆ : OpeningKey level₆.opening)
  (block₆ : Fin level₆.shape.blocks → Coord)
  (decode₆ : E → Option (Rq level₆.shape.innerDegree)) (weight₆ : ℕ)
  (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
    (Rq terminal.innerDegree))
  (blockL : Fin terminal.blocks → Coord) (decodeL : E → Option (Rq terminal.innerDegree))

local notation "zeroth" =>
  offBatch β key openKey (Note.claim ∘ Page.data : Wire r E → _) rootVars
    (And.left (root_sized r member)) 8 quotientDigits₀
    (And.left (And.right (root_sized r member))) rootBatch
    (And.left (And.right (And.right (root_sized r member)))) block decode Page.read mass (0 + 1)
    setupVars 0 scan

local notation "first" =>
  Tier.pinned (liftBatch β₁ key₁ openKey₁ (Note.claim₁ ∘ Page.data : Wire r E → _) vars₁
    (And.left (level₁_sized r member)) 16 quotientDigits
    (And.left (And.right (level₁_sized r member))) batch₁
    (And.left (And.right (And.right (level₁_sized r member)))) block₁ decode₁ Page.read mass₁
    (Tier.finish zeroth + 1)) (Batch.Pinned 0 (Prod.fst (commit (key₁ 0) 0)) 0)
    (fun ω _ => Batch.pinBreak key₁ openKey₁ _ 0 (Prod.snd (commit (key₁ 0) 0)) ω)

local notation "second" =>
  liftFold β₂ key₂ openKey₂ (Note.claim₂ ∘ Page.data : Wire r E → _) (Key20.Level.vars (level₂ r))
    (And.left (level₂_sized r member)) quotientDigits
    (And.left (And.right (level₂_sized r member))) (Key20.Level.batch (level₂ r))
    (And.left (And.right (And.right (level₂_sized r member)))) block₂ decode₂ Page.read weight₂
    (Tier.finish first)

local notation "third" =>
  Tier.tensored (liftFold β₃ key₃ openKey₃ (Note.claim₃ ∘ Page.data : Wire r E → _) level₃.vars
    (And.left level₃_sized) quotientDigits (And.left (And.right level₃_sized)) level₃.batch
    (And.left (And.right (And.right level₃_sized))) block₃ decode₃ Page.read weight₃
    (Tier.finish second)) level₃.tail

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
    Page.read weight₅ norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail

local notation "sixth" =>
  Tier.tensored (committedTraceSq (fold := level₆.fold) key₆ openKey₆
    (Note.claim₆ ∘ Page.data : Wire r E → _) level₆.vars
    (And.left (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))) level₆.batch
    (And.left (And.right (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))))
    block₆ decode₆ Page.read weight₆ norm₆.operator norm₆.limit (Tier.finish fifth))
    level₆.tail

/-- The verifier `offLinks` of the opening of a row of the schedule `key30.aks` accepts a proof:
the event of `Key30.tower_suffix` is not empty for the zero setup prefix. The wire messages are
`Wire`, the batching root is at the stage 0, the setup prefix `S^♭` is 0 with the commitment and
the reference decommitment of `commit` for the zero table, and the terminal is `terminalOf` at
the stage after the level 6. For each matrix of each level, each decoder of the fold challenges,
each split `scan` of the public weight of the root (Akita (132)), each table of the oracle whose
slots decode to fold challenges and each acceptance condition of the transcript protocol, a proof
exists whose complete transcript satisfies `offLinks`: the link `RootLinked` of the batching
root, the link `Tier.Offloaded` from the root fold to the two claims of level 1, the link
`RootLinked` of their batching stage, the links `Tier.Linked` of the levels 1 to 3, the links
`Tier.TraceLinked` of the levels 3 to 6, the link `Terminal.Linked` and the checks
`Terminal.Passed`. The proof is that of the honest prover for the zero tables of the key
polynomial, `W` and `V` (`Key30.tower_complete` with `batchFit_zero`, `stateFit_zero`,
`liftFit_zero` and `traceFit_zero`). The response of the group of the setup prefix at level 1 is a
sum of the blocks of the setup prefix, and a non-zero setup prefix makes each later witness
non-zero, so the theorem is for the zero setup prefix only. `inside`: the fold challenge of the
terminal is a stage of the transcript. `worked`: each work slot of the table has 57 zero work
bits. -/
theorem zero_accepts {total : ℕ} (inside : Tier.finish sixth < total)
    (accept : Prefix (Wire r E) (Option Coord → E) total → Prop)
    (table : Prefix (Wire r E) (Option Coord → E) total → Option Coord → E)
    (decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input b, (decode₁ b.1 (table input (some (block₁ b)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodes₆ : ∀ input i, (decode₆ (table input (some (block₆ i)))).isSome)
    (decodesL : ∀ input i, (decodeL (table input (some (blockL i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros E 57) :
    ∃ proof : ℕ → Wire r E,
      offLinks (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
        (route₅ := Route.euclidean norm₅.operator norm₅.limit)
        (route₆ := Route.euclidean norm₆.operator norm₆.limit) zeroth first second third fourth
        fifth sixth (terminalOf r total matrix blockL decodeL (Tier.finish sixth)) 0 Page.read
        (Note.commit ∘ Page.data : Wire r E → _) (Note.value ∘ Page.data : Wire r E → _)
        (Note.place ∘ Page.data : Wire r E → _) (Note.commits ∘ Page.data : Wire r E → _)
        (Note.values ∘ Page.data : Wire r E → _) (Note.places ∘ Page.data : Wire r E → _)
        (commit (key₁ 0) 0).1 (Note.commit₁ ∘ Page.data : Wire r E → _)
        (Note.commit₂ ∘ Page.data : Wire r E → _) (Note.commit₃ ∘ Page.data : Wire r E → _)
        (And.left quarters) (And.left (And.right quarters))
        (And.left (And.right (And.right quarters))) (And.right (And.right (And.right quarters)))
        (Note.commit₄ ∘ Page.data : Wire r E → _) (Note.commit₅ ∘ Page.data : Wire r E → _)
        (Note.commit₆ ∘ Page.data : Wire r E → _) (Note.state ∘ Page.data : Wire r E → _)
        ((transcript total accept).final (.done proof) table) := by
  obtain ⟨start, started⟩ : ∃ start, start = Tier.finish sixth := ⟨_, rfl⟩
  rw [← started] at inside ⊢
  obtain ⟨base, based⟩ : ∃ base : Note r E, base = {
      commit := fun g => (commit (key g) 0).1
      value := fun g => mle (List.replicate (![30, r.vars, 28] g) 0) (lifted E 0)
      place := fun g => List.replicate (![30, r.vars, 28] g) 0
      claim := Batch.honestClaim β key openKey (fun _ => 0)
        (fun g => Point.ofSegments ((root r).opening g) [] [] [] []) 0
      commit₁ := (commitSource (key₁ 1) 0).1
      commits := fun g => (commit (key₁ g) 0).1
      values := 0
      places := fun _ => []
      claim₁ := Batch.honestClaim β₁ key₁ openKey₁ (fun _ => 0)
        (fun g => Point.ofSegments ((level₁ r).opening g) [] [] [] []) 0
      commit₂ := (commitSource key₂ 0).1
      claim₂ := sourceClaim β₂ key₂ openKey₂ 0 (Point.ofSegments (level₂ r).opening [] [] [] [])
      commit₃ := (commitSource key₃ 0).1
      claim₃ := sourceClaim β₃ key₃ openKey₃ 0 (Point.ofSegments level₃.opening [] [] [] [])
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
  have padded : ((level₁ r).shape 0).pad 0 = 0 := funext fun cell => by
    unfold Shape.pad
    split <;> rfl
  have registered : Batch.Registered (key₁ 0) (commit (key₁ 0) 0).1 0 (commit (key₁ 0) 0).2 := by
    have commits := Batch.registered_commit (S := (level₁ r).shape 0) honest.2 (key₁ 0) 0
    rwa [padded] at commits
  have openings := level₁_openings r member
  have opens : ∀ g, ∃ table, (offState (key₁ 1) (commit (key₁ 0) 0).1 (commit (key₁ 0) 0).2 0
      g).2.Opens (key₁ g) (offState (key₁ 1) (commit (key₁ 0) 0).1 (commit (key₁ 0) 0).2 0 g).1
      table := Fin.forall_fin_two.mpr
    ⟨⟨0, registered.opens⟩, ⟨_, commitSource_opens (level₁_bounded r member) (key₁ 1) _⟩⟩
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
    (commit₁ := Note.commit₁ ∘ Page.data) (commit₂ := Note.commit₂ ∘ Page.data)
    (commit₃ := Note.commit₃ ∘ Page.data) (commit₄ := Note.commit₄ ∘ Page.data)
    (commit₅ := Note.commit₅ ∘ Page.data) (commit₆ := Note.commit₆ ∘ Page.data)
    (stateOf := Note.state ∘ Page.data) (statementOf := Note.statement ∘ Page.data)
    (partialsOf := Note.partials ∘ Page.data)
    (Page.sends (by decide)) le_rfl le_rfl (by decide)
    (Page.carries Note.claim fun target => ⟨{ base with claim := target }, rfl⟩)
    (Page.carries (fun note : Note r E => (note.commit₁, note.commits, note.values, note.places))
      fun target => ⟨{ base with
        commit₁ := target.1, commits := target.2.1, values := target.2.2.1,
        places := target.2.2.2 }, rfl⟩)
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
    registered started inside (fun _ => rfl) (fun _ => rfl) table decodes decodes₁ decodes₂ decodes₃
    decodes₄ decodes₅ decodes₆ decodesL worked
    (onPath_nil accept table (⟨base, fun _ => (fun _ => 0, 0)⟩ : Wire r E)) rfl
    (tables := fun _ => 0) (fun _ _ _ => rfl) (by rw [based]; rfl)
    (fun g => by rw [based]; exact List.length_replicate) (by rw [based]; rfl)
    (batchFit_zero (honest.1 r member) (fun g => (root_folds r member g).1)
      (fun g => (root_folds r member g).2) (root_path r member) (by decide)
      (stateFit_zero (fun g => (openings g).1) (fun g => (openings g).2)
        (fun g => (level₁_folds r member g).1) (fun g => (level₁_folds r member g).2)
        (level₁_path r member) opens zero (by decide)
        (liftFit_zero (lifts.1 r member).1 (lifts.1 r member).2.1 (lifts.1 r member).2.2.1
          (lifts.1 r member).2.2.2 (by decide : 2 ≤ 16)
          (liftFit_zero lifts.2.1 lifts.2.2.1 lifts.2.2.2.1 lifts.2.2.2.2 (by decide)
            (traceFit_zero quarters.1 (by decide) (by decide) trivial
              (traceFit_zero quarters.2.1 (by decide) (by decide)
                (by rw [vecNormSq_zero]; exact Nat.zero_le _)
                (traceFit_zero quarters.2.2.1 (by decide) (by decide)
                  (by rw [vecNormSq_zero]; exact Nat.zero_le _) fun c _ => by
                    rw [packed_zero, traceSource_zero, traceResponse_zero, vecNormSq_zero]
                    exact Nat.zero_le _)))))))
  obtain ⟨proof, _, holds⟩ := OnPath.accepts on whole each
  exact ⟨proof, holds⟩

end NightstreamFPrime.Spec.Compression.Akita.Key30
