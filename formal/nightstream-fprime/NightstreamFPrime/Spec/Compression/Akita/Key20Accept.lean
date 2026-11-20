import NightstreamFPrime.Spec.Compression.Akita.Key20Root
import NightstreamFPrime.Spec.Compression.Akita.Extract.Page
import NightstreamFPrime.Spec.Compression.Akita.Extract.ZeroFit

/-! The verifier `towerLinks` of the opening of Akita for the schedule file
`crates/ns-compress/artifacts/key20.aks` accepts a proof (Akita, ePrint 2026/1983, revision of
2026-10-15, Theorem 10.9, Figure 4 and Section 8.2). The theorem `Key20.tower_suffix` bounds an
event of this verifier, and `Key20.tower_complete` has hypotheses on the wire messages and on the
responses of the honest prover. This file has no such hypothesis.

- `Note` has the data of a wire message other than the messages of the checks: the commitments,
  the points and the values of the three claims, the claim of the root fold, the commitment and
  the claim of each level, and the state, the statement, the clear partials and the clear
  response of the terminal. `Wire` is `Page` for this data, with 4 cells of 8 elements and a
  number up to `3029804899`.
- `terminalOf` is the terminal that reads its statement and its clear partials from the message
  before its fold challenges.
- `zero_accepts`: for each table of the oracle whose slots decode to fold challenges, a proof
  exists whose transcript satisfies `towerLinks`. It is the proof of the honest prover for the
  zero tables of the key polynomial, `W` and `V`, for which each response is 0
  (`Extract/ZeroFit.lean`). The matrices, the decoders of the fold challenges and the coordinates
  of their slots are free. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key20

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

/-- The data of a wire message of the opening other than the messages of the checks (Akita
Figure 4: the messages of the prover). -/
structure Note (E : Type) [CommRing E] [Algebra Zq E] where
  commit : (g : Fin 3) → (root.shape g).Payload
  value : Fin 3 → E
  place : Fin 3 → List E
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

/-- A wire message of the opening: the data `Note`, and 4 cells of 8 elements of `E` and a
number up to `3029804899` for the messages of the checks. -/
abbrev Wire (E : Type) [CommRing E] [Algebra Zq E] : Type := Page (Note E) E 4 8 3029804899

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The terminal with the matrix `A_term`, the fold challenges at the stage `start`, and the
statement and the clear partials of the message before this stage (Akita Section 8.2). -/
def terminalOf (total : ℕ)
    (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
      (Rq terminal.innerDegree))
    (block : Fin terminal.blocks → Coord) (decode : E → Option (Rq terminal.innerDegree))
    (start : ℕ) :
    Terminal terminal E (Wire E) Coord E total
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

variable (β : ∀ g, Module.Basis (Fin (root.planes g)) Zq E)
  (key : (g : Fin 3) → Key (root.shape g)) (openKey : root.OpenKey) (block : root.Block → Coord)
  (decode : (g : Fin 3) → E → Option (Rq (root.carrierDegree g))) (mass : Fin 3 → ℕ)
  (β₁ : Module.Basis (Fin level₁.opening.planes) Zq E) (key₁ : Key level₁.shape)
  (openKey₁ : OpeningKey level₁.opening) (block₁ : Fin level₁.shape.blocks → Coord)
  (decode₁ : E → Option (Rq level₁.opening.carrierDegree)) (weight₁ : ℕ)
  (key₂ : Key level₂.shape) (openKey₂ : OpeningKey level₂.opening)
  (block₂ : Fin level₂.shape.blocks → Coord)
  (decode₂ : E → Option (Rq level₂.shape.innerDegree)) (weight₂ : ℕ)
  (key₃ : Key level₃.shape) (openKey₃ : OpeningKey level₃.opening)
  (block₃ : Fin level₃.shape.blocks → Coord)
  (decode₃ : E → Option (Rq level₃.shape.innerDegree)) (weight₃ : ℕ)
  (key₄ : Key level₄.shape) (openKey₄ : OpeningKey level₄.opening)
  (block₄ : Fin level₄.shape.blocks → Coord)
  (decode₄ : E → Option (Rq level₄.shape.innerDegree)) (weight₄ : ℕ)
  (key₅ : Key level₅.shape) (openKey₅ : OpeningKey level₅.opening)
  (block₅ : Fin level₅.shape.blocks → Coord)
  (decode₅ : E → Option (Rq level₅.shape.innerDegree)) (weight₅ : ℕ)
  (matrix : Matrix (Fin terminal.innerRank) (Fin terminal.blockLength × Fin terminal.sourceDigits)
    (Rq terminal.innerDegree))
  (blockL : Fin terminal.blocks → Coord) (decodeL : E → Option (Rq terminal.innerDegree))

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

/-- The verifier `towerLinks` of the opening of the schedule `key20.aks` accepts a proof: the
event of `Key20.tower_suffix` is not empty. The wire messages are `Wire`, the batching root is at
the stage 0, and the terminal is `terminalOf` at the stage after the level 5. For each matrix of
each level, each decoder of the fold challenges, each table of the oracle whose slots decode to
fold challenges and each acceptance condition of the transcript protocol, a proof exists whose
complete transcript has the link `RootLinked` of the batching root, the link `Tier.Linked` from
the root to the level 1, the links `Tier.TraceLinked` of the levels 1 to 5, the link
`Terminal.Linked` and the checks `Terminal.Passed`. The proof is that of the honest prover for
the zero tables of the key polynomial, `W` and `V` (`Key20.tower_complete` with `batchFit_zero`,
`liftFit_zero` and `traceFit_zero`). `inside`: the fold challenge of the terminal is a stage of
the transcript. -/
theorem zero_accepts {total : ℕ} (inside : Tier.finish fifth < total)
    (accept : Prefix (Wire E) (Coord → E) total → Prop)
    (table : Prefix (Wire E) (Coord → E) total → Coord → E)
    (decodes : ∀ input b, (decode b.1 (table input (block b))).isSome)
    (decodes₁ : ∀ input i, (decode₁ (table input (block₁ i))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (block₂ i))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (block₃ i))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (block₄ i))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (block₅ i))).isSome)
    (decodesL : ∀ input i, (decodeL (table input (blockL i))).isSome) :
    ∃ proof : ℕ → Wire E,
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
  obtain ⟨base, based⟩ : ∃ base : Note E, base = {
      commit := fun g => (commit (key g) 0).1
      value := fun g => mle (List.replicate (![20, 16, 16] g) 0) (lifted E 0)
      place := fun g => List.replicate (![20, 16, 16] g) 0
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
      response := 0 } := ⟨_, rfl⟩
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
    rfl inside (fun _ => rfl) (fun _ => rfl) table decodes decodes₁ decodes₂ decodes₃ decodes₄
    decodes₅ decodesL
    (onPath_nil accept table (⟨base, fun _ => (fun _ => 0, 0)⟩ : Wire E)) rfl
    (tables := fun _ => 0) (fun _ _ _ => rfl) (by rw [based]; rfl)
    (fun g => by rw [based]; exact List.length_replicate) (by rw [based]; rfl)
    (batchFit_zero root_honest (fun g => (root_folds g).1) (fun g => (root_folds g).2) paths.1
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
                  exact Nat.zero_le _))))))
  obtain ⟨proof, _, holds⟩ := OnPath.accepts on whole each
  exact ⟨proof, holds⟩

end NightstreamFPrime.Spec.Compression.Akita.Key20
