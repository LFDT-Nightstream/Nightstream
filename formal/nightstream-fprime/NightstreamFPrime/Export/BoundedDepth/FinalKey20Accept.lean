import NightstreamFPrime.Export.BoundedDepth.FinalKey20
import NightstreamFPrime.Spec.Compression.Akita.Key20Accept

/-! The final verifier over the Akita opening of the schedule file
`crates/ns-compress/artifacts/key20.aks` accepts the honest prover, for one type of wire messages
(SPEC.md Algorithm 9 and Algorithm 10; Akita, ePrint 2026/1983, revision of 2026-11-23,
Theorem 10.9 and Figure 4).

`FinalKey20.complete` has hypotheses on the wire messages. This file gives them for the messages
`Sheet`.

- `Sheet` is `Page` for the values of the first message (`FinalAkita.First`), the points of the
  three opening claims as lists, and the data `Key20.Note` of the Akita opening, with `slots`
  cells of `width` elements of `K` and a number up to `limit`. The points of `Key20.Note` are
  vectors, and the run of the compression argument (`Compose.accepts`) carries lists.
- `sheetBound` has the commitments that a stage input fixes: those of the key polynomial and of
  `W` in the first message, and that of `V` in the message after the lookup challenges.
  `sheetTerminal` is the terminal that reads its statement and its clear partials from the
  message before its fold challenges.
- `accepts`: for a setup with `FinalKey20.Use` and each table of the oracle with the conditions of
  `FinalKey20.complete`, a proof exists whose first message has the values `given` and whose
  complete transcript satisfies `towerAccepts`, the verifier of `FinalKey20.sound`. The honest
  prover of Π_CCS on the fresh instance (`leads`) and the fit conditions (`fit`) are
  hypotheses. -/

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
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier (Mode Claims)
open NightstreamFPrime.Export.BoundedDepth.FinalAkita
open Key20 (root level₁ level₂ level₃ level₄ level₅ norm₃ norm₄ norm₅ normTerminal rootVars
  rootBatch quotientDigits root_sized root_room level₁_sized sized₂ sized₃ sized₄ sized₅ quarters)

attribute [local instance low] Classical.propDecidable

noncomputable local instance : Algebra Zq K := inferInstanceAs (Algebra F K)

noncomputable local instance : Degree2 K :=
  ⟨AlgEquiv.ofRingEquiv (f := carrierEquiv.ringEquiv) fun _ => rfl⟩

/-- A wire message of a final proof: the values `First` of the first message, the points of the
three opening claims, the data `Key20.Note` of the Akita opening, `slots` cells of `width`
elements of `K` and a number up to `limit` for the messages of the checks, and the nonce of the
response search. -/
abbrev Sheet (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) (slots width limit : ℕ) : Type :=
  Page ((First logicalWidth publicFits × (Fin 3 → List K)) × Key20.Note K) K slots width limit

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode} (setup : FinalVerifier.Setup mode logicalWidth publicFits)
  (packageWidths : Fin 2 → ℕ) (step : AppState → AppWitness → AppState) (cap : ℕ)
  (control : AppState → AppWitness → Fin 2) (shape : Claims logicalWidth publicFits)
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {total slots width limit : ℕ}

/-- The commitments that a stage input fixes: the commitments of the key polynomial and of `W`
of the first message, and the commitment of `V` of the message after the lookup challenges. -/
noncomputable def sheetBound (anchor : Prefix (Sheet logicalWidth publicFits slots width limit)
  (Option Coord → K) total)
    (g : Fin 3) : (root.shape g).Payload :=
  if g = 2 then
    (anchor.wire
      ((Compression.Fresh.costs CompressionFresh.oneShape 8).length + 1)).data.2.commit g
  else (anchor.wire 0).data.2.commit g

/-- The terminal with the matrix `A_term`, the fold challenges at the stage `start`, and the
statement, the clear partials and the nonce of the response search of the message before this
stage (Akita Section 8.2). -/
def sheetTerminal (total : ℕ)
    (matrix : Matrix (Fin Key20.terminal.innerRank)
      (Fin Key20.terminal.blockLength × Fin Key20.terminal.sourceDigits)
      (Rq Key20.terminal.innerDegree))
    (block : Fin Key20.terminal.blocks → Coord)
    (decode : K → Option (Rq Key20.terminal.innerDegree)) (start : ℕ) :
    Terminal Key20.terminal K (Sheet logicalWidth publicFits slots width limit) Coord K total
      (.euclidean normTerminal.operator normTerminal.limit) where
  matrix := matrix
  read point := (point.wire start).data.2.statement
  partials point := (point.wire start).data.2.partials
  response wire := wire.data.2.response
  nonce wire := wire.nonce
  block := block
  decode := decode
  weight := 0
  bound := 0
  start := start

variable (β : ∀ g, Module.Basis (Fin (root.planes g)) Zq K)
  (key : (g : Fin 3) → Key (root.shape g)) (openKey : root.OpenKey) (block : root.Block → Coord)
  (decode : (g : Fin 3) → K → Option (Rq (root.carrierDegree g))) (mass : Fin 3 → ℕ)
  (β₁ : Module.Basis (Fin level₁.opening.planes) Zq K) (key₁ : Key level₁.shape)
  (openKey₁ : OpeningKey level₁.opening) (block₁ : Fin level₁.shape.blocks → Coord)
  (decode₁ : K → Option (Rq level₁.opening.carrierDegree)) (weight₁ : ℕ)
  (key₂ : Key level₂.shape) (openKey₂ : OpeningKey level₂.opening)
  (block₂ : Fin level₂.shape.blocks → Coord)
  (decode₂ : K → Option (Rq level₂.shape.innerDegree)) (weight₂ : ℕ)
  (key₃ : Key level₃.shape) (openKey₃ : OpeningKey level₃.opening)
  (block₃ : Fin level₃.shape.blocks → Coord)
  (decode₃ : K → Option (Rq level₃.shape.innerDegree)) (weight₃ : ℕ)
  (key₄ : Key level₄.shape) (openKey₄ : OpeningKey level₄.opening)
  (block₄ : Fin level₄.shape.blocks → Coord)
  (decode₄ : K → Option (Rq level₄.shape.innerDegree)) (weight₄ : ℕ)
  (key₅ : Key level₅.shape) (openKey₅ : OpeningKey level₅.opening)
  (block₅ : Fin level₅.shape.blocks → Coord)
  (decode₅ : K → Option (Rq level₅.shape.innerDegree)) (weight₅ : ℕ)
  (matrix : Matrix (Fin Key20.terminal.innerRank)
    (Fin Key20.terminal.blockLength × Fin Key20.terminal.sourceDigits)
    (Rq Key20.terminal.innerDegree))
  (blockL : Fin Key20.terminal.blocks → Coord)
  (decodeL : K → Option (Rq Key20.terminal.innerDegree))

local notation "Leaf" => Sheet logicalWidth publicFits slots width limit

local notation "noteOf" => (Prod.snd ∘ Page.data : Leaf → Key20.Note K)

local notation "Content" =>
  ((First logicalWidth publicFits × (Fin 3 → List K)) × Key20.Note K)

local notation "placeOf" => (Prod.snd ∘ Prod.fst ∘ Page.data : Leaf → Fin 3 → List K)

local notation "C" =>
  compose setup packageWidths step cap control shape (Page.read : Reader Leaf)
    (Prod.fst ∘ Prod.fst ∘ Page.data : Leaf → _) (sheetBound (Coord := Coord) (total := total))

local notation "vf" => FinalVerifier.Setup.verifier setup shape

local notation "zeroth" =>
  liftBatch β key openKey (Key20.Note.claim ∘ noteOf) rootVars (And.left root_sized) 8
    quotientDigits root_room rootBatch (And.left (And.right root_sized)) block decode
    (Compose.read C) mass (Compose.root C vf + 1)

local notation "first" =>
  Tier.tensored (liftFold β₁ key₁ openKey₁ (Key20.Note.claim₁ ∘ noteOf) level₁.vars
    (And.left level₁_sized) quotientDigits (And.left (And.right level₁_sized)) level₁.batch
    (And.left (And.right (And.right level₁_sized))) block₁ decode₁ (Compose.read C) weight₁
    (Tier.finish zeroth)) level₁.tail

local notation "second" =>
  Tier.tensored (committedTrace (fold := level₂.fold) key₂ openKey₂
    (Key20.Note.claim₂ ∘ noteOf) level₂.vars (And.left sized₂) level₂.batch
    (And.left (And.right sized₂)) block₂ decode₂ (Compose.read C) weight₂ (Tier.finish first))
    level₂.tail

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := level₃.fold) key₃ openKey₃
    (Key20.Note.claim₃ ∘ noteOf) level₃.vars (And.left sized₃) level₃.batch
    (And.left (And.right sized₃)) block₃ decode₃ (Compose.read C) weight₃
    norm₃.operator norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) key₄ openKey₄
    (Key20.Note.claim₄ ∘ noteOf) level₄.vars (And.left sized₄) level₄.batch
    (And.left (And.right sized₄)) block₄ decode₄ (Compose.read C) weight₄
    norm₄.operator norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅
    (Key20.Note.claim₅ ∘ noteOf) level₅.vars (And.left sized₅) level₅.batch
    (And.left (And.right sized₅)) block₅ decode₅ (Compose.read C) weight₅
    norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail

/-- The final verifier `towerAccepts` of `sound` accepts the honest prover of a final proof with
the wire messages `Sheet`, without the probability that a response of the Akita opening does not
fit (`complete` with `Page.sends`, `Cell.sendable_values` and `Page.carries`). The terminal is
`sheetTerminal` at the stage after the level 5, and the commitments of a stage input are
`sheetBound`. For a setup with `Use`, each matrix of each level, each decoder of the fold
challenges and each table of the oracle with `calm`, `regular` and `decodes` to `decodesL`, a
proof exists whose first message has the values `given` and whose complete transcript satisfies
`towerAccepts`.

`wide`, `capped`, `room`: a cell has 8 elements and a number up to `3029804899`, and a message
has 4 cells. `inside`: the fold challenge of the terminal is a stage of the transcript. `checks`
and `gate`: the values `given` pass steps 1 to 5 of SPEC.md Algorithm 10. `W` is the table of
the witnesses and the counts, with honest counts and 0 from the cell `2^16`. `leads`: the prover
of Π_CCS on the fresh instance has a message of a cell for each draw sequence such that the run
ends in a probe whose claims satisfy `Shaped` and `Relation` for `W`. No Lean result gives
`leads`. `fit`: the fit conditions of `Key20.tower_complete` for the tables of the key
polynomial, `W` and the honest inverses of each lookup challenges that are no table values. -/
theorem accepts (use : Use setup shape) (wide : 8 ≤ width) (capped : 3029804899 ≤ limit)
    (room : 3 < slots) (inside : Tier.finish fifth < total)
    (accept : Prefix Leaf (Option Coord → K) total → Prop)
    (table : Address Leaf total → Option Coord → K)
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
    (decodesL : ∀ input i, (decodeL (table input (some (blockL i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros K 57)
    (given : First logicalWidth publicFits)
    (checks : FinalVerifier.Checks mode given.statement (given.claims shape))
    (gate : setup.Gate (given.claims shape)) (W : Table)
    (counts : Compression.Verifier.HonestCounts (setup.params shape) W)
    (tail : ∀ cell, 2 ^ 16 ≤ cell → W cell = 0)
    (leads : WinsOff (Compression.Verifier.Rejects (setup.params shape))
      (Cell.Sendable K width limit) slots (setup.freshRun given.statement (given.claims shape) W)
      (Compression.Fresh.costs CompressionFresh.oneShape 8).length 0 fun result _ _ =>
        ∃ probe, result = some probe ∧
          Compression.Verifier.Shaped (setup.params shape)
            (setup.claimsOf given.statement (given.claims shape) probe) ∧
          Compression.Verifier.Relation (setup.params shape)
            (setup.claimsOf given.statement (given.claims shape) probe) W)
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
        (∀ i, ∃ slot, decodeL slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource
          (And.right (And.right (And.right (And.right quarters)))) (packed w₅ : ℕ → K))) ≤
          normTerminal.limit) :
    ∃ proof : ℕ → Leaf, (proof 0).data.1.1 = given ∧
      towerAccepts (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) C vf
        Route.coefficient (Route.euclidean norm₃.operator norm₃.limit)
        (Route.euclidean norm₄.operator norm₄.limit) (Route.euclidean norm₅.operator norm₅.limit)
        zeroth first second third fourth fifth
        (sheetTerminal total matrix blockL decodeL (Tier.finish fifth))
        (Key20.Note.commit ∘ noteOf) (Key20.Note.value ∘ noteOf) placeOf (And.left quarters)
        (And.left (And.right quarters)) (And.left (And.right (And.right quarters)))
        (And.left (And.right (And.right (And.right quarters))))
        (And.right (And.right (And.right (And.right quarters))))
        (Key20.Note.commit₁ ∘ noteOf) (Key20.Note.commit₂ ∘ noteOf)
        (Key20.Note.commit₃ ∘ noteOf) (Key20.Note.commit₄ ∘ noteOf)
        (Key20.Note.commit₅ ∘ noteOf) (Key20.Note.state ∘ noteOf)
        ((transcript total accept).final (.done proof) table) := by
  obtain ⟨start, started⟩ : ∃ start, start = Tier.finish fifth := ⟨_, rfl⟩
  rw [← started] at inside ⊢
  have base : Key20.Note K := {
    commit := fun g => (Compression.Akita.commit (key g) 0).1
    value := fun _ => 0
    point := fun _ _ => 0
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
    response := 0 }
  have stage : (Compose.front C).finish =
      (Compression.Fresh.costs CompressionFresh.oneShape 8).length + 1 := by
    show 0 + (_ ++ [_]).length = _
    rw [List.length_append, List.length_singleton, Nat.zero_add]
    rfl
  obtain ⟨proof, zero, holds⟩ := complete setup packageWidths step cap control shape
    (Page.read : Reader Leaf) (Prod.fst ∘ Prod.fst ∘ Page.data : Leaf → _)
    (sheetBound (Coord := Coord) (total := total)) (β := β) (key := key) (openKey := openKey)
    (claim := (Key20.Note.claim ∘ noteOf)) (block := block) (decode := decode)
    (mass := mass) (β₁ := β₁) (key₁ := key₁) (openKey₁ := openKey₁)
    (claim₁ := (Key20.Note.claim₁ ∘ noteOf)) (block₁ := block₁) (decode₁ := decode₁)
    (weight₁ := weight₁) (key₂ := key₂) (openKey₂ := openKey₂)
    (claim₂ := (Key20.Note.claim₂ ∘ noteOf)) (block₂ := block₂) (decode₂ := decode₂)
    (weight₂ := weight₂) (key₃ := key₃) (openKey₃ := openKey₃)
    (claim₃ := (Key20.Note.claim₃ ∘ noteOf)) (block₃ := block₃) (decode₃ := decode₃)
    (weight₃ := weight₃) (key₄ := key₄) (openKey₄ := openKey₄)
    (claim₄ := (Key20.Note.claim₄ ∘ noteOf)) (block₄ := block₄) (decode₄ := decode₄)
    (weight₄ := weight₄) (key₅ := key₅) (openKey₅ := openKey₅)
    (claim₅ := (Key20.Note.claim₅ ∘ noteOf)) (block₅ := block₅) (decode₅ := decode₅)
    (weight₅ := weight₅) (L := sheetTerminal total matrix blockL decodeL start)
    (commit := (Key20.Note.commit ∘ noteOf))
    (value := (Key20.Note.value ∘ noteOf))
    (place := placeOf)
    (commit₁ := (Key20.Note.commit₁ ∘ noteOf))
    (commit₂ := (Key20.Note.commit₂ ∘ noteOf))
    (commit₃ := (Key20.Note.commit₃ ∘ noteOf))
    (commit₄ := (Key20.Note.commit₄ ∘ noteOf))
    (commit₅ := (Key20.Note.commit₅ ∘ noteOf))
    (stateOf := (Key20.Note.state ∘ noteOf))
    (statementOf := (Key20.Note.statement ∘ noteOf))
    (partialsOf := (Key20.Note.partials ∘ noteOf)) (accept := accept) use
    (Page.sends (by omega)) wide capped room
    (fun values zero => Cell.sendable_values (by omega) fun index large => zero index (by omega))
    (Page.carries (fun data : Content => data.2.commit)
      fun target => ⟨((given, fun _ => []), { base with commit := target }), rfl⟩)
    (Page.carries (fun data : Content => (data.2.commit, data.2.value, data.1.2))
      fun target => ⟨((given, target.2.2),
        { base with commit := target.1, value := target.2.1 }), rfl⟩)
    (Page.carries (fun data : Content => data.2.claim)
      fun target => ⟨((given, fun _ => []), { base with claim := target }), rfl⟩)
    (Page.carries (fun data : Content => (data.2.commit₁, data.2.claim₁)) fun target =>
      ⟨((given, fun _ => []), { base with commit₁ := target.1, claim₁ := target.2 }), rfl⟩)
    (Page.carries (fun data : Content => (data.2.commit₂, data.2.claim₂)) fun target =>
      ⟨((given, fun _ => []), { base with commit₂ := target.1, claim₂ := target.2 }), rfl⟩)
    (Page.carries (fun data : Content => (data.2.commit₃, data.2.claim₃)) fun target =>
      ⟨((given, fun _ => []), { base with commit₃ := target.1, claim₃ := target.2 }), rfl⟩)
    (Page.carries (fun data : Content => (data.2.commit₄, data.2.claim₄)) fun target =>
      ⟨((given, fun _ => []), { base with commit₄ := target.1, claim₄ := target.2 }), rfl⟩)
    (Page.carries (fun data : Content => (data.2.commit₅, data.2.claim₅)) fun target =>
      ⟨((given, fun _ => []), { base with commit₅ := target.1, claim₅ := target.2 }), rfl⟩)
    (Page.carries (fun data : Content =>
        (data.2.state, data.2.statement, data.2.partials))
      fun target => ⟨((given, fun _ => []), { base with
        state := target.1, statement := target.2.1, partials := target.2.2 }), rfl⟩)
    (Page.carries (fun data : Content => data.2.response)
      fun target => ⟨((given, fun _ => []), { base with response := target }), rfl⟩)
    started inside (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) table calm regular decodes decodes₁
    decodes₂ decodes₃ decodes₄ decodes₅ decodesL worked
    (⟨((given, fun _ => []), { base with commit := fun g => (Compression.Akita.commit (key g)
      (![keyBase (setup.params shape), W, 0] g)).1 }), fun _ => (fun _ => 0, 0), 0⟩ : Leaf)
    checks gate W counts tail leads
    (fun anchor rooted =>
      (congrArg (fun leaf : Leaf => leaf.data.2.commit 0) rooted).trans rfl)
    (fun anchor rooted =>
      (congrArg (fun leaf : Leaf => leaf.data.2.commit 1) rooted).trans rfl)
    (fun anchor => congrArg (fun index => (anchor.wire index).data.2.commit 2) stage.symm)
    fit
  exact ⟨proof, by rw [zero], holds⟩

end NightstreamFPrime.Export.BoundedDepth.FinalKey20
