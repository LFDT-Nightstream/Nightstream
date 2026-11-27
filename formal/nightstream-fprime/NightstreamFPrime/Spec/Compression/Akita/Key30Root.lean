import NightstreamFPrime.Spec.Compression.Akita.Key30Complete
import NightstreamFPrime.Spec.Compression.Akita.Key30Tower
import NightstreamFPrime.Spec.Compression.Akita.Extract.OffComplete

/-! The honest prover of the opening of Akita for a row of the schedule file
`crates/ns-compress/artifacts/key30.aks`: the batching root, the offloaded root fold of the key
polynomial, `W` and `V`, the level 1 with the group of the setup prefix and the group of the
witness, the packing levels 2 and 3, the trace levels 4 to 6 and the terminal (Akita, ePrint
2026/1983, revision of 2026-11-23, Theorem 10.9 with Lemma 10.7, Section 9 with (136), Figure 4
and Section 8.2).

- `tower_complete` is `Extract.offTower_complete` for the numbers of a row of `Key30.lean`: the
  verifier `offLinks` of `Key30.tower_suffix` accepts the honest prover. The conditions on the
  numbers are theorems of `Key30.lean` and `Key30Complete.lean`. The premise
  `Batch.Registered` of the setup prefix and the fit conditions `BatchFit`, `StateFit`, `LiftFit`
  and `TraceFit` stay hypotheses, so the theorem holds for each setup prefix `flat`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key30

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

variable {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E]
  [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {read : Reader Wire} {start : ℕ} {r : Row} {member : r ∈ rows}
  {accept : Prefix Wire (Option Coord → Slot) total → Prop}
  {β : ∀ g, Module.Basis (Fin ((root r).planes g)) Zq E}
  {key : (g : Fin 3) → Key ((root r).shape g)} {openKey : (root r).OpenKey}
  {claim : Wire → (root r).Claim E} {block : (root r).Block → Coord}
  {decode : (g : Fin 3) → Slot → Option (Rq ((root r).carrierDegree g))} {mass : Fin 3 → ℕ}
  {flat : Table}
  {scan : ∀ (wire : Wire) (challenge : (root r).Challenge),
    SetupScan (Batch.liftedReduction β ((claim wire).statement key openKey) challenge rootVars
      (root_sized r member).1 8 quotientDigits₀ (root_sized r member).2.1 rootBatch
      (root_sized r member).2.2.1) setupVars flat}
  {β₁ : ∀ g, Module.Basis (Fin ((level₁ r).planes g)) Zq E}
  {key₁ : (g : Fin 2) → Key ((level₁ r).shape g)} {openKey₁ : (level₁ r).OpenKey}
  {claim₁ : Wire → (level₁ r).Claim E} {block₁ : (level₁ r).Block → Coord}
  {decode₁ : (g : Fin 2) → Slot → Option (Rq ((level₁ r).carrierDegree g))} {mass₁ : Fin 2 → ℕ}
  {registry : ((level₁ r).shape 0).Payload} {reference : Decommitment ((level₁ r).shape 0)}
  {β₂ : Module.Basis (Fin (level₂ r).opening.planes) Zq E} {key₂ : Key (level₂ r).shape}
  {openKey₂ : OpeningKey (level₂ r).opening} {claim₂ : Wire → Claim (level₂ r).opening E}
  {block₂ : Fin (level₂ r).shape.blocks → Coord}
  {decode₂ : Slot → Option (Rq (level₂ r).opening.carrierDegree)} {weight₂ : ℕ}
  {β₃ : Module.Basis (Fin (level₃ r).opening.planes) Zq E} {key₃ : Key (level₃ r).shape}
  {openKey₃ : OpeningKey (level₃ r).opening} {claim₃ : Wire → Claim (level₃ r).opening E}
  {block₃ : Fin (level₃ r).shape.blocks → Coord}
  {decode₃ : Slot → Option (Rq (level₃ r).opening.carrierDegree)} {weight₃ : ℕ}
  {key₄ : Key level₄.shape} {openKey₄ : OpeningKey level₄.opening}
  {claim₄ : Wire → TraceClaim level₄.opening E} {block₄ : Fin level₄.shape.blocks → Coord}
  {decode₄ : Slot → Option (Rq level₄.shape.innerDegree)} {weight₄ : ℕ}
  {key₅ : Key level₅.shape} {openKey₅ : OpeningKey level₅.opening}
  {claim₅ : Wire → TraceClaim level₅.opening E} {block₅ : Fin level₅.shape.blocks → Coord}
  {decode₅ : Slot → Option (Rq level₅.shape.innerDegree)} {weight₅ : ℕ}
  {key₆ : Key level₆.shape} {openKey₆ : OpeningKey level₆.opening}
  {claim₆ : Wire → TraceClaim level₆.opening E} {block₆ : Fin level₆.shape.blocks → Coord}
  {decode₆ : Slot → Option (Rq level₆.shape.innerDegree)} {weight₆ : ℕ}
  {L : Terminal terminal E Wire Coord Slot total
    (.euclidean normTerminal.operator normTerminal.limit)}
  {commit : Wire → (g : Fin 3) → ((root r).shape g).Payload} {value : Wire → Fin 3 → E}
  {place : Wire → Fin 3 → List E}
  {commits : Wire → (g : Fin 2) → ((level₁ r).shape g).Payload} {values : Wire → Fin 2 → E}
  {places : Wire → Fin 2 → List E}
  {spots : Wire → (g : Fin 2) → Fin (![setupVars, rootVars] g) → E}
  {commit₁ : Wire → ((level₁ r).shape 1).Payload}
  {commit₂ : Wire → (level₂ r).shape.Payload} {commit₃ : Wire → (level₃ r).shape.Payload}
  {commit₄ : Wire → level₄.shape.Payload} {commit₅ : Wire → level₅.shape.Payload}
  {commit₆ : Wire → level₆.shape.Payload}
  {stateOf : Wire → Fin terminal.blocks → Fin terminal.innerRank → Rq terminal.innerDegree}
  {statementOf : Wire → TerminalStatement terminal E}
  {partialsOf : Wire → Fin terminal.blocks → Rq terminal.innerDegree}

local notation "zeroth" =>
  offBatch β key openKey claim rootVars (And.left (root_sized r member)) 8 quotientDigits₀
    (And.left (And.right (root_sized r member))) rootBatch
    (And.left (And.right (And.right (root_sized r member)))) block decode read mass (start + 1)
    setupVars flat scan

local notation "first" =>
  Tier.pinned (liftBatch β₁ key₁ openKey₁ claim₁ vars₁ (And.left (level₁_sized r member)) 16
    quotientDigits (And.left (And.right (level₁_sized r member))) batch₁
    (And.left (And.right (And.right (level₁_sized r member)))) block₁ decode₁ read mass₁
    (Tier.finish zeroth + 1)) (Batch.Pinned 0 registry flat)
    (fun ω _ => Batch.pinBreak key₁ openKey₁ _ 0 reference ω)

local notation "second" =>
  liftFold β₂ key₂ openKey₂ claim₂ (Key20.Level.vars (level₂ r))
    (And.left (level₂_sized r member)) quotientDigits
    (And.left (And.right (level₂_sized r member))) (Key20.Level.batch (level₂ r))
    (And.left (And.right (And.right (level₂_sized r member)))) block₂ decode₂ read weight₂
    (Tier.finish first)

local notation "third" =>
  Tier.tensored (liftFold β₃ key₃ openKey₃ claim₃ (Key20.Level.vars (level₃ r)) (And.left (level₃_sized r member))
    quotientDigits (And.left (And.right (level₃_sized r member))) (Key20.Level.batch (level₃ r))
    (And.left (And.right (And.right (level₃_sized r member)))) block₃ decode₃ read weight₃
    (Tier.finish second)) (Key20.Level.tail (level₃ r))

local notation "fourth" =>
  Tier.tensored (committedTrace (fold := level₄.fold) key₄ openKey₄ claim₄ level₄.vars
    (And.left level₄_sized) level₄.batch (And.left (And.right level₄_sized)) block₄ decode₄ read
    weight₄ (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) key₅ openKey₅ claim₅ level₅.vars
    (And.left (tails_sized level₅ (List.mem_cons_self ..))) level₅.batch
    (And.left (And.right (tails_sized level₅ (List.mem_cons_self ..)))) block₅ decode₅
    read weight₅ (Key20.Squared.operator (norm₅ r)) (Key20.Squared.limit (norm₅ r)) (Tier.finish fourth)) level₅.tail

local notation "sixth" =>
  Tier.tensored (committedTraceSq (fold := level₆.fold) key₆ openKey₆ claim₆ level₆.vars
    (And.left (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))) level₆.batch
    (And.left (And.right (tails_sized level₆ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))))
    block₆ decode₆ read weight₆ norm₆.operator norm₆.limit (Tier.finish fifth))
    level₆.tail

local notation "accepts" =>
  offLinks (weight₅ := weight₅) (weight₆ := weight₆) (route₄ := Route.coefficient)
    (route₅ := Route.euclidean (Key20.Squared.operator (norm₅ r)) (Key20.Squared.limit (norm₅ r)))
    (route₆ := Route.euclidean norm₆.operator norm₆.limit) zeroth first second third fourth fifth
    sixth L start read commit value place commits values places registry commit₁ commit₂ commit₃
    (And.left quarters) (And.left (And.right quarters))
    (And.left (And.right (And.right quarters))) (And.right (And.right (And.right quarters)))
    commit₄ commit₅ commit₆ stateOf

omit [FieldSlot Slot E] in
/-- Akita Theorem 10.9 for the opening of a row of the schedule `key30.aks`, without the
probability that a response does not fit and without the term of (155) for a zero tensor factor
(`Extract.offTower_complete`): the verifier `offLinks` of `Key30.tower_suffix` accepts the honest
prover. For each table of the oracle whose slots decode to fold challenges, and each transcript
prefix of `start` moves on its path whose last message has the commitment of a table of each
group of the root, a point of 30, `vars` and 26 coordinates and the value of the table at this
point, later messages exist such that the complete transcript satisfies `offLinks`: the link
`RootLinked` of the batching root, the link `Tier.Offloaded` from the root fold to the two claims
of level 1, the link `RootLinked` of their batching stage, the links `Tier.Linked` of the levels
1 to 3, the links `Tier.TraceLinked` of the levels 3 to 6, the link `Terminal.Linked` and the
checks `Terminal.Passed`. The conditions on the shapes are theorems of `Key30.lean` and
`Key30Complete.lean`.

`sends`, `wide`, `capped`, `room`, `carriesR` to `carriesL`: a wire message can have the data of
the prover, with messages of 8 field elements, the energy `3029804899` and 4 messages of the
checks. `spotted`: the points of the two claims of level 1 are the lists of the vectors `spots` of
23 and 28 coordinates. `registered`: the premise of Akita Lemma 9.1 for the commitment `registry`
of the setup prefix `flat` with the reference decommitment `reference`. `next`, `inside`,
`reads`, `clears`: the stage of the terminal and its messages. `counts`: the terminal reads the
nonce of the response search with `read`. `live`: the table of the key polynomial is 0 after
`2^30` cells, the table of `W` is 0 after `2^vars` cells, and the table of `V` is 0 after `2^26`
cells. `committed`, `lengths`, `valued`: the last message of `early` has the commitments `commit`
of the tables, points of 30, `vars` and 26 coordinates, and the values of the tables at these
points. `fit`: the fit conditions `BatchFit` of the root, `StateFit` of level 1 for the reference
decommitment of the setup prefix and the bounded source of the witness, `LiftFit` of the levels 2
and 3, `TraceFit` of the levels 4 to 6 with `S_max` of `(norm₅ r)` and `norm₆`, and
`‖z‖_2² ≤ S_max,term` at the terminal, for each fold challenge vector of the families. -/
theorem tower_complete {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
    (sends : Sends sendable E width cap) (wide : 8 ≤ width) (capped : 3029804899 ≤ cap)
    (room : 3 < slots) (carriesR : Carries sendable slots read claim)
    (spotted : ∀ wire, places wire = fun g => List.ofFn (spots wire g))
    (carriesO : Carries sendable slots read fun wire =>
      (commit₁ wire, commits wire, values wire, spots wire))
    (carries₁R : Carries sendable slots read claim₁)
    (carries₁ : Carries sendable slots read fun wire => (commit₂ wire, claim₂ wire))
    (carries₂ : Carries sendable slots read fun wire => (commit₃ wire, claim₃ wire))
    (carries₃ : Carries sendable slots read fun wire => (commit₄ wire, claim₄ wire))
    (carries₄ : Carries sendable slots read fun wire => (commit₅ wire, claim₅ wire))
    (carries₅ : Carries sendable slots read fun wire => (commit₆ wire, claim₆ wire))
    (carries₆ : Carries sendable slots read fun wire =>
      (stateOf wire, statementOf wire, partialsOf wire))
    (carriesL : Carries sendable slots read (Terminal.response L))
    (registered : Batch.Registered (key₁ 0) registry flat reference)
    (next : Terminal.start L = Tier.finish sixth) (inside : Terminal.start L < total)
    (reads : ∀ point, Terminal.read L point = statementOf (point.wire (Terminal.start L)))
    (clears : ∀ point, Terminal.partials L point = partialsOf (point.wire (Terminal.start L)))
    (counts : ∀ wire, Terminal.nonce L wire = read.nonce wire)
    (table : Prefix Wire (Option Coord → Slot) total → Option Coord → Slot)
    (decodes : ∀ input b, (decode b.1 (table input (some (block b)))).isSome)
    (decodes₁ : ∀ input b, (decode₁ b.1 (table input (some (block₁ b)))).isSome)
    (decodes₂ : ∀ input i, (decode₂ (table input (some (block₂ i)))).isSome)
    (decodes₃ : ∀ input i, (decode₃ (table input (some (block₃ i)))).isSome)
    (decodes₄ : ∀ input i, (decode₄ (table input (some (block₄ i)))).isSome)
    (decodes₅ : ∀ input i, (decode₅ (table input (some (block₅ i)))).isSome)
    (decodes₆ : ∀ input i, (decode₆ (table input (some (block₆ i)))).isSome)
    (decodesL : ∀ input i, (Terminal.decode L (table input (some (Terminal.block L i)))).isSome)
    (worked : ∀ input, table input none ∈ workZeros Slot 57)
    {wire₀ : Wire} {early : List ((Option Coord → Slot) × Wire)} (on : OnPath accept table wire₀ early)
    (starts : early.length = start) {tables : Fin 3 → Table}
    (live : ∀ g cell, 2 ^ ![30, r.vars, 26] g ≤ cell → tables g cell = 0)
    (committed : commit (lastWire wire₀ early) = fun g => (Akita.commit (key g) (tables g)).1)
    (lengths : ∀ g, (place (lastWire wire₀ early) g).length = ![30, r.vars, 26] g)
    (valued : value (lastWire wire₀ early) =
      fun g => mle (place (lastWire wire₀ early) g) (lifted E (tables g)))
    (fit : BatchFit β key openKey rootVars (And.left (root_sized r member)) 8 quotientDigits₀
        decode tables fun w₀ =>
      StateFit β₁ key₁ openKey₁ vars₁ (And.left (level₁_sized r member)) 16 quotientDigits decode₁
        (offState (key₁ 1) registry reference w₀) fun w₁ =>
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
        (∀ i, ∃ slot, Terminal.decode L slot = some (c i)) →
        vecNormSq (traceResponse c (traceSource (And.right (And.right (And.right quarters)))
          (packed w₆ : ℕ → E))) ≤ normTerminal.limit) :
    ∃ later, OnPath accept table wire₀ (early ++ later) ∧ (early ++ later).length = total ∧
      ∀ stage : Prefix Wire (Option Coord → Slot) total, stage.first = wire₀ →
        stage.rest = early ++ later → accepts stage := by
  have pointed := root_points r member
  have segment : ∀ g, ![4, 4, 2] g ≤ (place (lastWire wire₀ early) g).length ∧
      7 ≤ (place (lastWire wire₀ early) g).length - ![4, 4, 2] g ∧
      ![10, r.vars - 19, 9] g ≤ (place (lastWire wire₀ early) g).length - ![4, 4, 2] g - 7 ∧
      (place (lastWire wire₀ early) g).length - ![4, 4, 2] g - 7 - ![10, r.vars - 19, 9] g =
        ![9, 8, 8] g := fun g => by
    have sum := (pointed g).2.2.2.2
    have length := lengths g
    omega
  have points₁ := level₁_points r member
  have opening₁ := level₁_openings r member
  have link := links r member
  have lift := lifts r member
  have point := points r member
  have cells := live_cells r member
  exact offTower_complete (scan := scan) (fix := fun ω _ => Batch.pinBreak key₁ openKey₁ _ 0
    reference ω) sends (by omega) room carriesR spotted carriesO carries₁R carries₁ carries₂
    carries₃ carries₄ carries₅ carries₆ carriesL (honest.1 r member)
    (fun g => (root_folds r member g).1) (fun g => (root_folds r member g).2) (root_path r member) (root_sized r member).2.2.2.2.1
    covers.1.1 covers.1.2 ((by decide : rangeWidth 8 ≤ 6).trans (by omega))
    (fun g => (opening₁ g).1) (fun g => (opening₁ g).2) (fun g => (level₁_folds r member g).1)
    (fun g => (level₁_folds r member g).2) (level₁_path r member)
    (level₁_sized r member).2.2.2.2.1 covers.2.1 covers.2.2
    ((by decide : rangeWidth 16 ≤ 6).trans (by omega)) (level₁_bounded r member) registered
    (points₁ 0).2.2.1 (points₁ 0).2.2.2.1 (points₁ 0).2.1 (points₁ 0).1
    (points₁ 1).2.2.1 (points₁ 1).2.2.2.1 (points₁ 1).2.1 (points₁ 1).1
    (by decide) (by decide) cells.1.1
    lift.1.1 lift.1.2.1 lift.1.2.2.1 lift.1.2.2.2
    covers.2.1 covers.2.2 ((by decide : rangeWidth 16 ≤ 6).trans (by omega))
    lift.2.1 lift.2.2.1 lift.2.2.2.1 lift.2.2.2.2 covers.2.1 covers.2.2
    ((by decide : rangeWidth 16 ≤ 6).trans (by omega))
    point.1.1 point.1.2.1 point.1.2.2.1
    point.1.2.2.2.1 point.1.2.2.2.2.1.symm cells.1.2.1
    point.2.1.1 point.2.1.2.1 point.2.1.2.2.1 point.2.1.2.2.2
    point.1.2.2.2.2.2.symm cells.1.2.2
    quarters.1 quarters.2.1 quarters.2.2.1 quarters.2.2.2 source₄ source₅ source₆ sourceTerm
    (fold₄.mono wide) (fold₅.mono wide) (fold₆.mono wide) link.1.1 link.1.2.1 link.1.2.2.1
    link.1.2.2.2 link.2.1.1 link.2.1.2.1 link.2.1.2.2.1 link.2.1.2.2.2 link.2.2.1
    link.2.2.2.1 link.2.2.2.2.1 link.2.2.2.2.2
    (norms_sized r member (level₅, (norm₅ r)) (List.mem_cons_self ..)).1
    (norms_sized r member (level₆, norm₆) (List.mem_cons_of_mem _ (List.mem_cons_self ..))).1
    ((by decide : rangeWidthSq level₅.shape.certifiedBase ≤ 8).trans wide)
    ((by decide : rangeWidthSq level₆.shape.certifiedBase ≤ 8).trans wide)
    ((Key30.capped r member).1.trans capped)
    ((by decide : norm₆.limit ≤ 3029804899).trans capped) next inside reads clears counts table
    decodes decodes₁ decodes₂ decodes₃ decodes₄ decodes₅ decodes₆ decodesL worked on starts
    (blockOf := fun g => (((place (lastWire wire₀ early) g).drop (![4, 4, 2] g)).drop 7).drop
      (![10, r.vars - 19, 9] g))
    (positionOf := fun g => (((place (lastWire wire₀ early) g).drop (![4, 4, 2] g)).drop 7).take
      (![10, r.vars - 19, 9] g))
    (packOf := fun g => (place (lastWire wire₀ early) g).take (![4, 4, 2] g))
    (tailOf := fun g => ((place (lastWire wire₀ early) g).drop (![4, 4, 2] g)).take 7)
    (fun g => (pointed g).1.le.trans_eq (congrArg (2 ^ ·) (by
      simp only [List.length_drop]
      exact (segment g).2.2.2.symm)))
    (fun g => (pointed g).2.1.trans (congrArg (2 ^ ·) (by
      simp only [List.length_take, List.length_drop]
      exact (Nat.min_eq_left (segment g).2.2.1).symm)))
    (fun g => (pointed g).2.2.1.trans (congrArg (2 ^ ·) (by
      simp only [List.length_take]
      exact (Nat.min_eq_left (segment g).1).symm)))
    (fun g => (pointed g).2.2.2.1.trans (congrArg (2 ^ ·) (by
      simp only [List.length_take, List.length_drop]
      exact (Nat.min_eq_left (segment g).2.1).symm)))
    (fun g cell large => live g cell ((root_cells r member g).symm.trans_le large)) committed
    (funext fun g => by simp only [List.take_append_drop]) valued fit

end NightstreamFPrime.Spec.Compression.Akita.Key30
