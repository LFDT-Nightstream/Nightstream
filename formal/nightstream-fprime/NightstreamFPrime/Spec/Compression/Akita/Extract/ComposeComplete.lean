import NightstreamFPrime.Spec.Compression.Akita.Extract.BackComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.Compose
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceComplete

/-! The honest prover of the compression argument before the Akita opening: the two checkpoint
segments of `Compose` (SPEC.md Algorithm 9, steps 5 to 11 and Rejects; Algorithm 10; Akita,
ePrint 2026/1983, revision of 2026-11-23, Theorem 10.9).

- `Compose.Ran.lengths`: on a transcript with the conditions `Compose.Ran`, the points of the
  three claims of the Akita opening have `laneVars + keyBlockVars + rowVars`, `witnessVars` and
  `spanVars + columnVars + zeros` coordinates.
- `Compose.accepts`: the verifier of the argument over an Akita opening accepts the honest
  prover, for an honest prover `tower` of the opening.
- `Compose.run` gives the transcript prefix of the honest prover from the stage `Compose.start`
  to the stage `Compose.root`, for a table of the oracle whose answers decode to no challenge in
  `Verifier.Rejects`. The game of each checkpoint ends for each transcript with this prefix, and
  the last message has the three opening claims, which are true for the key polynomial, `W` and
  the honest inverses `V`. The honest prover of `lead` is a hypothesis. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib (mle)
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

variable {E : Type} [Field E] [Fintype E]
  {Slot : Type} [Fintype Slot] [Nonempty Slot] [FieldSlot Slot E] [DecidableEq E] [Algebra Zq E]

namespace Compose

variable {P : Verifier.Params Zq E} {B : Batch (Fin 3)} {Wire Coord Lead : Type} {total : ℕ}
  [Fintype Coord] [DecidableEq Coord] {C : Compose P B Wire Coord Slot Lead total}
  {vf : Verifier.VerifierData P} {commit : Wire → (g : Fin 3) → (B.shape g).Payload}
  {value : Wire → Fin 3 → E} {place : Wire → Fin 3 → List E}

omit [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Fintype Coord] [DecidableEq Coord] in
/-- The table of the key polynomial is 0 from the cell `2^(laneVars + keyBlockVars + rowVars)`. -/
theorem keyBase_zero (rows : P.rows ≤ 2 ^ P.rowVars) {cell : ℕ}
    (large : 2 ^ (P.laneVars + P.keyBlockVars + P.rowVars) ≤ cell) : keyBase P cell = 0 := by
  have high : 2 ^ P.rowVars ≤ cell / (P.lanes * 2 ^ P.keyBlockVars) :=
    (Nat.le_div_iff_mul_le (Nat.mul_pos (Nat.two_pow_pos _) (Nat.two_pow_pos _))).mpr (by
      rwa [pow_add, pow_add, Nat.mul_comm] at large)
  exact ite_eq_right (not_lt.mpr (rows.trans high))

omit [Nonempty Slot] [FieldSlot Slot E] in
/-- The points of the opening claims at the end of the game of the checkpoint of `(W, V)` have
`laneVars + keyBlockVars + rowVars`, `witnessVars` and `spanVars + columnVars + zeros`
coordinates (`Verifier.back_ends`). -/
theorem back_lengths {point : Prefix Wire (Option Coord → Slot) total} (wf : Verifier.WellFormed P)
    {claims : Verifier.Finals E}
    (ended : (C.back vf).game (fun _ _ => 0) point = .pure (some claims)) :
    claims.key.1.length = P.laneVars + P.keyBlockVars + P.rowVars ∧
      claims.w.1.length = P.witnessVars ∧
      claims.v.1.length = P.spanVars + vf.columnVars + vf.zeros := by
  have ends : ((C.back vf).check (point.take (C.back vf).start) fun _ _ => 0).Ends
      fun result => ∀ claims, result = some claims →
        claims.key.1.length = P.laneVars + P.keyBlockVars + P.rowVars ∧
          claims.w.1.length = P.witnessVars ∧
          claims.v.1.length = P.spanVars + vf.columnVars + vf.zeros := by
    show (C.backCheck vf ((point.take (C.back vf).start).wire 0) (fun _ _ => 0)
      (C.ran (point.take (C.back vf).start))).Ends _
    rcases C.ran (point.take (C.back vf).start) with _ | ⟨lead, τ⟩
    · intro claims none_eq
      cases none_eq
    · exact Verifier.back_ends vf wf _ _ _ _
  have at_end : ((C.back vf).game (fun _ _ => 0) point).Ends _ := ends_position C.read ends _ _
  rw [ended] at at_end
  exact at_end claims rfl

omit [Nonempty Slot] [FieldSlot Slot E] in
/-- On a transcript with the conditions `Ran`, the points of the claims of the Akita opening
have `laneVars + keyBlockVars + rowVars` coordinates for the key polynomial, `witnessVars` for
`W` and `spanVars + columnVars + zeros` for `V` (`back_lengths`). -/
theorem Ran.lengths {point : Prefix Wire (Option Coord → Slot) total} (wf : Verifier.WellFormed P)
    (ran : C.Ran vf commit value place point) :
    (place (point.wire (C.root vf)) 0).length = P.laneVars + P.keyBlockVars + P.rowVars ∧
      (place (point.wire (C.root vf)) 1).length = P.witnessVars ∧
      (place (point.wire (C.root vf)) 2).length = P.spanVars + vf.columnVars + vf.zeros := by
  obtain ⟨claims, ended, places, -⟩ := ran.ended
  rw [places]
  exact back_lengths wf ended

omit [FieldSlot Slot E] in
/-- The transcript prefix of the honest prover of the compression argument, from the stage
`start` to the first stage `root` of the Akita opening (SPEC.md Algorithm 9, steps 5 to 11).

`early` has the moves before the stage `start`, on the path of the table of the oracle.
`dodges`: an answer of the table decodes to no challenge in `Verifier.Rejects`. `W` is the table
of the honest prover, with honest counts. `settled` and `leads`: `lead` starts at a draw, and the
prover of `lead` has a message for each draw sequence outside `Verifier.Rejects` such that `lead`
ends in a value whose claims satisfy `Shaped` and `Relation` for `W`. `sends`, `wide`, `room`,
`short`, `carriesM` and `carriesR`: a wire message can have the messages of the prover, the data
`middle` and the commitments, the values and the points of the Akita opening. The other
hypotheses are conditions of `Sized`.

Then moves exist through the stage `root`, on the path, with lookup challenges `τ` that are no
table values and opening claims that are true for the key polynomial, `W` and the honest inverses
`V` of `τ`. The last message has the commitments `payload τ`, these claims and a nonce of the
response search below `responseTries`. For each
transcript with this prefix, the message after `τ` has the data `mid τ`, and the games of the two
checkpoints end in a value with `τ` and in the claims: the conditions `led` and `ended` of
`Ran`. -/
theorem run [Nonempty Coord] {accept : Prefix Wire (Option Coord → Slot) total → Prop}
    {sendable : (M : Type) → M → Prop} {slots width cap : ℕ} {Mid : Type} {middle : Wire → Mid}
    (wf : Verifier.WellFormed P) (matrix : Verifier.MatrixSpec vf)
    (columns : P.lookups ≤ 2 ^ vf.columnVars) (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars)
    (lead_bare : ∀ wire W other, bare (C.lead wire W) = bare (C.lead wire other))
    (fits : C.root vf ≤ total) (sends : Sends sendable E width cap) (wide : 4 ≤ width)
    (room : 0 < slots)
    (short : ∀ values : Verifier.Values E, (∀ k, 2 ^ vf.columnVars ≤ k → values.2.1 k = 0) →
      sendable (Verifier.Values E) values)
    (carriesM : Carries sendable slots C.read middle)
    (carriesR : Carries sendable slots C.read fun wire => (commit wire, value wire, place wire))
    (table : Prefix Wire (Option Coord → Slot) total → Option Coord → Slot)
    (dodges : ∀ input (T : Type) [Fintype T] [Nonempty T],
      ¬ Verifier.Rejects P T (sample (Coord → Slot) T (Extract.view (table input))))
    {first : Wire} {early : List ((Option Coord → Slot) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = C.start) (W : Table) (counts : Verifier.HonestCounts P W)
    (distinct : Verifier.TablesDistinct P) (settled : Settled (C.lead first W))
    (leads : WinsOff (Verifier.Rejects P) sendable slots (C.lead first W) C.leadCosts.length 0
      fun result _ _ => ∃ lead, result = some lead ∧
        Verifier.Shaped P (C.statement first lead) ∧ Verifier.Relation P (C.statement first lead) W)
    (mid : (Fin P.lookups → Zq) → Mid)
    (payload : (Fin P.lookups → Zq) → (g : Fin 3) → (B.shape g).Payload) :
    ∃ (moves : List ((Option Coord → Slot) × Wire)) (τ : Fin P.lookups → Zq) (claims : Verifier.Finals
      E),
      (early ++ moves).length = C.root vf ∧ OnPath accept table first (early ++ moves) ∧
      C.read.nonce (lastWire first (early ++ moves)) < responseTries ∧
      Verifier.Apart P τ ∧
      Verifier.Opened (Verifier.keyCells P) (lifted E W)
        (lifted E (Verifier.honestInverses P W τ)) claims ∧
      commit (lastWire first (early ++ moves)) = payload τ ∧
      place (lastWire first (early ++ moves)) = (fun g => (claimOf claims g).1) ∧
      value (lastWire first (early ++ moves)) = (fun g => (claimOf claims g).2) ∧
      ∀ stage : Prefix Wire (Option Coord → Slot) total, stage.first = first →
        (∃ later, stage.rest = early ++ moves ++ later) →
        middle (stage.wire C.front.finish) = mid τ ∧
          stage.wire (C.root vf) = lastWire first (early ++ moves) ∧
          (∃ lead, C.front.game 0 stage = .pure (some (lead, τ))) ∧
          (C.back vf).game (fun _ _ => 0) stage = .pure (some claims) := by
  have frontFits : C.front.finish ≤ total := (Nat.le_add_right _ _).trans fits
  have short₀ : early.length ≤ total := by
    rw [starts]
    exact (Nat.le_add_right _ _).trans frontFits
  have frontWins : WinsOff (Verifier.Rejects P) sendable slots
      (C.front.check ⟨first, early, short₀⟩ ((0 : Table), W)) C.front.costs.length 0
      fun result _ _ => ∃ lead τ, result = some (lead, τ) ∧
        (Verifier.Shaped P (C.statement first lead) ∧
          Verifier.Relation P (C.statement first lead) W) ∧ Verifier.Apart P τ := by
    rw [show C.front.costs.length = C.leadCosts.length + 1 from by
      show (C.leadCosts ++ [Verifier.tauError P]).length = _
      rw [List.length_append, List.length_singleton]]
    exact Verifier.led_wins W leads
  obtain ⟨moves₁, result₁, counted₁, ⟨lead, τ, rfl, ⟨shaped, relation⟩, apart⟩, on₁, held₁,
      _, each₁⟩ :=
    C.front.run carriesM on starts frontFits short₀ ((0 : Table), W)
      (Verifier.led_settled settled W) frontWins dodges
      (List.append_ne_nil_of_right_ne_nil _ (List.cons_ne_nil _ _))
      (fun game => mid (((finished game).map Prod.snd).getD fun _ => 0))
  have length₁ : (early ++ moves₁).length = C.front.finish := by
    rw [List.length_append, starts, counted₁]
    rfl
  have short₁ : (early ++ moves₁).length ≤ total := length₁.le.trans frontFits
  have ran₁ : C.ran ⟨first, early ++ moves₁, short₁⟩ = some (lead, τ) := by
    have ended := (each₁ ⟨first, early ++ moves₁, short₁⟩ rfl
      ⟨[], (List.append_nil _).symm⟩).2
    have same : C.front.game 0 ⟨first, early ++ moves₁, short₁⟩ = .pure (some (lead, τ)) :=
      ended_of_bare C.read (Verifier.led_bare (lead_bare _ _ _) _ _) ended
    unfold Compose.ran
    rw [same]
    rfl
  have backCheck : (C.back vf).check ⟨first, early ++ moves₁, short₁⟩
      ![0, W, Verifier.honestInverses P W τ] =
      Verifier.back vf (C.statement first lead) W (Verifier.honestInverses P W τ) τ := by
    show C.backCheck vf first _ (C.ran _) = _
    rw [ran₁]
    rfl
  obtain ⟨moves₂, result₂, counted₂, ⟨claims, rfl, opened⟩, on₂, held₂, fresh, each₂⟩ :=
    (C.back vf).run (reject := Verifier.Rejects P) carriesR
      (goal := fun result => ∃ claims, result = some claims ∧
        Verifier.Opened (Verifier.keyCells P) (fun x => algebraMap Zq E (W x))
          (fun x => algebraMap Zq E (Verifier.honestInverses P W τ x)) claims)
      on₁ length₁ fits short₁ ![0, W, Verifier.honestInverses P W τ]
      (by rw [backCheck]; exact trivial)
      (by
        rw [backCheck]
        exact Verifier.back_wins vf sends wide room short wf matrix columns rows key_fits _ W τ
          shaped relation counts distinct apart)
      dodges (List.cons_ne_nil _ _)
      (fun game => (payload τ,
        (fun g => ((finished game).map fun claims => (claimOf claims g).2).getD 0),
        fun g => ((finished game).map fun claims => (claimOf claims g).1).getD []))
  refine ⟨moves₁ ++ moves₂, τ, claims, ?_, ?_, ?_, apart, opened, ?_, ?_, ?_, ?_⟩
  · rw [← List.append_assoc, List.length_append, length₁, counted₂]
    rfl
  · rwa [List.append_assoc] at on₂
  · rw [← List.append_assoc]
    exact fresh
  · rw [← List.append_assoc]
    exact congrArg Prod.fst held₂
  · rw [← List.append_assoc]
    exact congrArg (fun data => data.2.2) held₂
  · rw [← List.append_assoc]
    exact congrArg (fun data => data.2.1) held₂
  · rintro stage rooted ⟨later, shape⟩
    have shape₂ : stage.rest = early ++ moves₁ ++ moves₂ ++ later := by
      rw [shape]
      simp only [List.append_assoc]
    have shape₁ : stage.rest = early ++ moves₁ ++ (moves₂ ++ later) := by
      rw [shape₂, List.append_assoc]
    obtain ⟨wire₁, ended₁⟩ := each₁ stage rooted ⟨_, shape₁⟩
    obtain ⟨wire₂, ended₂⟩ := each₂ stage rooted ⟨_, shape₂⟩
    have led : C.front.game 0 stage = .pure (some (lead, τ)) :=
      ended_of_bare C.read (Verifier.led_bare (lead_bare _ _ _) _ _) ended₁
    have ended : (C.back vf).game (fun _ _ => 0) stage = .pure (some claims) :=
      back_game _ _ ended₂
    refine ⟨?_, ?_, ⟨lead, led⟩, ended⟩
    · rw [wire₁]
      exact held₁
    · rw [← List.append_assoc]
      exact wire₂

omit [FieldSlot Slot E] in
/-- The verifier of the compression argument over an Akita opening accepts the honest prover
(SPEC.md Algorithm 9 and Algorithm 10; Akita Theorem 10.9): `run` and then the honest prover
`tower` of the Akita opening. `links` is the part of the verifier for the Akita opening.

`bound₀`, `bound₁` and `bound₂`: the commitments that a stage input fixes are the commitments of
the key polynomial and of `W`, and the commitment of `V` in the message after `τ`. `tower`: for
lookup challenges `τ` that are no table values and a transcript prefix of `root` moves on the
path whose last message has a nonce of the response search below `responseTries`, the
commitments of the key polynomial, `W` and the honest inverses `V` of `τ`, points of the lengths
of `back_lengths` and the values of the tables at these points, later messages exist such that
each transcript with them satisfies `links`. The other hypotheses are those of `run`.

Then a proof with the first message `first` exists whose complete transcript satisfies `Ran` and
`links`. -/
theorem accepts [Nonempty Coord] {accept : Prefix Wire (Option Coord → Slot) total → Prop}
    {sendable : (M : Type) → M → Prop} {slots width cap : ℕ} (wf : Verifier.WellFormed P)
    (matrix : Verifier.MatrixSpec vf) (columns : P.lookups ≤ 2 ^ vf.columnVars)
    (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars)
    (lead_bare : ∀ wire W other, bare (C.lead wire W) = bare (C.lead wire other))
    (fits : C.root vf ≤ total) (sends : Sends sendable E width cap) (wide : 4 ≤ width)
    (room : 0 < slots)
    (short : ∀ values : Verifier.Values E, (∀ k, 2 ^ vf.columnVars ≤ k → values.2.1 k = 0) →
      sendable (Verifier.Values E) values)
    (carriesM : Carries sendable slots C.read commit)
    (carriesR : Carries sendable slots C.read fun wire => (commit wire, value wire, place wire))
    (table : Prefix Wire (Option Coord → Slot) total → Option Coord → Slot)
    (dodges : ∀ input (T : Type) [Fintype T] [Nonempty T],
      ¬ Verifier.Rejects P T (sample (Coord → Slot) T (Extract.view (table input))))
    {first : Wire} {early : List ((Option Coord → Slot) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = C.start) (W : Table) (counts : Verifier.HonestCounts P W)
    (distinct : Verifier.TablesDistinct P) (settled : Settled (C.lead first W))
    (leads : WinsOff (Verifier.Rejects P) sendable slots (C.lead first W) C.leadCosts.length 0
      fun result _ _ => ∃ lead, result = some lead ∧
        Verifier.Shaped P (C.statement first lead) ∧ Verifier.Relation P (C.statement first lead) W)
    (key : (g : Fin 3) → Key (B.shape g))
    (bound₀ : ∀ anchor : Prefix Wire (Option Coord → Slot) total, anchor.first = first →
      C.bound anchor 0 = (Akita.commit (key 0) (keyBase P)).1)
    (bound₁ : ∀ anchor : Prefix Wire (Option Coord → Slot) total, anchor.first = first →
      C.bound anchor 1 = (Akita.commit (key 1) W).1)
    (bound₂ : ∀ anchor : Prefix Wire (Option Coord → Slot) total,
      C.bound anchor 2 = commit (anchor.wire C.front.finish) 2)
    (links : Prefix Wire (Option Coord → Slot) total → Prop)
    (tower : ∀ τ : Fin P.lookups → Zq, Verifier.Apart P τ →
      ∀ moves : List ((Option Coord → Slot) × Wire), OnPath accept table first moves →
      moves.length = C.root vf → C.read.nonce (lastWire first moves) < responseTries →
      commit (lastWire first moves) = (fun g =>
        (Akita.commit (key g) (![keyBase P, W, Verifier.honestInverses P W τ] g)).1) →
      ((place (lastWire first moves) 0).length = P.laneVars + P.keyBlockVars + P.rowVars ∧
        (place (lastWire first moves) 1).length = P.witnessVars ∧
        (place (lastWire first moves) 2).length = P.spanVars + vf.columnVars + vf.zeros) →
      value (lastWire first moves) = (fun g => mle (place (lastWire first moves) g)
        (lifted E (![keyBase P, W, Verifier.honestInverses P W τ] g))) →
      ∃ later, OnPath accept table first (moves ++ later) ∧ (moves ++ later).length = total ∧
        ∀ stage : Prefix Wire (Option Coord → Slot) total, stage.first = first →
          stage.rest = moves ++ later → links stage) :
    ∃ proof : ℕ → Wire, proof 0 = first ∧
      C.Ran vf commit value place (pointPath total accept proof table total) ∧
      links (pointPath total accept proof table total) := by
  obtain ⟨moves, τ, claims, length, on', fresh, apart, opened, committed, places, values, each⟩ :=
    run wf matrix columns rows key_fits lead_bare fits sends wide room short carriesM carriesR
      table dodges on starts W counts distinct settled leads
      (fun τ g => (Akita.commit (key g) (![keyBase P, W, Verifier.honestInverses P W τ] g)).1)
      (fun τ g => (Akita.commit (key g) (![keyBase P, W, Verifier.honestInverses P W τ] g)).1)
  have short' : (early ++ moves).length ≤ total := length.le.trans fits
  obtain ⟨-, -, -, ended₀⟩ := each ⟨first, early ++ moves, short'⟩ rfl
    ⟨[], (List.append_nil _).symm⟩
  have sizes := back_lengths wf ended₀
  obtain ⟨later, on'', whole, linked⟩ := tower τ apart (early ++ moves) on' length fresh
    committed
    (by rw [places]; exact sizes)
    (by
      rw [values, places]
      funext g
      fin_cases g
      · show claims.key.2 = mle claims.key.1 (lifted E (keyBase P))
        rw [lifted_keyBase]
        exact opened.1
      · exact opened.2.1
      · exact opened.2.2)
  obtain ⟨proof, zero, holds⟩ := OnPath.accepts
    (holds := fun stage => C.Ran vf commit value place stage ∧ links stage) on'' whole
    (fun stage rooted shape => by
      obtain ⟨middle, last, led, ended⟩ := each stage rooted ⟨later, shape⟩
      refine ⟨⟨led.imp fun lead ended => ⟨τ, ended⟩, ?_,
        claims, ended, by rw [last]; exact places, by rw [last]; exact values⟩,
        linked stage rooted shape⟩
      rw [last, committed]
      funext g
      fin_cases g
      · exact (bound₀ (stage.take C.front.finish) rooted).symm
      · exact (bound₁ (stage.take C.front.finish) rooted).symm
      · show _ = C.bound (stage.take C.front.finish) 2
        rw [bound₂, Prefix.wire_take le_rfl, middle]
        rfl)
  exact ⟨proof, zero, holds⟩

end Compose

end NightstreamFPrime.Spec.Compression.Akita.Extract
