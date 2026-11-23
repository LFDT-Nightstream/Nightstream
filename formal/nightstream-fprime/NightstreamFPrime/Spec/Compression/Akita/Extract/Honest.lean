import NightstreamFPrime.Spec.Compression.Akita.FoldComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.Chain

/-! The honest prover of the checks of a packing fold on a Fiat–Shamir transcript (Akita,
ePrint 2026/1983, revision of 2026-11-23, Theorem 10.9 and Lemma 10.7).

- `Witness.Alphabets.rangeHolds` and `Witness.Alphabets.binaryOn` give the digit range statement
  and the binary statement of the flat witness of digit vectors with the certified alphabets.
- `flatFold_winsTo` is `foldCheck_winsTo` for digit vectors with the exact branch relation.
- `settle_winsTo`, `advance_winsTo` and `play` give the messages of the honest prover as wire
  messages: for each answer sequence of the oracle, the game of a transcript prefix ends in the
  goal. `keep` is a condition on the other parts of each wire message.
- `OnPath` states that a stage input is on the path of each proof with its messages for the
  table of the oracle, and `OnPath.final` gives the complete transcript of such a proof.
  `OnPath.answers` states that its answers are the answers of the table, so `Tier.path_worked`
  gives the work bits of its stages from the hypothesis that each work slot of the table has 57
  zero work bits (`Tier.Capped`: no site has more). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

section Cells

variable {d m : ℕ} [NeZero d]

theorem Digits.sequence {lo hi : ℤ} {v : Fin m → Rq d} (digits : Digits lo hi v) (low : lo ≤ 0)
    (high : 0 ≤ hi) (x : ℕ) :
    lo ≤ (Akita.sequence v x).valMinAbs ∧ (Akita.sequence v x).valMinAbs ≤ hi := by
  unfold Akita.sequence
  split
  · exact digits _ _ (Nat.mod_lt x (NeZero.pos d))
  · rw [ZMod.valMinAbs_zero]
    exact ⟨low, high⟩

theorem PathDigits.Binary.firstTable : {chain : Option ChainShape} → {digits : PathDigits chain} →
    PathDigits.Binary chain digits → ∀ x,
      -1 ≤ (PathDigits.firstTable chain digits x).valMinAbs ∧
        (PathDigits.firstTable chain digits x).valMinAbs ≤ 0
  | none, _, _, _ => by
      rw [show PathDigits.firstTable none _ _ = 0 from rfl, ZMod.valMinAbs_zero]
      exact ⟨by decide, le_rfl⟩
  | some _, _, binary, x => Digits.sequence binary.1 (by decide) le_rfl x

theorem PathDigits.Binary.secondTable : {chain : Option ChainShape} →
    {digits : PathDigits chain} → PathDigits.Binary chain digits → ∀ x,
      -1 ≤ (PathDigits.secondTable chain digits x).valMinAbs ∧
        (PathDigits.secondTable chain digits x).valMinAbs ≤ 0
  | none, _, _, _ => by
      rw [show PathDigits.secondTable none _ _ = 0 from rfl, ZMod.valMinAbs_zero]
      exact ⟨by decide, le_rfl⟩
  | some _, _, binary, x => Digits.sequence binary.2 (by decide) le_rfl x

end Cells

section Flat

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {w : Witness O fold}

/-- Each cell of the flat witness of digit vectors with the certified alphabets is in `A_{b⋆}`,
and each cell from the first digit of `F_1` is in `{−1, 0}` (Akita (76) and Remark 6.2), for
`b⋆ ≥ 2`. -/
theorem Witness.Alphabets.flat_cell (alphabets : w.Alphabets) (two : 2 ≤ S.certifiedBase)
    (x : ℕ) :
    (-((S.certifiedBase / 2 : ℕ) : ℤ) ≤ (w.flat x).valMinAbs ∧
      (w.flat x).valMinAbs ≤ ((S.certifiedBase / 2 : ℕ) : ℤ) - 1) ∧
      (O.chainStart fold ≤ x → -1 ≤ (w.flat x).valMinAbs ∧ (w.flat x).valMinAbs ≤ 0) := by
  have low : -((S.certifiedBase / 2 : ℕ) : ℤ) ≤ 0 := by omega
  have high : (0 : ℤ) ≤ ((S.certifiedBase / 2 : ℕ) : ℤ) - 1 := by omega
  have binary : ∀ (z : Zq) (late : Prop), -1 ≤ z.valMinAbs ∧ z.valMinAbs ≤ 0 →
      (-((S.certifiedBase / 2 : ℕ) : ℤ) ≤ z.valMinAbs ∧
        z.valMinAbs ≤ ((S.certifiedBase / 2 : ℕ) : ℤ) - 1) ∧
        (late → -1 ≤ z.valMinAbs ∧ z.valMinAbs ≤ 0) :=
    fun z _ range => ⟨⟨by omega, by omega⟩, fun _ => range⟩
  by_cases response : x < S.responseCells fold
  · rw [Witness.flat_response _ response]
    exact ⟨Digits.sequence (fun j => alphabets.responseDigits _) low high x, fun start => by
      unfold OpeningShape.chainStart OpeningShape.innerStart at start
      omega⟩
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt response)
  by_cases partials : x < O.partialCells
  · rw [Witness.flat_partial _ partials]
    exact ⟨Digits.sequence (fun j => alphabets.partialDigits _) low high x, fun start => by
      unfold OpeningShape.chainStart OpeningShape.innerStart at start
      omega⟩
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt partials)
  rw [← Nat.add_assoc, ← show O.innerStart fold = S.responseCells fold + O.partialCells from rfl]
  by_cases inner : x < S.innerCells
  · rw [Witness.flat_inner _ inner]
    refine ⟨Digits.sequence (fun entry k small => ?_) low high x, fun start => by
      unfold OpeningShape.chainStart at start
      omega⟩
    show _ ≤ (coeff (ofCoeffs _) k).valMinAbs ∧ (coeff (ofCoeffs _) k).valMinAbs ≤ _
    rw [coeff_ofCoeffs _ small]
    exact alphabets.innerDigits _ _ _ (S.address_lt _ small)
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt inner)
  rw [← Nat.add_assoc, ← show O.chainStart fold = O.innerStart fold + S.innerCells from rfl]
  by_cases first : x < firstCells S.chain
  · rw [Witness.flat_first _ first]
    exact binary _ _ (alphabets.chainDigits.firstTable x)
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt first)
  rw [← Nat.add_assoc,
    ← show O.openFirstStart fold = O.chainStart fold + firstCells S.chain from rfl]
  by_cases openFirst : x < firstCells O.chain
  · rw [Witness.flat_openFirst _ openFirst]
    exact binary _ _ (alphabets.openDigits.firstTable x)
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt openFirst)
  rw [← Nat.add_assoc,
    ← show O.secondStart fold = O.openFirstStart fold + firstCells O.chain from rfl]
  by_cases second : x < secondCells S.chain
  · rw [Witness.flat_second _ second]
    exact binary _ _ (alphabets.chainDigits.secondTable x)
  obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt second)
  rw [← Nat.add_assoc,
    ← show O.openSecondStart fold = O.secondStart fold + secondCells S.chain from rfl]
  by_cases openSecond : x < secondCells O.chain
  · rw [Witness.flat_openSecond _ openSecond]
    exact binary _ _ (alphabets.openDigits.secondTable x)
  · rw [Witness.flat_of_le _ (show O.cells fold ≤ O.openSecondStart fold + x by
      unfold OpeningShape.cells
      omega), ZMod.valMinAbs_zero]
    exact ⟨⟨low, high⟩, fun _ => ⟨by decide, le_rfl⟩⟩

/-- The digit range statement of Akita Section 6.1 for the flat witness of digit vectors with
the certified alphabets. -/
theorem Witness.Alphabets.rangeHolds (alphabets : w.Alphabets) (two : 2 ≤ S.certifiedBase)
    (small : S.certifiedBase < fieldModulus) (μ : ℕ) : RangeHolds S.certifiedBase μ w.flat :=
  (rangeHolds_iff small μ _).mpr fun x _ => (alphabets.flat_cell two x).1

/-- The binary statement of Akita Remark 6.2 for the flat witness of digit vectors with the
certified alphabets. -/
theorem Witness.Alphabets.binaryOn (alphabets : w.Alphabets) (two : 2 ≤ S.certifiedBase)
    (μ : ℕ) : BinaryOn (O.binarySupport fold) μ w.flat :=
  fun x _ member => (mul_add_one_eq_zero_iff _).mpr ((alphabets.flat_cell two x).2 member.1)

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  {β : Module.Basis (Fin O.planes) Zq E} {st : Statement O E}
  {challenge : Fin S.blocks → Rq O.carrierDegree} {μ : ℕ} {cells : O.cells fold ≤ 2 ^ μ}

variable (fits : Type → Prop)

/-- The completeness of the checks of a last packing fold (Akita Lemma 10.7 with Theorem 10.9):
for digit vectors with the exact branch relation, the prover has a message for each draw
sequence, and the checks end in the claim `(r_2, w̃(r_2))` for the flat witness. -/
theorem flatFold_winsTo (scalar : fits E) (range : ∀ n, n ≤ rangeWidth S.certifiedBase → fits (Fin n → E))
    (cubic : fits (Fin (3 + 1) → E)) (two : 2 ≤ S.certifiedBase)
    (small : S.certifiedBase < fieldModulus) (batch : ℕ) (rowsFit : O.rowCount + 1 ≤ 2 ^ batch)
    (satisfies : w.Satisfies β st challenge) :
    WinsTo fits (foldCheck (flatReduction β st challenge μ cells batch rowsFit) S.certifiedBase
      (O.binarySupport fold) (O.compressed fold) w.flat) (foldDraws μ S.certifiedBase (O.compressed
        fold)) true fun result _ _ =>
        ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w.flat)) :=
  foldCheck_winsTo fits (flatReduction β st challenge μ cells batch rowsFit) S.certifiedBase
    scalar range cubic _ (satisfies.flat μ cells) (satisfies.alphabets.rangeHolds two small μ)
    (satisfies.alphabets.binaryOn two μ)

end Flat

namespace Extract

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

theorem treeCosts_length (E : Type) [Fintype E] (μ last : ℕ) :
    ∀ (arities : List ℕ) (nodes : ℕ),
      (treeCosts E μ arities nodes).length + last = treeDraws μ last arities := by
  intro arities
  induction arities with
  | nil => simp [treeCosts, treeDraws]
  | cons a rest step =>
      intro nodes
      simp only [treeCosts, treeDraws, List.length_cons, List.length_append,
        List.length_replicate, ← step (nodes * a)]
      omega

theorem foldCosts_length (E : Type) [Fintype E] (badBound vars batch base : ℕ) (bin : Bool) :
    (foldCosts E badBound vars batch base bin).length = foldDraws vars base bin := by
  simp only [foldCosts, rangeCosts, foldDraws, rangeDraws, List.length_cons, List.length_append,
    List.length_replicate, binCosts_length, ← treeCosts_length E vars vars (rangeArities base) 1]
  omega

section Play

variable {Wire Reply α : Type} [Fintype Reply] (read : Reader Wire) (fits : Type → Prop)
  {goal : α → Prop}

/-- The wire message of the honest prover before a draw: when the prover has a message for each
draw sequence, a wire message with the condition `keep` gives the game its message, and the
prover has a message for each draw sequence of the game after it. `some` and `free` state that a
wire message can have each message of a type of `fits` with the condition `keep`. -/
theorem settle_winsTo {game : Game α} {draws : ℕ} (keep : Game α → Wire → Prop)
    (some : ∀ after, ∃ wire, keep after wire)
    (free : ∀ (M : Type) [Nonempty M] (message : M) (after : Game α), fits M →
      ∃ wire, keep after wire ∧ read.message M wire 0 = message)
    (wins : WinsTo fits game draws true fun result _ _ => goal result) :
    ∃ wire, keep (settle read wire game 0) wire ∧
      WinsTo fits (settle read wire game 0) draws false fun result _ _ => goal result := by
  cases game with
  | pure result =>
      obtain ⟨wire, kept⟩ := some (.pure result)
      exact ⟨wire, kept, wins⟩
  | draw C bad next =>
      obtain ⟨wire, kept⟩ := some (.draw C bad next)
      refine ⟨wire, kept, ?_⟩
      cases draws <;> exact wins
  | recv M next =>
      obtain ⟨_, sendable, message, won⟩ := wins
      obtain ⟨wire, kept, sent⟩ := free M message (next message) sendable
      have settled : settle read wire (.recv M next) 0 = next message := by
        show settle read wire (next (read.message M wire 0)) 1 = next message
        rw [sent]
        generalize next message = after at won
        cases after with
        | pure result => rfl
        | draw C bad later => rfl
        | recv M' later => exact absurd won.1 (by decide)
      exact ⟨wire, settled ▸ kept, settled ▸ won⟩

/-- The game of the honest prover after one answer of the oracle. -/
theorem advance_winsTo {game : Game α} {draws : ℕ} (reply : Reply)
    (wins : WinsTo fits game (draws + 1) false fun result _ _ => goal result) :
    WinsTo fits (advance reply game) draws true fun result _ _ => goal result := by
  cases game with
  | pure result => exact wins
  | draw C bad next => exact wins _
  | recv M next => exact absurd wins.1 (by decide)

theorem ended_of_winsTo {game : Game α}
    (wins : WinsTo fits game 0 false fun result _ _ => goal result) :
    ∃ result, game = .pure result ∧ goal result := by
  cases game with
  | pure result => exact ⟨result, rfl, wins⟩
  | draw C bad next => exact wins.elim
  | recv M next => exact absurd wins.1 (by decide)

/-- The transcript prefix of the honest prover for a game with `draws` draws: for the answers
`reply` of the oracle, which can read the earlier messages, wire messages exist such that the
game ends in the goal. The game reads each answer through `look`. `keep index` is the condition on
wire message `index`, with the game after it, and `invariant` is a property of the prefix that
each step with `keep` preserves. -/
theorem play {Answer : Type} (look : Answer → Reply) {start : Game α} {draws : ℕ}
    (wins : WinsTo fits start draws true fun result _ _ => goal result)
    (keep : ℕ → Game α → Wire → Prop) (some : ∀ index after, ∃ wire, keep index after wire)
    (free : ∀ (index : ℕ) (M : Type) [Nonempty M] (message : M) (after : Game α), fits M →
      ∃ wire, keep index after wire ∧ read.message M wire 0 = message)
    (reply : Wire → List (Answer × Wire) → Answer)
    (invariant : Wire → List (Answer × Wire) → Prop)
    (base : ∀ first, keep 0 (settle read first start 0) first → invariant first [])
    (step : ∀ first moves wire, moves.length < draws → invariant first moves →
      keep (moves.length + 1) (position read start first
        ((moves ++ [(reply first moves, wire)]).map fun move => (look move.1, move.2))) wire →
      invariant first (moves ++ [(reply first moves, wire)])) :
    ∃ first moves result, moves.length = draws ∧ invariant first moves ∧
      position read start first (moves.map fun move => (look move.1, move.2)) = .pure result ∧
        goal result := by
  have build : ∀ count ≤ draws, ∃ first moves, moves.length = count ∧ invariant first moves ∧
      WinsTo fits (position read start first (moves.map fun move => (look move.1, move.2)))
        (draws - count) false fun result _ _ => goal result := by
    intro count
    induction count with
    | zero =>
        intro _
        obtain ⟨first, kept, won⟩ := settle_winsTo read fits (keep 0) (some 0) (free 0) wins
        exact ⟨first, [], rfl, base first kept, won⟩
    | succ count induction =>
        intro inside
        obtain ⟨first, moves, length, kept, won⟩ := induction (by omega)
        rw [show draws - count = draws - (count + 1) + 1 by omega] at won
        obtain ⟨wire, next, later⟩ := settle_winsTo read fits (keep (count + 1))
          (some (count + 1)) (free (count + 1))
          (advance_winsTo fits (look (reply first moves)) won)
        rw [← position_map_append] at next later
        exact ⟨first, moves ++ [(reply first moves, wire)], by simp [length],
          step first moves wire (by omega) kept (length ▸ next), later⟩
  obtain ⟨first, moves, length, kept, won⟩ := build draws le_rfl
  rw [Nat.sub_self] at won
  obtain ⟨result, ended, good⟩ := ended_of_winsTo fits won
  exact ⟨first, moves, result, length, kept, ended, good⟩

end Play

section Path

variable {Wire Reply : Type} [Nonempty Reply] {total : ℕ}
  (accept : Prefix Wire Reply total → Prop) (table : Prefix Wire Reply total → Reply)

/-- The answer of the table of the oracle at the stage input with the message `first` and the
moves `moves`. -/
noncomputable def answer (first : Wire) (moves : List (Reply × Wire)) : Reply :=
  if short : moves.length ≤ total then table ⟨first, moves, short⟩
  else Classical.arbitrary Reply

theorem answer_eq (first : Wire) {moves : List (Reply × Wire)} (short : moves.length ≤ total) :
    answer table first moves = table ⟨first, moves, short⟩ :=
  dite_eq_left short

/-- The stage input with the message `first` and the moves `moves` is on the path of each proof
with these messages, for the table of the oracle. -/
def OnPath (first : Wire) (moves : List (Reply × Wire)) : Prop :=
  ∀ proof : ℕ → Wire, proof 0 = first →
    (∀ index (inside : index < moves.length), proof (index + 1) = moves[index].2) →
    ∃ short : moves.length ≤ total,
      pointPath total accept proof table moves.length = ⟨first, moves, short⟩

theorem onPath_nil (first : Wire) : OnPath accept table first [] := fun _ zero _ =>
  ⟨Nat.zero_le _, Prefix.ext zero rfl⟩

variable {accept table}

theorem OnPath.snoc {first : Wire} {moves : List (Reply × Wire)}
    (on : OnPath accept table first moves) (room : moves.length < total) (wire : Wire) :
    OnPath accept table first (moves ++ [(answer table first moves, wire)]) := by
  intro proof zero later
  obtain ⟨short, here⟩ := on proof zero fun index inside => by
    have same := later index (by rw [List.length_append]; omega)
    rwa [List.getElem_append_left inside] at same
  have last := later moves.length (by simp)
  simp only [List.getElem_concat_length] at last
  have length : (moves ++ [(answer table first moves, wire)]).length = moves.length + 1 := by
    simp
  have grown : (moves ++ [(answer table first moves, wire)]).length ≤ total := by omega
  refine ⟨grown, (congrArg (pointPath total accept proof table) length).trans ?_⟩
  show (transcript total accept).step proof (pointPath total accept proof table
    moves.length) (table (pointPath total accept proof table moves.length)) = _
  rw [here, transcript_step proof _ _ room]
  refine Prefix.ext rfl ?_
  show moves ++ [(table ⟨first, moves, short⟩, proof (moves.length + 1))] =
    moves ++ [(answer table first moves, wire)]
  rw [last, answer_eq table first short]

/-- A stage input of the last stage on the path is the complete transcript of a proof: the
stage input of the last stage of the path of this proof. -/
theorem OnPath.final {first : Wire} {moves : List (Reply × Wire)}
    (on : OnPath accept table first moves) (full : moves.length = total) :
    ∃ (proof : ℕ → Wire) (short : moves.length ≤ total), proof 0 = first ∧
      pointPath total accept proof table total = ⟨first, moves, short⟩ := by
  obtain ⟨short, here⟩ := on (⟨first, moves, full.le⟩ : Prefix Wire Reply total).wire rfl
    fun index inside => by simp [Prefix.wire, List.getElem?_eq_getElem inside]
  exact ⟨(⟨first, moves, full.le⟩ : Prefix Wire Reply total).wire, short, rfl,
    (congrArg (pointPath total accept _ table) full).symm.trans here⟩

variable (accept table) in
/-- Each answer of a transcript on the path of a table is an answer of the table. -/
theorem path_answers (proof : ℕ → Wire) : ∀ (level : ℕ),
    ∀ move ∈ (pointPath total accept proof table level).rest, ∃ input, move.1 = table input
  | 0 => fun _ member => absurd member (by simp [pointPath, transcript])
  | level + 1 => fun move member => by
      have earlier := path_answers proof level
      by_cases room : (pointPath total accept proof table level).rest.length < total
      · have grown : (pointPath total accept proof table (level + 1)).rest =
            (pointPath total accept proof table level).rest ++
              [(table (pointPath total accept proof table level), proof
                ((pointPath total accept proof table level).rest.length + 1))] := by
          show ((transcript total accept).step proof _ _).rest = _
          rw [transcript_step proof _ _ room]
          rfl
        rw [grown] at member
        rcases List.mem_append.mp member with old | new
        · exact earlier move old
        · rw [List.mem_singleton.mp new]
          exact ⟨_, rfl⟩
      · have same : (pointPath total accept proof table (level + 1)).rest =
            (pointPath total accept proof table level).rest := by
          have stay : (transcript total accept).step proof
              (pointPath total accept proof table level)
              (table (pointPath total accept proof table level)) =
                pointPath total accept proof table level := dite_eq_right room
          show ((transcript total accept).step proof _ _).rest = _
          rw [stay]
        rw [same] at member
        exact earlier move member

/-- Each answer of the moves of a stage input on the path is an answer of the table. -/
theorem OnPath.answers {first : Wire} {moves : List (Reply × Wire)}
    (on : OnPath accept table first moves) : ∀ move ∈ moves, ∃ input, move.1 = table input := by
  intro move member
  have short : moves.length ≤ total := by
    by_contra long
    have bound := (on (fun index => if index = 0 then first else
      ((moves[index - 1]?).map Prod.snd).getD first) rfl fun index inside => by
        simp [List.getElem?_eq_getElem inside]).1
    exact long bound
  obtain ⟨_, here⟩ := on (⟨first, moves, short⟩ : Prefix Wire Reply total).wire rfl
    fun index inside => by simp [Prefix.wire, List.getElem?_eq_getElem inside]
  have rest := congrArg Prefix.rest here
  exact path_answers accept table _ moves.length move (by rw [rest]; exact member)

end Path

namespace Tier

variable {Block : Type} {Ring : Block → Type}
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type}
  (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source) {total : ℕ}
  [Fintype Slot] [Nonempty Slot] {accept : Prefix Wire (Option Coord → Slot) total → Prop}
  {table : Prefix Wire (Option Coord → Slot) total → Option Coord → Slot}

/-- The work bits of the level are at most 57, the largest target of the Akita code
(`jolt_transcript::MAX_POW_BITS`). -/
def Capped : Prop := T.foldBits ≤ 57 ∧ ∀ g ∈ T.bits, g ≤ 57

omit [Fintype Slot] [Nonempty Slot] in
theorem Capped.getD {T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source}
    (capped : T.Capped) (index : ℕ) : T.bits.getD index 0 ≤ 57 := by
  rw [List.getD_eq_getElem?_getD]
  rcases found : T.bits[index]? with _ | g
  · exact Nat.zero_le _
  · exact capped.2 g (List.mem_of_getElem? found)

/-- On the path of a table whose work slots have 57 zero work bits, the answer of each stage of
the moves has `g ≤ 57` zero work bits. -/
theorem path_worked {first : Wire} {moves later : List ((Option Coord → Slot) × Wire)}
    {stage : Prefix Wire (Option Coord → Slot) total} (on : OnPath accept table first moves)
    (shape : stage.rest = moves ++ later) (worked : ∀ input, table input none ∈ workZeros Slot 57)
    {index g : ℕ} (inside : index < moves.length) (capped : g ≤ 57) :
    ∀ reply, stage.reply index = some reply → reply none ∈ workZeros Slot g := by
  intro reply answered
  have found : stage.reply index = some moves[index].1 := by
    simp [Prefix.reply, shape, List.getElem?_append_left inside, List.getElem?_eq_getElem inside]
  rw [found] at answered
  cases answered
  obtain ⟨input, same⟩ := on.answers _ (List.getElem_mem inside)
  rw [same]
  exact workZeros_anti Slot capped (worked input)

end Tier

end Extract

end NightstreamFPrime.Spec.Compression.Akita
