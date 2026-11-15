import NightstreamFPrime.Lifecycle.RandomOracle.Schedule

/-! The framing of a Poseidon2 transcript (`SECURITY_MODEL.md`, section Hashed encodings).

Rust `absorb_v1_2` adds each 12 words of a message to the rate lanes and permutes
(`Sponge.absorbCalls`). It writes no length word and no padding, so two messages with distinct
lengths can have one call list (`absorbCalls_not_injective`). A transcript is thus a list of
pieces (`Piece`): an absorb of a message, or a number of challenge reads. `calls` is the list of
sponge calls of a piece list, with the zero block that the sponge absorbs after each read at
position 5.

The call list fixes the pieces when the two piece lists are alike at each position
(`calls_injective`): two reads have one count, and the calls of two messages fix their length
(`Piece.Alike`). Two messages of one length are alike (`Piece.alike_of_length`), and so are two
messages with a count word that fixes their length (`Piece.alike_of_counted`). A fold transcript
is the piece list `Schedule.pieces`. Its pieces have one length at each position for every
statement and every prover message (`Schedule.pieces_alike`), and its calls are the calls of the
schedule (`Schedule.calls_pieces`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.Sponge

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle

/-- The calls of a message before any later calls fix the message, when its length is fixed. -/
theorem absorbCalls_append_inj {left right : List F} {rest rest' : List Call}
    (lengthEqual : left.length = right.length)
    (same : absorbCalls left ++ rest = absorbCalls right ++ rest') :
    left = right ∧ rest = rest' := by
  obtain ⟨calls, tail⟩ := List.append_inj same
    (by rw [absorbCalls_length, absorbCalls_length, lengthEqual])
  exact ⟨absorbCalls_injective lengthEqual calls, tail⟩

/-- Word `position` of a message is lane `position mod 12` of call `position / 12`. -/
theorem absorbCalls_word (words : List F) {position : ℕ} (inside : position < words.length) :
    ((absorbCalls words)[position / Poseidon2.rate]?.map fun call =>
      call ⟨position % Poseidon2.rate, Nat.mod_lt _ (by decide)⟩) = some words[position] := by
  have chunk : position / Poseidon2.rate <
      (words.length + Poseidon2.rate - 1) / Poseidon2.rate := by
    unfold Poseidon2.rate at *
    omega
  have split : position / Poseidon2.rate * Poseidon2.rate + position % Poseidon2.rate =
      position := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod position Poseidon2.rate
  simp [absorbCalls, chunk, padChunk, List.getD_eq_getElem?_getD,
    Nat.mod_lt _ (show 0 < Poseidon2.rate by decide), List.getElem?_drop, split,
    List.getElem?_eq_getElem inside]

/-- The calls of two messages, before any later calls, fix each word at a position of both. -/
theorem absorbCalls_append_word {left right : List F} {rest rest' : List Call}
    (same : absorbCalls left ++ rest = absorbCalls right ++ rest') {position : ℕ}
    (inLeft : position < left.length) (inRight : position < right.length) :
    left[position] = right[position] := by
  have bound : ∀ {words : List F}, position < words.length →
      position / Poseidon2.rate < (absorbCalls words).length := by
    intro words inside
    rw [absorbCalls_length]
    unfold Poseidon2.rate at *
    omega
  have call := congrArg (fun calls : List Call => calls[position / Poseidon2.rate]?) same
  simp only [List.getElem?_append_left (bound inLeft),
    List.getElem?_append_left (bound inRight)] at call
  have word := absorbCalls_word left inLeft
  rw [call, absorbCalls_word right inRight] at word
  exact (Option.some.inj word).symm

/-- An absorb has no length word and no padding: a message and the same message with one more
zero word have one call list. -/
theorem absorbCalls_not_injective :
    absorbCalls [1] = absorbCalls [1, 0] ∧ ([1] : List F) ≠ [1, 0] := by
  refine ⟨?_, by decide⟩
  unfold absorbCalls
  rw [show (([1] : List F).length + Poseidon2.rate - 1) / Poseidon2.rate = 1 from rfl,
    show (([1, 0] : List F).length + Poseidon2.rate - 1) / Poseidon2.rate = 1 from rfl]
  simp only [List.range_one, List.map_cons, List.map_nil, Nat.zero_mul, List.drop_zero]
  congr 1
  funext lane
  fin_cases lane <;> rfl

/-- One operation of a transcript: absorb the words of a message, or read `count` challenges. -/
inductive Piece where
  | absorb (words : List F)
  | draw (count : ℕ)

/-- The sponge calls of a piece list after `cursor` reads since the last call. An absorb of no
word makes no call. The reads since the last call take the positions 0, 1, and so on, and the
sponge absorbs a zero block after each read at position 5 (`Transcript.squeezeAt`). -/
def calls : ℕ → List Piece → List Call
  | _, [] => []
  | cursor, .absorb words :: rest =>
      absorbCalls words ++ calls (if words.isEmpty then cursor else 0) rest
  | cursor, .draw count :: rest =>
      List.replicate ((cursor + count) / Transcript.coinsPerChunk) squeeze ++
        calls ((cursor + count) % Transcript.coinsPerChunk) rest

/-- Two pieces of one transcript position: two reads of one count, or two messages whose calls,
before any later calls, fix their length. -/
def Piece.Alike : Piece → Piece → Prop
  | .absorb words, .absorb words' => ∀ rest rest' : List Call,
      absorbCalls words ++ rest = absorbCalls words' ++ rest' → words.length = words'.length
  | .draw count, .draw count' => count = count'
  | _, _ => False

theorem Piece.alike_of_length {words words' : List F} (lengthEqual : words.length = words'.length) :
    Piece.Alike (.absorb words) (.absorb words') :=
  fun _ _ _ => lengthEqual

/-- Two messages are alike when each has a front of one length, then a count word that fixes
the length of the words after it. -/
theorem Piece.alike_of_counted {front front' body body' : List F} {count count' : F}
    (fronts : front.length = front'.length)
    (rule : count = count' → body.length = body'.length) :
    Piece.Alike (.absorb (front ++ count :: body)) (.absorb (front' ++ count' :: body')) := by
  intro rest rest' same
  have word := absorbCalls_append_word same (position := front.length) (by simp)
    (by simp [fronts])
  have counts : count = count' := by simpa [fronts] using word
  simp [fronts, rule counts]

/-- The calls of a transcript fix its pieces, when the pieces are alike at each position. -/
theorem calls_injective {left right : List Piece} (alike : List.Forall₂ Piece.Alike left right) :
    ∀ cursor : ℕ, calls cursor left = calls cursor right → left = right := by
  induction alike with
  | nil => exact fun _ _ => rfl
  | @cons piece piece' rest rest' head _ tail =>
      intro cursor same
      cases piece with
      | absorb words =>
          cases piece' with
          | absorb words' =>
              simp only [calls] at same
              obtain ⟨equal, later⟩ := absorbCalls_append_inj (head _ _ same) same
              subst equal
              rw [tail _ later]
          | draw _ => exact absurd head id
      | draw count =>
          cases piece' with
          | absorb _ => exact absurd head id
          | draw count' =>
              have counts : count = count' := head
              subst counts
              simp only [calls] at same
              rw [tail _ (List.append_cancel_left same)]

end NightstreamFPrime.Lifecycle.RandomOracle.Sponge

namespace NightstreamFPrime.Lifecycle.RandomOracle.Schedule

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.RandomOracle.Sponge

variable (schedule : Schedule) {width : Nat}

/-- The pieces of one sum-check round: its polynomial, then the round challenge. -/
def roundPieces (messages : schedule.Messages width) (round : Fin schedule.shape.cubeVariables) :
    List Piece :=
  [.absorb (Transcript.serializeMessage (messages.rounds round).toMessage), .draw 1]

/-- The pieces of a fold transcript up to its output (SuperNeo v1.2 Sections 7.3 and 7.4): the
domain, the statement, `α` and `γ`, each round with its challenge, and the output. -/
def pieces (statement : schedule.Statement) (messages : schedule.Messages width) : List Piece :=
  [.absorb schedule.domainChunk, .absorb (schedule.statementWords statement),
      .draw (schedule.shape.cubeVariables + 1)] ++
    (canonicalFinIndices schedule.shape.cubeVariables).flatMap (schedule.roundPieces messages) ++
    [.absorb (schedule.outputWords messages.output)]

private theorem roundWords_length' (polynomial : SumCheck.Finite.FixedPolynomial K width) :
    (Transcript.serializeMessage polynomial.toMessage).length = 2 * (width + 1) := by
  have pairs : ∀ values : List K, (values.flatMap serializeK).length = 2 * values.length := by
    intro values
    induction values with
    | nil => rfl
    | cons _ _ ih => simp only [List.flatMap_cons, List.length_append, ih, serializeK,
        List.length_cons, List.length_nil]; omega
  rw [Transcript.serializeMessage, pairs]
  exact congrArg (2 * ·) polynomial.coefficients_length

private theorem rounds_alike (messages messages' : schedule.Messages width) :
    ∀ rounds : List (Fin schedule.shape.cubeVariables),
      List.Forall₂ Piece.Alike (rounds.flatMap (schedule.roundPieces messages))
        (rounds.flatMap (schedule.roundPieces messages'))
  | [] => .nil
  | round :: rounds => by
      simp only [List.flatMap_cons, roundPieces, List.cons_append, List.nil_append]
      exact .cons (Piece.alike_of_length (by rw [roundWords_length', roundWords_length']))
        (.cons rfl (rounds_alike messages messages' rounds))

/-- The fixed-length condition of a fold transcript: for every two statements and every two lists
of prover messages, the pieces are alike at each position. -/
theorem pieces_alike (statement statement' : schedule.Statement)
    (messages messages' : schedule.Messages width) :
    List.Forall₂ Piece.Alike (schedule.pieces statement messages)
      (schedule.pieces statement' messages') := by
  unfold pieces
  refine List.rel_append (List.rel_append ?_ (schedule.rounds_alike messages messages' _)) ?_
  · exact .cons (Piece.alike_of_length rfl)
      (.cons (Piece.alike_of_length (schedule.statementWords_length statement statement'))
        (.cons rfl .nil))
  · exact .cons (Piece.alike_of_length
      (schedule.outputWords_length messages.output messages'.output)) .nil

private theorem round_nonempty (messages : schedule.Messages width)
    (round : Fin schedule.shape.cubeVariables) :
    (Transcript.serializeMessage (messages.rounds round).toMessage).isEmpty = false := by
  rw [List.isEmpty_eq_false_iff, ← List.length_pos_iff, roundWords_length']
  omega

private theorem calls_rounds (messages : schedule.Messages width) (tail : List Piece) :
    ∀ (rounds : List (Fin schedule.shape.cubeVariables)) (cursor : ℕ), rounds ≠ [] →
      calls cursor (rounds.flatMap (schedule.roundPieces messages) ++ tail) =
        rounds.flatMap (schedule.roundCalls messages) ++ calls 1 tail
  | [], _, nonempty => absurd rfl nonempty
  | round :: rounds, cursor, _ => by
      have six : Transcript.coinsPerChunk = 6 := rfl
      simp only [List.flatMap_cons, roundPieces, List.cons_append, List.nil_append, calls,
        schedule.round_nonempty messages round, Bool.false_eq_true, ite_false, six, roundCalls,
        Nat.zero_add, Nat.reduceDiv, Nat.reduceMod, List.replicate_zero, List.append_assoc]
      by_cases last : rounds = []
      · subst last
        simp
      · rw [calls_rounds messages tail rounds 1 last]

/-- The calls of the pieces of a fold transcript are the calls of the schedule before its Π_RLC
challenges (`outputCalls`). -/
theorem calls_pieces (statement : schedule.Statement) (messages : schedule.Messages width) :
    calls 0 (schedule.pieces statement messages) = schedule.outputCalls statement messages := by
  have domain : schedule.domainChunk.isEmpty = false := by
    rw [List.isEmpty_eq_false_iff, ← List.length_pos_iff, schedule.domainChunk_length]
    decide
  have output : ∀ cursor, calls cursor [Piece.absorb (schedule.outputWords messages.output)] =
      absorbCalls (schedule.outputWords messages.output) := fun cursor => by simp [calls]
  simp only [pieces, List.cons_append, List.nil_append, calls, domain, Bool.false_eq_true,
    ite_false, outputCalls, roundsCalls, preRoundCalls, statementCalls, List.append_assoc]
  congr 2
  by_cases empty : (schedule.statementWords statement).isEmpty
  all_goals simp only [empty, ite_true, Bool.false_eq_true, ite_false, Nat.zero_add]
  all_goals congr 1
  all_goals
    by_cases none : canonicalFinIndices schedule.shape.cubeVariables = []
    · simp [none, output]
    · rw [schedule.calls_rounds messages _ _ _ none, output]

end NightstreamFPrime.Lifecycle.RandomOracle.Schedule
