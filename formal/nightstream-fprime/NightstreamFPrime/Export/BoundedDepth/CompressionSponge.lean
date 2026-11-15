import NightstreamFPrime.Lifecycle.BoundedDepth.State
import NightstreamFPrime.Lifecycle.RandomOracle.ByteWords
import NightstreamFPrime.Spec.Compression.Rust.Transcript

/-! The words of the compression transcript (`SPEC.md` section 5.4, rows 1 to 17) and the session
bytes of the Akita opening.

A run (`Run`) holds what the final verifier absorbs: the fresh claim of the Π_CCS run, its rounds
and its output, then the messages of the argument (`Messages`): the bytes of the compression key,
the claims, the two Akita commitments, the rounds and the values of both sum-checks. `pieces` is
the transcript of a run as a list of absorbs and reads (`Lifecycle.RandomOracle.Sponge.Piece`).
Rows 4 to 17 have the operations and the word counts of `Spec.Compression.Rust.schedule`
(`argumentPieces_fit`), which a Rust test replays on a real proof.

No message has a length word of the sponge. Each length is a size of the statement (`Sizes`,
`Spec.Compression.Rust.Shape`), or the message is a byte string, whose first word is its byte
count. So two runs of one statement shape are alike at each position (`pieces_alike`), and the
sponge calls of the transcript fix the run (`calls_injective`). The session bytes are a label and
lanes 0 to 3 of the sponge state after row 17. Two runs with one session are equal, or their
transcripts are two distinct lists of sponge calls with the same lanes 0 to 3 (`session_binds`).
Akita reads these session bytes with its own sponge; Lean does not model that sponge. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.CompressionSponge

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Compression
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.RandomOracle.Sponge

def ccsDomain : List F := BoundedDepth.domainChunk "Nightstream/OptionB/compress/ccs/v1"

def argumentDomain : List F := BoundedDepth.domainChunk "Nightstream/compress/akita/v2"

def sessionLabel : List Byte :=
  "Nightstream/compress/akita/v2/opening".toList.map fun char => Fin.ofNat 256 char.toNat

/-- A CE claim as the transcript absorbs it (row 6): the ring columns, the table bound plus 1 or
0 for no table, then the words of the commitment, of the public input, of the point, of `Eval_K`
and of `Eval_A`. A value of `K` is two words. -/
@[ext] structure Claim where
  blocks : ℕ
  norm : ℕ
  commitment : List F
  cells : List F
  point : List F
  evalK : List F
  evalA : List F

/-- The five size words of a claim: `w`, the norm word, `m_in`, `ℓ` and `t`. -/
def Claim.sizes (claim : Claim) : List F :=
  [claim.blocks, claim.norm, claim.cells.length, claim.point.length / 2,
    claim.evalA.length / (2 * 64)].map Poseidon2.ofNat

def Claim.words (claim : Claim) : List F :=
  claim.sizes ++ claim.commitment ++ claim.cells ++ claim.point ++ claim.evalK ++ claim.evalA

/-- A claim has the sizes `(m_in, ℓ, t)` of the statement, with `κ = 18` and `d = 64`. -/
structure Claim.Shaped (claim : Claim) (shape : ℕ × ℕ × ℕ) : Prop where
  blocks : claim.blocks < fieldModulus
  norm : claim.norm < fieldModulus
  commitment : claim.commitment.length = 18 * 64
  cells : claim.cells.length = shape.1
  point : claim.point.length = 2 * shape.2.1
  evalK : claim.evalK.length = 2 * 64
  evalA : claim.evalA.length = 2 * (shape.2.2 * 64)

theorem Claim.words_length {claim : Claim} {shape : ℕ × ℕ × ℕ} (shaped : claim.Shaped shape) :
    claim.words.length = 5 + 18 * 64 + shape.1 + 2 * (shape.2.1 + 64 + shape.2.2 * 64) := by
  simp only [Claim.words, Claim.sizes, List.length_append, List.length_map, List.length_cons,
    List.length_nil, shaped.commitment, shaped.cells, shaped.point, shaped.evalK, shaped.evalA]
  omega

/-- The words of a claim fix the claim, among the claims of one shape. -/
theorem Claim.words_injective {left right : Claim} {shape : ℕ × ℕ × ℕ}
    (leftShaped : left.Shaped shape) (rightShaped : right.Shaped shape)
    (same : left.words = right.words) : left = right := by
  unfold Claim.words at same
  obtain ⟨same, evalA⟩ := List.append_inj' same (by rw [leftShaped.evalA, rightShaped.evalA])
  obtain ⟨same, evalK⟩ := List.append_inj' same (by rw [leftShaped.evalK, rightShaped.evalK])
  obtain ⟨same, point⟩ := List.append_inj' same (by rw [leftShaped.point, rightShaped.point])
  obtain ⟨same, cells⟩ := List.append_inj' same (by rw [leftShaped.cells, rightShaped.cells])
  obtain ⟨sizes, commitment⟩ := List.append_inj' same
    (by rw [leftShaped.commitment, rightShaped.commitment])
  simp only [Claim.sizes, List.map_cons, List.cons.injEq] at sizes
  exact Claim.ext (ofNat_injective leftShaped.blocks rightShaped.blocks sizes.1)
    (ofNat_injective leftShaped.norm rightShaped.norm sizes.2.1) commitment cells point evalK evalA

/-- The messages of the argument (rows 5 to 17). -/
@[ext] structure Messages where
  key : List Byte
  claims : List Claim
  witness : List Byte
  inverse : List Byte
  rounds : List (List F)
  witnessValue : List F
  inverseValues : List F
  keyWeight : List F
  keyRounds : List (List F)
  keyValue : List F

structure Messages.Shaped (messages : Messages) (shape : Rust.Shape) : Prop where
  key : messages.key.length < fieldModulus
  claims : List.Forall₂ Claim.Shaped messages.claims shape.claims
  witness : messages.witness.length < fieldModulus
  inverse : messages.inverse.length < fieldModulus
  rounds : messages.rounds.length = shape.witnessVars
  round : ∀ round ∈ messages.rounds, round.length = 8
  witnessValue : messages.witnessValue.length = 2
  inverseValues : messages.inverseValues.length = 8
  keyWeight : messages.keyWeight.length = 2
  keyRounds : messages.keyRounds.length = Rust.laneVars + shape.keyBlockVars
  keyRound : ∀ round ∈ messages.keyRounds, round.length = 6
  keyValue : messages.keyValue.length = 2

/-- One round of a sum-check: its coefficients, then the round challenge. -/
def roundPieces (round : List F) : List Piece := [.absorb round, .draw 1]

/-- Rows 4 to 17: the transcript of the argument. -/
def argumentPieces (shape : Rust.Shape) (messages : Messages) : List Piece :=
  .absorb argumentDomain :: .absorb (byteWords messages.key) ::
    ((messages.claims.map fun claim => Piece.absorb claim.words) ++
      ([.absorb (byteWords messages.witness), .draw 2, .absorb (byteWords messages.inverse),
          .draw (4 + 4 + 2 * shape.tables + shape.spanVars)] ++
        (messages.rounds.flatMap roundPieces ++
          ([.absorb messages.witnessValue, .absorb messages.inverseValues,
              .absorb messages.keyWeight, .draw 2] ++
            (messages.keyRounds.flatMap roundPieces ++ [.absorb messages.keyValue])))))

/-- The sizes of the Π_CCS run of the fresh instance (rows 1 to 3): the words of the fresh claim,
the rounds, the words of a round and the words of the output. -/
structure Sizes where
  fresh : ℕ
  rounds : ℕ
  roundWords : ℕ
  output : ℕ

/-- What the final verifier absorbs: the fresh claim, the rounds and the output of the Π_CCS run,
then the messages of the argument. Row 2 and row 7 absorb the same commitment of `W`. -/
@[ext] structure Run where
  fresh : List F
  ccsRounds : List (List F)
  ccsOutput : List F
  argument : Messages

structure Run.Shaped (run : Run) (sizes : Sizes) (shape : Rust.Shape) : Prop where
  fresh : run.fresh.length = sizes.fresh
  ccsRounds : run.ccsRounds.length = sizes.rounds
  ccsRound : ∀ round ∈ run.ccsRounds, round.length = sizes.roundWords
  ccsOutput : run.ccsOutput.length = sizes.output
  argument : run.argument.Shaped shape

/-- Rows 1 to 3: the transcript of the Π_CCS run of the fresh instance. -/
def ccsPieces (sizes : Sizes) (run : Run) : List Piece :=
  .absorb ccsDomain :: .absorb (run.fresh ++ byteWords run.argument.witness) ::
    .draw (sizes.rounds + 1) :: (run.ccsRounds.flatMap roundPieces ++ [.absorb run.ccsOutput])

/-- Rows 1 to 17: the compression transcript. -/
def pieces (sizes : Sizes) (shape : Rust.Shape) (run : Run) : List Piece :=
  ccsPieces sizes run ++ argumentPieces shape run.argument

/-- A piece has the operation and the word count of a call of `Spec.Compression.Rust.calls`. A
byte string has fewer than `q` bytes. -/
def Fits : Rust.Call → Piece → Prop
  | .absorb count, .absorb words => words.length = count
  | .bytes, .absorb words =>
      ∃ bytes : List Byte, bytes.length < fieldModulus ∧ words = byteWords bytes
  | .draw count, .draw count' => count' = count
  | _, _ => False

theorem Fits.words {words : List F} {count : ℕ} (lengthEqual : words.length = count) :
    Fits (.absorb count) (.absorb words) :=
  lengthEqual

theorem Fits.bytes {bytes : List Byte} (small : bytes.length < fieldModulus) :
    Fits .bytes (.absorb (byteWords bytes)) :=
  ⟨bytes, small, rfl⟩

theorem Fits.reads (count : ℕ) : Fits (.draw count) (.draw count) :=
  rfl

private theorem alikeReads (count : ℕ) : Piece.Alike (.draw count) (.draw count) :=
  rfl

/-- Two pieces that fit one call are alike. -/
theorem Fits.alike {call : Rust.Call} {left right : Piece} (leftFits : Fits call left)
    (rightFits : Fits call right) : Piece.Alike left right := by
  cases call with
  | absorb count =>
      cases left <;> cases right <;>
        first
        | exact False.elim leftFits
        | exact False.elim rightFits
        | exact Piece.alike_of_length (Eq.trans leftFits (Eq.symm rightFits))
  | bytes =>
      cases left with
      | draw _ => exact False.elim leftFits
      | absorb _ =>
          cases right with
          | draw _ => exact False.elim rightFits
          | absorb _ =>
              obtain ⟨bytes, small, rfl⟩ := leftFits
              obtain ⟨bytes', small', rfl⟩ := rightFits
              exact byteWords_alike (front := []) (front' := []) rfl small small'
  | draw count =>
      cases left <;> cases right <;>
        first
        | exact False.elim leftFits
        | exact False.elim rightFits
        | exact Eq.trans leftFits (Eq.symm rightFits)

theorem alike_of_fits : ∀ {calls : List Rust.Call} {left right : List Piece},
    List.Forall₂ Fits calls left → List.Forall₂ Fits calls right →
      List.Forall₂ Piece.Alike left right
  | _, _, _, .nil, .nil => .nil
  | _, _, _, .cons leftFits leftRest, .cons rightFits rightRest =>
      .cons (leftFits.alike rightFits) (alike_of_fits leftRest rightRest)

/-- The calls of `Spec.Compression.Rust.schedule`, in order. -/
def scheduleCalls (shape : Rust.Shape) : List Rust.Call := (Rust.schedule shape).flatMap (·.2)

private theorem replicate_calls (first second : Rust.Label × List Rust.Call) : ∀ count : ℕ,
    List.flatMap (fun step => step.2) (List.replicate count [first, second]).flatten =
      (List.replicate count (first.2 ++ second.2)).flatten
  | 0 => rfl
  | count + 1 => by simp [List.replicate_succ, replicate_calls first second count]

theorem scheduleCalls_eq (shape : Rust.Shape) : scheduleCalls shape =
    .absorb 12 :: .bytes :: ((shape.claims.map fun claim =>
        Rust.Call.absorb (5 + 18 * 64 + claim.1 + 2 * (claim.2.1 + 64 + claim.2.2 * 64))) ++
      ([.bytes, .draw 2, .bytes, .draw (4 + 4 + 2 * shape.tables + shape.spanVars)] ++
        ((List.replicate shape.witnessVars [Rust.Call.absorb 8, .draw 1]).flatten ++
          ([.absorb 2, .absorb 8, .absorb 2, .draw 2] ++
            ((List.replicate (Rust.laneVars + shape.keyBlockVars)
                [Rust.Call.absorb 6, .draw 1]).flatten ++ [.absorb 2]))))) := by
  simp [scheduleCalls, Rust.schedule, Rust.labels, Rust.calls, replicate_calls]

private theorem rounds_fit {width : ℕ} : ∀ {rounds : List (List F)},
    (∀ round ∈ rounds, round.length = width) →
      List.Forall₂ Fits (List.replicate rounds.length [Rust.Call.absorb width, .draw 1]).flatten
        (rounds.flatMap roundPieces)
  | [], _ => .nil
  | round :: rounds, widths => by
      simp only [List.length_cons, List.replicate_succ, List.flatten_cons, List.flatMap_cons,
        roundPieces, List.cons_append, List.nil_append]
      exact .cons (Fits.words (widths round List.mem_cons_self))
        (.cons (Fits.reads 1)
          (rounds_fit fun other member => widths other (List.mem_cons_of_mem _ member)))

private theorem claims_fit : ∀ {claims : List Claim} {shapes : List (ℕ × ℕ × ℕ)},
    List.Forall₂ Claim.Shaped claims shapes →
      List.Forall₂ Fits
        (shapes.map fun claim =>
          Rust.Call.absorb (5 + 18 * 64 + claim.1 + 2 * (claim.2.1 + 64 + claim.2.2 * 64)))
        (claims.map fun claim => Piece.absorb claim.words)
  | _, _, .nil => .nil
  | _, _, .cons head tail => .cons (Fits.words (Claim.words_length head)) (claims_fit tail)

/-- Rows 4 to 17 have the operations and the word counts of `Spec.Compression.Rust.schedule`. -/
theorem argumentPieces_fit {shape : Rust.Shape} {messages : Messages}
    (shaped : messages.Shaped shape) :
    List.Forall₂ Fits (scheduleCalls shape) (argumentPieces shape messages) := by
  have rounds := rounds_fit shaped.round
  have keyRounds := rounds_fit shaped.keyRound
  rw [shaped.rounds] at rounds
  rw [shaped.keyRounds] at keyRounds
  rw [scheduleCalls_eq]
  have domain : argumentDomain.length = 12 := by
    simp [argumentDomain, BoundedDepth.domainChunk, Poseidon2.rate]
  exact .cons (Fits.words domain) (.cons (Fits.bytes shaped.key)
    (List.rel_append (claims_fit shaped.claims)
      (List.rel_append
        (.cons (Fits.bytes shaped.witness) (.cons (Fits.reads _)
          (.cons (Fits.bytes shaped.inverse) (.cons (Fits.reads _) .nil))))
        (List.rel_append rounds
          (List.rel_append
            (.cons (Fits.words shaped.witnessValue) (.cons (Fits.words shaped.inverseValues)
              (.cons (Fits.words shaped.keyWeight) (.cons (Fits.reads _) .nil))))
            (List.rel_append keyRounds (.cons (Fits.words shaped.keyValue) .nil)))))))

variable {sizes : Sizes} {shape : Rust.Shape} {left right : Run}

/-- Two runs of one statement shape are alike at each position of the transcript. -/
theorem pieces_alike (leftShaped : left.Shaped sizes shape) (rightShaped : right.Shaped sizes shape) :
    List.Forall₂ Piece.Alike (pieces sizes shape left) (pieces sizes shape right) := by
  have leftRounds := rounds_fit leftShaped.ccsRound
  have rightRounds := rounds_fit rightShaped.ccsRound
  rw [leftShaped.ccsRounds] at leftRounds
  rw [rightShaped.ccsRounds] at rightRounds
  exact List.rel_append
    (.cons (Piece.alike_of_length rfl)
      (.cons (byteWords_alike (by rw [leftShaped.fresh, rightShaped.fresh])
          leftShaped.argument.witness rightShaped.argument.witness)
        (.cons (alikeReads _) (List.rel_append (alike_of_fits leftRounds rightRounds)
          (.cons (Piece.alike_of_length (by rw [leftShaped.ccsOutput, rightShaped.ccsOutput]))
            .nil)))))
    (alike_of_fits (argumentPieces_fit leftShaped.argument)
      (argumentPieces_fit rightShaped.argument))

private theorem rounds_append_inj : ∀ {rounds rounds' : List (List F)} {rest rest' : List Piece},
    rounds.length = rounds'.length →
      rounds.flatMap roundPieces ++ rest = rounds'.flatMap roundPieces ++ rest' →
        rounds = rounds' ∧ rest = rest'
  | [], [], _, _, _, same => ⟨rfl, by simpa using same⟩
  | [], _ :: _, _, _, lengths, _ => by simp at lengths
  | _ :: _, [], _, _, lengths, _ => by simp at lengths
  | round :: rounds, round' :: rounds', rest, rest', lengths, same => by
      simp only [List.flatMap_cons, roundPieces, List.cons_append, List.nil_append,
        List.cons.injEq, Piece.absorb.injEq, true_and] at same
      obtain ⟨tail, later⟩ := rounds_append_inj (by simpa using lengths) same.2
      exact ⟨by rw [same.1, tail], later⟩

private theorem claims_injective : ∀ {claims claims' : List Claim} {shapes : List (ℕ × ℕ × ℕ)},
    List.Forall₂ Claim.Shaped claims shapes → List.Forall₂ Claim.Shaped claims' shapes →
      (claims.map fun claim => Piece.absorb claim.words) =
        (claims'.map fun claim => Piece.absorb claim.words) → claims = claims'
  | _, _, _, .nil, .nil, _ => rfl
  | _, _, _, .cons head tail, .cons head' tail', same => by
      simp only [List.map_cons, List.cons.injEq, Piece.absorb.injEq] at same
      rw [Claim.words_injective head head' same.1, claims_injective tail tail' same.2]

/-- The pieces of the transcript fix the run, among the runs of one statement shape. -/
theorem pieces_injective (leftShaped : left.Shaped sizes shape)
    (rightShaped : right.Shaped sizes shape)
    (same : pieces sizes shape left = pieces sizes shape right) : left = right := by
  have leftArgument := leftShaped.argument
  have rightArgument := rightShaped.argument
  simp only [pieces, ccsPieces, argumentPieces, List.cons_append, List.cons.injEq,
    Piece.absorb.injEq, true_and, List.append_assoc] at same
  obtain ⟨statement, same⟩ := same
  obtain ⟨fresh, witnessWords⟩ := List.append_inj statement
    (by rw [leftShaped.fresh, rightShaped.fresh])
  obtain ⟨ccsRounds, same⟩ := rounds_append_inj
    (by rw [leftShaped.ccsRounds, rightShaped.ccsRounds]) same
  simp only [List.nil_append, List.cons.injEq, Piece.absorb.injEq, true_and] at same
  obtain ⟨ccsOutput, keyWords, same⟩ := same
  obtain ⟨claimPieces, same⟩ := List.append_inj same
    (by rw [List.length_map, List.length_map, leftArgument.claims.length_eq,
      rightArgument.claims.length_eq])
  simp only [List.cons.injEq, Piece.absorb.injEq, true_and] at same
  obtain ⟨-, inverseWords, same⟩ := same
  obtain ⟨rounds, same⟩ := rounds_append_inj
    (by rw [leftArgument.rounds, rightArgument.rounds]) same
  simp only [List.cons.injEq, Piece.absorb.injEq, true_and] at same
  obtain ⟨witnessValue, inverseValues, keyWeight, same⟩ := same
  obtain ⟨keyRounds, same⟩ := rounds_append_inj
    (by rw [leftArgument.keyRounds, rightArgument.keyRounds]) same
  simp only [List.cons.injEq, Piece.absorb.injEq, and_true] at same
  exact Run.ext fresh ccsRounds ccsOutput
    (Messages.ext (byteWords_injective leftArgument.key rightArgument.key keyWords)
      (claims_injective leftArgument.claims rightArgument.claims claimPieces)
      (byteWords_injective leftArgument.witness rightArgument.witness witnessWords)
      (byteWords_injective leftArgument.inverse rightArgument.inverse inverseWords)
      rounds witnessValue inverseValues keyWeight keyRounds same)

/-- The sponge calls of the compression transcript fix the run, among the runs of one statement
shape. -/
theorem calls_injective (leftShaped : left.Shaped sizes shape)
    (rightShaped : right.Shaped sizes shape)
    (same : calls 0 (pieces sizes shape left) = calls 0 (pieces sizes shape right)) :
    left = right :=
  pieces_injective leftShaped rightShaped
    (RandomOracle.Sponge.calls_injective (pieces_alike leftShaped rightShaped) 0 same)

/-- The sponge state after row 17. -/
def finalState (sizes : Sizes) (shape : Rust.Shape) (data : Run) : Transcript.State :=
  RandomOracle.Sponge.run Transcript.initialState (calls 0 (pieces sizes shape data))

/-- The session bytes of the Akita opening: the label, then lanes 0 to 3 of the sponge state
after row 17. -/
def session (sizes : Sizes) (shape : Rust.Shape) (data : Run) : List Byte :=
  sessionBytes sessionLabel (finalState sizes shape data)

/-- Two runs of one statement shape with one session are equal, or their transcripts are two
distinct lists of sponge calls whose states have the same lanes 0 to 3. -/
theorem session_binds (leftShaped : left.Shaped sizes shape)
    (rightShaped : right.Shaped sizes shape)
    (same : session sizes shape left = session sizes shape right) :
    left = right ∨
      (calls 0 (pieces sizes shape left) ≠ calls 0 (pieces sizes shape right) ∧
        (finalState sizes shape left).take Poseidon2.digestLen =
          (finalState sizes shape right).take Poseidon2.digestLen) := by
  by_cases equal : calls 0 (pieces sizes shape left) = calls 0 (pieces sizes shape right)
  · exact .inl (calls_injective leftShaped rightShaped equal)
  · exact .inr ⟨equal, sessionBytes_injective same⟩

end NightstreamFPrime.Export.BoundedDepth.CompressionSponge
