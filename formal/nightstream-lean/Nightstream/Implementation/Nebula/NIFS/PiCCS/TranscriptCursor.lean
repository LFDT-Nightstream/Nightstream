import Nightstream.Implementation.Nebula.NIFS.PiCCS.TranscriptRows
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor

/-!
Contract: exact post-PiCCS cursor for the selected V2 transcript.

Owns the structural proof that all twenty-five SumCheck rounds end with a
challenge gate at cursor zero, the complete PiCCS output has exactly 25,724
fields, and its absorption leaves the shared transcript at cursor four.

Does not own field values, Poseidon2 row semantics, PiRLC candidates,
physical placement, cryptographic security, Rust, or NIFS acceptance.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductPiCcsTranscriptCursor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductPiCcsTranscriptRows
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private theorem length_flatMap_uniform
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha -> List Beta)
    (count : Nat) (uniform : forall item, (values item).length = count) :
    (items.flatMap values).length = items.length * count := by
  induction items with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [uniform, inductionHypothesis, Nat.add_mul, Nat.add_comm]

theorem fullOutputPayload_length (input : Input) :
    (fullOutputPayload input).length = 25704 := by
  have coefficientsLength : forall source matrix,
      ((canonicalFinIndices selectedShape.coefficientCount).flatMap
        fun coefficient =>
          carriedFields (input.fullOutput source matrix coefficient)).length =
        108 := by
    intro source matrix
    rw [length_flatMap_uniform _ _ 2 (fun _ => rfl),
      canonicalFinIndices_length]
    rfl
  have matricesLength : forall source,
      ((canonicalFinIndices selectedShape.matrixCount).flatMap fun matrix =>
        (canonicalFinIndices selectedShape.coefficientCount).flatMap
          fun coefficient =>
            carriedFields
              (input.fullOutput source matrix coefficient)).length = 1512 := by
    intro source
    rw [length_flatMap_uniform _ _ 108 (coefficientsLength source),
      canonicalFinIndices_length]
    rfl
  unfold fullOutputPayload
  rw [length_flatMap_uniform _ _ 1512 matricesLength,
    canonicalFinIndices_length]
  rfl

theorem fullOutputFields_length (input : Input) :
    (fullOutputFields input).length = 25724 := by
  unfold fullOutputFields proverMessageFields
  rw [List.length_append, fullOutputPayload_length]
  decide

/-- A nonempty fixed-width round replay ends at the last challenge gate. -/
theorem replayRoundsGo_absorbed (input : Input) :
    forall rounds index builder,
      (replayRoundsGo input rounds index builder).builder.absorbed =
        match rounds with
        | [] => builder.absorbed
        | _ :: _ => 0
  | [], _, _ => rfl
  | round :: rest, index, builder => by
      let absorbed := SymbolicDuplex.absorbMany input.transcriptBase
        (roundFields index round) builder
      let sampled := squeezeVerifierChallenge
        (4 + 2 * index) (3 + index) 46 [] input absorbed
      cases rest with
      | nil => rfl
      | cons next tail =>
          simpa [replayRoundsGo, absorbed, sampled] using
            replayRoundsGo_absorbed input (next :: tail) (index + 1) sampled.2

theorem replayRounds_absorbed (input : Input) :
    (replayRounds input).builder.absorbed = 0 := by
  unfold replayRounds
  rw [replayRoundsGo_absorbed]
  rfl

private theorem after_four_four : SymbolicDuplexCursor.after 4 4 = 4 := by
  decide

private theorem after_four_four_mul (blocks : Nat) :
    SymbolicDuplexCursor.after 4 (4 * blocks) = 4 := by
  induction blocks with
  | zero => rfl
  | succ blocks inductionHypothesis =>
      rw [Nat.mul_succ, SymbolicDuplexCursor.after_add,
        inductionHypothesis, after_four_four]

theorem after_zero_25724 :
    SymbolicDuplexCursor.after 0 25724 = 4 := by
  rw [show 25724 = 4 + 4 * 6430 by decide,
    SymbolicDuplexCursor.after_add]
  change SymbolicDuplexCursor.after 4 (4 * 6430) = 4
  exact after_four_four_mul 6430

/-- The selected complete PiCCS output fixes the common PiRLC start cursor.
This is derived from the serialized output length, not supplied by a caller. -/
theorem afterFullOutput_absorbed (input : Input) :
    (afterFullOutput input).absorbed = 4 := by
  unfold afterFullOutput
  rw [SymbolicDuplexCursor.absorbMany_absorbed,
    fullOutputFields_length, replayRounds_absorbed, after_zero_25724]

end Nightstream.Implementation.Nebula.ProductPiCcsTranscriptCursor
