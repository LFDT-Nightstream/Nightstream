import Nightstream.Implementation.Nebula.NIFS.PiCCS.TranscriptRowsFor
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor

/-!
Contract: exact post-PiCCS cursor at any generated-relation exponent.

The SumCheck round count changes with `rowVariables`, but every round ends in
a challenge gate at cursor zero. The complete PiCCS output has 22,700 fields
for every exponent because source, matrix, coefficient, and limb counts are
profile constants. Its absorption therefore leaves cursor four.

Assurance tier: exponent-indexed transcript geometry.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductPiCcsTranscriptCursorFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductPiCcsTranscriptRowsFor
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

theorem fullOutputPayload_length
    {rowVariables : Nat} (input : Input rowVariables) :
    (fullOutputPayload input).length = 22680 := by
  have coefficientsLength : forall source matrix,
      ((canonicalFinIndices (Shape rowVariables).coefficientCount).flatMap
        fun coefficient =>
          carriedFields (input.fullOutput source matrix coefficient)).length =
        108 := by
    intro source matrix
    rw [length_flatMap_uniform _ _ 2 (fun _ => rfl),
      canonicalFinIndices_length]
    rfl
  have matricesLength : forall source,
      ((canonicalFinIndices (Shape rowVariables).matrixCount).flatMap
        fun matrix =>
          (canonicalFinIndices (Shape rowVariables).coefficientCount).flatMap
            fun coefficient =>
              carriedFields
                (input.fullOutput source matrix coefficient)).length =
        1512 := by
    intro source
    rw [length_flatMap_uniform _ _ 108 (coefficientsLength source),
      canonicalFinIndices_length]
    rfl
  unfold fullOutputPayload
  rw [length_flatMap_uniform _ _ 1512 matricesLength,
    canonicalFinIndices_length]
  rfl

theorem fullOutputFields_length
    {rowVariables : Nat} (input : Input rowVariables) :
    (fullOutputFields input).length = 22700 := by
  unfold fullOutputFields ProductPiCcsTranscriptRowsFor.proverMessageFields
    ProductPiCcsTranscriptRows.proverMessageFields
  simp only [List.length_append, List.length_map, List.length_cons,
    List.length_nil, fullOutputPayload_length]
  have labelLength :
      ProductPoseidon2.proverMessageLabelFields.length = 16 := by
    decide
  rw [labelLength]

private theorem replayRoundsGo_absorbed_zero
    {rowVariables : Nat} (input : Input rowVariables) :
    forall rounds index builder,
      builder.absorbed = 0 ->
        (replayRoundsGo input rounds index builder).builder.absorbed = 0
  | [], _, _, zero => zero
  | round :: rest, index, builder, _ => by
      apply replayRoundsGo_absorbed_zero input rest (index + 1)
      exact SymbolicDuplex.squeezeK_absorbed _ _

theorem replayRounds_absorbed
    {rowVariables : Nat} (input : Input rowVariables) :
    (replayRounds input).builder.absorbed = 0 := by
  apply replayRoundsGo_absorbed_zero input
  exact SymbolicDuplex.squeezeK_absorbed _ _

private theorem after_four_four : SymbolicDuplexCursor.after 4 4 = 4 := by
  decide

private theorem after_four_four_mul (blocks : Nat) :
    SymbolicDuplexCursor.after 4 (4 * blocks) = 4 := by
  induction blocks with
  | zero => rfl
  | succ blocks inductionHypothesis =>
      rw [Nat.mul_succ, SymbolicDuplexCursor.after_add,
        inductionHypothesis, after_four_four]

theorem after_zero_22700 :
    SymbolicDuplexCursor.after 0 22700 = 4 := by
  rw [show 22700 = 4 + 4 * 5674 by decide,
    SymbolicDuplexCursor.after_add]
  change SymbolicDuplexCursor.after 4 (4 * 5674) = 4
  exact after_four_four_mul 5674

theorem afterFullOutput_absorbed
    {rowVariables : Nat} (input : Input rowVariables) :
    (afterFullOutput input).absorbed = 4 := by
  unfold afterFullOutput
  rw [SymbolicDuplexCursor.absorbMany_absorbed,
    fullOutputFields_length, replayRounds_absorbed, after_zero_22700]

end Nightstream.Implementation.Nebula.ProductPiCcsTranscriptCursorFor
