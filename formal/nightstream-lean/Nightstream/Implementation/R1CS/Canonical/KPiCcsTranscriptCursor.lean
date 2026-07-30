import Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscript
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor

/-!
Contract: expose the exact cursor handed from the canonical PiCCS transcript
to the PiRLC sampler.

Owns: the zero cursor after every verifier squeeze, the exact PiCCS output
serialization length, and the resulting post-output cursor formula.

Does not own: selection of the fixed thirteen-matrix relation, PiRLC rows,
call-frame decoding, or random-oracle security.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptCursor

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS.Canonical

@[simp] theorem derivePreSumcheck_builder_absorbed
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) :
    (KPiCcsTranscript.derivePreSumcheck input).builder.absorbed = 0 := by
  rfl

theorem replayRounds_builder_absorbed_of_zero
    {degree : Nat} (base : Nat) :
    ∀ (rounds : List (KFixedPhaseSumCheck.Round degree))
      (index : Nat) (builder : SymbolicDuplex.Builder),
      builder.absorbed = 0 →
      (KPiCcsTranscript.replayRounds base rounds index builder).builder.absorbed =
        0
  | [], _, _, zero => zero
  | round :: rest, index, builder, _ => by
      simp only [KPiCcsTranscript.replayRounds]
      apply replayRounds_builder_absorbed_of_zero base rest (index + 1)
      rfl

@[simp] theorem replay_beforeOutput_absorbed
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) :
    (KPiCcsTranscript.replay input).beforeOutput.absorbed = 0 := by
  unfold KPiCcsTranscript.replay
  apply replayRounds_builder_absorbed_of_zero
  exact derivePreSumcheck_builder_absorbed input

private theorem freshFields_length
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) :
    (((canonicalFinIndices shape.freshCount).flatMap fun source =>
      (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
        KPiCcsTranscript.carriedFields
          (input.freshMatrixImage source matrix))).length =
      shape.freshCount * (shape.matrixCount * 2) := by
  calc
    _ = (canonicalFinIndices shape.freshCount).length *
        (shape.matrixCount * 2) := by
      apply Poseidon2Program.length_flatMap_uniform
      intro source
      calc
        _ = (canonicalFinIndices shape.matrixCount).length * 2 := by
          apply Poseidon2Program.length_flatMap_uniform
          intro matrix
          rfl
        _ = shape.matrixCount * 2 := by
          rw [canonicalFinIndices_length]
    _ = shape.freshCount * (shape.matrixCount * 2) := by
      rw [canonicalFinIndices_length]

private theorem sourceFields_length
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) :
    (((canonicalFinIndices shape.sourceCount).flatMap fun source =>
      KPiCcsTranscript.carriedFields
        (input.sourceAssignment source))).length =
      shape.sourceCount * 2 := by
  calc
    _ = (canonicalFinIndices shape.sourceCount).length * 2 := by
      apply Poseidon2Program.length_flatMap_uniform
      intro source
      rfl
    _ = shape.sourceCount * 2 := by
      rw [canonicalFinIndices_length]

private theorem carriedFields_length
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) :
    (((canonicalCarriedCoordinates shape).flatMap fun coordinate =>
      KPiCcsTranscript.carriedFields
        (input.carriedImage coordinate))).length =
      shape.carriedEvaluationCount * 2 := by
  calc
    _ = (canonicalCarriedCoordinates shape).length * 2 := by
      apply Poseidon2Program.length_flatMap_uniform
      intro coordinate
      rfl
    _ = shape.carriedEvaluationCount * 2 := by
      rw [canonicalCarriedCoordinates_length]

/-- Exact number of field expressions absorbed after PiCCS verification. -/
theorem outputFields_length
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) :
    (KPiCcsTranscript.outputFields input).length =
      1 + shape.freshCount * (shape.matrixCount * 2)
        + shape.sourceCount * 2
        + shape.carriedEvaluationCount * 2 := by
  unfold KPiCcsTranscript.outputFields
  rw [List.length_append, List.length_append, List.length_append,
    freshFields_length input, sourceFields_length input,
    carriedFields_length input]
  rfl

/-- The post-output cursor is determined solely by the prior zero cursor and
the exact typed output serialization length. -/
theorem replay_afterOutput_absorbed
    {shape : Shape} {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) :
    (KPiCcsTranscript.replay input).afterOutput.absorbed =
      SymbolicDuplexCursor.after 0
        (KPiCcsTranscript.outputFields input).length := by
  unfold KPiCcsTranscript.replay
  rw [SymbolicDuplexCursor.absorbMany_absorbed]
  congr
  exact replay_beforeOutput_absorbed input

end Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptCursor
