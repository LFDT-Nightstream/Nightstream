import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerOutputLaw
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.BatchOutputLaw

/-!
Owns ordered Option collection of the existing scalar decoder, its finite
17-scalar output comparison, and the actual batch output/state equality.
The finite experiment retains every abort. It bounds one fixed batch;
it assigns no input law to Poseidon2 and no query, retry, or extractor work.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerBatchOutputLaw

open NightstreamFPrime.Spec
open Sampling Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet

/-- Collect the existing scalar decoder in increasing source order. -/
def fieldDecodeBatch {count : Nat} (fields : Fin count → FieldShortfall.FieldWindow) :
    Option (List Scalar) :=
  (List.ofFn fields).mapM SamplerOutputLaw.fieldDecode

private theorem collect_ofFn_map {Input Output : Type*} (decode : Input → Option Output)
    (count : Nat) (inputs : Fin count → Input) :
    (List.ofFn (fun index => decode (inputs index))).mapM id = (List.ofFn inputs).mapM decode := by
  simpa only [List.map_ofFn, Function.comp_def, id_eq] using
    (List.mapM_map (m := Option) (f := decode) (g := id) (l := List.ofFn inputs))

private theorem collect_ofFn_some {Output : Type*} (count : Nat) (values : Fin count → Output) :
    (List.ofFn (fun index => some (values index))).mapM id = some (List.ofFn values) := by
  rw [collect_ofFn_map (some : Output → Option Output) count values]
  simpa only [List.map_id] using
    (List.mapM_pure (m := Option) (l := List.ofFn values) (f := id))

/-- The independent field-window experiment is compared with uniform
successful ordered scalar lists. Its `none` mass is retained; the target
gives `none` mass zero. The multiplier is the selected arity, seventeen. -/
theorem field_batch_output_event_error_le (event : Option (List Scalar) → Prop) :
    |(Nat.card {fields : FieldBatchShortfall.FieldBatch // event (fieldDecodeBatch fields)} : ℚ) /
          ((goldilocksModulus : ℚ) ^ FieldShortfall.fieldLaneCount) ^ FieldBatchShortfall.batchCount -
      (Nat.card {scalars : Fin FieldBatchShortfall.batchCount → Scalar //
          event (some (List.ofFn scalars))} : ℚ) /
          ((alphabetSize : ℚ) ^ coefficientCount) ^ FieldBatchShortfall.batchCount| ≤
      17 * (32 * FieldPairLaw.pairDeviation +
        (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11) := by
  have scalarCard : Nat.card Scalar = alphabetSize ^ coefficientCount := by
    rw [Nat.card_fun, Nat.card_fin, Nat.card_fin]
  have fieldPositive : (0 : ℚ) < Nat.card FieldShortfall.FieldWindow := by
    rw [FieldShortfall.field_window_cardinality, Nat.cast_pow]
    exact pow_pos (by norm_num [goldilocksModulus]) _
  have scalarPositive : (0 : ℚ) < Nat.card Scalar := by
    rw [scalarCard, Nat.cast_pow]
    exact pow_pos (by norm_num [alphabetSize]) _
  have single (scalarEvent : Option Scalar → Prop) :
      |(Nat.card {fields : FieldShortfall.FieldWindow //
            scalarEvent (SamplerOutputLaw.fieldDecode fields)} : ℚ) / Nat.card FieldShortfall.FieldWindow -
        (Nat.card {scalar : Scalar // scalarEvent (some scalar)} : ℚ) / Nat.card Scalar| ≤
        32 * FieldPairLaw.pairDeviation +
          (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11 := by
    rw [FieldShortfall.field_window_cardinality, scalarCard, Nat.cast_pow, Nat.cast_pow]
    exact SamplerOutputLaw.field_output_event_error_le scalarEvent
  let collect (values : Fin FieldBatchShortfall.batchCount → Option Scalar) : Option (List Scalar) :=
    (List.ofFn values).mapM id
  have compared := BatchOutputLaw.independent_batch_event_error_le
    SamplerOutputLaw.fieldDecode (some : Scalar → Option Scalar)
    (32 * FieldPairLaw.pairDeviation +
      (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11)
    fieldPositive scalarPositive single FieldBatchShortfall.batchCount
    (fun values => event (collect values))
  have fieldCount :
      Nat.card {fields : FieldBatchShortfall.FieldBatch //
        event (collect (fun index => SamplerOutputLaw.fieldDecode (fields index)))} =
      Nat.card {fields : FieldBatchShortfall.FieldBatch // event (fieldDecodeBatch fields)} :=
    Nat.card_congr (Equiv.subtypeEquivRight (fun fields => iff_of_eq
      (congrArg event (collect_ofFn_map SamplerOutputLaw.fieldDecode
        FieldBatchShortfall.batchCount fields))))
  have scalarCount :
      Nat.card {scalars : Fin FieldBatchShortfall.batchCount → Scalar //
        event (collect (fun index => some (scalars index)))} =
      Nat.card {scalars : Fin FieldBatchShortfall.batchCount → Scalar //
        event (some (List.ofFn scalars))} :=
    Nat.card_congr (Equiv.subtypeEquivRight (fun scalars => iff_of_eq
      (congrArg event (collect_ofFn_some FieldBatchShortfall.batchCount scalars))))
  have scalarBatchCard : Nat.card (Fin FieldBatchShortfall.batchCount → Scalar) =
      (alphabetSize ^ coefficientCount) ^ FieldBatchShortfall.batchCount := by
    rw [Nat.card_fun, Nat.card_fin, scalarCard]
  rw [fieldCount, scalarCount, FieldBatchShortfall.field_batch_cardinality, scalarBatchCard] at compared
  simpa only [Nat.cast_pow, FieldBatchShortfall.batchCount_eq] using compared

private theorem mapM_ofFn_last {Input Output : Type*} (decode : Input → Option Output)
    (count : Nat) (inputs : Fin (count + 1) → Input) :
    (List.ofFn inputs).mapM decode =
      ((List.ofFn (fun index : Fin count => inputs index.castSucc)).mapM decode).bind
        (fun prior => (decode (inputs (Fin.last count))).map (fun last => prior ++ [last])) := by
  rw [List.ofFn_succ_last, List.mapM_append, List.mapM_cons, List.mapM_nil]
  generalize (List.ofFn (fun index : Fin count => inputs index.castSucc)).mapM decode = prior
  generalize decode (inputs (Fin.last count)) = last
  cases prior <;> cases last <;> rfl

private theorem mapM_output_map {Input Output Mapped : Type*}
    (decode : Input → Option Output) (encode : Output → Mapped) (inputs : List Input) :
    inputs.mapM (fun input => (decode input).map encode) =
      (inputs.mapM decode).map (List.map encode) := by
  induction inputs with
  | nil => rfl
  | cons input suffix inductionHypothesis =>
      rw [List.mapM_cons, List.mapM_cons, inductionHypothesis]
      generalize decode input = head
      generalize suffix.mapM decode = tail
      cases head <;> cases tail <;> rfl

private theorem list_collect_of_recurrence {Output : Type*}
    (sample : Nat → Option Output) (collected : Nat → Option (List Output))
    (empty : collected 0 = some [])
    (step : ∀ count, collected (count + 1) =
      (collected count).bind
        (fun prior => (sample count).map (fun last => prior ++ [last])))
    (count : Nat) :
    collected count = (List.ofFn (fun index : Fin count => sample index.val)).mapM id := by
  induction count with
  | zero => rw [empty, List.ofFn_zero, List.mapM_nil] <;> rfl
  | succ count inductionHypothesis =>
      rw [step, mapM_ofFn_last]
      simpa only [Fin.val_castSucc, Fin.val_last, id_eq] using
        congrArg (fun prior : Option (List Output) =>
          prior.bind (fun values => (sample count).map (fun last => values ++ [last])))
          inductionHypothesis

private theorem sampleBatch_list_step (initial : Transcript.State) (count : Nat) :
    (Transcript.PiRlcSampler.sampleBatch initial (count + 1)).map
        (fun batch => List.ofFn batch.challenges) =
      ((Transcript.PiRlcSampler.sampleBatch initial count).map
        (fun batch => List.ofFn batch.challenges)).bind
          (fun prior => (Transcript.PiRlcSampler.sampleRingChallenge initial count).map
            (fun last => prior ++ [last])) := by
  rw [Transcript.PiRlcSampler.sampleBatch]
  generalize Transcript.PiRlcSampler.sampleBatch initial count = prior
  generalize Transcript.PiRlcSampler.sampleRingChallenge initial count = sampled
  generalize stateAt Transcript.PiRlcSampler.specification initial (count + 1) = finalState
  cases prior <;> cases sampled <;>
    simp only [Option.map_none, Option.map_some, Option.bind_none, Option.bind_some,
      List.ofFn_succ_last, Fin.lastCases_castSucc, Fin.lastCases_last]

private theorem sampleBatch_list_eq_mapM (initial : Transcript.State) (count : Nat) :
    (Transcript.PiRlcSampler.sampleBatch initial count).map
        (fun batch => List.ofFn batch.challenges) =
      (List.ofFn (fun index : Fin count =>
        Transcript.PiRlcSampler.sampleRingChallenge initial index.val)).mapM id := by
  exact list_collect_of_recurrence (Transcript.PiRlcSampler.sampleRingChallenge initial)
    (fun size => (Transcript.PiRlcSampler.sampleBatch initial size).map
      (fun batch => List.ofFn batch.challenges))
    (by
      change (Transcript.PiRlcSampler.sampleBatch initial 0).map
        (fun batch => List.ofFn batch.challenges) = some []
      rw [Transcript.PiRlcSampler.sampleBatch.eq_def, Option.map_some, List.ofFn_zero])
    (sampleBatch_list_step initial) count

/-- The actual batch's ring list is the ordered field-window decoder on
its own deterministic source. No independent-field premise occurs here. -/
theorem sampleBatch_ring_list_eq_fieldDecodeBatch (initial : Transcript.State) (count : Nat) :
    (Transcript.PiRlcSampler.sampleBatch initial count).map
        (fun batch => List.ofFn batch.challenges) =
      (fieldDecodeBatch (fun index : Fin count =>
        SamplerFieldShortfall.fieldWindow initial index.val)).map (List.map Phi81StrongSet.embedScalar) := by
  rw [sampleBatch_list_eq_mapM]
  have pointwise :
      (fun index : Fin count => Transcript.PiRlcSampler.sampleRingChallenge initial index.val) =
        (fun index : Fin count =>
          (SamplerOutputLaw.fieldDecode
            (SamplerFieldShortfall.fieldWindow initial index.val)).map Phi81StrongSet.embedScalar) := by
    funext index
    rw [Transcript.PiRlcSampler.sampleRingChallenge, SamplerOutputLaw.sampleScalar_eq_fieldDecode]
  rw [pointwise, collect_ofFn_map
    (fun fields : FieldShortfall.FieldWindow =>
      (SamplerOutputLaw.fieldDecode fields).map Phi81StrongSet.embedScalar)
    count (fun index : Fin count => SamplerFieldShortfall.fieldWindow initial index.val)]
  exact mapM_output_map SamplerOutputLaw.fieldDecode Phi81StrongSet.embedScalar _

private theorem option_map_pair_fixed {Input Value State : Type*}
    (input : Option Input) (value : Input → Value) (state : Input → State) (finalState : State)
    (fixed : ∀ output, input = some output → state output = finalState) :
    input.map (fun output => (value output, state output)) =
      (input.map value).map (fun output => (output, finalState)) := by
  cases input with
  | none => rfl
  | some output => simp only [Option.map_some, fixed output rfl]

/-- Successful output preserves both source order and the exact state
after every fixed digest block. Failure remains `none` on both sides. -/
theorem sampleBatch_output_state_eq_fieldDecodeBatch (initial : Transcript.State) (count : Nat) :
    (Transcript.PiRlcSampler.sampleBatch initial count).map
        (fun batch => (List.ofFn batch.challenges, batch.finalState)) =
      (fieldDecodeBatch (fun index : Fin count =>
        SamplerFieldShortfall.fieldWindow initial index.val)).map
          (fun scalars => (scalars.map Phi81StrongSet.embedScalar,
            stateAt Transcript.PiRlcSampler.specification initial count)) := by
  calc
    _ = ((Transcript.PiRlcSampler.sampleBatch initial count).map
          (fun batch => List.ofFn batch.challenges)).map
            (fun values => (values, stateAt Transcript.PiRlcSampler.specification initial count)) :=
      option_map_pair_fixed (Transcript.PiRlcSampler.sampleBatch initial count)
        (fun batch => List.ofFn batch.challenges) (fun batch => batch.finalState)
        (stateAt Transcript.PiRlcSampler.specification initial count)
        (fun _ success => Transcript.PiRlcSampler.piRlcChallengesWithState_finalState success)
    _ = ((fieldDecodeBatch (fun index : Fin count =>
          SamplerFieldShortfall.fieldWindow initial index.val)).map
            (List.map Phi81StrongSet.embedScalar)).map
              (fun values => (values, stateAt Transcript.PiRlcSampler.specification initial count)) :=
      congrArg (fun output : Option (List RingF) =>
        output.map (fun values => (values, stateAt Transcript.PiRlcSampler.specification initial count)))
        (sampleBatch_ring_list_eq_fieldDecodeBatch initial count)
    _ = _ := by simp only [Option.map_map, Function.comp_def]

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerBatchOutputLaw
