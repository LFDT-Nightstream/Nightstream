import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerBatchOutputLaw

/-!
Owns scalarwise totalization for the finite comparison experiment only.
The supplied Scalar fallback already has the five-symbol coefficient bounds.
Actual sampling still aborts. Agreement requires actual batch success; no
failed-state, Poseidon2 distribution, inverse-codec, or Fiat-Shamir claim is made.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerTotalizedOutputLaw

open NightstreamFPrime.Spec
open Sampling Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet

/-- Replace one scalar failure only in the comparison decoder. -/
def totalizedFieldDecode (fallback : Scalar) (fields : FieldShortfall.FieldWindow) : Scalar :=
  (SamplerOutputLaw.fieldDecode fields).getD fallback

/-- Totalize each coordinate separately and preserve its source index. -/
def totalizedFieldBatch (fallback : Scalar) {count : Nat}
    (fields : Fin count → FieldShortfall.FieldWindow) : Fin count → Scalar :=
  fun index => totalizedFieldDecode fallback (fields index)

/-- Option-event precomposition retains the previous scalar error bound
while the comparison target is uniform on Scalar itself. -/
theorem scalar_event_error_le (fallback : Scalar) (event : Scalar → Prop) :
    |(Nat.card {fields : FieldShortfall.FieldWindow // event (totalizedFieldDecode fallback fields)} : ℚ) /
          (goldilocksModulus : ℚ) ^ FieldShortfall.fieldLaneCount -
      (Nat.card {scalar : Scalar // event scalar} : ℚ) /
          (alphabetSize : ℚ) ^ coefficientCount| ≤
      32 * FieldPairLaw.pairDeviation +
        (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11 := by
  simpa only [totalizedFieldDecode, Option.getD_some] using
    SamplerOutputLaw.field_output_event_error_le (fun output => event (output.getD fallback))

/-- The selected independent field batch has the derived seventeen-scalar
error. Totalization is coordinatewise, including on inputs with failures. -/
theorem batch_event_error_le (fallback : Scalar)
    (event : (Fin FieldBatchShortfall.batchCount → Scalar) → Prop) :
    |(Nat.card {fields : FieldBatchShortfall.FieldBatch // event (totalizedFieldBatch fallback fields)} : ℚ) /
          ((goldilocksModulus : ℚ) ^ FieldShortfall.fieldLaneCount) ^ FieldBatchShortfall.batchCount -
      (Nat.card {scalars : Fin FieldBatchShortfall.batchCount → Scalar // event scalars} : ℚ) /
          ((alphabetSize : ℚ) ^ coefficientCount) ^ FieldBatchShortfall.batchCount| ≤
      544 * FieldPairLaw.pairDeviation +
        17 * ((Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11) := by
  have scalarCard : Nat.card Scalar = alphabetSize ^ coefficientCount := by
    rw [Nat.card_fun, Nat.card_fin, Nat.card_fin]
  have fieldPositive : (0 : ℚ) < Nat.card FieldShortfall.FieldWindow := by
    rw [FieldShortfall.field_window_cardinality, Nat.cast_pow]
    exact pow_pos (by norm_num [goldilocksModulus]) _
  have scalarPositive : (0 : ℚ) < Nat.card Scalar := by
    rw [scalarCard, Nat.cast_pow]
    exact pow_pos (by norm_num [alphabetSize]) _
  have single (scalarEvent : Scalar → Prop) :
      |(Nat.card {fields : FieldShortfall.FieldWindow //
            scalarEvent (totalizedFieldDecode fallback fields)} : ℚ) / Nat.card FieldShortfall.FieldWindow -
        (Nat.card {scalar : Scalar // scalarEvent (id scalar)} : ℚ) / Nat.card Scalar| ≤
        32 * FieldPairLaw.pairDeviation +
          (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11 := by
    rw [FieldShortfall.field_window_cardinality, scalarCard, Nat.cast_pow, Nat.cast_pow]
    exact scalar_event_error_le fallback scalarEvent
  have compared := BatchOutputLaw.independent_batch_event_error_le
    (totalizedFieldDecode fallback) (id : Scalar → Scalar)
    (32 * FieldPairLaw.pairDeviation +
      (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11)
    fieldPositive scalarPositive single FieldBatchShortfall.batchCount event
  have scalarBatchCard : Nat.card (Fin FieldBatchShortfall.batchCount → Scalar) =
      (alphabetSize ^ coefficientCount) ^ FieldBatchShortfall.batchCount := by
    rw [Nat.card_fun, Nat.card_fin, scalarCard]
  rw [FieldBatchShortfall.field_batch_cardinality, scalarBatchCard] at compared
  have normalized :
      |(Nat.card {fields : FieldBatchShortfall.FieldBatch // event (totalizedFieldBatch fallback fields)} : ℚ) /
            ((goldilocksModulus : ℚ) ^ FieldShortfall.fieldLaneCount) ^ FieldBatchShortfall.batchCount -
        (Nat.card {scalars : Fin FieldBatchShortfall.batchCount → Scalar // event scalars} : ℚ) /
            ((alphabetSize : ℚ) ^ coefficientCount) ^ FieldBatchShortfall.batchCount| ≤
        17 * (32 * FieldPairLaw.pairDeviation +
          (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11) := by
    simpa only [totalizedFieldBatch, id_eq, Nat.cast_pow, FieldBatchShortfall.batchCount_eq] using compared
  calc
    _ ≤ 17 * (32 * FieldPairLaw.pairDeviation +
        (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11) := normalized
    _ = _ := by ring

private theorem mapM_totalized_eq_of_success {Input Output : Type*}
    (decode : Input → Option Output) (fallback : Output) (inputs : List Input)
    (outputs : List Output) (success : inputs.mapM decode = some outputs) :
    inputs.map (fun input => (decode input).getD fallback) = outputs := by
  induction inputs generalizing outputs with
  | nil =>
      have same : ([] : List Output) = outputs := Option.some.inj success
      exact same
  | cons input suffix inductionHypothesis =>
      rw [List.mapM_cons] at success
      generalize firstEq : decode input = first at success
      generalize tailEq : suffix.mapM decode = tail at success
      cases first with
      | none =>
          have impossible : (none : Option (List Output)) = some outputs := success
          exact (Option.some_ne_none outputs impossible.symm).elim
      | some first =>
          cases tail with
          | none =>
              have impossible : (none : Option (List Output)) = some outputs := success
              exact (Option.some_ne_none outputs impossible.symm).elim
          | some tail =>
              have same : first :: tail = outputs := Option.some.inj success
              calc
                _ = first :: tail := by
                  simp only [List.map_cons, firstEq, Option.getD_some,
                    inductionHypothesis tail tailEq]
                _ = outputs := same

private theorem totalized_ofFn (fallback : Scalar) {count : Nat}
    (fields : Fin count → FieldShortfall.FieldWindow) :
    List.ofFn (totalizedFieldBatch fallback fields) =
      (List.ofFn fields).map (fun fields => (SamplerOutputLaw.fieldDecode fields).getD fallback) := by
  simpa only [totalizedFieldBatch, totalizedFieldDecode, Function.comp_def] using
    (List.map_ofFn (f := fields)
      (g := fun fields : FieldShortfall.FieldWindow => (SamplerOutputLaw.fieldDecode fields).getD fallback)).symm

/-- Successful actual batches have exactly the same ordered ring list.
This theorem does not change an abort into protocol acceptance. -/
theorem sampleBatch_success_ring_list_eq (fallback : Scalar) (initial : Transcript.State) (count : Nat)
    (batch : Transcript.PiRlcSampler.Batch count)
    (success : Transcript.PiRlcSampler.sampleBatch initial count = some batch) :
    List.ofFn batch.challenges =
      (List.ofFn (totalizedFieldBatch fallback
        (fun index : Fin count => SamplerFieldShortfall.fieldWindow initial index.val))).map
          Phi81StrongSet.embedScalar := by
  let fields : Fin count → FieldShortfall.FieldWindow :=
    fun index => SamplerFieldShortfall.fieldWindow initial index.val
  have agreement := SamplerBatchOutputLaw.sampleBatch_ring_list_eq_fieldDecodeBatch initial count
  rw [success, Option.map_some] at agreement
  have collected : (SamplerBatchOutputLaw.fieldDecodeBatch fields).map (List.map Phi81StrongSet.embedScalar) =
      some (List.ofFn batch.challenges) := agreement.symm
  obtain ⟨scalars, sampled, encoded⟩ := Option.map_eq_some_iff.mp collected
  have totalized : List.ofFn (totalizedFieldBatch fallback fields) = scalars :=
    (totalized_ofFn fallback fields).trans
      (mapM_totalized_eq_of_success SamplerOutputLaw.fieldDecode fallback
        (List.ofFn fields) scalars sampled)
  exact encoded.symm.trans (congrArg (List.map Phi81StrongSet.embedScalar) totalized.symm)

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerTotalizedOutputLaw
