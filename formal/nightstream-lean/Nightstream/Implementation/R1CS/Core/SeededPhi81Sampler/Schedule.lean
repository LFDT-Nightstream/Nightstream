import Nightstream.Implementation.R1CS.Core.SeededPhi81Sampler.Vector

/-!
Unbounded semantics for seeded-Phi81 vector, chunk, and output traversal.

Assurance tier: executable-semantics refinement. The relations in this file
compose the one-vector `SampleVector` meaning without mentioning Rust or R1CS.
Successful bounded traversal is proved to satisfy the corresponding unbounded
relations at every nesting level.

Owns: vector sequencing within one seed; seed/chunk traversal for one output;
output traversal for a complete schedule; and canonicality propagation.

Does not own: one-vector rejection semantics; ChaCha8; Rust `rand_chacha`;
seed derivation; Phi81 rotation; SIS security; R1CS rows; Poseidon2;
transcript authority; row removal; or cost totals.

Emits constraints: no.

Authority boundary: every relation retains the explicit `stream`, seed,
message-column count, and chunk size. Production correspondence must first
identify the stream and seeds with verifier-owned semantics.

| Protocol | Phase | Mathematical branch | Definition/theorem | Exact guarantee |
|---|---|---|---|---|
| seeded SIS | coefficient sampling | vectors in one chunk | `SamplesVectors` | sequential replacement cursors connect every sampled vector |
| seeded SIS | coefficient sampling | chunks in one output | `SamplesOutput` | chunk sizes and seed order match `chunkMessageCount` |
| seeded SIS | coefficient sampling | all outputs | `SamplesSchedule` | output order and grouping match the schedule |
| seeded SIS | coefficient sampling | executable refinement | `Schedule.baseRotations_sound` | every successful complete bounded schedule satisfies all three relations |
| seeded SIS | coefficient sampling | canonicality | `SamplesSchedule.vectors_canonical` | every produced coefficient is a canonical Goldilocks word |
-/

namespace Nightstream.Implementation.R1CS.SeededPhi81Sampler

/-- Unbounded sequential meaning of `count` vectors sampled from one seed. -/
inductive SamplesVectors (stream : WordStream) (seed : List Nat) :
    Nat -> Nat -> List (List Nat) -> Prop
  | nil (wordPosition : Nat) :
      SamplesVectors stream seed 0 wordPosition []
  | cons (count wordPosition nextPosition : Nat)
      (vector : List Nat) (tail : List (List Nat))
      (headSample : SampleVector stream seed wordPosition vector nextPosition)
      (tailSamples : SamplesVectors stream seed count nextPosition tail) :
      SamplesVectors stream seed (count + 1) wordPosition (vector :: tail)

private theorem sampleVectors_go_sound
    {stream : WordStream} {seed : List Nat} {fuel count wordPosition : Nat}
    {reversed result : List (List Nat)}
    (success : sampleVectors.go stream seed fuel count wordPosition reversed =
      some result) :
    exists sampled,
      SamplesVectors stream seed count wordPosition sampled /\
      result = reversed.reverse ++ sampled := by
  induction count generalizing wordPosition reversed result with
  | zero =>
      have resultEq : reversed.reverse = result :=
        Option.some.inj (by simpa [sampleVectors.go] using success)
      exact ⟨[], .nil wordPosition, by simpa using resultEq.symm⟩
  | succ count ih =>
      cases headEq : sampleVector stream seed fuel wordPosition with
      | none => simp [sampleVectors.go, headEq] at success
      | some headResult =>
          rcases headResult with ⟨vector, nextPosition⟩
          have tailSuccess :
              sampleVectors.go stream seed fuel count nextPosition
                  (vector :: reversed) = some result := by
            simpa [sampleVectors.go, headEq] using success
          rcases ih tailSuccess with ⟨tail, tailSamples, resultEq⟩
          refine ⟨vector :: tail,
            .cons count wordPosition nextPosition vector tail
              (sampleVector_sound headEq) tailSamples, ?_⟩
          simpa [List.reverse_cons, List.append_assoc] using resultEq

theorem sampleVectors_sound
    {stream : WordStream} {seed : List Nat} {fuel count wordPosition : Nat}
    {vectors : List (List Nat)}
    (success : sampleVectors stream seed fuel count wordPosition =
      some vectors) :
    SamplesVectors stream seed count wordPosition vectors := by
  rcases sampleVectors_go_sound (by simpa [sampleVectors] using success) with
    ⟨sampled, samples, vectorsEq⟩
  have equal : vectors = sampled := by simpa using vectorsEq
  simpa [equal] using samples

theorem SamplesVectors.length
    {stream : WordStream} {seed : List Nat} {count wordPosition : Nat}
    {vectors : List (List Nat)}
    (samples : SamplesVectors stream seed count wordPosition vectors) :
    vectors.length = count := by
  induction samples with
  | nil => rfl
  | cons _ _ _ _ _ _ _ ih => simp [ih]

theorem SampleVector.values_canonical
    {stream : WordStream} {seed : List Nat}
    {wordPosition : Nat} {values : List Nat} {finalPosition : Nat}
    (sampled : SampleVector stream seed wordPosition values finalPosition) :
    forall value, value ∈ values -> value < modulus :=
  Repairs.values_canonical sampled

theorem SamplesVectors.vectors_canonical
    {stream : WordStream} {seed : List Nat} {count wordPosition : Nat}
    {vectors : List (List Nat)}
    (samples : SamplesVectors stream seed count wordPosition vectors) :
    forall vector, vector ∈ vectors ->
      forall value, value ∈ vector -> value < modulus := by
  induction samples with
  | nil => simp
  | cons _ _ _ vector _ headSample _ ih =>
      intro candidate membership
      simp only [List.mem_cons] at membership
      rcases membership with rfl | membership
      · exact headSample.values_canonical
      · exact ih candidate membership

/-- Unbounded seed/chunk traversal for one SIS output. -/
inductive SamplesOutput (stream : WordStream) (messageCols chunkSize : Nat) :
    Nat -> List (List Nat) -> List (List Nat) -> Prop
  | nil (chunkIndex : Nat) :
      SamplesOutput stream messageCols chunkSize chunkIndex [] []
  | cons (chunkIndex : Nat) (seed : List Nat) (seeds : List (List Nat))
      (vectors rest : List (List Nat))
      (chunkSamples : SamplesVectors stream seed
        (chunkMessageCount messageCols chunkSize chunkIndex) 0 vectors)
      (tailSamples : SamplesOutput stream messageCols chunkSize
        (chunkIndex + 1) seeds rest) :
      SamplesOutput stream messageCols chunkSize chunkIndex (seed :: seeds)
        (vectors ++ rest)

theorem sampleOutput_sound
    {stream : WordStream} {messageCols chunkSize fuel chunkIndex : Nat}
    {seeds : List (List Nat)} {vectors : List (List Nat)}
    (success : sampleOutput stream messageCols chunkSize fuel chunkIndex seeds =
      some vectors) :
    SamplesOutput stream messageCols chunkSize chunkIndex seeds vectors := by
  induction seeds generalizing chunkIndex vectors with
  | nil =>
      have pairEq : [] = vectors :=
        Option.some.inj (by simpa [sampleOutput] using success)
      subst vectors
      exact .nil chunkIndex
  | cons seed seeds ih =>
      cases chunkEq : sampleVectors stream seed fuel
          (chunkMessageCount messageCols chunkSize chunkIndex) 0 with
      | none => simp [sampleOutput, chunkEq] at success
      | some chunkVectors =>
          cases tailEq : sampleOutput stream messageCols chunkSize fuel
              (chunkIndex + 1) seeds with
          | none => simp [sampleOutput, chunkEq, tailEq] at success
          | some rest =>
              have vectorsEq : chunkVectors ++ rest = vectors :=
                Option.some.inj (by
                  simpa [sampleOutput, chunkEq, tailEq] using success)
              subst vectors
              exact .cons chunkIndex seed seeds chunkVectors rest
                (sampleVectors_sound chunkEq) (ih tailEq)

theorem SamplesOutput.vectors_canonical
    {stream : WordStream} {messageCols chunkSize chunkIndex : Nat}
    {seeds : List (List Nat)} {vectors : List (List Nat)}
    (samples : SamplesOutput stream messageCols chunkSize chunkIndex seeds vectors) :
    forall vector, vector ∈ vectors ->
      forall value, value ∈ vector -> value < modulus := by
  induction samples with
  | nil => simp
  | cons _ _ _ chunkVectors rest chunkSamples _ ih =>
      intro vector membership
      rcases List.mem_append.mp membership with membership | membership
      · exact chunkSamples.vectors_canonical vector membership
      · exact ih vector membership

/-- Unbounded traversal of every output group in a complete seed schedule. -/
inductive SamplesSchedule (stream : WordStream) (messageCols chunkSize : Nat) :
    List (List (List Nat)) -> List (List (List Nat)) -> Prop
  | nil : SamplesSchedule stream messageCols chunkSize [] []
  | cons (seeds : List (List Nat)) (seedTail : List (List (List Nat)))
      (vectors : List (List Nat)) (outputTail : List (List (List Nat)))
      (outputSamples : SamplesOutput stream messageCols chunkSize 0 seeds vectors)
      (tailSamples : SamplesSchedule stream messageCols chunkSize
        seedTail outputTail) :
      SamplesSchedule stream messageCols chunkSize (seeds :: seedTail)
        (vectors :: outputTail)

theorem sampleScheduleOutputs_sound
    {stream : WordStream} {messageCols chunkSize fuel : Nat}
    {seedsByOutput : List (List (List Nat))}
    {outputs : List (List (List Nat))}
    (success : sampleScheduleOutputs stream messageCols chunkSize fuel
      seedsByOutput = some outputs) :
    SamplesSchedule stream messageCols chunkSize seedsByOutput outputs := by
  induction seedsByOutput generalizing outputs with
  | nil =>
      have outputsEq : [] = outputs :=
        Option.some.inj (by simpa [sampleScheduleOutputs] using success)
      subst outputs
      exact .nil
  | cons seeds seedTail ih =>
      cases outputEq : sampleOutput stream messageCols chunkSize fuel 0 seeds with
      | none => simp [sampleScheduleOutputs, outputEq] at success
      | some vectors =>
          cases tailEq : sampleScheduleOutputs stream messageCols chunkSize fuel
              seedTail with
          | none => simp [sampleScheduleOutputs, outputEq, tailEq] at success
          | some outputTail =>
              have outputsEq : vectors :: outputTail = outputs :=
                Option.some.inj (by
                  simpa [sampleScheduleOutputs, outputEq, tailEq] using success)
              subst outputs
              exact .cons seeds seedTail vectors outputTail
                (sampleOutput_sound outputEq) (ih tailEq)

theorem Schedule.baseRotations_sound
    {schedule : Schedule} {stream : WordStream} {messageCols : Nat}
    {outputs : List (List (List Nat))}
    (success : schedule.baseRotations stream messageCols = some outputs) :
    SamplesSchedule stream messageCols schedule.chunkSize
      schedule.seedsByOutput outputs :=
  sampleScheduleOutputs_sound (by
    simpa [Schedule.baseRotations] using success)

theorem SamplesSchedule.vectors_canonical
    {stream : WordStream} {messageCols chunkSize : Nat}
    {seedsByOutput : List (List (List Nat))}
    {outputs : List (List (List Nat))}
    (samples : SamplesSchedule stream messageCols chunkSize
      seedsByOutput outputs) :
    forall output, output ∈ outputs ->
      forall vector, vector ∈ output ->
        forall value, value ∈ vector -> value < modulus := by
  induction samples with
  | nil => simp
  | cons _ _ vectors _ outputSamples _ ih =>
      intro output membership
      simp only [List.mem_cons] at membership
      rcases membership with rfl | membership
      · exact outputSamples.vectors_canonical
      · exact ih output membership

end Nightstream.Implementation.R1CS.SeededPhi81Sampler
