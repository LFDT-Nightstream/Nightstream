import Nightstream.Implementation.R1CS.Core.SeededPhi81Sampler

/-!
Unbounded semantics and executable refinement for one seeded-Phi81 vector.

Assurance tier: executable-semantics refinement. `Repairs` is an inductive
mathematical relation: canonical initial words are retained and every rejected
word is replaced by the first later canonical stream word. The bounded
implementation is proved sound and complete with respect to that relation;
fuel therefore affects termination only, never the accepted vector meaning.

Owns: one-vector repair semantics; bounded/unbounded soundness and
completeness; output canonicality; and vector cursor meaning.

Does not own: ChaCha8; Rust `rand_chacha`; seed derivation; multi-vector,
chunk, or output traversal; Phi81 rotation; SIS security; R1CS rows;
Poseidon2; transcript authority; row removal; or cost totals.

Emits constraints: no.

Authority boundary: `stream` and `seed` are explicit verifier inputs to this
semantic layer. A later refinement identifies them with verifier-owned pure
ChaCha8 and then with the production implementation.

| Protocol | Phase | Mathematical branch | Definition/theorem | Exact guarantee |
|---|---|---|---|---|
| seeded SIS | coefficient sampling | rejected-word repair | `Repairs` | retain canonical words and replace each rejection by the first later canonical stream word |
| seeded SIS | coefficient sampling | bounded soundness | `repairRejected_sound` | every successful bounded repair satisfies `Repairs` |
| seeded SIS | coefficient sampling | bounded completeness | `Repairs.exists_fuel` | every finite `Repairs` derivation is produced with some fuel |
| seeded SIS | coefficient sampling | canonical output | `Repairs.values_canonical` | every repaired coefficient is below the Goldilocks modulus |
| seeded SIS | coefficient sampling | vector cursor | `SampleVector` | the initial 54 words and replacement cursor have an unbounded meaning |
-/

namespace Nightstream.Implementation.R1CS.SeededPhi81Sampler

/-- Unbounded semantics for repairing a finite initial coefficient vector.
The replacement cursor advances only when an initial word is rejected. -/
inductive Repairs (stream : WordStream) (seed : List Nat) :
    List Nat -> Nat -> List Nat -> Nat -> Prop
  | nil (wordPosition : Nat) :
      Repairs stream seed [] wordPosition [] wordPosition
  | keep (candidate : Nat) (tail values : List Nat)
      (wordPosition finalPosition : Nat)
      (accepted : candidate < modulus)
      (rest : Repairs stream seed tail wordPosition values finalPosition) :
      Repairs stream seed (candidate :: tail) wordPosition
        (candidate :: values) finalPosition
  | replace (candidate : Nat) (tail : List Nat)
      (wordPosition value nextPosition : Nat) (values : List Nat)
      (finalPosition : Nat)
      (rejected : modulus <= candidate)
      (replacement : FirstAccepted stream seed wordPosition value nextPosition)
      (rest : Repairs stream seed tail nextPosition values finalPosition) :
      Repairs stream seed (candidate :: tail) wordPosition
        (value :: values) finalPosition

theorem FirstAccepted.value_lt_modulus
    {stream : WordStream} {seed : List Nat}
    {wordPosition value nextPosition : Nat}
    (accepted : FirstAccepted stream seed wordPosition value nextPosition) :
    value < modulus := by
  induction accepted with
  | here _ accepted => exact accepted
  | later _ _ _ _ _ ih => exact ih

theorem nextAccepted_add_fuel
    {stream : WordStream} {seed : List Nat}
    {wordPosition fuel value nextPosition : Nat}
    (success : nextAccepted stream seed wordPosition fuel =
      some (value, nextPosition))
    (extra : Nat) :
    nextAccepted stream seed wordPosition (fuel + extra) =
      some (value, nextPosition) := by
  induction fuel generalizing wordPosition with
  | zero => simp [nextAccepted] at success
  | succ fuel ih =>
      by_cases accepted : candidateAt stream seed wordPosition < modulus
      · rw [Nat.succ_add]
        simpa [nextAccepted, accepted] using success
      · simp only [Nat.succ_add, nextAccepted, accepted, ↓reduceIte] at success ⊢
        exact ih success

theorem nextAccepted_mono
    {stream : WordStream} {seed : List Nat}
    {wordPosition fuel largerFuel value nextPosition : Nat}
    (success : nextAccepted stream seed wordPosition fuel =
      some (value, nextPosition))
    (fuelLe : fuel <= largerFuel) :
    nextAccepted stream seed wordPosition largerFuel =
      some (value, nextPosition) := by
  obtain ⟨extra, rfl⟩ := Nat.exists_eq_add_of_le fuelLe
  exact nextAccepted_add_fuel success extra

theorem repairRejected_sound
    {stream : WordStream} {seed : List Nat}
    {candidates : List Nat} {wordPosition fuel : Nat}
    {values : List Nat} {finalPosition : Nat}
    (success : repairRejected stream seed fuel candidates wordPosition =
      some (values, finalPosition)) :
    Repairs stream seed candidates wordPosition values finalPosition := by
  induction candidates generalizing wordPosition values with
  | nil =>
      have pairEq : ([], wordPosition) = (values, finalPosition) :=
        Option.some.inj (by simpa [repairRejected] using success)
      have valuesEq : [] = values := congrArg Prod.fst pairEq
      have finalEq : wordPosition = finalPosition := congrArg Prod.snd pairEq
      subst values
      subst finalPosition
      exact .nil wordPosition
  | cons candidate tail ih =>
      by_cases accepted : candidate < modulus
      · cases tailEq : repairRejected stream seed fuel tail wordPosition with
        | none => simp [repairRejected, accepted, tailEq] at success
        | some result =>
            rcases result with ⟨repaired, final⟩
            have pairEq : (candidate :: repaired, final) =
                (values, finalPosition) :=
              Option.some.inj (by
                simpa [repairRejected, accepted, tailEq] using success)
            have valuesEq : candidate :: repaired = values :=
              congrArg Prod.fst pairEq
            have finalEq : final = finalPosition := congrArg Prod.snd pairEq
            subst values
            subst finalPosition
            exact .keep candidate tail repaired wordPosition final accepted
              (ih tailEq)
      · cases replacementEq : nextAccepted stream seed wordPosition fuel with
        | none => simp [repairRejected, accepted, replacementEq] at success
        | some replacementResult =>
            rcases replacementResult with ⟨value, nextPosition⟩
            cases tailEq : repairRejected stream seed fuel tail nextPosition with
            | none =>
                simp [repairRejected, accepted, replacementEq, tailEq] at success
            | some result =>
                rcases result with ⟨repaired, final⟩
                have pairEq : (value :: repaired, final) =
                    (values, finalPosition) :=
                  Option.some.inj (by
                    simpa [repairRejected, accepted, replacementEq, tailEq]
                      using success)
                have valuesEq : value :: repaired = values :=
                  congrArg Prod.fst pairEq
                have finalEq : final = finalPosition := congrArg Prod.snd pairEq
                subst values
                subst finalPosition
                exact .replace candidate tail wordPosition value
                  nextPosition repaired final (Nat.le_of_not_gt accepted)
                  (nextAccepted_sound replacementEq) (ih tailEq)

theorem Repairs.exists_fuel_ge
    {stream : WordStream} {seed : List Nat}
    {candidates : List Nat} {wordPosition : Nat}
    {values : List Nat} {finalPosition : Nat}
    (repairs : Repairs stream seed candidates wordPosition values finalPosition)
    (minimumFuel : Nat) :
    exists fuel,
      minimumFuel <= fuel /\
      repairRejected stream seed fuel candidates wordPosition =
        some (values, finalPosition) := by
  induction repairs generalizing minimumFuel with
  | nil wordPosition =>
      exact ⟨minimumFuel, Nat.le_refl _, by simp [repairRejected]⟩
  | keep candidate tail values wordPosition finalPosition accepted _ ih =>
      rcases ih minimumFuel with ⟨fuel, fuelGe, success⟩
      exact ⟨fuel, fuelGe, by simp [repairRejected, accepted, success]⟩
  | replace candidate tail wordPosition value nextPosition values finalPosition
      rejected replacement _ tailIH =>
      rcases FirstAccepted.exists_fuel replacement with
        ⟨replacementFuel, replacementSuccess⟩
      rcases tailIH (Nat.max minimumFuel replacementFuel) with
        ⟨fuel, fuelGeMax, tailSuccess⟩
      have minimumLe : minimumFuel <= fuel :=
        Nat.le_trans (Nat.le_max_left _ _) fuelGeMax
      have replacementLe : replacementFuel <= fuel :=
        Nat.le_trans (Nat.le_max_right _ _) fuelGeMax
      have replacementSuccess' :=
        nextAccepted_mono replacementSuccess replacementLe
      exact ⟨fuel, minimumLe, by
        simp [repairRejected, Nat.not_lt.mpr rejected,
          replacementSuccess', tailSuccess]⟩

theorem Repairs.exists_fuel
    {stream : WordStream} {seed : List Nat}
    {candidates : List Nat} {wordPosition : Nat}
    {values : List Nat} {finalPosition : Nat}
    (repairs : Repairs stream seed candidates wordPosition values finalPosition) :
    exists fuel,
      repairRejected stream seed fuel candidates wordPosition =
        some (values, finalPosition) := by
  rcases repairs.exists_fuel_ge 0 with ⟨fuel, _, success⟩
  exact ⟨fuel, success⟩

theorem Repairs.values_canonical
    {stream : WordStream} {seed : List Nat}
    {candidates : List Nat} {wordPosition : Nat}
    {values : List Nat} {finalPosition : Nat}
    (repairs : Repairs stream seed candidates wordPosition values finalPosition) :
    forall value, value ∈ values -> value < modulus := by
  induction repairs with
  | nil => simp
  | keep candidate _ _ _ _ accepted _ ih =>
      intro value membership
      simp only [List.mem_cons] at membership
      rcases membership with rfl | membership
      · exact accepted
      · exact ih value membership
  | replace _ _ _ value _ _ _ _ replacement _ ih =>
      intro candidate membership
      simp only [List.mem_cons] at membership
      rcases membership with rfl | membership
      · exact replacement.value_lt_modulus
      · exact ih candidate membership

/-- Unbounded meaning of one sampled vector. The first `dimension` words are
read in one slice; replacement words start immediately after that slice. -/
def SampleVector (stream : WordStream) (seed : List Nat)
    (wordPosition : Nat) (values : List Nat) (finalPosition : Nat) : Prop :=
  Repairs stream seed (stream seed wordPosition dimension)
    (wordPosition + 2 * dimension) values finalPosition

theorem sampleVector_sound
    {stream : WordStream} {seed : List Nat}
    {fuel wordPosition : Nat} {values : List Nat} {finalPosition : Nat}
    (success : sampleVector stream seed fuel wordPosition =
      some (values, finalPosition)) :
    SampleVector stream seed wordPosition values finalPosition := by
  exact repairRejected_sound (by simpa [sampleVector] using success)

theorem SampleVector.exists_fuel
    {stream : WordStream} {seed : List Nat}
    {wordPosition : Nat} {values : List Nat} {finalPosition : Nat}
    (sampled : SampleVector stream seed wordPosition values finalPosition) :
    exists fuel,
      sampleVector stream seed fuel wordPosition =
        some (values, finalPosition) := by
  have repairs : Repairs stream seed (stream seed wordPosition dimension)
      (wordPosition + 2 * dimension) values finalPosition := sampled
  rcases Repairs.exists_fuel repairs with ⟨fuel, success⟩
  exact ⟨fuel, by simpa [sampleVector] using success⟩

end Nightstream.Implementation.R1CS.SeededPhi81Sampler
