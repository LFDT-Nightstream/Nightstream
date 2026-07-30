import Nightstream.Implementation.Lowering.Goldilocks.Codec

/-!
Contract: prove when every exact-width field-coordinate list has one semantic
preimage for a codec.

Owns: closure of exact-width recovery under products, finite functions,
fixed lists, fixed arrays, and pullbacks with an explicit inverse.

Does not own: row constraints, application-specific canonicality, protocol
acceptance, Rust, or generated artifacts.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks

universe u v

namespace Codec

/-- Every coordinate list of the declared width represents an admissible
semantic value. -/
def ExactWidthRecoverable
    {α : Type u}
    (codec : Codec α) : Prop :=
  ∀ coordinates,
    coordinates.length = codec.width →
      ∃ value,
        codec.Admissible value ∧ codec.encode value = coordinates

/-- Exact-width recovery gives one successful decoder result. -/
theorem decode_exists_of_exactWidthRecoverable
    {α : Type u}
    {codec : Codec α}
    (recoverable : codec.ExactWidthRecoverable)
    (coordinates : List Field)
    (lengthExact : coordinates.length = codec.width) :
    ∃ value, codec.decode coordinates = some value := by
  rcases recoverable coordinates lengthExact with
    ⟨value, admissible, encoded⟩
  refine ⟨value, ?_⟩
  rw [← encoded]
  exact codec.decode_encode value admissible

/-- One canonical field coordinate is recoverable at every exact-width
input. -/
theorem fieldCodec_exactWidthRecoverable :
    fieldCodec.ExactWidthRecoverable := by
  intro coordinates lengthExact
  cases coordinates with
  | nil => simp [fieldCodec] at lengthExact
  | cons head tail =>
      cases tail with
      | nil => exact ⟨head, True.intro, rfl⟩
      | cons next rest => simp [fieldCodec] at lengthExact

/-- One bounded natural coordinate recovers its canonical field
representative. -/
theorem boundedNatCodec_exactWidthRecoverable :
    boundedNatCodec.ExactWidthRecoverable := by
  intro coordinates lengthExact
  cases coordinates with
  | nil => simp [boundedNatCodec] at lengthExact
  | cons head tail =>
      cases tail with
      | nil =>
          refine ⟨head.val, head.isLt, ?_⟩
          have decoded :
              boundedNatCodec.decode [head] = some head.val :=
            (boundedNatCodec_decode_singleton_iff head head.val).2 rfl
          exact (boundedNatCodec.encode_decode [head] head.val decoded).2
      | cons next rest => simp [boundedNatCodec] at lengthExact

/-- Exact-width recovery is closed under ordered codec products. -/
theorem product_exactWidthRecoverable
    {α : Type u}
    {β : Type v}
    (left : Codec α)
    (right : Codec β)
    (leftRecoverable : left.ExactWidthRecoverable)
    (rightRecoverable : right.ExactWidthRecoverable) :
    (product left right).ExactWidthRecoverable := by
  intro coordinates lengthExact
  have leftLength :
      (coordinates.take left.width).length = left.width := by
    simp [List.length_take, lengthExact, product,
      ofInjectiveEncoding]
  have rightLength :
      (coordinates.drop left.width).length = right.width := by
    simp [List.length_drop, lengthExact, product,
      ofInjectiveEncoding]
  rcases leftRecoverable _ leftLength with
    ⟨leftValue, leftAdmissible, leftEncoded⟩
  rcases rightRecoverable _ rightLength with
    ⟨rightValue, rightAdmissible, rightEncoded⟩
  refine
    ⟨(leftValue, rightValue), ⟨leftAdmissible, rightAdmissible⟩, ?_⟩
  change
    left.encode leftValue ++ right.encode rightValue = coordinates
  rw [leftEncoded, rightEncoded, List.take_append_drop]

/-- Exact-width recovery is closed under fixed finite functions. -/
theorem finFunction_exactWidthRecoverable
    {α : Type u}
    (codec : Codec α)
    (recoverable : codec.ExactWidthRecoverable)
    (count : Nat) :
    (finFunction count codec).ExactWidthRecoverable := by
  induction count with
  | zero =>
      intro coordinates lengthExact
      have empty : coordinates = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa [finFunction, ofInjectiveEncoding] using lengthExact
      subst coordinates
      refine ⟨Fin.elim0, ?_, rfl⟩
      intro index
      exact Fin.elim0 index
  | succ count inductionHypothesis =>
      intro coordinates lengthExact
      have headLength :
          (coordinates.take codec.width).length = codec.width := by
        simp [List.length_take, lengthExact, finFunction,
          ofInjectiveEncoding, Nat.succ_mul]
      have tailLength :
          (coordinates.drop codec.width).length =
            (finFunction count codec).width := by
        simp [List.length_drop, lengthExact, finFunction,
          ofInjectiveEncoding, Nat.succ_mul]
      rcases recoverable _ headLength with
        ⟨head, headAdmissible, headEncoded⟩
      rcases inductionHypothesis _ tailLength with
        ⟨tail, tailAdmissible, tailEncoded⟩
      let values : Fin (count + 1) → α :=
        Fin.cases head tail
      refine ⟨values, ?_, ?_⟩
      · intro index
        exact Fin.cases headAdmissible tailAdmissible index
      · change encodeFin codec (count + 1) values = coordinates
        simp only [encodeFin, values, Fin.cases_zero,
          Fin.cases_succ]
        have tailEncoded' :
            encodeFin codec count tail =
              coordinates.drop codec.width := by
          simpa [finFunction, ofInjectiveEncoding] using tailEncoded
        rw [headEncoded, tailEncoded', List.take_append_drop]

/-- Exact-width recovery is closed under exact-length lists. -/
theorem fixedList_exactWidthRecoverable
    {α : Type u}
    (count : Nat)
    (default : α)
    (codec : Codec α)
    (recoverable : codec.ExactWidthRecoverable) :
    (fixedList count default codec).ExactWidthRecoverable := by
  intro coordinates lengthExact
  have functionLength :
      coordinates.length = (finFunction count codec).width := by
    simpa [fixedList, finFunction, ofInjectiveEncoding] using lengthExact
  rcases finFunction_exactWidthRecoverable codec recoverable count
      coordinates functionLength with
    ⟨values, valuesAdmissible, valuesEncoded⟩
  let list := List.ofFn values
  refine ⟨list, ?_, ?_⟩
  · constructor
    · simp [list]
    · intro index
      simpa [list, List.getD_eq_getElem?_getD, index.isLt] using
        valuesAdmissible index
  · change
      encodeFin codec count
          (fun index => list.getD index.val default) =
        coordinates
    have valuesExact :
        (fun index : Fin count => list.getD index.val default) =
          values := by
      funext index
      simp [list, List.getD_eq_getElem?_getD, index.isLt]
    have valuesEncoded' :
        encodeFin codec count values = coordinates := by
      simpa [finFunction, ofInjectiveEncoding] using valuesEncoded
    rw [valuesExact, valuesEncoded']

/-- Exact-width recovery is closed under exact-size arrays. -/
theorem fixedArray_exactWidthRecoverable
    {α : Type u}
    (count : Nat)
    (default : α)
    (codec : Codec α)
    (recoverable : codec.ExactWidthRecoverable) :
    (fixedArray count default codec).ExactWidthRecoverable := by
  intro coordinates lengthExact
  have functionLength :
      coordinates.length = (finFunction count codec).width := by
    simpa [fixedArray, finFunction, ofInjectiveEncoding] using lengthExact
  rcases finFunction_exactWidthRecoverable codec recoverable count
      coordinates functionLength with
    ⟨values, valuesAdmissible, valuesEncoded⟩
  let array := Array.ofFn values
  refine ⟨array, ?_, ?_⟩
  · constructor
    · simp [array]
    · intro index
      simpa [array] using valuesAdmissible index
  · change
      encodeFin codec count
          (fun index => array.getD index.val default) =
        coordinates
    have valuesExact :
        (fun index : Fin count => array.getD index.val default) =
          values := by
      funext index
      simp [array, index.isLt]
    have valuesEncoded' :
        encodeFin codec count values = coordinates := by
      simpa [finFunction, ofInjectiveEncoding] using valuesEncoded
    rw [valuesExact, valuesEncoded']

/-- Pullback recovery requires a concrete inverse for every recovered target
value. -/
theorem pullback_exactWidthRecoverable
    {α : Type u}
    {β : Type v}
    (target : Codec β)
    (toTarget : α → β)
    (toInjective : Function.Injective toTarget)
    (fromTarget : β → α)
    (rightInverse :
      ∀ value, target.Admissible value →
        toTarget (fromTarget value) = value)
    (targetRecoverable : target.ExactWidthRecoverable) :
    (pullback target toTarget toInjective).ExactWidthRecoverable := by
  intro coordinates lengthExact
  rcases targetRecoverable coordinates lengthExact with
    ⟨targetValue, targetAdmissible, targetEncoded⟩
  refine
    ⟨fromTarget targetValue, ?_, ?_⟩
  · change target.Admissible (toTarget (fromTarget targetValue))
    rw [rightInverse targetValue targetAdmissible]
    exact targetAdmissible
  · change target.encode (toTarget (fromTarget targetValue)) = coordinates
    rw [rightInverse targetValue targetAdmissible]
    exact targetEncoded

/-- Domain-restricted pullback recovery requires a concrete inverse whose
result is inside the selected source domain. -/
theorem pullbackOn_exactWidthRecoverable
    {α : Type u}
    {β : Type v}
    (target : Codec β)
    (sourceAdmissible : α → Prop)
    (toTarget : α → β)
    (targetAdmissible :
      ∀ value, sourceAdmissible value →
        target.Admissible (toTarget value))
    (toInjective :
      ∀ {left right},
        sourceAdmissible left →
        sourceAdmissible right →
        toTarget left = toTarget right →
        left = right)
    (fromTarget : β → α)
    (fromAdmissible :
      ∀ value, target.Admissible value →
        sourceAdmissible (fromTarget value))
    (rightInverse :
      ∀ value, target.Admissible value →
        toTarget (fromTarget value) = value)
    (targetRecoverable : target.ExactWidthRecoverable) :
    (pullbackOn target sourceAdmissible toTarget
      targetAdmissible toInjective).ExactWidthRecoverable := by
  intro coordinates lengthExact
  rcases targetRecoverable coordinates lengthExact with
    ⟨targetValue, targetValueAdmissible, targetEncoded⟩
  refine
    ⟨fromTarget targetValue, fromAdmissible _ targetValueAdmissible, ?_⟩
  change target.encode (toTarget (fromTarget targetValue)) = coordinates
  rw [rightInverse targetValue targetValueAdmissible, targetEncoded]

end Codec

end Nightstream.Implementation.Lowering.Goldilocks
