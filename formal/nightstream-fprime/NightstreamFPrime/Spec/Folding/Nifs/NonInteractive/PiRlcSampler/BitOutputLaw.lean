import Mathlib.Data.Vector.Basic
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldOutputLaw
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-!
Owns the finite output law of the existing 54-of-64 decoder under uniform
16-bit candidates. Accepted-prefix residue permutations preserve all input
positions and rejection flags. No law is assigned to the transcript.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.BitOutputLaw

open ProductionAlphabet ProductionStrongSet Sampling

private def acceptedCoordinates : AcceptedChunk ≃ Fin acceptedQuotientCount × Coefficient where
  toFun := factor
  invFun := combine
  left_inv := combine_factor
  right_inv := factor_combine

private def acceptedPerm (permutation : Equiv.Perm Coefficient) : Equiv.Perm AcceptedChunk :=
  (acceptedCoordinates.trans
    (Equiv.prodCongr (Equiv.refl (Fin acceptedQuotientCount)) permutation)).trans
      acceptedCoordinates.symm

private def chunkPerm (permutation : Equiv.Perm Coefficient) : Equiv.Perm Chunk :=
  (acceptedPerm permutation).subtypeCongr (Equiv.refl _)

private theorem chunkPerm_accepted (permutation : Equiv.Perm Coefficient) (candidate : Chunk)
    (accepted : verifier.accepts candidate = true) :
    chunkPerm permutation candidate = (acceptedPerm permutation ⟨candidate, accepted⟩).val :=
  Equiv.Perm.subtypeCongr.left_apply
    (p := fun value : Chunk => verifier.accepts value = true) (a := candidate)
    (acceptedPerm permutation) (Equiv.refl {value : Chunk // ¬ verifier.accepts value = true}) accepted

private theorem chunkPerm_rejected (permutation : Equiv.Perm Coefficient) (candidate : Chunk)
    (rejected : ¬ verifier.accepts candidate = true) : chunkPerm permutation candidate = candidate :=
  Equiv.Perm.subtypeCongr.right_apply
    (p := fun value : Chunk => verifier.accepts value = true) (a := candidate)
    (acceptedPerm permutation) (Equiv.refl {value : Chunk // ¬ verifier.accepts value = true}) rejected

private theorem chunkPerm_accepts (permutation : Equiv.Perm Coefficient) (candidate : Chunk) :
    verifier.accepts (chunkPerm permutation candidate) = verifier.accepts candidate := by
  by_cases accepted : verifier.accepts candidate = true
  · rw [chunkPerm_accepted permutation candidate accepted]
    exact (acceptedPerm permutation ⟨candidate, accepted⟩).property.trans accepted.symm
  · rw [chunkPerm_rejected permutation candidate accepted]

private theorem chunkPerm_symbol (permutation : Equiv.Perm Coefficient) (candidate : Chunk)
    (accepted : verifier.accepts candidate = true) :
    verifier.symbol (chunkPerm permutation candidate) = permutation (verifier.symbol candidate) := by
  rw [chunkPerm_accepted permutation candidate accepted]
  change (factor (combine ((factor ⟨candidate, accepted⟩).1,
    permutation (symbol candidate)))).2 = permutation (symbol candidate)
  exact congrArg Prod.snd (factor_combine _)

private def scan (permutations : Nat → Equiv.Perm Chunk) : Nat → List Chunk → List Chunk
  | _, [] => []
  | rank, candidate :: suffix =>
      permutations rank candidate ::
        scan permutations (rank + if verifier.accepts candidate then 1 else 0) suffix

private theorem scan_length (permutations : Nat → Equiv.Perm Chunk)
    (rank : Nat) (candidates : List Chunk) :
    (scan permutations rank candidates).length = candidates.length := by
  induction candidates generalizing rank with
  | nil => rfl
  | cons candidate suffix inductionHypothesis =>
      simp only [scan, List.length_cons, inductionHypothesis]

private theorem acceptedCount_cons (candidate : Chunk) (suffix : List Chunk) :
    FirstAccepted.acceptedCount verifier (candidate :: suffix) =
      (if verifier.accepts candidate then 1 else 0) + FirstAccepted.acceptedCount verifier suffix := by
  cases accepted : verifier.accepts candidate <;>
    simp [FirstAccepted.acceptedCount, FirstAccepted.acceptedCandidates, accepted, Nat.add_comm]

private theorem acceptedSymbols_cons (candidate : Chunk) (suffix : List Chunk) :
    FirstAccepted.acceptedSymbols verifier (candidate :: suffix) =
      if verifier.accepts candidate then
        verifier.symbol candidate :: FirstAccepted.acceptedSymbols verifier suffix
      else FirstAccepted.acceptedSymbols verifier suffix := by
  cases accepted : verifier.accepts candidate <;>
    simp [FirstAccepted.acceptedSymbols, FirstAccepted.acceptedCandidates, accepted]

private theorem scan_acceptedCount (permutations : Nat → Equiv.Perm Chunk)
    (preserves : ∀ rank candidate,
      verifier.accepts (permutations rank candidate) = verifier.accepts candidate)
    (rank : Nat) (candidates : List Chunk) :
    FirstAccepted.acceptedCount verifier (scan permutations rank candidates) =
      FirstAccepted.acceptedCount verifier candidates := by
  induction candidates generalizing rank with
  | nil => rfl
  | cons candidate suffix inductionHypothesis =>
      simp only [scan, acceptedCount_cons, preserves, inductionHypothesis]

private theorem scan_inverse (permutations : Nat → Equiv.Perm Chunk)
    (preserves : ∀ rank candidate,
      verifier.accepts (permutations rank candidate) = verifier.accepts candidate)
    (rank : Nat) (candidates : List Chunk) :
    scan (fun rank => (permutations rank).symm) rank (scan permutations rank candidates) =
      candidates := by
  induction candidates generalizing rank with
  | nil => rfl
  | cons candidate suffix inductionHypothesis =>
      simp only [scan, preserves, Equiv.symm_apply_apply, inductionHypothesis]

private theorem inverse_preserves (permutations : Nat → Equiv.Perm Chunk)
    (preserves : ∀ rank candidate,
      verifier.accepts (permutations rank candidate) = verifier.accepts candidate)
    (rank : Nat) (candidate : Chunk) :
    verifier.accepts ((permutations rank).symm candidate) = verifier.accepts candidate := by
  have same := preserves rank ((permutations rank).symm candidate)
  simpa only [Equiv.apply_symm_apply] using same.symm

private def vectorPerm (permutations : Nat → Equiv.Perm Chunk)
    (preserves : ∀ rank candidate,
      verifier.accepts (permutations rank candidate) = verifier.accepts candidate)
    (count : Nat) : Equiv.Perm (List.Vector Chunk count) where
  toFun input := ⟨scan permutations 0 input.val, (scan_length _ _ _).trans input.property⟩
  invFun input :=
    ⟨scan (fun rank => (permutations rank).symm) 0 input.val,
      (scan_length _ _ _).trans input.property⟩
  left_inv input := Subtype.ext (scan_inverse permutations preserves 0 input.val)
  right_inv input := by
    apply Subtype.ext
    simpa only [Equiv.symm_symm] using
      scan_inverse (fun rank => (permutations rank).symm)
        (inverse_preserves permutations preserves) 0 input.val

private def windowPerm (permutations : Nat → Equiv.Perm Chunk)
    (preserves : ∀ rank candidate,
      verifier.accepts (permutations rank candidate) = verifier.accepts candidate)
    (count : Nat) : Equiv.Perm (Fin count → Chunk) :=
  ((Equiv.vectorEquivFin Chunk count).symm.trans
    (vectorPerm permutations preserves count)).trans
      (Equiv.vectorEquivFin Chunk count)

private theorem ofFn_windowPerm (permutations : Nat → Equiv.Perm Chunk)
    (preserves : ∀ rank candidate,
      verifier.accepts (permutations rank candidate) = verifier.accepts candidate)
    (count : Nat) (window : Fin count → Chunk) :
    List.ofFn (windowPerm permutations preserves count window) =
      scan permutations 0 (List.ofFn window) := by
  have copied := congrArg List.Vector.toList (List.Vector.ofFn_get
    (vectorPerm permutations preserves count
      ((Equiv.vectorEquivFin Chunk count).symm window)))
  rw [List.Vector.toList_ofFn] at copied
  change List.ofFn (windowPerm permutations preserves count window) =
    scan permutations 0 (List.Vector.ofFn window).toList at copied
  rw [List.Vector.toList_ofFn] at copied
  exact copied

private theorem scan_symbols (permutations : Nat → Equiv.Perm Coefficient)
    (rank : Nat) (candidates : List Chunk) :
    FirstAccepted.acceptedSymbols verifier
        (scan (fun rank => chunkPerm (permutations rank)) rank candidates) =
      (FirstAccepted.acceptedSymbols verifier candidates).mapIdx
        (fun index value => permutations (rank + index) value) := by
  induction candidates generalizing rank with
  | nil => rfl
  | cons candidate suffix inductionHypothesis =>
      cases accepted : verifier.accepts candidate <;>
        simp [scan, acceptedSymbols_cons, chunkPerm_accepts, accepted,
          chunkPerm_symbol, inductionHypothesis, Nat.add_comm, Nat.add_left_comm]

private theorem take_mapIdx {Element Result : Type*} (action : Nat → Element → Result)
    (values : List Element) (count : Nat) :
    (values.mapIdx action).take count = (values.take count).mapIdx action := by
  induction values generalizing count action with
  | nil => simp
  | cons value suffix inductionHypothesis =>
      cases count <;> simp [List.mapIdx_cons, inductionHypothesis]

private theorem firstAccepted_scan (permutations : Nat → Equiv.Perm Coefficient)
    (rank need : Nat) (candidates : List Chunk) :
    FirstAccepted.firstAccepted verifier need
        (scan (fun rank => chunkPerm (permutations rank)) rank candidates) =
      (FirstAccepted.firstAccepted verifier need candidates).mapIdx
        (fun index value => permutations (rank + index) value) := by
  unfold FirstAccepted.firstAccepted
  rw [scan_symbols, take_mapIdx]

private theorem boundedSample_scan (permutations : Nat → Equiv.Perm Coefficient)
    (need : Nat) (candidates : List Chunk) :
    FirstAccepted.boundedSample verifier need
        (scan (fun rank => chunkPerm (permutations rank)) 0 candidates) =
      (FirstAccepted.boundedSample verifier need candidates).map
        (fun values => values.mapIdx (fun index value => permutations index value)) := by
  have sameCount := scan_acceptedCount (fun rank => chunkPerm (permutations rank))
    (fun rank candidate => chunkPerm_accepts (permutations rank) candidate) 0 candidates
  unfold FirstAccepted.boundedSample
  rw [sameCount]
  by_cases enough : need ≤ FirstAccepted.acceptedCount verifier candidates
  · simp only [enough, if_true, Option.map_some]
    rw [firstAccepted_scan]
    simp only [Nat.zero_add]
  · simp only [enough, if_false, Option.map_none]

private theorem symbolMap_injective (permutations : Nat → Equiv.Perm Coefficient) :
    Function.Injective
      (fun values : List Coefficient => values.mapIdx (fun index value => permutations index value)) := by
  have inverse : Function.LeftInverse
      (fun values : List Coefficient =>
        values.mapIdx (fun index value => (permutations index).symm value))
      (fun values : List Coefficient =>
        values.mapIdx (fun index value => permutations index value)) := by
    intro values
    apply List.ext_getElem
    · simp only [List.length_mapIdx]
    · intro index leftBound rightBound
      simp only [List.getElem_mapIdx, Equiv.symm_apply_apply]
  exact inverse.injective

private def between (left right : Scalar) (index : Nat) : Equiv.Perm Coefficient :=
  if within : index < coefficientCount then
    Equiv.swap (left ⟨index, within⟩) (right ⟨index, within⟩)
  else Equiv.refl Coefficient

private theorem between_maps_left (left right : Scalar) :
    (List.ofFn left).mapIdx (fun index value => between left right index value) = List.ofFn right := by
  apply List.ext_getElem
  · simp only [List.length_mapIdx, List.length_ofFn]
  · intro index leftBound rightBound
    have within : index < coefficientCount := by simpa only [List.length_ofFn] using rightBound
    simp only [List.getElem_mapIdx, List.getElem_ofFn, between, dif_pos within,
      Equiv.swap_apply_left]

private def successFiberEquiv (left right : Scalar) :
    {window : ShortfallBound.Window // FirstAccepted.boundedSample verifier coefficientCount
        (List.ofFn window) = some (List.ofFn left)} ≃
      {window : ShortfallBound.Window // FirstAccepted.boundedSample verifier coefficientCount
        (List.ofFn window) = some (List.ofFn right)} :=
  Equiv.subtypeEquiv
    (p := fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window) = some (List.ofFn left))
    (q := fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window) = some (List.ofFn right))
    (windowPerm (fun rank => chunkPerm (between left right rank))
      (fun rank candidate => chunkPerm_accepts (between left right rank) candidate)
      candidateBound) (by
    intro window
    dsimp only
    rw [ofFn_windowPerm, boundedSample_scan]
    constructor
    · intro success
      rw [success, Option.map_some, between_maps_left]
    · intro success
      obtain ⟨output, sampled, mapped⟩ := Option.map_eq_some_iff.mp success
      have same := symbolMap_injective (between left right)
        (mapped.trans (between_maps_left left right).symm)
      simpa only [same] using sampled)

/-- Every 54-coefficient scalar has the same successful preimage count in
the complete 64-candidate window. The bijection preserves rejection positions. -/
theorem successful_fiber_card_eq (left right : Scalar) :
    Nat.card {window : ShortfallBound.Window // FirstAccepted.boundedSample verifier coefficientCount
        (List.ofFn window) = some (List.ofFn left)} =
      Nat.card {window : ShortfallBound.Window // FirstAccepted.boundedSample verifier coefficientCount
        (List.ofFn window) = some (List.ofFn right)} :=
  Nat.card_congr (successFiberEquiv left right)

attribute [local instance] Classical.propDecidable

private theorem encoded_event_count {Input Output Carrier : Type*}
    [Finite Input] [Finite Output]
    (encode : Output → Carrier) (decode : Input → Option Carrier)
    (injective : Function.Injective encode)
    (supported : ∀ input value, decode input = some value → ∃ output, encode output = value)
    (fiberSize : Nat)
    (balanced : ∀ output, Nat.card {input // decode input = some (encode output)} = fiberSize)
    (event : Option Carrier → Prop) :
    Nat.card {input // event (decode input)} =
      (if event none then Nat.card {input // decode input = none} else 0) +
        Nat.card {output // event (some (encode output))} * fiberSize := by
  let NonePart := {input : Input // decode input = none ∧ event none}
  let SomePart := Σ output : {output : Output // event (some (encode output))},
    {input : Input // decode input = some (encode output.val)}
  let project : NonePart ⊕ SomePart → {input : Input // event (decode input)}
    | Sum.inl input => ⟨input.val, input.property.1.symm ▸ input.property.2⟩
    | Sum.inr ⟨output, input⟩ => ⟨input.val, input.property.symm ▸ output.property⟩
  have projectInjective : Function.Injective project := by
    rintro (⟨left, leftNone, leftEvent⟩ | ⟨⟨first, firstEvent⟩, ⟨left, leftSome⟩⟩)
      (⟨right, rightNone, rightEvent⟩ | ⟨⟨second, secondEvent⟩, ⟨right, rightSome⟩⟩) same
    · have inputs : left = right := congrArg Subtype.val same
      subst right
      rfl
    · have inputs : left = right := congrArg Subtype.val same
      subst right
      exact (Option.some_ne_none _ (rightSome.symm.trans leftNone)).elim
    · have inputs : left = right := congrArg Subtype.val same
      subst right
      exact (Option.some_ne_none _ (leftSome.symm.trans rightNone)).elim
    · have inputs : left = right := congrArg Subtype.val same
      subst right
      have outputs := injective (Option.some.inj (leftSome.symm.trans rightSome))
      change first = second at outputs
      subst second
      rfl
  have projectSurjective : Function.Surjective project := by
    rintro ⟨input, included⟩
    cases sampled : decode input with
    | none => exact ⟨Sum.inl ⟨input, sampled, sampled ▸ included⟩, rfl⟩
    | some value =>
        obtain ⟨output, encoded⟩ := supported input value sampled
        subst value
        exact ⟨Sum.inr ⟨⟨output, sampled ▸ included⟩, ⟨input, sampled⟩⟩, rfl⟩
  have noneCount : Nat.card NonePart =
      if event none then Nat.card {input // decode input = none} else 0 := by
    by_cases included : event none
    · rw [if_pos included]
      exact Nat.card_congr (Equiv.subtypeEquivRight (fun input =>
        (and_iff_left included : (decode input = none ∧ event none) ↔ decode input = none)))
    · rw [if_neg included]
      letI : IsEmpty NonePart := ⟨fun input => included input.property.2⟩
      exact Nat.card_of_isEmpty
  have someCount : Nat.card SomePart =
      Nat.card {output : Output // event (some (encode output))} * fiberSize := by
    letI : Fintype {output : Output // event (some (encode output))} := Fintype.ofFinite _
    rw [Nat.card_sigma]
    simp only [balanced, Finset.sum_const, Finset.card_univ,
      Nat.nsmul_eq_mul, Nat.card_eq_fintype_card]
  calc
    _ = Nat.card (NonePart ⊕ SomePart) :=
      (Nat.card_eq_of_bijective project ⟨projectInjective, projectSurjective⟩).symm
    _ = Nat.card NonePart + Nat.card SomePart := Nat.card_sum
    _ = _ := by rw [noneCount, someCount]

private theorem balanced_option_law {Input Output Carrier : Type*}
    [Finite Input] [Finite Output]
    (encode : Output → Carrier) (decode : Input → Option Carrier)
    (injective : Function.Injective encode)
    (supported : ∀ input value, decode input = some value → ∃ output, encode output = value)
    (fiberSize : Nat)
    (balanced : ∀ output, Nat.card {input // decode input = some (encode output)} = fiberSize)
    (inputPositive : (0 : ℚ) < Nat.card Input) (outputPositive : (0 : ℚ) < Nat.card Output)
    (event : Option Carrier → Prop) :
    let abort : ℚ := (Nat.card {input // decode input = none} : ℚ) / Nat.card Input
    let uniform : ℚ := (Nat.card {output // event (some (encode output))} : ℚ) / Nat.card Output
    (Nat.card {input // event (decode input)} : ℚ) / Nat.card Input =
        (1 - abort) * uniform + (if event none then abort else 0) ∧
      |(Nat.card {input // event (decode input)} : ℚ) / Nat.card Input - uniform| ≤ abort := by
  dsimp only
  have counted := encoded_event_count encode decode injective supported fiberSize balanced event
  have total : Nat.card Input =
      Nat.card {input // decode input = none} + Nat.card Output * fiberSize := by
    simpa only [Nat.card_subtype_true, if_true] using
      encoded_event_count encode decode injective supported fiberSize balanced (fun _ => True)
  have totalQ : (Nat.card Input : ℚ) =
      (Nat.card {input // decode input = none} : ℚ) + (Nat.card Output : ℚ) * fiberSize := by
    exact_mod_cast total
  have totalForEvent := congrArg
    (fun count : ℚ => (Nat.card {output // event (some (encode output))} : ℚ) * count) totalQ
  have mixture : (Nat.card {input // event (decode input)} : ℚ) / Nat.card Input =
      (1 - (Nat.card {input // decode input = none} : ℚ) / Nat.card Input) *
        ((Nat.card {output // event (some (encode output))} : ℚ) / Nat.card Output) +
      (if event none then (Nat.card {input // decode input = none} : ℚ) / Nat.card Input else 0) := by
    rw [counted]
    split_ifs <;> simp only [Nat.cast_add, Nat.cast_mul, Nat.cast_zero] <;>
      field_simp [ne_of_gt inputPositive, ne_of_gt outputPositive] <;>
      nlinarith only [totalForEvent]
  refine ⟨mixture, ?_⟩
  have abortNonnegative :
      (0 : ℚ) ≤ (Nat.card {input // decode input = none} : ℚ) / Nat.card Input :=
    div_nonneg (Nat.cast_nonneg _) (le_of_lt inputPositive)
  have uniformNonnegative :
      (0 : ℚ) ≤ (Nat.card {output // event (some (encode output))} : ℚ) / Nat.card Output :=
    div_nonneg (Nat.cast_nonneg _) (le_of_lt outputPositive)
  have includedLe : (Nat.card {output // event (some (encode output))} : ℚ) ≤ Nat.card Output := by
    have counted : Nat.card {output : Output // event (some (encode output))} ≤ Nat.card Output :=
      Nat.card_le_card_of_injective Subtype.val Subtype.val_injective
    exact_mod_cast counted
  have uniformLe :
      (Nat.card {output // event (some (encode output))} : ℚ) / Nat.card Output ≤ 1 :=
    (div_le_one outputPositive).mpr includedLe
  have productNonnegative := mul_nonneg abortNonnegative uniformNonnegative
  have productLe := mul_le_mul_of_nonneg_left uniformLe abortNonnegative
  rw [mixture]
  split_ifs <;> rw [abs_le] <;> constructor <;>
    nlinarith only [abortNonnegative, productNonnegative, productLe]

private theorem scalar_cardinality : Nat.card Scalar = alphabetSize ^ coefficientCount := by
  rw [Nat.card_fun, Nat.card_fin, Nat.card_fin]

private theorem list_encoded_of_length {Element : Type*} {count : Nat}
    (values : List Element) (length : values.length = count) :
    ∃ output : Fin count → Element, List.ofFn output = values := by
  let vector : List.Vector Element count := ⟨values, length⟩
  refine ⟨vector.get, ?_⟩
  have copied := congrArg List.Vector.toList (List.Vector.ofFn_get vector)
  rw [List.Vector.toList_ofFn] at copied
  exact copied

private theorem boundedSample_supported (window : ShortfallBound.Window) (values : List Coefficient)
    (success : FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window) = some values) :
    ∃ scalar : Scalar, List.ofFn scalar = values :=
  list_encoded_of_length values (FirstAccepted.bounded_success_length success)

/-- Uniform mass on successful scalar encodings; `none` has zero mass. -/
noncomputable def uniformSomeFrequency (event : Option (List Coefficient) → Prop) : ℚ :=
  (Nat.card {scalar : Scalar // event (some (List.ofFn scalar))} : ℚ) /
    (alphabetSize : ℚ) ^ coefficientCount

private theorem selected_option_law (event : Option (List Coefficient) → Prop) :
    let decode := fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)
    FieldOutputLaw.bitDecodedFrequency decode event =
        (1 - ShortfallBound.iidBitShortfallProbability) * uniformSomeFrequency event +
          (if event none then ShortfallBound.iidBitShortfallProbability else 0) ∧
      |FieldOutputLaw.bitDecodedFrequency decode event - uniformSomeFrequency event| ≤
        ShortfallBound.iidBitShortfallProbability := by
  let base : Scalar := fun _ => ⟨0, by decide⟩
  have inputPositive : (0 : ℚ) < Nat.card ShortfallBound.Window := by
    rw [ShortfallBound.window_cardinality, Nat.cast_pow]
    exact pow_pos (by norm_num [chunkModulus]) _
  have outputPositive : (0 : ℚ) < Nat.card Scalar := by
    rw [scalar_cardinality, Nat.cast_pow]
    exact pow_pos (by norm_num [alphabetSize]) _
  have abortCount :
      Nat.card {window : ShortfallBound.Window //
        FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window) = none} =
      Nat.card {window : ShortfallBound.Window //
        FirstAccepted.Shortfall verifier coefficientCount (List.ofFn window)} :=
    Nat.card_congr (Equiv.subtypeEquivRight (fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample_eq_none_iff_shortfall (verifier := verifier)
        (need := coefficientCount) (candidates := List.ofFn window)))
  have law := balanced_option_law
    (List.ofFn : Scalar → List Coefficient)
    (fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window))
    List.ofFn_injective boundedSample_supported
    (Nat.card {window : ShortfallBound.Window //
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window) = some (List.ofFn base)})
    (fun output => successful_fiber_card_eq output base) inputPositive outputPositive event
  simpa only [FieldOutputLaw.bitDecodedFrequency, uniformSomeFrequency,
    ShortfallBound.iidBitShortfallProbability, abortCount, ShortfallBound.window_cardinality,
    scalar_cardinality, Nat.cast_pow] using law

/-- Exact full Option law: a uniform successful scalar and the retained abort mass. -/
theorem boundedSample_event_frequency_eq_mixture (event : Option (List Coefficient) → Prop) :
    let decode := fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)
    FieldOutputLaw.bitDecodedFrequency decode event =
      (1 - ShortfallBound.iidBitShortfallProbability) * uniformSomeFrequency event +
        (if event none then ShortfallBound.iidBitShortfallProbability else 0) :=
  (selected_option_law event).1

/-- Comparing with a uniform successful scalar costs at most the abort probability. -/
theorem boundedSample_event_error_le_shortfall (event : Option (List Coefficient) → Prop) :
    let decode := fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)
    |FieldOutputLaw.bitDecodedFrequency decode event - uniformSomeFrequency event| ≤
      ShortfallBound.iidBitShortfallProbability :=
  (selected_option_law event).2

/-- Uniform-field output bias and bit-sampler abort are separate finite error terms. -/
theorem boundedSample_field_event_error_le (event : Option (List Coefficient) → Prop) :
    let decode := fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)
    |FieldOutputLaw.fieldDecodedFrequency decode event - uniformSomeFrequency event| ≤
      32 * FieldPairLaw.pairDeviation + ShortfallBound.iidBitShortfallProbability := by
  let decode := fun window : ShortfallBound.Window =>
    FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)
  change |FieldOutputLaw.fieldDecodedFrequency decode event - uniformSomeFrequency event| ≤ _
  calc
    _ ≤ |FieldOutputLaw.fieldDecodedFrequency decode event -
          FieldOutputLaw.bitDecodedFrequency decode event| +
        |FieldOutputLaw.bitDecodedFrequency decode event - uniformSomeFrequency event| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add (FieldOutputLaw.boundedSample_event_frequency_error_le event)
      (boundedSample_event_error_le_shortfall event)

/-- The explicit scalar shortfall bound also gives an unconditional event bound. -/
theorem boundedSample_field_event_error_le_upper (event : Option (List Coefficient) → Prop) :
    let decode := fun window : ShortfallBound.Window =>
      FirstAccepted.boundedSample verifier coefficientCount (List.ofFn window)
    |FieldOutputLaw.fieldDecodedFrequency decode event - uniformSomeFrequency event| ≤
      32 * FieldPairLaw.pairDeviation +
        (Nat.choose candidateBound 11 : ℚ) / (chunkModulus : ℚ) ^ 11 :=
  (boundedSample_field_event_error_le event).trans
    (add_le_add (le_refl _) ShortfallBound.iid_bit_shortfall_probability_le)

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.BitOutputLaw
