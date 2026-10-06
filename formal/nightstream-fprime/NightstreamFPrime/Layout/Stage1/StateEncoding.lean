import NightstreamFPrime.Layout.PilotProduction
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits

/-!
Owns the injectivity facts for the Stage 1 state preimage and the radix-`2^17`
parent packing.

The preimage stores the Π_DEC parent public input once, so it is injective
only on states whose children are the canonical common-sign split of that
parent (`ChildrenCanonical`). That is the Π_DEC accepted language: the default
running instance and every accepted Π_DEC output satisfy it. `unpackWord`
inverts the packing on every bounded parent and `packWord ∘ unpackWord` is the
identity on every word, so the state decoder is exact.
-/

namespace NightstreamFPrime.Layout.Stage1.StateEncoding

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
  (EvaluationFamily)
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-! ## Canonical children -/

/-- The paper default running instance has zero children. -/
theorem defaultRunning_canonical :
    ChildrenCanonical (defaultRunning (logicalWidth := logicalWidth)
      (publicFits := publicFits)) := by
  intro column
  exact ⟨0, Or.inl rfl, fun _ => Or.inl rfl⟩

/-- Every running instance that the production NIFS verifier outputs is
canonical: its children are the checked split of a bounded parent. -/
theorem output_canonical
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    {running result : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof : Proof (ProductionKey.degreeBound relation)}
    (accepted : (ProductionKey.key relation ajtai).output running fresh proof = some result) :
    ChildrenCanonical result := by
  unfold NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Key.output at accepted
  rcases Option.bind_eq_some_iff.mp accepted with ⟨attempt, _, mapped⟩
  rcases Option.map_eq_some_iff.mp mapped with ⟨publicInputs, checkedEq, rfl⟩
  by_cases bounded : (ProductionKey.key relation ajtai).piDecPublicInputSplit.parentBounded
      attempt.parent.publicInput
  · rw [NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier.PublicInputSplit.checked_eq_some _ _ bounded] at checkedEq
    cases checkedEq
    intro column
    exact ⟨_, (Radix.UniformSignedDigits.honest_complete _ (bounded column)).constraint⟩
  · rw [NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier.PublicInputSplit.checked_eq_none _ _ bounded] at checkedEq
    cases checkedEq

/-- Exact validity conditions for one canonical Stage 1 state. -/
def WellFormed
    (preimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  PilotProduction.FixedPreimage preimage ∧
    preimage.iteration < goldilocksModulus ∧
    preimage.pc = 1 ∧
    ChildrenCanonical (preimage.running functionIndex)

/-! ## Fixed-width list lemmas -/

/-- Equal concatenations of fixed-width encodings agree position by position. -/
theorem finRange_flatMap_ext
    {count width : Nat}
    (left right : Fin count → List F)
    (leftLength : ∀ index, (left index).length = width)
    (rightLength : ∀ index, (right index).length = width)
    (equal : (List.finRange count).flatMap left =
      (List.finRange count).flatMap right)
    (index : Fin count) :
    left index = right index := by
  apply List.ext_getElem
  · rw [leftLength, rightLength]
  · intro inner leftBound rightBound
    have innerBound : inner < width := by
      rw [← leftLength index]
      exact leftBound
    have selected :
        ((List.finRange count).flatMap left).getD (index.val * width + inner) 0 =
          ((List.finRange count).flatMap right).getD (index.val * width + inner) 0 := by
      rw [equal]
    rw [finRange_flatMap_getD left leftLength index inner innerBound,
      finRange_flatMap_getD right rightLength index inner innerBound,
      List.getD_eq_getElem _ _ leftBound,
      List.getD_eq_getElem _ _ rightBound] at selected
    exact selected

private theorem finRange_map_injective
    {count : Nat} {left right : Fin count → F}
    (equal : (List.finRange count).map left = (List.finRange count).map right) :
    left = right := by
  funext index
  have selected := congrArg (fun words => words.getD index.val 0) equal
  simpa [List.getD_eq_getElem] using selected

private theorem serializeK_injective {left right : K}
    (equal : serializeK left = serializeK right) : left = right := by
  simp only [serializeK, List.cons.injEq, and_true] at equal
  cases left
  cases right
  simp_all

private theorem serializeCommitment_injective
    {left right : PaperAlgebra.Commitment}
    (equal : serializeCommitment left = serializeCommitment right) :
    left = right := by
  funext row
  have rowEqual := finRange_flatMap_ext
    (fun row => serializeRingF (left row))
    (fun row => serializeRingF (right row))
    (fun _ => serializeRingF_length _) (fun _ => serializeRingF_length _)
    equal row
  exact finRange_map_injective rowEqual

private theorem serializePublicInput_injective
    {left right : PaperAlgebra.PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits)}
    (equal : serializePublicInput (publicFits := publicFits) left =
      serializePublicInput (publicFits := publicFits) right) :
    left = right :=
  finRange_map_injective equal

private theorem serializeEvalK_injective
    {left right : EvaluationFamily K productionShape}
    (equal : serializeEvalK left = serializeEvalK right) :
    left.pad = right.pad := by
  funext coefficient
  exact serializeK_injective (finRange_flatMap_ext (width := 2)
    (fun coefficient => serializeK (left.pad coefficient))
    (fun coefficient => serializeK (right.pad coefficient))
    (fun _ => serializeK_length _) (fun _ => serializeK_length _) equal coefficient)

private theorem serializeEvalA_injective
    {left right : EvaluationFamily K productionShape}
    (equal : serializeEvalA left = serializeEvalA right) :
    left.matrix = right.matrix := by
  funext matrix coefficient
  have matrixEqual := finRange_flatMap_ext
    (width := productionShape.coefficientCount * 2)
    (fun matrix => (List.finRange productionShape.coefficientCount).flatMap
      fun coefficient => serializeK (left.matrix matrix coefficient))
    (fun matrix => (List.finRange productionShape.coefficientCount).flatMap
      fun coefficient => serializeK (right.matrix matrix coefficient))
    (fun _ => by simp) (fun _ => by simp) equal matrix
  exact serializeK_injective (finRange_flatMap_ext (width := 2)
    (fun coefficient => serializeK (left.matrix matrix coefficient))
    (fun coefficient => serializeK (right.matrix matrix coefficient))
    (fun _ => serializeK_length _) (fun _ => serializeK_length _) matrixEqual coefficient)

private theorem flatMap_serializeK_injective :
    ∀ {left right : List K}, left.length = right.length →
      left.flatMap serializeK = right.flatMap serializeK → left = right
  | [], [], _, _ => rfl
  | [], _ :: _, length, _ => by simp at length
  | _ :: _, [], length, _ => by simp at length
  | leftHead :: leftTail, rightHead :: rightTail, length, equal => by
      simp only [List.flatMap_cons] at equal
      rcases List.append_inj equal rfl with ⟨headEqual, tailEqual⟩
      rw [serializeK_injective headEqual,
        flatMap_serializeK_injective (by simpa using length) tailEqual]

private theorem serializePoint_injective
    {left right : CubePoint K cubeVariables}
    (equal : serializePoint left = serializePoint right) : left = right := by
  have coordinates := flatMap_serializeK_injective
    (by rw [left.dimension, right.dimension]) equal
  cases left
  cases right
  simp_all

/-! ## Layout injectivity -/

theorem running_ext
    {left right : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (point : left.point = right.point)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Equal shared running fields identify the point, commitments, and every
separate `Eval_K` and `Eval_A` value. -/
theorem serializeRunningFields_injective
    {left right : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (equal : serializeRunningFields left = serializeRunningFields right) :
    left.point = right.point ∧ left.commitments = right.commitments ∧
      left.evaluations = right.evaluations := by
  unfold serializeRunningFields at equal
  rcases List.append_inj equal
      (by simp only [List.length_append, serializeCommitments_length,
        serializeEvalKs_length, serializeEvalAs_length]) with
    ⟨fieldsEqual, pointEqual⟩
  rcases List.append_inj fieldsEqual
      (by simp only [List.length_append, serializeCommitments_length,
        serializeEvalKs_length]) with
    ⟨headEqual, evalAsEqual⟩
  rcases List.append_inj headEqual
      (by simp only [serializeCommitments_length]) with
    ⟨commitmentsEqual, evalKsEqual⟩
  refine ⟨serializePoint_injective pointEqual, ?_, ?_⟩
  · funext source
    exact serializeCommitment_injective (finRange_flatMap_ext _ _
      (fun _ => serializeCommitment_length _)
      (fun _ => serializeCommitment_length _) commitmentsEqual source)
  · funext source
    have padEqual := serializeEvalK_injective (finRange_flatMap_ext _ _
      (fun _ => serializeEvalK_length _) (fun _ => serializeEvalK_length _)
      evalKsEqual source)
    have matrixEqual := serializeEvalA_injective (finRange_flatMap_ext _ _
      (fun _ => serializeEvalA_length _) (fun _ => serializeEvalA_length _)
      evalAsEqual source)
    cases hLeft : left.evaluations source
    cases hRight : right.evaluations source
    rw [hLeft, hRight] at padEqual matrixEqual
    simp_all

private theorem natWord_injective_below_modulus
    {left right : Nat}
    (leftBound : left < goldilocksModulus)
    (rightBound : right < goldilocksModulus)
    (equal : natWord left = natWord right) :
    left = right := by
  have valuesEqual := congrArg Fin.val equal
  simpa [natWord, Spec.Poseidon2.ofNat,
    Nat.mod_eq_of_lt leftBound, Nat.mod_eq_of_lt rightBound] using valuesEqual

private theorem fin_slot_eq_functionIndex (index : Fin slotCount) :
    index = functionIndex := by
  apply Fin.ext
  have bound := index.isLt
  simp only [slotCount] at bound
  change index.val = 0
  omega

/-- Equal tails with the fixed ABI identify the key, the iteration word and
both application states. -/
private theorem serializeTail_injective
    {left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftFixed : PilotProduction.FixedPreimage left)
    (rightFixed : PilotProduction.FixedPreimage right)
    (equal : serializeTail left = serializeTail right) :
    left.verifierKeys functionIndex = right.verifierKeys functionIndex ∧
      natWord left.iteration = natWord right.iteration ∧
      left.z0 = right.z0 ∧ left.current = right.current := by
  rcases leftFixed with ⟨leftKey, leftZ0, leftCurrent⟩
  rcases rightFixed with ⟨rightKey, rightZ0, rightCurrent⟩
  unfold serializeTail at equal
  rcases List.append_inj equal
      (by simp only [List.length_append, List.length_singleton]; omega) with
    ⟨headEqual, currentEqual⟩
  rcases List.append_inj headEqual
      (by simp only [List.length_append, List.length_singleton]; omega) with
    ⟨prefixEqual, z0Equal⟩
  rcases List.append_inj prefixEqual (by omega) with ⟨keyEqual, iterationEqual⟩
  exact ⟨keyEqual, by simpa using iterationEqual, z0Equal, currentEqual⟩

private theorem preimage_ext
    {left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (key : left.verifierKeys functionIndex = right.verifierKeys functionIndex)
    (iteration : left.iteration = right.iteration)
    (z0 : left.z0 = right.z0) (current : left.current = right.current)
    (running : left.running functionIndex = right.running functionIndex)
    (pc : left.pc = right.pc) :
    left = right := by
  have keys : left.verifierKeys = right.verifierKeys := by
    funext index
    rw [fin_slot_eq_functionIndex index]
    exact key
  have runnings : left.running = right.running := by
    funext index
    rw [fin_slot_eq_functionIndex index]
    exact running
  cases left
  cases right
  simp_all

/-! ## Packed parent injectivity -/

private theorem ofNat_val (value : F) : Poseidon2.ofNat value.val = value := by
  apply Fin.ext
  simp [Poseidon2.ofNat, Nat.mod_eq_of_lt value.isLt]

private theorem ofNat_add (left right : Nat) :
    Poseidon2.ofNat left + Poseidon2.ofNat right =
      Poseidon2.ofNat (left + right) := by
  apply Fin.ext
  simp [Poseidon2.ofNat, Fin.val_add, Nat.add_mod]

private theorem ofNat_mul (left right : Nat) :
    Poseidon2.ofNat left * Poseidon2.ofNat right =
      Poseidon2.ofNat (left * right) := by
  apply Fin.ext
  simp [Poseidon2.ofNat, Fin.val_mul, Nat.mul_mod]

/-- A coordinate of centered magnitude below `2^16`, shifted by `2^16`, is a
natural number below `2^17`. -/
private theorem shifted_val_lt {value : F}
    (bounded : centeredMagnitude value < 2 ^ 16) :
    (value + packOffset).val < 131072 := by
  have valueBound := value.isLt
  have small : value.val < 65536 ∨ goldilocksModulus - value.val < 65536 := by
    unfold centeredMagnitude at bounded
    rw [min_lt_iff] at bounded
    norm_num at bounded
    exact bounded
  have offsetValue : packOffset.val = 65536 := by
    simp [packOffset, Poseidon2.ofNat, goldilocksModulus]
  rw [Fin.val_add, offsetValue]
  unfold goldilocksModulus at *
  rcases small with small | large
  · rw [Nat.mod_eq_of_lt (by omega)]
    omega
  · rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
    omega

open Fin.CommRing in
private theorem packWord_shift (low middle high : F) :
    packWord (low + packOffset) (middle + packOffset) (high + packOffset) =
      packWord low middle high + packShift := by
  simp only [packShift, packWord]
  ring

private theorem packWord_val {low middle high : F}
    (lowBound : low.val < 131072) (middleBound : middle.val < 131072)
    (highBound : high.val < 131072) :
    (packWord low middle high).val =
      low.val + 131072 * middle.val + 17179869184 * high.val := by
  have asNat : packWord low middle high =
      Poseidon2.ofNat (low.val + 2 ^ 17 * middle.val + 2 ^ 17 * 2 ^ 17 * high.val) := by
    calc packWord low middle high
        = Poseidon2.ofNat low.val + Poseidon2.ofNat (2 ^ 17) * Poseidon2.ofNat middle.val +
            Poseidon2.ofNat (2 ^ 17) * Poseidon2.ofNat (2 ^ 17) *
              Poseidon2.ofNat high.val := by
          rw [ofNat_val, ofNat_val, ofNat_val]
          rfl
      _ = _ := by
          rw [ofNat_mul, ofNat_mul, ofNat_mul, ofNat_add, ofNat_add]
  rw [asNat]
  simp only [Poseidon2.ofNat]
  norm_num
  apply Nat.mod_eq_of_lt
  unfold goldilocksModulus
  omega

private theorem radix_unique {a b c d e f : Nat}
    (aBound : a < 131072) (bBound : b < 131072)
    (dBound : d < 131072) (eBound : e < 131072)
    (equal : a + 131072 * b + 17179869184 * c = d + 131072 * e + 17179869184 * f) :
    a = d ∧ b = e ∧ c = f := by
  omega

/-- Radix-`2^17` packing is injective on coordinates of centered magnitude
below `2^16`. -/
theorem packWord_injective {leftLow leftMiddle leftHigh
      rightLow rightMiddle rightHigh : F}
    (leftLowBound : centeredMagnitude leftLow < 2 ^ 16)
    (leftMiddleBound : centeredMagnitude leftMiddle < 2 ^ 16)
    (leftHighBound : centeredMagnitude leftHigh < 2 ^ 16)
    (rightLowBound : centeredMagnitude rightLow < 2 ^ 16)
    (rightMiddleBound : centeredMagnitude rightMiddle < 2 ^ 16)
    (rightHighBound : centeredMagnitude rightHigh < 2 ^ 16)
    (equal : packWord leftLow leftMiddle leftHigh =
      packWord rightLow rightMiddle rightHigh) :
    leftLow = rightLow ∧ leftMiddle = rightMiddle ∧ leftHigh = rightHigh := by
  have shiftedEqual :
      packWord (leftLow + packOffset) (leftMiddle + packOffset) (leftHigh + packOffset) =
        packWord (rightLow + packOffset) (rightMiddle + packOffset)
          (rightHigh + packOffset) := by
    rw [packWord_shift, packWord_shift, equal]
  have valuesEqual := congrArg Fin.val shiftedEqual
  have leftLowShift := shifted_val_lt leftLowBound
  have leftMiddleShift := shifted_val_lt leftMiddleBound
  have leftHighShift := shifted_val_lt leftHighBound
  have rightLowShift := shifted_val_lt rightLowBound
  have rightMiddleShift := shifted_val_lt rightMiddleBound
  have rightHighShift := shifted_val_lt rightHighBound
  rw [packWord_val leftLowShift leftMiddleShift leftHighShift,
    packWord_val rightLowShift rightMiddleShift rightHighShift] at valuesEqual
  rcases radix_unique leftLowShift leftMiddleShift rightLowShift
      rightMiddleShift valuesEqual with ⟨low, middle, high⟩
  exact ⟨add_right_cancel (Fin.ext low), add_right_cancel (Fin.ext middle),
    add_right_cancel (Fin.ext high)⟩

private theorem packWord_ofNat (low middle high : Nat) :
    packWord (Poseidon2.ofNat low) (Poseidon2.ofNat middle) (Poseidon2.ofNat high) =
      Poseidon2.ofNat (low + 2 ^ 17 * middle + 2 ^ 17 * 2 ^ 17 * high) := by
  unfold packWord packRadix
  rw [ofNat_mul, ofNat_mul, ofNat_mul, ofNat_add, ofNat_add]

/-- Packing the unpacked lanes returns every word. -/
theorem packWord_unpackWord (word : F) :
    packWord (unpackWord word 0) (unpackWord word 1) (unpackWord word 2) = word := by
  have shifted :
      packWord (unpackWord word 0 + packOffset) (unpackWord word 1 + packOffset)
          (unpackWord word 2 + packOffset) = word + packShift := by
    simp only [unpackWord, sub_add_cancel]
    rw [packWord_ofNat]
    have digits : (word + packShift).val % 2 ^ 17 +
        2 ^ 17 * ((word + packShift).val / 2 ^ 17 % 2 ^ 17) +
        2 ^ 17 * 2 ^ 17 * ((word + packShift).val / 2 ^ 34) =
          (word + packShift).val := by
      omega
    rw [digits, ofNat_val]
  rw [packWord_shift] at shifted
  exact add_right_cancel shifted

private theorem radix_digits {a b c : Nat}
    (aBound : a < 131072) (bBound : b < 131072) (cBound : c < 131072) :
    (a + 131072 * b + 17179869184 * c) % 2 ^ 17 = a ∧
      (a + 131072 * b + 17179869184 * c) / 2 ^ 17 % 2 ^ 17 = b ∧
      (a + 131072 * b + 17179869184 * c) / 2 ^ 34 = c := by
  omega

/-- Unpacking returns every lane of centered magnitude below `2^16`. -/
theorem unpackWord_packWord {low middle high : F}
    (lowBound : centeredMagnitude low < 2 ^ 16)
    (middleBound : centeredMagnitude middle < 2 ^ 16)
    (highBound : centeredMagnitude high < 2 ^ 16) :
    unpackWord (packWord low middle high) 0 = low ∧
      unpackWord (packWord low middle high) 1 = middle ∧
      unpackWord (packWord low middle high) 2 = high := by
  have lowShift := shifted_val_lt lowBound
  have middleShift := shifted_val_lt middleBound
  have highShift := shifted_val_lt highBound
  have value : (packWord low middle high + packShift).val =
      (low + packOffset).val + 131072 * (middle + packOffset).val +
        17179869184 * (high + packOffset).val := by
    rw [← packWord_shift, packWord_val lowShift middleShift highShift]
  have lane (lane : F) : Poseidon2.ofNat (lane + packOffset).val - packOffset = lane := by
    rw [ofNat_val, add_sub_cancel_right]
  rcases radix_digits lowShift middleShift highShift with ⟨lowDigit, middleDigit, highDigit⟩
  refine ⟨?_, ?_, ?_⟩
  · simp only [unpackWord]
    rw [value, lowDigit, lane]
  · simp only [unpackWord]
    rw [value, middleDigit, lane]
  · simp only [unpackWord]
    rw [value, highDigit, lane]

theorem packedColumn_cover
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    ∃ word lane, packedColumn (logicalWidth := logicalWidth) (publicFits := publicFits)
      word lane = column := by
  have columnBound : column.val < 3 * packedParentWords :=
    lt_of_lt_of_eq column.isLt publicWidth_eq
  refine ⟨⟨column.val / 3, by unfold packedParentWords at *; omega⟩,
    ⟨column.val % 3, by omega⟩, ?_⟩
  apply Fin.ext
  simp only [packedColumn]
  omega

/-- Equal packed parents identify the children of two canonical running
instances. -/
theorem serializeParentPublic_injective
    {left right : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftCanonical : ChildrenCanonical left)
    (rightCanonical : ChildrenCanonical right)
    (equal : serializeParentPublic left = serializeParentPublic right) :
    left.publicInputs = right.publicInputs := by
  have parentEqual : ∀ column, parentPublic left column = parentPublic right column := by
    intro column
    rcases packedColumn_cover column with ⟨word, lane, rfl⟩
    have wordEqual := congrArg (fun words => words.getD word.val 0) equal
    simp only [serializeParentPublic] at wordEqual
    rw [List.getD_eq_getElem _ _ (by simp), List.getD_eq_getElem _ _ (by simp)] at wordEqual
    simp only [List.getElem_map, List.getElem_finRange, Fin.eta] at wordEqual
    rcases packWord_injective
        (leftCanonical.parentBounded _) (leftCanonical.parentBounded _)
        (leftCanonical.parentBounded _) (rightCanonical.parentBounded _)
        (rightCanonical.parentBounded _) (rightCanonical.parentBounded _)
        wordEqual with
      ⟨low, middle, high⟩
    fin_cases lane
    · exact low
    · exact middle
    · exact high
  funext source column
  rcases leftCanonical.accepted column with ⟨leftSign, leftAccepted⟩
  rcases rightCanonical.accepted column with ⟨rightSign, rightAccepted⟩
  have leftSplit := leftAccepted.digits_eq_splitScalar
  have rightSplit := rightAccepted.digits_eq_splitScalar
  rw [parentEqual column] at leftSplit
  have digitsEqual : childDigits left column = childDigits right column := by
    rw [leftSplit, rightSplit]
  have selected := congrFun digitsEqual (Fin.cast runningCount_eq_radixChildCount source)
  simpa [childDigits] using selected

/-- The running serializer is injective on canonical running instances. -/
theorem serializeRunning_injective
    {left right : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftCanonical : ChildrenCanonical left)
    (rightCanonical : ChildrenCanonical right)
    (equal : serializeRunning left = serializeRunning right) :
    left = right := by
  unfold serializeRunning at equal
  rcases List.append_inj equal
      (by simp only [serializeRunningFields_length]) with
    ⟨fieldsEqual, parentEqual⟩
  rcases serializeRunningFields_injective fieldsEqual with
    ⟨pointEqual, commitmentsEqual, evaluationsEqual⟩
  exact running_ext pointEqual commitmentsEqual
    (serializeParentPublic_injective leftCanonical rightCanonical parentEqual)
    evaluationsEqual

/-! ## Preimage injectivity -/

/-- Distinct well-formed Stage 1 states have distinct preimages. -/
theorem serializePreimage_injective
    {left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftWellFormed : WellFormed left)
    (rightWellFormed : WellFormed right)
    (encodedEqual :
      serializePreimage (publicFits := publicFits) left =
        serializePreimage (publicFits := publicFits) right) :
    left = right := by
  rcases leftWellFormed with ⟨leftFixed, leftIteration, leftPc, leftCanonical⟩
  rcases rightWellFormed with ⟨rightFixed, rightIteration, rightPc, rightCanonical⟩
  unfold serializePreimage at encodedEqual
  simp only [List.append_assoc] at encodedEqual
  have afterDomain := List.append_cancel_left encodedEqual
  rcases List.append_inj afterDomain
      (by simp only [serializeRunning_length]) with
    ⟨runningEqual, tailEqual⟩
  unfold serializeRunning at runningEqual
  rcases List.append_inj runningEqual
      (by simp only [serializeRunningFields_length]) with
    ⟨fieldsEqual, parentEqual⟩
  rcases serializeRunningFields_injective fieldsEqual with
    ⟨pointEqual, commitmentsEqual, evaluationsEqual⟩
  rcases serializeTail_injective leftFixed rightFixed tailEqual with
    ⟨keyEqual, iterationWord, z0Equal, currentEqual⟩
  exact preimage_ext keyEqual
    (natWord_injective_below_modulus leftIteration rightIteration iterationWord)
    z0Equal currentEqual
    (running_ext pointEqual commitmentsEqual
      (serializeParentPublic_injective leftCanonical rightCanonical parentEqual)
      evaluationsEqual)
    (by rw [leftPc, rightPc])

/-- No canonical preimage is another canonical preimage followed by a
nonempty suffix. This is stronger than the required trailing-zero case. -/
theorem serializePreimage_not_trailing_extension
    {left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftWellFormed : WellFormed left)
    (rightWellFormed : WellFormed right)
    {suffix : List F}
    (suffixNonempty : suffix ≠ []) :
    serializePreimage (publicFits := publicFits) left ≠
      serializePreimage (publicFits := publicFits) right ++ suffix := by
  intro encodedEqual
  have lengthEqual := congrArg List.length encodedEqual
  rw [PilotProduction.serializePreimage_length_fixed left leftWellFormed.1, List.length_append,
    PilotProduction.serializePreimage_length_fixed right rightWellFormed.1] at lengthEqual
  have suffixLength : suffix.length = 0 := by omega
  exact suffixNonempty (List.eq_nil_of_length_eq_zero suffixLength)

/-- In particular, appending one zero word cannot produce a valid encoding. -/
theorem serializePreimage_not_trailing_zero
    {left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftWellFormed : WellFormed left)
    (rightWellFormed : WellFormed right) :
    serializePreimage (publicFits := publicFits) left ≠
      serializePreimage (publicFits := publicFits) right ++ [0] := by
  exact serializePreimage_not_trailing_extension leftWellFormed rightWellFormed
    (by simp)

private theorem serializePreimage_tail
    (left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (encodedEqual : serializePreimage (publicFits := publicFits) left =
      serializePreimage (publicFits := publicFits) right) :
    serializeTail left = serializeTail right := by
  unfold serializePreimage at encodedEqual
  simp only [List.append_assoc] at encodedEqual
  exact (List.append_inj (List.append_cancel_left encodedEqual)
    (by simp only [serializeRunning_length])).2

/-- Equal preimages carry equal running words. -/
theorem serializePreimage_eq_implies_running_eq
    (left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (encodedEqual : serializePreimage (publicFits := publicFits) left =
      serializePreimage (publicFits := publicFits) right) :
    serializeRunning (publicFits := publicFits) (left.running functionIndex) =
      serializeRunning (publicFits := publicFits) (right.running functionIndex) := by
  unfold serializePreimage at encodedEqual
  simp only [List.append_assoc] at encodedEqual
  exact (List.append_inj (List.append_cancel_left encodedEqual)
    (by simp only [serializeRunning_length])).1

/-- Equal preimages identify the context words without requiring
injectivity of the later natural counter encoding. -/
theorem serializePreimage_eq_implies_context_eq
    (left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (lengthEqual : (left.verifierKeys functionIndex).length =
      (right.verifierKeys functionIndex).length)
    (encodedEqual : serializePreimage (publicFits := publicFits) left =
      serializePreimage (publicFits := publicFits) right) :
    left.verifierKeys functionIndex = right.verifierKeys functionIndex := by
  have tailEqual := serializePreimage_tail left right encodedEqual
  unfold serializeTail at tailEqual
  simp only [List.append_assoc] at tailEqual
  exact (List.append_inj tailEqual lengthEqual).1

/-- The iteration word follows the context words in the same fixed frame. -/
theorem serializePreimage_eq_implies_iteration_word_eq
    (left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (lengthEqual : (left.verifierKeys functionIndex).length =
      (right.verifierKeys functionIndex).length)
    (encodedEqual : serializePreimage (publicFits := publicFits) left =
      serializePreimage (publicFits := publicFits) right) :
    natWord left.iteration = natWord right.iteration := by
  have tailEqual := serializePreimage_tail left right encodedEqual
  unfold serializeTail at tailEqual
  simp only [List.append_assoc] at tailEqual
  have afterKey := (List.append_inj tailEqual lengthEqual).2
  simpa using (List.cons.inj afterKey).1

/-- Every state hash has the exact four-word public ABI, independently of
the input length. The proof follows the round structure symbolically. -/
theorem stateHash_length
    (preimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (stateHash preimage).length = 4 := by
  have roundsLength (roundStep : Nat → Poseidon2.State → Poseidon2.State)
      (stepLength : ∀ round state, (roundStep round state).length = Poseidon2.width)
      (rounds : List Nat) (state : Poseidon2.State)
      (stateLength : state.length = Poseidon2.width) :
      (rounds.foldl (fun current round => roundStep round current) state).length =
        Poseidon2.width := by
    induction rounds generalizing state with
    | nil => exact stateLength
    | cons round rest inductionHypothesis =>
        exact inductionHypothesis _ (stepLength round state)
  have permuteLength (state : Poseidon2.State) :
      (Poseidon2.permute state).length = Poseidon2.width := by
    unfold Poseidon2.permute Poseidon2.rounds
    apply roundsLength
    · intro round current
      simp [Poseidon2.fullRound, Poseidon2.externalLayer]
    · apply roundsLength
      · intro round current
        simp [Poseidon2.partialRound, Poseidon2.internalLayer]
      · apply roundsLength
        · intro round current
          simp [Poseidon2.fullRound, Poseidon2.externalLayer]
        · simp [Poseidon2.externalLayer]
  unfold stateHash Poseidon2.hash
  dsimp only
  rw [List.length_take, permuteLength]
  norm_num [Poseidon2.digestLen, Poseidon2.width]

/-- A decoded predecessor below the modulus cannot wrap to a positive,
canonical terminal iteration with the same field word. -/
theorem natWord_successor_eq_below_modulus
    (prior current : Nat)
    (priorBound : prior < goldilocksModulus)
    (currentPositive : 0 < current)
    (currentBound : current < goldilocksModulus)
    (wordEqual : natWord (prior + 1) = natWord current) :
    prior + 1 = current := by
  by_cases nextBound : prior + 1 < goldilocksModulus
  · exact natWord_injective_below_modulus nextBound currentBound wordEqual
  · have atModulus : prior + 1 = goldilocksModulus := by omega
    have valuesEqual := congrArg Fin.val wordEqual
    simp [natWord, Spec.Poseidon2.ofNat, atModulus,
      Nat.mod_eq_of_lt currentBound] at valuesEqual
    omega

end NightstreamFPrime.Layout.Stage1.StateEncoding
