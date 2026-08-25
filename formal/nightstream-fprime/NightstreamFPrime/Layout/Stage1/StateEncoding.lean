import NightstreamFPrime.Layout.Stage1.PiCCSRepresentation

/-!
Owns canonical-decoding facts for the fixed Stage 1 state-hash preimage.

The serializer stores `iteration` as one Goldilocks word. A well-formed
preimage therefore requires the natural-number iteration to be below the
field modulus. Stage 1 has one function, so its one-based program counter is
exactly one. These conditions, together with the fixed four-word context and
application states, make the complete serializer injective.
-/

namespace NightstreamFPrime.Layout.Stage1.StateEncoding

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
  (EvaluationFamily)
open NightstreamFPrime.Layout.Stage1.PiCCSRepresentation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Exact validity conditions for one canonical Stage 1 hash preimage. -/
def WellFormed
    (preimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  PilotProduction.FixedPreimage preimage ∧
    preimage.iteration < goldilocksModulus ∧
    preimage.pc = 1

/-- The `Eval_K` segment of one serialized evaluation family. -/
def evalKWords (evaluations : EvaluationFamily K productionShape) : List F :=
  (List.finRange productionShape.coefficientCount).flatMap
    (fun coefficient => serializeK (evaluations.pad coefficient))

/-- The 14-matrix `Eval_A` segment of one serialized evaluation family. -/
def evalAWords (evaluations : EvaluationFamily K productionShape) : List F :=
  (List.finRange productionShape.matrixCount).flatMap fun matrix =>
    (List.finRange productionShape.coefficientCount).flatMap fun coefficient =>
      serializeK (evaluations.matrix matrix coefficient)

theorem evalKWords_length (evaluations : EvaluationFamily K productionShape) :
    (evalKWords evaluations).length = 108 := by
  simp [evalKWords, productionShape, Phi81MatrixSource.phi81Shape, ringDegree]

theorem evalAWords_length (evaluations : EvaluationFamily K productionShape) :
    (evalAWords evaluations).length = 1512 := by
  simp [evalAWords, productionShape, Phi81MatrixSource.phi81Shape,
    productionProfile, ringDegree]

/-- The digest preimage has a proved boundary between `Eval_K` and all
separate `Eval_A` values. -/
theorem serializeEvaluations_eq_evalK_append_evalA
    (evaluations : EvaluationFamily K productionShape) :
    serializeEvaluations evaluations =
      evalKWords evaluations ++ evalAWords evaluations := by
  rfl

theorem serializeEvaluations_take_evalK
    (evaluations : EvaluationFamily K productionShape) :
    (serializeEvaluations evaluations).take 108 = evalKWords evaluations := by
  rw [serializeEvaluations_eq_evalK_append_evalA]
  simp [evalKWords_length]

theorem serializeEvaluations_drop_evalK
    (evaluations : EvaluationFamily K productionShape) :
    (serializeEvaluations evaluations).drop 108 = evalAWords evaluations := by
  rw [serializeEvaluations_eq_evalK_append_evalA]
  simp [evalKWords_length]

private theorem split_append
    {leftHead rightHead leftTail rightTail : List F}
    (lengthEqual : leftHead.length = rightHead.length)
    (encodedEqual : leftHead ++ leftTail = rightHead ++ rightTail) :
    leftHead = rightHead ∧ leftTail = rightTail := by
  constructor
  · have selected := congrArg (List.take leftHead.length) encodedEqual
    simpa [lengthEqual] using selected
  · have selected := congrArg (List.drop leftHead.length) encodedEqual
    simpa [lengthEqual] using selected

private theorem block_injective : Function.Injective block := by
  intro left right equal
  unfold block at equal
  exact (List.cons.inj equal).2

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

/-- Distinct well-formed Stage 1 states have distinct canonical encodings. -/
theorem serializePreimage_injective
    {left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftWellFormed : WellFormed left)
    (rightWellFormed : WellFormed right)
    (encodedEqual :
      serializePreimage (publicFits := publicFits) left =
        serializePreimage (publicFits := publicFits) right) :
    left = right := by
  rcases leftWellFormed with
    ⟨leftFixed, leftIterationBound, leftPc⟩
  rcases rightWellFormed with
    ⟨rightFixed, rightIterationBound, rightPc⟩
  rcases leftFixed with ⟨leftKeyLength, leftZ0Length, leftCurrentLength⟩
  rcases rightFixed with
    ⟨rightKeyLength, rightZ0Length, rightCurrentLength⟩
  simp only [serializePreimage, List.append_assoc] at encodedEqual
  have afterTag := List.append_cancel_left encodedEqual
  have keyBlockLength :
      (block (left.verifierKeys functionIndex)).length =
        (block (right.verifierKeys functionIndex)).length := by
    simp [leftKeyLength, rightKeyLength]
  rcases split_append
      (leftHead := block (left.verifierKeys functionIndex))
      (rightHead := block (right.verifierKeys functionIndex))
      keyBlockLength afterTag with
    ⟨keyBlockEqual, afterKey⟩
  have keyEqual :
      left.verifierKeys functionIndex = right.verifierKeys functionIndex :=
    block_injective keyBlockEqual
  rcases split_append (show ([natWord left.iteration] : List F).length =
      [natWord right.iteration].length by rfl) afterKey with
    ⟨iterationBlockEqual, afterIteration⟩
  have iterationWordEqual :
      natWord left.iteration = natWord right.iteration := by
    simpa using iterationBlockEqual
  have iterationEqual : left.iteration = right.iteration :=
    natWord_injective_below_modulus leftIterationBound rightIterationBound
      iterationWordEqual
  have z0BlockLength :
      (block left.z0).length = (block right.z0).length := by
    simp [leftZ0Length, rightZ0Length]
  rcases split_append z0BlockLength afterIteration with
    ⟨z0BlockEqual, afterZ0⟩
  have z0Equal : left.z0 = right.z0 := block_injective z0BlockEqual
  have currentBlockLength :
      (block left.current).length = (block right.current).length := by
    simp [leftCurrentLength, rightCurrentLength]
  rcases split_append currentBlockLength afterZ0 with
    ⟨currentBlockEqual, afterCurrent⟩
  have currentEqual : left.current = right.current :=
    block_injective currentBlockEqual
  have runningLength :
      (serializeRunning (publicFits := publicFits)
          (left.running functionIndex)).length =
        (serializeRunning (publicFits := publicFits)
          (right.running functionIndex)).length := by
    rw [serializeRunning_length, serializeRunning_length]
  rcases split_append runningLength afterCurrent with
    ⟨runningEncodingEqual, _afterRunning⟩
  have runningAtIndexEqual :
      left.running functionIndex = right.running functionIndex :=
    serializeRunning_injective runningEncodingEqual
  have verifierKeysEqual : left.verifierKeys = right.verifierKeys := by
    funext index
    rw [fin_slot_eq_functionIndex index]
    exact keyEqual
  have runningEqual : left.running = right.running := by
    funext index
    rw [fin_slot_eq_functionIndex index]
    exact runningAtIndexEqual
  cases left
  cases right
  simp_all

/-- No canonical state encoding is another canonical state encoding followed
by a nonempty suffix. This is stronger than the required trailing-zero case. -/
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
  have leftLength := PilotProduction.serializePreimage_length_fixed left
    leftWellFormed.1
  have rightLength := PilotProduction.serializePreimage_length_fixed right
    rightWellFormed.1
  rw [leftLength, List.length_append, rightLength] at lengthEqual
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

end NightstreamFPrime.Layout.Stage1.StateEncoding
