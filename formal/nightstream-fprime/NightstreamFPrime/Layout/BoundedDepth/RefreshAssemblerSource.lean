import NightstreamFPrime.Layout.BoundedDepth.RefreshAssembler
import NightstreamFPrime.Layout.BoundedDepth.LightAssemblerSource

/-! Values for the refresh step's source columns (CW-3.8). The columns below the children's
commitments are the light step's (`LightAssemblerSource.env`): counters, key digests, states,
witness, slots, digests and the fresh instance. After them come the children's commitments and
evaluation families, the digit words, the rounds and the Π_CCS output of all 17 sources, each
decoded from its position in its group (`tailColumn`). The readback theorems give the value of
each of these shared words on an environment that holds the columns. -/

namespace NightstreamFPrime.Layout.BoundedDepth.RefreshAssemblerSource

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
open NightstreamFPrime.Layout.BoundedDepth.LightAssembler (kAt slotAt slotWordCount keysStart
  statesStart witnessStart slotsStart digestsStart freshStart)
open NightstreamFPrime.Layout.BoundedDepth.LightAssemblerSource (finAt finAt_eq finAt_val kColumn
  kAt_eval kColumn_low kColumn_high ofFn_getD roundColumn)
open NightstreamFPrime.Layout.BoundedDepth.RefreshAssembler

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-- The 16 children's commitments, 1,188 words each. -/
def childCommitmentColumn (split : Spec.Folding.Refresh.Split PaperAlgebra.Commitment
    productionShape) (word : Nat) : F :=
  split.commitments (finAt 16 (by decide) (word / 1188))
    (finAt 22 (by decide) (word % 1188 / 54)) (finAt 54 (by decide) word)

/-- The 16 children's evaluation families, 270 values of `K` each: the pad family, then the
four matrix families. -/
def childEvaluationColumn (split : Spec.Folding.Refresh.Split PaperAlgebra.Commitment
    productionShape) (word : Nat) : F :=
  kColumn (fun index =>
    if index % 270 < 54 then
      (split.evaluations (finAt 16 (by decide) (index / 270))).pad
        (finAt 54 (by decide) (index % 270))
    else
      (split.evaluations (finAt 16 (by decide) (index / 270))).matrix
        (finAt 4 (by decide) ((index % 270 - 54) / 54)) (finAt 54 (by decide) (index % 270 - 54)))
    word

/-- The 16 children's digit words, 270 each. -/
def digitColumn (digits : Fin productionShape.runningCount →
    PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (word : Nat) : F :=
  digits (finAt 16 (by decide) (word / 270)) (finAt 270 (by decide) word)

/-- The Π_CCS output of all 17 sources: per source 54 pad and 216 matrix coordinates. -/
def outputColumn (output : FullOutputCoordinates.FullOutput K productionShape) (word : Nat) : F :=
  kColumn (fun index =>
    if index % 270 < 54 then
      output.padCoordinate (finAt 17 (by decide) (index / 270)) (finAt 54 (by decide) (index % 270))
    else
      output.matrixCoordinate (finAt 17 (by decide) (index / 270))
        (finAt 4 (by decide) ((index % 270 - 54) / 54)) (finAt 54 (by decide) (index % 270 - 54)))
    word

/-- One value for each group of the refresh step's shared words. -/
structure Values (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  counters : List F
  keys : Fin 2 → List F
  states : Fin 3 → List F
  witness : List F
  slots : Fin 7 → SlotData logicalWidth publicFits
  digests : Fin 5 → List F
  fresh : StepFresh logicalWidth publicFits
  split : Spec.Folding.Refresh.Split PaperAlgebra.Commitment productionShape
  digits : Fin productionShape.runningCount →
    PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)
  rounds : Fin productionShape.cubeVariables → SumCheck.Finite.FixedPolynomial K 8
  output : FullOutputCoordinates.FullOutput K productionShape
  digestSelect : Fin 3 → List F

/-- The light step's values for the shared groups; the light rounds, output and digest select lie
above the columns that the refresh step reads from them. -/
def lightValues (values : Values logicalWidth publicFits) :
    LightAssemblerSource.Values logicalWidth publicFits where
  counters := values.counters
  keys := values.keys
  states := values.states
  witness := values.witness
  slots := values.slots
  digests := values.digests
  fresh := values.fresh
  rounds := values.rounds
  output := ⟨fun _ _ => ⟨0, 0⟩, fun _ _ _ => ⟨0, 0⟩⟩
  digestSelect := fun _ => []

/-- The refresh step's own groups, from the children's commitments. -/
def tailColumn (values : Values logicalWidth publicFits) (word : Nat) : F :=
  if word < 19008 then childCommitmentColumn values.split word
  else if word < 27648 then childEvaluationColumn values.split (word - 19008)
  else if word < 31968 then digitColumn values.digits (word - 27648)
  else if word < 32472 then roundColumn values.rounds (word - 31968)
  else if word < 41652 then outputColumn values.output (word - 32472)
  else LightAssemblerSource.quadColumn (by decide) values.digestSelect (word - 41652)

/-- The source columns from `base`, in the layout of `RefreshAssembler.interface`. -/
def env (base witnessWordCount : Nat) (values : Values logicalWidth publicFits) : Env :=
  fun column =>
    if column < childCommitmentsStart base witnessWordCount then
      LightAssemblerSource.env base witnessWordCount (lightValues values) column
    else tailColumn values (column - childCommitmentsStart base witnessWordCount)

variable (base witnessWordCount : Nat) (values : Values logicalWidth publicFits)

theorem env_light (column : Nat) (bound : column < childCommitmentsStart base witnessWordCount) :
    env base witnessWordCount values column =
      LightAssemblerSource.env base witnessWordCount (lightValues values) column := by
  simp only [env, bound, ↓reduceIte]

theorem env_tail (word : Nat) :
    env base witnessWordCount values (childCommitmentsStart base witnessWordCount + word) =
      tailColumn values word := by
  simp only [env, show ¬childCommitmentsStart base witnessWordCount + word <
    childCommitmentsStart base witnessWordCount by omega, ↓reduceIte, Nat.add_sub_cancel_left]

theorem tail_commitment (word : Nat) (bound : word < 19008) :
    tailColumn values word = childCommitmentColumn values.split word := by
  simp only [tailColumn, bound, ↓reduceIte]

theorem tail_evaluation (word : Nat) (bound : word < 8640) :
    tailColumn values (19008 + word) = childEvaluationColumn values.split word := by
  simp only [tailColumn, show ¬19008 + word < 19008 by omega,
    show 19008 + word < 27648 by omega, ↓reduceIte, Nat.add_sub_cancel_left]

theorem tail_digit (word : Nat) (bound : word < 4320) :
    tailColumn values (27648 + word) = digitColumn values.digits word := by
  simp only [tailColumn, show ¬27648 + word < 19008 by omega,
    show ¬27648 + word < 27648 by omega, show 27648 + word < 31968 by omega, ↓reduceIte,
    Nat.add_sub_cancel_left]

theorem tail_round (word : Nat) (bound : word < 504) :
    tailColumn values (31968 + word) = roundColumn values.rounds word := by
  simp only [tailColumn, show ¬31968 + word < 19008 by omega,
    show ¬31968 + word < 27648 by omega, show ¬31968 + word < 31968 by omega,
    show 31968 + word < 32472 by omega, ↓reduceIte, Nat.add_sub_cancel_left]

theorem tail_output (word : Nat) (bound : word < 9180) :
    tailColumn values (32472 + word) = outputColumn values.output word := by
  simp only [tailColumn, show ¬32472 + word < 19008 by omega,
    show ¬32472 + word < 27648 by omega, show ¬32472 + word < 31968 by omega,
    show ¬32472 + word < 32472 by omega, show 32472 + word < 41652 by omega, ↓reduceIte,
    Nat.add_sub_cancel_left]

theorem tail_digestSelect (word : Nat) :
    tailColumn values (41652 + word) =
      LightAssemblerSource.quadColumn (by decide) values.digestSelect word := by
  simp only [tailColumn, show ¬41652 + word < 19008 by omega,
    show ¬41652 + word < 27648 by omega, show ¬41652 + word < 31968 by omega,
    show ¬41652 + word < 32472 by omega, show ¬41652 + word < 41652 by omega, ↓reduceIte,
    Nat.add_sub_cancel_left]

section Readback

variable {base witnessWordCount} (env : Env)

private theorem fixedPolynomial_ext {left right : SumCheck.Finite.FixedPolynomial K 8}
    (same : left.coefficients = right.coefficients) : left = right := by
  cases left
  cases right
  simp_all

/-- The children's commitment words carry the split's commitments whose columns `env` holds. -/
theorem childCommitment_eval (split : Spec.Folding.Refresh.Split PaperAlgebra.Commitment
      productionShape)
    (columns : ∀ word, word < 19008 →
      env (childCommitmentsStart base witnessWordCount + word) = childCommitmentColumn split word)
    (child : Fin productionShape.runningCount) (row : Fin productionProfile.commitmentWidth)
    (lane : Fin ringDegree) :
    ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        witnessWordCount).childCommitment child row lane).eval env =
      split.commitments child row lane := by
  have childBound := child.isLt
  have rowBound := row.isLt
  have laneBound := lane.isLt
  change child.val < 16 at childBound
  change row.val < 22 at rowBound
  change lane.val < 54 at laneBound
  change env (childCommitmentsStart base witnessWordCount + child.val * 1188 +
    row.val * ringDegree + lane.val) = _
  rw [show childCommitmentsStart base witnessWordCount + child.val * 1188 + row.val * ringDegree +
      lane.val = childCommitmentsStart base witnessWordCount +
        (child.val * 1188 + row.val * 54 + lane.val) by simp only [ringDegree]; omega,
    columns _ (by omega)]
  unfold childCommitmentColumn
  exact congrArg₂ (fun first second => split.commitments first second _)
    (Fin.ext (by rw [finAt_val]; omega)) (Fin.ext (by rw [finAt_val]; omega)) |>.trans
    (congrArg (split.commitments child row) (Fin.ext (by rw [finAt_val]; omega)))

/-- The children's evaluation words carry the split's evaluation families whose columns `env`
holds. -/
theorem childEvaluation_eval (split : Spec.Folding.Refresh.Split PaperAlgebra.Commitment
      productionShape)
    (columns : ∀ word, word < 8640 →
      env (childEvaluationsStart base witnessWordCount + word) =
        childEvaluationColumn split word)
    (child : Fin productionShape.runningCount) :
    StatementAbsorption.evalEvaluation
        ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          witnessWordCount).childEvaluation child) env =
      split.evaluations child := by
  have childBound := child.isLt
  change child.val < 16 at childBound
  unfold StatementAbsorption.evalEvaluation
  congr 1
  · funext coefficient
    have coefficientBound := coefficient.isLt
    change coefficient.val < 54 at coefficientBound
    have position : (child.val * 270 + coefficient.val) % 270 = coefficient.val := by omega
    apply kAt_eval
    · rw [show childEvaluationsStart base witnessWordCount + child.val * 540 +
          2 * coefficient.val = childEvaluationsStart base witnessWordCount +
            2 * (child.val * 270 + coefficient.val) by omega,
        columns _ (by omega)]
      unfold childEvaluationColumn
      rw [kColumn_low]
      simp only [position, show coefficient.val < 54 from coefficientBound, ↓reduceIte]
      exact congrArg₂ (fun first second => ((split.evaluations first).pad second).c0)
        (Fin.ext (by rw [finAt_val]; omega)) (Fin.ext (by rw [finAt_val]; omega))
    · rw [show childEvaluationsStart base witnessWordCount + child.val * 540 +
          2 * coefficient.val + 1 = childEvaluationsStart base witnessWordCount +
            (2 * (child.val * 270 + coefficient.val) + 1) by omega,
        columns _ (by omega)]
      unfold childEvaluationColumn
      rw [kColumn_high]
      simp only [position, show coefficient.val < 54 from coefficientBound, ↓reduceIte]
      exact congrArg₂ (fun first second => ((split.evaluations first).pad second).c1)
        (Fin.ext (by rw [finAt_val]; omega)) (Fin.ext (by rw [finAt_val]; omega))
  · funext matrix coefficient
    have matrixBound := matrix.isLt
    have coefficientBound := coefficient.isLt
    change matrix.val < 4 at matrixBound
    change coefficient.val < 54 at coefficientBound
    have position : (child.val * 270 + 54 + matrix.val * 54 + coefficient.val) % 270 =
        54 + matrix.val * 54 + coefficient.val := by omega
    have notPad : ¬54 + matrix.val * 54 + coefficient.val < 54 := by omega
    have value := fun (part : K → F) =>
      congrArg₂ (fun first second => part ((split.evaluations first).matrix second
          (finAt 54 (by decide) (54 + matrix.val * 54 + coefficient.val - 54))))
        (show finAt 16 (by decide) ((child.val * 270 + 54 + matrix.val * 54 + coefficient.val) /
          270) = child from Fin.ext (by rw [finAt_val]; omega))
        (show finAt 4 (by decide) ((54 + matrix.val * 54 + coefficient.val - 54) / 54) = matrix
          from Fin.ext (by rw [finAt_val]; omega)) |>.trans
        (congrArg (fun third => part ((split.evaluations child).matrix matrix third))
          (show finAt 54 (by decide) (54 + matrix.val * 54 + coefficient.val - 54) = coefficient
            from Fin.ext (by rw [finAt_val]; omega)))
    apply kAt_eval
    · rw [show childEvaluationsStart base witnessWordCount + child.val * 540 + 108 +
          2 * (matrix.val * 54 + coefficient.val) = childEvaluationsStart base witnessWordCount +
            2 * (child.val * 270 + 54 + matrix.val * 54 + coefficient.val) by omega,
        columns _ (by omega)]
      unfold childEvaluationColumn
      rw [kColumn_low]
      simp only [position, notPad, ↓reduceIte]
      exact value K.c0
    · rw [show childEvaluationsStart base witnessWordCount + child.val * 540 + 108 +
          2 * (matrix.val * 54 + coefficient.val) + 1 =
            childEvaluationsStart base witnessWordCount +
              (2 * (child.val * 270 + 54 + matrix.val * 54 + coefficient.val) + 1) by omega,
        columns _ (by omega)]
      unfold childEvaluationColumn
      rw [kColumn_high]
      simp only [position, notPad, ↓reduceIte]
      exact value K.c1

/-- The digit words carry the digits whose columns `env` holds. -/
theorem digit_eval (digits : Fin productionShape.runningCount →
      PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (columns : ∀ word, word < 4320 →
      env (digitsStart base witnessWordCount + word) = digitColumn digits word)
    (child : Fin productionShape.runningCount)
    (column : Fin (PaperAlgebra.FullShape logicalWidth publicFits).publicWidth) :
    ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        witnessWordCount).digit child column).eval env = digits child column := by
  have childBound := child.isLt
  have columnBound := column.isLt
  change child.val < 16 at childBound
  change column.val < 270 at columnBound
  change env (digitsStart base witnessWordCount + child.val * 270 + column.val) = _
  rw [Nat.add_assoc, columns _ (by omega)]
  unfold digitColumn
  exact congrArg₂ digits (Fin.ext (by rw [finAt_val]; omega))
    (Fin.ext (by rw [finAt_val]; omega))

/-- The round words carry the round polynomials whose columns `env` holds. -/
theorem round_eval (rounds : Fin productionShape.cubeVariables →
      SumCheck.Finite.FixedPolynomial K 8)
    (columns : ∀ word, word < 504 →
      env (roundsStart base witnessWordCount + word) = roundColumn rounds word)
    (roundIndex : Fin productionShape.cubeVariables) :
    ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        witnessWordCount).round roundIndex).semanticPolynomial env = rounds roundIndex := by
  have roundBound := roundIndex.isLt
  change roundIndex.val < 28 at roundBound
  apply fixedPolynomial_ext
  change (List.ofFn fun coefficient : Fin 9 => kAt (roundsStart base witnessWordCount +
    2 * (roundIndex.val * 9 + coefficient.val))).map (KExpr.eval env) = _
  rw [List.map_ofFn, ← ofFn_getD ⟨0, 0⟩ (rounds roundIndex).coefficients
    (rounds roundIndex).coefficients_length]
  apply congrArg List.ofFn
  funext coefficient
  have coefficientBound := coefficient.isLt
  apply kAt_eval
  · rw [columns (2 * (roundIndex.val * 9 + coefficient.val)) (by omega)]
    unfold roundColumn
    rw [kColumn_low, show (roundIndex.val * 9 + coefficient.val) % 9 = coefficient.val by omega]
    exact congrArg (fun index => ((rounds index).coefficients.getD coefficient.val ⟨0, 0⟩).c0)
      (Fin.ext (by rw [finAt_val]; omega))
  · rw [Nat.add_assoc, columns (2 * (roundIndex.val * 9 + coefficient.val) + 1) (by omega)]
    unfold roundColumn
    rw [kColumn_high, show (roundIndex.val * 9 + coefficient.val) % 9 = coefficient.val by omega]
    exact congrArg (fun index => ((rounds index).coefficients.getD coefficient.val ⟨0, 0⟩).c1)
      (Fin.ext (by rw [finAt_val]; omega))

/-- The output words carry the Π_CCS output whose columns `env` holds. -/
theorem outputPad_eval (output : FullOutputCoordinates.FullOutput K productionShape)
    (columns : ∀ word, word < 9180 →
      env (outputStart base witnessWordCount + word) = outputColumn output word)
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) :
    ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        witnessWordCount).output.padCoordinate source coefficient).eval env =
      output.padCoordinate source coefficient := by
  have sourceBound := source.isLt
  have coefficientBound := coefficient.isLt
  change source.val < 17 at sourceBound
  change coefficient.val < 54 at coefficientBound
  have position : (source.val * 270 + coefficient.val) % 270 = coefficient.val := by omega
  apply kAt_eval
  · rw [columns (2 * (source.val * 270 + coefficient.val)) (by omega)]
    unfold outputColumn
    rw [kColumn_low]
    simp only [position, show coefficient.val < 54 from coefficientBound, ↓reduceIte]
    exact congrArg₂ (fun first second => (output.padCoordinate first second).c0)
      (Fin.ext (by rw [finAt_val]; omega)) (Fin.ext (by rw [finAt_val]; omega))
  · rw [Nat.add_assoc, columns (2 * (source.val * 270 + coefficient.val) + 1) (by omega)]
    unfold outputColumn
    rw [kColumn_high]
    simp only [position, show coefficient.val < 54 from coefficientBound, ↓reduceIte]
    exact congrArg₂ (fun first second => (output.padCoordinate first second).c1)
      (Fin.ext (by rw [finAt_val]; omega)) (Fin.ext (by rw [finAt_val]; omega))

theorem outputMatrix_eval (output : FullOutputCoordinates.FullOutput K productionShape)
    (columns : ∀ word, word < 9180 →
      env (outputStart base witnessWordCount + word) = outputColumn output word)
    (source : Fin productionShape.sourceCount) (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        witnessWordCount).output.matrixCoordinate source matrix coefficient).eval env =
      output.matrixCoordinate source matrix coefficient := by
  have sourceBound := source.isLt
  have matrixBound := matrix.isLt
  have coefficientBound := coefficient.isLt
  change source.val < 17 at sourceBound
  change matrix.val < 4 at matrixBound
  change coefficient.val < 54 at coefficientBound
  have position : (source.val * 270 + 54 + matrix.val * 54 + coefficient.val) % 270 =
      54 + matrix.val * 54 + coefficient.val := by omega
  have notPad : ¬54 + matrix.val * 54 + coefficient.val < 54 := by omega
  have value := fun (part : K → F) =>
    congrArg₂ (fun first second => part (output.matrixCoordinate first second
        (finAt 54 (by decide) (54 + matrix.val * 54 + coefficient.val - 54))))
      (show finAt 17 (by decide) ((source.val * 270 + 54 + matrix.val * 54 + coefficient.val) /
        270) = source from Fin.ext (by rw [finAt_val]; omega))
      (show finAt 4 (by decide) ((54 + matrix.val * 54 + coefficient.val - 54) / 54) = matrix
        from Fin.ext (by rw [finAt_val]; omega)) |>.trans
      (congrArg (fun third => part (output.matrixCoordinate source matrix third))
        (show finAt 54 (by decide) (54 + matrix.val * 54 + coefficient.val - 54) = coefficient
          from Fin.ext (by rw [finAt_val]; omega)))
  apply kAt_eval
  · rw [columns (2 * (source.val * 270 + 54 + matrix.val * 54 + coefficient.val)) (by omega)]
    unfold outputColumn
    rw [kColumn_low]
    simp only [position, notPad, ↓reduceIte]
    exact value K.c0
  · rw [Nat.add_assoc,
      columns (2 * (source.val * 270 + 54 + matrix.val * 54 + coefficient.val) + 1) (by omega)]
    unfold outputColumn
    rw [kColumn_high]
    simp only [position, notPad, ↓reduceIte]
    exact value K.c1

end Readback

section Honest

open NightstreamFPrime.Spec.HyperNova.Construction2

variable (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
  (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
  (selected : Fin 2) (folded : SlotData logicalWidth publicFits)

/-- The honest values of a refresh step: the light step's shared groups (`folded` is the
selected slot's new claim at depth 0), the proof's children, the digit words that split the
selected slot's public input, and the proof's rounds and Π_CCS output. -/
def honestValues : Values logicalWidth publicFits where
  counters := [natWord input.iteration, Hint.inverse (natWord input.iteration),
    if input.iteration = 0 then 0 else 1, natWord input.priorPc, natWord selected.val]
  keys := fun key => (circuitKeys key).digest
  states := fun index =>
    if index.val = 0 then input.z0 else if index.val = 1 then input.zi else output.zNext
  witness := input.witness
  slots := fun slot =>
    if slot.val < 2 then input.running (finAt 2 (by decide) slot.val)
    else if slot.val = 2 then input.running selected
    else if slot.val < 5 then
      (if slot.val - 3 = selected.val then folded
        else input.running (finAt 2 (by decide) (slot.val - 3)))
    else output.runningNext (finAt 2 (by decide) (slot.val - 5))
  digests := fun index =>
    if index.val < 2 then slotDigest (input.running (finAt 2 (by decide) index.val))
    else if index.val < 4 then slotDigest (output.runningNext (finAt 2 (by decide) (index.val - 2)))
    else output.x
  fresh := input.fresh
  split := proof.split
  digits := fun child => (PaperAlgebra.publicInputSplit ajtai).split
    (input.running selected).1.publicInput (RefreshFold.outputIndex child)
  rounds := proof.nifs.piCcsRounds
  output := proof.nifs.piCcsOutput
  digestSelect := fun index =>
    if index.val = 0 then slotDigest folded
    else if index.val - 1 = selected.val then slotDigest folded
    else slotDigest (input.running (finAt 2 (by decide) (index.val - 1)))

/-- The honest source columns. -/
abbrev honestEnv (base witnessWordCount : Nat) : Env :=
  env base witnessWordCount (honestValues ajtai circuitKeys input output proof selected folded)

omit ajtai circuitKeys input output proof selected folded in
theorem childStart_eq (base witnessWordCount : Nat) :
    childCommitmentsStart base witnessWordCount = base + witnessWordCount + 15888 := by
  unfold childCommitmentsStart freshStart digestsStart slotsStart witnessStart slotWordCount
  omega

variable (base witnessWordCount : Nat) (target : Env)
  (agree : ∀ column, column < sharedEnd base witnessWordCount →
    target column = honestEnv ajtai circuitKeys input output proof selected folded base
      witnessWordCount column)

include agree

/-- Below the children's commitments the target holds the light step's honest columns. -/
theorem target_light (column : Nat) (bound : column < childCommitmentsStart base witnessWordCount) :
    target column = LightAssemblerSource.env base witnessWordCount
      (lightValues (honestValues ajtai circuitKeys input output proof selected folded)) column :=
  (agree column (by
    have := childStart_eq base witnessWordCount
    have := sharedEnd_eq base witnessWordCount
    omega)).trans (env_light base witnessWordCount _ column bound)

theorem target_tail (word : Nat) (bound : word < 41664) :
    target (childCommitmentsStart base witnessWordCount + word) =
      tailColumn (honestValues ajtai circuitKeys input output proof selected folded) word :=
  (agree _ (by
    have := childStart_eq base witnessWordCount
    have := sharedEnd_eq base witnessWordCount
    omega)).trans (env_tail base witnessWordCount _ word)

theorem target_counter (word : Nat) (bound : word < 5) :
    target (base + word) =
      (honestValues ajtai circuitKeys input output proof selected folded).counters.getD word 0 :=
  (target_light ajtai circuitKeys input output proof selected folded base witnessWordCount target
    agree _ (by have := childStart_eq base witnessWordCount; omega)).trans
    (LightAssemblerSource.env_counter base witnessWordCount _ word bound)

theorem target_key (word : Nat) (bound : word < 8) :
    target (keysStart base + word) =
      LightAssemblerSource.quadColumn (by decide)
        (honestValues ajtai circuitKeys input output proof selected folded).keys word :=
  (target_light ajtai circuitKeys input output proof selected folded base witnessWordCount target
    agree _ (by have := childStart_eq base witnessWordCount; unfold keysStart; omega)).trans
    (LightAssemblerSource.env_key base witnessWordCount _ word bound)

theorem target_state (word : Nat) (bound : word < 12) :
    target (statesStart base + word) =
      LightAssemblerSource.quadColumn (by decide)
        (honestValues ajtai circuitKeys input output proof selected folded).states word :=
  (target_light ajtai circuitKeys input output proof selected folded base witnessWordCount target
    agree _ (by have := childStart_eq base witnessWordCount; unfold statesStart; omega)).trans
    (LightAssemblerSource.env_state base witnessWordCount _ word bound)

theorem target_witness (word : Nat) (bound : word < witnessWordCount) :
    target (witnessStart base + word) = input.witness.getD word 0 :=
  (target_light ajtai circuitKeys input output proof selected folded base witnessWordCount target
    agree _ (by have := childStart_eq base witnessWordCount; unfold witnessStart; omega)).trans
    (LightAssemblerSource.env_witness base witnessWordCount _ word bound)

theorem target_slot (slot : Fin 7) (word : Nat) (bound : word < slotWordCount) :
    target (slotsStart base witnessWordCount + slot.val * slotWordCount + word) =
      LightAssemblerSource.slotColumn
        ((honestValues ajtai circuitKeys input output proof selected folded).slots slot) word :=
  (target_light ajtai circuitKeys input output proof selected folded base witnessWordCount target
    agree _ (by
      have := childStart_eq base witnessWordCount
      have slotBound := slot.isLt
      unfold slotsStart witnessStart
      simp only [slotWordCount] at bound ⊢
      omega)).trans (LightAssemblerSource.env_slot base witnessWordCount _ slot word bound)

theorem target_digest (word : Nat) (bound : word < 20) :
    target (digestsStart base witnessWordCount + word) =
      LightAssemblerSource.quadColumn (by decide)
        (honestValues ajtai circuitKeys input output proof selected folded).digests word :=
  (target_light ajtai circuitKeys input output proof selected folded base witnessWordCount target
    agree _ (by
      have := childStart_eq base witnessWordCount
      unfold digestsStart slotsStart witnessStart slotWordCount
      omega)).trans (LightAssemblerSource.env_digest base witnessWordCount _ word bound)

theorem target_fresh (word : Nat) (bound : word < 1458) :
    target (freshStart base witnessWordCount + word) =
      LightAssemblerSource.freshColumn input.fresh word :=
  (target_light ajtai circuitKeys input output proof selected folded base witnessWordCount target
    agree _ (by
      have := childStart_eq base witnessWordCount
      unfold freshStart digestsStart slotsStart witnessStart slotWordCount
      omega)).trans (LightAssemblerSource.env_fresh base witnessWordCount _ word bound)

theorem target_childCommitment (word : Nat) (bound : word < 19008) :
    target (childCommitmentsStart base witnessWordCount + word) =
      childCommitmentColumn proof.split word :=
  (target_tail ajtai circuitKeys input output proof selected folded base witnessWordCount target
    agree word (by omega)).trans (tail_commitment _ word bound)

theorem target_childEvaluation (word : Nat) (bound : word < 8640) :
    target (childEvaluationsStart base witnessWordCount + word) =
      childEvaluationColumn proof.split word := by
  rw [show childEvaluationsStart base witnessWordCount + word =
    childCommitmentsStart base witnessWordCount + (19008 + word) by
      unfold childEvaluationsStart; omega]
  exact (target_tail ajtai circuitKeys input output proof selected folded base witnessWordCount
    target agree _ (by omega)).trans (tail_evaluation _ word bound)

theorem target_digit (word : Nat) (bound : word < 4320) :
    target (digitsStart base witnessWordCount + word) =
      digitColumn (honestValues ajtai circuitKeys input output proof selected folded).digits
        word := by
  rw [show digitsStart base witnessWordCount + word =
    childCommitmentsStart base witnessWordCount + (27648 + word) by
      unfold digitsStart childEvaluationsStart; omega]
  exact (target_tail ajtai circuitKeys input output proof selected folded base witnessWordCount
    target agree _ (by omega)).trans (tail_digit _ word bound)

theorem target_round (word : Nat) (bound : word < 504) :
    target (roundsStart base witnessWordCount + word) =
      roundColumn proof.nifs.piCcsRounds word := by
  rw [show roundsStart base witnessWordCount + word =
    childCommitmentsStart base witnessWordCount + (31968 + word) by
      unfold roundsStart digitsStart childEvaluationsStart; omega]
  exact (target_tail ajtai circuitKeys input output proof selected folded base witnessWordCount
    target agree _ (by omega)).trans (tail_round _ word bound)

theorem target_output (word : Nat) (bound : word < 9180) :
    target (outputStart base witnessWordCount + word) =
      outputColumn proof.nifs.piCcsOutput word := by
  rw [show outputStart base witnessWordCount + word =
    childCommitmentsStart base witnessWordCount + (32472 + word) by
      unfold outputStart roundsStart digitsStart childEvaluationsStart; omega]
  exact (target_tail ajtai circuitKeys input output proof selected folded base witnessWordCount
    target agree _ (by omega)).trans (tail_output _ word bound)

theorem target_digestSelect (word : Nat) (bound : word < 12) :
    target (digestSelectStart base witnessWordCount + word) =
      LightAssemblerSource.quadColumn (by decide)
        (honestValues ajtai circuitKeys input output proof selected folded).digestSelect word := by
  rw [show digestSelectStart base witnessWordCount + word =
    childCommitmentsStart base witnessWordCount + (41652 + word) by
      unfold digestSelectStart outputStart roundsStart digitsStart childEvaluationsStart; omega]
  exact (target_tail ajtai circuitKeys input output proof selected folded base witnessWordCount
    target agree _ (by omega)).trans (tail_digestSelect _ word)

/-- The words of slot group `slot` carry the honest slot of that group. -/
theorem target_slot_words (slot : Fin 7)
    (depth : ((honestValues ajtai circuitKeys input output proof selected folded).slots slot).2 <
      fieldModulus) :
    Hash.evalList target (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        (slotsStart base witnessWordCount + slot.val * slotWordCount)).words =
      serializeSlot ((honestValues ajtai circuitKeys input output proof selected folded).slots
        slot) :=
  LightAssemblerSource.slot_words target _ _ (fun word bound => target_slot ajtai circuitKeys
    input output proof selected folded base witnessWordCount target agree slot word bound) depth

theorem target_slot_value (slot : Fin 7)
    (depth : ((honestValues ajtai circuitKeys input output proof selected folded).slots slot).2 <
      fieldModulus) :
    (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        (slotsStart base witnessWordCount + slot.val * slotWordCount)).value target =
      (honestValues ajtai circuitKeys input output proof selected folded).slots slot :=
  LightAssemblerSource.slot_value target _ _ (fun word bound => target_slot ajtai circuitKeys
    input output proof selected folded base witnessWordCount target agree slot word bound) depth

omit agree in
theorem honest_prior_slot (slot : Fin 2) :
    (honestValues ajtai circuitKeys input output proof selected folded).slots
        ⟨slot.val, by omega⟩ = input.running slot := by
  have bound := slot.isLt
  simp only [honestValues, show slot.val < 2 from bound, ↓reduceIte]
  exact congrArg input.running (finAt_eq _ slot _ (Nat.mod_eq_of_lt bound))

private theorem state_eval (block : Fin 3) (lengths : ∀ block,
      ((honestValues ajtai circuitKeys input output proof selected folded).states block).length =
        4) :
    List.ofFn (fun lane : Fin 4 => target (statesStart base + 4 * block.val + lane.val)) =
      (honestValues ajtai circuitKeys input output proof selected folded).states block := by
  rw [← ofFn_getD 0 _ (lengths block)]
  apply congrArg List.ofFn
  funext lane
  have blockBound := block.isLt
  have laneBound := lane.isLt
  rw [Nat.add_assoc]
  exact (target_state ajtai circuitKeys input output proof selected folded base witnessWordCount
    target agree (4 * block.val + lane.val) (by omega)).trans
    (LightAssemblerSource.quadColumn_eq _ _ block _ laneBound)

omit agree in
/-- On columns that agree with the honest values below `sharedEnd`, `StepRepresents` holds. -/
theorem stepRepresents (program : Stage1.Application.Program) (circuit : Fin 2)
    (agree : ∀ column, column < sharedEnd base program.witnessWordCount →
      target column = honestEnv ajtai circuitKeys input output proof selected folded base
        program.witnessWordCount column)
    (iterationBound : input.iteration < fieldModulus)
    (priorPcBound : input.priorPc < fieldModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < fieldModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount)
    (xLength : output.x.length = 4)
    (pcNext : output.pcNext = circuit)
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .refresh proof) :
    RefreshStepSoundness.StepRepresents program
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount) ajtai circuitKeys circuit input output proof target := by
  set values := honestValues ajtai circuitKeys input output proof selected folded
  have stateLengths : ∀ block, (values.states block).length = 4 := by
    intro block
    simp only [values, honestValues]
    split_ifs
    · exact z0Length
    · exact ziLength
    · exact zNextLength
  have stateBlock := fun (block : Fin 3) => state_eval ajtai circuitKeys input output proof
    selected folded base program.witnessWordCount target agree block stateLengths
  refine {
    iteration := ?_
    iterationBound := iterationBound
    priorPc := ?_
    priorPcBound := priorPcBound
    verifierKeys := fun key => ?_
    initialState := ?_
    currentState := ?_
    nextState := ?_
    witness := ?_
    priorSlots := fun slot => ?_
    priorDepth := priorDepth
    pcNext := pcNext
    outputDigest := ?_
    fresh := ?_
    childCommitments := ?_
    childEvaluations := ?_
    rounds := ?_
    outputPad := ?_
    outputMatrix := ?_
    proof := proofEq }
  · have word := target_counter ajtai circuitKeys input output proof selected folded base
      program.witnessWordCount target agree 0 (by decide)
    rw [Nat.add_zero] at word
    exact word
  · exact target_counter ajtai circuitKeys input output proof selected folded base
      program.witnessWordCount target agree 3 (by decide)
  · rw [← ofFn_getD 0 _ (keyLength key)]
    apply congrArg List.ofFn
    funext lane
    have keyBound := key.isLt
    have laneBound := lane.isLt
    change target (keysStart base + 4 * key.val + lane.val) = _
    rw [Nat.add_assoc]
    exact (target_key ajtai circuitKeys input output proof selected folded base
      program.witnessWordCount target agree (4 * key.val + lane.val) (by omega)).trans
      (LightAssemblerSource.quadColumn_eq _ _ key _ laneBound)
  · have block := stateBlock ⟨0, by decide⟩
    simp only [Nat.mul_zero, Nat.add_zero] at block
    exact block
  · exact stateBlock ⟨1, by decide⟩
  · exact stateBlock ⟨2, by decide⟩
  · rw [← ofFn_getD 0 _ witnessLength]
    apply congrArg List.ofFn
    funext index
    exact target_witness ajtai circuitKeys input output proof selected folded base
      program.witnessWordCount target agree _ index.isLt
  · have words := target_slot_words ajtai circuitKeys input output proof selected folded base
      program.witnessWordCount target agree ⟨slot.val, by omega⟩ (by
        rw [honest_prior_slot ajtai circuitKeys input output proof selected folded]
        exact priorDepth slot)
    rw [honest_prior_slot ajtai circuitKeys input output proof selected folded] at words
    exact words
  · rw [← ofFn_getD 0 _ xLength]
    apply congrArg List.ofFn
    funext lane
    have laneBound := lane.isLt
    have digestWord := (target_digest ajtai circuitKeys input output proof selected folded base
      program.witnessWordCount target agree (16 + lane.val) (by omega)).trans
      (LightAssemblerSource.quadColumn_eq (by decide) values.digests ⟨4, by decide⟩ lane.val
        laneBound)
    have lastBlock : values.digests ⟨4, by decide⟩ = output.x := by
      simp only [values, honestValues, show ¬(4 : Nat) < 2 by decide,
        show ¬(4 : Nat) < 4 by decide, ↓reduceIte]
    rw [lastBlock] at digestWord
    change _ = target (digestsStart base program.witnessWordCount + 16 + lane.val)
    rw [Nat.add_assoc]
    exact digestWord.symm
  · exact LightAssemblerSource.fresh_eval target input.fresh (fun word bound => target_fresh
      ajtai circuitKeys input output proof selected folded base program.witnessWordCount target
      agree word bound)
  · exact childCommitment_eval target proof.split (fun word bound => target_childCommitment
      ajtai circuitKeys input output proof selected folded base program.witnessWordCount target
      agree word bound)
  · exact childEvaluation_eval target proof.split (fun word bound => target_childEvaluation
      ajtai circuitKeys input output proof selected folded base program.witnessWordCount target
      agree word bound)
  · exact round_eval target proof.nifs.piCcsRounds (fun word bound => target_round ajtai
      circuitKeys input output proof selected folded base program.witnessWordCount target agree
      word bound)
  · exact outputPad_eval target proof.nifs.piCcsOutput (fun word bound => target_output ajtai
      circuitKeys input output proof selected folded base program.witnessWordCount target agree
      word bound)
  · exact outputMatrix_eval target proof.nifs.piCcsOutput (fun word bound => target_output ajtai
      circuitKeys input output proof selected folded base program.witnessWordCount target agree
      word bound)

end Honest

end NightstreamFPrime.Layout.BoundedDepth.RefreshAssemblerSource
