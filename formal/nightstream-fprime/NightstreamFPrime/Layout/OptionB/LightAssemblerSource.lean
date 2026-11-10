import NightstreamFPrime.Layout.OptionB.LightAssembler

/-! Values for the light step's source columns (assembler A3d-1). `Values` holds one value for
each group of shared words of `LightAssembler.interface`, and `env` writes them into the columns
from `base`: each column decodes its value from its position in its group. The readback
theorems give the value of each shared word on `env`: a slot's words are the slot
(`slot_value`), the fresh words are the fresh instance (`fresh_eval`), the round words are the
round polynomials (`round_eval`), and the output words are the Π_CCS output (`outputPad_eval`,
`outputMatrix_eval`). -/

namespace NightstreamFPrime.Layout.OptionB.LightAssemblerSource

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Layout.OptionB.LightAssembler

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- `index` modulo `count`, as an index below `count`. -/
def finAt (count : Nat) (positive : 0 < count) (index : Nat) : Fin count :=
  ⟨index % count, Nat.mod_lt _ positive⟩

theorem finAt_eq {count : Nat} (positive : 0 < count) (index : Fin count) (extra : Nat)
    (same : extra % count = index.val) : finAt count positive extra = index :=
  Fin.ext same

theorem finAt_val (count : Nat) (positive : 0 < count) (extra : Nat) :
    (finAt count positive extra).val = extra % count :=
  rfl

/-- Values of `K` in pairs of columns: word `2i` is `c0` of value `i` and word `2i + 1` its `c1`. -/
def kColumn (values : Nat → K) (word : Nat) : F :=
  if word % 2 = 0 then (values (word / 2)).c0 else (values (word / 2)).c1

theorem kAt_eval (env : Env) (start index : Nat) (value : K)
    (low : env (start + 2 * index) = value.c0) (high : env (start + 2 * index + 1) = value.c1) :
    (kAt (start + 2 * index)).eval env = value := by
  change K.mk (env (start + 2 * index)) (env (start + 2 * index + 1)) = value
  rw [low, high]

theorem kColumn_low (values : Nat → K) (index : Nat) :
    kColumn values (2 * index) = (values index).c0 := by
  simp only [kColumn, show 2 * index % 2 = 0 by omega, show 2 * index / 2 = index by omega,
    ↓reduceIte]

theorem kColumn_high (values : Nat → K) (index : Nat) :
    kColumn values (2 * index + 1) = (values index).c1 := by
  simp only [kColumn, show ¬(2 * index + 1) % 2 = 0 by omega,
    show (2 * index + 1) / 2 = index by omega, ↓reduceIte]

/-- Groups of four words: word `4b + lane` is word `lane` of block `b`. -/
def quadColumn {count : Nat} (positive : 0 < count) (blocks : Fin count → List F)
    (word : Nat) : F :=
  (blocks (finAt count positive (word / 4))).getD (word % 4) 0

theorem quadColumn_eq {count : Nat} (positive : 0 < count) (blocks : Fin count → List F)
    (block : Fin count) (lane : Nat) (bound : lane < 4) :
    quadColumn positive blocks (4 * block.val + lane) = (blocks block).getD lane 0 := by
  have blockBound := block.isLt
  unfold quadColumn
  rw [finAt_eq positive block _ (by
      rw [show (4 * block.val + lane) / 4 = block.val by omega]
      exact Nat.mod_eq_of_lt blockBound),
    show (4 * block.val + lane) % 4 = lane by omega]

theorem ofFn_getD {α : Type} {count : Nat} (default : α) (words : List α)
    (length : words.length = count) :
    List.ofFn (fun index : Fin count => words.getD index.val default) = words := by
  apply List.ext_getElem (by simp [length])
  intro index left right
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem right]

/-- The 2,055 words of a slot in the order of `SlotExpr.words`. -/
def slotColumn (data : SlotData logicalWidth publicFits) (word : Nat) : F :=
  if word < 1188 then
    data.1.commitment (finAt 22 (by decide) (word / 54)) (finAt 54 (by decide) word)
  else if word < 1296 then
    kColumn (fun index => data.1.evaluations.pad (finAt 54 (by decide) index)) (word - 1188)
  else if word < 1728 then
    kColumn (fun index => data.1.evaluations.matrix (finAt 4 (by decide) (index / 54))
      (finAt 54 (by decide) index)) (word - 1296)
  else if word < 1784 then
    kColumn (fun index => data.1.point.coordinates.getD index ⟨0, 0⟩) (word - 1728)
  else if word < 2054 then
    data.1.publicInput (finAt 270 (by decide) (word - 1784))
  else natWord data.2

/-- The fresh instance's 1,458 words: the commitment, then the public input. -/
def freshColumn (fresh : StepFresh logicalWidth publicFits) (word : Nat) : F :=
  if word < 1188 then
    fresh.commitments ⟨0, by decide⟩ (finAt 22 (by decide) (word / 54)) (finAt 54 (by decide) word)
  else fresh.publicInputs ⟨0, by decide⟩ (finAt 270 (by decide) (word - 1188))

/-- The round polynomials' coefficients, nine per round. -/
def roundColumn (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.FixedPolynomial K 8)
    (word : Nat) : F :=
  kColumn (fun index => (rounds (finAt 28 (by decide) (index / 9))).coefficients.getD
    (index % 9) ⟨0, 0⟩) word

/-- The Π_CCS output of both sources: per source 54 pad and 216 matrix coordinates. -/
def outputColumn (output : FullOutputCoordinates.FullOutput K LightShape) (word : Nat) : F :=
  kColumn (fun index =>
    if index % 270 < 54 then
      output.padCoordinate (finAt 2 (by decide) (index / 270)) (finAt 54 (by decide) (index % 270))
    else
      output.matrixCoordinate (finAt 2 (by decide) (index / 270))
        (finAt 4 (by decide) ((index % 270 - 54) / 54)) (finAt 54 (by decide) (index % 270 - 54)))
    word

/-- One value for each group of shared words. -/
structure Values (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  counters : List F
  keys : Fin 2 → List F
  states : Fin 3 → List F
  witness : List F
  slots : Fin 7 → SlotData logicalWidth publicFits
  digests : Fin 5 → List F
  fresh : StepFresh logicalWidth publicFits
  rounds : Fin productionShape.cubeVariables → SumCheck.Finite.FixedPolynomial K 8
  output : FullOutputCoordinates.FullOutput K LightShape
  digestSelect : Fin 4 → List F

/-- The source columns from `base`, in the layout of `LightAssembler.interface`. -/
def env (base witnessWordCount : Nat) (values : Values logicalWidth publicFits) : Env :=
  fun column =>
    if column < base then 0
    else if column < keysStart base then values.counters.getD (column - base) 0
    else if column < statesStart base then
      quadColumn (by decide) values.keys (column - keysStart base)
    else if column < witnessStart base then
      quadColumn (by decide) values.states (column - statesStart base)
    else if column < slotsStart base witnessWordCount then
      values.witness.getD (column - witnessStart base) 0
    else if column < digestsStart base witnessWordCount then
      slotColumn (values.slots (finAt _ (by decide)
          ((column - slotsStart base witnessWordCount) / slotWordCount)))
        ((column - slotsStart base witnessWordCount) % slotWordCount)
    else if column < freshStart base witnessWordCount then
      quadColumn (by decide) values.digests (column - digestsStart base witnessWordCount)
    else if column < roundsStart base witnessWordCount then
      freshColumn values.fresh (column - freshStart base witnessWordCount)
    else if column < outputStart base witnessWordCount then
      roundColumn values.rounds (column - roundsStart base witnessWordCount)
    else if column < digestSelectStart base witnessWordCount then
      outputColumn values.output (column - outputStart base witnessWordCount)
    else quadColumn (by decide) values.digestSelect (column - digestSelectStart base witnessWordCount)

variable (base witnessWordCount : Nat) (values : Values logicalWidth publicFits)

theorem env_counter (word : Nat) (bound : word < 5) :
    env base witnessWordCount values (base + word) = values.counters.getD word 0 := by
  simp only [env, show ¬base + word < base by omega,
    show base + word < keysStart base by unfold keysStart; omega, ↓reduceIte,
    Nat.add_sub_cancel_left]

theorem env_key (word : Nat) (bound : word < 8) :
    env base witnessWordCount values (keysStart base + word) =
      quadColumn (by decide) values.keys word := by
  simp only [env, show ¬keysStart base + word < base by unfold keysStart; omega,
    show ¬keysStart base + word < keysStart base by omega,
    show keysStart base + word < statesStart base by unfold keysStart statesStart; omega,
    ↓reduceIte, Nat.add_sub_cancel_left]

theorem env_state (word : Nat) (bound : word < 12) :
    env base witnessWordCount values (statesStart base + word) =
      quadColumn (by decide) values.states word := by
  simp only [env, show ¬statesStart base + word < base by unfold statesStart; omega,
    show ¬statesStart base + word < keysStart base by unfold statesStart keysStart; omega,
    show ¬statesStart base + word < statesStart base by omega,
    show statesStart base + word < witnessStart base by
      unfold statesStart witnessStart; omega,
    ↓reduceIte, Nat.add_sub_cancel_left]

theorem env_witness (word : Nat) (bound : word < witnessWordCount) :
    env base witnessWordCount values (witnessStart base + word) =
      values.witness.getD word 0 := by
  simp only [env, show ¬witnessStart base + word < base by unfold witnessStart; omega,
    show ¬witnessStart base + word < keysStart base by unfold witnessStart keysStart; omega,
    show ¬witnessStart base + word < statesStart base by
      unfold witnessStart statesStart; omega,
    show ¬witnessStart base + word < witnessStart base by omega,
    show witnessStart base + word < slotsStart base witnessWordCount by
      unfold slotsStart; omega,
    ↓reduceIte, Nat.add_sub_cancel_left]

theorem env_slot (slot : Fin 7) (word : Nat) (bound : word < slotWordCount) :
    env base witnessWordCount values
        (slotsStart base witnessWordCount + slot.val * slotWordCount + word) =
      slotColumn (values.slots slot) word := by
  have slotBound := slot.isLt
  have before : slot.val * slotWordCount + word < 7 * slotWordCount := by
    simp only [slotWordCount] at bound ⊢
    omega
  simp only [env, show ¬slotsStart base witnessWordCount + slot.val * slotWordCount + word < base by
      unfold slotsStart witnessStart; omega,
    show ¬slotsStart base witnessWordCount + slot.val * slotWordCount + word < keysStart base by
      unfold slotsStart witnessStart keysStart; omega,
    show ¬slotsStart base witnessWordCount + slot.val * slotWordCount + word < statesStart base by
      unfold slotsStart witnessStart statesStart; omega,
    show ¬slotsStart base witnessWordCount + slot.val * slotWordCount + word < witnessStart base by
      unfold slotsStart; omega,
    show ¬slotsStart base witnessWordCount + slot.val * slotWordCount + word <
      slotsStart base witnessWordCount by omega,
    show slotsStart base witnessWordCount + slot.val * slotWordCount + word <
      digestsStart base witnessWordCount by unfold digestsStart; omega,
    ↓reduceIte]
  rw [show slotsStart base witnessWordCount + slot.val * slotWordCount + word -
      slotsStart base witnessWordCount = slot.val * slotWordCount + word by omega,
    finAt_eq _ slot _ (by simp only [slotWordCount] at bound ⊢; omega),
    show (slot.val * slotWordCount + word) % slotWordCount = word by
      simp only [slotWordCount] at bound ⊢; omega]

theorem env_digest (word : Nat) (bound : word < 20) :
    env base witnessWordCount values (digestsStart base witnessWordCount + word) =
      quadColumn (by decide) values.digests word := by
  have far : witnessStart base ≤ slotsStart base witnessWordCount := by
    unfold slotsStart; omega
  simp only [env, show ¬digestsStart base witnessWordCount + word < base by
      unfold digestsStart slotsStart witnessStart; omega,
    show ¬digestsStart base witnessWordCount + word < keysStart base by
      unfold digestsStart slotsStart witnessStart keysStart; omega,
    show ¬digestsStart base witnessWordCount + word < statesStart base by
      unfold digestsStart slotsStart witnessStart statesStart; omega,
    show ¬digestsStart base witnessWordCount + word < witnessStart base by
      unfold digestsStart; omega,
    show ¬digestsStart base witnessWordCount + word < slotsStart base witnessWordCount by
      unfold digestsStart; omega,
    show ¬digestsStart base witnessWordCount + word < digestsStart base witnessWordCount by
      omega,
    show digestsStart base witnessWordCount + word < freshStart base witnessWordCount by
      unfold freshStart; omega,
    ↓reduceIte, Nat.add_sub_cancel_left]

theorem env_fresh (word : Nat) (bound : word < 1458) :
    env base witnessWordCount values (freshStart base witnessWordCount + word) =
      freshColumn values.fresh word := by
  simp only [env, show ¬freshStart base witnessWordCount + word < base by
      unfold freshStart digestsStart slotsStart witnessStart; omega,
    show ¬freshStart base witnessWordCount + word < keysStart base by
      unfold freshStart digestsStart slotsStart witnessStart keysStart; omega,
    show ¬freshStart base witnessWordCount + word < statesStart base by
      unfold freshStart digestsStart slotsStart witnessStart statesStart; omega,
    show ¬freshStart base witnessWordCount + word < witnessStart base by
      unfold freshStart digestsStart slotsStart; omega,
    show ¬freshStart base witnessWordCount + word < slotsStart base witnessWordCount by
      unfold freshStart digestsStart; omega,
    show ¬freshStart base witnessWordCount + word < digestsStart base witnessWordCount by
      unfold freshStart; omega,
    show ¬freshStart base witnessWordCount + word < freshStart base witnessWordCount by omega,
    show freshStart base witnessWordCount + word < roundsStart base witnessWordCount by
      unfold roundsStart; omega,
    ↓reduceIte, Nat.add_sub_cancel_left]

theorem env_round (word : Nat) (bound : word < 504) :
    env base witnessWordCount values (roundsStart base witnessWordCount + word) =
      roundColumn values.rounds word := by
  simp only [env, show ¬roundsStart base witnessWordCount + word < base by
      unfold roundsStart freshStart digestsStart slotsStart witnessStart; omega,
    show ¬roundsStart base witnessWordCount + word < keysStart base by
      unfold roundsStart freshStart digestsStart slotsStart witnessStart keysStart; omega,
    show ¬roundsStart base witnessWordCount + word < statesStart base by
      unfold roundsStart freshStart digestsStart slotsStart witnessStart statesStart; omega,
    show ¬roundsStart base witnessWordCount + word < witnessStart base by
      unfold roundsStart freshStart digestsStart slotsStart; omega,
    show ¬roundsStart base witnessWordCount + word < slotsStart base witnessWordCount by
      unfold roundsStart freshStart digestsStart; omega,
    show ¬roundsStart base witnessWordCount + word < digestsStart base witnessWordCount by
      unfold roundsStart freshStart; omega,
    show ¬roundsStart base witnessWordCount + word < freshStart base witnessWordCount by
      unfold roundsStart; omega,
    show ¬roundsStart base witnessWordCount + word < roundsStart base witnessWordCount by omega,
    show roundsStart base witnessWordCount + word < outputStart base witnessWordCount by
      unfold outputStart; omega,
    ↓reduceIte, Nat.add_sub_cancel_left]

theorem env_output (word : Nat) (bound : word < 1080) :
    env base witnessWordCount values (outputStart base witnessWordCount + word) =
      outputColumn values.output word := by
  simp only [env, show ¬outputStart base witnessWordCount + word < base by
      unfold outputStart roundsStart freshStart digestsStart slotsStart witnessStart; omega,
    show ¬outputStart base witnessWordCount + word < keysStart base by
      unfold outputStart roundsStart freshStart digestsStart slotsStart witnessStart keysStart
      omega,
    show ¬outputStart base witnessWordCount + word < statesStart base by
      unfold outputStart roundsStart freshStart digestsStart slotsStart witnessStart statesStart
      omega,
    show ¬outputStart base witnessWordCount + word < witnessStart base by
      unfold outputStart roundsStart freshStart digestsStart slotsStart; omega,
    show ¬outputStart base witnessWordCount + word < slotsStart base witnessWordCount by
      unfold outputStart roundsStart freshStart digestsStart; omega,
    show ¬outputStart base witnessWordCount + word < digestsStart base witnessWordCount by
      unfold outputStart roundsStart freshStart; omega,
    show ¬outputStart base witnessWordCount + word < freshStart base witnessWordCount by
      unfold outputStart roundsStart; omega,
    show ¬outputStart base witnessWordCount + word < roundsStart base witnessWordCount by
      unfold outputStart; omega,
    show ¬outputStart base witnessWordCount + word < outputStart base witnessWordCount by omega,
    show outputStart base witnessWordCount + word < digestSelectStart base witnessWordCount by
      unfold digestSelectStart; omega,
    ↓reduceIte, Nat.add_sub_cancel_left]

theorem env_digestSelect (word : Nat) :
    env base witnessWordCount values (digestSelectStart base witnessWordCount + word) =
      quadColumn (by decide) values.digestSelect word := by
  simp only [env, show ¬digestSelectStart base witnessWordCount + word < base by
      unfold digestSelectStart outputStart roundsStart freshStart digestsStart slotsStart witnessStart
      omega,
    show ¬digestSelectStart base witnessWordCount + word < keysStart base by
      unfold digestSelectStart outputStart roundsStart freshStart digestsStart slotsStart witnessStart keysStart
      omega,
    show ¬digestSelectStart base witnessWordCount + word < statesStart base by
      unfold digestSelectStart outputStart roundsStart freshStart digestsStart slotsStart witnessStart statesStart
      omega,
    show ¬digestSelectStart base witnessWordCount + word < witnessStart base by
      unfold digestSelectStart outputStart roundsStart freshStart digestsStart slotsStart; omega,
    show ¬digestSelectStart base witnessWordCount + word < slotsStart base witnessWordCount by
      unfold digestSelectStart outputStart roundsStart freshStart digestsStart; omega,
    show ¬digestSelectStart base witnessWordCount + word < digestsStart base witnessWordCount by
      unfold digestSelectStart outputStart roundsStart freshStart; omega,
    show ¬digestSelectStart base witnessWordCount + word < freshStart base witnessWordCount by
      unfold digestSelectStart outputStart roundsStart; omega,
    show ¬digestSelectStart base witnessWordCount + word < roundsStart base witnessWordCount by
      unfold digestSelectStart outputStart; omega,
    show ¬digestSelectStart base witnessWordCount + word < outputStart base witnessWordCount by
      unfold digestSelectStart; omega,
    show ¬digestSelectStart base witnessWordCount + word <
      digestSelectStart base witnessWordCount by omega,
    ↓reduceIte, Nat.add_sub_cancel_left]

section Readback

variable {base witnessWordCount} (env : Env)

private theorem fixedPolynomial_ext {left right : SumCheck.Finite.FixedPolynomial K 8}
    (same : left.coefficients = right.coefficients) : left = right := by
  cases left
  cases right
  simp_all

/-- The words of a slot placed at `start` carry the slot whose columns `env` holds. -/
theorem slot_value (start : Nat) (data : SlotData logicalWidth publicFits)
    (columns : ∀ word, word < slotWordCount → env (start + word) = slotColumn data word)
    (depth : data.2 < goldilocksModulus) :
    (slotAt start).value env = data := by
  obtain ⟨⟨commitment, publicInput, point, evaluations⟩, depthValue⟩ := data
  simp only [slotWordCount] at columns
  have kWord := fun (offset : Nat) (index : Nat) (value : K)
      (low : slotColumn (⟨commitment, publicInput, point, evaluations⟩, depthValue)
        (offset + 2 * index) = value.c0)
      (high : slotColumn (⟨commitment, publicInput, point, evaluations⟩, depthValue)
        (offset + 2 * index + 1) = value.c1)
      (bound : offset + 2 * index + 1 < 2055) =>
    kAt_eval env (start + offset) index value
      (by rw [show start + offset + 2 * index = start + (offset + 2 * index) by omega,
        columns _ (by omega), low])
      (by rw [show start + offset + 2 * index + 1 = start + (offset + 2 * index + 1) by omega,
        columns _ (by omega), high])
  refine Prod.ext ?_ ?_
  · simp only [SlotExpr.value]
    congr 1
    · funext row lane
      have rowBound := row.isLt
      have laneBound := lane.isLt
      change row.val < 22 at rowBound
      change lane.val < 54 at laneBound
      change env (start + row.val * ringDegree + lane.val) = commitment row lane
      rw [show start + row.val * ringDegree + lane.val = start + (row.val * 54 + lane.val) by
        simp only [ringDegree]; omega, columns _ (by omega)]
      simp only [slotColumn, show row.val * 54 + lane.val < 1188 by omega, ↓reduceIte]
      exact congrArg₂ commitment (Fin.ext (by rw [finAt_val]; omega))
        (Fin.ext (by rw [finAt_val]; omega))
    · funext column
      have columnBound := column.isLt
      change column.val < 270 at columnBound
      change env (start + 1784 + column.val) = publicInput column
      rw [show start + 1784 + column.val = start + (1784 + column.val) by omega,
        columns _ (by omega)]
      simp only [slotColumn, show ¬1784 + column.val < 1188 by omega,
        show ¬1784 + column.val < 1296 by omega, show ¬1784 + column.val < 1728 by omega,
        show ¬1784 + column.val < 1784 by omega, show 1784 + column.val < 2054 by omega,
        ↓reduceIte, Nat.add_sub_cancel_left]
      exact congrArg publicInput (Fin.ext (by rw [finAt_val]; omega))
    · obtain ⟨coordinates, dimension⟩ := point
      simp only [StatementAbsorption.evalPoint, CubePoint.mk.injEq]
      change coordinates.length = 28 at dimension
      rw [← ofFn_getD ⟨0, 0⟩ coordinates dimension]
      apply congrArg List.ofFn
      funext coordinate
      have coordinateBound := coordinate.isLt
      change coordinate.val < 28 at coordinateBound
      apply kWord 1728 coordinate.val
      · simp only [slotColumn, show ¬1728 + 2 * coordinate.val < 1188 by omega,
          show ¬1728 + 2 * coordinate.val < 1296 by omega,
          show ¬1728 + 2 * coordinate.val < 1728 by omega,
          show 1728 + 2 * coordinate.val < 1784 by omega, ↓reduceIte, Nat.add_sub_cancel_left,
          kColumn_low]
      · simp only [slotColumn, show ¬1728 + 2 * coordinate.val + 1 < 1188 by omega,
          show ¬1728 + 2 * coordinate.val + 1 < 1296 by omega,
          show ¬1728 + 2 * coordinate.val + 1 < 1728 by omega,
          show 1728 + 2 * coordinate.val + 1 < 1784 by omega, ↓reduceIte,
          show 1728 + 2 * coordinate.val + 1 - 1728 = 2 * coordinate.val + 1 by omega,
          kColumn_high]
      · omega
    · obtain ⟨pad, matrix⟩ := evaluations
      simp only [StatementAbsorption.evalEvaluation]
      congr 1
      · funext coefficient
        have coefficientBound := coefficient.isLt
        change coefficient.val < 54 at coefficientBound
        apply kWord 1188 coefficient.val
        · simp only [slotColumn, show ¬1188 + 2 * coefficient.val < 1188 by omega,
            show 1188 + 2 * coefficient.val < 1296 by omega, ↓reduceIte,
            Nat.add_sub_cancel_left, kColumn_low]
          exact congrArg (fun index => (pad index).c0) (Fin.ext (by rw [finAt_val]; omega))
        · simp only [slotColumn, show ¬1188 + 2 * coefficient.val + 1 < 1188 by omega,
            show 1188 + 2 * coefficient.val + 1 < 1296 by omega, ↓reduceIte,
            show 1188 + 2 * coefficient.val + 1 - 1188 = 2 * coefficient.val + 1 by omega,
            kColumn_high]
          exact congrArg (fun index => (pad index).c1) (Fin.ext (by rw [finAt_val]; omega))
        · omega
      · funext matrixIndex coefficient
        have matrixBound := matrixIndex.isLt
        have coefficientBound := coefficient.isLt
        change matrixIndex.val < 4 at matrixBound
        change coefficient.val < 54 at coefficientBound
        change (kAt (start + 1296 + 2 * (matrixIndex.val * ringDegree + coefficient.val))).eval
          env = _
        simp only [ringDegree]
        apply kWord 1296 (matrixIndex.val * 54 + coefficient.val)
        · simp only [slotColumn,
            show ¬1296 + 2 * (matrixIndex.val * 54 + coefficient.val) < 1188 by omega,
            show ¬1296 + 2 * (matrixIndex.val * 54 + coefficient.val) < 1296 by omega,
            show 1296 + 2 * (matrixIndex.val * 54 + coefficient.val) < 1728 by omega,
            ↓reduceIte, Nat.add_sub_cancel_left, kColumn_low]
          exact congrArg₂ (fun first second => (matrix first second).c0)
            (Fin.ext (by rw [finAt_val]; omega)) (Fin.ext (by rw [finAt_val]; omega))
        · simp only [slotColumn,
            show ¬1296 + 2 * (matrixIndex.val * 54 + coefficient.val) + 1 < 1188 by omega,
            show ¬1296 + 2 * (matrixIndex.val * 54 + coefficient.val) + 1 < 1296 by omega,
            show 1296 + 2 * (matrixIndex.val * 54 + coefficient.val) + 1 < 1728 by omega,
            ↓reduceIte,
            show 1296 + 2 * (matrixIndex.val * 54 + coefficient.val) + 1 - 1296 =
              2 * (matrixIndex.val * 54 + coefficient.val) + 1 by omega, kColumn_high]
          exact congrArg₂ (fun first second => (matrix first second).c1)
            (Fin.ext (by rw [finAt_val]; omega)) (Fin.ext (by rw [finAt_val]; omega))
        · omega
  · change (env (start + 2054)).val = depthValue
    rw [columns 2054 (by decide)]
    simp only [slotColumn, show ¬2054 < 1188 by decide, show ¬2054 < 1296 by decide,
      show ¬2054 < 1728 by decide, show ¬2054 < 1784 by decide, show ¬2054 < 2054 by decide,
      ↓reduceIte]
    simp [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt depth]

/-- A slot's words evaluate to the slot's serialization. -/
theorem slot_words (start : Nat) (data : SlotData logicalWidth publicFits)
    (columns : ∀ word, word < slotWordCount → env (start + word) = slotColumn data word)
    (depth : data.2 < goldilocksModulus) :
    Hash.evalList env (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits) start).words =
      serializeSlot data := by
  rw [SlotExpr.words_eval, slot_value env start data columns depth]

/-- The fresh words carry the fresh instance whose columns `env` holds. -/
theorem fresh_eval (fresh : StepFresh logicalWidth publicFits)
    (columns : ∀ word, word < 1458 →
      env (freshStart base witnessWordCount + word) = freshColumn fresh word) :
    StatementAbsorption.evalFresh
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          witnessWordCount).fresh env = fresh := by
  obtain ⟨commitments, publicInputs⟩ := fresh
  simp only [StatementAbsorption.evalFresh]
  have only : ∀ source : Fin productionShape.freshCount, source = ⟨0, by decide⟩ :=
    fun source => Fin.ext (by
      have bound := source.isLt
      change source.val < 1 at bound
      show source.val = 0
      omega)
  congr 1
  · funext source row lane
    have rowBound := row.isLt
    have laneBound := lane.isLt
    change row.val < 22 at rowBound
    change lane.val < 54 at laneBound
    change env (freshStart base witnessWordCount + row.val * ringDegree + lane.val) = _
    rw [show freshStart base witnessWordCount + row.val * ringDegree + lane.val =
      freshStart base witnessWordCount + (row.val * 54 + lane.val) by simp only [ringDegree]; omega,
      columns _ (by omega)]
    simp only [freshColumn, show row.val * 54 + lane.val < 1188 by omega, ↓reduceIte]
    rw [only source]
    exact congrArg₂ (commitments ⟨0, by decide⟩) (Fin.ext (by rw [finAt_val]; omega))
      (Fin.ext (by rw [finAt_val]; omega))
  · funext source column
    have columnBound := column.isLt
    change column.val < 270 at columnBound
    change env (freshStart base witnessWordCount + 1188 + column.val) = _
    rw [show freshStart base witnessWordCount + 1188 + column.val =
      freshStart base witnessWordCount + (1188 + column.val) by omega, columns _ (by omega)]
    simp only [freshColumn, show ¬1188 + column.val < 1188 by omega, ↓reduceIte,
      Nat.add_sub_cancel_left]
    rw [only source]
    exact congrArg (publicInputs ⟨0, by decide⟩) (Fin.ext (by rw [finAt_val]; omega))

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
theorem outputPad_eval (output : FullOutputCoordinates.FullOutput K LightShape)
    (columns : ∀ word, word < 1080 →
      env (outputStart base witnessWordCount + word) = outputColumn output word)
    (source : Fin LightShape.sourceCount) (coefficient : Fin LightShape.coefficientCount) :
    ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        witnessWordCount).output.padCoordinate source coefficient).eval env =
      output.padCoordinate source coefficient := by
  have sourceBound := source.isLt
  have coefficientBound := coefficient.isLt
  change source.val < 2 at sourceBound
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

theorem outputMatrix_eval (output : FullOutputCoordinates.FullOutput K LightShape)
    (columns : ∀ word, word < 1080 →
      env (outputStart base witnessWordCount + word) = outputColumn output word)
    (source : Fin LightShape.sourceCount) (matrix : Fin LightShape.matrixCount)
    (coefficient : Fin LightShape.coefficientCount) :
    ((interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        witnessWordCount).output.matrixCoordinate source matrix coefficient).eval env =
      output.matrixCoordinate source matrix coefficient := by
  have sourceBound := source.isLt
  have matrixBound := matrix.isLt
  have coefficientBound := coefficient.isLt
  change source.val < 2 at sourceBound
  change matrix.val < 4 at matrixBound
  change coefficient.val < 54 at coefficientBound
  have position : (source.val * 270 + 54 + matrix.val * 54 + coefficient.val) % 270 =
      54 + matrix.val * 54 + coefficient.val := by omega
  have notPad : ¬54 + matrix.val * 54 + coefficient.val < 54 := by omega
  apply kAt_eval
  · rw [columns (2 * (source.val * 270 + 54 + matrix.val * 54 + coefficient.val)) (by omega)]
    unfold outputColumn
    rw [kColumn_low]
    simp only [position, notPad, ↓reduceIte]
    rw [show (output.matrixCoordinate (finAt 2 (by decide)
        ((source.val * 270 + 54 + matrix.val * 54 + coefficient.val) / 270))
        (finAt 4 (by decide) ((54 + matrix.val * 54 + coefficient.val - 54) / 54))
        (finAt 54 (by decide) (54 + matrix.val * 54 + coefficient.val - 54))) =
      output.matrixCoordinate source matrix coefficient from
        congrArg₂ (fun first second => output.matrixCoordinate first second _)
          (Fin.ext (by rw [finAt_val]; omega)) (Fin.ext (by rw [finAt_val]; omega)) |>.trans
          (congrArg (output.matrixCoordinate source matrix) (Fin.ext (by rw [finAt_val]; omega)))]
  · rw [Nat.add_assoc,
      columns (2 * (source.val * 270 + 54 + matrix.val * 54 + coefficient.val) + 1) (by omega)]
    unfold outputColumn
    rw [kColumn_high]
    simp only [position, notPad, ↓reduceIte]
    rw [show (output.matrixCoordinate (finAt 2 (by decide)
        ((source.val * 270 + 54 + matrix.val * 54 + coefficient.val) / 270))
        (finAt 4 (by decide) ((54 + matrix.val * 54 + coefficient.val - 54) / 54))
        (finAt 54 (by decide) (54 + matrix.val * 54 + coefficient.val - 54))) =
      output.matrixCoordinate source matrix coefficient from
        congrArg₂ (fun first second => output.matrixCoordinate first second _)
          (Fin.ext (by rw [finAt_val]; omega)) (Fin.ext (by rw [finAt_val]; omega)) |>.trans
          (congrArg (output.matrixCoordinate source matrix) (Fin.ext (by rw [finAt_val]; omega)))]

end Readback

section Honest

open NightstreamFPrime.Spec.HyperNova.Construction2

variable (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
  (proof : Spec.Folding.Light.Proof LightShape 8) (selected : Fin 2)
  (folded : SlotData logicalWidth publicFits)

/-- The honest values of a light step: the counters with the iteration's inverse, the recursive
flag and the selected slot; the key digests; the start, current and next states; the witness; the
prior slots, the active slot (the selected prior slot), the inner slots (`folded` for the
selected slot, the prior slot otherwise) and the next slots; the slot and output digests; the
fresh instance; the light proof; and the new digest (of `folded`), the inner digests and the active
digest (of the selected prior slot). -/
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
  rounds := proof.piCcsRounds
  output := proof.piCcsOutput
  digestSelect := fun index =>
    if index.val = 0 then slotDigest folded
    else if index.val = 3 then slotDigest (input.running selected)
    else if index.val - 1 = selected.val then slotDigest folded
    else slotDigest (input.running (finAt 2 (by decide) (index.val - 1)))

/-- The honest source columns. -/
abbrev honestEnv (base witnessWordCount : Nat) : Env :=
  env base witnessWordCount (honestValues circuitKeys input output proof selected folded)

variable (base witnessWordCount : Nat) (target : Env)
  (agree : ∀ column, column < sharedEnd base witnessWordCount →
    target column = honestEnv circuitKeys input output proof selected folded base witnessWordCount
      column)

omit circuitKeys input output proof selected folded in
private theorem starts : keysStart base = base + 5 ∧ statesStart base = base + 13 ∧
    witnessStart base = base + 25 ∧
    slotsStart base witnessWordCount = base + 25 + witnessWordCount ∧
    digestsStart base witnessWordCount = base + 25 + witnessWordCount + 14385 ∧
    freshStart base witnessWordCount = base + 25 + witnessWordCount + 14405 ∧
    roundsStart base witnessWordCount = base + 25 + witnessWordCount + 15863 ∧
    outputStart base witnessWordCount = base + 25 + witnessWordCount + 16367 ∧
    digestSelectStart base witnessWordCount = base + 25 + witnessWordCount + 17447 ∧
    sharedEnd base witnessWordCount = base + 25 + witnessWordCount + 17463 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp only [keysStart, statesStart, witnessStart, slotsStart, digestsStart, freshStart,
      roundsStart, outputStart, digestSelectStart, sharedEnd, slotWordCount]

include agree

theorem target_counter (word : Nat) (bound : word < 5) :
    target (base + word) =
      (honestValues circuitKeys input output proof selected folded).counters.getD word 0 :=
  (agree _ (by have := starts base witnessWordCount; omega)).trans
    (env_counter base witnessWordCount _ word bound)

theorem target_key (word : Nat) (bound : word < 8) :
    target (keysStart base + word) =
      quadColumn (by decide) (honestValues circuitKeys input output proof selected folded).keys
        word :=
  (agree _ (by have := starts base witnessWordCount; omega)).trans
    (env_key base witnessWordCount _ word bound)

theorem target_state (word : Nat) (bound : word < 12) :
    target (statesStart base + word) =
      quadColumn (by decide) (honestValues circuitKeys input output proof selected folded).states
        word :=
  (agree _ (by have := starts base witnessWordCount; omega)).trans
    (env_state base witnessWordCount _ word bound)

theorem target_witness (word : Nat) (bound : word < witnessWordCount) :
    target (witnessStart base + word) = input.witness.getD word 0 :=
  (agree _ (by have := starts base witnessWordCount; omega)).trans
    (env_witness base witnessWordCount _ word bound)

theorem target_slot (slot : Fin 7) (word : Nat) (bound : word < slotWordCount) :
    target (slotsStart base witnessWordCount + slot.val * slotWordCount + word) =
      slotColumn ((honestValues circuitKeys input output proof selected folded).slots slot) word :=
  (agree _ (by
    have := starts base witnessWordCount
    have slotBound := slot.isLt
    simp only [slotWordCount] at bound ⊢
    omega)).trans (env_slot base witnessWordCount _ slot word bound)

theorem target_digest (word : Nat) (bound : word < 20) :
    target (digestsStart base witnessWordCount + word) =
      quadColumn (by decide) (honestValues circuitKeys input output proof selected folded).digests
        word :=
  (agree _ (by have := starts base witnessWordCount; omega)).trans
    (env_digest base witnessWordCount _ word bound)

theorem target_fresh (word : Nat) (bound : word < 1458) :
    target (freshStart base witnessWordCount + word) = freshColumn input.fresh word :=
  (agree _ (by have := starts base witnessWordCount; omega)).trans
    (env_fresh base witnessWordCount _ word bound)

theorem target_round (word : Nat) (bound : word < 504) :
    target (roundsStart base witnessWordCount + word) = roundColumn proof.piCcsRounds word :=
  (agree _ (by have := starts base witnessWordCount; omega)).trans
    (env_round base witnessWordCount _ word bound)

theorem target_output (word : Nat) (bound : word < 1080) :
    target (outputStart base witnessWordCount + word) = outputColumn proof.piCcsOutput word :=
  (agree _ (by have := starts base witnessWordCount; omega)).trans
    (env_output base witnessWordCount _ word bound)

theorem target_digestSelect (word : Nat) (bound : word < 16) :
    target (digestSelectStart base witnessWordCount + word) =
      quadColumn (by decide)
        (honestValues circuitKeys input output proof selected folded).digestSelect word :=
  (agree _ (by have := starts base witnessWordCount; omega)).trans
    (env_digestSelect base witnessWordCount _ word)

/-- The words of slot group `slot` carry the honest slot of that group. -/
theorem target_slot_words (slot : Fin 7)
    (depth : ((honestValues circuitKeys input output proof selected folded).slots slot).2 <
      goldilocksModulus) :
    Hash.evalList target (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        (slotsStart base witnessWordCount + slot.val * slotWordCount)).words =
      serializeSlot ((honestValues circuitKeys input output proof selected folded).slots slot) :=
  slot_words target _ _ (fun word bound => target_slot circuitKeys input output proof selected
    folded base witnessWordCount target agree slot word bound) depth

theorem target_slot_value (slot : Fin 7)
    (depth : ((honestValues circuitKeys input output proof selected folded).slots slot).2 <
      goldilocksModulus) :
    (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        (slotsStart base witnessWordCount + slot.val * slotWordCount)).value target =
      (honestValues circuitKeys input output proof selected folded).slots slot :=
  slot_value target _ _ (fun word bound => target_slot circuitKeys input output proof selected
    folded base witnessWordCount target agree slot word bound) depth

omit agree in
theorem honest_prior_slot (slot : Fin 2) :
    (honestValues circuitKeys input output proof selected folded).slots ⟨slot.val, by omega⟩ =
      input.running slot := by
  have bound := slot.isLt
  simp only [honestValues, show slot.val < 2 from bound, ↓reduceIte]
  exact congrArg input.running (finAt_eq _ slot _ (Nat.mod_eq_of_lt bound))

private theorem state_eval (block : Fin 3) (lengths : ∀ block,
      ((honestValues circuitKeys input output proof selected folded).states block).length = 4) :
    List.ofFn (fun lane : Fin 4 => target (statesStart base + 4 * block.val + lane.val)) =
      (honestValues circuitKeys input output proof selected folded).states block := by
  rw [← ofFn_getD 0 _ (lengths block)]
  apply congrArg List.ofFn
  funext lane
  have blockBound := block.isLt
  have laneBound := lane.isLt
  rw [Nat.add_assoc]
  exact (target_state circuitKeys input output proof selected folded base witnessWordCount
    target agree (4 * block.val + lane.val) (by omega)).trans
    (quadColumn_eq _ _ block _ laneBound)

omit agree in
/-- On columns that agree with the honest values below `sharedEnd`, `StepRepresents` holds. -/
theorem stepRepresents (program : Stage1.Application.Program)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuit : Fin 2)
    (agree : ∀ column, column < sharedEnd base program.witnessWordCount →
      target column = honestEnv circuitKeys input output proof selected folded base
        program.witnessWordCount column)
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount)
    (xLength : output.x.length = 4)
    (pcNext : output.pcNext = circuit)
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .light proof) :
    LightStepSoundness.StepRepresents program
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount) ajtai circuitKeys circuit input output proof target := by
  set values := honestValues circuitKeys input output proof selected folded
  have stateLengths : ∀ block, (values.states block).length = 4 := by
    intro block
    simp only [values, honestValues]
    split_ifs
    · exact z0Length
    · exact ziLength
    · exact zNextLength
  have stateBlock := fun (block : Fin 3) => state_eval circuitKeys input output proof selected
    folded base program.witnessWordCount target agree block stateLengths
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
    active := ?_
    otherDigests := fun _ slot _ => ?_
    pcNext := pcNext
    outputDigest := ?_
    fresh := ?_
    rounds := ?_
    outputPad := ?_
    outputMatrix := ?_
    proof := proofEq }
  · have word := target_counter circuitKeys input output proof selected folded base
      program.witnessWordCount target agree 0 (by decide)
    rw [Nat.add_zero] at word
    exact word
  · exact target_counter circuitKeys input output proof selected folded base
      program.witnessWordCount target agree 3 (by decide)
  · rw [← ofFn_getD 0 _ (keyLength key)]
    apply congrArg List.ofFn
    funext lane
    have keyBound := key.isLt
    have laneBound := lane.isLt
    change target (keysStart base + 4 * key.val + lane.val) = _
    rw [Nat.add_assoc]
    exact (target_key circuitKeys input output proof selected folded base
      program.witnessWordCount target agree (4 * key.val + lane.val) (by omega)).trans
      (quadColumn_eq _ _ key _ laneBound)
  · have block := stateBlock ⟨0, by decide⟩
    simp only [Nat.mul_zero, Nat.add_zero] at block
    exact block
  · exact stateBlock ⟨1, by decide⟩
  · exact stateBlock ⟨2, by decide⟩
  · rw [← ofFn_getD 0 _ witnessLength]
    apply congrArg List.ofFn
    funext index
    exact target_witness circuitKeys input output proof selected folded base
      program.witnessWordCount target agree _ index.isLt
  · have activeSlot : values.slots ⟨2, by decide⟩ = input.running selected := by
      simp only [values, honestValues, show ¬(2 : Nat) < 2 by decide, ↓reduceIte]
    have selectedEq : LightStepLayout.selectedSlot program
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount) target = selected := by
      have word := target_counter circuitKeys input output proof selected folded base
        program.witnessWordCount target agree 4 (by decide)
      change (if target (base + 4) = 1 then (1 : Fin 2) else 0) = selected
      rw [word]
      simp only [honestValues, List.getD_cons_succ, List.getD_cons_zero]
      fin_cases selected <;> decide
    rw [selectedEq, ← activeSlot]
    exact target_slot_value circuitKeys input output proof selected folded base
      program.witnessWordCount target agree ⟨2, by decide⟩ (by rw [activeSlot]; exact priorDepth _)
  · have slotBound := slot.isLt
    have block : values.digests ⟨slot.val, by omega⟩ = slotDigest (input.running slot) := by
      simp only [values, honestValues, show slot.val < 2 from slotBound, ↓reduceIte]
      exact congrArg (fun index => slotDigest (input.running index))
        (finAt_eq _ slot _ (Nat.mod_eq_of_lt slotBound))
    unfold digestValue
    rw [← block, ← ofFn_getD 0 (values.digests ⟨slot.val, by omega⟩)
      (by rw [block, slotDigest, Poseidon2.hash_length])]
    apply congrArg List.ofFn
    funext lane
    have laneBound := lane.isLt
    change target (digestsStart base program.witnessWordCount + 4 * slot.val + lane.val) = _
    rw [Nat.add_assoc]
    exact (target_digest circuitKeys input output proof selected folded base
      program.witnessWordCount target agree (4 * slot.val + lane.val) (by omega)).trans
      (quadColumn_eq (by decide) values.digests ⟨slot.val, by omega⟩ lane.val laneBound)
  · rw [← ofFn_getD 0 _ xLength]
    apply congrArg List.ofFn
    funext lane
    have laneBound := lane.isLt
    have digestWord := (target_digest circuitKeys input output proof selected folded base
      program.witnessWordCount target agree (16 + lane.val) (by omega)).trans
      (quadColumn_eq (by decide) values.digests ⟨4, by decide⟩ lane.val laneBound)
    have lastBlock : values.digests ⟨4, by decide⟩ = output.x := by
      simp only [values, honestValues, show ¬(4 : Nat) < 2 by decide,
        show ¬(4 : Nat) < 4 by decide, ↓reduceIte]
    rw [lastBlock] at digestWord
    change _ = target (digestsStart base program.witnessWordCount + 16 + lane.val)
    rw [Nat.add_assoc]
    exact digestWord.symm
  · exact fresh_eval target input.fresh (fun word bound => target_fresh circuitKeys input
      output proof selected folded base program.witnessWordCount target agree word bound)
  · exact round_eval target proof.piCcsRounds (fun word bound => target_round circuitKeys input
      output proof selected folded base program.witnessWordCount target agree word bound)
  · exact outputPad_eval target proof.piCcsOutput (fun word bound => target_output circuitKeys
      input output proof selected folded base program.witnessWordCount target agree word bound)
  · exact outputMatrix_eval target proof.piCcsOutput (fun word bound => target_output
      circuitKeys input output proof selected folded base program.witnessWordCount target agree
      word bound)

end Honest

end NightstreamFPrime.Layout.OptionB.LightAssemblerSource
