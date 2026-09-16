import NightstreamFPrime.Export.Stage1.PiCCSCarriedAccumulation
import NightstreamFPrime.Export.Stage1.PiCCSCarriedSource

/-! Selected carried accumulation from original assignments. The Pad block
kernel and optional numeric matrix reads produce the two full moments.
This is a pure kernel theorem, not a proof of the IO task or cache loop. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedComplete

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle
open NumericCompletionSum (numericSum)

private abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev selectedProgram := PerApplicationMatrixProgram.matrixProgram
  Poseidon2HashChainV1Package.application
private abbrev selectedLayout :=
  (Lifecycle.PiRLC.v1_1.InputBinding.relationSource selectedRelation).cubeLayout
private noncomputable abbrev selectedStatement (input : PiCCSPublicReplay.Input) :=
  (ProductionKey.key selectedRelation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
    (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)
private noncomputable abbrev sourceData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth) :=
  (selectedStatement input).sourceProtocolData K.embed witness
private abbrev powers (gamma : K) := TargetPolynomial.power extensionOps.toOps gamma
private abbrev basis (gamma : K) :=
  PiCCSAggregatedImages.prepare (PiDECParentSparseRead.prepare ()) (powers gamma)
private abbrev blocks
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) := PiCCSAggregatedImages.combinedBlock (powers gamma) witness.assignments
private def weight (input : PiCCSPublicReplay.Input) : Nat → K :=
  NumericBooleanDomain.tensorWeightCoordinates extensionOps
    (PiCCSPublicReplay.verifierInput input).priorPoint.coordinates.tail

private def atRow (value : BooleanVertex cubeVariables → K) (row : Nat) : K :=
  if inside : row < 2 ^ cubeVariables then
    value (NumericBooleanDomain.vertex cubeVariables ⟨row, inside⟩)
  else extensionOps.zero

private noncomputable def padRow (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) : Nat → K := atRow (PiCCSCarriedSource.originalPad input witness gamma)
private noncomputable def matrixRow (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) : Nat → K := atRow (PiCCSCarriedSource.originalMatrix input witness gamma)

private theorem source_prior (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth) :
    (sourceData input witness).priorPoint = (PiCCSPublicReplay.verifierInput input).priorPoint := by
  have source := StrongReduction.Statement.sourceProtocolData_toVerifierInput
    K.embed (selectedStatement input) witness
  have publicEq := PiCCSPublicReplay.verifierInput_eq_key input selectedRelation
    Poseidon2HashChainV1Setup.productionAjtaiKey
  exact congrArg (fun value : ProtocolPolynomial.VerifierInput K productionShape => value.priorPoint)
    (source.trans publicEq.symm)

private theorem fullShape_carrierWidth (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :
    (PaperAlgebra.FullShape logicalWidth publicFits).carrierWidth =
      Phi81CarrierLayout.carrierWidth logicalWidth := rfl

private theorem carrier_count : PiCCSSourceImages.blockCount * ringDegree =
    Phi81CarrierLayout.carrierWidth PiCCSSourceImages.logicalWidth := by
  rw [PiCCSCarriedSource.blockCount_eq_authority, Phi81CarrierLayout.blockCount_carrierWidth,
    ← Phi81CarrierLayout.carrierWidth_eq]

private theorem rows_fit : selectedProgram.rowCount ≤ 2 ^ cubeVariables := by
  rw [PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits]
  exact PerApplicationFixedPoint.structuralPlan_rowCount_le
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

private theorem atRow_toVertex {columns : Nat} (covered : columns ≤ 2 ^ cubeVariables)
    (value : BooleanVertex cubeVariables → K) (column : Fin columns) :
    atRow value column.val =
      value ((Folding.PiCCS.CanonicalRowLayout.layout cubeVariables columns covered).toVertex column) := by
  rw [atRow, dif_pos (Nat.lt_of_lt_of_le column.isLt covered)]
  rfl

private theorem carrierColumn_val {logicalWidth : Nat}
    (block : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (lane : Fin ringDegree) :
    (Phi81CarrierLayout.carrierColumn (logicalWidth := logicalWidth) block lane).val =
      block.val * ringDegree + lane.val := rfl

private theorem pad_dot (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (block : Nat) (live : block < PiCCSSourceImages.blockCount) (lane : Fin ringDegree) :
    PiCCSWeightedBasis.dotK ((basis gamma).1.get lane) (blocks witness gamma block).get =
      padRow input witness gamma (block * ringDegree + lane.val) := by
  let typedBlock := Fin.cast PiCCSCarriedSource.blockCount_eq_authority (⟨block, live⟩ : Fin PiCCSSourceImages.blockCount)
  have source := PiCCSCarriedSource.pad_block_sourceProtocolData input witness gamma typedBlock lane
  have flat := atRow_toVertex selectedRelation.cubeFits
    (PiCCSCarriedSource.originalPad input witness gamma)
    (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) typedBlock lane)
  rw [carrierColumn_val] at flat
  exact source.trans flat.symm

/-- The same complete-block kernel used by the Pad runner, projected to one
of its two parity accumulators. The zero-block branch remains proved. -/
def padMoment (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (bit : Fin 2) : K :=
  numericSum extensionOps PiCCSSourceImages.blockCount fun block =>
    let value := PiCCSPadBlockMoment.blockMoment (basis gamma).1 (weight input) block (blocks witness gamma block)
    if bit.val = 0 then value.1 else value.2

private theorem padMoment_value (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (bit : Fin 2) :
    padMoment input witness gamma bit =
      numericSum extensionOps (Phi81CarrierLayout.carrierWidth PiCCSSourceImages.logicalWidth) (fun row =>
        if row % 2 = bit.val then extensionOps.mul (weight input (row / 2)) (padRow input witness gamma row)
        else extensionOps.zero) := by
  have flat := PiCCSCarriedAccumulation.blockMoments_eq_flat_sum
    (basis gamma).1 (weight input) (blocks witness gamma) PiCCSSourceImages.blockCount bit
    (padRow input witness gamma) (pad_dot input witness gamma)
  rw [carrier_count] at flat
  exact flat

/-- One actual numeric matrix row, with a failure retained as none. The
selected row bound keeps every accumulated index inside the Boolean domain. -/
def matrixRow?
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (row : Nat) : Option K :=
  if inside : row < 2 ^ cubeVariables then
    (PiCCSCarriedSource.matrixAt? witness gamma
      (NumericBooleanDomain.vertex cubeVariables ⟨row, inside⟩)).map (PiCCSCarriedSource.slotTotal gamma)
  else none

private theorem matrixRow?_value (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (row : Nat) (inside : row < 2 ^ cubeVariables) :
    matrixRow? witness gamma row = some (matrixRow input witness gamma row) := by
  rw [matrixRow?, dif_pos inside, matrixRow, atRow, dif_pos inside]
  exact PiCCSCarriedSource.matrix_sourceProtocolData input witness gamma _

/-- Scalar projection of the complete optional matrix-row accumulation.
It uses the same slot sum and absolute row parity as the runner. -/
def matrixMoment? (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (bit : Fin 2) : Option K :=
  Nat.fold selectedProgram.rowCount (fun row _ previous => do
    let total ← previous
    let value ← matrixRow? witness gamma row
    pure (extensionOps.add total (if row % 2 = bit.val then
      extensionOps.mul (weight input (row / 2)) value else extensionOps.zero)))
    (some extensionOps.zero)

private theorem optionalSum_value (action : Nat → Option K) (transform : Nat → K → K)
    (value : Nat → K) (count : Nat) :
    (∀ row, row < count → action row = some (value row)) →
    Nat.fold count (fun row _ previous => do
      let total ← previous
      let returned ← action row
      pure (extensionOps.add total (transform row returned))) (some extensionOps.zero) =
      some (numericSum extensionOps count (fun row => transform row (value row))) := by
  induction count with
  | zero => intro _; rfl
  | succ count ih =>
      intro returned
      have previous := ih (fun row inside => returned row (Nat.lt_trans inside (Nat.lt_succ_self count)))
      rw [Nat.fold_succ, previous, returned count (Nat.lt_succ_self count)]
      simp only [numericSum, Nat.fold_succ]
      rfl

private theorem matrixMoment?_value (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (bit : Fin 2) :
    matrixMoment? input witness gamma bit =
      some (numericSum extensionOps selectedProgram.rowCount (fun row =>
        if row % 2 = bit.val then extensionOps.mul (weight input (row / 2)) (matrixRow input witness gamma row)
        else extensionOps.zero)) := by
  apply optionalSum_value
  intro row inside
  exact matrixRow?_value input witness gamma row (Nat.lt_of_lt_of_le inside rows_fit)

private theorem endpoint_bound (pair : Fin (2 ^ 27)) (bit : Fin 2) :
    2 * pair.val + bit.val < 2 ^ cubeVariables := by
  have pairBound := pair.isLt
  have bitBound := bit.isLt
  change 2 * pair.val + bit.val < 268435456
  change pair.val < 134217728 at pairBound
  omega

private theorem endpoint_vertex (pair : Fin (2 ^ 27)) (bit : Fin 2) :
    PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) (bit.val == 1)
        (NumericBooleanDomain.vertex 27 pair) =
      NumericBooleanDomain.vertex cubeVariables ⟨2 * pair.val + bit.val, endpoint_bound pair bit⟩ := by
  let vertex := PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) (bit.val == 1)
    (NumericBooleanDomain.vertex 27 pair)
  have indexed : NumericBooleanDomain.index vertex = 2 * pair.val + bit.val := by
    have casesBit : bit.val = 0 ∨ bit.val = 1 := by have bound := bit.isLt; omega
    rcases casesBit with low | high
    · simp only [vertex, low, show (0 == 1) = false by decide]
      change 0 + 2 * NumericBooleanDomain.index (NumericBooleanDomain.vertex 27 pair) = 2 * pair.val + 0
      rw [NumericBooleanDomain.index_vertex]
      omega
    · simp only [vertex, high, show (1 == 1) = true by decide]
      change 1 + 2 * NumericBooleanDomain.index (NumericBooleanDomain.vertex 27 pair) = 2 * pair.val + 1
      rw [NumericBooleanDomain.index_vertex]
      omega
  calc
    _ = NumericBooleanDomain.vertex cubeVariables
        ⟨NumericBooleanDomain.index vertex, NumericBooleanDomain.index_lt_twoPow vertex⟩ :=
      (NumericBooleanDomain.vertex_index vertex).symm
    _ = _ := congrArg (NumericBooleanDomain.vertex cubeVariables) (Fin.ext indexed)

private theorem padRow_zero (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (row : Nat)
    (lower : Phi81CarrierLayout.carrierWidth PiCCSSourceImages.logicalWidth ≤ row)
    (upper : row < 2 ^ cubeVariables) : padRow input witness gamma row = extensionOps.zero := by
  rw [padRow, atRow, dif_pos upper]
  apply PiCCSCarriedSource.originalPad_padding
  rw [NumericBooleanDomain.index_vertex]
  have width := fullShape_carrierWidth PiCCSSourceImages.logicalWidth PiCCSSourceImages.publicFits
  change PiCCSSourceImages.shape.carrierWidth = _ at width
  rw [width]
  exact lower

private theorem matrixRow_zero (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (row : Nat) (lower : selectedProgram.rowCount ≤ row) (upper : row < 2 ^ cubeVariables) :
    matrixRow input witness gamma row = extensionOps.zero := by
  rw [matrixRow, atRow, dif_pos upper]
  apply PiCCSCarriedSource.originalMatrix_padding
  simpa only [NumericBooleanDomain.index_vertex] using lower

/-- The selected computed Pad and optional matrix accumulators give the
complete original-source carried moment. No source-correctness, row-coverage,
or assignment-validity hypothesis remains. The optional result retains all
matrix lookup failures, which the selected source theorem rules out. -/
theorem selected_moment (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (gamma : K) (bit : Fin 2) :
    (matrixMoment? input witness gamma bit).map (fun matrix =>
      extensionOps.add (padMoment input witness gamma bit)
        (extensionOps.mul (powers gamma productionShape.matrixEvaluationOffset) matrix)) =
      some (PiCCSCarriedMoments.moment extensionOps (sourceData input witness) gamma
        (by decide : productionShape.cubeVariables = 27 + 1) (bit.val == 1)) := by
  rw [matrixMoment?_value, Option.map_some, padMoment_value]
  apply congrArg some
  have complete := PiCCSCarriedAccumulation.moment_of_exact_row_callbacks extensionOps extensionLaws
    (sourceData input witness) gamma (by decide : productionShape.cubeVariables = 27 + 1)
    (padRow input witness gamma) (matrixRow input witness gamma)
    (Phi81CarrierLayout.carrierWidth PiCCSSourceImages.logicalWidth) selectedProgram.rowCount bit
    selectedRelation.cubeFits rows_fit
    (padRow_zero input witness gamma) (matrixRow_zero input witness gamma)
    (by
      intro pair
      rw [padRow, atRow, dif_pos (endpoint_bound pair bit), ← endpoint_vertex pair bit]
      rfl)
    (by
      intro pair
      rw [matrixRow, atRow, dif_pos (endpoint_bound pair bit), ← endpoint_vertex pair bit]
      rfl)
  rw [source_prior input witness] at complete
  exact complete

end NightstreamFPrime.Export.Stage1.PiCCSCarriedComplete
