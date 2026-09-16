import NightstreamFPrime.Export.Stage1.PiCCSFirstRoundComposition
import NightstreamFPrime.Export.Stage1.PiCCSPadPrefix
import NightstreamFPrime.Export.Stage1.PiCCSPrefixRound

/-! Proof-only transport of original carried scalar reads through the existing
prefix fold to the original weighted message fields. Pad and matrix prefixes
have separate extents. File decoding and parity accumulation are not owned here. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedPrefixSource

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle
open FiniteSumAlgebra (sumMap)
open PiCCSAggregatedImages (selectedProgram selectedLayout)

private abbrev power (gamma : K) := TargetPolynomial.power extensionOps.toOps gamma
private abbrev basis (gamma : K) :=
  PiCCSAggregatedImages.prepare (PiDECParentSparseRead.prepare ()) (power gamma)
private abbrev blocks (masks : Array (Array (Nat × Nat))) (gamma : K) :=
  PiCCSAggregatedImages.combinedBlock (power gamma) (PiCCSNormSource.assignments masks)

private theorem rows_fit : selectedProgram.rowCount ≤ 2 ^ cubeVariables := by
  rw [PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits]
  exact PerApplicationFixedPoint.structuralPlan_rowCount_le
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

-- These prefixes collect only the existing scalar initializers. They are
-- proof-side array views, not an executable replay or a second prover.
def padPrefix (masks : Array (Array (Nat × Nat))) (gamma : K) : Array K :=
  Array.ofFn fun column : Fin PiCCSSourceImages.shape.carrierWidth =>
    (PiCCSPadPrefix.blockValues (basis gamma).1
      (blocks masks gamma (column.val / ringDegree))).get
        ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩

def matrixPrefix? (masks : Array (Array (Nat × Nat))) (gamma : K) : Option (Array K) :=
  Array.ofFnM fun row : Fin selectedProgram.rowCount =>
    PiCCSCarriedComplete.matrixRow? (PiCCSFirstRoundComposition.witness masks) gamma row.val

private def reference {arity : Nat} (value : BooleanVertex arity → K)
    (count : Nat) (fits : count ≤ 2 ^ arity) : Array K :=
  Array.ofFn fun row : Fin count =>
    value (NumericBooleanDomain.vertex arity ⟨row.val, Nat.lt_of_lt_of_le row.isLt fits⟩)

private theorem pad_column_source (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K)
    (column : Fin PiCCSSourceImages.shape.carrierWidth) :
    (PiCCSPadPrefix.blockValues (basis gamma).1
      (blocks masks gamma (column.val / ringDegree))).get
        ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ =
      PiCCSCarriedSource.originalPad input (PiCCSFirstRoundComposition.witness masks) gamma
        (selectedLayout.toVertex column) := by
  rw [PiCCSPadPrefix.blockValues_value]
  exact PiCCSCarriedSource.pad_column_sourceProtocolData input
    (PiCCSFirstRoundComposition.witness masks) gamma column

/-- All retained Pad lanes, including the full carrier tail, come from the
original source Pad family. The whole-block zero shortcut is already proved. -/
theorem padPrefix_value (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K) :
    padPrefix masks gamma =
      reference (PiCCSCarriedSource.originalPad input
        (PiCCSFirstRoundComposition.witness masks) gamma)
        PiCCSSourceImages.shape.carrierWidth selectedLayout.columns_le := by
  unfold padPrefix reference
  apply congrArg Array.ofFn
  funext column
  exact pad_column_source input masks gamma column

/-- Every active matrix scalar load succeeds and gives the original local
gamma total. No successful-load or expected-value premise is required. -/
theorem matrixPrefix_value (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K) :
    matrixPrefix? masks gamma =
      some (reference (PiCCSCarriedSource.originalMatrix input
        (PiCCSFirstRoundComposition.witness masks) gamma)
        selectedProgram.rowCount rows_fit) := by
  unfold matrixPrefix? reference
  have actions :
      (fun row : Fin selectedProgram.rowCount =>
        PiCCSCarriedComplete.matrixRow? (PiCCSFirstRoundComposition.witness masks) gamma row.val) =
      (fun row : Fin selectedProgram.rowCount =>
        some (PiCCSCarriedSource.originalMatrix input
          (PiCCSFirstRoundComposition.witness masks) gamma
          (NumericBooleanDomain.vertex cubeVariables
            ⟨row.val, Nat.lt_of_lt_of_le row.isLt rows_fit⟩))) := by
    funext row
    rw [PiCCSCarriedComplete.matrixRow?, dif_pos (Nat.lt_of_lt_of_le row.isLt rows_fit)]
    exact PiCCSCarriedSource.matrix_sourceProtocolData input
      (PiCCSFirstRoundComposition.witness masks) gamma _
  rw [actions]
  exact Array.ofFnM_pure

private theorem padPrefix_block (masks : Array (Array (Nat × Nat))) (gamma : K)
    (block : Fin PiCCSSourceImages.blockCount) (lane : Fin ringDegree) :
    (padPrefix masks gamma).getD (block.val * ringDegree + lane.val) K.zero =
      PiCCSWeightedBasis.dotK ((basis gamma).1.get lane) (blocks masks gamma block.val).get := by
  let typedBlock := Fin.cast PiCCSCarriedSource.blockCount_eq_authority block
  have live : block.val * ringDegree + lane.val < PiCCSSourceImages.shape.carrierWidth :=
    Phi81CarrierLayout.flatIndex_lt_carrierWidth
      (logicalWidth := PiCCSSourceImages.logicalWidth) typedBlock lane
  have laneBound : lane.val < 54 := lane.isLt
  have quotient : (block.val * ringDegree + lane.val) / ringDegree = block.val := by
    change (block.val * 54 + lane.val) / 54 = block.val
    omega
  have remainder : (block.val * ringDegree + lane.val) % ringDegree = lane.val := by
    change (block.val * 54 + lane.val) % 54 = lane.val
    omega
  simp only [padPrefix, Array.getD_eq_getD_getElem?, Array.getElem?_ofFn,
    dif_pos live, Option.getD_some, quotient, remainder, PiCCSPadPrefix.blockValues_value]

/-- Each actual 27-value block emitted by the first retained Pad fold is
exactly its part of the complete full-carrier scalar fold. Block boundaries
do not alter the canonical adjacent pairing. -/
theorem pad_foldedBlock (masks : Array (Array (Nat × Nat))) (gamma challenge : K)
    (block : Fin PiCCSSourceImages.blockCount) (pair : Fin PiCCSNormSource.PairCount) :
    (PiCCSPadPrefix.foldedBlock (basis gamma).1
      (blocks masks gamma block.val) challenge).getD pair.val K.zero =
      (PrefixFold.foldOne extensionOps (padPrefix masks gamma) challenge).getD
        (PiCCSNormSource.pairIndex block.val pair) K.zero := by
  have lowLive : 2 * pair.val < ringDegree := (PiCCSNormSource.lowLane pair).isLt
  have highLive : 2 * pair.val + 1 < ringDegree := (PiCCSNormSource.highLane pair).isLt
  rw [PiCCSPadPrefix.foldedBlock_getD, dif_pos lowLive, dif_pos highLive]
  have folded := PrefixFold.foldOne_getD extensionOps extensionLaws
    (padPrefix masks gamma) challenge (PiCCSNormSource.pairIndex block.val pair)
  rw [← PiCCSNormSource.highLane_global block.val pair,
    ← PiCCSNormSource.lowLane_global block.val pair] at folded
  have low := padPrefix_block masks gamma block (PiCCSNormSource.lowLane pair)
  have high := padPrefix_block masks gamma block (PiCCSNormSource.highLane pair)
  exact (congrArg₂ (PrefixFold.interpolate extensionOps challenge) low high).symm.trans folded.symm

private theorem reference_table {arity : Nat} (value : BooleanVertex arity → K)
    (count : Nat) (fits : count ≤ 2 ^ arity)
    (padding : ∀ vertex, count ≤ NumericBooleanDomain.index vertex → value vertex = K.zero) :
    PrefixFold.zeroExtend extensionOps arity (reference value count fits) =
      BooleanTable.tabulate value := by
  unfold PrefixFold.zeroExtend
  apply congrArg BooleanTable.tabulate
  funext vertex
  simp only [reference, Array.getD_eq_getD_getElem?, Array.getElem?_ofFn]
  by_cases live : NumericBooleanDomain.index vertex < count
  · simp only [dif_pos live, Option.getD_some, NumericBooleanDomain.vertex_index]
  · simp only [dif_neg live, Option.getD_none]
    exact (padding vertex (Nat.le_of_not_lt live)).symm

private theorem fold_reference_evaluate {arity remaining : Nat}
    (value : BooleanVertex arity → K) (count : Nat) (fits : count ≤ 2 ^ arity)
    (padding : ∀ vertex, count ≤ NumericBooleanDomain.index vertex → value vertex = K.zero)
    (challenges : List K) (dimension : arity = remaining + challenges.length)
    (suffix : CubePoint K remaining) :
    (PrefixFold.zeroExtend extensionOps remaining
      (PrefixFold.foldPrefix extensionOps (reference value count fits) challenges)).evaluate
        extensionOps suffix =
      (BooleanTable.tabulate value).evaluate extensionOps
        ⟨challenges ++ suffix.coordinates, by
          simp only [List.length_append, suffix.dimension]
          omega⟩ := by
  subst arity
  rw [PrefixFold.foldPrefix_evaluate extensionOps extensionLaws _ challenges suffix
    (by simpa only [reference, Array.size_ofFn] using fits),
    reference_table value count fits padding]

private theorem weighted_evaluate {Index : Type} {arity : Nat}
    (indices : List Index) (weights : Index → K) (tables : Index → BooleanTable K arity)
    (point : CubePoint K arity) :
    (BooleanTable.tabulate fun vertex =>
      sumMap extensionOps indices (fun index =>
        extensionOps.mul (weights index) ((tables index).valueAt vertex))).evaluate
          extensionOps point =
      sumMap extensionOps indices (fun index =>
        extensionOps.mul (weights index) ((tables index).evaluate extensionOps point)) := by
  rw [← BooleanReproduction.equalityWeighted_tabulate_eq_evaluate extensionOps extensionLaws,
    BooleanReproduction.equalityWeighted_sumMap extensionOps extensionLaws]
  apply FiniteSumAlgebra.sumMap_congr
  intro index _
  apply congrArg (extensionOps.mul (weights index))
  exact (BooleanTable.evaluate_eq_equalityWeightedSum extensionOps extensionLaws
    (tables index) point).symm

private theorem pad_evaluate (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K)
    (point : CubePoint K cubeVariables) :
    (BooleanTable.tabulate (PiCCSCarriedSource.originalPad input
      (PiCCSFirstRoundComposition.witness masks) gamma)).evaluate extensionOps point =
      sumMap extensionOps (canonicalPadCoordinates productionShape) (fun coordinate =>
        extensionOps.mul (power gamma coordinate.localGammaExponent)
          ((ProtocolPolynomial.messageAt extensionOps
            (PiCCSFirstRoundComposition.sourceData input masks) point).padImage coordinate)) := by
  exact weighted_evaluate (canonicalPadCoordinates productionShape)
    (fun coordinate => power gamma coordinate.localGammaExponent)
    (PiCCSFirstRoundComposition.sourceData input masks).padImages point

private theorem matrix_evaluate (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K)
    (point : CubePoint K cubeVariables) :
    (BooleanTable.tabulate (PiCCSCarriedSource.originalMatrix input
      (PiCCSFirstRoundComposition.witness masks) gamma)).evaluate extensionOps point =
      sumMap extensionOps (canonicalMatrixCoordinates productionShape) (fun coordinate =>
        extensionOps.mul (power gamma coordinate.localGammaExponent)
          ((ProtocolPolynomial.messageAt extensionOps
            (PiCCSFirstRoundComposition.sourceData input masks) point).matrixImage coordinate)) := by
  exact weighted_evaluate (canonicalMatrixCoordinates productionShape)
    (fun coordinate => power gamma coordinate.localGammaExponent)
    (PiCCSFirstRoundComposition.sourceData input masks).matrixImages point

/-- Every folded Pad value is the full local-gamma sum of original padImage
evaluations at prefix ++ suffix. Only the full carrier's zero suffix is used. -/
theorem padPrefix_evaluate (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K) (challenges : List K)
    {remaining : Nat} (dimension : cubeVariables = remaining + challenges.length)
    (suffix : CubePoint K remaining) :
    (PrefixFold.zeroExtend extensionOps remaining
      (PrefixFold.foldPrefix extensionOps (padPrefix masks gamma) challenges)).evaluate
        extensionOps suffix =
      sumMap extensionOps (canonicalPadCoordinates productionShape) (fun coordinate =>
        extensionOps.mul (power gamma coordinate.localGammaExponent)
          ((ProtocolPolynomial.messageAt extensionOps
            (PiCCSFirstRoundComposition.sourceData input masks)
            ⟨challenges ++ suffix.coordinates, by
              simp only [List.length_append, suffix.dimension]
              change challenges.length + remaining = cubeVariables
              omega⟩).padImage coordinate)) := by
  rw [padPrefix_value input masks gamma]
  exact (fold_reference_evaluate _ PiCCSSourceImages.shape.carrierWidth selectedLayout.columns_le
    (PiCCSCarriedSource.originalPad_padding input
      (PiCCSFirstRoundComposition.witness masks) gamma) challenges dimension suffix).trans
    (pad_evaluate input masks gamma _)

/-- Every optional matrix prefix returns the full local-gamma sum of original
matrixImage evaluations. Padding starts at the active program row count.
The global matrixEvaluationOffset factor is not applied at this boundary. -/
theorem matrixPrefix_evaluate (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K) (challenges : List K)
    {remaining : Nat} (dimension : cubeVariables = remaining + challenges.length)
    (suffix : CubePoint K remaining) :
    (matrixPrefix? masks gamma).map (fun values =>
      (PrefixFold.zeroExtend extensionOps remaining
        (PrefixFold.foldPrefix extensionOps values challenges)).evaluate extensionOps suffix) =
      some (sumMap extensionOps (canonicalMatrixCoordinates productionShape) (fun coordinate =>
        extensionOps.mul (power gamma coordinate.localGammaExponent)
          ((ProtocolPolynomial.messageAt extensionOps
            (PiCCSFirstRoundComposition.sourceData input masks)
            ⟨challenges ++ suffix.coordinates, by
              simp only [List.length_append, suffix.dimension]
              change challenges.length + remaining = cubeVariables
              omega⟩).matrixImage coordinate))) := by
  rw [matrixPrefix_value input masks gamma, Option.map_some]
  apply congrArg some
  rw [fold_reference_evaluate _ _ rows_fit
    (PiCCSCarriedSource.originalMatrix_padding input
      (PiCCSFirstRoundComposition.witness masks) gamma) challenges dimension suffix]
  exact matrix_evaluate input masks gamma _

/-- Exact next-round Pad endpoint after any consumed challenge prefix. -/
theorem padPrefix_endpoint (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K) (challenges : List K)
    {remaining : Nat} (dimension : cubeVariables = challenges.length + remaining + 1)
    (bit : Bool) (suffix : BooleanVertex remaining) :
    (PrefixFold.foldPrefix extensionOps (padPrefix masks gamma) challenges).getD
        (NumericBooleanDomain.index (.cons bit suffix)) K.zero =
      sumMap extensionOps (canonicalPadCoordinates productionShape) (fun coordinate =>
        extensionOps.mul (power gamma coordinate.localGammaExponent)
          ((ProtocolPolynomial.messageAt extensionOps
            (PiCCSFirstRoundComposition.sourceData input masks)
            (PiCCSPrefixRound.point extensionOps challenges dimension
              (if bit then K.one else K.zero) suffix)).padImage coordinate)) := by
  have dimension' : cubeVariables = (remaining + 1) + challenges.length := by omega
  have value := padPrefix_evaluate input masks gamma challenges dimension'
    ((BooleanVertex.cons bit suffix).toCubePoint extensionOps)
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt extensionOps extensionLaws] at value
  simp only [PrefixFold.zeroExtend, BooleanTable.valueAt_tabulate] at value
  cases bit <;> simpa only [PiCCSPrefixRound.point,
    BooleanVertex.toCubePoint_coordinates, BooleanVertex.fieldCoordinates,
    SumCheckTruthPath.VertexEncoding.fieldCoordinates, Bool.false_eq_true,
    if_false, if_true] using value

/-- Exact next-round matrix endpoint, with successful scalar initialization
proved and the complete matrix family retained in canonical coordinate order. -/
theorem matrixPrefix_endpoint (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K) (challenges : List K)
    {remaining : Nat} (dimension : cubeVariables = challenges.length + remaining + 1)
    (bit : Bool) (suffix : BooleanVertex remaining) :
    (matrixPrefix? masks gamma).map (fun values =>
      (PrefixFold.foldPrefix extensionOps values challenges).getD
        (NumericBooleanDomain.index (.cons bit suffix)) K.zero) =
      some (sumMap extensionOps (canonicalMatrixCoordinates productionShape) (fun coordinate =>
        extensionOps.mul (power gamma coordinate.localGammaExponent)
          ((ProtocolPolynomial.messageAt extensionOps
            (PiCCSFirstRoundComposition.sourceData input masks)
            (PiCCSPrefixRound.point extensionOps challenges dimension
              (if bit then K.one else K.zero) suffix)).matrixImage coordinate))) := by
  have dimension' : cubeVariables = (remaining + 1) + challenges.length := by omega
  have value := matrixPrefix_evaluate input masks gamma challenges dimension'
    ((BooleanVertex.cons bit suffix).toCubePoint extensionOps)
  simp only [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt extensionOps extensionLaws,
    PrefixFold.zeroExtend, BooleanTable.valueAt_tabulate] at value
  cases bit <;> simpa only [PiCCSPrefixRound.point,
    BooleanVertex.toCubePoint_coordinates, BooleanVertex.fieldCoordinates,
    SumCheckTruthPath.VertexEncoding.fieldCoordinates, Bool.false_eq_true,
    if_false, if_true] using value

/-- The Pad initializer retains the complete carrier, including its tail. -/
theorem padPrefix_size (masks : Array (Array (Nat × Nat))) (gamma : K) :
    (padPrefix masks gamma).size = PiCCSSourceImages.shape.carrierWidth := by
  simp only [padPrefix, Array.size_ofFn]

/-- The selected matrix initializer succeeds at the exact active-row extent. -/
theorem matrixPrefix_size (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (gamma : K) :
    (matrixPrefix? masks gamma).map Array.size = some selectedProgram.rowCount := by
  rw [matrixPrefix_value input masks gamma, Option.map_some]
  simp only [reference, Array.size_ofFn]

end NightstreamFPrime.Export.Stage1.PiCCSCarriedPrefixSource
