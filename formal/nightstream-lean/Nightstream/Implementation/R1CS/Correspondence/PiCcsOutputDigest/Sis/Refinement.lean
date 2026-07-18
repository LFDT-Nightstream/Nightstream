import Nightstream.Implementation.R1CS.Core.SeededPhi81
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.CanonicalWord
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Sis.Semantics

/-!
Generic refinement from a compact seeded-Phi81 R1CS block to the independent
SIS linear-map semantics.

Assurance tier: implementation/R1CS correspondence. The conclusions are exact
coordinate and flattened equalities. They are conditional on canonical
input-word agreement and on `Block.Holds`; they never assume a commitment or
digest claim.

Owns: projection of a compact production block to an abstract `LinearMap`;
the word-agreement contract; equality of sparse production terms and semantic
terms; and coordinate-level and flattened output refinement.

Does not own: any concrete `Pi_CCS` block; proof of canonical word rows;
generated seed correctness; Rust/ChaCha stream conformance; Poseidon2;
collision resistance; transcript authority; row removal; or cost totals.

Emits constraints: no.

Authority boundary: `mapOfBlock` exposes the coefficient function being
proved about. A later concrete module must separately pin that function to
the public protocol seed expansion. `Block.Holds` alone is not called a
cryptographic commitment theorem.

| Protocol | Phase | Constraint family | Theorem | Exact guarantee |
|---|---|---|---|---|
| `Pi_CCS` | output digest | canonical-word inputs | `WordAgreement` | every consumed input coordinate is the independent centered digit |
| `Pi_CCS` | output digest | seeded linear rows | `valueTerms_eq_semanticTerms` | sparse R1CS inputs equal abstract message terms |
| `Pi_CCS` | output digest | seeded linear rows | `outputCoordinate_eq` | each accepted output coordinate equals `Semantics.applyCoordinate` |
| `Pi_CCS` | output digest | seeded linear block | `outputs_eq_apply` | all output columns equal the flattened abstract linear-map result |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Refinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

/-- Forget production placement while retaining the exact coefficient
function computed by the compact block. -/
def mapOfBlock (block : SeededPhi81.Block) : Sis.Semantics.LinearMap where
  kappa := block.kappa
  messageCols := block.messageCols
  coefficient := block.coefficient

/-- Exact premise required at a compact block's word boundary. -/
structure WordAgreement (block : SeededPhi81.Block) (fields : List Nat)
    (assignment : Nat -> Nat) : Prop where
  wordWidth : block.wordWidth = Sis.Semantics.digitCount
  fieldCount : fields.length = block.wordStarts.length
  digit : forall wordIndex digitIndex,
    wordIndex < fields.length -> digitIndex < Sis.Semantics.digitCount ->
    assignment (block.wordStarts.getD wordIndex 0 + digitIndex) =
      Sis.Semantics.canonicalDigit (fields.getD wordIndex 0) digitIndex

/-- Production sparse terms after replacing columns by their assigned
values. This definition is only a bridge object, not the independent map. -/
def valueTerms (block : SeededPhi81.Block) (assignment : Nat -> Nat)
    (output coordinate : Nat) : List (Nat × Nat) :=
  (block.terms output coordinate).map fun term =>
    (assignment term.1, term.2)

theorem lcEval_eq_evalValueTerms (block : SeededPhi81.Block)
    (assignment : Nat -> Nat) (output coordinate : Nat) :
    lcEval assignment (block.terms output coordinate) =
      Sis.Semantics.evalTerms (valueTerms block assignment output coordinate) := by
  simp [lcEval, Sis.Semantics.evalTerms, valueTerms, List.foldl_map,
    Sis.Semantics.modulus, goldilocksP,
    Nightstream.SuperNeo.Concrete.goldilocksModulus]

/-- The independent and production centered-digit formulas are definitionally
the same arithmetic statement, despite living in opposite dependency layers. -/
theorem abstractDigit_eq_productionDigit (fieldValue index : Nat) :
    Sis.Semantics.canonicalDigit fieldValue index =
      ShiftedTernaryCanonicalWord.canonicalDigit fieldValue index := by
  rfl

/-- Exact sparse-term refinement. The proof follows the public row-major
index, including the final zero padding, instead of comparing sampled outputs. -/
theorem valueTerms_eq_semanticTerms
    {block : SeededPhi81.Block} {fields : List Nat}
    {assignment : Nat -> Nat}
    (agreement : WordAgreement block fields assignment)
    (output coordinate : Nat) :
    valueTerms block assignment output coordinate =
      Sis.Semantics.coordinateTerms (mapOfBlock block) fields output coordinate := by
  unfold valueTerms SeededPhi81.Block.terms Sis.Semantics.coordinateTerms
  simp only [List.map_flatMap, List.map_filterMap]
  apply congrArg List.flatten
  apply List.map_congr_left
  intro messageColumn _messageColumnMember
  congr 1
  funext messageRow
  let index := messageRow * block.messageCols + messageColumn
  have digitCountNonzero : Sis.Semantics.digitCount ≠ 0 := by decide
  by_cases inBounds : index < fields.length * Sis.Semantics.digitCount
  · have blockBounds :
        index < block.wordStarts.length * block.wordWidth := by
      simpa [agreement.fieldCount, agreement.wordWidth] using inBounds
    have wordIndexLt : index / Sis.Semantics.digitCount < fields.length := by
      apply Nat.div_lt_of_lt_mul
      simpa [Nat.mul_comm] using inBounds
    have digitIndexLt :
        index % Sis.Semantics.digitCount < Sis.Semantics.digitCount :=
      Nat.mod_lt _ (by decide)
    have digitValue := agreement.digit
      (index / Sis.Semantics.digitCount) (index % Sis.Semantics.digitCount)
      wordIndexLt digitIndexLt
    have blockBoundsExpanded :
        messageRow * block.messageCols + messageColumn <
          block.wordStarts.length * Sis.Semantics.digitCount := by
      simpa [index, agreement.wordWidth] using blockBounds
    have inBoundsExpanded :
        messageRow * block.messageCols + messageColumn <
          fields.length * Sis.Semantics.digitCount := by
      simpa [index] using inBounds
    have digitValueExpanded := digitValue
    simp only [index] at digitValueExpanded
    by_cases coefficientZero :
        block.coefficient output messageColumn messageRow coordinate = 0
    · simp [SeededPhi81.Block.bitColumn, agreement.wordWidth,
        digitCountNonzero, blockBoundsExpanded,
        Sis.Semantics.messageIndex, mapOfBlock,
        inBoundsExpanded, coefficientZero]
    · simp [SeededPhi81.Block.bitColumn, agreement.wordWidth,
        digitCountNonzero, blockBoundsExpanded,
        Sis.Semantics.messageValue, Sis.Semantics.messageIndex, mapOfBlock,
        inBoundsExpanded, coefficientZero]
      exact digitValueExpanded
  · have blockBounds :
        ¬ (index < block.wordStarts.length * block.wordWidth) := by
      simpa [agreement.fieldCount, agreement.wordWidth] using inBounds
    have blockBoundsExpanded :
        ¬ (messageRow * block.messageCols + messageColumn <
          block.wordStarts.length * Sis.Semantics.digitCount) := by
      simpa [index, agreement.wordWidth] using blockBounds
    have inBoundsExpanded :
        ¬ (messageRow * block.messageCols + messageColumn <
          fields.length * Sis.Semantics.digitCount) := by
      simpa [index] using inBounds
    simp [SeededPhi81.Block.bitColumn, agreement.wordWidth,
      digitCountNonzero, blockBoundsExpanded,
      Sis.Semantics.messageIndex, mapOfBlock,
      inBoundsExpanded]

/-- Every output row of a holding compact block is exactly the corresponding
coordinate of the assignment-free abstract linear map. -/
theorem outputCoordinate_eq
    {block : SeededPhi81.Block} {fields : List Nat}
    {assignment : Nat -> Nat}
    (holds : block.Holds assignment)
    (agreement : WordAgreement block fields assignment)
    (output coordinate : Nat) (outputLt : output < block.kappa)
    (coordinateLt : coordinate < Sis.Semantics.dimension) :
    assignment
        (block.outputColumns.getD
          (output * SeededPhi81.dimension + coordinate) 0) =
      Sis.Semantics.applyCoordinate (mapOfBlock block) fields output coordinate := by
  have definitionMember :
      block.definition output coordinate ∈ block.definitions := by
    unfold SeededPhi81.Block.definitions
    rw [List.mem_flatMap]
    refine ⟨output, List.mem_range.mpr outputLt, ?_⟩
    exact List.mem_map.mpr
      ⟨coordinate, List.mem_range.mpr coordinateLt, rfl⟩
  have definitionHolds := holds (block.definition output coordinate)
    definitionMember
  change assignment
      (block.outputColumns.getD
        (output * SeededPhi81.dimension + coordinate) 0) =
      lcEval assignment (block.terms output coordinate) at definitionHolds
  rw [definitionHolds, lcEval_eq_evalValueTerms,
    valueTerms_eq_semanticTerms agreement]
  rfl

private theorem map_range_eq_ofFn
    {Item : Type}
    (count : Nat)
    (value : Nat -> Item) :
    (List.range count).map value =
      List.ofFn fun index : Fin count => value index.val := by
  apply List.ext_get
  · simp
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_map, List.getElem_range,
      List.getElem_ofFn]

private theorem getD_ofFn
    {Item : Type}
    {count : Nat}
    (items : Fin count -> Item)
    (index : Fin count)
    (default : Item) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

private theorem getD_flatten_ofFn_ofFn
    {Item : Type}
    {outer inner : Nat}
    (items : Fin outer -> Fin inner -> Item)
    (outerIndex : Fin outer)
    (innerIndex : Fin inner)
    (default : Item) :
    ((List.ofFn fun outerPosition =>
        List.ofFn fun innerPosition =>
          items outerPosition innerPosition).flatten).getD
      (outerIndex.val * inner + innerIndex.val) default =
        items outerIndex innerIndex := by
  induction outer with
  | zero => exact Fin.elim0 outerIndex
  | succ outer inductionHypothesis =>
      refine Fin.cases ?_ (fun index => ?_) outerIndex
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_zero,
          Nat.zero_mul, Nat.zero_add]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_left (by simp)]
        change (List.ofFn (items 0)).getD innerIndex.val default = _
        exact getD_ofFn _ innerIndex default
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_succ]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_right (by
          simp only [List.length_ofFn]
          rw [Nat.add_mul, Nat.one_mul]
          omega)]
        simp only [List.length_ofFn]
        change ((List.ofFn fun outerPosition =>
          List.ofFn fun innerPosition =>
            items outerPosition.succ innerPosition).flatten).getD
          ((index.val + 1) * inner + innerIndex.val - inner) default = _
        have indexArithmetic :
            (index.val + 1) * inner + innerIndex.val - inner =
              index.val * inner + innerIndex.val := by
          rw [Nat.add_mul, Nat.one_mul]
          omega
        rw [indexArithmetic]
        exact inductionHypothesis
          (fun outerPosition innerPosition =>
            items outerPosition.succ innerPosition)
          index

/-- A valid holding compact block materializes the complete abstract SIS
result in its declared output-column order. This is only a linear-map
refinement; it gives the columns no digest or transcript authority. -/
theorem outputs_eq_apply
    {block : SeededPhi81.Block}
    {fields : List Nat}
    {assignment : Nat -> Nat}
    (valid : block.Valid)
    (holds : block.Holds assignment)
    (agreement : WordAgreement block fields assignment) :
    block.outputColumns.map assignment =
      Sis.Semantics.apply (mapOfBlock block) fields := by
  have dimensionEq :
      SeededPhi81.dimension = Sis.Semantics.dimension := by
    rfl
  have outputLength :
      block.outputColumns.length =
        SeededPhi81.dimension * block.kappa :=
    valid.2.2.2.2.2.1
  have sameLength :
      (block.outputColumns.map assignment).length =
        (Sis.Semantics.apply (mapOfBlock block) fields).length := by
    rw [List.length_map, outputLength, Sis.Semantics.apply_length]
    simp [mapOfBlock, dimensionEq, Nat.mul_comm]
  have applyShape :
      Sis.Semantics.apply (mapOfBlock block) fields =
        (List.ofFn fun output : Fin block.kappa =>
          List.ofFn fun coordinate : Fin Sis.Semantics.dimension =>
            Sis.Semantics.applyCoordinate (mapOfBlock block) fields
              output.val coordinate.val).flatten := by
    simp [Sis.Semantics.apply, List.flatMap_def, map_range_eq_ofFn,
      mapOfBlock]
  apply List.ext_getElem sameLength
  intro index leftLt rightLt
  have sourceLt : index < block.outputColumns.length := by
    simpa using leftLt
  have indexLt : index < block.kappa * Sis.Semantics.dimension := by
    simpa [outputLength, Nat.mul_comm] using sourceLt
  let output : Fin block.kappa :=
    ⟨index / Sis.Semantics.dimension,
      (Nat.div_lt_iff_lt_mul
        (by decide : 0 < Sis.Semantics.dimension)).2 indexLt⟩
  let coordinate : Fin Sis.Semantics.dimension :=
    ⟨index % Sis.Semantics.dimension,
      Nat.mod_lt _ (by decide)⟩
  have indexEq :
      output.val * Sis.Semantics.dimension + coordinate.val = index := by
    simpa [output, coordinate, Nat.mul_comm] using
      Nat.div_add_mod index Sis.Semantics.dimension
  have physicalAt := outputCoordinate_eq holds agreement
    output.val coordinate.val output.isLt coordinate.isLt
  rw [dimensionEq, indexEq] at physicalAt
  have semanticAt :
      (Sis.Semantics.apply (mapOfBlock block) fields).getD index 0 =
        Sis.Semantics.applyCoordinate (mapOfBlock block) fields
          output.val coordinate.val := by
    rw [applyShape, ← indexEq]
    exact getD_flatten_ofFn_ofFn
      (fun output coordinate =>
        Sis.Semantics.applyCoordinate (mapOfBlock block) fields
          output.val coordinate.val)
      output coordinate 0
  simp only [List.getElem_map]
  rw [List.getElem_eq_getD (l := block.outputColumns) (h := sourceLt) 0,
    physicalAt,
    List.getElem_eq_getD
      (l := Sis.Semantics.apply (mapOfBlock block) fields)
      (h := rightLt) 0,
    semanticAt]

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.Refinement
