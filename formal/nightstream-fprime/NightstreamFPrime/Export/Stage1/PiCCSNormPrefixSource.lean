import NightstreamFPrime.Export.Stage1.PiCCSFirstRoundComposition
import NightstreamFPrime.Export.Stage1.PiCCSPrefixCodeFold
import NightstreamFPrime.Export.Stage1.PiCCSPrefixRound

/-! Proof-only transport from the existing signed source decoder through the
existing retained norm folds to the original sourceAssignment message field.
The source index remains separate. File IO and range accumulation are not
proved here. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormPrefixSource

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle

private abbrev code (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (column : Nat) : Fin 3 :=
  PiCCSNormSource.sourceCode (masks[column / ringDegree]?.getD #[]) source
    ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩

private theorem fullShape_width (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (PaperAlgebra.FullShape logicalWidth publicFits).carrierWidth =
      Phi81CarrierLayout.carrierWidth logicalWidth := rfl

private theorem completeWidth :
    PiCCSSourceImages.shape.carrierWidth =
      PiCCSSourceImages.blockCount * ringDegree := by
  have width := fullShape_width PiCCSSourceImages.logicalWidth PiCCSSourceImages.publicFits
  have blocks : PiCCSSourceImages.blockCount * ringDegree =
      Phi81CarrierLayout.carrierWidth PiCCSSourceImages.logicalWidth := by
    rw [PiCCSCarriedSource.blockCount_eq_authority,
      Phi81CarrierLayout.blockCount_carrierWidth, ← Phi81CarrierLayout.carrierWidth_eq]
  exact width.trans blocks.symm

private theorem code_outside (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (column : Nat) (outside : PiCCSSourceImages.shape.carrierWidth ≤ column) :
    PiCCSNormCache.signedValue (code masks source column) = (0 : F) := by
  have beyond : masks.size ≤ column / ringDegree := by
    apply Nat.le_trans loaded
    apply (Nat.le_div_iff_mul_le (by decide : 0 < ringDegree)).2
    rw [← completeWidth]
    exact outside
  rw [code, PiCCSNormSource.signedValue_sourceCode,
    Array.getElem?_eq_none beyond, Option.getD_none]
  simp only [SignedUnitSourceInput.scalar, Array.getElem?_empty, Option.getD_none,
    Nat.zero_testBit, Bool.false_eq_true, if_false]

private theorem table_ext {arity : Nat} (left right : BooleanTable K arity)
    (equal : ∀ vertex, left.valueAt vertex = right.valueAt vertex) : left = right := by
  induction arity with
  | zero =>
      cases left with
      | leaf left =>
          cases right with
          | leaf right => exact congrArg BooleanTable.leaf (equal .nil)
  | succ arity ih =>
      cases left with
      | branch low high =>
          cases right with
          | branch otherLow otherHigh =>
              have lowEqual := ih low otherLow (fun vertex => equal (.cons false vertex))
              have highEqual := ih high otherHigh (fun vertex => equal (.cons true vertex))
              rw [lowEqual, highEqual]

private theorem decoded_prefix_value
    (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (count : Nat) (covered : PiCCSSourceImages.shape.carrierWidth ≤ count)
    (vertex : BooleanVertex cubeVariables) :
    (Array.ofFn fun entry : Fin count =>
      K.embed (PiCCSNormCache.signedValue (code masks source entry.val))).getD
        (NumericBooleanDomain.index vertex) K.zero =
      K.embed ((PiCCSNormSource.canonicalLayout ()).paddedValue
        (0 : F) (PiCCSNormSource.assignments masks source) vertex) := by
  by_cases live : NumericBooleanDomain.index vertex < PiCCSSourceImages.shape.carrierWidth
  · have stored : NumericBooleanDomain.index vertex < count := Nat.lt_of_lt_of_le live covered
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn,
      dif_pos stored, Option.getD_some]
    change K.embed (PiCCSNormCache.signedValue (code masks source
        (NumericBooleanDomain.index vertex))) =
      K.embed ((Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
        PiCCSSourceImages.shape.carrierWidth (PiCCSNormSource.canonicalLayout ()).columns_le).paddedValue
        (0 : F) (PiCCSNormSource.assignments masks source) vertex)
    simp only [ColumnLayout.paddedValue, Folding.PiCCS.CanonicalRowLayout.layout,
      dif_pos live]
    rw [code, PiCCSNormSource.signedValue_sourceCode]
    rfl
  · have outside := Nat.le_of_not_lt live
    have padding : (PiCCSNormSource.canonicalLayout ()).toColumn? vertex = none :=
      (Folding.PiCCS.CanonicalRowLayout.toColumn?_eq_none_iff cubeVariables
        PiCCSSourceImages.shape.carrierWidth (PiCCSNormSource.canonicalLayout ()).columns_le vertex).2 outside
    rw [ColumnLayout.paddedValue, padding]
    by_cases stored : NumericBooleanDomain.index vertex < count
    · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn,
        dif_pos stored, Option.getD_some]
      rw [code_outside masks source loaded _ outside]
    · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn,
        dif_neg stored, Option.getD_none]
      rfl

/-- A full decoded source prefix, with optional extra zero scalars, is the
actual source table of the selected statement. The length premise is about
the decoder's allocated block array, not a wanted source-value equality. -/
theorem decoded_prefix_table
    (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (count : Nat) (covered : PiCCSSourceImages.shape.carrierWidth ≤ count) :
    PrefixFold.zeroExtend extensionOps cubeVariables
      (Array.ofFn fun entry : Fin count =>
        K.embed (PiCCSNormCache.signedValue (code masks source entry.val))) =
      (PiCCSFirstRoundComposition.sourceData input masks).sourceAssignments source := by
  apply table_ext
  intro vertex
  rw [PrefixFold.zeroExtend, BooleanTable.valueAt_tabulate]
  exact (decoded_prefix_value masks source loaded count covered vertex).trans
    (PiCCSSourceImages.assignment_sourceProtocolData input
      (PiCCSFirstRoundComposition.witness masks) source vertex)

private theorem fold_table_evaluate {arity remaining : Nat}
    (table : BooleanTable K arity) (values : Array K)
    (tableEqual : PrefixFold.zeroExtend extensionOps arity values = table)
    (fits : values.size ≤ 2 ^ arity)
    (challenges : List K) (dimension : arity = remaining + challenges.length)
    (suffix : CubePoint K remaining) :
    (PrefixFold.zeroExtend extensionOps remaining
      (PrefixFold.foldPrefix extensionOps values challenges)).evaluate extensionOps suffix =
      table.evaluate extensionOps ⟨challenges ++ suffix.coordinates, by
        simp only [List.length_append, suffix.dimension]
        omega⟩ := by
  subst arity
  rw [PrefixFold.foldPrefix_evaluate extensionOps extensionLaws _ challenges suffix fits,
    tableEqual]

/-- The existing array fold evaluates the original source table after an
arbitrary challenge prefix. All 17 sources remain individually indexed. -/
theorem decoded_prefix_evaluate
    (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (count : Nat) (covered : PiCCSSourceImages.shape.carrierWidth ≤ count)
    (fits : count ≤ 2 ^ cubeVariables)
    (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = remaining + challenges.length)
    (suffix : CubePoint K remaining) :
    (PrefixFold.zeroExtend extensionOps remaining
      (PrefixFold.foldPrefix extensionOps
        (Array.ofFn fun entry : Fin count =>
          K.embed (PiCCSNormCache.signedValue (code masks source entry.val)))
        challenges)).evaluate extensionOps suffix =
      ((PiCCSFirstRoundComposition.sourceData input masks).sourceAssignments source).evaluate
        extensionOps ⟨challenges ++ suffix.coordinates, by
          simp only [List.length_append, suffix.dimension]
          change challenges.length + remaining = cubeVariables
          omega⟩ := by
  exact fold_table_evaluate _ _
    (decoded_prefix_table input masks source loaded count covered)
    (by simpa only [Array.size_ofFn] using fits) challenges dimension suffix

/-- Decode the actual 81-entry norm code table, then consume any remaining
challenges with the existing scalar fold. Its value is the original norm
source MLE. The two leading challenges and source order are unchanged. -/
theorem quadCodes_evaluate
    (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (groups : Nat) (covered : PiCCSSourceImages.shape.carrierWidth ≤ 4 * groups)
    (fits : 4 * groups ≤ 2 ^ cubeVariables)
    (first second : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = remaining + (first :: second :: challenges).length)
    (suffix : CubePoint K remaining) :
    (PrefixFold.zeroExtend extensionOps remaining
      (PrefixFold.foldPrefix extensionOps
        (PiCCSPrefixCodeFold.decode
          (PiCCSPrefixCodeFold.pairedTable (PiCCSPrefixNorm.values first) second)
          (PiCCSPrefixCodeFold.quadCodes (code masks source) 0 groups))
        challenges)).evaluate extensionOps suffix =
      ((PiCCSFirstRoundComposition.sourceData input masks).sourceAssignments source).evaluate
        extensionOps ⟨(first :: second :: challenges) ++ suffix.coordinates, by
          simp only [List.length_append, suffix.dimension]
          change (first :: second :: challenges).length + remaining = cubeVariables
          omega⟩ := by
  rw [PiCCSPrefixCodeFold.decode_quadCodes_twoFolds]
  simpa only [Nat.mul_zero, Nat.zero_add, PrefixFold.foldPrefix] using
    decoded_prefix_evaluate input masks source loaded (4 * groups) covered fits
      (first :: second :: challenges) dimension suffix

/-- The retained norm entry at either numeric endpoint is the exact
sourceAssignment component of messageAt used by PiCCSPrefixRound.
The unused suffix is Boolean; the consumed prefix remains arbitrary. -/
theorem quadCodes_endpoint
    (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (groups : Nat) (covered : PiCCSSourceImages.shape.carrierWidth ≤ 4 * groups)
    (fits : 4 * groups ≤ 2 ^ cubeVariables)
    (first second : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = (first :: second :: challenges).length + remaining + 1)
    (bit : Bool) (suffix : BooleanVertex remaining) :
    (PrefixFold.foldPrefix extensionOps
      (PiCCSPrefixCodeFold.decode
        (PiCCSPrefixCodeFold.pairedTable (PiCCSPrefixNorm.values first) second)
        (PiCCSPrefixCodeFold.quadCodes (code masks source) 0 groups))
      challenges).getD (NumericBooleanDomain.index (.cons bit suffix)) K.zero =
      (ProtocolPolynomial.messageAt extensionOps
        (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPrefixRound.point extensionOps (first :: second :: challenges) dimension
          (if bit then K.one else K.zero) suffix)).sourceAssignment source := by
  have dimension' : cubeVariables =
      (remaining + 1) + (first :: second :: challenges).length := by omega
  have value := quadCodes_evaluate input masks source loaded groups covered fits
    first second challenges dimension' ((BooleanVertex.cons bit suffix).toCubePoint extensionOps)
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt extensionOps extensionLaws] at value
  simp only [PrefixFold.zeroExtend, BooleanTable.valueAt_tabulate] at value
  cases bit <;> simpa only [ProtocolPolynomial.messageAt, PiCCSPrefixRound.point,
    BooleanVertex.toCubePoint_coordinates, BooleanVertex.fieldCoordinates,
    SumCheckTruthPath.VertexEncoding.fieldCoordinates, Bool.false_eq_true,
    if_false, if_true] using! value

end NightstreamFPrime.Export.Stage1.PiCCSNormPrefixSource
