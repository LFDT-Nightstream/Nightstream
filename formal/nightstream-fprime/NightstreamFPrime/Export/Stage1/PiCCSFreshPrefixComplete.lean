import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixContribution
import NightstreamFPrime.Export.Stage1.PiCCSCarriedPrefixComplete

/-! Complete pure fresh-prefix coefficient range. The shortened range keeps
its odd final pair. Selected positive-degree syntax proves the omitted suffix
zero. The optional source constructor retains all load failures. This module
does not prove file provenance, decoding, or mutable IO task execution. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixComplete

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open PiCCSAggregatedImages (selectedProgram)

universe uField
variable {Field : Type uField}

/-- Exact coefficient surface of the existing FixedPolynomial.add fold.
This leaf keeps failure and order; it does not parse an expected polynomial. -/
def coefficientRange? (ops : InterpolationOps Field) (degree : Nat)
    (action : Nat → Option (List Field)) (first count : Nat) : Option (List Field) :=
  Nat.fold count (fun offset _ previous => do
    let total ← previous
    let next ← action (first + offset)
    pure (List.zipWith ops.add total next))
    (some (FixedPolynomial.zero ops.toOps degree).coefficients)

private theorem coefficientRange_succ (ops : InterpolationOps Field) (degree : Nat)
    (action : Nat → Option (List Field)) (first count : Nat) :
    coefficientRange? ops degree action first (count + 1) =
      (do
        let total ← coefficientRange? ops degree action first count
        let next ← action (first + count)
        pure (List.zipWith ops.add total next)) := by
  simp only [coefficientRange?, Nat.fold_succ]

/-- Generic optional accumulation is proved before any selected source data
is supplied. Equality covers complete fixed-width coefficient lists. -/
theorem coefficientRange_eq_range (ops : InterpolationOps Field) {degree : Nat}
    (action : Nat → Option (List Field)) (term : Nat → FixedPolynomial Field degree)
    (first count : Nat)
    (returned : ∀ offset, offset < count →
      action (first + offset) = some (term (first + offset)).coefficients) :
    coefficientRange? ops degree action first count =
      some (PiCCSPolynomialRange.range ops first count term).coefficients := by
  revert returned
  induction count with
  | zero => intro _; rfl
  | succ count ih =>
      intro returned
      have previous := ih (fun offset inside =>
        returned offset (Nat.lt_trans inside (Nat.lt_succ_self count)))
      have next := returned count (Nat.lt_succ_self count)
      rw [coefficientRange_succ, previous, next]
      simp only [bind, Option.bind]
      apply congrArg some
      have step :
          PiCCSPolynomialRange.range ops first (count + 1) term =
            FixedPolynomial.add ops.toOps
              (PiCCSPolynomialRange.range ops first count term) (term (first + count)) := by
        simp only [PiCCSPolynomialRange.range, Nat.fold_succ]
      rw [step, PiCCSPolynomialRange.add_coefficients]

/-- The runtime rejects a pair outside the remaining Boolean domain. -/
def pairCoefficients? (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K)
    (remaining index : Nat) : Option (List K) :=
  if inside : index < 2 ^ remaining then
    PiCCSFreshPrefixContribution.coefficients? input masks challenges
      (NumericBooleanDomain.vertex remaining ⟨index, inside⟩)
  else none

/-- Same complete coefficients and numeric pair order as the pure projection
of freshFromPrefix. Its caller supplies a numeric range, not wanted values. -/
def rangeCoefficients? (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K)
    (remaining first count : Nat) : Option (List K) :=
  coefficientRange? extensionOps (PiCCSPublicReplay.verifierInput input).sumcheckDegreeBound
    (pairCoefficients? input masks challenges remaining) first count

def retainedRows (consumed : Nat) : Nat :=
  (selectedProgram.rowCount + 2 ^ consumed - 1) / 2 ^ consumed

/-- Ceil(retained rows / 2) includes the final mixed real/zero pair. -/
def activePairs (consumed : Nat) : Nat := (retainedRows consumed + 1) / 2

def coefficients? (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K) (remaining : Nat) :
    Option (List K) :=
  rangeCoefficients? input masks challenges remaining 0 (activePairs challenges.length)

private def numericTerm (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1) (index : Nat) :
    FixedPolynomial K data.toVerifierInput.sumcheckDegreeBound :=
  if inside : index < 2 ^ remaining then
    PiCCSPrefixComposition.freshTerm data alpha gamma challenges dimension
      (NumericBooleanDomain.vertex remaining ⟨index, inside⟩)
  else FixedPolynomial.zero extensionOps.toOps data.toVerifierInput.sumcheckDegreeBound

/-- Every covered range has the original source term coefficients. The only
range premise is the actual Boolean-domain bound; no source-value premise remains. -/
theorem range_coefficients_eq (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (first count : Nat) (covered : first + count ≤ 2 ^ remaining) :
    rangeCoefficients? input masks challenges remaining first count =
      some (PiCCSPolynomialRange.range extensionOps first count
        (numericTerm (PiCCSFirstRoundComposition.sourceData input masks)
          (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
          challenges dimension)).coefficients := by
  have inputAgreement := PiCCSFreshComplete.verifierInput_eq_sourceProtocolData input
    (PiCCSFirstRoundComposition.witness masks)
  have degreeAgreement := congrArg
    (fun verifier : ProtocolPolynomial.VerifierInput K productionShape => verifier.sumcheckDegreeBound)
    inputAgreement
  unfold rangeCoefficients?
  refine (congrArg
    (fun degree => coefficientRange? extensionOps degree
      (pairCoefficients? input masks challenges remaining) first count)
    degreeAgreement).trans ?_
  apply coefficientRange_eq_range
  intro offset inside
  have inDomain : first + offset < 2 ^ remaining := by omega
  simp only [pairCoefficients?, dif_pos inDomain, numericTerm]
  simpa only [dif_pos inDomain] using
    PiCCSFreshPrefixContribution.coefficients_eq_freshTerm input masks challenges dimension
      (NumericBooleanDomain.vertex remaining ⟨first + offset, inDomain⟩)

/-- Vector-row folding has the same exact extent as the existing scalar fold. -/
theorem foldRowsPrefix_size
    (rows : Array (Vector K ProductionRelation.matrixCount)) (challenges : List K) :
    (PiCCSFreshPrefixSource.foldRowsPrefix rows challenges).size =
      (rows.size + 2 ^ challenges.length - 1) / 2 ^ challenges.length := by
  have scalar := PiCCSFreshPrefixSource.portValues_foldRowsPrefix rows challenges ⟨0, by decide⟩
  have sizes := congrArg Array.size scalar
  rw [PiCCSCarriedPrefixComplete.foldPrefix_size] at sizes
  simpa only [PiCCSFreshPrefix.portValues, Array.size_map] using sizes

private theorem foldRowsPrefix_fits
    (rows : Array (Vector K ProductionRelation.matrixCount)) (challenges : List K)
    (remaining : Nat) (fits : rows.size ≤ 2 ^ ((remaining + 1) + challenges.length)) :
    (PiCCSFreshPrefixSource.foldRowsPrefix rows challenges).size ≤ 2 ^ (remaining + 1) := by
  have scalar := PrefixFold.foldPrefix_fits extensionOps
    (PiCCSFreshPrefix.portValues rows ⟨0, by decide⟩) challenges (remaining + 1)
    (by simpa only [PiCCSFreshPrefix.portValues, Array.size_map] using fits)
  rw [← PiCCSFreshPrefixSource.portValues_foldRowsPrefix] at scalar
  simpa only [PiCCSFreshPrefix.portValues, Array.size_map] using scalar

private theorem endpoint_zero
    (rows : Array (Vector K ProductionRelation.matrixCount)) (challenges : List K)
    {remaining : Nat} (suffix : BooleanVertex remaining) (bit : Bool)
    (outside : (PiCCSFreshPrefixSource.foldRowsPrefix rows challenges).size ≤
      2 * NumericBooleanDomain.index suffix) :
    (PiCCSFreshPrefixSource.foldRowsPrefix rows challenges).getD
        (NumericBooleanDomain.index (.cons bit suffix))
        (Vector.replicate ProductionRelation.matrixCount extensionOps.zero) =
      Vector.replicate ProductionRelation.matrixCount extensionOps.zero := by
  have beyond : (PiCCSFreshPrefixSource.foldRowsPrefix rows challenges).size ≤
      NumericBooleanDomain.index (.cons bit suffix) := by
    cases bit
    · change _ ≤ 0 + 2 * NumericBooleanDomain.index suffix
      omega
    · change _ ≤ 1 + 2 * NumericBooleanDomain.index suffix
      omega
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none beyond, Option.getD_none]

private theorem fields_zero
    (action : Option (Vector K ProductionRelation.matrixCount))
    (wanted : ProtocolPolynomial.OutputMessage K productionShape)
    (loaded : action = some (Vector.replicate ProductionRelation.matrixCount extensionOps.zero))
    (ports : ∀ port, action.map (fun row => row.get port) =
      some (wanted.freshMatrixImage ⟨0, by decide⟩ port)) :
    ∀ source port, wanted.freshMatrixImage source port = extensionOps.zero := by
  intro source port
  have sourceEqual : source = (⟨0, by decide⟩ : Fin productionShape.freshCount) := by
    have bound : source.val < 1 := source.isLt
    have zero : source.val = 0 := by omega
    exact Fin.ext zero
  subst source
  have value := ports port
  rw [loaded, Option.map_some] at value
  have zeroGet : (Vector.replicate ProductionRelation.matrixCount extensionOps.zero).get port =
      extensionOps.zero := by
    change (Vector.replicate ProductionRelation.matrixCount extensionOps.zero)[port.val] = _
    rw [Vector.getElem_replicate]
  exact (Option.some.inj value).symm.trans zeroGet

private theorem positive_transport
    (left right : ProtocolPolynomial.VerifierInput K productionShape) (same : left = right)
    (positive : ∀ term, term ∈ left.constraintPolynomial.terms → 0 < term.totalDegree) :
    ∀ term, term ∈ right.constraintPolynomial.terms → 0 < term.totalDegree := by
  subst right
  exact positive

/-- Zero endpoints imply a zero fresh polynomial only under explicit
positive-degree syntax. No such claim is made for a generic CCS constant term. -/
private theorem freshTerm_zero
    (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining)
    (positive : ∀ term, term ∈ data.toVerifierInput.constraintPolynomial.terms → 0 < term.totalDegree)
    (lowZero : ∀ source port,
      (ProtocolPolynomial.messageAt extensionOps data
        (PiCCSPrefixRound.point extensionOps challenges dimension extensionOps.zero suffix)).freshMatrixImage
          source port = extensionOps.zero)
    (highZero : ∀ source port,
      (ProtocolPolynomial.messageAt extensionOps data
        (PiCCSPrefixRound.point extensionOps challenges dimension extensionOps.one suffix)).freshMatrixImage
          source port = extensionOps.zero) :
    PiCCSPrefixComposition.freshTerm data alpha gamma challenges dimension suffix =
      FixedPolynomial.zero extensionOps.toOps data.toVerifierInput.sumcheckDegreeBound := by
  have zero := PiCCSFreshPolynomial.reference_ccsPolynomialWithPowers_zero
    extensionOps extensionLaws data.toVerifierInput (TargetPolynomial.power extensionOps.toOps gamma)
    (PiCCSPrefixSelector.selector extensionOps challenges suffix alpha)
    (ProtocolPolynomial.messageAt extensionOps data
      (PiCCSPrefixRound.point extensionOps challenges dimension extensionOps.zero suffix))
    (ProtocolPolynomial.messageAt extensionOps data
      (PiCCSPrefixRound.point extensionOps challenges dimension extensionOps.one suffix))
    positive lowZero highZero
  exact (congrArg (PiCCSFreshComplete.outerFresh data.toVerifierInput
    (TargetPolynomial.power extensionOps.toOps gamma)) zero).trans
      (PiCCSFreshComplete.outerFresh_zero data.toVerifierInput
        (TargetPolynomial.power extensionOps.toOps gamma))

private theorem numericTerm_padding (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (rows : Array (Vector K ProductionRelation.matrixCount))
    (loaded : PiCCSFreshPrefixSource.rows? masks = some rows)
    (rowSize : rows.size = selectedProgram.rowCount)
    (index : Nat) (lower : activePairs challenges.length ≤ index) (upper : index < 2 ^ remaining) :
    numericTerm (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
        challenges dimension index =
      FixedPolynomial.zero extensionOps.toOps
        (PiCCSFirstRoundComposition.sourceData input masks).toVerifierInput.sumcheckDegreeBound := by
  let suffix := NumericBooleanDomain.vertex remaining ⟨index, upper⟩
  have size := foldRowsPrefix_size rows challenges
  rw [rowSize] at size
  have outside : (PiCCSFreshPrefixSource.foldRowsPrefix rows challenges).size ≤
      2 * NumericBooleanDomain.index suffix := by
    rw [size, NumericBooleanDomain.index_vertex]
    change (retainedRows challenges.length + 1) / 2 ≤ index at lower
    change retainedRows challenges.length ≤ 2 * index
    omega
  have zeroAction (bit : Bool) :
      PiCCSFreshPrefixSource.endpointRow? masks challenges bit suffix =
        some (Vector.replicate ProductionRelation.matrixCount extensionOps.zero) := by
    unfold PiCCSFreshPrefixSource.endpointRow?
    rw [loaded, Option.map_some, endpoint_zero rows challenges suffix bit outside]
  have lowZero := fields_zero
    (PiCCSFreshPrefixSource.endpointRow? masks challenges false suffix)
    (ProtocolPolynomial.messageAt extensionOps (PiCCSFirstRoundComposition.sourceData input masks)
      (PiCCSPrefixRound.point extensionOps challenges dimension extensionOps.zero suffix))
    (zeroAction false)
    (PiCCSFreshPrefixSource.endpoint_port input masks challenges dimension false suffix)
  have highZero := fields_zero
    (PiCCSFreshPrefixSource.endpointRow? masks challenges true suffix)
    (ProtocolPolynomial.messageAt extensionOps (PiCCSFirstRoundComposition.sourceData input masks)
      (PiCCSPrefixRound.point extensionOps challenges dimension extensionOps.one suffix))
    (zeroAction true)
    (PiCCSFreshPrefixSource.endpoint_port input masks challenges dimension true suffix)
  have positive := positive_transport (PiCCSPublicReplay.verifierInput input)
    (PiCCSFirstRoundComposition.sourceData input masks).toVerifierInput
    (PiCCSFreshComplete.verifierInput_eq_sourceProtocolData input
      (PiCCSFirstRoundComposition.witness masks))
    (PiCCSFreshPolynomial.selected_terms_positive input)
  rw [numericTerm, dif_pos upper]
  exact freshTerm_zero _ _ _ challenges dimension suffix positive lowZero highZero

private theorem fullRange_eq_sum
    (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1) :
    PiCCSPolynomialRange.range extensionOps 0 (2 ^ remaining)
        (numericTerm data alpha gamma challenges dimension) =
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSPrefixComposition.freshTerm data alpha gamma challenges dimension
          (NumericBooleanDomain.vertex remaining index) := by
  rw [PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)))
  funext index
  simp only [Nat.zero_add, numericTerm, dif_pos index.isLt]

/-- The exact shortened fresh-prefix range gives the complete original
fresh-term coefficient sum. Source success, row extent, padding and every
pair value are derived. No source-correctness or expected-artifact premise
is accepted. The prefix dimension is the only operational hypothesis. -/
theorem coefficients_eq_fullSum (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1) :
    coefficients? input masks challenges remaining =
      some ((FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSPrefixComposition.freshTerm (PiCCSFirstRoundComposition.sourceData input masks)
          (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
          challenges dimension (NumericBooleanDomain.vertex remaining index)).coefficients) := by
  have sizeResult : (PiCCSFreshPrefixSource.rows? masks).map Array.size =
      some selectedProgram.rowCount :=
    PiCCSFreshPrefix.rows?_size input (PiCCSFirstRoundComposition.witness masks) ⟨0, by decide⟩
  obtain ⟨rows, loaded, rowSize⟩ := Option.map_eq_some_iff.mp sizeResult
  have rowFits : rows.size ≤ 2 ^ cubeVariables := by
    rw [rowSize,
      PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
        Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits]
    exact PerApplicationFixedPoint.structuralPlan_rowCount_le
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
  have shape : cubeVariables = (remaining + 1) + challenges.length := by omega
  have foldedFits := foldRowsPrefix_fits rows challenges remaining (by rw [← shape]; exact rowFits)
  rw [foldRowsPrefix_size, rowSize] at foldedFits
  have pairFits : activePairs challenges.length ≤ 2 ^ remaining := by
    change retainedRows challenges.length ≤ 2 ^ (remaining + 1) at foldedFits
    rw [Nat.pow_succ] at foldedFits
    unfold activePairs
    omega
  let term := numericTerm (PiCCSFirstRoundComposition.sourceData input masks)
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma challenges dimension
  have coverage : activePairs challenges.length +
      (2 ^ remaining - activePairs challenges.length) = 2 ^ remaining := by omega
  have complete := PiCCSPolynomialRange.range_append_zero extensionOps extensionLaws
    0 (activePairs challenges.length) (2 ^ remaining - activePairs challenges.length) term (by
      intro index lower upper
      exact numericTerm_padding input masks challenges dimension rows loaded rowSize index
        (by simpa only [Nat.zero_add] using lower)
        (by simpa only [Nat.zero_add, coverage] using upper))
  rw [coverage] at complete
  calc
    _ = some (PiCCSPolynomialRange.range extensionOps 0 (activePairs challenges.length) term).coefficients :=
      range_coefficients_eq input masks challenges dimension 0 (activePairs challenges.length)
        (by simpa only [Nat.zero_add] using pairFits)
    _ = some (PiCCSPolynomialRange.range extensionOps 0 (2 ^ remaining) term).coefficients :=
      congrArg (fun polynomial => some polynomial.coefficients) complete.symm
    _ = _ := congrArg (fun polynomial => some polynomial.coefficients)
      (fullRange_eq_sum (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma challenges dimension)

end NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixComplete
