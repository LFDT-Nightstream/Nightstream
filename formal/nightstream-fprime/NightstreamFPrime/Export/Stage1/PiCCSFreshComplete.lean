import NightstreamFPrime.Export.Stage1.PiCCSFreshPadding
import NightstreamFPrime.Export.Stage1.PiCCSCachedSelector
import NightstreamFPrime.Export.Stage1.PiCCSAggregatedImages
import NightstreamFPrime.Export.Stage1.PiCCSSourceImagesPreservation

/-! Proof-only closure of the fresh numeric pair sum. Optional row failures
are preserved and then discharged by the selected original-source theorem.
This does not verify the executable's IO task or invocation-cache loop. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFreshComplete

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout
open PiCCSPolynomialRange (coefficients_ext)

private abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev selectedProgram := PerApplicationMatrixProgram.matrixProgram
  Poseidon2HashChainV1Package.application
private abbrev selectedSource := fun (row : Nat) =>
  (PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)[row]?
private abbrev canonicalSource := PerApplicationCanonicalPackage.sourceRow
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private noncomputable abbrev selectedStatement (input : PiCCSPublicReplay.Input) :=
  (ProductionKey.key selectedRelation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
    (PiCCSPublicReplay.running input) (PiCCSPublicReplay.fresh input)
private noncomputable abbrev message (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (vertex : BooleanVertex cubeVariables) :=
  ProtocolPolynomial.vertexMessage ((selectedStatement input).sourceProtocolData K.embed witness) vertex

/-- The public input used by the fresh runner is the verifier projection
of the same original-source protocol data. No key or matrix is unfolded. -/
theorem verifierInput_eq_sourceProtocolData (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth) :
    PiCCSPublicReplay.verifierInput input =
      ((selectedStatement input).sourceProtocolData K.embed witness).toVerifierInput := by
  rw [StrongReduction.Statement.sourceProtocolData_toVerifierInput]
  exact PiCCSPublicReplay.verifierInput_eq_key input selectedRelation
    Poseidon2HashChainV1Setup.productionAjtaiKey

private theorem selectedSource_value : selectedSource = canonicalSource := by
  funext row
  exact PiDECCanonicalSourceCache.stored_value Poseidon2HashChainV1Package.application
    Poseidon2HashChainV1Package.fits row

/-- The unchanged outer fresh gamma shift and degree widening in Main. -/
def outerFresh (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K)
    (inner : FixedPolynomial K input.constraintPolynomial.canonicalEqualityGatedDegreeBound) :
    FixedPolynomial K input.sumcheckDegreeBound :=
  FixedPolynomial.scale extensionOps.toOps (powers productionShape.constraintOffset)
    (FixedPolynomial.widen extensionOps.toOps (Nat.le_max_left _ _) inner)

/-- The same numeric fresh endpoint operations as Main, before its degree-9
cast. No failed row lookup is replaced with a polynomial value. -/
def numericPairFresh? (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables) (index : Nat) :
    Option (FixedPolynomial K (PiCCSPublicReplay.verifierInput input).sumcheckDegreeBound) :=
  if inside : index < 2 ^ (cubeVariables - 1) then do
    let suffix := NumericBooleanDomain.vertex (cubeVariables - 1) ⟨index, inside⟩
    let lowVertex := PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false suffix
    let highVertex := PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true suffix
    let low ← PiCCSSourceImages.freshMatrixImage? selectedProgram selectedSource
      (witness.assignments (freshSourceIndex ⟨0, by decide⟩)) lowVertex
    let high ← PiCCSSourceImages.freshMatrixImage? selectedProgram selectedSource
      (witness.assignments (freshSourceIndex ⟨0, by decide⟩)) highVertex
    return outerFresh (PiCCSPublicReplay.verifierInput input) powers
      (PiCCSFreshPolynomial.ccsPolynomialWithPowers extensionOps
        (PiCCSPublicReplay.verifierInput input) powers
        (PiCCSCachedSelector.equalitySelector extensionOps suffix alpha
          (PiCCSTensorWeights.prepare extensionOps alpha.coordinates.tail))
        (PiCCSAggregatedImages.nonlinearMessage layout witness.assignments lowVertex low)
        (PiCCSAggregatedImages.nonlinearMessage layout witness.assignments highVertex high))
  else some (FixedPolynomial.zero extensionOps.toOps
    (PiCCSPublicReplay.verifierInput input).sumcheckDegreeBound)

/-- The original-source reference fresh term over the full numeric domain.
Only the existing public input and sourceProtocolData endpoint messages occur. -/
noncomputable def referencePair (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables) (index : Nat) :
    FixedPolynomial K (PiCCSPublicReplay.verifierInput input).sumcheckDegreeBound :=
  if inside : index < 2 ^ (cubeVariables - 1) then
    let suffix := NumericBooleanDomain.vertex (cubeVariables - 1) ⟨index, inside⟩
    outerFresh (PiCCSPublicReplay.verifierInput input) powers
      (PiCCSFirstRoundPair.ccsPolynomialWithPowers extensionOps
        (PiCCSPublicReplay.verifierInput input) powers
        (PiCCSFirstRound.equalitySelector extensionOps suffix alpha)
        (message input witness (PiCCSFirstRound.endpointVertex (arity := cubeVariables)
          (by decide) false suffix))
        (message input witness (PiCCSFirstRound.endpointVertex (arity := cubeVariables)
          (by decide) true suffix)))
  else FixedPolynomial.zero extensionOps.toOps
    (PiCCSPublicReplay.verifierInput input).sumcheckDegreeBound

private theorem canonical_fresh_value (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (source : Fin productionShape.freshCount) (matrix : Fin ProductionRelation.matrixCount)
    (vertex : BooleanVertex cubeVariables) :
    (PiCCSSourceImages.freshMatrixImage? selectedProgram canonicalSource
      (witness.assignments (freshSourceIndex source)) vertex).map
      (fun values => K.embed (values.get matrix)) =
        some ((message input witness vertex).freshMatrixImage source matrix) :=
  PiCCSSourceImages.freshMatrix_sourceProtocolData input witness source matrix vertex

private theorem embedded_replicate_zero {count : Nat} (index : Fin count) :
    K.embed ((Vector.replicate count (0 : F)).get index) = extensionOps.zero := by
  change K.embed ((Vector.replicate count (0 : F))[index.val]) = extensionOps.zero
  rw [Vector.getElem_replicate]
  rfl

private theorem fresh_loaded (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (vertex : BooleanVertex cubeVariables) :
    ∃ values, PiCCSSourceImages.freshMatrixImage? selectedProgram selectedSource
      (witness.assignments (freshSourceIndex ⟨0, by decide⟩)) vertex = some values := by
  have original := canonical_fresh_value input witness
    ⟨0, by decide⟩ ⟨0, by decide⟩ vertex
  rw [← selectedSource_value] at original
  obtain ⟨values, loaded, _⟩ := Option.map_eq_some_iff.mp original
  exact ⟨values, loaded⟩

private theorem fresh_fields (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (vertex : BooleanVertex cubeVariables) (values : Vector F ProductionRelation.matrixCount)
    (loaded : PiCCSSourceImages.freshMatrixImage? selectedProgram selectedSource
      (witness.assignments (freshSourceIndex ⟨0, by decide⟩)) vertex = some values) :
    (PiCCSAggregatedImages.nonlinearMessage layout witness.assignments vertex values).freshMatrixImage =
      (message input witness vertex).freshMatrixImage := by
  funext source matrix
  have sourceZero : source = (⟨0, by decide⟩ : Fin productionShape.freshCount) := by
    apply Fin.ext
    change source.val = 0
    have bound : source.val < 1 := source.isLt
    omega
  subst source
  have original := canonical_fresh_value input witness
    ⟨0, by decide⟩ matrix vertex
  rw [← selectedSource_value, loaded, Option.map_some] at original
  exact Option.some.inj original

private theorem reference_fresh_congr
    (input : ProtocolPolynomial.VerifierInput K productionShape) (powers : Nat → K)
    (selector : FixedPolynomial K 1)
    (low high low' high' : ProtocolPolynomial.OutputMessage K productionShape)
    (lowFields : low.freshMatrixImage = low'.freshMatrixImage)
    (highFields : high.freshMatrixImage = high'.freshMatrixImage) :
    PiCCSFirstRoundPair.ccsPolynomialWithPowers extensionOps input powers selector low high =
      PiCCSFirstRoundPair.ccsPolynomialWithPowers extensionOps input powers selector low' high' := by
  unfold PiCCSFirstRoundPair.ccsPolynomialWithPowers
  rw [lowFields, highFields]

/-- Selected source semantics discharge both loads and identify every fresh
field. This theorem applies to arbitrary original complete assignments. -/
theorem numericPairFresh?_value (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables) (index : Nat) :
    numericPairFresh? input witness layout powers alpha index =
      some (referencePair input witness powers alpha index) := by
  by_cases inside : index < 2 ^ (cubeVariables - 1)
  · let suffix := NumericBooleanDomain.vertex (cubeVariables - 1) ⟨index, inside⟩
    let lowVertex := PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false suffix
    let highVertex := PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true suffix
    obtain ⟨low, lowLoaded⟩ := fresh_loaded input witness lowVertex
    obtain ⟨high, highLoaded⟩ := fresh_loaded input witness highVertex
    have lowFields := fresh_fields input witness layout lowVertex low lowLoaded
    have highFields := fresh_fields input witness layout highVertex high highLoaded
    simp only [numericPairFresh?, referencePair, dif_pos inside]
    change (do
      let low ← PiCCSSourceImages.freshMatrixImage? selectedProgram selectedSource
        (witness.assignments (freshSourceIndex ⟨0, by decide⟩)) lowVertex
      let high ← PiCCSSourceImages.freshMatrixImage? selectedProgram selectedSource
        (witness.assignments (freshSourceIndex ⟨0, by decide⟩)) highVertex
      pure (outerFresh (PiCCSPublicReplay.verifierInput input) powers
        (PiCCSFreshPolynomial.ccsPolynomialWithPowers extensionOps
          (PiCCSPublicReplay.verifierInput input) powers
          (PiCCSCachedSelector.equalitySelector extensionOps suffix alpha
            (PiCCSTensorWeights.prepare extensionOps alpha.coordinates.tail))
          (PiCCSAggregatedImages.nonlinearMessage layout witness.assignments lowVertex low)
          (PiCCSAggregatedImages.nonlinearMessage layout witness.assignments highVertex high)))) = _
    rw [lowLoaded, highLoaded]
    simp only [bind, Option.bind]
    apply congrArg some
    apply congrArg (outerFresh (PiCCSPublicReplay.verifierInput input) powers)
    rw [PiCCSFreshPolynomial.ccsPolynomialWithPowers_value extensionOps extensionLaws,
      PiCCSCachedSelector.equalitySelector_prepare extensionOps extensionLaws (by decide)]
    exact reference_fresh_congr _ powers _ _ _ _ _ lowFields highFields
  · simp only [numericPairFresh?, referencePair, dif_neg inside]

/-- The same coefficient addition loop with Option failure retained.
The first argument is an absolute pair index, as in each worker range. -/
def freshRange? (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables) (first count : Nat) :
    Option (FixedPolynomial K (PiCCSPublicReplay.verifierInput input).sumcheckDegreeBound) :=
  Nat.fold count (fun offset _ previous => do
    let total ← previous
    let value ← numericPairFresh? input witness layout powers alpha (first + offset)
    pure (FixedPolynomial.add extensionOps.toOps total value))
    (some (FixedPolynomial.zero extensionOps.toOps
      (PiCCSPublicReplay.verifierInput input).sumcheckDegreeBound))

/-- Every actual worker interval equals the same interval of original-source
reference terms. This supplies the worker values for the ordered range merger. -/
theorem freshRange?_value (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables) (first count : Nat) :
    freshRange? input witness layout powers alpha first count =
      some (PiCCSPolynomialRange.range extensionOps first count
        (referencePair input witness powers alpha)) := by
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [freshRange?, Nat.fold_succ]
      change (do
        let total ← freshRange? input witness layout powers alpha first count
        let value ← numericPairFresh? input witness layout powers alpha (first + count)
        pure (FixedPolynomial.add extensionOps.toOps total value)) = _
      rw [ih, numericPairFresh?_value]
      change some (FixedPolynomial.add extensionOps.toOps
          (PiCCSPolynomialRange.range extensionOps first count
            (referencePair input witness powers alpha))
          (referencePair input witness powers alpha (first + count))) = _
      simp only [PiCCSPolynomialRange.range, Nat.fold_succ]

/-- One final mixed pair is retained when the active row count is odd. -/
def activePairs : Nat := (selectedProgram.rowCount + 1) / 2

private theorem activePairs_le_domain : activePairs ≤ 2 ^ (cubeVariables - 1) := by
  have rows : selectedProgram.rowCount ≤ 2 ^ cubeVariables := by
    rw [PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits]
    exact PerApplicationFixedPoint.structuralPlan_rowCount_le
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
  change selectedProgram.rowCount ≤ 268435456 at rows
  change (selectedProgram.rowCount + 1) / 2 ≤ 134217728
  omega

private theorem reference_fresh_zero (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (suffix : BooleanVertex 27) (beyond : selectedProgram.rowCount ≤ 2 * NumericBooleanDomain.index suffix)
    (bit : Bool) (source : Fin productionShape.freshCount)
    (matrix : Fin ProductionRelation.matrixCount) :
    (message input witness (PiCCSFirstRound.endpointVertex (arity := cubeVariables)
      (by decide) bit suffix)).freshMatrixImage source matrix = extensionOps.zero := by
  have original := canonical_fresh_value input witness source matrix
    (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) bit suffix)
  rw [PiCCSFreshPolynomial.freshMatrixImage?_zero_of_pair_beyond selectedProgram canonicalSource
    (witness.assignments (freshSourceIndex source)) suffix beyond bit, Option.map_some] at original
  exact (Option.some.inj original).symm.trans (embedded_replicate_zero matrix)

private theorem outerFresh_zero (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) :
    outerFresh input powers (FixedPolynomial.zero extensionOps.toOps
      input.constraintPolynomial.canonicalEqualityGatedDegreeBound) =
      FixedPolynomial.zero extensionOps.toOps input.sumcheckDegreeBound := by
  have widened : FixedPolynomial.widen extensionOps.toOps (Nat.le_max_left _ _)
      (FixedPolynomial.zero extensionOps.toOps input.constraintPolynomial.canonicalEqualityGatedDegreeBound) =
        FixedPolynomial.zero extensionOps.toOps input.sumcheckDegreeBound := by
    apply coefficients_ext
    change List.replicate (input.constraintPolynomial.canonicalEqualityGatedDegreeBound + 1)
        extensionOps.zero ++
      List.replicate (input.sumcheckDegreeBound - input.constraintPolynomial.canonicalEqualityGatedDegreeBound)
        extensionOps.zero = List.replicate (input.sumcheckDegreeBound + 1) extensionOps.zero
    rw [← List.replicate_add]
    congr 1
    have bound : input.constraintPolynomial.canonicalEqualityGatedDegreeBound ≤ input.sumcheckDegreeBound :=
      Nat.le_max_left _ _
    omega
  exact (congrArg (FixedPolynomial.scale extensionOps.toOps
    (powers productionShape.constraintOffset)) widened).trans
      (PiCCSPolynomialRange.scale_zero_polynomial extensionOps extensionLaws _ _)

private theorem referencePair_padding (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables) (index : Nat)
    (lower : activePairs ≤ index) (upper : index < 2 ^ (cubeVariables - 1)) :
    referencePair input witness powers alpha index =
      FixedPolynomial.zero extensionOps.toOps (PiCCSPublicReplay.verifierInput input).sumcheckDegreeBound := by
  let suffix := NumericBooleanDomain.vertex (cubeVariables - 1) ⟨index, upper⟩
  have beyond : selectedProgram.rowCount ≤ 2 * NumericBooleanDomain.index suffix := by
    rw [NumericBooleanDomain.index_vertex]
    change selectedProgram.rowCount ≤ 2 * index
    change (selectedProgram.rowCount + 1) / 2 ≤ index at lower
    omega
  have zero := PiCCSFreshPolynomial.reference_ccsPolynomialWithPowers_zero extensionOps extensionLaws
    (PiCCSPublicReplay.verifierInput input) powers
    (PiCCSFirstRound.equalitySelector extensionOps suffix alpha)
    (message input witness (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false suffix))
    (message input witness (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true suffix))
    (PiCCSFreshPolynomial.selected_terms_positive input)
    (fun source matrix => reference_fresh_zero input witness suffix beyond false source matrix)
    (fun source matrix => reference_fresh_zero input witness suffix beyond true source matrix)
  simp only [referencePair, dif_pos upper]
  exact (congrArg (outerFresh (PiCCSPublicReplay.verifierInput input) powers) zero).trans
    (outerFresh_zero _ powers)

/-- The complete active-row fresh scan is the full 2^27-pair reference fresh
sum of the original sourceProtocolData. The selector, every coefficient,
degree widening, outer gamma shift and final mixed pair are retained.
Prepared gamma lookup can be substituted by PiCCSGammaPowers.lookup_prepare. -/
theorem freshRange_eq_fullPairSum (input : PiCCSPublicReplay.Input)
    (witness : StrongReduction.OutputWitness productionShape PiCCSSourceImages.shape.carrierWidth)
    (layout : ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables) :
    freshRange? input witness layout powers alpha 0 activePairs =
      some (PiCCSPolynomialRange.range extensionOps 0 (2 ^ (cubeVariables - 1))
        (referencePair input witness powers alpha)) := by
  have coverage : activePairs + (2 ^ (cubeVariables - 1) - activePairs) =
      2 ^ (cubeVariables - 1) := by
    have bound := activePairs_le_domain
    omega
  have complete := PiCCSPolynomialRange.range_append_zero extensionOps extensionLaws 0 activePairs
    (2 ^ (cubeVariables - 1) - activePairs) (referencePair input witness powers alpha)
    (by
      intro index lower upper
      apply referencePair_padding input witness powers alpha index
      · simpa only [Nat.zero_add] using lower
      · simpa only [Nat.zero_add, coverage] using upper)
  rw [coverage] at complete
  exact (freshRange?_value input witness layout powers alpha 0 activePairs).trans
    (congrArg some complete.symm)

end NightstreamFPrime.Export.Stage1.PiCCSFreshComplete
