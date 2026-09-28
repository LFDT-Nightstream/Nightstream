import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixSource
import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixPolynomial

/-! The prepared fresh contribution uses the original source endpoints.
Only the existing row fold, selector cache and power cache are composed here. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixContribution

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle

private def sourceMessage (data : ProtocolPolynomial.Data K productionShape)
    (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (bit : Bool) (suffix : BooleanVertex remaining) :=
  ProtocolPolynomial.messageAt extensionOps data
    (PiCCSPrefixRound.point extensionOps challenges dimension
      (if bit then K.one else K.zero) suffix)

private theorem message_fields (row : Vector K ProductionRelation.matrixCount)
    (wanted : ProtocolPolynomial.OutputMessage K productionShape)
    (fields : ∀ port, row.get port = wanted.freshMatrixImage ⟨0, by decide⟩ port) :
    (PiCCSFreshPrefixPolynomial.message row).freshMatrixImage = wanted.freshMatrixImage := by
  funext source port
  have same : source = (⟨0, by decide⟩ : Fin productionShape.freshCount) := by
    have bound : source.val < 1 := source.isLt
    have zero : source.val = 0 := by omega
    exact Fin.ext zero
  subst source
  exact fields port

private theorem ccs_fields (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (selector : FixedPolynomial K 1)
    (low high low' high' : ProtocolPolynomial.OutputMessage K productionShape)
    (lowFields : low.freshMatrixImage = low'.freshMatrixImage)
    (highFields : high.freshMatrixImage = high'.freshMatrixImage) :
    PiCCSFirstRoundPair.ccsPolynomialWithPowers extensionOps input powers selector low high =
      PiCCSFirstRoundPair.ccsPolynomialWithPowers extensionOps input powers selector low' high' := by
  unfold PiCCSFirstRoundPair.ccsPolynomialWithPowers
  rw [lowFields, highFields]

private theorem prepared_coefficients
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (data : ProtocolPolynomial.Data K productionShape) (agreement : input = data.toVerifierInput)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) (low high : Vector K ProductionRelation.matrixCount)
    (lowFields : (PiCCSFreshPrefixPolynomial.message low).freshMatrixImage =
      (sourceMessage data challenges dimension false suffix).freshMatrixImage)
    (highFields : (PiCCSFreshPrefixPolynomial.message high).freshMatrixImage =
      (sourceMessage data challenges dimension true suffix).freshMatrixImage) :
    (PiCCSFreshPrefixPolynomial.contribution input alpha challenges suffix
      (PiCCSGammaPowers.lookup extensionOps.toOps gamma
        (PiCCSGammaPowers.prepare extensionOps.toOps gamma
          (PiCCSFirstRoundPair.powerCount productionShape)))
      (PiCCSTensorWeights.prepare extensionOps
        (PiCCSPrefixSelector.dropPoint alpha challenges.length).coordinates.tail) low high).coefficients =
      (PiCCSPrefixComposition.freshTerm data alpha gamma challenges dimension suffix).coefficients := by
  subst input
  refine (congrArg FixedPolynomial.coefficients
    (PiCCSFreshPrefixPolynomial.contribution_prepared data.toVerifierInput
      alpha gamma challenges suffix dimension low high)).trans ?_
  apply congrArg (fun polynomial =>
    (PiCCSFreshComplete.outerFresh data.toVerifierInput
      (TargetPolynomial.power extensionOps.toOps gamma) polynomial).coefficients)
  exact ccs_fields _ _ _ _ _ _ _ lowFields highFields

private theorem optional_coefficients
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (data : ProtocolPolynomial.Data K productionShape) (agreement : input = data.toVerifierInput)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining)
    (lowAction highAction : Option (Vector K ProductionRelation.matrixCount))
    (lowPorts : ∀ port, lowAction.map (fun row => row.get port) =
      some ((sourceMessage data challenges dimension false suffix).freshMatrixImage ⟨0, by decide⟩ port))
    (highPorts : ∀ port, highAction.map (fun row => row.get port) =
      some ((sourceMessage data challenges dimension true suffix).freshMatrixImage ⟨0, by decide⟩ port)) :
    (do
      let low ← lowAction
      let high ← highAction
      pure (PiCCSFreshPrefixPolynomial.contribution input alpha challenges suffix
        (PiCCSGammaPowers.lookup extensionOps.toOps gamma
          (PiCCSGammaPowers.prepare extensionOps.toOps gamma
            (PiCCSFirstRoundPair.powerCount productionShape)))
        (PiCCSTensorWeights.prepare extensionOps
          (PiCCSPrefixSelector.dropPoint alpha challenges.length).coordinates.tail)
        low high).coefficients) =
      some ((PiCCSPrefixComposition.freshTerm data alpha gamma challenges dimension suffix).coefficients) := by
  obtain ⟨low, lowLoaded, _⟩ := Option.map_eq_some_iff.mp (lowPorts ⟨0, by decide⟩)
  obtain ⟨high, highLoaded, _⟩ := Option.map_eq_some_iff.mp (highPorts ⟨0, by decide⟩)
  have lowFields :
      (PiCCSFreshPrefixPolynomial.message low).freshMatrixImage =
        (sourceMessage data challenges dimension false suffix).freshMatrixImage := by
    apply message_fields
    intro port
    have value := lowPorts port
    rw [lowLoaded, Option.map_some] at value
    exact Option.some.inj value
  have highFields :
      (PiCCSFreshPrefixPolynomial.message high).freshMatrixImage =
        (sourceMessage data challenges dimension true suffix).freshMatrixImage := by
    apply message_fields
    intro port
    have value := highPorts port
    rw [highLoaded, Option.map_some] at value
    exact Option.some.inj value
  simp only [lowLoaded, highLoaded, bind, Option.bind]
  exact congrArg some (prepared_coefficients input data agreement alpha gamma challenges dimension
    suffix low high lowFields highFields)

/-- The same prepared contribution called by the fresh-prefix runner,
with its two rows supplied by the existing source row fold. -/
def coefficients? (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K)
    {remaining : Nat} (suffix : BooleanVertex remaining) : Option (List K) := do
  let low ← PiCCSFreshPrefixSource.endpointRow? masks challenges false suffix
  let high ← PiCCSFreshPrefixSource.endpointRow? masks challenges true suffix
  let coins := PiCCSPublicReplay.pre input
  return (PiCCSFreshPrefixPolynomial.contribution (PiCCSPublicReplay.verifierInput input)
    coins.alpha challenges suffix
    (PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma
      (PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
        (PiCCSFirstRoundPair.powerCount productionShape)))
    (PiCCSTensorWeights.prepare extensionOps
      (PiCCSPrefixSelector.dropPoint coins.alpha challenges.length).coordinates.tail)
    low high).coefficients

/-- Every coefficient of one prepared fresh contribution is the original
source term. Source reads, both row endpoints and cache correctness are derived. -/
theorem coefficients_eq_freshTerm (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) :
    coefficients? input masks challenges suffix =
      some ((PiCCSPrefixComposition.freshTerm
        (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
        challenges dimension suffix).coefficients) := by
  exact optional_coefficients (PiCCSPublicReplay.verifierInput input)
    (PiCCSFirstRoundComposition.sourceData input masks)
    (PiCCSFreshComplete.verifierInput_eq_sourceProtocolData input
      (PiCCSFirstRoundComposition.witness masks))
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma challenges dimension suffix
    (PiCCSFreshPrefixSource.endpointRow? masks challenges false suffix)
    (PiCCSFreshPrefixSource.endpointRow? masks challenges true suffix)
    (PiCCSFreshPrefixSource.endpoint_port input masks challenges dimension false suffix)
    (PiCCSFreshPrefixSource.endpoint_port input masks challenges dimension true suffix)

end NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixContribution
