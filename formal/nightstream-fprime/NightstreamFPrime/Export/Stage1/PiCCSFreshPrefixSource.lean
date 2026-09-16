import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixPreservation
import NightstreamFPrime.Export.Stage1.PiCCSPrefixComposition

/-! Source values of the existing fresh-row prefix folds. The optional initial
row traversal retains failures; selected source ownership proves success.
This module does not certify external files or IO task order. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixSource

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle
open PiCCSAggregatedImages (selectedProgram selectedSource)

private theorem rows_fit : selectedProgram.rowCount ≤ 2 ^ cubeVariables := by
  rw [PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits]
  exact PerApplicationFixedPoint.structuralPlan_rowCount_le
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

/-- The original optional scalar row initializer for the selected fresh source. -/
def rows? (masks : Array (Array (Nat × Nat))) :
    Option (Array (Vector K ProductionRelation.matrixCount)) :=
  PiCCSFreshPrefix.rows? selectedProgram selectedSource
    ((PiCCSFirstRoundComposition.witness masks).assignments (freshSourceIndex ⟨0, by decide⟩))
    rows_fit

/-- Repeat the existing vector-row interpolation in challenge order. -/
def foldRowsPrefix : Array (Vector K ProductionRelation.matrixCount) → List K →
    Array (Vector K ProductionRelation.matrixCount)
  | rows, [] => rows
  | rows, challenge :: challenges =>
      foldRowsPrefix (PiCCSFreshPrefix.foldRows rows challenge) challenges

/-- Every repeated vector port is exactly the existing scalar prefix fold. -/
theorem portValues_foldRowsPrefix (rows : Array (Vector K ProductionRelation.matrixCount))
    (challenges : List K) (port : Fin ProductionRelation.matrixCount) :
    PiCCSFreshPrefix.portValues (foldRowsPrefix rows challenges) port =
      PrefixFold.foldPrefix extensionOps (PiCCSFreshPrefix.portValues rows port) challenges := by
  induction challenges generalizing rows with
  | nil => rfl
  | cons challenge challenges ih =>
      simp only [foldRowsPrefix, PrefixFold.foldPrefix]
      rw [ih, PiCCSFreshPrefix.portValues_foldRows]

/-- One low or high endpoint, including the zero extension after active rows. -/
def endpointRow? (masks : Array (Array (Nat × Nat))) (challenges : List K)
    {remaining : Nat} (bit : Bool) (suffix : BooleanVertex remaining) :
    Option (Vector K ProductionRelation.matrixCount) :=
  (rows? masks).map fun rows =>
    (foldRowsPrefix rows challenges).getD (NumericBooleanDomain.index (.cons bit suffix))
      (Vector.replicate ProductionRelation.matrixCount extensionOps.zero)

private theorem port_fold_endpoint (masks : Array (Array (Nat × Nat)))
    (challenges : List K) (port : Fin ProductionRelation.matrixCount)
    {remaining : Nat} (bit : Bool) (suffix : BooleanVertex remaining) :
    (endpointRow? masks challenges bit suffix).map (fun row => row.get port) =
      (rows? masks).map (fun rows =>
        (PrefixFold.foldPrefix extensionOps (PiCCSFreshPrefix.portValues rows port) challenges).getD
          (NumericBooleanDomain.index (.cons bit suffix)) extensionOps.zero) := by
  simp only [endpointRow?, Option.map_map]
  apply congrArg (fun action => (rows? masks).map action)
  funext rows
  dsimp only [Function.comp_apply]
  rw [← PiCCSFreshPrefix.portValues_getD, portValues_foldRowsPrefix]

/-- Each retained fresh endpoint port is the matching original message field
at the same challenge prefix. The selected source has exactly one fresh row family. -/
theorem endpoint_port (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K)
    {remaining : Nat} (dimension : cubeVariables = challenges.length + remaining + 1)
    (bit : Bool) (suffix : BooleanVertex remaining) (port : Fin ProductionRelation.matrixCount) :
    (endpointRow? masks challenges bit suffix).map (fun row => row.get port) =
      some ((ProtocolPolynomial.messageAt extensionOps
        (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPrefixRound.point extensionOps challenges dimension
          (if bit then K.one else K.zero) suffix)).freshMatrixImage ⟨0, by decide⟩ port) := by
  have dimension' : cubeVariables = (remaining + 1) + challenges.length := by omega
  have value := PiCCSFreshPrefix.fold_evaluate input
    (PiCCSFirstRoundComposition.witness masks) ⟨0, by decide⟩ port challenges dimension'
    ((BooleanVertex.cons bit suffix).toCubePoint extensionOps)
  simp only [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt extensionOps extensionLaws,
    PrefixFold.zeroExtend, BooleanTable.valueAt_tabulate] at value
  rw [port_fold_endpoint]
  cases bit <;> simpa only [rows?, PiCCSFreshPrefix.portValues,
    ProtocolPolynomial.messageAt, PiCCSPrefixRound.point,
    BooleanVertex.toCubePoint_coordinates, BooleanVertex.fieldCoordinates,
    SumCheckTruthPath.VertexEncoding.fieldCoordinates, Bool.false_eq_true,
    if_false, if_true] using value

end NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixSource
