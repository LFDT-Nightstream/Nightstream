import NightstreamFPrime.Layout.Stage1.PiCCSProofInputs

/-!
Owns the inverse from an actual selected NIFS proof to the existing PiCCS
source values. The production degree remains nine. Round coefficients and
both output families are read from that proof; D messages stay in its template.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSProofReadback

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Existing source values read directly from the actual fresh commitment
and the actual degree-nine NIFS proof. -/
def ofProof (commitment : PaperAlgebra.Commitment) (actual : Proof 9) :
    PiCCSProofInputs.ProofValues where
  freshCommitment := commitment
  roundCoefficient := fun round coefficient =>
    (actual.piCcsRounds round).coefficients.get
      ⟨coefficient.val, by
        rw [(actual.piCcsRounds round).coefficients_length]
        exact coefficient.isLt⟩
  outputEval_K := actual.piCcsOutput.padCoordinate
  outputEval_A := actual.piCcsOutput.matrixCoordinate

private theorem polynomial_ext
    (left right : SumCheck.Finite.FixedPolynomial K 9)
    (same : left.coefficients = right.coefficients) : left = right := by
  cases left
  cases right
  cases same
  rfl

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)

/-- Source reconstruction returns the exact selected proof. The degree
conversion uses the existing production degree-nine definition and
`ProductionKey.degreeBound_eq`; it adds no restriction on proof values. -/
theorem relationProof_ofProof (commitment : PaperAlgebra.Commitment)
    (actual : Proof (ProductionKey.degreeBound relation)) :
    PiCCSProofInputs.relationProof relation (ofProof commitment actual) actual = actual := by
  have rounds : (PiCCSProofInputs.relationProof relation
      (ofProof commitment actual) actual).piCcsRounds = actual.piCcsRounds := by
    funext round
    apply polynomial_ext
    change List.ofFn (fun coefficient : Fin (9 + 1) =>
      (actual.piCcsRounds round).coefficients.get
        ⟨coefficient.val, _⟩) = (actual.piCcsRounds round).coefficients
    apply List.ext_get
    · simp only [List.length_ofFn, (actual.piCcsRounds round).coefficients_length,
        ProductionKey.degreeBound_eq]
    · intro index leftBound rightBound
      rw [List.get_ofFn]
      rfl
  exact congrArg (fun messages => ({
    piCcsRounds := messages
    piCcsOutput := actual.piCcsOutput
    piDecCommitments := actual.piDecCommitments
    piDecEvaluations := actual.piDecEvaluations } : Proof 9)) rounds

/-- The unique fresh source is reconstructed with its exact commitment and
public input, without a new fresh-instance representation. -/
theorem protocolFresh_ofProof
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (actual : Proof (ProductionKey.degreeBound relation)) :
    PiCCSProofInputs.protocolFresh logicalWidth publicFits
      (fresh.publicInputs ⟨0, by decide⟩)
      (ofProof (fresh.commitments ⟨0, by decide⟩) actual) = fresh := by
  have unique (source : Fin productionShape.freshCount) : source = ⟨0, by decide⟩ := by
    apply Fin.ext
    have bound := source.isLt
    change source.val < 1 at bound
    change source.val = 0
    omega
  have commitments : (PiCCSProofInputs.protocolFresh logicalWidth publicFits
      (fresh.publicInputs ⟨0, by decide⟩)
      (ofProof (fresh.commitments ⟨0, by decide⟩) actual)).commitments = fresh.commitments := by
    funext source
    change fresh.commitments ⟨0, by decide⟩ = fresh.commitments source
    rw [unique source]
  have publicInputs : (PiCCSProofInputs.protocolFresh logicalWidth publicFits
      (fresh.publicInputs ⟨0, by decide⟩)
      (ofProof (fresh.commitments ⟨0, by decide⟩) actual)).publicInputs = fresh.publicInputs := by
    funext source
    change fresh.publicInputs ⟨0, by decide⟩ = fresh.publicInputs source
    rw [unique source]
  exact congrArg₂ (fun commitments publicInputs =>
    ({ commitments := commitments, publicInputs := publicInputs } :
      Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)))
    commitments publicInputs

end NightstreamFPrime.Layout.Stage1.PiCCSProofReadback
