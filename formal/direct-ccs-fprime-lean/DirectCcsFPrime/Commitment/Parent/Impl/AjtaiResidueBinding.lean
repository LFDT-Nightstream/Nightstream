import DirectCcsFPrime.Commitment.Parent.Security.ParentOpeningAuthorization
import SuperNeo.ProofSystem.Lattice
import SuperNeo.ProofSystem.LatticeReductions

/-!
Ajtai-backed residue binding for reduced parent handles.

This module bridges the local direct-CCS obligation
`CommitMapResiduesFunctional` to the concrete Ajtai opening/collision surface
from `formal/superneo-lean`.
-/

namespace DirectCcsFPrime

namespace AjtaiResidueBinding

/-- Prop-level absence of a concrete Ajtai binding collision. -/
def NoAjtaiBindingCollision
    (params : SuperNeo.ProofSystem.AjtaiParams) : Prop :=
  ¬ Nonempty (SuperNeo.ProofSystem.BindingCollision params)

private theorem invPolyBound_one_lt_one_of_pos
    {n : Nat}
    (hn : 0 < n) :
    SuperNeo.ProofSystem.invPolyBound 1 n < (1 : Rat) := by
  unfold SuperNeo.ProofSystem.invPolyBound
  have hDenGtOne : (1 : Rat) < (n + 1 : Rat) := by
    exact_mod_cast Nat.lt_add_of_pos_left hn
  simpa [one_div] using (inv_lt_one_of_one_lt₀ hDenGtOne)

/--
An Ajtai binding advantage bound with negligible error rules out a concrete
binding collision in the theorem-facing Prop model.

This is still a cryptographic boundary: the caller must supply the advantage
bound. The theorem removes the extra local `NoAjtaiBindingCollision` premise
once that boundary is available.
-/
theorem noAjtaiBindingCollision_of_advantageBound
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {eps : SuperNeo.ProofSystem.ErrorFn}
    (hBound :
      SuperNeo.ProofSystem.AjtaiBindingAdvantageBound params eps)
    (hNeg :
      SuperNeo.ProofSystem.IsNegligible eps) :
    NoAjtaiBindingCollision params := by
  intro hCollision
  rcases hNeg 1 with ⟨N, hTail⟩
  let n := max N 1
  have hN : N ≤ n := by
    exact Nat.le_max_left N 1
  have hnPos : 0 < n := by
    exact Nat.lt_of_lt_of_le Nat.zero_lt_one (Nat.le_max_right N 1)
  have hAdv :
      SuperNeo.ProofSystem.AjtaiBindingAdvantage
        SuperNeo.ProofSystem.truthProb
        (SuperNeo.ProofSystem.canonicalAjtaiBindingGame params)
        n ≤ eps n :=
    hBound SuperNeo.ProofSystem.truthProb n
  have hAdvOne :
      SuperNeo.ProofSystem.AjtaiBindingAdvantage
        SuperNeo.ProofSystem.truthProb
        (SuperNeo.ProofSystem.canonicalAjtaiBindingGame params)
        n = 1 := by
    simp [
      SuperNeo.ProofSystem.AjtaiBindingAdvantage,
      SuperNeo.ProofSystem.canonicalAjtaiBindingGame,
      SuperNeo.ProofSystem.truthProb,
      hCollision
    ]
  have hOneLe : (1 : Rat) ≤ eps n := by
    simpa [hAdvOne] using hAdv
  have hEpsLe : eps n ≤ SuperNeo.ProofSystem.invPolyBound 1 n :=
    hTail n hN
  have hInvLt : SuperNeo.ProofSystem.invPolyBound 1 n < (1 : Rat) :=
    invPolyBound_one_lt_one_of_pos hnPos
  exact not_lt_of_ge (le_trans hOneLe hEpsLe) hInvLt

/--
The theorem-facing Ajtai binding assumption also rules out concrete binding
collisions.
-/
theorem noAjtaiBindingCollision_of_ajtaiBindingAssumption
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (hAssumption :
      SuperNeo.ProofSystem.AjtaiBindingAssumption params) :
    NoAjtaiBindingCollision params := by
  rcases hAssumption with ⟨eps, hNeg, hBound⟩
  exact noAjtaiBindingCollision_of_advantageBound hBound hNeg

/--
The SuperNeo MSIS-to-Ajtai reduction surface plus MSIS hardness rules out
concrete Ajtai binding collisions.

This reuses the theorem-facing lattice reduction package from
`formal/superneo-lean`; the direct-CCS project does not introduce a separate
Ajtai assumption.
-/
theorem noAjtaiBindingCollision_of_msis
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params) :
    NoAjtaiBindingCollision params := by
  exact
    noAjtaiBindingCollision_of_ajtaiBindingAssumption
      (SuperNeo.ProofSystem.ajtaiBinding_of_msis hRed hMsis)

/--
Adapter connecting an abstract CE `commitMap : Coeffs -> Commitment` to the
concrete Ajtai opening relation.

The projector states how the Ajtai opening witness exposes exactly the flat
residues consumed by private `Pi_DEC`.
-/
structure AssignmentOpeningAdapter
    (n : Nat)
    (params : SuperNeo.ProofSystem.AjtaiParams)
    (commitMap : SuperNeo.Coeffs → SuperNeo.ProofSystem.Commitment) where
  toOpening : SuperNeo.Coeffs → SuperNeo.ProofSystem.Opening
  projectWitnessResidues : Array SuperNeo.Coeffs → Fin n → Nat
  opens :
    ∀ assignment,
      SuperNeo.ProofSystem.opensTo
        params
        (commitMap assignment)
        (toOpening assignment)
  bounded :
    ∀ assignment,
      (toOpening assignment).normBound < params.bindingNormBound
  residueSound :
    ∀ assignment,
      projectWitnessResidues (toOpening assignment).witness =
        ParentOpeningAuthorization.assignmentResidues
          (n := n)
          assignment

/--
Canonical Ajtai commitment induced by a fixed public matrix and an opening
witness.

The payload is the theorem-level `(M || Mz)` encoding used by
`SuperNeo.ProofSystem.opensTo`.
-/
def commitmentOfOpening
    (params : SuperNeo.ProofSystem.AjtaiParams)
    (matrixFlat : Array SuperNeo.Coeffs)
    (opening : SuperNeo.ProofSystem.Opening) :
    SuperNeo.ProofSystem.Commitment where
  payload :=
    matrixFlat ++
      SuperNeo.ProofSystem.matVecMul
        params
        matrixFlat
        opening.witness

/--
Concrete Ajtai commitment map for a fixed public matrix and assignment-opening
encoder.

This is the theorem-facing form of the SuperNeo/Ajtai commitment map:
`assignment ↦ M || Mz`.
-/
def ajtaiCommitMap
    (params : SuperNeo.ProofSystem.AjtaiParams)
    (matrixFlat : Array SuperNeo.Coeffs)
    (toOpening :
      SuperNeo.Coeffs →
        SuperNeo.ProofSystem.Opening) :
    SuperNeo.Coeffs → SuperNeo.ProofSystem.Commitment :=
  fun assignment =>
    commitmentOfOpening params matrixFlat (toOpening assignment)

/--
The minimal theorem-facing condition saying an abstract CE commitment map is
actually backed by canonical Ajtai commitments.

This is the bridge that the abstract SuperNeo `GlobalParams.commitMap` cannot
provide by itself: the commitment map must be tied to a fixed public Ajtai
matrix, to accepted openings, and to the residue projector consumed by private
`Pi_DEC`.
-/
structure AjtaiBackedCommitMap
    (n : Nat)
    (params : SuperNeo.ProofSystem.AjtaiParams)
    (commitMap : SuperNeo.Coeffs → SuperNeo.ProofSystem.Commitment) where
  matrixFlat : Array SuperNeo.Coeffs
  matrixShape : matrixFlat.size = params.matrixFlatLen
  toOpening : SuperNeo.Coeffs → SuperNeo.ProofSystem.Opening
  projectWitnessResidues : Array SuperNeo.Coeffs → Fin n → Nat
  commitMap_eq :
    ∀ assignment,
      commitMap assignment =
        commitmentOfOpening params matrixFlat (toOpening assignment)
  openingWellFormed :
    ∀ assignment,
      SuperNeo.ProofSystem.Opening.WellFormed params (toOpening assignment)
  openingNormSound :
    ∀ assignment,
      SuperNeo.ProofSystem.Opening.NormSound (toOpening assignment)
  bounded :
    ∀ assignment,
      (toOpening assignment).normBound < params.bindingNormBound
  residueSound :
    ∀ assignment,
      projectWitnessResidues (toOpening assignment).witness =
        ParentOpeningAuthorization.assignmentResidues
          (n := n)
          assignment

/--
A canonical Ajtai commitment opens to the opening used to construct it.
-/
theorem opensTo_commitmentOfOpening
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {matrixFlat : Array SuperNeo.Coeffs}
    {opening : SuperNeo.ProofSystem.Opening}
    (hMatrix : matrixFlat.size = params.matrixFlatLen)
    (hOpeningWF :
      SuperNeo.ProofSystem.Opening.WellFormed params opening)
    (hNorm :
      SuperNeo.ProofSystem.Opening.NormSound opening) :
    SuperNeo.ProofSystem.opensTo
      params
      (commitmentOfOpening params matrixFlat opening)
      opening := by
  let value :=
    SuperNeo.ProofSystem.matVecMul
      params
      matrixFlat
      opening.witness
  have hValueSize : value.size = params.kappa := by
    simpa [value] using
      SuperNeo.ProofSystem.matVecMul_size
        params
        matrixFlat
        opening.witness
  have hPayloadWF :
      SuperNeo.ProofSystem.Commitment.WellFormed
        params
        (commitmentOfOpening params matrixFlat opening) := by
    simp [
      commitmentOfOpening,
      SuperNeo.ProofSystem.Commitment.WellFormed,
      SuperNeo.ProofSystem.AjtaiParams.payloadLen,
      SuperNeo.ProofSystem.AjtaiParams.commitmentLen,
      hMatrix,
      hValueSize,
      value
    ]
  have hMatrixPart :
      SuperNeo.ProofSystem.Commitment.ppMatrixFlat
        params
        (commitmentOfOpening params matrixFlat opening) =
      matrixFlat := by
    calc
      SuperNeo.ProofSystem.Commitment.ppMatrixFlat
          params
          (commitmentOfOpening params matrixFlat opening)
          =
            (matrixFlat ++ value).extract 0 matrixFlat.size := by
              simp [
                SuperNeo.ProofSystem.Commitment.ppMatrixFlat,
                commitmentOfOpening,
                hMatrix,
                value
              ]
      _ = matrixFlat.extract := by
            exact Array.extract_append_left
      _ = matrixFlat := by
            exact Array.extract_size
  have hValuePart :
      SuperNeo.ProofSystem.Commitment.valueVec
        params
        (commitmentOfOpening params matrixFlat opening) =
      value := by
    calc
      SuperNeo.ProofSystem.Commitment.valueVec
          params
          (commitmentOfOpening params matrixFlat opening)
          =
            (matrixFlat ++ value).extract
              matrixFlat.size
              (matrixFlat.size + value.size) := by
              simp [
                SuperNeo.ProofSystem.Commitment.valueVec,
                commitmentOfOpening,
                hMatrix,
                hValueSize,
                value
              ]
      _ = value.extract 0 value.size := by
            exact Array.extract_append_right
      _ = value := by
            exact Array.extract_size
  refine ⟨?_, hOpeningWF, hNorm, ?_⟩
  · exact hPayloadWF
  · rw [hMatrixPart, hValuePart]

/--
Canonical Ajtai-backed commitment maps induce the assignment-opening adapter
used by the terminal reduced-handle soundness theorem.
-/
def assignmentOpeningAdapter_of_ajtaiBackedCommitMap
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {commitMap : SuperNeo.Coeffs → SuperNeo.ProofSystem.Commitment}
    (backing : AjtaiBackedCommitMap n params commitMap) :
    AssignmentOpeningAdapter n params commitMap where
  toOpening := backing.toOpening
  projectWitnessResidues := backing.projectWitnessResidues
  opens := by
    intro assignment
    rw [backing.commitMap_eq assignment]
    exact
      opensTo_commitmentOfOpening
        backing.matrixShape
        (backing.openingWellFormed assignment)
        (backing.openingNormSound assignment)
  bounded := backing.bounded
  residueSound := backing.residueSound

/--
The concrete `assignment ↦ M || Mz` Ajtai commitment map is backed by canonical
Ajtai openings when the fixed matrix, openings, bounds, and residue projection
are well formed.
-/
def ajtaiBackedCommitMap_of_ajtaiCommitMap
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {matrixFlat : Array SuperNeo.Coeffs}
    {toOpening :
      SuperNeo.Coeffs →
        SuperNeo.ProofSystem.Opening}
    {projectWitnessResidues : Array SuperNeo.Coeffs → Fin n → Nat}
    (hMatrix : matrixFlat.size = params.matrixFlatLen)
    (hOpeningWF :
      ∀ assignment,
        SuperNeo.ProofSystem.Opening.WellFormed
          params
          (toOpening assignment))
    (hOpeningNorm :
      ∀ assignment,
        SuperNeo.ProofSystem.Opening.NormSound
          (toOpening assignment))
    (hBounded :
      ∀ assignment,
        (toOpening assignment).normBound < params.bindingNormBound)
    (hResidues :
      ∀ assignment,
        projectWitnessResidues (toOpening assignment).witness =
          ParentOpeningAuthorization.assignmentResidues
            (n := n)
            assignment) :
    AjtaiBackedCommitMap
      n
      params
      (ajtaiCommitMap params matrixFlat toOpening) where
  matrixFlat := matrixFlat
  matrixShape := hMatrix
  toOpening := toOpening
  projectWitnessResidues := projectWitnessResidues
  commitMap_eq := by
    intro assignment
    rfl
  openingWellFormed := hOpeningWF
  openingNormSound := hOpeningNorm
  bounded := hBounded
  residueSound := hResidues

/--
If concrete Ajtai binding has no collision, two accepted openings of the same
commitment have equal witness arrays.
-/
theorem openingWitness_eq_of_noAjtaiBindingCollision
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {commitment : SuperNeo.ProofSystem.Commitment}
    {openingA openingB : SuperNeo.ProofSystem.Opening}
    (hNoCollision : NoAjtaiBindingCollision params)
    (hOpenA :
      SuperNeo.ProofSystem.opensTo params commitment openingA)
    (hOpenB :
      SuperNeo.ProofSystem.opensTo params commitment openingB)
    (hBoundA : openingA.normBound < params.bindingNormBound)
    (hBoundB : openingB.normBound < params.bindingNormBound) :
    openingA.witness = openingB.witness := by
  by_contra hDistinct
  exact
    hNoCollision
      ⟨{
        commitment := commitment
        opening1 := openingA
        opening2 := openingB
        distinct := hDistinct
        opens1 := hOpenA
        opens2 := hOpenB
        bounded1 := hBoundA
        bounded2 := hBoundB
      }⟩

/--
Concrete Ajtai no-collision plus an opening adapter proves the exact
commitment-map residue binding obligation needed by the reduced parent-handle
argument.
-/
theorem commitMapResiduesFunctional_of_noAjtaiBindingCollision
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {commitMap : SuperNeo.Coeffs → SuperNeo.ProofSystem.Commitment}
    (hNoCollision : NoAjtaiBindingCollision params)
    (adapter : AssignmentOpeningAdapter n params commitMap) :
    ParentOpeningAuthorization.CommitMapResiduesFunctional
      (n := n)
      commitMap := by
  intro assignmentA assignmentB hCommit
  have hOpenA :
      SuperNeo.ProofSystem.opensTo
        params
        (commitMap assignmentA)
        (adapter.toOpening assignmentA) :=
    adapter.opens assignmentA
  have hOpenB :
      SuperNeo.ProofSystem.opensTo
        params
        (commitMap assignmentA)
        (adapter.toOpening assignmentB) := by
    simpa [hCommit] using adapter.opens assignmentB
  have hWitness :
      (adapter.toOpening assignmentA).witness =
        (adapter.toOpening assignmentB).witness :=
    openingWitness_eq_of_noAjtaiBindingCollision
      hNoCollision
      hOpenA
      hOpenB
      (adapter.bounded assignmentA)
      (adapter.bounded assignmentB)
  calc
    ParentOpeningAuthorization.assignmentResidues
          (n := n)
          assignmentA
        = adapter.projectWitnessResidues
            (adapter.toOpening assignmentA).witness := by
            exact (adapter.residueSound assignmentA).symm
    _ = adapter.projectWitnessResidues
            (adapter.toOpening assignmentB).witness := by
            rw [hWitness]
    _ = ParentOpeningAuthorization.assignmentResidues
          (n := n)
          assignmentB := adapter.residueSound assignmentB

/--
Adapter connecting an accepted CE opening to the concrete Ajtai opening
relation.

Unlike `AssignmentOpeningAdapter`, this is local to accepted `CE.Holds`
witnesses. This is the exact protocol obligation: the parent residues used by
private `Pi_DEC` only need to be bound for CE openings that the terminal proof
accepts.
-/
structure CEOpeningAdapter
    (n : Nat)
    (params : SuperNeo.ProofSystem.AjtaiParams)
    (ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment) where
  toOpening :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Statement
      SuperNeo.ProofSystem.Commitment →
    SuperNeo.ProofSystem.ConstraintSystem.CE.Witness →
    SuperNeo.ProofSystem.Opening
  projectWitnessResidues : Array SuperNeo.Coeffs → Fin n → Nat
  opens :
    ∀ stmt wit,
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmt wit →
        SuperNeo.ProofSystem.opensTo
          params
          stmt.commitment
          (toOpening stmt wit)
  bounded :
    ∀ stmt wit,
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmt wit →
        (toOpening stmt wit).normBound < params.bindingNormBound
  residueSound :
    ∀ stmt wit,
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmt wit →
        projectWitnessResidues (toOpening stmt wit).witness =
          ParentOpeningAuthorization.assignmentResidues
            (n := n)
            wit.assignment

/--
An assignment-level Ajtai opening adapter for `ce.commitMap` induces the
CE-local adapter used by terminal parent-opening soundness.

The only extra fact needed is already part of `CE.Holds`:
`stmt.commitment = ce.commitMap wit.assignment`.
-/
def ceOpeningAdapter_of_assignmentOpeningAdapter
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    (adapter :
      AssignmentOpeningAdapter
        n
        params
        ce.commitMap) :
    CEOpeningAdapter n params ce where
  toOpening := fun _stmt wit => adapter.toOpening wit.assignment
  projectWitnessResidues := adapter.projectWitnessResidues
  opens := by
    intro stmt wit hHolds
    simpa [hHolds.1] using adapter.opens wit.assignment
  bounded := by
    intro _stmt wit _hHolds
    exact adapter.bounded wit.assignment
  residueSound := by
    intro _stmt wit _hHolds
    exact adapter.residueSound wit.assignment

/--
Concrete Ajtai no-collision plus a local CE-opening adapter proves fixed-CE
opening residue binding.
-/
theorem fixedCEOpeningResiduesFunctional_of_noAjtaiBindingCollision
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    (hNoCollision : NoAjtaiBindingCollision params)
    (adapter : CEOpeningAdapter n params ce) :
    ParentOpeningAuthorization.FixedCEOpeningResiduesFunctional
      (n := n)
      ce := by
  intro stmtA stmtB witA witB hCommitment hHoldsA hHoldsB
  let openingA := adapter.toOpening stmtA witA
  let openingB := adapter.toOpening stmtB witB
  have hOpenA :
      SuperNeo.ProofSystem.opensTo
        params
        stmtA.commitment
        openingA :=
    adapter.opens stmtA witA hHoldsA
  have hOpenB :
      SuperNeo.ProofSystem.opensTo
        params
        stmtA.commitment
        openingB := by
    simpa [openingB, hCommitment] using
      adapter.opens stmtB witB hHoldsB
  have hWitness : openingA.witness = openingB.witness :=
    openingWitness_eq_of_noAjtaiBindingCollision
      hNoCollision
      hOpenA
      hOpenB
      (adapter.bounded stmtA witA hHoldsA)
      (adapter.bounded stmtB witB hHoldsB)
  calc
    ParentOpeningAuthorization.witnessResidues
          (n := n)
          witA
        = ParentOpeningAuthorization.assignmentResidues
            (n := n)
            witA.assignment := by
            exact
              ParentOpeningAuthorization.witnessResidues_eq_assignmentResidues
                witA
    _ = adapter.projectWitnessResidues openingA.witness := by
            exact (adapter.residueSound stmtA witA hHoldsA).symm
    _ = adapter.projectWitnessResidues openingB.witness := by
            rw [hWitness]
    _ = ParentOpeningAuthorization.assignmentResidues
          (n := n)
          witB.assignment := adapter.residueSound stmtB witB hHoldsB
    _ = ParentOpeningAuthorization.witnessResidues
          (n := n)
          witB := by
            exact
              (ParentOpeningAuthorization.witnessResidues_eq_assignmentResidues
                witB).symm

/--
Concrete Ajtai no-collision plus CE-opening adapter and statement-encoding
consistency proves encoded-parent opening residue binding for a fixed CE
relation.
-/
theorem encodedParentCEBOpeningResiduesFunctionalFor_of_noAjtaiBindingCollision
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision : NoAjtaiBindingCollision params)
    (adapter : CEOpeningAdapter n params ce) :
    ParentOpeningAuthorization.EncodedParentCEBOpeningResiduesFunctionalFor
      (n := n)
      ce
      StatementEncodes :=
  ParentOpeningAuthorization.encodedParentCEBOpeningResiduesFunctionalFor_of_fixedCEBinding
    hEncoding
    (fixedCEOpeningResiduesFunctional_of_noAjtaiBindingCollision
      hNoCollision
      adapter)

end AjtaiResidueBinding

end DirectCcsFPrime
