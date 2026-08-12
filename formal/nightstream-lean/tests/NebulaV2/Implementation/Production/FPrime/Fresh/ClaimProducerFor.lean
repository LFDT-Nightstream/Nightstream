import Nightstream.Implementation.NebulaV2.Production.FPrime.Fresh.ClaimProducerFor

open Nightstream.Implementation.NebulaV2.ProductionFreshClaimProducerFor

#check value_bundle_opens
#check value_ccs_fullMatches
#check value_canonical
#check EncodedFreshRelationWitnessForRows
#check EncodedFreshRelationWitnessForRows.toFreshRelationWitness
#check FreshRelationWitnessForRows
#check FreshRelationWitnessForRows.publicOutput
#check FreshRelationWitnessForRows.norm
#check FreshRelationWitnessForRows.relation
#check FreshRelationWitnessForRows.authorityRows
#check FreshRelationWitnessForRows.selectedBranch
#check FreshRelationWitnessForRows.exists_of_ccsHolds
#check RelationAuthority.ExactDecodedBranch
#check RelationAuthority.selectedBranchOfCcsPublic
#check RelationAuthority.selectedBranchOfCcsAssignment
#check freshStatement_holds
#check freshStatement_holds_from_rows
#check freshStatement_holds_iff_exists_rows
#check producedFresh_commitment
#check producedFresh_publicInput

open Nightstream.Implementation.R1CS

def weakSeparateAssignments
    (rows : List Row) (committed semantic : Nat -> Nat) : Prop :=
  Satisfies rows committed /\ Satisfies rows semantic

/-- Two separate satisfying-assignment premises do not bind row semantics to
the committed witness. The empty relation gives the smallest counterexample. -/
theorem weak_separate_assignments_can_disagree :
    weakSeparateAssignments [] (fun _ => 0) (fun _ => 1) /\
      (fun _ : Nat => 0) ≠ (fun _ => 1) := by
  constructor
  · exact ⟨by simp [Satisfies], by simp [Satisfies]⟩
  · intro equal
    have atZero := congrFun equal 0
    omega
