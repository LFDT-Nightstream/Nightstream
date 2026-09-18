import SuperNeo.SumCheck

open SuperNeo

namespace tests

private def f (n : Nat) : F :=
  F.ofNat n

def proofOneTable : Array F := #[f 1, f 2]

def proofOneInst : SumCheckInstance :=
  { rounds := 1
    maxDegree := 1
    domainSize := 1
    claimedValue := sumcheckTableSum proofOneTable }

def proofOneStmt : SumCheckStatement proofOneInst :=
  { parameterConsistent := by
      simp [sumcheckParameterConsistent, proofOneInst]
    degreeCompatible := by
      simp [sumcheckDegreeCompatible, proofOneInst]
    table := proofOneTable
    tableSize := by rfl
    hypercubeSumEqClaimed := by rfl }

def proofOneTr : SumCheckTranscript :=
  sumcheckHonestTranscript proofOneStmt

def proofOneBadChallenges : SumCheckTranscript :=
  { proofOneTr with challenges := #[] }

def proofOneBadInitial : SumCheckTranscript :=
  { proofOneTr with roundPolys := #[#[f 0, f 0]] }

def proofOneWrongTable : Array F := #[f 0, f 3]

def proofOneWrongStmt : SumCheckStatement proofOneInst :=
  { parameterConsistent := by
      simp [sumcheckParameterConsistent, proofOneInst]
    degreeCompatible := by
      simp [sumcheckDegreeCompatible, proofOneInst]
    table := proofOneWrongTable
    tableSize := by rfl
    hypercubeSumEqClaimed := by rfl }

example : sumcheckAccepted proofOneInst proofOneTr :=
  sumcheckHonestTranscript_accepted_of_statement proofOneStmt

example : sumcheckFinalOracleConsistent proofOneInst proofOneStmt proofOneTr :=
  sumcheckFinalOracleConsistent_of_statement_constructive proofOneStmt

example :
    sumcheckStatementTranscriptConsistent proofOneInst proofOneStmt proofOneTr :=
  sumcheckStatementTranscriptConsistent_of_statement_constructive proofOneStmt

example : ¬ sumcheckAccepted proofOneInst proofOneBadChallenges := by
  apply sumcheckAccepted_not_of_challenge_size_ne
  simp [proofOneBadChallenges, proofOneInst]

example : ¬ sumcheckAccepted proofOneInst proofOneBadInitial := by
  apply sumcheckAccepted_not_of_bad_initial_round
  change ¬
    (sumcheckEvalPoly #[f 0, f 0] 0 + sumcheckEvalPoly #[f 0, f 0] 1 =
      proofOneInst.claimedValue)
  native_decide

example : ¬ sumcheckFinalOracleConsistent proofOneInst proofOneWrongStmt proofOneTr := by
  intro hFinal
  have hEval :
      sumcheckEvalPoly (proofOneTr.roundPolys[proofOneInst.rounds - 1]!)
        (proofOneTr.challenges[proofOneInst.rounds - 1]!) =
      mleByFolding proofOneWrongStmt.table proofOneTr.challenges := hFinal.2
  have hEvalNe :
      ¬
        (sumcheckEvalPoly (proofOneTr.roundPolys[proofOneInst.rounds - 1]!)
          (proofOneTr.challenges[proofOneInst.rounds - 1]!) =
        mleByFolding proofOneWrongStmt.table proofOneTr.challenges) := by
    native_decide
  exact hEvalNe hEval

def proofTwoTable : Array F := #[f 1, f 2, f 3, f 4]

def proofTwoInst : SumCheckInstance :=
  { rounds := 2
    maxDegree := 1
    domainSize := 1
    claimedValue := sumcheckTableSum proofTwoTable }

def proofTwoStmt : SumCheckStatement proofTwoInst :=
  { parameterConsistent := by
      simp [sumcheckParameterConsistent, proofTwoInst]
    degreeCompatible := by
      simp [sumcheckDegreeCompatible, proofTwoInst]
    table := proofTwoTable
    tableSize := by rfl
    hypercubeSumEqClaimed := by rfl }

def proofTwoTr : SumCheckTranscript :=
  sumcheckHonestTranscript proofTwoStmt

def proofTwoBadRoundPolyCount : SumCheckTranscript :=
  { proofTwoTr with roundPolys := #[proofTwoTr.roundPolys[0]!] }

def proofTwoBadShape : SumCheckTranscript :=
  { proofTwoTr with roundPolys := #[proofTwoTr.roundPolys[0]!, #[f 0, f 0, f 0]] }

def proofTwoBadFold : SumCheckTranscript :=
  { proofTwoTr with roundPolys := #[proofTwoTr.roundPolys[0]!, #[f 0, f 0]] }

example : sumcheckAccepted proofTwoInst proofTwoTr :=
  sumcheckHonestTranscript_accepted_of_statement proofTwoStmt

example :
    sumcheckStatementTranscriptConsistent proofTwoInst proofTwoStmt proofTwoTr :=
  sumcheckStatementTranscriptConsistent_of_statement_constructive proofTwoStmt

example : ¬ sumcheckAccepted proofTwoInst proofTwoBadRoundPolyCount := by
  apply sumcheckAccepted_not_of_roundpoly_size_ne
  simp [proofTwoBadRoundPolyCount, proofTwoInst]

example : ¬ sumcheckAccepted proofTwoInst proofTwoBadShape := by
  apply sumcheckAccepted_not_of_bad_round_shape
  refine ⟨⟨1, by decide⟩, ?_⟩
  simp [sumcheckRoundPolyShape, proofTwoBadShape, proofTwoInst]

example : ¬ sumcheckAccepted proofTwoInst proofTwoBadFold := by
  intro hAcc
  have hStep := sumcheckAccepted_fold_step hAcc (i := 0) (by decide : 0 + 1 < proofTwoBadFold.roundPolys.size)
  change
    sumcheckEvalPoly #[f 0, f 0] 0 + sumcheckEvalPoly #[f 0, f 0] 1 =
      sumcheckEvalPoly (proofTwoTr.roundPolys[0]!) (proofTwoTr.challenges[0]!) at hStep
  have hImpossible :
      ¬
        (sumcheckEvalPoly #[f 0, f 0] 0 + sumcheckEvalPoly #[f 0, f 0] 1 =
          sumcheckEvalPoly (proofTwoTr.roundPolys[0]!) (proofTwoTr.challenges[0]!)) := by
    native_decide
  exact hImpossible hStep

def main : IO Unit := pure ()

end tests
