import SuperNeo.SumCheck

open SuperNeo

namespace tests

private def f (n : Nat) : F :=
  F.ofNat n

private def sumcheckRoundConsistentB
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Bool :=
  decide (tr.challenges.size = inst.rounds) &&
    decide (tr.roundPolys.size = inst.rounds)

private def sumcheckRoundPolyShapeB
  (inst : SumCheckInstance)
  (poly : Array F) : Bool :=
  decide (poly.size = inst.maxDegree + 1)

private def sumcheckRoundShapesB
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Bool :=
  tr.roundPolys.all (sumcheckRoundPolyShapeB inst)

private def sumcheckFoldConsistentB
  (tr : SumCheckTranscript) : Bool :=
  decide (tr.challenges.size = tr.roundPolys.size) &&
    (List.range tr.roundPolys.size).all fun i =>
      if _h : i + 1 < tr.roundPolys.size then
        decide
          (sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
              sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
            sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!))
      else
        true

private def sumcheckInitialRoundConsistentB
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Bool :=
  if _hZero : tr.roundPolys.size = 0 then
    true
  else
    decide
      (sumcheckEvalPoly (tr.roundPolys[0]!) 0 +
          sumcheckEvalPoly (tr.roundPolys[0]!) 1 =
        inst.claimedValue)

private def sumcheckParameterConsistentB (inst : SumCheckInstance) : Bool :=
  decide (inst.maxDegree ≤ inst.domainSize)

private def sumcheckDegreeCompatibleB (inst : SumCheckInstance) : Bool :=
  decide (inst.rounds = 0 ∨ 0 < inst.maxDegree)

private def sumcheckAcceptedB
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Bool :=
  sumcheckParameterConsistentB inst &&
    sumcheckDegreeCompatibleB inst &&
    sumcheckRoundConsistentB inst tr &&
    sumcheckRoundShapesB inst tr &&
    sumcheckInitialRoundConsistentB inst tr &&
    sumcheckFoldConsistentB tr

private def sumcheckFinalOracleConsistentWithTableB
  (inst : SumCheckInstance)
  (table : Array F)
  (tr : SumCheckTranscript) : Bool :=
  decide (table.size = 2 ^ inst.rounds) &&
    if _hZero : inst.rounds = 0 then
      decide (mleByFolding table #[] = inst.claimedValue)
    else
      decide
        (sumcheckEvalPoly (tr.roundPolys[inst.rounds - 1]!) (tr.challenges[inst.rounds - 1]!) =
          mleByFolding table tr.challenges)

private def sumcheckAcceptedForTableB
  (inst : SumCheckInstance)
  (table : Array F)
  (tr : SumCheckTranscript) : Bool :=
  sumcheckAcceptedB inst tr &&
    sumcheckFinalOracleConsistentWithTableB inst table tr

def zeroInst : SumCheckInstance :=
  { rounds := 0
    maxDegree := 0
    domainSize := 0
    claimedValue := 0 }

def zeroTr : SumCheckTranscript :=
  { challenges := #[]
    roundPolys := #[] }

def zeroTable : Array F := #[0]

#guard sumcheckAcceptedB zeroInst zeroTr = true
#guard sumcheckAcceptedForTableB zeroInst zeroTable zeroTr = true

def zeroBadTr : SumCheckTranscript :=
  { challenges := #[f 0]
    roundPolys := #[] }

#guard sumcheckAcceptedB zeroInst zeroBadTr = false

def zeroBadPolyTr : SumCheckTranscript :=
  { challenges := #[]
    roundPolys := #[#[f 0]] }

#guard sumcheckAcceptedB zeroInst zeroBadPolyTr = false

def zeroWrongSizeTable : Array F := #[]

#guard sumcheckAcceptedForTableB zeroInst zeroWrongSizeTable zeroTr = false

def zeroWrongValueTable : Array F := #[f 1]

#guard sumcheckAcceptedForTableB zeroInst zeroWrongValueTable zeroTr = false

def zeroBadDegreeInst : SumCheckInstance :=
  { rounds := 1
    maxDegree := 0
    domainSize := 0
    claimedValue := 0 }

#guard sumcheckAcceptedB zeroBadDegreeInst zeroTr = false

def oneTable : Array F := #[f 1, f 2]

def oneInst : SumCheckInstance :=
  { rounds := 1
    maxDegree := 1
    domainSize := 1
    claimedValue := sumcheckTableSum oneTable }

def oneStmt : SumCheckStatement oneInst :=
  { parameterConsistent := by
      simp [sumcheckParameterConsistent, oneInst]
    degreeCompatible := by
      simp [sumcheckDegreeCompatible, oneInst]
    table := oneTable
    tableSize := by rfl
    hypercubeSumEqClaimed := by rfl }

def oneTr : SumCheckTranscript :=
  sumcheckHonestTranscript oneStmt

#guard sumcheckAcceptedB oneInst oneTr = true
#guard sumcheckAcceptedForTableB oneInst oneTable oneTr = true

def oneBadChallenges : SumCheckTranscript :=
  { oneTr with challenges := #[] }

#guard sumcheckAcceptedB oneInst oneBadChallenges = false

def oneBadInitial : SumCheckTranscript :=
  { oneTr with roundPolys := #[#[f 0, f 0]] }

#guard sumcheckAcceptedB oneInst oneBadInitial = false

def oneBadShape : SumCheckTranscript :=
  { oneTr with roundPolys := #[#[f 0]] }

#guard sumcheckAcceptedB oneInst oneBadShape = false

def oneWrongTable : Array F := #[f 0, f 2]

#guard sumcheckAcceptedForTableB oneInst oneWrongTable oneTr = false

def oneWrongSizeTable : Array F := #[f 3]

#guard sumcheckAcceptedForTableB oneInst oneWrongSizeTable oneTr = false

def oneBadDegreeInst : SumCheckInstance :=
  { rounds := 1
    maxDegree := 0
    domainSize := 0
    claimedValue := oneInst.claimedValue }

def oneBadDegreeTr : SumCheckTranscript :=
  { challenges := #[f 0]
    roundPolys := #[#[oneInst.claimedValue]] }

#guard sumcheckAcceptedB oneBadDegreeInst oneBadDegreeTr = false

def twoTable : Array F := #[f 1, f 2, f 3, f 4]

def twoInst : SumCheckInstance :=
  { rounds := 2
    maxDegree := 1
    domainSize := 1
    claimedValue := sumcheckTableSum twoTable }

def twoStmt : SumCheckStatement twoInst :=
  { parameterConsistent := by
      simp [sumcheckParameterConsistent, twoInst]
    degreeCompatible := by
      simp [sumcheckDegreeCompatible, twoInst]
    table := twoTable
    tableSize := by rfl
    hypercubeSumEqClaimed := by rfl }

def twoTr : SumCheckTranscript :=
  sumcheckHonestTranscript twoStmt

#guard sumcheckAcceptedB twoInst twoTr = true
#guard sumcheckAcceptedForTableB twoInst twoTable twoTr = true

def twoBadFold : SumCheckTranscript :=
  { challenges := twoTr.challenges
    roundPolys := #[twoTr.roundPolys[0]!, #[f 0, f 0]] }

#guard sumcheckAcceptedB twoInst twoBadFold = false

def twoWrongTable : Array F := #[f 9, f 2, f 3, f 4]

#guard sumcheckAcceptedForTableB twoInst twoWrongTable twoTr = false

def twoBadChallenges : SumCheckTranscript :=
  { twoTr with challenges := #[twoTr.challenges[0]! ] }

#guard sumcheckAcceptedB twoInst twoBadChallenges = false

def twoBadShape : SumCheckTranscript :=
  { twoTr with roundPolys := #[twoTr.roundPolys[0]!, #[f 0, f 0, f 0]] }

#guard sumcheckAcceptedB twoInst twoBadShape = false

def main : IO Unit := pure ()

end tests
