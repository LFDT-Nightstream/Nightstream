import SuperNeo.FPrimeRecursiveVerifierInterface

namespace tests.FPrimeRecursiveVerifier

open SuperNeo.FPrimeRecursiveVerifier

/-!
A two-block circuit in which both blocks enforce the same target predicate.
The second block is deliberately redundant, so the test exercises certified
block removal and verifies the structural row delta.
-/

inductive SmokeCheck where
  | required
  | duplicate
deriving Repr, DecidableEq

def smokeSemantics (_check : SmokeCheck) (input : Bool) : Prop :=
  input = true

def smokeTarget (input : Bool) : Prop :=
  input = true

def smokeChecks : Finset SmokeCheck :=
  { .required, .duplicate }

def smokeSemanticPlan :
    CertifiedPlan smokeSemantics smokeTarget where
  checks := smokeChecks
  sound := by
    intro input hAccepts
    exact hAccepts .required (by simp [smokeChecks])
  complete := by
    intro input hTarget _check _hCheck
    exact hTarget

def smokeColumn (index coefficient : Nat) : LinearCombination Nat :=
  { terms := [(index, coefficient)] }

def smokeRow : R1csConstraint Nat :=
  { a := smokeColumn 0 1
    b := smokeColumn 1 1
    c := smokeColumn 1 1 }

def smokeBlock : R1csBlock Nat :=
  { columns := 2
    constraints := [smokeRow] }

def smokeAssignment
    (_check : SmokeCheck)
    (input : Bool)
    (_witness : Unit)
    (column : Nat) : Nat :=
  if column = 0 then
    if input = true then 1 else 0
  else if column = 1 then
    1
  else
    0

def smokeEncoding :
    ModularR1csEncoding Nat Bool SmokeCheck Unit :=
  { block := fun _check => smokeBlock
    assignment := smokeAssignment }

def smokeRefinement :
    BlockRefinement smokeEncoding smokeSemantics where
  wellFormed := by
    intro check
    simp [smokeEncoding, smokeBlock, smokeRow, smokeColumn,
      R1csBlock.WellFormed, R1csConstraint.WellFormed,
      LinearCombination.WellFormed]
  sound := by
    intro check input witness hSatisfied
    cases input with
    | false =>
        have hRow :
            smokeRow.Holds
              (smokeEncoding.assignment check false witness) :=
          hSatisfied smokeRow (by simp [smokeEncoding, smokeBlock])
        norm_num [smokeEncoding, smokeAssignment, smokeRow, smokeColumn,
          R1csConstraint.Holds, LinearCombination.eval] at hRow
    | true =>
        rfl

def smokeCompilerComplete :
    PlanWitnessComplete smokeEncoding smokeSemantics smokeChecks := by
  intro input hAccepts
  cases input with
  | false =>
      have hRequired := hAccepts .required (by simp [smokeChecks])
      simp [smokeSemantics] at hRequired
  | true =>
      refine ⟨(), ?_⟩
      intro check hCheck
      simp [smokeEncoding, smokeBlock, smokeRow, smokeColumn,
        smokeAssignment, R1csBlock.Satisfied, R1csConstraint.Holds,
        LinearCombination.eval]

def smokeCandidate :
    CertifiedR1csPlan smokeEncoding smokeSemantics smokeTarget where
  semanticPlan := smokeSemanticPlan
  refinement := smokeRefinement
  compilerComplete := smokeCompilerComplete

theorem duplicate_redundant :
    Redundant smokeSemantics smokeChecks .duplicate := by
  intro input hWithout
  exact hWithout .required (by simp [smokeChecks])

def prunedSmokeCandidate :
    CertifiedR1csPlan smokeEncoding smokeSemantics smokeTarget :=
  smokeCandidate.eraseRedundant .duplicate duplicate_redundant

example :
    prunedSmokeCandidate.semanticPlan.checks = { .required } := by
  decide

example :
    R1csExactForTarget
      smokeEncoding smokeTarget
      prunedSmokeCandidate.semanticPlan.checks :=
  prunedSmokeCandidate.exact

example : (compiledCost smokeEncoding smokeChecks).rows = 2 := by
  native_decide

example :
    (compiledCost
      smokeEncoding prunedSmokeCandidate.semanticPlan.checks).rows = 1 := by
  native_decide

example : (compiledCost smokeEncoding smokeChecks).nonzeros = 6 := by
  native_decide

example :
    (compiledCost
      smokeEncoding prunedSmokeCandidate.semanticPlan.checks).nonzeros = 3 := by
  native_decide

/-! Regression checks for the concrete theorem surfaces. -/

example :
    InclusionMinimalSound
      (checkSemantics booleanPredicates)
      (PaperRecursiveStep booleanPredicates)
      essentialChecks :=
  booleanEssentialPlan_inclusionMinimalSound

example :
    fullPostRlcPlanWithoutDec.checks = minimalPostRlcChecks :=
  fullPostRlcPlanWithoutDec_checks

def projectionContext :
    PiCcsOutputContext Nat Nat Nat Nat Nat Nat Nat :=
  { shape := 1
    commitment := 2
    publicX := 3
    rowPoint := 4
    columnPoint := 5
    foldDigest := 6
    sidecars := 7 }

def projectionMessage : PiCcsOutputMessage Nat Nat :=
  { yRing := 11, yZcol := 13 }

example :
    piCcsOutputMessage
      (reconstructPiCcsOutput (fun y => y + 1)
        projectionContext projectionMessage) = projectionMessage := by
  rfl

end tests.FPrimeRecursiveVerifier
