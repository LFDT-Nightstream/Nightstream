import Nightstream.SuperNeo.SumCheck

/-! Positive and adversarial semantic tests for the SumCheck reduction. -/

namespace NightstreamTests.SumCheck

open Nightstream.SuperNeo.SumCheck

def natOps : Ops Nat Nat where
  zero := 0
  one := 1
  add := Nat.add

def honestPolynomial : Nat → Nat
  | 0 => 2
  | 1 => 3
  | _ => 7

def honestRound : Round Nat Nat where
  claimed := honestPolynomial
  expected := honestPolynomial
  challenge := 2
  degree := 2

def honestTranscript : Instance Nat Nat where
  claimedInitial := 5
  trueInitial := 5
  terminal := 7
  rounds := [honestRound]
  maxDegree := 2
  challengeSetSize := 97

example : TruthPath natOps honestTranscript := by
  simp [TruthPath, Chain, honestTranscript, honestRound, honestPolynomial, natOps]

example : Honest honestTranscript := by
  refine ⟨rfl, ?_, by simp [honestTranscript, honestRound]⟩
  intro round roundInTranscript
  simp only [honestTranscript, List.mem_cons, List.not_mem_nil, or_false] at roundInTranscript
  subst round
  rfl

example : Accepted natOps honestTranscript :=
  complete natOps honestTranscript (by
    simp [TruthPath, Chain, honestTranscript, honestRound, honestPolynomial, natOps]) (by
    refine ⟨rfl, ?_, by simp [honestTranscript, honestRound]⟩
    intro round roundInTranscript
    simp only [honestTranscript, List.mem_cons, List.not_mem_nil, or_false] at roundInTranscript
    subst round
    rfl)

example : check natOps honestTranscript = true := by decide

def forgedPolynomial : Nat → Nat
  | 0 => 4
  | 1 => 4
  | _ => 7

/-- The prover changes the claimed sum from five to eight but collides at challenge two. -/
def forgedRound : Round Nat Nat where
  claimed := forgedPolynomial
  expected := honestPolynomial
  challenge := 2
  degree := 2

def forgedTranscript : Instance Nat Nat where
  claimedInitial := 8
  trueInitial := 5
  terminal := 7
  rounds := [forgedRound]
  maxDegree := 2
  challengeSetSize := 97

example : Accepted natOps forgedTranscript := by
  simp [Accepted, Chain, forgedTranscript, forgedRound, forgedPolynomial, natOps]

/-- Acceptance alone is intentionally possible for this false claim. -/
example : check natOps forgedTranscript = true := by decide

example : TruthPath natOps forgedTranscript := by
  simp [TruthPath, Chain, forgedTranscript, forgedRound, honestPolynomial, natOps]

example : ¬ Claim.True forgedTranscript := by
  simp [Claim.True, forgedTranscript]

/-- False acceptance is not called valid; it exposes the sampled collision. -/
example : ∃ round, BadChallenge forgedTranscript round :=
  false_acceptance_implies_bad_challenge natOps forgedTranscript
    (by
      simp [Accepted, Chain, forgedTranscript, forgedRound, forgedPolynomial, natOps])
    (by
      simp [TruthPath, Chain, forgedTranscript, forgedRound, honestPolynomial, natOps])
    (by simp [Claim.True, forgedTranscript])

/-- A malformed claimed-chain equation is rejected before any security theorem. -/
example : ¬ Accepted natOps { forgedTranscript with claimedInitial := 9 } := by
  simp [Accepted, Chain, forgedTranscript, forgedRound, forgedPolynomial, natOps]

example : check natOps { forgedTranscript with claimedInitial := 9 } = false := by decide

end NightstreamTests.SumCheck
