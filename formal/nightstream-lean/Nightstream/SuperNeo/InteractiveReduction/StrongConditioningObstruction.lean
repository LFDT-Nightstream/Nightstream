import Nightstream.SuperNeo.InteractiveReduction.Paper

/-!
Kernel counterexample to reusing an unconditional repeated-witness error after
the first-success conditioning in SuperNeo Appendix D.4.

Owns: one four-seed experiment with two successful witnesses, the exact raw
two-run disagreement count, the exact first-success-conditioned disagreement
count, and their strict probability comparison.

Does not own: `Pi_CCS`, an unbounded rejection sampler, asymptotic negligible
functions, SumCheck, Schwartz--Zippel, Fiat--Shamir, Rust, R1CS, or costs.

Appendix D.4 first rejection-samples an ambient-successful run and then makes
one fresh run.  Conditioning the first run divides a raw two-run disagreement
probability by the ambient success probability.  Therefore an exact
quantitative strong-reduction theorem cannot subtract the unchanged raw
disagreement bound unless it separately accounts for that conditioning.
-/

namespace Nightstream.SuperNeo.InteractiveReduction.StrongConditioningObstruction

/-- Four equally likely verifier executions.  Seeds zero and one succeed and
carry different witnesses; seeds two and three reject. -/
abbrev Seed := Fin 4

def allSeeds : List Seed :=
  List.ofFn fun seed => seed

def output (seed : Seed) : Option Bool :=
  if seed.val = 0 then some false
  else if seed.val = 1 then some true
  else none

def succeeds (seed : Seed) : Bool :=
  (output seed).isSome

def successfulSeeds : List Seed :=
  allSeeds.filter succeeds

def pairs (left right : List Seed) : List (Seed × Seed) :=
  left.flatMap fun leftSeed =>
    right.map fun rightSeed => (leftSeed, rightSeed)

def successfulWitnessDisagreement (sample : Seed × Seed) : Bool :=
  match output sample.1, output sample.2 with
  | some left, some right => left != right
  | _, _ => false

/-- The Definition-10-style raw experiment uses two independent unconditioned
runs. -/
def rawPairs : List (Seed × Seed) :=
  pairs allSeeds allSeeds

/-- The Appendix-D.4 extractor conditions its first run on success and leaves
the second run fresh and unconditioned. -/
def firstSuccessConditionedPairs : List (Seed × Seed) :=
  pairs successfulSeeds allSeeds

def rawDisagreementCount : Nat :=
  (rawPairs.filter successfulWitnessDisagreement).length

def conditionedDisagreementCount : Nat :=
  (firstSuccessConditionedPairs.filter successfulWitnessDisagreement).length

theorem exact_counts :
    allSeeds.length = 4 /\
    successfulSeeds.length = 2 /\
    rawPairs.length = 16 /\
    firstSuccessConditionedPairs.length = 8 /\
    rawDisagreementCount = 2 /\
    conditionedDisagreementCount = 2 := by
  decide

/-- Cross multiplication proves that the extractor's conditioned disagreement
probability `2/8` is strictly larger than the raw repeated-run probability
`2/16`.  The numerator happens to be unchanged; the conditioning halves the
sample-space denominator. -/
theorem raw_bound_does_not_bound_conditioned_disagreement :
    rawDisagreementCount * firstSuccessConditionedPairs.length <
      conditionedDisagreementCount * rawPairs.length := by
  decide

/-- Headline obstruction with every count exposed for audit. -/
theorem unchanged_raw_uniqueness_budget_counterexample :
    rawDisagreementCount = 2 /\
    rawPairs.length = 16 /\
    conditionedDisagreementCount = 2 /\
    firstSuccessConditionedPairs.length = 8 /\
    rawDisagreementCount * firstSuccessConditionedPairs.length <
      conditionedDisagreementCount * rawPairs.length := by
  exact ⟨exact_counts.2.2.2.2.1, exact_counts.2.2.1,
    exact_counts.2.2.2.2.2, exact_counts.2.2.2.1,
    raw_bound_does_not_bound_conditioned_disagreement⟩

end Nightstream.SuperNeo.InteractiveReduction.StrongConditioningObstruction
