import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth

/-! The deterministic part of HyperNova Lemma 17 (Appendix H.3) for one slot of the bounded-depth scheme, in
the form the probabilistic layer uses. The reverse extractor's stage `j` hands its NIFS
extractor an `Attempt`: the `j`-th fold from the end with a witness of the fold's output claim,
which is stage `j − 1`'s answer. `returned_of_no_failure`: on an accepted history, when no stage
fails, the answers open every input claim at its recovered bound and every fresh instance. Every
bound stays within `B + cap · T · (b − 1)`, so Assumption 1 is needed only there.

A link carries the prior state that the adversary outputs with the fold (`Link.prior`), and an
accepted history and a winning attempt require `PriorLink` of it. A fold's transcript absorbs the
fresh instance and not the running claim; the fresh instance's public input commits to the prior
state, whose slot holds the running claim. Without an output prior, the link would hold for
almost every running claim, because many states share one digest (PR #155). The relations are
keyed: a claim holds under the key of the slot that it lives in. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthReverse

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth
open NightstreamFPrime.Spec.Folding.Schedule (Step extractedBefore)

variable {Prior Key Claim Fresh Proof ClaimWitness FreshWitness : Type}

/-- A fold, the running claim it starts from, and the prior state that the adversary says the
fold's fresh instance commits to. -/
structure Link (Prior Key Claim Fresh Proof : Type) where
  prior : Prior
  start : Running Claim
  fold : Fold Key Claim Fresh Proof

def chainBound (p : GlobalParams) (cap : Nat) : Nat :=
  p.bigB + cap * (p.expansionT * (p.b - 1))

/-- The folds a slot with key `key` receives, last fold first, ending at running claim `final`,
each linked to its prior state. -/
def RunsBack (cap : Nat) (folds : Folds Key Claim Fresh Proof)
    (PriorLink : Prior → Running Claim → Fold Key Claim Fresh Proof → Prop) (key : Key) :
    Running Claim → List (Link Prior Key Claim Fresh Proof) → Prop
  | _, [] => True
  | final, link :: older =>
      link.fold.output = final ∧ link.fold.input = link.start.1 ∧ link.fold.key = key ∧
        PriorLink link.prior link.start link.fold ∧
        Accepts (verifier cap folds link.fold.circuit) link.fold.key link.start link.fold.fresh
          link.fold.proof link.fold.output ∧
        RunsBack cap folds PriorLink key link.start older

/-- A fold and a witness of its output claim at `bound`. -/
structure Attempt (Prior Key Claim Fresh Proof ClaimWitness : Type) where
  link : Link Prior Key Claim Fresh Proof
  bound : Nat
  witness : ClaimWitness

section Events

variable (p : GlobalParams) (cap : Nat) (folds : Folds Key Claim Fresh Proof)
  (PriorLink : Prior → Running Claim → Fold Key Claim Fresh Proof → Prop)
  (Holds : Key → Nat → Claim → ClaimWitness → Prop)
  (FreshHolds : Key → Fresh → FreshWitness → Prop)

/-- The NIFS adversary wins: the fold is linked to the prior state that the adversary outputs,
the fold is accepted, its output claim opens at `bound`, and `bound` is within the chain bound. -/
def Succeeds (attempt : Attempt Prior Key Claim Fresh Proof ClaimWitness) : Prop :=
  PriorLink attempt.link.prior attempt.link.start attempt.link.fold ∧
    attempt.link.fold.input = attempt.link.start.1 ∧
    Accepts (verifier cap folds attempt.link.fold.circuit) attempt.link.fold.key attempt.link.start
      attempt.link.fold.fresh attempt.link.fold.proof attempt.link.fold.output ∧
    Holds attempt.link.fold.key attempt.bound attempt.link.fold.output.1 attempt.witness ∧
    attempt.bound ≤ chainBound p cap

/-- The answer opens the fold's input claim at `extractedBefore` of the bound, and its fresh
instance. -/
def Extracted (attempt : Attempt Prior Key Claim Fresh Proof ClaimWitness)
    (answer : ClaimWitness × FreshWitness) : Prop :=
  Holds attempt.link.fold.key
      (extractedBefore p (circuitStep attempt.link.fold.circuit) attempt.bound)
      attempt.link.fold.input answer.1 ∧
    FreshHolds attempt.link.fold.key attempt.link.fold.fresh answer.2

/-- The stages' attempts, last fold first: each stage starts from the previous stage's answer. -/
def attempts : Nat → ClaimWitness → List (Link Prior Key Claim Fresh Proof) →
    List (ClaimWitness × FreshWitness) → List (Attempt Prior Key Claim Fresh Proof ClaimWitness)
  | _, _, [], _ => []
  | bound, witness, link :: older, results =>
      ⟨link, bound, witness⟩ :: match results with
        | answer :: rest =>
            attempts (extractedBefore p (circuitStep link.fold.circuit) bound) answer.1 older rest
        | [] => []

/-- Stage `j` fails: its attempt succeeds and its answer does not extract. -/
def Fails (bound : Nat) (witness : ClaimWitness) (back : List (Link Prior Key Claim Fresh Proof))
    (results : List (ClaimWitness × FreshWitness)) (j : Nat) : Prop :=
  ∃ attempt answer, (attempts p bound witness back results)[j]? = some attempt ∧
    results[j]? = some answer ∧ Succeeds p cap folds PriorLink Holds attempt ∧
    ¬ Extracted p Holds FreshHolds attempt answer

/-- The answers extract every fold of the history. -/
def Returned (bound : Nat) (witness : ClaimWitness) (back : List (Link Prior Key Claim Fresh Proof))
    (results : List (ClaimWitness × FreshWitness)) : Prop :=
  ∀ j < back.length, ∃ attempt answer, (attempts p bound witness back results)[j]? = some attempt ∧
    results[j]? = some answer ∧ Extracted p Holds FreshHolds attempt answer

end Events

variable {p : GlobalParams} {cap : Nat} {folds : Folds Key Claim Fresh Proof}
  {PriorLink : Prior → Running Claim → Fold Key Claim Fresh Proof → Prop}
  {Holds : Key → Nat → Claim → ClaimWitness → Prop}
  {FreshHolds : Key → Fresh → FreshWitness → Prop}

/-- Stage `j`'s attempt reads only the first `j` answers. -/
theorem attempts_take :
    ∀ (back : List (Link Prior Key Claim Fresh Proof)) (bound : Nat) (witness : ClaimWitness)
      (results : List (ClaimWitness × FreshWitness)) (j : Nat),
      (attempts p bound witness back results)[j]? =
        (attempts p bound witness back (results.take j))[j]?
  | [], _, _, _, _ => rfl
  | _ :: _, _, _, _, 0 => by
      simp only [attempts, List.getElem?_cons_zero]
  | _ :: _, _, _, [], _ + 1 => rfl
  | link :: older, bound, _, answer :: rest, j + 1 => by
      simp only [attempts, List.take_succ_cons, List.getElem?_cons_succ]
      exact attempts_take older _ answer.1 rest j

theorem fails_take (bound : Nat) (witness : ClaimWitness)
    (back : List (Link Prior Key Claim Fresh Proof)) (results : List (ClaimWitness × FreshWitness))
    (j : Nat) :
    Fails p cap folds PriorLink Holds FreshHolds bound witness back results j ↔
      Fails p cap folds PriorLink Holds FreshHolds bound witness back (results.take (j + 1)) j := by
  have sameAttempt : (attempts p bound witness back results)[j]? =
      (attempts p bound witness back (results.take (j + 1)))[j]? := by
    rw [attempts_take back bound witness results j,
      attempts_take back bound witness (results.take (j + 1)) j, List.take_take,
      Nat.min_eq_left (Nat.le_succ j)]
  have sameAnswer : results[j]? = (results.take (j + 1))[j]? := by
    simp
  simp only [Fails, sameAttempt, sameAnswer]

/-- HyperNova Lemma 17, one slot, without probabilities: on an accepted history whose stages do
not fail, the answers extract every fold. The bound invariant is the one of
`Folding.Schedule.recoveredBound_le`, read from the last fold back. -/
theorem returned_of_no_failure (key : Key) :
    ∀ (back : List (Link Prior Key Claim Fresh Proof)) (final : Running Claim) (bound : Nat)
      (witness : ClaimWitness) (results : List (ClaimWitness × FreshWitness)),
      RunsBack cap folds PriorLink key final back → Holds key bound final.1 witness →
      bound ≤ p.bigB + (cap - final.2) * (p.expansionT * (p.b - 1)) →
      back.length ≤ results.length →
      (∀ j < back.length, ¬ Fails p cap folds PriorLink Holds FreshHolds bound witness back results j) →
      Returned p Holds FreshHolds bound witness back results
  | [], _, _, _, _, _, _, _, _, _ => fun j small => absurd small (Nat.not_lt_zero j)
  | _ :: _, _, _, _, [], _, _, _, long, _ => by
      simp at long
  | link :: older, final, bound, witness, answer :: rest, runs, holds, within, long,
      noFail => by
      obtain ⟨outputEq, inputEq, keyEq, linked, accepted, olderRuns⟩ := runs
      have shrink : (cap - final.2) * (p.expansionT * (p.b - 1)) ≤
          cap * (p.expansionT * (p.b - 1)) :=
        Nat.mul_le_mul_right _ (Nat.sub_le _ _)
      have succeeds : Succeeds p cap folds PriorLink Holds ⟨link, bound, witness⟩ := by
        refine ⟨linked, inputEq, accepted, ?_, ?_⟩
        · show Holds link.fold.key bound link.fold.output.1 witness
          rw [outputEq, keyEq]
          exact holds
        · show bound ≤ p.bigB + cap * (p.expansionT * (p.b - 1))
          omega
      have extracted : Extracted p Holds FreshHolds ⟨link, bound, witness⟩ answer :=
        Classical.byContradiction fun missing =>
          noFail 0 (by simp) ⟨_, answer, rfl, rfl, succeeds, missing⟩
      have nextHolds : Holds key (extractedBefore p (circuitStep link.fold.circuit) bound)
          link.start.1 answer.1 := by
        rw [← inputEq, ← keyEq]
        exact extracted.1
      have nextWithin : extractedBefore p (circuitStep link.fold.circuit) bound ≤
          p.bigB + (cap - link.start.2) * (p.expansionT * (p.b - 1)) := by
        obtain ⟨room, _, moved⟩ := (accepts_iff cap folds link.fold.circuit link.fold.key
          link.start.1 link.start.2 link.fold.fresh link.fold.proof link.fold.output).mp accepted
        rw [outputEq] at moved
        cases kind : circuitStep link.fold.circuit with
        | light =>
            have below := room kind
            simp only [kind, Folding.Schedule.nextDepth] at moved
            have steps : cap - link.start.2 = (cap - final.2) + 1 := by omega
            simp only [extractedBefore]
            rw [steps, Nat.add_mul, Nat.one_mul]
            omega
        | refresh =>
            simp only [extractedBefore]
            omega
      have later := returned_of_no_failure key older link.start _ answer.1 rest olderRuns
        nextHolds nextWithin (by simp at long; omega) (fun j small failed => by
          apply noFail (j + 1) (by simp; omega)
          obtain ⟨attempt, given, attemptEq, answerEq, wins, fails⟩ := failed
          exact ⟨attempt, given, by simpa [attempts] using attemptEq, by simpa using answerEq,
            wins, fails⟩)
      intro j small
      cases j with
      | zero => exact ⟨_, answer, rfl, rfl, extracted⟩
      | succ j =>
          obtain ⟨attempt, given, attemptEq, answerEq, extracts⟩ := later j (by simp at small; omega)
          exact ⟨attempt, given, by simpa [attempts] using attemptEq, by simpa using answerEq,
            extracts⟩

end NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthReverse
