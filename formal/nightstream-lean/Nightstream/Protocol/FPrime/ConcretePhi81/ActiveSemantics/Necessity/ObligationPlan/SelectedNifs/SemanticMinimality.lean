import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs.CarriedEvaluations
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs.PaperRelation
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.Running
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.Tail

/-!
Proof-bearing necessity ledger for the selected-NIFS semantic plan.

Assurance tier: model-level.

Owns: the exhaustive, disjoint classification of every exact semantic leaf,
the now-empty open list, and a proof record matching the proved list.

Does not own: executable transcript/SumCheck refinement, Rust/R1CS
correspondence, physical rows, costs, security reductions, global gate
minimality, or row removal.

Emits constraints: no.

Authority boundary: every proved entry refers to a kernel-checked removal
counterexample against the independent target. The explicit open list remains
fail-closed for future plan changes; historical circuit behavior and profiler
totals cannot close it.

| Phase | Child owner | Closed leaves | Open leaves |
|---|---|---|---|
| baseline | `SemanticMinimality.Baseline` | accepted independent fixture | none |
| `Pi_CCS` | `SemanticMinimality.PiCcs`, `PiCcs.PaperRelation`, `PiCcs.CarriedEvaluations` | fresh CCS, all-source norm, carried evaluations, polynomial input, source product | none |
| incoming | `SemanticMinimality.Running` | incoming authority | none |
| `Pi_RLC` / `Pi_DEC` | `SemanticMinimality.Tail` | challenge set, exact parent, exact children | none |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

/-- Leaves with a closed removal witness in the child-owner modules. -/
def provedNecessaryLeaves : List SemanticFold.ObligationPlan.Leaf :=
  [.freshCcs, .allSourceNorm, .carriedEvaluations, .polynomialInput,
    .sourceProduct, .incomingAuthority, .challengeStrongSet, .parentExact,
    .childrenExact]

/-- No leaf in the current exact plan remains without a removal witness. -/
def openNecessityLeaves : List SemanticFold.ObligationPlan.Leaf :=
  []

/-- The proved ledger is exactly the semantic plan, in the same stable review
order. -/
theorem provedNecessaryLeaves_eq_checks :
    provedNecessaryLeaves = SemanticFold.ObligationPlan.checks := by
  rfl

/-- The open ledger is mechanically empty. -/
theorem openNecessityLeaves_eq_nil : openNecessityLeaves = [] := by
  rfl

/-- Every exact NIFS semantic leaf is present in one side of the current
necessity ledger. -/
theorem necessity_classified (leaf : SemanticFold.ObligationPlan.Leaf) :
    leaf ∈ provedNecessaryLeaves ∨ leaf ∈ openNecessityLeaves := by
  cases leaf <;> simp [provedNecessaryLeaves, openNecessityLeaves]

/-- No semantic leaf is simultaneously reported as proved and open. -/
theorem necessity_classification_disjoint
    (leaf : SemanticFold.ObligationPlan.Leaf) :
    ¬(leaf ∈ provedNecessaryLeaves ∧ leaf ∈ openNecessityLeaves) := by
  cases leaf <;> simp [provedNecessaryLeaves, openNecessityLeaves]

/-- Proof-bearing closure record for exactly the leaves listed as proved. -/
structure NecessityWitnesses : Prop where
  freshCcs :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .freshCcs
  allSourceNorm :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .allSourceNorm
  carriedEvaluations :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .carriedEvaluations
  polynomialInput :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .polynomialInput
  sourceProduct :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .sourceProduct
  incomingAuthority :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .incomingAuthority
  challengeStrongSet :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .challengeStrongSet
  parentExact :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .parentExact
  childrenExact :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .childrenExact

/-- Closed proof record matching `provedNecessaryLeaves`. -/
theorem necessityWitnesses : NecessityWitnesses := {
  freshCcs := freshCcs_necessary
  allSourceNorm := allSourceNorm_necessary
  carriedEvaluations := carriedEvaluations_necessary
  polynomialInput := polynomialInput_necessary
  sourceProduct := sourceProduct_necessary
  incomingAuthority := incomingAuthority_necessary
  challengeStrongSet := challengeStrongSet_necessary
  parentExact := parentExact_necessary
  childrenExact := childrenExact_necessary
}

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality
