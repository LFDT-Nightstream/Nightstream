import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity

/-!
Accepted independent baseline for selected-NIFS semantic minimality.

Assurance tier: model-level.

Owns: the exact fixed-active 270-coordinate candidate profile and one direct
accepted realization of the nine-leaf NIFS semantic plan.

Does not own: removal witnesses, physical certificates, Rust/R1CS refinement,
rows, costs, security reductions, or row removal.

Emits constraints: no.

Authority boundary: the realization is built from independent source
semantics, checked incoming-parent authority, and an explicit centered-zero
bounded sampler. No implementation acceptance bit or historical circuit is
used as semantic authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.active.nifs.semantic.baseline.witness` | choose one typed point and challenge vector | direct model data | `baselineWitness` |
| `fprime.active.nifs.semantic.baseline.outputs` | compute the canonical parent and children | computed | `baselineCandidate` |
| `fprime.active.nifs.semantic.baseline.acceptance` | satisfy all nine independent leaves | derived | `baselineTargetHolds`, `baselineAccepted` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline

/-- Exact semantic-plan realization type at the explicit fixed-active model
profile. -/
abbrev BaselineRealization :=
  SemanticFold.ObligationPlan.Necessity.Realization
    (shape := Sources.shape)
    (State := Unit)
    (publicRingColumns := Context.publicRingColumns)
    (verifierRows := Context.verifierRows)
    (publicFits := Context.publicFits)
    (arity := FixedActive.arity)

/-- Raw nine-leaf candidate type at the same explicit model profile. -/
abbrev BaselineCandidate :=
  SemanticFold.ObligationPlan.Candidate Sources.shape Unit
    Context.publicRingColumns Context.publicFits Context.verifierRows
    FixedActive.arity

/-- Profile-specialized leaf interpretation used by every closed witness. -/
abbrev baselineSemantics :
    SemanticFold.ObligationPlan.Leaf -> BaselineCandidate -> Prop :=
  SemanticFold.ObligationPlan.semantics

/-- Profile-specialized independent target used by every closed witness. -/
abbrev baselineTarget : BaselineCandidate -> Prop :=
  SemanticFold.ObligationPlan.target

/-- Exact raw point/challenge witness used by the direct semantic baseline. -/
def baselineWitness : SemanticFold.Witness Context.context where
  point := Context.zeroPoint Sources.shape.rowVariables
  challenges := Context.zeroChallenges

/-- Direct raw candidate whose result surfaces are computed from the
independent source data. -/
def baselineCandidate : BaselineCandidate := {
  context := Context.context
  data := Sources.data
  point := baselineWitness.point
  challenges := baselineWitness.challenges
  parent := SemanticFold.parentOf Context.context Sources.data baselineWitness
  children :=
    SemanticFold.childrenOf Context.context Sources.data baselineWitness
}

/-- The direct candidate satisfies the witness-indexed independent target;
no physical certificate or accepted implementation execution is used. -/
theorem baselineTargetHolds : baselineTarget baselineCandidate := by
  exact {
    paper := Sources.paperHolds
    input := Context.semanticInput
    running := Context.runningAccepted
    challengesValid := by
      intro coordinate
      exact (Context.samplerBound ()).challengeValid coordinate
    parent_eq := rfl
    children_eq := rfl
  }

/-- The exact nine-leaf plan accepts the direct semantic baseline. -/
theorem baselineAccepted :
    CheckPlan.Accepts baselineSemantics SemanticFold.ObligationPlan.checks
      baselineCandidate :=
  (SemanticFold.ObligationPlan.accepts_iff_target baselineCandidate).mpr
    baselineTargetHolds

/-- One closed proof-bearing realization used by every removal witness. -/
theorem baseline : Nonempty BaselineRealization :=
  ⟨{
    candidate := baselineCandidate
    accepted := baselineAccepted
  }⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality
