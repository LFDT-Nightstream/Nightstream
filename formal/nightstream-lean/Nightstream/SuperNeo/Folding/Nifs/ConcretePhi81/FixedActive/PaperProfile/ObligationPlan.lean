import Nightstream.SuperNeo.CheckPlan
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile

/-!
Fixed-candidate obligation plan for the production-shaped SuperNeo NIFS paper
profile.

Protocol: SuperNeo Sections 7.3--7.5.
Phase: one fixed `Pi_CCS -> Pi_RLC -> Pi_DEC` public-verifier realization.
Constraint family: semantic obligations only; this file emits no rows.

Assurance tier: model-level.

Owns: one raw candidate fixing the profile, public source, independent source
data, row point, challenge vector, and public target; six stable macro
obligations; and exact equivalence between their conjunction and the indexed
paper-profile realization.

Does not own: per-obligation necessity, transcript replay, SumCheck security,
child-opening extraction, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: no existential choice remains inside `target`. In
particular, a failed obligation cannot be repaired by substituting different
source data, a different row point, different challenges, or a different
target. `piDecAcceptance` is the operational paper verifier; it does not add
child CE membership as a public check.

| Phase | Leaf | Mathematical obligation | Authority class |
|---|---|---|---|
| `Pi_CCS` | `freshCcs` | every fresh assignment satisfies CCS | checked |
| `Pi_CCS` | `allSourceNorm` | every source coordinate has strict norm `< 2` | checked |
| `Pi_CCS` | `carriedEvaluations` | every running claim is the authoritative prior evaluation | checked |
| `Pi_CCS` | `sourceBinding` | the complete public source product binds the independent source family | checked |
| `Pi_RLC` | `challengeStrongSet` | every fixed challenge belongs to the verifier-owned strong set | checked |
| `Pi_DEC` | `piDecAcceptance` | the fixed target passes exact operational decomposition against the computed parent | checked/computed |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.ObligationPlan

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- Stable paper phase ownership. -/
inductive Phase where
  | piCcs
  | piRlc
  | piDec
  deriving DecidableEq

/-- The six macro obligations retained by the corrected fixed-active paper
profile. -/
inductive Leaf where
  | freshCcs
  | allSourceNorm
  | carriedEvaluations
  | sourceBinding
  | challengeStrongSet
  | piDecAcceptance
  deriving DecidableEq

/-- Semantic authority classification. -/
inductive AuthorityClass where
  | checked
  | checkedComputed
  deriving DecidableEq

def phase : Leaf -> Phase
  | .freshCcs | .allSourceNorm | .carriedEvaluations | .sourceBinding =>
      .piCcs
  | .challengeStrongSet => .piRlc
  | .piDecAcceptance => .piDec

def authority : Leaf -> AuthorityClass
  | .piDecAcceptance => .checkedComputed
  | _ => .checked

/-- Stable review path. A physical family must refine one of these paths. -/
def path : Leaf -> String
  | .freshCcs => "nifs.paper.fixed.pi_ccs.relation.fresh_ccs"
  | .allSourceNorm => "nifs.paper.fixed.pi_ccs.relation.all_source_norm"
  | .carriedEvaluations =>
      "nifs.paper.fixed.pi_ccs.relation.carried_evaluations"
  | .sourceBinding => "nifs.paper.fixed.pi_ccs.input.source_binding"
  | .challengeStrongSet => "nifs.paper.fixed.pi_rlc.challenge.strong_set"
  | .piDecAcceptance => "nifs.paper.fixed.pi_dec.acceptance"

/-- Stable paper order. -/
def checks : List Leaf :=
  [.freshCcs, .allSourceNorm, .carriedEvaluations, .sourceBinding,
    .challengeStrongSet, .piDecAcceptance]

theorem mem_checks (leaf : Leaf) : leaf ∈ checks := by
  cases leaf <;> simp [checks]

/-- A fully fixed raw paper candidate. No validity proposition is hidden in
the carrier. -/
structure Candidate
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  profile : Profile shape publicRingColumns publicFits verifierRows
  source : Source shape publicRingColumns publicFits verifierRows
  data : Data shape
  witness : Witness shape
  target : Target shape publicRingColumns publicFits verifierRows

section

variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Each leaf owns one field of the indexed paper realization. -/
def semantics :
    Leaf ->
      Candidate shape publicRingColumns publicFits verifierRows -> Prop
  | .freshCcs, candidate =>
      Semantics.Paper.FreshCcsHolds candidate.data
  | .allSourceNorm, candidate =>
      Semantics.Paper.AllSourceNormsHold candidate.data
  | .carriedEvaluations, candidate =>
      Semantics.Paper.CarriedEvaluationsHold candidate.data
  | .sourceBinding, candidate =>
      InputBound candidate.profile candidate.source candidate.data
  | .challengeStrongSet, candidate =>
      ChallengesValid candidate.profile candidate.witness
  | .piDecAcceptance, candidate =>
      PiDEC.PaperVerifier.OutputAccepted (decAlgebra candidate.profile)
        (decPublicInputSplit candidate.profile)
        (decEvaluationArity candidate.profile)
        (parentOf candidate.profile candidate.source candidate.data
          candidate.witness)
        candidate.target

/-- The independent target is indexed by every raw candidate field. -/
def target
    (candidate :
      Candidate shape publicRingColumns publicFits verifierRows) : Prop :=
  Realization candidate.profile candidate.source candidate.data
    candidate.target candidate.witness

/-- The six-leaf conjunction is exactly the fixed paper realization. -/
theorem accepts_iff_target
    (candidate :
      Candidate shape publicRingColumns publicFits verifierRows) :
    CheckPlan.Accepts semantics checks candidate ↔ target candidate := by
  constructor
  · intro accepted
    exact {
      paper := ⟨
        accepted .freshCcs (mem_checks .freshCcs),
        accepted .allSourceNorm (mem_checks .allSourceNorm),
        accepted .carriedEvaluations (mem_checks .carriedEvaluations)⟩
      input := accepted .sourceBinding (mem_checks .sourceBinding)
      challengesValid :=
        accepted .challengeStrongSet (mem_checks .challengeStrongSet)
      piDecAccepted :=
        accepted .piDecAcceptance (mem_checks .piDecAcceptance)
    }
  · intro realized leaf _member
    cases leaf with
    | freshCcs => exact realized.paper.1
    | allSourceNorm => exact realized.paper.2.1
    | carriedEvaluations => exact realized.paper.2.2
    | sourceBinding => exact realized.input
    | challengeStrongSet => exact realized.challengesValid
    | piDecAcceptance => exact realized.piDecAccepted

/-- Exactness in the generic inclusion-minimality calculus. -/
theorem exact :
    CheckPlan.Exact
      (semantics (shape := shape)
        (publicRingColumns := publicRingColumns)
        (publicFits := publicFits) (verifierRows := verifierRows))
      target checks := by
  intro candidate
  exact accepts_iff_target candidate

end

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.ObligationPlan
