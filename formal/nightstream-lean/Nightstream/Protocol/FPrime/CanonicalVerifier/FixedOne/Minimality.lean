import Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality

/-!
Inclusion-minimal obligation ledger for the canonical fixed-one F' step.

Assurance tier: model-level.

The outer fixed-one executable treats its selected `NIFS.V` call atomically.
`ExecutableOuter` proves the two outer obligations minimal against an actual
one-slot `FixedOne.eval`.  Separately, `SelectedPaper` reuses the audited
six-leaf `FixedActive.PaperProfile.ObligationPlan` minimality result.  The
eight-leaf result below is their branch-conditioned composition ledger; it
does not pretend that an arbitrary opaque `Setup.nifs` exposes those six
internal leaves.

Owns:
- exactness and inclusion-minimality of an executable two-leaf outer model;
- one base-only retained equality, `z0 = zi`;
- one recursive outer retained equality, the prior public link;
- the six retained selected-NIFS paper leaves and their existing kernel
  countermodels;
- branch-conditioned exactness and inclusion-minimal soundness;
- explicit classification of dispatch and raw prior-counter range as derived,
  not retained.

Does not own: production ConcretePhi81 decoding or authority, Rust, R1CS,
physical rows, costs, global arithmetization minimality, or primitive security.
The imported ConcretePhi81-path theorem is used only as model-level evidence
for the selected paper profile; it is not implementation authority.

Emits constraints: no.
-/

namespace Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.SuperNeo

universe uKey uDigest uState uWitness uRunning uFresh uProof uEncoded

namespace SelectedPaper

abbrev Leaf :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.PaperLeaf

abbrev Candidate :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.PaperCandidate

abbrev Semantics : Leaf -> Candidate -> Prop :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.paperSemantics

abbrev Target : Candidate -> Prop :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.paperTarget

abbrev checks : List Leaf :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.paperChecks

theorem mem_checks (leaf : Leaf) :
    leaf ∈ checks :=
  Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.ObligationPlan.mem_checks
    leaf

theorem accepts_iff_target (candidate : Candidate) :
    CheckPlan.Accepts Semantics checks candidate <-> Target candidate :=
  Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.ObligationPlan.accepts_iff_target
    candidate

abbrev baseline : Candidate :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.paperBaselineCandidate

theorem baselineAccepted :
    CheckPlan.Accepts Semantics checks baseline :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.paperBaselineAccepted

theorem freshCcs_necessary :
    CheckPlan.NecessaryForSoundness Semantics Target checks .freshCcs :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.freshCcs_necessary

theorem allSourceNorm_necessary :
    CheckPlan.NecessaryForSoundness Semantics Target checks .allSourceNorm :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.allSourceNorm_necessary

theorem carriedEvaluations_necessary :
    CheckPlan.NecessaryForSoundness Semantics Target checks
      .carriedEvaluations :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.carriedEvaluations_necessary

theorem sourceBinding_necessary :
    CheckPlan.NecessaryForSoundness Semantics Target checks .sourceBinding :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.sourceBinding_necessary

theorem challengeStrongSet_necessary :
    CheckPlan.NecessaryForSoundness Semantics Target checks
      .challengeStrongSet :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.challengeStrongSet_necessary

theorem piDecAcceptance_necessary :
    CheckPlan.NecessaryForSoundness Semantics Target checks
      .piDecAcceptance :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.piDecAcceptance_necessary

theorem inclusionMinimalSound :
    CheckPlan.InclusionMinimalSound Semantics Target checks :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.inclusionMinimalSound

end SelectedPaper

/-! ## Actual fixed-one executable outer obligations -/

namespace ExecutableOuter

/-- The two obligations visible outside the opaque selected NIFS call. -/
inductive Leaf where
  | baseInitialEndpoint
  | recursivePriorPublicLink
deriving DecidableEq

def checks : List Leaf :=
  [.baseInitialEndpoint, .recursivePriorPublicLink]

theorem mem_checks (leaf : Leaf) :
    leaf ∈ checks := by
  cases leaf <;> simp [checks]

theorem checks_length :
    checks.length = 2 :=
  rfl

/-- A concrete one-slot setup whose selected NIFS call always returns the
unique running value.  This isolates the two outer checks without smuggling
another NIFS predicate into the model. -/
def setup : Setup Unit Unit Bool Unit 1 where
  verifierKeys := fun _ => ()
  nifs := {
    verify := fun _ _ _ _ => some ()
  }
  defaultRunning := ()

/-- A concrete outer machine.  Its public link exposes the fresh Boolean,
while its verifier-owned hash is always `false`. -/
def machine : Machine Unit Bool Bool Unit Unit Bool Bool 1 where
  control := fun _ _ => selected
  step := fun _ state _ => state
  freshPublic := fun fresh => fresh
  encodeInstance := fun digest => digest
  hash := fun _ => false

abbrev ModelInput :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input
    Bool Unit Unit Bool Unit

abbrev ModelOutput :=
  Output Bool Bool Unit 1

/-- Construct a complete executable input while exposing only the fields
needed by the two outer removal witnesses. -/
def inputAt (iteration : Nat) (z0 zi fresh : Bool) : ModelInput where
  iteration := iteration
  z0 := z0
  zi := zi
  running := fun _ => ()
  fresh := fresh
  witness := ()
  nifsProof := ()

/-- The output is computed by the same fixed-one helper used by `eval`; it is
not another caller-supplied obligation. -/
def computedOutput (input : ModelInput) : ModelOutput :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
    setup machine input (fun _ => ())

/-- The exact verifier-owned prior-public-link equation. -/
def PriorPublicLink (input : ModelInput) : Prop :=
  machine.freshPublic input.fresh =
    machine.encodeInstance
      (machine.hash
        (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
          setup input))

/-- Branch-conditioned outer predicates.  The selected NIFS call remains an
opaque atomic call here and is fixed to acceptance by `setup`. -/
def semantics : Leaf -> ModelInput -> Prop
  | .baseInitialEndpoint, input =>
      input.iteration = 0 -> input.z0 = input.zi
  | .recursivePriorPublicLink, input =>
      input.iteration ≠ 0 -> PriorPublicLink input

/-- Independent target for the outer plan: actual computed acceptance by the
fixed-one executable on its computed output. -/
def target (input : ModelInput) : Prop :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts
    setup machine input (computedOutput input)

/-- A two-element plan is exactly the conjunction of its named leaves. -/
theorem planAccepts_iff (input : ModelInput) :
    CheckPlan.Accepts semantics checks input <->
      semantics .baseInitialEndpoint input /\
        semantics .recursivePriorPublicLink input := by
  constructor
  · intro accepted
    exact ⟨accepted .baseInitialEndpoint (mem_checks _),
      accepted .recursivePriorPublicLink (mem_checks _)⟩
  · rintro ⟨baseEndpoint, priorLink⟩ leaf _member
    cases leaf with
    | baseInitialEndpoint => exact baseEndpoint
    | recursivePriorPublicLink => exact priorLink

/-- The two-leaf plan agrees extensionally with an actual
`CanonicalVerifier.FixedOne.eval` on this explicit one-slot model. -/
theorem accepts_iff_fixedOne_eval (input : ModelInput) :
    CheckPlan.Accepts semantics checks input <-> target input := by
  rw [planAccepts_iff]
  by_cases iterationZero : input.iteration = 0
  · by_cases endpoint : input.z0 = input.zi
    · simp [semantics, target,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
        computedOutput, iterationZero, endpoint]
    · simp [semantics, target,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
        computedOutput, iterationZero, endpoint]
  · by_cases priorLink : PriorPublicLink input
    · simp [semantics, target,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
        computedOutput, iterationZero, priorLink, PriorPublicLink, setup]
    · simp [semantics, target,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts,
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
        computedOutput, iterationZero, priorLink, PriorPublicLink, setup]

def unequalBase : ModelInput :=
  inputAt 0 false true false

def brokenPriorLink : ModelInput :=
  inputAt 1 false false true

/-- Removing the base endpoint admits an actual fixed-one input rejected only
by its unequal initial endpoint. -/
theorem baseInitialEndpoint_necessary :
    CheckPlan.NecessaryForSoundness semantics target checks
      .baseInitialEndpoint := by
  refine ⟨unequalBase, ?_, ?_⟩
  · intro leaf member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases leaf with
    | baseInitialEndpoint => exact (retained rfl).elim
    | recursivePriorPublicLink =>
        intro notZero
        exact (notZero rfl).elim
  · intro accepted
    have plan := (accepts_iff_fixedOne_eval unequalBase).mpr accepted
    have endpoint :=
      plan .baseInitialEndpoint (mem_checks .baseInitialEndpoint) rfl
    change false = true at endpoint
    exact Bool.false_ne_true endpoint

/-- Removing the prior-public link admits an actual recursive fixed-one input
whose selected atomic NIFS call succeeds but whose public link is false. -/
theorem recursivePriorPublicLink_necessary :
    CheckPlan.NecessaryForSoundness semantics target checks
      .recursivePriorPublicLink := by
  refine ⟨brokenPriorLink, ?_, ?_⟩
  · intro leaf member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases leaf with
    | baseInitialEndpoint =>
        intro iterationZero
        exact (Nat.one_ne_zero iterationZero).elim
    | recursivePriorPublicLink => exact (retained rfl).elim
  · intro accepted
    have plan := (accepts_iff_fixedOne_eval brokenPriorLink).mpr accepted
    have priorLink :=
      plan .recursivePriorPublicLink
        (mem_checks .recursivePriorPublicLink) Nat.one_ne_zero
    change true = false at priorLink
    exact (by decide : true ≠ false) priorLink

theorem retained_necessary
    (leaf : Leaf)
    (_member : leaf ∈ checks) :
    CheckPlan.NecessaryForSoundness semantics target checks leaf := by
  cases leaf with
  | baseInitialEndpoint => exact baseInitialEndpoint_necessary
  | recursivePriorPublicLink =>
      exact recursivePriorPublicLink_necessary

theorem sound :
    CheckPlan.Sound semantics target checks := by
  intro input accepted
  exact (accepts_iff_fixedOne_eval input).mp accepted

/-- Inclusion-minimality of the two checks visible at the actual executable
outer boundary. -/
theorem inclusionMinimalSound :
    CheckPlan.InclusionMinimalSound semantics target checks :=
  CheckPlan.inclusionMinimalSound_of_witnesses sound retained_necessary

end ExecutableOuter

/-- Exact retained leaves of the branch-conditioned composition ledger.
Executable exactness is owned separately by `ExecutableOuter`; the six NIFS
entries name the selected paper-profile checker. -/
inductive Leaf where
  | baseInitialEndpoint
  | recursivePriorPublicLink
  | nifsFreshCcs
  | nifsAllSourceNorm
  | nifsCarriedEvaluations
  | nifsSourceBinding
  | nifsChallengeStrongSet
  | nifsPiDecAcceptance
deriving DecidableEq

inductive Branch where
  | base
  | recursive
deriving DecidableEq

def branch : Leaf -> Branch
  | .baseInitialEndpoint => .base
  | _ => .recursive

def checks : List Leaf :=
  [.baseInitialEndpoint, .recursivePriorPublicLink,
    .nifsFreshCcs, .nifsAllSourceNorm, .nifsCarriedEvaluations,
    .nifsSourceBinding, .nifsChallengeStrongSet, .nifsPiDecAcceptance]

theorem mem_checks (leaf : Leaf) :
    leaf ∈ checks := by
  cases leaf <;> simp [checks]

theorem checks_length :
    checks.length = 8 :=
  rfl

/-- Checks eliminated by the fixed-one carrier rather than retained in the
plan. -/
inductive DerivedCheck where
  | dispatch
  | rawPriorPcRange
deriving DecidableEq

def derivedChecks : List DerivedCheck :=
  [.dispatch, .rawPriorPcRange]

theorem derivedChecks_length :
    derivedChecks.length = 2 :=
  rfl

/-- Minimal base-branch carrier used by the endpoint countermodel. -/
structure BaseCandidate where
  z0 : Bool
  zi : Bool

/-- Recursive branch carrier: the outer public link plus the fixed selected
paper-NIFS candidate. -/
structure RecursiveCandidate where
  priorPublicLink : Bool
  selectedNifs : SelectedPaper.Candidate

/-- Branch-conditioned raw candidate.  Base cases contain no NIFS candidate
and therefore do not execute or constrain selected-NIFS leaves. -/
inductive Candidate where
  | base (candidate : BaseCandidate)
  | recursive (candidate : RecursiveCandidate)

def ofOuterLeaf : ExecutableOuter.Leaf -> Leaf
  | .baseInitialEndpoint => .baseInitialEndpoint
  | .recursivePriorPublicLink => .recursivePriorPublicLink

def ofPaperLeaf : SelectedPaper.Leaf -> Leaf
  | .freshCcs => .nifsFreshCcs
  | .allSourceNorm => .nifsAllSourceNorm
  | .carriedEvaluations => .nifsCarriedEvaluations
  | .sourceBinding => .nifsSourceBinding
  | .challengeStrongSet => .nifsChallengeStrongSet
  | .piDecAcceptance => .nifsPiDecAcceptance

theorem ofPaperLeaf_injective :
    Function.Injective ofPaperLeaf := by
  intro left right equal
  cases left <;> cases right <;> simp_all [ofPaperLeaf]

theorem ofPaperLeaf_mem_checks (leaf : SelectedPaper.Leaf) :
    ofPaperLeaf leaf ∈ checks :=
  mem_checks (ofPaperLeaf leaf)

/-- The aggregate list is literally the executable outer pair followed by the
six selected-paper leaves. -/
theorem checks_eq_composed :
    checks =
      ExecutableOuter.checks.map ofOuterLeaf ++
        SelectedPaper.checks.map ofPaperLeaf :=
  rfl

/-- Each retained predicate is active only on its actual execution branch. -/
def semantics : Leaf -> Candidate -> Prop
  | .baseInitialEndpoint, .base candidate =>
      candidate.z0 = candidate.zi
  | .baseInitialEndpoint, .recursive _ => True
  | .recursivePriorPublicLink, .base _ => True
  | .recursivePriorPublicLink, .recursive candidate =>
      candidate.priorPublicLink = true
  | .nifsFreshCcs, .base _ => True
  | .nifsFreshCcs, .recursive candidate =>
      SelectedPaper.Semantics .freshCcs candidate.selectedNifs
  | .nifsAllSourceNorm, .base _ => True
  | .nifsAllSourceNorm, .recursive candidate =>
      SelectedPaper.Semantics .allSourceNorm candidate.selectedNifs
  | .nifsCarriedEvaluations, .base _ => True
  | .nifsCarriedEvaluations, .recursive candidate =>
      SelectedPaper.Semantics .carriedEvaluations candidate.selectedNifs
  | .nifsSourceBinding, .base _ => True
  | .nifsSourceBinding, .recursive candidate =>
      SelectedPaper.Semantics .sourceBinding candidate.selectedNifs
  | .nifsChallengeStrongSet, .base _ => True
  | .nifsChallengeStrongSet, .recursive candidate =>
      SelectedPaper.Semantics .challengeStrongSet candidate.selectedNifs
  | .nifsPiDecAcceptance, .base _ => True
  | .nifsPiDecAcceptance, .recursive candidate =>
      SelectedPaper.Semantics .piDecAcceptance candidate.selectedNifs

/-- Intended branch target: the base endpoint equality, or the recursive
public link together with the complete chosen paper-NIFS target. -/
def target : Candidate -> Prop
  | .base candidate => candidate.z0 = candidate.zi
  | .recursive candidate =>
      candidate.priorPublicLink = true /\
        SelectedPaper.Target candidate.selectedNifs

/-- The eight-leaf branch-conditioned plan is exactly the stated target. -/
theorem accepts_iff_target (candidate : Candidate) :
    CheckPlan.Accepts semantics checks candidate <-> target candidate := by
  cases candidate with
  | base candidate =>
      constructor
      · intro accepted
        exact accepted .baseInitialEndpoint (mem_checks _)
      · intro endpoint leaf _member
        cases leaf with
        | baseInitialEndpoint => exact endpoint
        | recursivePriorPublicLink => trivial
        | nifsFreshCcs => trivial
        | nifsAllSourceNorm => trivial
        | nifsCarriedEvaluations => trivial
        | nifsSourceBinding => trivial
        | nifsChallengeStrongSet => trivial
        | nifsPiDecAcceptance => trivial
  | recursive candidate =>
      constructor
      · intro accepted
        refine ⟨accepted .recursivePriorPublicLink (mem_checks _), ?_⟩
        apply (SelectedPaper.accepts_iff_target candidate.selectedNifs).mp
        intro leaf _member
        cases leaf with
        | freshCcs =>
            exact accepted .nifsFreshCcs (mem_checks _)
        | allSourceNorm =>
            exact accepted .nifsAllSourceNorm (mem_checks _)
        | carriedEvaluations =>
            exact accepted .nifsCarriedEvaluations (mem_checks _)
        | sourceBinding =>
            exact accepted .nifsSourceBinding (mem_checks _)
        | challengeStrongSet =>
            exact accepted .nifsChallengeStrongSet (mem_checks _)
        | piDecAcceptance =>
            exact accepted .nifsPiDecAcceptance (mem_checks _)
      · rintro ⟨priorLink, selectedTarget⟩
        have selectedAccepted :=
          (SelectedPaper.accepts_iff_target candidate.selectedNifs).mpr
            selectedTarget
        intro leaf _member
        cases leaf with
        | baseInitialEndpoint => trivial
        | recursivePriorPublicLink => exact priorLink
        | nifsFreshCcs =>
            exact selectedAccepted .freshCcs (SelectedPaper.mem_checks _)
        | nifsAllSourceNorm =>
            exact selectedAccepted .allSourceNorm (SelectedPaper.mem_checks _)
        | nifsCarriedEvaluations =>
            exact selectedAccepted .carriedEvaluations
              (SelectedPaper.mem_checks _)
        | nifsSourceBinding =>
            exact selectedAccepted .sourceBinding (SelectedPaper.mem_checks _)
        | nifsChallengeStrongSet =>
            exact selectedAccepted .challengeStrongSet
              (SelectedPaper.mem_checks _)
        | nifsPiDecAcceptance =>
            exact selectedAccepted .piDecAcceptance
              (SelectedPaper.mem_checks _)

def unequalBase : Candidate :=
  .base { z0 := false, zi := true }

/-- Dropping the base endpoint equality admits unequal initial endpoints. -/
theorem baseInitialEndpoint_necessary :
    CheckPlan.NecessaryForSoundness semantics target checks
      .baseInitialEndpoint := by
  refine ⟨unequalBase, ?_, ?_⟩
  · intro leaf member
    cases leaf with
    | baseInitialEndpoint =>
        exact ((CheckPlan.mem_without_iff.mp member).2 rfl).elim
    | recursivePriorPublicLink => trivial
    | nifsFreshCcs => trivial
    | nifsAllSourceNorm => trivial
    | nifsCarriedEvaluations => trivial
    | nifsSourceBinding => trivial
    | nifsChallengeStrongSet => trivial
    | nifsPiDecAcceptance => trivial
  · intro endpoint
    change false = true at endpoint
    exact Bool.false_ne_true endpoint

def missingPriorLink : Candidate :=
  .recursive {
    priorPublicLink := false
    selectedNifs := SelectedPaper.baseline
  }

/-- Dropping the recursive prior-public-link admits an otherwise valid
selected paper fold with a false outer link. -/
theorem recursivePriorPublicLink_necessary :
    CheckPlan.NecessaryForSoundness semantics target checks
      .recursivePriorPublicLink := by
  refine ⟨missingPriorLink, ?_, ?_⟩
  · intro leaf member
    cases leaf with
    | baseInitialEndpoint => trivial
    | recursivePriorPublicLink =>
        exact ((CheckPlan.mem_without_iff.mp member).2 rfl).elim
    | nifsFreshCcs =>
        exact SelectedPaper.baselineAccepted .freshCcs
          (SelectedPaper.mem_checks _)
    | nifsAllSourceNorm =>
        exact SelectedPaper.baselineAccepted .allSourceNorm
          (SelectedPaper.mem_checks _)
    | nifsCarriedEvaluations =>
        exact SelectedPaper.baselineAccepted .carriedEvaluations
          (SelectedPaper.mem_checks _)
    | nifsSourceBinding =>
        exact SelectedPaper.baselineAccepted .sourceBinding
          (SelectedPaper.mem_checks _)
    | nifsChallengeStrongSet =>
        exact SelectedPaper.baselineAccepted .challengeStrongSet
          (SelectedPaper.mem_checks _)
    | nifsPiDecAcceptance =>
        exact SelectedPaper.baselineAccepted .piDecAcceptance
          (SelectedPaper.mem_checks _)
  · rintro ⟨priorLink, _selectedTarget⟩
    change false = true at priorLink
    exact Bool.false_ne_true priorLink

private theorem paperMember_without_of_combinedMember
    (candidate removed : SelectedPaper.Leaf)
    (member :
      ofPaperLeaf candidate ∈
        CheckPlan.without checks (ofPaperLeaf removed)) :
    candidate ∈ CheckPlan.without SelectedPaper.checks removed := by
  apply CheckPlan.mem_without_iff.mpr
  refine ⟨SelectedPaper.mem_checks candidate, ?_⟩
  intro equal
  exact (CheckPlan.mem_without_iff.mp member).2
    (congrArg ofPaperLeaf equal)

/-- Lift a selected-paper removal witness into the recursive fixed-one branch.
The outer prior link is held true, so only the selected leaf is removed. -/
theorem liftSelectedNecessary
    (leaf : SelectedPaper.Leaf)
    (necessary :
      CheckPlan.NecessaryForSoundness SelectedPaper.Semantics
        SelectedPaper.Target SelectedPaper.checks leaf) :
    CheckPlan.NecessaryForSoundness semantics target checks
      (ofPaperLeaf leaf) := by
  rcases necessary with ⟨selectedCandidate, weakened, rejected⟩
  refine ⟨.recursive {
    priorPublicLink := true
    selectedNifs := selectedCandidate
  }, ?_, ?_⟩
  · intro check member
    cases check with
    | baseInitialEndpoint => trivial
    | recursivePriorPublicLink => rfl
    | nifsFreshCcs =>
        exact weakened .freshCcs
          (paperMember_without_of_combinedMember .freshCcs leaf member)
    | nifsAllSourceNorm =>
        exact weakened .allSourceNorm
          (paperMember_without_of_combinedMember .allSourceNorm leaf member)
    | nifsCarriedEvaluations =>
        exact weakened .carriedEvaluations
          (paperMember_without_of_combinedMember .carriedEvaluations leaf
            member)
    | nifsSourceBinding =>
        exact weakened .sourceBinding
          (paperMember_without_of_combinedMember .sourceBinding leaf member)
    | nifsChallengeStrongSet =>
        exact weakened .challengeStrongSet
          (paperMember_without_of_combinedMember .challengeStrongSet leaf
            member)
    | nifsPiDecAcceptance =>
        exact weakened .piDecAcceptance
          (paperMember_without_of_combinedMember .piDecAcceptance leaf member)
  · rintro ⟨_priorLink, selectedTarget⟩
    exact rejected selectedTarget

/-- Every retained leaf has an ordinary kernel-checked removal witness. -/
theorem retained_necessary
    (leaf : Leaf)
    (_member : leaf ∈ checks) :
    CheckPlan.NecessaryForSoundness semantics target checks leaf := by
  cases leaf with
  | baseInitialEndpoint => exact baseInitialEndpoint_necessary
  | recursivePriorPublicLink =>
      exact recursivePriorPublicLink_necessary
  | nifsFreshCcs =>
      exact liftSelectedNecessary .freshCcs
        SelectedPaper.freshCcs_necessary
  | nifsAllSourceNorm =>
      exact liftSelectedNecessary .allSourceNorm
        SelectedPaper.allSourceNorm_necessary
  | nifsCarriedEvaluations =>
      exact liftSelectedNecessary .carriedEvaluations
        SelectedPaper.carriedEvaluations_necessary
  | nifsSourceBinding =>
      exact liftSelectedNecessary .sourceBinding
        SelectedPaper.sourceBinding_necessary
  | nifsChallengeStrongSet =>
      exact liftSelectedNecessary .challengeStrongSet
        SelectedPaper.challengeStrongSet_necessary
  | nifsPiDecAcceptance =>
      exact liftSelectedNecessary .piDecAcceptance
        SelectedPaper.piDecAcceptance_necessary

/-- The full retained plan is sound. -/
theorem sound :
    CheckPlan.Sound semantics target checks := by
  intro candidate accepted
  exact (accepts_iff_target candidate).mp accepted

/-- Inclusion-minimality relative to the fixed-one typed branch carrier and
the chosen six-leaf paper-NIFS primitive.  This is not a row or global gate
lower bound. -/
theorem inclusionMinimalSound :
    CheckPlan.InclusionMinimalSound semantics target checks := by
  exact CheckPlan.inclusionMinimalSound_of_witnesses sound retained_necessary

/-- Dispatch remains absent because `Fin 1` computes it. -/
theorem dispatch_derived
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Encoded : Type uEncoded}
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (state : State)
    (witness : Witness) :
    machine.control state witness = selected :=
  control_eq_selected machine state witness

/-- The raw prior-counter range check remains absent because `toGeneric`
constructs the sole valid counter. -/
theorem rawPriorPcRange_derived
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input :
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input
        State Witness Running Fresh Proof) :
    InRange 1 (input.toGeneric (Key := Key)).priorPc :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric_priorPcValid
    input

structure Obligation8Classification : Prop where
  retainedExact :
    checks =
      [.baseInitialEndpoint, .recursivePriorPublicLink,
        .nifsFreshCcs, .nifsAllSourceNorm, .nifsCarriedEvaluations,
        .nifsSourceBinding, .nifsChallengeStrongSet,
        .nifsPiDecAcceptance]
  derivedExact :
    derivedChecks = [.dispatch, .rawPriorPcRange]
  composedChecksExact :
    checks =
      ExecutableOuter.checks.map ofOuterLeaf ++
        SelectedPaper.checks.map ofPaperLeaf
  executableOuterExact :
    forall input : ExecutableOuter.ModelInput,
      CheckPlan.Accepts ExecutableOuter.semantics
          ExecutableOuter.checks input <->
        ExecutableOuter.target input
  executableOuterMinimal :
    CheckPlan.InclusionMinimalSound ExecutableOuter.semantics
      ExecutableOuter.target ExecutableOuter.checks
  selectedPaperMinimal :
    CheckPlan.InclusionMinimalSound SelectedPaper.Semantics
      SelectedPaper.Target SelectedPaper.checks
  composedLedgerMinimal :
    CheckPlan.InclusionMinimalSound semantics target checks

/-- Final obligation-8 classification.  The outer pair is tied to an actual
fixed-one executable; the internal six are tied to the selected paper
profile; the aggregate eight-leaf theorem records their honest
branch-conditioned composition. -/
theorem obligation8_classification :
    Obligation8Classification := {
  retainedExact := rfl
  derivedExact := rfl
  composedChecksExact := checks_eq_composed
  executableOuterExact := ExecutableOuter.accepts_iff_fixedOne_eval
  executableOuterMinimal := ExecutableOuter.inclusionMinimalSound
  selectedPaperMinimal := SelectedPaper.inclusionMinimalSound
  composedLedgerMinimal := inclusionMinimalSound
}

end Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality
