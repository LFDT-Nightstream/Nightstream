import SuperNeo.FPrimeRecursiveVerifier.Cost

/-!
Semantic obligations for one recursive `F'` step.

The target follows HyperNova Construction 2 at the verifier boundary: the
application transition, recursive public link, complete SuperNeo reduction,
accumulator continuity, state advance, and public output must all hold. Legacy
hashes and serialization sidecars are modeled separately so they can be added
or removed without silently changing the target relation.
-/

namespace SuperNeo.FPrimeRecursiveVerifier

universe u

/-- Independently selectable verifier obligations. -/
inductive FPrimeCheck where
  | verifierContext
  | canonicalEncoding
  | applicationTransition
  | recursivePublicLink
  | piCCS
  | piRLC
  | decRecomposition
  | parentTranscript
  | incomingAccumulator
  | outgoingAccumulator
  | stateTransition
  | outputBinding
  | decChildrenTranscriptHash
  | duplicateAccumulatorHash
  | sidecarConsistency
deriving Repr, DecidableEq

/--
Concrete predicates supplied by an implementation model for one recursive
step. The first twelve fields are the paper/Construction-2 obligations. The
last three are implementation checks whose redundancy must be proved before
they are removed from a complete legacy plan.
-/
structure FPrimePredicates (Step : Type u) where
  verifierContext : Step → Prop
  canonicalEncoding : Step → Prop
  applicationTransition : Step → Prop
  recursivePublicLink : Step → Prop
  piCCS : Step → Prop
  piRLC : Step → Prop
  decRecomposition : Step → Prop
  parentTranscript : Step → Prop
  incomingAccumulator : Step → Prop
  outgoingAccumulator : Step → Prop
  stateTransition : Step → Prop
  outputBinding : Step → Prop
  decChildrenTranscriptHash : Step → Prop
  duplicateAccumulatorHash : Step → Prop
  sidecarConsistency : Step → Prop

/-- Interpret a selectable check using a concrete step model. -/
def checkSemantics
    {Step : Type u}
    (predicates : FPrimePredicates Step) :
    FPrimeCheck → Step → Prop
  | .verifierContext => predicates.verifierContext
  | .canonicalEncoding => predicates.canonicalEncoding
  | .applicationTransition => predicates.applicationTransition
  | .recursivePublicLink => predicates.recursivePublicLink
  | .piCCS => predicates.piCCS
  | .piRLC => predicates.piRLC
  | .decRecomposition => predicates.decRecomposition
  | .parentTranscript => predicates.parentTranscript
  | .incomingAccumulator => predicates.incomingAccumulator
  | .outgoingAccumulator => predicates.outgoingAccumulator
  | .stateTransition => predicates.stateTransition
  | .outputBinding => predicates.outputBinding
  | .decChildrenTranscriptHash => predicates.decChildrenTranscriptHash
  | .duplicateAccumulatorHash => predicates.duplicateAccumulatorHash
  | .sidecarConsistency => predicates.sidecarConsistency

/-- The exact semantic target for one recursive augmented-function step. -/
def PaperRecursiveStep
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (step : Step) : Prop :=
  predicates.verifierContext step ∧
    predicates.canonicalEncoding step ∧
    predicates.applicationTransition step ∧
    predicates.recursivePublicLink step ∧
    predicates.piCCS step ∧
    predicates.piRLC step ∧
    predicates.decRecomposition step ∧
    predicates.parentTranscript step ∧
    predicates.incomingAccumulator step ∧
    predicates.outgoingAccumulator step ∧
    predicates.stateTransition step ∧
    predicates.outputBinding step

/-- All and only the semantic obligations of `PaperRecursiveStep`. -/
def essentialChecks : Finset FPrimeCheck :=
  { .verifierContext
  , .canonicalEncoding
  , .applicationTransition
  , .recursivePublicLink
  , .piCCS
  , .piRLC
  , .decRecomposition
  , .parentTranscript
  , .incomingAccumulator
  , .outgoingAccumulator
  , .stateTransition
  , .outputBinding }

/-- The previous over-specified language, including three derived side checks. -/
def legacyChecks : Finset FPrimeCheck :=
  insert .decChildrenTranscriptHash
    (insert .duplicateAccumulatorHash
      (insert .sidecarConsistency essentialChecks))

theorem essential_accepts_iff_paper
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (step : Step) :
    Accepts (checkSemantics predicates) essentialChecks step ↔
      PaperRecursiveStep predicates step := by
  simp [Accepts, essentialChecks, checkSemantics, PaperRecursiveStep]

/-- The canonical semantic plan is sound and complete by construction. -/
def essentialPlan
    {Step : Type u}
    (predicates : FPrimePredicates Step) :
    CertifiedPlan (checkSemantics predicates) (PaperRecursiveStep predicates) where
  checks := essentialChecks
  sound := by
    intro step hAccepts
    exact (essential_accepts_iff_paper predicates step).mp hAccepts
  complete := by
    intro step hPaper
    exact (essential_accepts_iff_paper predicates step).mpr hPaper

/--
Proofs that legacy checks are derived on every valid paper transition. This is
the only authority that permits those checks to be erased while preserving
completeness as well as soundness.
-/
structure DerivedCheckLaws
    {Step : Type u}
    (predicates : FPrimePredicates Step) where
  decChildrenTranscriptHash :
    ∀ step, PaperRecursiveStep predicates step →
      predicates.decChildrenTranscriptHash step
  duplicateAccumulatorHash :
    ∀ step, PaperRecursiveStep predicates step →
      predicates.duplicateAccumulatorHash step
  sidecarConsistency :
    ∀ step, PaperRecursiveStep predicates step →
      predicates.sidecarConsistency step

/-- The legacy plan is exact only after all three derived-check laws are proved. -/
def legacyPlan
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates) :
    CertifiedPlan (checkSemantics predicates) (PaperRecursiveStep predicates) where
  checks := legacyChecks
  sound := by
    intro step hLegacy
    apply (essential_accepts_iff_paper predicates step).mp
    exact accepts_of_superset (by
      intro check hCheck
      simp [legacyChecks, hCheck]) hLegacy
  complete := by
    intro step hPaper check hCheck
    have hEssential := (essential_accepts_iff_paper predicates step).mpr hPaper
    rcases Finset.mem_insert.mp hCheck with hChild | hCheck
    · subst check
      exact laws.decChildrenTranscriptHash step hPaper
    rcases Finset.mem_insert.mp hCheck with hDuplicate | hCheck
    · subst check
      exact laws.duplicateAccumulatorHash step hPaper
    rcases Finset.mem_insert.mp hCheck with hSidecar | hEssentialMember
    · subst check
      exact laws.sidecarConsistency step hPaper
    · exact hEssential check hEssentialMember

/-- Any derived check is redundant when all essential checks survive its removal. -/
theorem derivedCheck_redundant
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (checks : Finset FPrimeCheck)
    (check : FPrimeCheck)
    (hEssential : essentialChecks ⊆ checks.erase check)
    (hDerived :
      ∀ step, PaperRecursiveStep predicates step →
        checkSemantics predicates check step) :
    Redundant (checkSemantics predicates) checks check := by
  intro step hWithout
  apply hDerived step
  apply (essential_accepts_iff_paper predicates step).mp
  exact accepts_of_superset hEssential hWithout

/-- First legacy hash sidecar can be removed. -/
theorem decChildrenTranscriptHash_redundant
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates) :
    Redundant
      (checkSemantics predicates)
      legacyChecks
      .decChildrenTranscriptHash := by
  apply derivedCheck_redundant predicates legacyChecks
  · decide
  · exact laws.decChildrenTranscriptHash

/-- Duplicate accumulator hashing remains redundant after the first removal. -/
theorem duplicateAccumulatorHash_redundant_after_child
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates) :
    Redundant
      (checkSemantics predicates)
      (legacyChecks.erase .decChildrenTranscriptHash)
      .duplicateAccumulatorHash := by
  apply derivedCheck_redundant predicates
    (legacyChecks.erase .decChildrenTranscriptHash)
  · decide
  · exact laws.duplicateAccumulatorHash

/-- The final serialization sidecar remains redundant after both hash removals. -/
theorem sidecarConsistency_redundant_after_hashes
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates) :
    Redundant
      (checkSemantics predicates)
      ((legacyChecks.erase .decChildrenTranscriptHash).erase
        .duplicateAccumulatorHash)
      .sidecarConsistency := by
  apply derivedCheck_redundant predicates
    ((legacyChecks.erase .decChildrenTranscriptHash).erase
      .duplicateAccumulatorHash)
  · decide
  · exact laws.sidecarConsistency

theorem erase_all_derived_checks_eq_essential :
    (((legacyChecks.erase .decChildrenTranscriptHash).erase
      .duplicateAccumulatorHash).erase .sidecarConsistency) =
      essentialChecks := by
  decide

/-- Mechanical semantic pruning of all three proved-derived legacy checks. -/
def prunedLegacyPlan
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates) :
    CertifiedPlan
      (checkSemantics predicates)
      (PaperRecursiveStep predicates) :=
  let noChild :=
    (legacyPlan predicates laws).eraseRedundant
      .decChildrenTranscriptHash
      (decChildrenTranscriptHash_redundant predicates laws)
  let noDuplicate :=
    noChild.eraseRedundant
      .duplicateAccumulatorHash
      (duplicateAccumulatorHash_redundant_after_child predicates laws)
  noDuplicate.eraseRedundant
    .sidecarConsistency
    (sidecarConsistency_redundant_after_hashes predicates laws)

theorem prunedLegacyPlan_checks
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates) :
    (prunedLegacyPlan predicates laws).checks = essentialChecks := by
  exact erase_all_derived_checks_eq_essential

/--
Counterexample family required to claim inclusion-minimality. Each retained
check must have an invalid step accepted after exactly that check is removed.
-/
structure EssentialNecessityWitnesses
    {Step : Type u}
    (predicates : FPrimePredicates Step) where
  witness : FPrimeCheck → Step
  acceptsWithout :
    ∀ check, check ∈ essentialChecks →
      Accepts (checkSemantics predicates) (essentialChecks.erase check) (witness check)
  violatesPaper :
    ∀ check, check ∈ essentialChecks →
      ¬ PaperRecursiveStep predicates (witness check)

theorem essentialPlan_inclusionMinimalSound
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (witnesses : EssentialNecessityWitnesses predicates) :
    InclusionMinimalSound
      (checkSemantics predicates)
      (PaperRecursiveStep predicates)
      essentialChecks := by
  apply inclusionMinimalSound_of_witnesses (essentialPlan predicates).sound
  intro check hCheck
  exact ⟨witnesses.witness check,
    witnesses.acceptsWithout check hCheck,
    witnesses.violatesPaper check hCheck⟩

end SuperNeo.FPrimeRecursiveVerifier
