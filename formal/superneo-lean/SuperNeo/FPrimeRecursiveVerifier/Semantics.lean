import SuperNeo.FPrimeRecursiveVerifier.Cost

/-!
Owns: a generic recursive `F'` check vocabulary, candidate essential and
legacy plans, and pruning obligations relative to caller-supplied predicates.

Does not own: the independent SuperNeo/HyperNova transition semantics, concrete
SuperNeo predicates, Poseidon2 schedules, R1CS rows, or Rust conformance.

Emits constraints: no.

Authority boundary: this file supplies no protocol authority. `PaperRecursiveStep`
is a named conjunction over predicates supplied by a later instantiation. A
separate theorem must prove those predicates equivalent to the independent
paper-level transition before any plan or row removal is protocol-authoritative.
Legacy digests and sidecars are removable only through explicit derived-check
laws at that concrete instantiation.

| Obligation | Lean owner | Guarantee |
|---|---|---|
| Candidate checklist | `PaperRecursiveStep` | Conjoins caller-supplied recursive-step predicates |
| Definitional plan exactness | `essential_accepts_iff_paper` | Selected checks equal that same conjunction; does not prove paper semantics |
| Derived checks | `DerivedCheckLaws`, `prunedLegacyPlan` | Removes only proved redundant legacy checks |
| Minimality hook | `EssentialNecessityWitnesses` | Requires one counterexample per essential check removal |
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
Predicates supplied by an implementation model for one recursive step. The
first twelve fields are candidate categories intended to refine the paper and
Construction-2 obligations; this structure does not prove that refinement.
The last three are implementation checks whose redundancy must be proved
before they are removed from a complete legacy plan.
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

/--
Candidate obligation checklist for one recursive augmented-function step.

This definition is not an independent paper semantics: every proposition is
supplied through `FPrimePredicates`. Protocol authority therefore requires a
separate instantiation theorem relating these fields to the paper-level
PiCCS/PiRLC/PiDEC composition and HyperNova F' transition.
-/
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

/--
The candidate plan is definitionally exact for its supplied checklist. This is
planning infrastructure, not a proof that the checklist is a sound SuperNeo
or HyperNova recursive verifier.
-/
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
Proofs that legacy checks are derived from every accepted concrete checklist
instantiation. This permits checklist-relative pruning; protocol-authoritative
pruning additionally requires the independent paper-semantics refinement
described above.
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
