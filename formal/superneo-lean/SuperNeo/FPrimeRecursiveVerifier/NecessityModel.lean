import SuperNeo.FPrimeRecursiveVerifier.Semantics

/-!
Owns: a Boolean independence model and inclusion-minimality witness for the
essential-check vocabulary.

Does not own: concrete protocol attacks, Rust counterexamples, or independence
of checks in a production implementation.

Emits constraints: no.

Authority boundary: each Boolean coordinate is a model witness only; production
row removal still requires concrete refinement and protocol-specific necessity.

| Obligation | Lean owner | Guarantee |
|---|---|---|
| Independent checks | `booleanPredicates` | Gives each check a separate Boolean coordinate |
| Removal witness | `allBut` | Falsifies exactly one selected coordinate |
| Inclusion minimality | `booleanEssentialPlan_inclusionMinimalSound` | Proves the generic essential plan is model-minimal |
-/

namespace SuperNeo.FPrimeRecursiveVerifier

/-- A step whose check outcomes can be varied independently. -/
abbrev BooleanStep := FPrimeCheck → Bool

/-- Interpret each semantic predicate as its corresponding Boolean coordinate. -/
def booleanPredicates : FPrimePredicates BooleanStep where
  verifierContext := fun step => step .verifierContext = true
  canonicalEncoding := fun step => step .canonicalEncoding = true
  applicationTransition := fun step => step .applicationTransition = true
  recursivePublicLink := fun step => step .recursivePublicLink = true
  piCCS := fun step => step .piCCS = true
  piRLC := fun step => step .piRLC = true
  decRecomposition := fun step => step .decRecomposition = true
  parentTranscript := fun step => step .parentTranscript = true
  incomingAccumulator := fun step => step .incomingAccumulator = true
  outgoingAccumulator := fun step => step .outgoingAccumulator = true
  stateTransition := fun step => step .stateTransition = true
  outputBinding := fun step => step .outputBinding = true
  decChildrenTranscriptHash :=
    fun step => step .decChildrenTranscriptHash = true
  duplicateAccumulatorHash :=
    fun step => step .duplicateAccumulatorHash = true
  sidecarConsistency := fun step => step .sidecarConsistency = true

@[simp] theorem checkSemantics_boolean
    (check : FPrimeCheck)
    (step : BooleanStep) :
    checkSemantics booleanPredicates check step ↔ step check = true := by
  cases check <;> rfl

/-- A removal attack: every coordinate is true except the omitted check. -/
def allBut (omitted : FPrimeCheck) : BooleanStep :=
  fun check => decide (check ≠ omitted)

theorem allBut_accepts_without
    (omitted : FPrimeCheck)
    (_hEssential : omitted ∈ essentialChecks) :
    Accepts
      (checkSemantics booleanPredicates)
      (essentialChecks.erase omitted)
      (allBut omitted) := by
  intro check hCheck
  rw [checkSemantics_boolean]
  have hNe : check ≠ omitted := (Finset.mem_erase.mp hCheck).1
  simp [allBut, hNe]

theorem allBut_violates_paper
    (omitted : FPrimeCheck)
    (hEssential : omitted ∈ essentialChecks) :
    ¬ PaperRecursiveStep booleanPredicates (allBut omitted) := by
  intro hPaper
  have hAccepts :
      Accepts
        (checkSemantics booleanPredicates)
        essentialChecks
        (allBut omitted) :=
    (essential_accepts_iff_paper
      booleanPredicates (allBut omitted)).mpr hPaper
  have hOmitted := hAccepts omitted hEssential
  rw [checkSemantics_boolean] at hOmitted
  simp [allBut] at hOmitted

/-- Concrete witnesses showing that no essential coordinate is erased by default. -/
def booleanNecessityWitnesses :
    EssentialNecessityWitnesses booleanPredicates where
  witness := allBut
  acceptsWithout := allBut_accepts_without
  violatesPaper := allBut_violates_paper

/-- Regression theorem: the complete essential vocabulary is inclusion-minimal. -/
theorem booleanEssentialPlan_inclusionMinimalSound :
    InclusionMinimalSound
      (checkSemantics booleanPredicates)
      (PaperRecursiveStep booleanPredicates)
      essentialChecks :=
  essentialPlan_inclusionMinimalSound
    booleanPredicates booleanNecessityWitnesses

end SuperNeo.FPrimeRecursiveVerifier
