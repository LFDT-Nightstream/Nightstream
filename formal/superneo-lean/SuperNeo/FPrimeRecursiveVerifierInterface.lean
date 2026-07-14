import SuperNeo.FPrimeRecursiveVerifier

/-!
Contract interface for `SuperNeo.FPrimeRecursiveVerifier`.

Spec: `specs/FPrimeRecursiveVerifier.spec.md`
-/

namespace SuperNeo.FPrimeRecursiveVerifierInterface

universe u v w x

/-! ## Semantic Plan Surface -/

abbrev FPrimeCheck := FPrimeRecursiveVerifier.FPrimeCheck
abbrev FPrimePredicates (Step : Type u) :=
  FPrimeRecursiveVerifier.FPrimePredicates Step
abbrev Accepts
    {Input : Type u} {Check : Type v}
    (semantics : Check → Input → Prop)
    (checks : Finset Check)
    (input : Input) : Prop :=
  FPrimeRecursiveVerifier.Accepts semantics checks input
abbrev InclusionMinimalSound
    {Input : Type u} {Check : Type v} [DecidableEq Check]
    (semantics : Check → Input → Prop)
    (target : Input → Prop)
    (checks : Finset Check) : Prop :=
  FPrimeRecursiveVerifier.InclusionMinimalSound semantics target checks
abbrev PaperRecursiveStep
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (step : Step) : Prop :=
  FPrimeRecursiveVerifier.PaperRecursiveStep predicates step
abbrev essentialChecks := FPrimeRecursiveVerifier.essentialChecks
abbrev legacyChecks := FPrimeRecursiveVerifier.legacyChecks
abbrev DerivedCheckLaws
    {Step : Type u}
    (predicates : FPrimePredicates Step) :=
  FPrimeRecursiveVerifier.DerivedCheckLaws predicates

/-- [Role: Theorem-Target] Essential checks accept exactly the fixed target. -/
theorem essential_accepts_iff_paper
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (step : Step) :
    Accepts
      (FPrimeRecursiveVerifier.checkSemantics predicates)
      essentialChecks step ↔
      PaperRecursiveStep predicates step :=
  FPrimeRecursiveVerifier.essential_accepts_iff_paper predicates step

/-- [Role: Theorem-Target] Pruning all three derived sidecars leaves the essential plan. -/
theorem prunedLegacyPlan_checks
    {Step : Type u}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates) :
    (FPrimeRecursiveVerifier.prunedLegacyPlan predicates laws).checks =
      essentialChecks :=
  FPrimeRecursiveVerifier.prunedLegacyPlan_checks predicates laws

/-- [Role: Regression] Essential coordinates are independently necessary in the Boolean model. -/
theorem booleanEssentialPlan_inclusionMinimalSound :
    InclusionMinimalSound
      (FPrimeRecursiveVerifier.checkSemantics
        FPrimeRecursiveVerifier.booleanPredicates)
      (PaperRecursiveStep FPrimeRecursiveVerifier.booleanPredicates)
      essentialChecks :=
  FPrimeRecursiveVerifier.booleanEssentialPlan_inclusionMinimalSound

/-! ## Parent Authority Surface -/

abbrev AuthorityCoreStep
    (Parent : Type u) (Children : Type v)
    (Digest : Type w) (Challenge : Type x) :=
  FPrimeRecursiveVerifier.AuthorityCoreStep
    Parent Children Digest Challenge
abbrev ParentAuthorityModel
    (Parent : Type u) (Children : Type v)
    (Digest : Type w) (Challenge : Type x) :=
  FPrimeRecursiveVerifier.ParentAuthorityModel
    Parent Children Digest Challenge
abbrev AuthorityCoreAccepts
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    (model : ParentAuthorityModel Parent Children Digest Challenge)
    (step : AuthorityCoreStep Parent Children Digest Challenge) : Prop :=
  FPrimeRecursiveVerifier.AuthorityCoreAccepts model step
abbrev AuthorityLegacyAccepts
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    (model : ParentAuthorityModel Parent Children Digest Challenge)
    (step : FPrimeRecursiveVerifier.AuthorityLegacyStep
      Parent Children Digest Challenge) : Prop :=
  FPrimeRecursiveVerifier.AuthorityLegacyAccepts model step

/-- [Role: Theorem-Target] A canonical child-digest sidecar changes no accepted core execution. -/
theorem canonical_legacy_accepts_iff_core
    {Parent Children Digest Challenge : Type}
    (model : ParentAuthorityModel Parent Children Digest Challenge)
    (step : AuthorityCoreStep Parent Children Digest Challenge) :
    AuthorityLegacyAccepts model
      (FPrimeRecursiveVerifier.canonicalLegacyStep model step) ↔
      AuthorityCoreAccepts model step :=
  FPrimeRecursiveVerifier.canonical_legacy_accepts_iff_core model step

abbrev PostRlcCheck := FPrimeRecursiveVerifier.PostRlcCheck
abbrev PostRlcTarget
    (ctx : SuperNeo.ProtocolTargetContext) : Prop :=
  FPrimeRecursiveVerifier.PostRlcTarget ctx
abbrev minimalPostRlcChecks :=
  FPrimeRecursiveVerifier.minimalPostRlcChecks

/-- [Role: Theorem-Target] The RLC parent alone implies the theorem-level DEC statement. -/
theorem minimalPostRlc_accepts_iff_target
    (ctx : SuperNeo.ProtocolTargetContext) :
    Accepts
      FPrimeRecursiveVerifier.postRlcCheckSemantics
      minimalPostRlcChecks ctx ↔
      PostRlcTarget ctx :=
  FPrimeRecursiveVerifier.minimalPostRlc_accepts_iff_target ctx

/-- [Role: Theorem-Target] Deriving DEC does not remove checked-child recomposition. -/
theorem postRlcCheckedMinimal_iff_target
    {Children : Type u}
    (childrenValidate :
      SuperNeo.ProtocolTargetContext → Children → Prop)
    (step : FPrimeRecursiveVerifier.PostRlcStep Children) :
    FPrimeRecursiveVerifier.PostRlcCheckedMinimal childrenValidate step ↔
      FPrimeRecursiveVerifier.PostRlcCheckedTarget childrenValidate step :=
  FPrimeRecursiveVerifier.postRlcCheckedMinimal_iff_target
    childrenValidate step

/-! ## R1CS Refinement Surface -/

abbrev LinearCombination (R : Type u) :=
  FPrimeRecursiveVerifier.LinearCombination R
abbrev R1csConstraint (R : Type u) :=
  FPrimeRecursiveVerifier.R1csConstraint R
abbrev R1csBlock (R : Type u) :=
  FPrimeRecursiveVerifier.R1csBlock R
abbrev R1csCost := FPrimeRecursiveVerifier.R1csCost
abbrev ModularR1csEncoding
    (R : Type u) (Input : Type v)
    (Check : Type w) (Witness : Type x) :=
  FPrimeRecursiveVerifier.ModularR1csEncoding
    R Input Check Witness
abbrev BlockRefinement
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (semantics : Check → Input → Prop) :=
  FPrimeRecursiveVerifier.BlockRefinement encoding semantics
abbrev PlanWitnessComplete
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (semantics : Check → Input → Prop)
    (checks : Finset Check) : Prop :=
  FPrimeRecursiveVerifier.PlanWitnessComplete
    encoding semantics checks
abbrev CertifiedR1csPlan
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} [DecidableEq Check]
    {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (semantics : Check → Input → Prop)
    (target : Input → Prop) :=
  FPrimeRecursiveVerifier.CertifiedR1csPlan
    encoding semantics target

/-- [Role: Construction] Package an exact essential-check R1CS candidate. -/
abbrev certifyEssentialR1cs
    {R : Type u} [Semiring R]
    {Step : Type v} {Witness : Type v}
    (predicates : FPrimePredicates Step)
    (encoding : ModularR1csEncoding R Step FPrimeCheck Witness)
    (refinement :
      BlockRefinement encoding
        (FPrimeRecursiveVerifier.checkSemantics predicates))
    (compilerComplete :
      PlanWitnessComplete encoding
        (FPrimeRecursiveVerifier.checkSemantics predicates)
        essentialChecks) :=
  FPrimeRecursiveVerifier.certifyEssentialR1cs
    predicates encoding refinement compilerComplete

end SuperNeo.FPrimeRecursiveVerifierInterface
