import SuperNeo.FPrimeRecursiveVerifier.NecessityModel
import SuperNeo.FPrimeRecursiveVerifier.OutputMessageProjection
import SuperNeo.FPrimeRecursiveVerifier.R1csRefinement
import SuperNeo.FPrimeRecursiveVerifier.SuperNeoBridge
import SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding

/-!
Owns: certification and pruning of modular R1CS plans for one recursive `F'`
step.

Does not own: semantic predicates, concrete constraint blocks, or witness
compiler implementations.

Emits constraints: no. It packages semantic and R1CS proof obligations.

Authority boundary: the packaged exactness is relative to the caller-supplied
`FPrimePredicates`. Block refinement and honest-witness completeness connect an
R1CS to that checklist, but do not establish that the checklist is the
independent paper relation. A concrete paper-semantics equivalence theorem is
an additional required gate; block cost never supplies authority.

| Obligation | Lean owner | Guarantee |
|---|---|---|
| Essential plan | `certifyEssentialR1cs` | Packages the exact essential-check lowering |
| Legacy pruning | `pruneLegacyR1cs` | Erases only checks covered by derived-check laws |
| Post-RLC plan | `certifyMinimalPostRlcR1cs` | Packages the parent-authority lowering |
-/

namespace SuperNeo.FPrimeRecursiveVerifier

universe u v

/--
Package an essential-check lowering relative to a supplied predicate checklist.
This does not certify the checklist against independent paper semantics.
-/
def certifyEssentialR1cs
    {R : Type u} [Semiring R]
    {Step : Type v} {Witness : Type v}
    (predicates : FPrimePredicates Step)
    (encoding :
      ModularR1csEncoding R Step FPrimeCheck Witness)
    (refinement :
      BlockRefinement encoding (checkSemantics predicates))
    (compilerComplete :
      PlanWitnessComplete
        encoding (checkSemantics predicates) essentialChecks) :
    CertifiedR1csPlan
      encoding
      (checkSemantics predicates)
      (PaperRecursiveStep predicates) where
  semanticPlan := essentialPlan predicates
  refinement := refinement
  compilerComplete := compilerComplete

/-- The packaged R1CS accepts exactly the supplied recursive-step checklist. -/
theorem certifiedEssentialR1cs_exact
    {R : Type u} [Semiring R]
    {Step : Type v} {Witness : Type v}
    (predicates : FPrimePredicates Step)
    (encoding :
      ModularR1csEncoding R Step FPrimeCheck Witness)
    (refinement :
      BlockRefinement encoding (checkSemantics predicates))
    (compilerComplete :
      PlanWitnessComplete
        encoding (checkSemantics predicates) essentialChecks) :
    R1csExactForTarget
      encoding
      (PaperRecursiveStep predicates)
      essentialChecks :=
  (certifyEssentialR1cs
    predicates encoding refinement compilerComplete).exact

/-- Package the over-specified legacy lowering before certified pruning. -/
def certifyLegacyR1cs
    {R : Type u} [Semiring R]
    {Step : Type v} {Witness : Type v}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates)
    (encoding :
      ModularR1csEncoding R Step FPrimeCheck Witness)
    (refinement :
      BlockRefinement encoding (checkSemantics predicates))
    (compilerComplete :
      PlanWitnessComplete
        encoding (checkSemantics predicates) legacyChecks) :
    CertifiedR1csPlan
      encoding
      (checkSemantics predicates)
      (PaperRecursiveStep predicates) where
  semanticPlan := legacyPlan predicates laws
  refinement := refinement
  compilerComplete := compilerComplete

/--
Delete all three proved-derived legacy blocks. The result reuses the legacy
witness compiler restricted to the remaining essential blocks.
-/
def pruneLegacyR1cs
    {R : Type u} [Semiring R]
    {Step : Type v} {Witness : Type v}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates)
    (encoding :
      ModularR1csEncoding R Step FPrimeCheck Witness)
    (refinement :
      BlockRefinement encoding (checkSemantics predicates))
    (compilerComplete :
      PlanWitnessComplete
        encoding (checkSemantics predicates) legacyChecks) :
    CertifiedR1csPlan
      encoding
      (checkSemantics predicates)
      (PaperRecursiveStep predicates) :=
  let legacy :=
    certifyLegacyR1cs
      predicates laws encoding refinement compilerComplete
  let noChild :=
    legacy.eraseRedundant
      .decChildrenTranscriptHash
      (decChildrenTranscriptHash_redundant predicates laws)
  let noDuplicate :=
    noChild.eraseRedundant
      .duplicateAccumulatorHash
      (duplicateAccumulatorHash_redundant_after_child predicates laws)
  noDuplicate.eraseRedundant
    .sidecarConsistency
    (sidecarConsistency_redundant_after_hashes predicates laws)

theorem pruneLegacyR1cs_checks
    {R : Type u} [Semiring R]
    {Step : Type v} {Witness : Type v}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates)
    (encoding :
      ModularR1csEncoding R Step FPrimeCheck Witness)
    (refinement :
      BlockRefinement encoding (checkSemantics predicates))
    (compilerComplete :
      PlanWitnessComplete
        encoding (checkSemantics predicates) legacyChecks) :
    (pruneLegacyR1cs predicates laws encoding refinement compilerComplete).semanticPlan.checks =
      essentialChecks := by
  exact erase_all_derived_checks_eq_essential

/-- The mechanically pruned legacy R1CS remains exact. -/
theorem prunedLegacyR1cs_exact
    {R : Type u} [Semiring R]
    {Step : Type v} {Witness : Type v}
    (predicates : FPrimePredicates Step)
    (laws : DerivedCheckLaws predicates)
    (encoding :
      ModularR1csEncoding R Step FPrimeCheck Witness)
    (refinement :
      BlockRefinement encoding (checkSemantics predicates))
    (compilerComplete :
      PlanWitnessComplete
        encoding (checkSemantics predicates) legacyChecks) :
    R1csExactForTarget
      encoding
      (PaperRecursiveStep predicates)
      (pruneLegacyR1cs predicates laws encoding refinement compilerComplete).semanticPlan.checks :=
  (pruneLegacyR1cs
    predicates laws encoding refinement compilerComplete).exact

/-- Package a concrete SuperNeo post-RLC lowering with derived DEC authority. -/
def certifyMinimalPostRlcR1cs
    {R : Type u} [Semiring R]
    {Witness : Type v}
    (encoding : ModularR1csEncoding
      R SuperNeo.ProtocolTargetContext PostRlcCheck Witness)
    (refinement :
      BlockRefinement encoding postRlcCheckSemantics)
    (compilerComplete :
      PlanWitnessComplete
        encoding postRlcCheckSemantics minimalPostRlcChecks) :
    CertifiedR1csPlan encoding postRlcCheckSemantics PostRlcTarget where
  semanticPlan := minimalPostRlcPlan
  refinement := refinement
  compilerComplete := compilerComplete

/-- The minimal post-RLC circuit language is exactly the theorem target. -/
theorem certifiedMinimalPostRlcR1cs_exact
    {R : Type u} [Semiring R]
    {Witness : Type v}
    (encoding : ModularR1csEncoding
      R SuperNeo.ProtocolTargetContext PostRlcCheck Witness)
    (refinement :
      BlockRefinement encoding postRlcCheckSemantics)
    (compilerComplete :
      PlanWitnessComplete
        encoding postRlcCheckSemantics minimalPostRlcChecks) :
    R1csExactForTarget
      encoding PostRlcTarget minimalPostRlcChecks :=
  (certifyMinimalPostRlcR1cs
    encoding refinement compilerComplete).exact

end SuperNeo.FPrimeRecursiveVerifier
