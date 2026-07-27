import tests.FPrimeConcreteProductionNifsBridge
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the M3 production relation and concrete
paper-exact NIFS/F-prime bridge.
-/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ProductionRelation.relationShape_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ProductionRelation.relationShape_eq

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ProductionRelation.exactProfile' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ProductionRelation.exactProfile

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.CertificateRefinement.paperOutputEquations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.CertificateRefinement.paperOutputEquations

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary.SourceAuthority.ofCanonicalOpening' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary.SourceAuthority.ofCanonicalOpening

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.physicalChecks_refineConstruction2_of_paperTransition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.physicalChecks_refineConstruction2_of_paperTransition

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary.run_refinesConstruction2_or_namedFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary.run_refinesConstruction2_or_namedFailure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary.exists_run_and_construction2_or_samplerShortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary.exists_run_and_construction2_or_samplerShortfall
