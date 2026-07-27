import Nightstream.Protocol.FPrime.Frozen

/-!
Focused regression for the exact production relation and paper-exact
fixed-one NIFS/F-prime bridge.

Assurance tier: model-level.

The checks below exercise the frozen exports. They make no Rust, R1CS,
generated-row, concrete Fiat--Shamir, or probability-bound claim.
-/

set_option autoImplicit false

namespace tests.FPrimeConcreteProductionNifsBridge

open Nightstream.Protocol.FPrime.Frozen

#check ProductionRelation.relationShape_eq
#check ProductionRelation.publicWidth_eq
#check ProductionRelation.runningAssignment_exact
#check ProductionRelation.piDecPublicInput_roundTrip
#check ProductionRelation.exactProfile

#check ConcreteNifsBridge.check_eq_true_iff_accepted
#check ConcreteNifsBridge.run_eq_some_iff
#check ConcreteNifsBridge.SourceAuthority
#check ConcreteNifsBridge.ofCanonicalOpening
#check ConcreteNifsBridge.paperOutputEquations
#check ConcreteNifsBridge.run_refinesConstruction2_or_namedFailure
#check ConcreteNifsBridge.exists_run_and_construction2_or_samplerShortfall

-- Source authority is a typed input to the bridge, not a generic failure
-- constructor. The bridge retains exactly the three output/algebraic events.
#check Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary.NamedFailure.yRingBinding
#check Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary.NamedFailure.packedYZcolBinding
#check Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary.NamedFailure.piCcsAlgebraic

end tests.FPrimeConcreteProductionNifsBridge
