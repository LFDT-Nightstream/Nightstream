import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTraceRefinement
import tests.Axioms.Support

namespace NightstreamTests.Axioms.Implementation.Poseidon2CompactTraceRefinement

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement.trace_refines_compact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms trace_refines_compact

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement.trace_computes_reference' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms trace_computes_reference

end NightstreamTests.Axioms.Implementation.Poseidon2CompactTraceRefinement
