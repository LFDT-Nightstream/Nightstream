import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.HonestProver
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the complete exact `Pi_CCS` honest prover.

| Stage path | Guarded obligation | Emits constraints? |
|---|---|---|
| `nifs.pi_ccs.exact.prover` | paper obligations produce an accepted exact, source-bound transcript | no |
-/

/--
info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.HonestProver.complete_of_paperObligations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.HonestProver.complete_of_paperObligations
