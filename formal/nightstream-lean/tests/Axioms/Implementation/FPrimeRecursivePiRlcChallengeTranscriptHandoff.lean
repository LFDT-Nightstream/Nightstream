import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Transcript.Handoff
import tests.Axioms.Support

/-! Fail-closed dependencies for the active PiRLC transcript handoff. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Replay.inputsBound_of_boundary_equalities' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Replay.inputsBound_of_boundary_equalities

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Replay.accepted_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Replay.accepted_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Handoff.fieldCandidate_eq_transcriptCandidate' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Handoff.fieldCandidate_eq_transcriptCandidate

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Handoff.fieldDerivedChallenges_eq_transcriptDerivedChallenges' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Handoff.fieldDerivedChallenges_eq_transcriptDerivedChallenges

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Handoff.embeddedRows_bind_transcriptDerivedChallenges' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Handoff.embeddedRows_bind_transcriptDerivedChallenges
