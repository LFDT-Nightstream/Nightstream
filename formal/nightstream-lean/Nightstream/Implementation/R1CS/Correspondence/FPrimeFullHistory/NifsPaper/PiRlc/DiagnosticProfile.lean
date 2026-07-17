import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc

/-!
Fixed-carrier dimensions for the legacy generated `Pi_RLC` artifacts.

Assurance tier: artifact-checked. These constants describe the current
three-evaluation-row fixtures; they are not the active selective relation,
whose matrix count is derived independently by `SelectiveCcs.RelationProfile`.

Owns: the shared dimensions used to inspect the legacy recursive and terminal
projection artifacts.

Does not own: protocol semantics, production shape, Rust conformance, or any
compression from the selective relation's thirteen evaluations to three.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.diagnostic.shape` | fixture has three evaluation rows | non-authoritative structure | `matrixCount` |
| `nifs.pi_rlc.diagnostic.public` | fixture has `23 + 2 * 3 = 29` public leaves | derived | `publicLeafCount_eq_29` |
| `nifs.pi_rlc.diagnostic.delayed_nc` | two delayed-NC leaves follow the public leaves | non-authoritative structure | `delayedNcLeafCount` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.DiagnosticProfile

/-- Evaluation-row count hardcoded by the legacy generated fixtures. -/
def matrixCount : Nat := 3

/-- Public commitment, input, and evaluation leaves in the fixture. -/
def publicLeafCount : Nat := 23 + 2 * matrixCount

/-- Delayed-NC leaves stored after the paper-public fixture leaves. -/
def delayedNcLeafCount : Nat := 2

/-- Complete generated trace count for either diagnostic fixture. -/
def traceCount : Nat := publicLeafCount + delayedNcLeafCount

theorem publicLeafCount_eq_29 : publicLeafCount = 29 := by
  decide

theorem traceCount_eq_31 : traceCount = 31 := by
  decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.DiagnosticProfile
