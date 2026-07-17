import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Fe.Rows
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Fe.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Fe.WireFormat

/-!
Phase-to-family tree for terminal FE SumCheck refinement.

Owns: the parent boundary between the independently typed minimal FE language
and the legacy terminal artifact's physical owner tree.

Does not own: a claim that those two wire formats are equal, FE polynomial
soundness, a repaired Rust/R1CS lowering, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `Rows` and `Schedule` classify the old circuit. The
independent semantic language remains authoritative, and `WireFormat` records
the current non-refinement rather than widening the semantic certificate.

| Stage path | Mathematical obligation | Child owner |
|---|---|---|
| `nifs.pi_ccs.fe_sumcheck.legacy_owner` | exact 41-piece legacy owner address space | `Rows` |
| `nifs.pi_ccs.fe_sumcheck.legacy_tree` | prologue, first round, and six later rounds cover the legacy owner | `Schedule` |
| `nifs.pi_ccs.fe_sumcheck.refinement.width_gap` | typed lane width three differs from legacy wire width five | `WireFormat` |
-/
