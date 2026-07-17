import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc

/-!
Executable SumCheck layer of the independent production-shaped Split-NC
verifier.

Owns: child ownership and dependency direction only; this file emits no rows.

Does not own: protocol polynomials, transcript challenge derivation, output
authority, Rust, R1CS, row removal, or constraint counts.

Emits constraints: no.

| Child stage | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.sumcheck` | mixed-width FE messages, one cross-phase claimed chain, and fixed-challenge completeness adapter | no | `Verifier.SumCheck.Fe` |
| `nifs.pi_ccs.nc.sumcheck` | exact-width NC messages, generic replay adapter, completeness, and named deterministic bad events | no | `Verifier.SumCheck.Nc` |
-/
