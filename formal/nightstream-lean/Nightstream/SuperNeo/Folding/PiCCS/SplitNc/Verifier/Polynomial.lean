import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum.CarriedBridge
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.SourceProjection
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Terminal

/-!
Polynomial layer of the independent production-shaped Split-NC verifier.

Owns: child ownership and dependency direction only; this file emits no rows.

Does not own: equations already owned by its children, transcript derivation,
SumCheck execution, Rust, R1CS, row removal, or constraint counts.

Emits constraints: no.

| Child stage | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.parameters` | typed row/lane point and physical row/lane degree ceilings | no | `Polynomial.Fe.Parameters` |
| `nifs.pi_ccs.fe` | fresh CCS and running CE product-domain polynomial | no | `Polynomial.Fe` |
| `nifs.pi_ccs.fe.source` | source-derived fresh/running FE refinement | no | `Polynomial.Fe.SourceRefinement` |
| `nifs.pi_ccs.fe.initial` | exact Boolean-cube decomposition and independent residual mix | no | `Polynomial.Fe.InitialSum` |
| `nifs.pi_ccs.fe.initial.carried` | carried selector closure and unconditional honest initial-sum completeness | no | `Polynomial.Fe.InitialSum.CarriedBridge` |
| `nifs.pi_ccs.degree` | shared Boolean-MLE and equality-selector coordinate slices | no | `Polynomial.DegreeSupport` |
| `nifs.pi_ccs.fe.degree` | syntax-derived row and lane round widths | no | `Polynomial.Fe.Degree` |
| `nifs.pi_ccs.fe.soundness` | zero FE mix is truth or an explicit nonzero-residual compression root | no | `Polynomial.Fe.MixingSoundness` |
| `nifs.pi_ccs.nc.domain` | typed column/lane carrier and fail-closed decoding | no | `Polynomial.Nc` |
| `nifs.pi_ccs.nc.source_projection` | full padded source table, nested MLE, and Boolean cubic/truth equivalence | no | `Polynomial.Nc.SourceProjection` |
| `nifs.pi_ccs.nc.mixing` | explicit paper/joint/Split-V1 gamma schedules and equality-gated source mixing | no | `Polynomial.Nc.Mixing` |
| `nifs.pi_ccs.nc.initial` | exact source-specialization identity and honest zero-claim completeness | no | `Polynomial.Nc.InitialSum` |
| `nifs.pi_ccs.nc.degree` | five-coefficient per-variable ceiling for strict-`b = 2` NC | no | `Polynomial.Nc.Degree` |
| `nifs.pi_ccs.nc.soundness` | exact paper-relative truth/selector-root/gamma-root dichotomy | no | `Polynomial.Nc.MixingSoundness` |
| `nifs.pi_ccs.nc.terminal` | source-bound `yZcol` lane MLE and exact raw-message/semantic terminal equality | no | `Polynomial.Nc.Terminal` |
-/
