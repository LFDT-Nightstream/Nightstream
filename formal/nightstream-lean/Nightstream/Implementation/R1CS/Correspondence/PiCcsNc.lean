import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Semantics
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.SourceRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.CarrierCoverageRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Terminal.Identity
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.ProjectionNecessity
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.TerminalEqualityNecessity
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedParentProjection
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.ProjectionIdentity
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.ProjectionBinding

/-!
Owns: the curated implementation-refinement surface for the Π_CCS norm-check
phase.

Does not own: abstract Π_CCS protocol soundness, FE semantics, concrete
transcript rows not explicitly imported here, or any row-removal decision.

Emits constraints: no.

Authority boundary: the abstract protocol theorem becomes usable for the
production verifier only after semantics, transcript, SumCheck, terminal, and
outer-assignment correspondence are all connected. This root exports the
completed range and mixed-polynomial leaves plus conditional terminal and
delayed-authority semantics while leaving their production derivations open.

| Phase | Mathematical obligation | Emits constraints? | Current owner |
|---|---|---|---|
| range semantics | classify the embedded `b = 2` range-factor roots | no | `PiCcsNc.Semantics` |
| parameters | carry domain widths without hand-written fixed-profile constants | no | `PiCcsNc.Parameters` |
| true polynomial | define the independently evaluated direct-packed mixed NC polynomial | no | `PiCcsNc.Semantics.MixedPolynomial` |
| source refinement | connect the independent full carrier to the executable direct table | no | exact cells/norms; honest zero-claim only |
| carrier coverage refinement | connect one exact Rust packed pair to independent full-carrier NC truth | no | exact optimized `Pi_CCS` API accepts the counterexample; general and NIFS/F-prime refinement remain open |
| transcript | bind verifier-owned mixing and round challenges | yes in Rust | open Lean refinement |
| SumCheck | replay claimed rows and compare with the true polynomial | yes in Rust | claimed-chain correspondence only |
| terminal | connect carried `y_zcol` to packed assignments under `YZcolBound` | yes in Rust | conditional model theorem; verifier derivation open |
| authority necessity | show that erasing `y_zcol` can change the accepted NC statement | no | concrete counterexample; no row-removal permission |
| terminal-equality necessity | show that the terminal range scalar can agree despite a false `y_zcol` sidecar | no | concrete model-level counterexample; no row-removal permission |
| delayed authority | transfer a verified old-point raw-child projection to the state-bound parent and retain authoritative next outputs | not yet | conditional optimized model theorem; SumCheck, Π_RLC/Π_DEC, and state/commitment refinement open |
| delayed residual | lift the radix-combined raw-child evaluation at producer `beta` into the NC cube | not yet | model-level formula; transcript and SumCheck-row refinement open |
| delayed projection identity | expose 54 active coefficients, two limb evaluations, canonical padding, and exact-or-degree-53-bad-root semantics | not yet | model-level identity; generated parent-column, beta-ladder, transcript, and opening refinement open |
| delayed projection binding | transfer an accepted compact identity into the delayed NC cube sum | not yet | model-level exact-or-bad-root theorem; transcript, state, and row refinement open |
-/
