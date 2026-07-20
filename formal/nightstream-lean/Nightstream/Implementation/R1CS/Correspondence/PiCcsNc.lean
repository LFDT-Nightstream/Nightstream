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
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.OldPointBinding
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.ProductionRawChildren
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.FlatCombinedNc
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.FlatCombinedNc.Verifier

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
| delayed old-point binding | decode all 54 active lanes, append the ten explicit padding lanes, and promote the accepted identity to `OldPointSumcheckRelation` | not yet | model-level exact-or-bad-root theorem; production acceptance and padding-row derivation open |
| delayed raw-child authority | materialize the fixed-profile child table directly from typed running assignments | not yet | exact Lean dataflow; Rust/R1CS witness-table decoder open, and `CeClaim.y_zcol` is excluded |
| flat combined NC | combine the current 9-column/6-lane NC polynomial with the raw-child delayed residual, prove the quartic round bound and terminal formula, and extract NC truth plus old-point binding or named roots/collision | not yet | model-level fixed-phase theorem; transcript scheduling, generated padding rows, state continuity, commitment binding, and the Rust raw-witness handoff remain open |
| flat combined NC verifier | replay the exact five-coefficient messages through the Poseidon2 machine, derive all 15 challenges, and recompute the terminal from raw running assignments | not yet | model-level executable checker; producer challenge scheduling, generated padding rows, state continuity, commitment binding, and Rust/R1CS refinement remain open |
-/
