import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Types
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiRlc
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec.Checker
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan.Necessity
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Transition
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PiRlcParentOpening
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator.SemanticBoundary

/-!
Curated public surface for the exact concrete Phi81 NIFS composition.

This component is intentionally separate from the older abstract
`Nifs.PaperNifsTransition`. Its Split-NC phase refines the independent
Section 7.3 obligations directly, and its two tail phases use the complete
typed Phi81 algebras.

Owns: the curated imports and ownership map for the concrete Phi81 NIFS
semantic composition.

Does not own: F-prime lifecycle policy, Rust/R1CS refinement, constraint
emission, cost accounting, or row-removal authority.

Emits constraints: no.

| Child stage | Mathematical obligation | Emits constraints? | Lean owner |
|---|---|---|---|
| `nifs.concrete.types` | exact verifier context, raw certificate, and shared derived phase carriers | no | `ConcretePhi81.Types` |
| `nifs.concrete.pi_ccs.domain` | one production dimension record projects to the proved FE and block×lane NC domains | no | `ConcretePhi81.PiCcsDomains` |
| `nifs.concrete.equality` | executable finite equality for typed public carriers | no | `ConcretePhi81.CarrierEquality` |
| `nifs.concrete.running_authority` | bootstrap parent absence or strict active incoming-parent `Pi_DEC`, with derived shared running point | no | `ConcretePhi81.RunningAuthority` |
| `nifs.concrete.pi_rlc.sampler` | exact 54-of-64 bounded batch from the derived `Pi_CCS` outgoing state to RingF membership | no | `ConcretePhi81.Sampler` |
| `nifs.concrete.pi_rlc.sampler.checker` | executable finite batch check exactly equivalent to sampler acceptance | no | `ConcretePhi81.Sampler.Checker` |
| `nifs.concrete.pi_rlc.derived` | exact retained source-structure family; every other public `Pi_RLC` equation is computed | no | `ConcretePhi81.DerivedPiRlc` |
| `nifs.concrete.pi_dec.derived` | canonical child materialization and the exact three-equation recomposition boundary | no | `ConcretePhi81.DerivedPiDec` |
| `nifs.concrete.pi_dec.checker` | executable three-family recomposition check exact to the retained boundary | no | `ConcretePhi81.DerivedPiDec.Checker` |
| `nifs.semantic.fold` | certificate-independent honest outputs, combined parent, radix children, and relation | no | `ConcretePhi81.SemanticFold` |
| `nifs.semantic.obligation_plan` | exact protocol/phase/family leaves over a raw point and challenge candidate | no | `ConcretePhi81.SemanticFold.ObligationPlan` |
| `nifs.semantic.obligation_plan.necessity` | parent-only and child-only removal witnesses for the computed result equalities | no | `ConcretePhi81.SemanticFold.ObligationPlan.Necessity` |
| `nifs.concrete.transition` | physical acceptance, refinement into the semantic fold, named-bad-event soundness, and honest completeness | no | `ConcretePhi81.Transition` |
| `nifs.pi_rlc.verify.authority.parent` | canonical combined CE opening from source-authorized `yRing` values | no | `ConcretePhi81.Authority.PiRlcParentOpening` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol` | product-indexed packed sidecar authority, or explicit mixing, bad-root, or parent-projection mismatch | no | `ConcretePhi81.Authority.PackedYZcol` |
| `nifs.result` | one arity-independent derived parent/children result and its semantic projection | no | `ConcretePhi81.Result` |
| `nifs.fixed_bootstrap` | exact `1 + 0` carrier, absent incoming parent, and complete derived parent/children result | no | `ConcretePhi81.FixedBootstrap` |
| `nifs.paper.fixed_active` | concrete Phi81 refinement of the independent `1 CCS + 14 CE -> 14 CE` paper profile | no | `ConcretePhi81.FixedActive.PaperProfile` |
| `nifs.fixed_active` | exact `1 + 14` carrier and complete derived parent/children result | no | `ConcretePhi81.FixedActive` |
| `nifs.fixed_active.canonical` | verifier-constructed carrier, four-family incoming authority, and exact raw-certificate checker | no | `ConcretePhi81.FixedActive.Canonical` |
| `nifs.fixed_active.run` | fail-closed canonical evaluator and exact physical acceptance boundary | no | `ConcretePhi81.FixedActive.Evaluator` |
| `nifs.fixed_active.soundness` | conditional semantic closure with source/output authority and bad-event exclusion explicit | no | `ConcretePhi81.FixedActive.Evaluator.SemanticBoundary` |
-/
