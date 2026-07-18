import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-!
Canonical Π_CCS transcript dimensions for the concrete Phi81 NIFS profile.

Assurance tier: model-level.

Owns: the single production dimension record and its exact FE/block×lane NC
projections.

Does not own: relation-domain minimality, transcript operations, verifier
acceptance, Poseidon2, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: the values are assembled only from independently proved
relation-domain constants. Neither a transcript trace nor an R1CS artifact is
used as dimension authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.pi_ccs.domain.shared` | one `9/3/6` record owns both arithmetization views | computed | `production` |
| `nifs.concrete.pi_ccs.domain.fe` | FE projection is the proven `9/6` compatibility domain | derived | `production_fe` |
| `nifs.concrete.pi_ccs.domain.nc` | canonical NC projection is the proven `3/6` block domain | derived | `production_nc` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- Production dimensions represented once before either phase view is
projected. -/
def production : Domains where
  columnVariables := 9
  blockVariables := 3
  laneVariables := 6

@[simp] theorem production_fe : production.fe = PiCcsDomain.domain := by
  rfl

@[simp] theorem production_nc : production.nc = PiCcsDomain.blockDomain := by
  rfl

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains
