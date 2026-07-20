import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-!
Canonical Π_CCS transcript dimensions for the concrete Phi81 NIFS profiles.

Assurance tier: model-level.

Owns: the bounded five-ring public-prefix dimension record, the complete
fixed-point production dimension record, and their exact FE/block×lane NC
projections.

Does not own: relation-domain minimality, transcript operations, verifier
acceptance, Poseidon2, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: these values configure transcript capacity. Generated
shape agreement and minimality belong to the implementation correspondence;
neither a transcript trace nor an R1CS artifact supplies semantic authority
here.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.pi_ccs.domain.public_prefix` | one `9/3/6` record owns the bounded five-ring profile | computed | `publicPrefix` |
| `nifs.concrete.pi_ccs.domain.fixed_point` | one `24/19/6` record owns the complete fixed-point profile | computed | `fixedPointProduction` |
| `nifs.concrete.pi_ccs.domain.fe` | the fixed-point FE compatibility view has 24 column variables | derived | `fixedPointProduction_fe` |
| `nifs.concrete.pi_ccs.domain.nc` | canonical fixed-point NC has 19 block and 6 lane variables | derived | `fixedPointProduction_nc` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- Bounded public-prefix dimensions represented once before either phase
view is projected. This remains a diagnostic five-ring profile; it is not the
complete fixed-point witness domain. -/
def publicPrefix : Domains where
  columnVariables := 9
  blockVariables := 3
  laneVariables := 6

@[simp] theorem publicPrefix_fe : publicPrefix.fe = PiCcsDomain.domain := by
  rfl

@[simp] theorem publicPrefix_nc : publicPrefix.nc = PiCcsDomain.blockDomain := by
  rfl

/-- Complete fixed-point transcript capacity. The executable profile has
14,338,890 physical scalar columns, organized as exactly 265,535 Phi81 blocks
with 54 live lanes, so its FE and NC transcript capacities use 24 column
variables and 19 block variables. The implementation layer separately proves
that the generated fixed-point relation fits these protocol dimensions;
generated metadata is not imported as semantic authority here. -/
def fixedPointProduction : Domains where
  columnVariables := 24
  blockVariables := 19
  laneVariables := 6

@[simp] theorem fixedPointProduction_fe_columnVariables :
    fixedPointProduction.fe.columnVariables = 24 := by
  rfl

@[simp] theorem fixedPointProduction_fe_laneVariables :
    fixedPointProduction.fe.laneVariables = 6 := by
  rfl

@[simp] theorem fixedPointProduction_nc_blockVariables :
    fixedPointProduction.nc.blockVariables = 19 := by
  rfl

@[simp] theorem fixedPointProduction_nc_laneVariables :
    fixedPointProduction.nc.laneVariables = 6 := by
  rfl

theorem fixedPointProduction_flatRoundCount :
    fixedPointProduction.fe.columnVariables +
        fixedPointProduction.fe.laneVariables = 30 := by
  decide

theorem fixedPointProduction_blockRoundCount :
    fixedPointProduction.nc.blockVariables +
        fixedPointProduction.nc.laneVariables = 25 := by
  decide

/-- Active fixed-point contexts use the complete witness domain. The bounded
public-prefix profile remains available explicitly as `publicPrefix`. -/
abbrev production : Domains := fixedPointProduction

@[simp] theorem production_fe : production.fe = fixedPointProduction.fe := by
  rfl

@[simp] theorem production_nc : production.nc = fixedPointProduction.nc := by
  rfl

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains
