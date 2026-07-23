import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalRawOldBlockProjectionFinalScaleCompiler
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalRawOldBlockProjectionPhysicalIndex
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.ProductionEmitterLayout

/-!
Typed layout boundary for the active terminal raw-old-block projection.

This leaf turns the generated Rust row-at program into the compiler's compact
prefix `Layout` and the final-round-factorized `Layout`.  It also names the
structural property guaranteed by Rust's
crate-private `RawOldBlockProjectionColumnMap::new`: the physical column map
is well formed and its generated inverse recovers every canonical column
used by the program.  Neither property mentions assignment values, semantic
acceptance, child sidecars, digests, or commitment authority.

Owns: the typed fixed-production prefix and final-scale compiler layouts
decoded from the generated row-at plan, and the narrow `EmitterLayoutValid`
contract for runtime physical column allocation.

Does not own: proof that the concrete emitter layout is valid, column-inverse
correctness, row coefficients, assignment values, row satisfaction, or
semantic projection authority.

Emits constraints: no.  The Rust owner is
`paper/decider_ce_relation/old_block_projection.rs`; the generated owner is
`Execution/RawOldBlockProjectionRowAt.lean`.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.projection_layout.profile` | radix, child count, lane count, logical width, and block-variable count match the generated production plan | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.projection_layout.tensor` | eighteen generated prefix tensor levels retain exact emitter order | computed |
| `f_prime.pi_ccs_nc.delayed.projection_layout.final_scale` | old-block coordinate 18 and one generated scale trace per lane are associated explicitly | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.projection_layout.emitter_valid` | a physical emitter layout passes the generated range/disjointness check | checked contract |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt

private def profileRadix : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.radixBase
private def profileChildren : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.childCount
private def profileActiveLanes : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.activeLanes
private def profileLogicalWidth : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.logicalWidth
private def profileTensorVariables : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.tensorVariables
private def profileFactoredVariable : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.factoredVariable

/-- One exact round of the Rust compact-prefix tensor program. -/
def productionTensorLevel (round : Nat) : TensorLevel where
  multiplicationCount := tensorRoundMulCount round
  trace := fun parent => tensorTrace round parent.val

/-- The eighteen production prefix-tensor rounds, in emitter order. -/
def productionTensorLevels : List TensorLevel :=
  (List.range profileTensorVariables).map productionTensorLevel

/-- Canonical production layout before the runtime emitter column map is
applied.  Raw child columns are the ordered `FinalWitnessWires` allocations
also consumed by the terminal Ajtai opening. -/
def productionLayout : Layout where
  radix := profileRadix
  childCount := profileChildren
  activeLanes := profileActiveLanes
  logicalWidth := profileLogicalWidth
  blockVariables := profileTensorVariables
  oldBlock := fun round => oldBlockColumnsNat round.val
  parent := parentColumns
  childWitnessFirst := childWitnessFirst
  productFirst := productFirstColumn
  tensorLevels := productionTensorLevels

/-- The optimized production layout.  Its prefix is `productionLayout`; the
omitted nineteenth point coordinate is named directly, and every lane's
five-row scale trace is the generated Rust trace. -/
def productionFactoredLayout :
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler.Layout where
  base := productionLayout
  factor :=
    { factorFinalRound :=
        Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.factorFinalRound
      tensorVariables := profileTensorVariables
      factoredVariable := profileFactoredVariable
      fullOldBlock := oldBlockColumnsNat
      finalPoint := oldBlockColumnsNat profileFactoredVariable }
  scale := fun lane => finalScaleTrace lane.val

/-- Structural validity of a runtime emitter layout.  This is exactly the
generated fail-closed Rust constructor check.  In particular, callers cannot
supply a claimed map inverse; the column-map leaf derives it from this Boolean
fact. -/
structure EmitterLayoutValid (emitter : EmitterLayout) : Prop where
  checked : emitterColumnMapValid emitter = true

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
