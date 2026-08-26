import NightstreamFPrime.Export.Stage1.PiCCSArithmetic
import NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain
import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.PiRLCStarts

/-!
Owns the canonical ordinary-row packet for the production PiRLC sampler
chain.

The logical constraint list is the sampler child selected by the sole PiRLC
phase assembler. Its R1CS-fresh columns start after every logical PiRLC child,
as required by the full-phase lowering plan.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerRows

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def sharedInterface :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Interface
      logicalWidth publicFits :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset
    (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset

def samplerInterface :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerInterface
    (sharedInterface (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The exact sampler child list in the production PiRLC parent. -/
def constraints : List Expr :=
  NightstreamFPrime.Layout.PiRLC.v1_1.childConstraints
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerCircuit
      (sharedInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits)))
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart

/-- The sampler prefix lowered at the full phase's first fresh column. -/
def rows : List Rows.CompiledRow :=
  PiCCSArithmetic.compilePacket
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerRowStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerFreshStart
    (constraints (logicalWidth := logicalWidth) (publicFits := publicFits))

theorem constraints_eq_samplerChain :
    constraints (logicalWidth := logicalWidth) (publicFits := publicFits) =
      NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain.logicalConstraints
        (samplerInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits))
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart := by
  rfl

/-- Exact final-column image of the Lean-lowered sampler prefix. -/
theorem rows_toR1CS :
    (rows (logicalWidth := logicalWidth) (publicFits := publicFits)).map
        Rows.CompiledRow.toR1CS =
      NightstreamFPrime.Layout.Stage1.Spartan.remapRows
        (R1CS.lowerConstraints
          (constraints (logicalWidth := logicalWidth)
            (publicFits := publicFits))
          NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerFreshStart).rows := by
  exact PiCCSArithmetic.compilePacket_toR1CS _ _ _

theorem rows_length :
    (rows (logicalWidth := logicalWidth) (publicFits := publicFits)).length =
      1008848 := by
  rw [rows, PiCCSArithmetic.compilePacket_length,
    constraints_eq_samplerChain]
  exact
    NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain.totalRowCount_eq
      (samplerInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.samplerInputs
        (logicalWidth := logicalWidth) (publicFits := publicFits))

theorem freshCount_eq :
    R1CS.totalFreshCount
        (constraints (logicalWidth := logicalWidth) (publicFits := publicFits)) =
      743631 := by
  rw [constraints_eq_samplerChain]
  exact
    NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain.totalFreshCount_eq
      (samplerInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.samplerInputs
        (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- Satisfaction of the exported sampler rows implies the exact production
sampler-chain relation. -/
theorem rows_imply_relation (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.Assumptions
        (samplerInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits))
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
    (holds : R1CS.RowsHold env
      ((rows (logicalWidth := logicalWidth) (publicFits := publicFits)).map
        Rows.CompiledRow.toR1CS)) :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.RelationHolds
      (samplerInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  have logical := PiCCSArithmetic.compilePacket_sound
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerRowStart
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerFreshStart
    (constraints (logicalWidth := logicalWidth) (publicFits := publicFits)) env
    holds
  rw [constraints_eq_samplerChain] at logical
  apply NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.soundness
    (samplerInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) assumptions
  exact holdsFlat_implies_holds _ _ (by
    simpa only [NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain.logicalConstraints]
      using logical)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerRows
