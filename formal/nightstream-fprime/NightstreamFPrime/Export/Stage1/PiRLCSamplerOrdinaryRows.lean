import NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations

/-! Emits the checked wide reduction and its coefficient-word rows. The two
Poseidon2 permutations per scalar are owned by PiRLCSamplerInvocations. Every
ordinary row is compiled from the canonical circuit at its declared offset. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Layout NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiRLCSamplerInvocations (sourceInterface sourceLogicalStart)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

def rangeInterface (source : Nat) : WideReduction.Interface where
  source := fun lane _ => PiRLCSamplerInvocations.fastAdvanceState
    (logicalWidth := logicalWidth) (publicFits := publicFits) source (Sampler.rateLane lane)

theorem rangeInterface_eq (source : Nat) :
    rangeInterface (logicalWidth := logicalWidth) (publicFits := publicFits) source =
      Sampler.rangeInterface (sourceInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source) source (sourceLogicalStart source) := by
  unfold rangeInterface
  rw [PiRLCSamplerInvocations.fastAdvanceState_eq]
  rfl

def rangeConstraints (source : Nat) : List Expr :=
  flatConstraints (WideReduction.Program.operations
    (rangeInterface (logicalWidth := logicalWidth) (publicFits := publicFits) source)
    (Layout.Stage1.PiRLCStarts.rangeLogicalStart source))

def wordConstraints (source : Nat) : List Expr :=
  flatConstraints (SamplerWords.operations (Layout.Stage1.PiRLCStarts.rangeLogicalStart source)
    (Layout.Stage1.PiRLCStarts.challengeWordStart source))

def rangeRows (source : Nat) : List Rows.CompiledRow :=
  PiCCSArithmetic.compilePacket (Layout.Stage1.PiRLCStarts.rangeRowStart source)
    (Layout.Stage1.PiRLCStarts.rangeFreshStart source)
    (rangeConstraints (logicalWidth := logicalWidth) (publicFits := publicFits) source)

def wordRows (source : Nat) : List Rows.CompiledRow :=
  PiCCSArithmetic.compilePacket (Layout.Stage1.PiRLCStarts.challengeWordRowStart source)
    (Layout.Stage1.PiRLCStarts.rangeFreshStart source + 144) (wordConstraints source)

def sourceRows (source : Nat) : List Rows.CompiledRow :=
  rangeRows (logicalWidth := logicalWidth) (publicFits := publicFits) source ++ wordRows source

def rows : List Rows.CompiledRow :=
  (List.range PiRLCSamplerInvocations.sourceCount).flatMap
    (sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits))

theorem rangeConstraints_eq (source : Nat) :
    rangeConstraints (logicalWidth := logicalWidth) (publicFits := publicFits) source =
      flatConstraints (Circuit.ops
        (Sampler.rangeCircuit (sourceInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source) source (sourceLogicalStart source)).main
        (Layout.Stage1.PiRLCStarts.rangeLogicalStart source)) := by
  unfold rangeConstraints
  rw [rangeInterface_eq]
  rfl

theorem rangeRows_toR1CS (source : Nat) :
    (rangeRows (logicalWidth := logicalWidth) (publicFits := publicFits) source).map Rows.CompiledRow.toR1CS =
      Layout.Stage1.Spartan.remapRows (R1CS.lowerConstraints
        (rangeConstraints (logicalWidth := logicalWidth) (publicFits := publicFits) source)
        (Layout.Stage1.PiRLCStarts.rangeFreshStart source)).rows :=
  PiCCSArithmetic.compilePacket_toR1CS _ _ _

theorem wordRows_toR1CS (source : Nat) :
    (wordRows source).map Rows.CompiledRow.toR1CS =
      Layout.Stage1.Spartan.remapRows (R1CS.lowerConstraints (wordConstraints source)
        (Layout.Stage1.PiRLCStarts.rangeFreshStart source + 144)).rows :=
  PiCCSArithmetic.compilePacket_toR1CS _ _ _

@[simp] theorem rangeRows_length (source : Nat) :
    (rangeRows (logicalWidth := logicalWidth) (publicFits := publicFits) source).length = 825 := by
  rw [rangeRows, PiCCSArithmetic.compilePacket_length]
  unfold rangeConstraints
  rw [rangeInterface_eq, WideReduction.Program.constraints_eq]
  exact (Layout.Sampling.WideReduction.counts _ _ _ (fun lane =>
    Layout.PiRLC.v1_1.Sampler.entered_affine
      (sourceInterface (logicalWidth := logicalWidth) (publicFits := publicFits) source)
      source (sourceLogicalStart source) (Sampler.rateLane lane))).2

@[simp] theorem wordRows_length (source : Nat) : (wordRows source).length = 54 := by
  rw [wordRows, PiCCSArithmetic.compilePacket_length, R1CS.totalRowCount_eq_fresh_add_length]
  have fresh := Layout.PiRLC.v1_1.Sampler.words_fresh (sourceLogicalStart source)
  change R1CS.totalFreshCount (wordConstraints source) = 0 at fresh
  rw [fresh]
  change 0 + (flatConstraints (SamplerWords.operations _ _)).length = 54
  rw [SamplerWords.rowCount_eq, SamplerWords.count_eq]

@[simp] theorem sourceRows_length (source : Nat) :
    (sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits) source).length = 879 := by
  simp [sourceRows]

theorem rows_length :
    (rows (logicalWidth := logicalWidth) (publicFits := publicFits)).length = 14943 := by
  simp [rows, PiRLCSamplerInvocations.sourceCount]

theorem rows_imply_sourceRows (source : Fin PiRLCSamplerInvocations.sourceCount) (env : Env)
    (held : R1CS.RowsHold env ((rows (logicalWidth := logicalWidth)
      (publicFits := publicFits)).map Rows.CompiledRow.toR1CS)) :
    R1CS.RowsHold env ((sourceRows (logicalWidth := logicalWidth)
      (publicFits := publicFits) source.val).map Rows.CompiledRow.toR1CS) := by
  intro row member
  obtain ⟨compiled, inSource, rfl⟩ := List.mem_map.mp member
  apply held _
  apply List.mem_map.mpr
  exact ⟨compiled, List.mem_flatMap.mpr
    ⟨source.val, List.mem_range.mpr source.isLt, inSource⟩, rfl⟩

theorem rangeRows_imply_spec (source : Nat) (env : Env)
    (inputs : WideReduction.Assumptions
      (rangeInterface (logicalWidth := logicalWidth) (publicFits := publicFits) source)
      (Layout.Stage1.PiRLCStarts.rangeLogicalStart source))
    (held : R1CS.RowsHold env ((rangeRows (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).map Rows.CompiledRow.toR1CS)) :
    (WideReduction.Program.circuit
      (rangeInterface (logicalWidth := logicalWidth) (publicFits := publicFits) source)).spec
      (Layout.Stage1.PiRLCStarts.rangeLogicalStart source) (Layout.Stage1.Spartan.pullback env) := by
  have logical := PiCCSArithmetic.compilePacket_sound _ _ _ env held
  exact (WideReduction.Program.circuit _).soundness _ _ inputs
    (holdsFlat_implies_holds _ _ logical)

theorem wordRows_imply_spec (source : Nat) (env : Env)
    (held : R1CS.RowsHold env ((wordRows source).map Rows.CompiledRow.toR1CS)) :
    SamplerWords.SpecHolds (Layout.Stage1.PiRLCStarts.rangeLogicalStart source)
      (Layout.Stage1.PiRLCStarts.challengeWordStart source) (Layout.Stage1.Spartan.pullback env) := by
  have logical := PiCCSArithmetic.compilePacket_sound _ _ _ env held
  exact SamplerWords.soundness _ _ _ (holdsFlat_implies_holds _ _ logical)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
