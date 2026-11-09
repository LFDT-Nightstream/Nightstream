import CompressionLean.Spec.OptionBMatrix
import NightstreamFPrime.Spec.Folding.Light.Bound
import NightstreamFPrime.Spec.Folding.Schedule

/-! What a slot's layer-1 key covers (CW-10.6). Layer 1 checks a witness within the key's norm
bound, and the key commits only its own blocks (`Final.parent_complete`). The honest witness of a
next slot lies below the accumulator bound of the light cap plus one and is zero on every ring
block from the logical width of its circuit's package (F′'s `FinishCompleteness.finish_complete`).
`coverFailure` checks, on a step package and the layer-1 key that Rust writes for the package's
circuit, that the key's norm bound is at least that accumulator bound (`normCovers`) and that the
key's blocks hold the package's logical width (`blocksCover`). -/

namespace CompressionLean.Spec.OptionB

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Export.OptionB.StepPackage (Package)
open CompressionLean.Circuit (Layer1Key)

/-- The honest norm bound of a slot: the accumulator bound at the light cap
(`GlobalParams.accumulatorBound`, Rust's `option_b/compression.rs::compression_bound`). -/
abbrev slotBound : ℕ := productionGlobalParams.accumulatorBound Folding.Schedule.productionLightFoldCap

theorem slotBound_eq : slotBound = 17496 := by decide

/-- Layer-1 key `key` covers the honest witnesses of step package `p`'s circuit. -/
structure Covers (p : Package) (key : Layer1Key) : Prop where
  normCovers : slotBound ≤ key.relation.normBound
  blocksCover : ProductionAssignment.logicalWidth (slotsOf p) ≤ ringDegree * key.relation.key.blocks

def coverChecks (p : Package) (key : Layer1Key) : List Fact :=
  [⟨"normCovers", slotBound ≤ key.relation.normBound⟩,
    ⟨"blocksCover", ProductionAssignment.logicalWidth (slotsOf p) ≤ ringDegree * key.relation.key.blocks⟩]

def coverFailure (p : Package) (key : Layer1Key) : Option String :=
  firstFailure (coverChecks p key)

end CompressionLean.Spec.OptionB
