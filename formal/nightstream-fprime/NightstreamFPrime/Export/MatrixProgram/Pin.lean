import NightstreamFPrime.Layout.MatrixProgram.Pin
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram.Pin

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def Block.format : Format Block where
  encode := fun block => .array [
    .atom block.oneColumn,
    (list WireForm.format).encode block.values]
  decode
    | .array [.atom oneColumn, values] => do
        pure ⟨oneColumn, ← (list WireForm.format).decode values⟩
    | _ => .error "invalid pin matrix block"
  decode_encode := by
    intro block
    cases block
    simp [Format.decode_encode]

end NightstreamFPrime.Layout.MatrixProgram.Pin
