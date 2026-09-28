import NightstreamFPrime.Layout.MatrixProgram.SourceProjection
import NightstreamFPrime.Export.Codec

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout

def SourceProjectionRange.format : Format SourceProjectionRange where
  encode := fun range => .array [
    .atom range.packageStart,
    .atom range.sourceStart,
    .atom range.count]
  decode
    | .array [.atom packageStart, .atom sourceStart, .atom count] =>
        .ok ⟨packageStart, sourceStart, count⟩
    | _ => .error "invalid matrix source projection range"
  decode_encode := by
    intro range
    cases range
    rfl

def SourceProjection.format : Format SourceProjection where
  encode
    | .identity => .array [.atom 0]
    | .mapped items => .array [
        .atom 1, (list SourceProjectionRange.format).encode items]
  decode
    | .array [.atom 0] => .ok .identity
    | .array [.atom 1, items] => do
        pure (.mapped
          (← (list SourceProjectionRange.format).decode items))
    | _ => .error "invalid matrix source projection"
  decode_encode := by
    intro projection
    cases projection with
    | identity => rfl
    | mapped items =>
        simp [Format.decode_encode]

end NightstreamFPrime.Layout.MatrixProgram
