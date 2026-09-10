import NightstreamFPrime.Layout.MatrixProgram.Affine
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram.Affine

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

def Form.format : Format Form where
  encode := fun form => .array [.atom form.constant, WireForm.format.encode form.terms]
  decode
    | .array [.atom constant, terms] => do
        pure ⟨constant, ← WireForm.format.decode terms⟩
    | _ => .error "invalid affine source form"
  decode_encode := by
    intro form
    cases form
    simp [Format.decode_encode]

def Table.format : Format Table where
  encode := fun table => (list Form.format).encode table.values.toList
  decode := fun value => do
    pure ⟨(← (list Form.format).decode value).toArray⟩
  decode_encode := by
    intro table
    cases table
    simp [Format.decode_encode]

end NightstreamFPrime.Layout.MatrixProgram.Affine
