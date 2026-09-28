import NightstreamFPrime.Layout.MatrixProgram.Ordinary
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Export.MatrixProgram.SourceProjection

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram.Ordinary

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

def Block.format : Format Block where
  encode := fun block => .array [
    IndexSchedule.format.encode block.rows,
    .atom block.oneColumn,
    SourceSubstitution.format.encode block.substitution,
    SourceProjection.format.encode block.projection]
  decode
    | .array [rows, .atom oneColumn, substitution, projection] => do
      pure ⟨← IndexSchedule.format.decode rows, oneColumn,
        ← SourceSubstitution.format.decode substitution,
        ← SourceProjection.format.decode projection⟩
    | _ => .error "invalid ordinary matrix block"
  decode_encode := by
    intro block
    cases block
    simp [IndexSchedule.format.decode_encode,
      SourceSubstitution.format.decode_encode,
      SourceProjection.format.decode_encode]
    rfl

end NightstreamFPrime.Layout.MatrixProgram.Ordinary
