import NightstreamFPrime.Layout.MatrixProgram.Phi81Product
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram.Phi81Product

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def Family.format : Format Family where
  encode := fun family => .array [
    .atom family.sourceCount,
    .atom family.blockCount,
    .atom family.cellCount]
  decode
    | .array [.atom sourceCount, .atom blockCount, .atom cellCount] =>
        .ok ⟨sourceCount, blockCount, cellCount⟩
    | _ => .error "invalid Phi81 product family"
  decode_encode := by
    intro family
    cases family
    rfl

def Block.format : Format Block where
  encode := fun block => .array [
    (list Family.format).encode block.families,
    .atom block.oneColumn,
    RetainedBlock.format.encode block.challenge,
    .atom block.challengeSlotStart,
    .atom block.challengeSourceStride,
    SourceSubstitution.format.encode block.input,
    RetainedBlock.format.encode block.output,
    RetainedBlock.format.encode block.group]
  decode
    | .array [families, .atom oneColumn, challenge,
        .atom challengeSlotStart, .atom challengeSourceStride,
        input, output, group] => do
      pure {
        families := ← (list Family.format).decode families
        oneColumn
        challenge := ← RetainedBlock.format.decode challenge
        challengeSlotStart
        challengeSourceStride
        input := ← SourceSubstitution.format.decode input
        output := ← RetainedBlock.format.decode output
        group := ← RetainedBlock.format.decode group }
    | _ => .error "invalid Phi81 product block"
  decode_encode := by
    rintro ⟨families, oneColumn, challenge, challengeSlotStart,
      challengeSourceStride, input, output, group⟩
    simp only
    rw [(list Family.format).decode_encode,
      RetainedBlock.format.decode_encode,
      SourceSubstitution.format.decode_encode,
      RetainedBlock.format.decode_encode,
      RetainedBlock.format.decode_encode]
    rfl

end NightstreamFPrime.Layout.MatrixProgram.Phi81Product
