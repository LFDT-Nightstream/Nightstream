import NightstreamFPrime.Layout.MatrixProgram.AffineGrid
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram.AffineGrid

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

def Region.format : Format Region where
  encode := fun region => .array [
    .atom region.majorStart,
    .atom region.majorCount,
    .atom region.middleStart,
    .atom region.middleCount,
    .atom region.minorStart,
    .atom region.minorCount]
  decode
    | .array [.atom majorStart, .atom majorCount,
        .atom middleStart, .atom middleCount,
        .atom minorStart, .atom minorCount] =>
        .ok (Region.mk majorStart majorCount middleStart middleCount
          minorStart minorCount)
    | _ => .error "invalid affine-grid region"
  decode_encode := by
    intro region
    cases region
    rfl

def Term.format : Format Term where
  encode
    | .retained block slotBase majorStride middleStride minorStride
        coefficient => .array [
          .atom 0, RetainedBlock.format.encode block, .atom slotBase,
          .atom majorStride, .atom middleStride, .atom minorStride,
          .atom coefficient]
    | .constant coefficient => .array [.atom 1, .atom coefficient]
  decode
    | .array [.atom 0, block, .atom slotBase, .atom majorStride,
        .atom middleStride, .atom minorStride, .atom coefficient] => do
        pure (.retained (← RetainedBlock.format.decode block) slotBase
          majorStride middleStride minorStride coefficient)
    | .array [.atom 1, .atom coefficient] => .ok (.constant coefficient)
    | _ => .error "invalid affine-grid term"
  decode_encode := by
    intro term
    cases term <;> simp [RetainedBlock.format.decode_encode]

def Rule.format : Format Rule where
  encode := fun rule => .array [
    Region.format.encode rule.region,
    Term.format.encode rule.term]
  decode
    | .array [region, term] => do
        pure ⟨← Region.format.decode region, ← Term.format.decode term⟩
    | _ => .error "invalid affine-grid rule"
  decode_encode := by
    rintro ⟨region, term⟩
    simp only
    rw [Region.format.decode_encode, Term.format.decode_encode]
    rfl

def Program.format : Format Program where
  encode := fun program => (list Rule.format).encode program.rules
  decode := fun value => do
    pure ⟨← (list Rule.format).decode value⟩
  decode_encode := by
    intro program
    cases program
    simp [Format.decode_encode]

end NightstreamFPrime.Layout.MatrixProgram.AffineGrid
