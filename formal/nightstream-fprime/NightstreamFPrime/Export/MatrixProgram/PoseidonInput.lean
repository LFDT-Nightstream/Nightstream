import NightstreamFPrime.Layout.MatrixProgram.PoseidonInput
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Export.MatrixProgram.Affine

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram.PoseidonInput

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def Region.format : Format Region where
  encode := fun region => .array [
    .atom region.invocationStart,
    .atom region.invocationCount,
    .atom region.laneStart,
    .atom region.laneCount]
  decode
    | .array [.atom invocationStart, .atom invocationCount,
        .atom laneStart, .atom laneCount] =>
        .ok ⟨invocationStart, invocationCount, laneStart, laneCount⟩
    | _ => .error "invalid Poseidon2 input region"
  decode_encode := by
    intro region
    cases region
    rfl

def InvocationTag.format : Format InvocationTag where
  encode
    | .absorb => .atom 0
    | .squeezeFirst => .atom 1
    | .squeezeSecond => .atom 2
  decode
    | .atom 0 => .ok .absorb
    | .atom 1 => .ok .squeezeFirst
    | .atom 2 => .ok .squeezeSecond
    | _ => .error "invalid Poseidon2 invocation tag"
  decode_encode := by
    intro tag
    cases tag <;> rfl

def TagTable.format : Format TagTable where
  encode := fun table => (list InvocationTag.format).encode table.tags.toList
  decode := fun value => do
    pure ⟨(← (list InvocationTag.format).decode value).toArray⟩
  decode_encode := by
    intro table
    cases table with
    | mk tags =>
        simp [Format.decode_encode]

def OptionalConstantTable.format : Format OptionalConstantTable where
  encode := fun table =>
    (list (option nat)).encode table.values.toList
  decode := fun value => do
    pure ⟨(← (list (option nat)).decode value).toArray⟩
  decode_encode := by
    intro table
    cases table with
    | mk values =>
        simp [Format.decode_encode]

def Term.format : Format Term where
  encode
    | .retained block slotBase invocationStride laneStride => .array [
        .atom 0, RetainedBlock.format.encode block, .atom slotBase,
        .atom invocationStride, .atom laneStride]
    | .constant coefficient => .array [.atom 1, .atom coefficient]
    | .external block slotBase invocationStride => .array [
        .atom 2, RetainedBlock.format.encode block, .atom slotBase,
        .atom invocationStride]
    | .taggedRetained block tags required slotBase invocationStride
        laneStride => .array [
          .atom 3, RetainedBlock.format.encode block, TagTable.format.encode tags,
          InvocationTag.format.encode required, .atom slotBase,
          .atom invocationStride, .atom laneStride]
    | .optionalConstant values laneCount => .array [
        .atom 4, OptionalConstantTable.format.encode values, .atom laneCount]
    | .taggedAffine values substitution tags required laneCount => .array [
        .atom 5, Affine.Table.format.encode values,
        SourceSubstitution.format.encode substitution, TagTable.format.encode tags,
        InvocationTag.format.encode required, .atom laneCount]
  decode
    | .array [.atom 0, block, .atom slotBase, .atom invocationStride,
        .atom laneStride] => do
        pure (.retained (← RetainedBlock.format.decode block) slotBase
          invocationStride laneStride)
    | .array [.atom 1, .atom coefficient] => .ok (.constant coefficient)
    | .array [.atom 2, block, .atom slotBase, .atom invocationStride] => do
        pure (.external (← RetainedBlock.format.decode block) slotBase
          invocationStride)
    | .array [.atom 3, block, tags, required, .atom slotBase,
        .atom invocationStride, .atom laneStride] => do
        pure (.taggedRetained (← RetainedBlock.format.decode block)
          (← TagTable.format.decode tags) (← InvocationTag.format.decode required)
          slotBase invocationStride laneStride)
    | .array [.atom 4, values, .atom laneCount] => do
        pure (.optionalConstant (← OptionalConstantTable.format.decode values)
          laneCount)
    | .array [.atom 5, values, substitution, tags, required, .atom laneCount] => do
        pure (.taggedAffine (← Affine.Table.format.decode values)
          (← SourceSubstitution.format.decode substitution)
          (← TagTable.format.decode tags) (← InvocationTag.format.decode required)
          laneCount)
    | _ => .error "invalid Poseidon2 input term"
  decode_encode := by
    intro term
    cases term <;> simp [RetainedBlock.format.decode_encode,
      TagTable.format.decode_encode, InvocationTag.format.decode_encode,
      OptionalConstantTable.format.decode_encode, Affine.Table.format.decode_encode,
      SourceSubstitution.format.decode_encode] <;> rfl

def Rule.format : Format Rule where
  encode := fun rule => .array [
    Region.format.encode rule.region,
    Term.format.encode rule.term]
  decode
    | .array [region, term] => do
        pure ⟨← Region.format.decode region, ← Term.format.decode term⟩
    | _ => .error "invalid Poseidon2 input rule"
  decode_encode := by
    intro rule
    cases rule
    simp [Region.format.decode_encode, Term.format.decode_encode]
    rfl

def Program.format : Format Program where
  encode := fun program => (list Rule.format).encode program.rules
  decode := fun value => do
    pure ⟨← (list Rule.format).decode value⟩
  decode_encode := by
    intro program
    cases program
    simp [Format.decode_encode]

end NightstreamFPrime.Layout.MatrixProgram.PoseidonInput
