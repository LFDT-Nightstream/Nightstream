import Nightstream.Implementation.NebulaV2.FPrime.State.AuthoritativeOutputBinding

/-! Focused regressions for the complete typed V2 state-output relation. -/

namespace NightstreamTests.NebulaV2AuthoritativeStateOutput

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2

example : Function.Injective
    StateOutputAuthorityRows.payloadFields :=
  StateOutputAuthorityRows.payloadFields_injective

example : Function.Injective
    Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.u64Halves :=
  U64HalvesRows.u64Halves_injective

example {layout : AuthoritativeStateOutputRows.Layout}
    (valid : layout.Valid) :
    (AuthoritativeStateOutputRows.rows layout).length = 24497 :=
  AuthoritativeStateOutputRows.rows_length_exact valid

example [DecidableEq MemoryCarryParser.Block]
    {leftPayload rightPayload : StateOutputAuthorityRows.Payload}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    (leftCanonical : ∀ value ∈
      AuthoritativeStateOutputBinding.typedFrame leftPayload leftBlock,
      value < goldilocksP)
    (rightCanonical : ∀ value ∈
      AuthoritativeStateOutputBinding.typedFrame rightPayload rightBlock,
      value < goldilocksP)
    (equal : AuthoritativeStateOutputBinding.typedDigest leftPayload leftBlock =
      AuthoritativeStateOutputBinding.typedDigest rightPayload rightBlock) :
    (leftPayload = rightPayload ∧ leftBlock = rightBlock) ∨
      StateOutputPoseidonBinding.OuterCollision ∨
      MemoryCarryPoseidonBinding.PoseidonCollision :=
  AuthoritativeStateOutputBinding.typed_authority_eq_or_two_stage_collision
    (StateOutputAuthorityRows.fullFrame_length _ _)
    (StateOutputAuthorityRows.fullFrame_length _ _)
    leftCanonical rightCanonical equal

end NightstreamTests.NebulaV2AuthoritativeStateOutput
