import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource

/-!
Contract: typed six-phase source program for the native delayed F' public
input link.

Owns:
- exact verifier-input ownership and order for the three shape checks,
  affine-one check, ordered 256-coordinate body range, and trailing padding;
- a definitional obligation cost;
- universal executable equivalence with the plain source checker.

Does not own: a generated Rust-program receipt, Rust or MIR semantics,
lifecycle call-site arguments, R1CS rows, or the optional application suffix.

Emits constraints: no.

| Instruction | Input owner | Output | Cost |
|---|---|---|---|
| expected public length | verifier-owned layout | Boolean | 1 |
| claim `m_in` | claim metadata | Boolean | 1 |
| claim `x` length | claim vector shape | Boolean | 1 |
| affine one | claim coordinate 0 | Boolean | 1 |
| body range | expected bits and claim coordinates 1--256 | Boolean | range length |
| padding range | trailing claim coordinates | Boolean | range length |
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgram

open Nightstream.Implementation.Encoding.FPrime
open CanonicalPlainCarrierLink
open CanonicalPlainCarrierSource

/-- One retained phase of the source-level verifier program. -/
inductive Instruction where
  | expectedPublicInputLen (expected : Nat)
  | claimMIn (expected : Nat)
  | claimXLen (expected : Nat)
  | affineOne (claimIndex : Nat)
  | bodyRange (expectedOffset claimOffset length : Nat)
  | paddingZeroRange (claimOffset length : Nat)
deriving Repr, DecidableEq

namespace Instruction

/-- Number of scalar obligations retained by one source instruction. -/
def cost : Instruction -> Nat
  | .expectedPublicInputLen _ => 1
  | .claimMIn _ => 1
  | .claimXLen _ => 1
  | .affineOne _ => 1
  | .bodyRange _ _ length => length
  | .paddingZeroRange _ length => length

private def slice?
    (values : List Nat) (offset length : Nat) : Option (List Nat) :=
  if offset + length ≤ values.length then
    some ((values.drop offset).take length)
  else
    none

/-- Executable meaning of one typed source instruction. -/
def check
    (instruction : Instruction)
    (digest : Digest)
    (expectedPublicInputLen : Nat)
    (claim : RawClaim) : Bool :=
  match instruction with
  | .expectedPublicInputLen expected =>
      decide (expectedPublicInputLen = expected)
  | .claimMIn expected =>
      decide (claim.mIn = expected)
  | .claimXLen expected =>
      decide (claim.x.length = expected)
  | .affineOne claimIndex =>
      decide (claim.x.getD claimIndex 0 = 1)
  | .bodyRange expectedOffset claimOffset length =>
      decide
        (slice? claim.x claimOffset length =
          slice? (bodyCoordinates digest) expectedOffset length)
  | .paddingZeroRange claimOffset length =>
      decide (claim.x.drop claimOffset = List.replicate length 0)

end Instruction

abbrev Program := List Instruction

def cost (program : Program) : Nat :=
  (program.map Instruction.cost).sum

def run
    (program : Program)
    (digest : Digest)
    (expectedPublicInputLen : Nat)
    (claim : RawClaim) : Bool :=
  program.all fun instruction =>
    instruction.check digest expectedPublicInputLen claim

/-- Canonical plain-profile source program in production evaluation order. -/
def plain : Program :=
  [ .expectedPublicInputLen carrierWidth
  , .claimMIn carrierWidth
  , .claimXLen carrierWidth
  , .affineOne 0
  , .bodyRange 0 1 256
  , .paddingZeroRange 257 paddingWidth
  ]

theorem plain_cost : cost plain = 273 := by
  decide

private theorem paddingCoordinates_eq_replicate :
    paddingCoordinates = List.replicate paddingWidth 0 := by
  decide

/-- The typed program is extensionally the already-audited plain source
checker on every untrusted input. -/
theorem run_plain_eq_sourceCheck
    (digest : Digest)
    (claim : RawClaim) :
    run plain digest carrierWidth claim =
      sourceCheck digest claim := by
  apply Bool.eq_iff_iff.mpr
  cases claim with
  | mk mIn coordinates =>
      cases coordinates with
      | nil =>
          simp [run, plain, Instruction.check, Instruction.slice?,
            sourceCheck, carrierWidth]
      | cons one tail =>
          by_cases tailLength : tail.length = 269
          · have bodyTake :
                (bodyCoordinates digest).take 256 =
                  bodyCoordinates digest := by
              rw [← bodyCoordinates_length digest, List.take_length]
            simp [run, plain, Instruction.check, Instruction.slice?,
              sourceCheck, carrierWidth, paddingWidth, logicalWidth,
              tailLength, bodyCoordinates_length,
              paddingCoordinates_eq_replicate, bodyTake, Bool.and_assoc]
          · have totalLength : tail.length + 1 ≠ 270 := by
              omega
            simp [run, plain, Instruction.check, Instruction.slice?,
              sourceCheck, carrierWidth, totalLength]

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgram
