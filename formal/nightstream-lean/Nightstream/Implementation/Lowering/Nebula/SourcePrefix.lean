import Nightstream.Implementation.Lowering.Nebula.SourcePacking

/-!
Exact linear-prefix bridge for the Lean-owned Nebula memory fingerprint.

Assurance tier: model-level.

Owns the small field identity that turns the emitted linear prefix into
`gamma2 - packed(timestamp, globalIndex)`, and its operation and scan
instantiations. It does not unfold concrete 64-bit words during the algebraic
proof.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.Nebula.SourceSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint

private abbrev Lin := Rows.LinearCombination

private theorem sub_sub_eq_sub_add (left middle right : F) :
    (left - middle) - right = left - (middle + right) := by
  calc
    (left - middle) - right = (left + -middle) + -right := by
      rw [Fin.sub_eq_add_neg, Fin.sub_eq_add_neg]
    _ = left + (-middle + -right) := Lean.Grind.Fin.add_assoc _ _ _
    _ = left + -(middle + right) := by
      rw [Lean.Grind.AddCommGroup.neg_add]
    _ = left - (middle + right) := (Fin.sub_eq_add_neg _ _).symm

theorem extension_sub_sub_eq_sub_add (left middle right : K) :
    K.sub (K.sub left middle) right =
      K.sub left (K.add middle right) := by
  change K.mk _ _ = K.mk _ _
  rw [K.mk.injEq]
  exact
    ⟨sub_sub_eq_sub_add left.c0 middle.c0 right.c0,
      sub_sub_eq_sub_add left.c1 middle.c1 right.c1⟩

theorem nestedSub_eq_fingerprint
    (challengeValues : Challenges) (entry : MemTuple) :
    K.sub
        (K.sub challengeValues.gamma2
          (K.embed (Fingerprint.packed entry)))
        (K.mul challengeValues.gamma1
          (K.embed (Fingerprint.valueField entry))) =
      Fingerprint.fingerprint challengeValues entry := by
  unfold Fingerprint.fingerprint
  exact extension_sub_sub_eq_sub_add _ _ _

private theorem evaluatePrefix
    (assignment : Nat -> F) (gammaLow gammaHigh timestamp globalIndex : Lin) :
    evaluatePair assignment
        (Rows.LinearCombination.sub
          (Rows.LinearCombination.sub gammaLow timestamp)
          (Rows.LinearCombination.scale
            (Rows.LinearCombination.fieldTwoPower Layout.timestampBits)
            globalIndex))
        gammaHigh =
      K.sub (evaluatePair assignment gammaLow gammaHigh)
        (K.embed
          (Rows.LinearCombination.eval assignment timestamp +
            Rows.LinearCombination.fieldTwoPower Layout.timestampBits *
              Rows.LinearCombination.eval assignment globalIndex)) := by
  change K.mk _ _ = K.mk _ _
  rw [K.mk.injEq]
  constructor
  · simp only [evaluatePair, K.embed,
      Rows.LinearCombination.eval_sub, Rows.LinearCombination.eval_scale]
    simpa only [Fin.sub_eq_add_neg] using
      sub_sub_eq_sub_add
        (Rows.LinearCombination.eval assignment gammaLow)
        (Rows.LinearCombination.eval assignment timestamp)
        (Rows.LinearCombination.fieldTwoPower Layout.timestampBits *
          Rows.LinearCombination.eval assignment globalIndex)
  · simp only [evaluatePair, K.embed, Fin.sub_eq_add_neg,
      Lean.Grind.AddCommGroup.neg_zero, Fin.add_zero]

theorem operationPrefixPair
    (assignment : Nat -> F) (params : Params)
    (slot : Nat) (write : Bool) :
    evaluatePair assignment
        (operationFingerprintPrefix params slot write) (gammaWord 1 1) =
      K.sub (challenges assignment).gamma2
        (K.embed (Fingerprint.packed
          (operationEntry assignment params slot write))) := by
  rw [operationEntry_packed]
  change evaluatePair assignment
      (Rows.LinearCombination.sub
        (Rows.LinearCombination.sub (gammaWord 1 0)
          (if write then operationWriteTimestamp params slot
            else operationReadTimestamp params slot))
        (Rows.LinearCombination.scale
          (Rows.LinearCombination.fieldTwoPower Layout.timestampBits)
          (operationGlobalIndex params slot)))
      (gammaWord 1 1) =
    K.sub (evaluatePair assignment (gammaWord 1 0) (gammaWord 1 1))
      (K.embed
        (fieldValue assignment
            (if write then operationWriteTimestamp params slot
              else operationReadTimestamp params slot) +
          Rows.LinearCombination.fieldTwoPower Layout.timestampBits *
            fieldValue assignment (operationGlobalIndex params slot)))
  exact evaluatePrefix assignment (gammaWord 1 0) (gammaWord 1 1)
    (if write then operationWriteTimestamp params slot
      else operationReadTimestamp params slot)
    (operationGlobalIndex params slot)

theorem scanPrefixPair
    (assignment : Nat -> F) (params : Params)
    (final : Bool) (slot : Nat) :
    evaluatePair assignment
        (Rows.LinearCombination.sub
          (Rows.LinearCombination.sub (gammaWord 1 0)
            (scanTimestamp params final slot))
          (Rows.LinearCombination.scale
            (Rows.LinearCombination.fieldTwoPower Layout.timestampBits)
            (scanGlobalIndex params slot)))
        (gammaWord 1 1) =
      K.sub (challenges assignment).gamma2
        (K.embed (Fingerprint.packed
          (scanEntry assignment params final slot))) := by
  rw [scanEntry_packed]
  change evaluatePair assignment
      (Rows.LinearCombination.sub
        (Rows.LinearCombination.sub (gammaWord 1 0)
          (scanTimestamp params final slot))
        (Rows.LinearCombination.scale
          (Rows.LinearCombination.fieldTwoPower Layout.timestampBits)
          (scanGlobalIndex params slot)))
      (gammaWord 1 1) =
    K.sub (evaluatePair assignment (gammaWord 1 0) (gammaWord 1 1))
      (K.embed
        (fieldValue assignment (scanTimestamp params final slot) +
          Rows.LinearCombination.fieldTwoPower Layout.timestampBits *
            fieldValue assignment (scanGlobalIndex params slot)))
  exact evaluatePrefix assignment (gammaWord 1 0) (gammaWord 1 1)
    (scanTimestamp params final slot) (scanGlobalIndex params slot)

end Nightstream.Implementation.Lowering.Nebula.SourceSemantics
