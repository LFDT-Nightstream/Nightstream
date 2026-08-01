import Nightstream.Implementation.Lowering.Nebula.SourcePrefix

/-!
Source binding for the Lean-owned Nebula memory products.

Assurance tier: model-level.

Owns: decoding each operation or scan lane into the protocol `MemTuple`,
decoding both extension-field challenges, and proving that every physical
fingerprint factor is exactly the protocol fingerprint of that tuple.

Does not own: bit-to-Nat range bounds, the WASM port map, execution of a
memory update, terminal product balance, Fiat--Shamir unpredictability,
Rust, or a collision probability bound.
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

/-- Every active-operation factor is exactly the protocol fingerprint of the
decoded operation tuple. -/
theorem operationFactor_eq_fingerprint
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool) :
    operationFactor assignment params slot write =
      Fingerprint.fingerprint (challenges assignment)
        (operationEntry assignment params slot write) := by
  calc
    operationFactor assignment params slot write =
        K.sub
          (evaluatePair assignment
            (operationFingerprintPrefix params slot write)
            (gammaWord 1 1))
          (K.mul (challenges assignment).gamma1
            (K.embed (fieldValue assignment
              (if write then operationWriteValue params slot
                else operationReadValue params slot)))) := rfl
    _ = K.sub
        (K.sub (challenges assignment).gamma2
          (K.embed (Fingerprint.packed
            (operationEntry assignment params slot write))))
        (K.mul (challenges assignment).gamma1
          (K.embed (fieldValue assignment
            (if write then operationWriteValue params slot
              else operationReadValue params slot)))) := by
      rw [operationPrefixPair]
    _ = _ := by
      rw [← operationEntry_valueField]
      exact nestedSub_eq_fingerprint _ _

/-- Every scan factor is exactly the protocol fingerprint of the decoded
initial or final cell tuple. -/
theorem scanFactor_eq_fingerprint
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat) :
    scanFactor assignment params final slot =
      Fingerprint.fingerprint (challenges assignment)
        (scanEntry assignment params final slot) := by
  calc
    scanFactor assignment params final slot =
        K.sub
          (evaluatePair assignment
            (Rows.LinearCombination.sub
              (Rows.LinearCombination.sub (gammaWord 1 0)
                (scanTimestamp params final slot))
              (Rows.LinearCombination.scale
                (Rows.LinearCombination.fieldTwoPower Layout.timestampBits)
                (scanGlobalIndex params slot)))
            (gammaWord 1 1))
          (K.mul (challenges assignment).gamma1
            (K.embed (fieldValue assignment
              (scanValue params final slot)))) := rfl
    _ = K.sub
        (K.sub (challenges assignment).gamma2
          (K.embed (Fingerprint.packed
            (scanEntry assignment params final slot))))
        (K.mul (challenges assignment).gamma1
          (K.embed (fieldValue assignment
            (scanValue params final slot)))) := by
      rw [scanPrefixPair]
    _ = _ := by
      rw [← scanEntry_valueField]
      exact nestedSub_eq_fingerprint _ _

end Nightstream.Implementation.Lowering.Nebula.SourceSemantics
