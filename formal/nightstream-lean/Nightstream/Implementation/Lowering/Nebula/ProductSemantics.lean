import Nightstream.Implementation.Lowering.Nebula.Physical
import Nightstream.Protocol.Nebula.Memory

/-!
Semantic product refinement for the Lean-owned Nebula compiler.

Assurance tier: model-level.

Owns: the exact interpretation of each two-row quadratic-extension update,
the operation and scan factors computed from physical columns, and the
connection from emitted CCS satisfaction to running-product updates.

Does not own: bit decoding, WASM port binding, segment composition, the
terminal balance check, Fiat--Shamir challenge generation, Rust, or a
collision probability bound.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.Lowering.Nebula.ProductSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

private abbrev Lin := Rows.LinearCombination

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨Lean.Grind.Fin.add_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨Lean.Grind.Fin.add_comm⟩

private theorem mul_neg (left right : F) :
    left * -right = -(left * right) := by
  calc
    left * -right = -right * left := Fin.mul_comm _ _
    _ = -(right * left) := Lean.Grind.Fin.neg_mul _ _
    _ = -(left * right) := by rw [Fin.mul_comm right left]

private theorem mul_sub_expanded (left factor coefficient value : F) :
    left * (factor - coefficient * value) =
      left * factor + -(left * coefficient * value) := by
  rw [Fin.sub_eq_add_neg, Lean.Grind.Fin.left_distrib, mul_neg,
    ← Fin.mul_assoc]

private theorem update_component
    (left right pad active factor0 coefficient0 factor1 coefficient1
      value : F) :
    left * (pad + active * (factor0 - coefficient0 * value)) +
        right * (active * (factor1 - coefficient1 * value)) =
      left * pad + left * active * factor0 +
        -(left * active * coefficient0 * value) +
        right * active * factor1 +
        -(right * active * coefficient1 * value) := by
  rw [Lean.Grind.Fin.left_distrib,
    ← Fin.mul_assoc left active,
    mul_sub_expanded,
    ← Fin.mul_assoc right active,
    mul_sub_expanded]
  simp only [Lean.Grind.Fin.add_assoc]

private theorem move_last_to_front
    (a b c d e last : F) :
    a + b + c + d + e + last =
      last + a + b + c + d + e := by
  ac_rfl

def evaluatePair
    (assignment : Nat → F) (low high : Lin) : K :=
  ⟨Rows.LinearCombination.eval assignment low,
    Rows.LinearCombination.eval assignment high⟩

def fingerprintFactor
    (assignment : Nat → F)
    (fingerprint0 fingerprint1 value0 value1 value : Lin) : K :=
  K.sub
    (evaluatePair assignment fingerprint0 fingerprint1)
    (K.mul (evaluatePair assignment value0 value1)
      (K.embed (Rows.LinearCombination.eval assignment value)))

def gatedFactor
    (assignment : Nat → F)
    (pad active fingerprint0 fingerprint1 value0 value1 value : Lin) : K :=
  K.add
    (K.embed (Rows.LinearCombination.eval assignment pad))
    (K.mul
      (K.embed (Rows.LinearCombination.eval assignment active))
      (fingerprintFactor assignment fingerprint0 fingerprint1
        value0 value1 value))

/-- Two emitted extension rows are exactly one multiplication in
`F[X]/(X² - 7)`. This theorem does not assume a Boolean pad value. -/
theorem extensionRows_sound
    (family : Family) (slot : Nat)
    (output0 output1 previous0 previous1 pad active fingerprint0
      fingerprint1 value0 value1 value : Lin)
    (assignment : Nat → F)
    (satisfied : Satisfies
      (extensionRows family slot output0 output1 previous0 previous1
        pad active fingerprint0 fingerprint1 value0 value1 value)
      assignment) :
    evaluatePair assignment output0 output1 =
      K.mul (evaluatePair assignment previous0 previous1)
        (gatedFactor assignment pad active fingerprint0 fingerprint1
          value0 value1 value) := by
  have low := satisfied
    (extensionUpdateRow (id family slot 0 0)
      output0 previous0 (Rows.LinearCombination.scale 7 previous1)
      pad active fingerprint0 fingerprint1 value0 value1 value)
    (by simp [extensionRows])
  have high := satisfied
    (extensionUpdateRow (id family slot 1 0)
      output1 previous1 previous0 pad active fingerprint0 fingerprint1
      value0 value1 value)
    (by simp [extensionRows])
  rw [extensionUpdateRow_holds_iff] at low high
  simp only [Rows.LinearCombination.eval_scale] at low
  have lowExact :
      Rows.LinearCombination.eval assignment output0 =
        Rows.LinearCombination.eval assignment previous0 *
            Rows.LinearCombination.eval assignment pad +
          Rows.LinearCombination.eval assignment previous0 *
              Rows.LinearCombination.eval assignment active *
            Rows.LinearCombination.eval assignment fingerprint0 +
          -(Rows.LinearCombination.eval assignment previous0 *
              Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment value0 *
              Rows.LinearCombination.eval assignment value) +
          7 * Rows.LinearCombination.eval assignment previous1 *
              Rows.LinearCombination.eval assignment active *
            Rows.LinearCombination.eval assignment fingerprint1 +
          -(7 * Rows.LinearCombination.eval assignment previous1 *
              Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment value1 *
              Rows.LinearCombination.eval assignment value) := by
    apply (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp ?_).symm
    calc
      _ = -Rows.LinearCombination.eval assignment output0 +
              Rows.LinearCombination.eval assignment previous0 *
                Rows.LinearCombination.eval assignment pad +
              Rows.LinearCombination.eval assignment previous0 *
                  Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment fingerprint0 +
              -(Rows.LinearCombination.eval assignment previous0 *
                  Rows.LinearCombination.eval assignment active *
                  Rows.LinearCombination.eval assignment value0 *
                  Rows.LinearCombination.eval assignment value) +
              7 * Rows.LinearCombination.eval assignment previous1 *
                  Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment fingerprint1 +
              -(7 * Rows.LinearCombination.eval assignment previous1 *
                  Rows.LinearCombination.eval assignment active *
                  Rows.LinearCombination.eval assignment value1 *
                  Rows.LinearCombination.eval assignment value) := by
            rw [Fin.sub_eq_add_neg]
            exact move_last_to_front _ _ _ _ _ _
      _ = 0 := low
  have highExact :
      Rows.LinearCombination.eval assignment output1 =
        Rows.LinearCombination.eval assignment previous1 *
            Rows.LinearCombination.eval assignment pad +
          Rows.LinearCombination.eval assignment previous1 *
              Rows.LinearCombination.eval assignment active *
            Rows.LinearCombination.eval assignment fingerprint0 +
          -(Rows.LinearCombination.eval assignment previous1 *
              Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment value0 *
              Rows.LinearCombination.eval assignment value) +
          Rows.LinearCombination.eval assignment previous0 *
              Rows.LinearCombination.eval assignment active *
            Rows.LinearCombination.eval assignment fingerprint1 +
          -(Rows.LinearCombination.eval assignment previous0 *
              Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment value1 *
              Rows.LinearCombination.eval assignment value) := by
    apply (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp ?_).symm
    calc
      _ = -Rows.LinearCombination.eval assignment output1 +
              Rows.LinearCombination.eval assignment previous1 *
                Rows.LinearCombination.eval assignment pad +
              Rows.LinearCombination.eval assignment previous1 *
                  Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment fingerprint0 +
              -(Rows.LinearCombination.eval assignment previous1 *
                  Rows.LinearCombination.eval assignment active *
                  Rows.LinearCombination.eval assignment value0 *
                  Rows.LinearCombination.eval assignment value) +
              Rows.LinearCombination.eval assignment previous0 *
                  Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment fingerprint1 +
              -(Rows.LinearCombination.eval assignment previous0 *
                  Rows.LinearCombination.eval assignment active *
                  Rows.LinearCombination.eval assignment value1 *
                  Rows.LinearCombination.eval assignment value) := by
            rw [Fin.sub_eq_add_neg]
            exact move_last_to_front _ _ _ _ _ _
      _ = 0 := high
  change K.mk _ _ = K.mk _ _
  rw [K.mk.injEq]
  constructor
  · simp only [evaluatePair, gatedFactor, fingerprintFactor, K.mul, K.add,
      K.embed, K.sub, Fin.mul_zero, Fin.zero_mul, Fin.add_zero,
      Fin.zero_add]
    rw [lowExact]
    exact (update_component
      (Rows.LinearCombination.eval assignment previous0)
      (7 * Rows.LinearCombination.eval assignment previous1)
      (Rows.LinearCombination.eval assignment pad)
      (Rows.LinearCombination.eval assignment active)
      (Rows.LinearCombination.eval assignment fingerprint0)
      (Rows.LinearCombination.eval assignment value0)
      (Rows.LinearCombination.eval assignment fingerprint1)
      (Rows.LinearCombination.eval assignment value1)
      (Rows.LinearCombination.eval assignment value)).symm
  · simp only [evaluatePair, gatedFactor, fingerprintFactor, K.mul, K.add,
      K.embed, K.sub, Fin.mul_zero, Fin.zero_mul, Fin.add_zero,
      Fin.zero_add]
    rw [highExact]
    calc
      _ = Rows.LinearCombination.eval assignment previous1 *
            (Rows.LinearCombination.eval assignment pad +
              Rows.LinearCombination.eval assignment active *
                (Rows.LinearCombination.eval assignment fingerprint0 -
                  Rows.LinearCombination.eval assignment value0 *
                    Rows.LinearCombination.eval assignment value)) +
          Rows.LinearCombination.eval assignment previous0 *
            (Rows.LinearCombination.eval assignment active *
              (Rows.LinearCombination.eval assignment fingerprint1 -
                Rows.LinearCombination.eval assignment value1 *
                  Rows.LinearCombination.eval assignment value)) :=
        (update_component
          (Rows.LinearCombination.eval assignment previous1)
          (Rows.LinearCombination.eval assignment previous0)
          (Rows.LinearCombination.eval assignment pad)
          (Rows.LinearCombination.eval assignment active)
          (Rows.LinearCombination.eval assignment fingerprint0)
          (Rows.LinearCombination.eval assignment value0)
          (Rows.LinearCombination.eval assignment fingerprint1)
          (Rows.LinearCombination.eval assignment value1)
          (Rows.LinearCombination.eval assignment value)).symm
      _ = _ := Lean.Grind.Fin.add_comm _ _

/-- The converse of `extensionRows_sound`: an exact extension-product update
is sufficient for both emitted component rows. -/
theorem extensionRows_honest
    (family : Family) (slot : Nat)
    (output0 output1 previous0 previous1 pad active fingerprint0
      fingerprint1 value0 value1 value : Lin)
    (assignment : Nat → F)
    (exact :
      evaluatePair assignment output0 output1 =
        K.mul (evaluatePair assignment previous0 previous1)
          (gatedFactor assignment pad active fingerprint0 fingerprint1
            value0 value1 value)) :
    Satisfies
      (extensionRows family slot output0 output1 previous0 previous1
        pad active fingerprint0 fingerprint1 value0 value1 value)
      assignment := by
  have lowExact := congrArg K.c0 exact
  have highExact := congrArg K.c1 exact
  simp only [evaluatePair, gatedFactor, fingerprintFactor, K.mul, K.add,
    K.embed, K.sub, Fin.mul_zero, Fin.zero_mul, Fin.add_zero,
    Fin.zero_add] at lowExact highExact
  intro row member
  simp [extensionRows] at member
  rcases member with rfl | rfl
  · rw [extensionUpdateRow_holds_iff]
    simp only [Rows.LinearCombination.eval_scale]
    have expanded :
        Rows.LinearCombination.eval assignment previous0 *
              Rows.LinearCombination.eval assignment pad +
            Rows.LinearCombination.eval assignment previous0 *
                Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment fingerprint0 +
            -(Rows.LinearCombination.eval assignment previous0 *
                Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment value0 *
                Rows.LinearCombination.eval assignment value) +
            7 * Rows.LinearCombination.eval assignment previous1 *
                Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment fingerprint1 +
            -(7 * Rows.LinearCombination.eval assignment previous1 *
                Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment value1 *
                Rows.LinearCombination.eval assignment value) =
          Rows.LinearCombination.eval assignment output0 := by
      calc
        _ = Rows.LinearCombination.eval assignment previous0 *
                (Rows.LinearCombination.eval assignment pad +
                  Rows.LinearCombination.eval assignment active *
                    (Rows.LinearCombination.eval assignment fingerprint0 -
                      Rows.LinearCombination.eval assignment value0 *
                        Rows.LinearCombination.eval assignment value)) +
              7 * Rows.LinearCombination.eval assignment previous1 *
                (Rows.LinearCombination.eval assignment active *
                  (Rows.LinearCombination.eval assignment fingerprint1 -
                    Rows.LinearCombination.eval assignment value1 *
                      Rows.LinearCombination.eval assignment value)) :=
            (update_component
              (Rows.LinearCombination.eval assignment previous0)
              (7 * Rows.LinearCombination.eval assignment previous1)
              (Rows.LinearCombination.eval assignment pad)
              (Rows.LinearCombination.eval assignment active)
              (Rows.LinearCombination.eval assignment fingerprint0)
              (Rows.LinearCombination.eval assignment value0)
              (Rows.LinearCombination.eval assignment fingerprint1)
              (Rows.LinearCombination.eval assignment value1)
              (Rows.LinearCombination.eval assignment value)).symm
        _ = _ := lowExact.symm
    calc
      _ = Rows.LinearCombination.eval assignment previous0 *
              Rows.LinearCombination.eval assignment pad +
            Rows.LinearCombination.eval assignment previous0 *
                Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment fingerprint0 +
            -(Rows.LinearCombination.eval assignment previous0 *
                Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment value0 *
                Rows.LinearCombination.eval assignment value) +
            7 * Rows.LinearCombination.eval assignment previous1 *
                Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment fingerprint1 +
            -(7 * Rows.LinearCombination.eval assignment previous1 *
                Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment value1 *
                Rows.LinearCombination.eval assignment value) +
            -Rows.LinearCombination.eval assignment output0 :=
          (move_last_to_front _ _ _ _ _ _).symm
      _ = 0 := by
        rw [expanded, ← Fin.sub_eq_add_neg]
        exact Fin.sub_self
  · rw [extensionUpdateRow_holds_iff]
    have expanded :
        Rows.LinearCombination.eval assignment previous1 *
              Rows.LinearCombination.eval assignment pad +
            Rows.LinearCombination.eval assignment previous1 *
                Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment fingerprint0 +
            -(Rows.LinearCombination.eval assignment previous1 *
                Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment value0 *
                Rows.LinearCombination.eval assignment value) +
            Rows.LinearCombination.eval assignment previous0 *
                Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment fingerprint1 +
            -(Rows.LinearCombination.eval assignment previous0 *
                Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment value1 *
                Rows.LinearCombination.eval assignment value) =
          Rows.LinearCombination.eval assignment output1 := by
      calc
        _ = Rows.LinearCombination.eval assignment previous1 *
                (Rows.LinearCombination.eval assignment pad +
                  Rows.LinearCombination.eval assignment active *
                    (Rows.LinearCombination.eval assignment fingerprint0 -
                      Rows.LinearCombination.eval assignment value0 *
                        Rows.LinearCombination.eval assignment value)) +
              Rows.LinearCombination.eval assignment previous0 *
                (Rows.LinearCombination.eval assignment active *
                  (Rows.LinearCombination.eval assignment fingerprint1 -
                    Rows.LinearCombination.eval assignment value1 *
                      Rows.LinearCombination.eval assignment value)) :=
            (update_component
              (Rows.LinearCombination.eval assignment previous1)
              (Rows.LinearCombination.eval assignment previous0)
              (Rows.LinearCombination.eval assignment pad)
              (Rows.LinearCombination.eval assignment active)
              (Rows.LinearCombination.eval assignment fingerprint0)
              (Rows.LinearCombination.eval assignment value0)
              (Rows.LinearCombination.eval assignment fingerprint1)
              (Rows.LinearCombination.eval assignment value1)
              (Rows.LinearCombination.eval assignment value)).symm
        _ = _ := Lean.Grind.Fin.add_comm _ _
        _ = _ := highExact.symm
    calc
      _ = Rows.LinearCombination.eval assignment previous1 *
              Rows.LinearCombination.eval assignment pad +
            Rows.LinearCombination.eval assignment previous1 *
                Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment fingerprint0 +
            -(Rows.LinearCombination.eval assignment previous1 *
                Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment value0 *
                Rows.LinearCombination.eval assignment value) +
            Rows.LinearCombination.eval assignment previous0 *
                Rows.LinearCombination.eval assignment active *
              Rows.LinearCombination.eval assignment fingerprint1 +
            -(Rows.LinearCombination.eval assignment previous0 *
                Rows.LinearCombination.eval assignment active *
                Rows.LinearCombination.eval assignment value1 *
                Rows.LinearCombination.eval assignment value) +
            -Rows.LinearCombination.eval assignment output1 :=
          (move_last_to_front _ _ _ _ _ _).symm
      _ = 0 := by
        rw [expanded, ← Fin.sub_eq_add_neg]
        exact Fin.sub_self

def operationFactor
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool) : K :=
  fingerprintFactor assignment
    (operationFingerprintPrefix params slot write)
    (gammaWord 1 1)
    (gammaWord 0 0)
    (gammaWord 0 1)
    (if write then operationWriteValue params slot
      else operationReadValue params slot)

def operationGate
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool) : K :=
  gatedFactor assignment
    (operationPad params slot)
    (Rows.LinearCombination.sub one (operationPad params slot))
    (operationFingerprintPrefix params slot write)
    (gammaWord 1 1)
    (gammaWord 0 0)
    (gammaWord 0 1)
    (if write then operationWriteValue params slot
      else operationReadValue params slot)

def operationProduct
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool) : K :=
  evaluatePair assignment
    (if write then operationWriteProductWord params slot 0
      else operationReadProductWord params slot 0)
    (if write then operationWriteProductWord params slot 1
      else operationReadProductWord params slot 1)

def previousOperationProductValue
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool) : K :=
  evaluatePair assignment
    (previousOperationProduct params slot (if write then 1 else 0) 0)
    (previousOperationProduct params slot (if write then 1 else 0) 1)

private theorem previousOperationProductValue_succ
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool) :
    previousOperationProductValue assignment params (slot + 1) write =
      operationProduct assignment params slot write := by
  cases write <;>
    rfl

theorem operationProductRows_sound
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool)
    (satisfied : Satisfies (operationProductRows params slot write)
      assignment) :
    operationProduct assignment params slot write =
      K.mul (previousOperationProductValue assignment params slot write)
        (operationGate assignment params slot write) := by
  simpa [operationProductRows, operationProduct,
    previousOperationProductValue, operationGate] using
    extensionRows_sound
      (if write then Family.writeProduct else .readProduct) slot
      (if write then operationWriteProductWord params slot 0
        else operationReadProductWord params slot 0)
      (if write then operationWriteProductWord params slot 1
        else operationReadProductWord params slot 1)
      (previousOperationProduct params slot (if write then 1 else 0) 0)
      (previousOperationProduct params slot (if write then 1 else 0) 1)
      (operationPad params slot)
      (Rows.LinearCombination.sub one (operationPad params slot))
      (operationFingerprintPrefix params slot write)
      (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
      (if write then operationWriteValue params slot
        else operationReadValue params slot)
      assignment satisfied

theorem operationProductRows_honest
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool)
    (exact :
      operationProduct assignment params slot write =
        K.mul (previousOperationProductValue assignment params slot write)
          (operationGate assignment params slot write)) :
    Satisfies (operationProductRows params slot write) assignment := by
  simpa [operationProductRows, operationProduct,
    previousOperationProductValue, operationGate] using
    extensionRows_honest
      (if write then Family.writeProduct else .readProduct) slot
      (if write then operationWriteProductWord params slot 0
        else operationReadProductWord params slot 0)
      (if write then operationWriteProductWord params slot 1
        else operationReadProductWord params slot 1)
      (previousOperationProduct params slot (if write then 1 else 0) 0)
      (previousOperationProduct params slot (if write then 1 else 0) 1)
      (operationPad params slot)
      (Rows.LinearCombination.sub one (operationPad params slot))
      (operationFingerprintPrefix params slot write)
      (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
      (if write then operationWriteValue params slot
        else operationReadValue params slot)
      assignment exact

def scanFactor
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat) : K :=
  fingerprintFactor assignment
    (Rows.LinearCombination.sub
      (Rows.LinearCombination.sub (gammaWord 1 0)
        (scanTimestamp params final slot))
      (Rows.LinearCombination.scale
        (Rows.LinearCombination.fieldTwoPower timestampBits)
        (scanGlobalIndex params slot)))
    (gammaWord 1 1)
    (gammaWord 0 0)
    (gammaWord 0 1)
    (scanValue params final slot)

def scanProduct
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat) : K :=
  evaluatePair assignment
    (scanProductWord params final slot 0)
    (scanProductWord params final slot 1)

def previousScanProductValue
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat) : K :=
  evaluatePair assignment
    (previousScanProduct params final slot 0)
    (previousScanProduct params final slot 1)

theorem scanRowsForLane_sound
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (scanRowsForLane params final slot) assignment) :
    scanProduct assignment params final slot =
      K.mul (previousScanProductValue assignment params final slot)
        (scanFactor assignment params final slot) := by
  have extensionSatisfied : Satisfies
      (extensionRows
        (if final then Family.finalScanProduct else .initialScanProduct)
        slot
        (scanProductWord params final slot 0)
        (scanProductWord params final slot 1)
        (previousScanProduct params final slot 0)
        (previousScanProduct params final slot 1)
        Rows.LinearCombination.zero one
        (Rows.LinearCombination.sub
          (Rows.LinearCombination.sub (gammaWord 1 0)
            (scanTimestamp params final slot))
          (Rows.LinearCombination.scale
            (Rows.LinearCombination.fieldTwoPower timestampBits)
            (scanGlobalIndex params slot)))
        (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
        (scanValue params final slot)) assignment := by
    intro row member
    apply satisfied row
    simp only [scanRowsForLane]
    exact List.mem_append_right _ member
  have update := extensionRows_sound
      (if final then Family.finalScanProduct else .initialScanProduct)
      slot
      (scanProductWord params final slot 0)
      (scanProductWord params final slot 1)
      (previousScanProduct params final slot 0)
      (previousScanProduct params final slot 1)
      Rows.LinearCombination.zero one
      (Rows.LinearCombination.sub
        (Rows.LinearCombination.sub (gammaWord 1 0)
          (scanTimestamp params final slot))
        (Rows.LinearCombination.scale
          (Rows.LinearCombination.fieldTwoPower timestampBits)
          (scanGlobalIndex params slot)))
      (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
      (scanValue params final slot) assignment extensionSatisfied
  have gateExact :
      gatedFactor assignment Rows.LinearCombination.zero one
        (Rows.LinearCombination.sub
          (Rows.LinearCombination.sub (gammaWord 1 0)
            (scanTimestamp params final slot))
          (Rows.LinearCombination.scale
            (Rows.LinearCombination.fieldTwoPower timestampBits)
            (scanGlobalIndex params slot)))
        (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
        (scanValue params final slot) =
      scanFactor assignment params final slot := by
    simp [gatedFactor, scanFactor, one,
      Rows.LinearCombination.eval_zero,
      Rows.LinearCombination.eval_constant, constantWire,
      K.add, K.mul, K.embed,
      Fin.one_mul, Fin.zero_mul, Fin.mul_zero, Fin.add_zero,
      Fin.zero_add]
  simpa [scanProduct, previousScanProductValue, gateExact] using update

theorem scanProductRows_honest
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat)
    (constantWire : assignment 0 = 1)
    (exact :
      scanProduct assignment params final slot =
        K.mul (previousScanProductValue assignment params final slot)
          (scanFactor assignment params final slot)) :
    Satisfies
      (extensionRows
        (if final then Family.finalScanProduct else .initialScanProduct)
        slot
        (scanProductWord params final slot 0)
        (scanProductWord params final slot 1)
        (previousScanProduct params final slot 0)
        (previousScanProduct params final slot 1)
        Rows.LinearCombination.zero one
        (Rows.LinearCombination.sub
          (Rows.LinearCombination.sub (gammaWord 1 0)
            (scanTimestamp params final slot))
          (Rows.LinearCombination.scale
            (Rows.LinearCombination.fieldTwoPower timestampBits)
            (scanGlobalIndex params slot)))
        (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
        (scanValue params final slot)) assignment := by
  have gateExact :
      gatedFactor assignment Rows.LinearCombination.zero one
        (Rows.LinearCombination.sub
          (Rows.LinearCombination.sub (gammaWord 1 0)
            (scanTimestamp params final slot))
          (Rows.LinearCombination.scale
            (Rows.LinearCombination.fieldTwoPower timestampBits)
            (scanGlobalIndex params slot)))
        (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
        (scanValue params final slot) =
      scanFactor assignment params final slot := by
    simp [gatedFactor, scanFactor, one,
      Rows.LinearCombination.eval_zero,
      Rows.LinearCombination.eval_constant, constantWire,
      K.add, K.mul, K.embed,
      Fin.one_mul, Fin.zero_mul, Fin.mul_zero, Fin.add_zero,
      Fin.zero_add]
  apply extensionRows_honest
  simpa [scanProduct, previousScanProductValue, gateExact] using exact

private theorem satisfies_cons_iff
    (row : Row) (rest : List Row) (assignment : Nat → F) :
    Satisfies (row :: rest) assignment ↔
      row.Holds assignment ∧ Satisfies rest assignment := by
  constructor
  · intro satisfied
    constructor
    · exact satisfied row (List.mem_cons_self)
    · intro tail tailMember
      exact satisfied tail (List.mem_cons_of_mem row tailMember)
  · rintro ⟨head, tail⟩ candidate member
    rcases List.mem_cons.mp member with rfl | tailMember
    · exact head
    · exact tail candidate tailMember

private theorem satisfies_append_iff
    (left right : List Row) (assignment : Nat → F) :
    Satisfies (left ++ right) assignment ↔
      Satisfies left assignment ∧ Satisfies right assignment := by
  constructor
  · intro satisfied
    constructor
    · intro row member
      exact satisfied row (List.mem_append_left right member)
    · intro row member
      exact satisfied row (List.mem_append_right left member)
  · rintro ⟨leftSatisfied, rightSatisfied⟩ row member
    rcases List.mem_append.mp member with inLeft | inRight
    · exact leftSatisfied row inLeft
    · exact rightSatisfied row inRight

private theorem satisfies_numberRowsFrom_iff
    (position : Nat) (source : List Row) (assignment : Nat → F) :
    Satisfies (numberRowsFrom position source) assignment ↔
      Satisfies source assignment := by
  induction source generalizing position with
  | nil => simp [Satisfies, numberRowsFrom]
  | cons head tail inductionHypothesis =>
      rw [numberRowsFrom, satisfies_cons_iff, satisfies_cons_iff,
        Row.withPosition_holds_iff, inductionHypothesis]

theorem rawRows_satisfied_of_rows
    (assignment : Nat → F) (params : Params)
    (satisfied : Satisfies (rows params) assignment) :
    Satisfies (rawRows params) assignment := by
  exact (satisfies_numberRowsFrom_iff 0 (rawRows params) assignment).mp
    (by simpa [rows] using satisfied)

theorem operationRows_satisfied_of_rows
    (assignment : Nat → F) (params : Params)
    (satisfied : Satisfies (rows params) assignment)
    (slot : Nat) (slotBound : slot < params.operationSlots) :
    Satisfies (operationRows params slot) assignment := by
  have rawSatisfied := rawRows_satisfied_of_rows assignment params satisfied
  intro row member
  apply rawSatisfied row
  unfold rawRows
  apply List.mem_append_left
  apply List.mem_append_left
  apply List.mem_append_right
  exact List.mem_flatMap.mpr
    ⟨slot, List.mem_range.mpr slotBound, member⟩

theorem scanRows_satisfied_of_rows
    (assignment : Nat → F) (params : Params)
    (satisfied : Satisfies (rows params) assignment)
    (slot : Nat) (slotBound : slot < params.scanSlots) :
    Satisfies (scanRows params slot) assignment := by
  have rawSatisfied := rawRows_satisfied_of_rows assignment params satisfied
  intro row member
  apply rawSatisfied row
  unfold rawRows
  apply List.mem_append_left
  apply List.mem_append_right
  exact List.mem_flatMap.mpr
    ⟨slot, List.mem_range.mpr slotBound, member⟩

theorem boundaryRows_satisfied_of_rows
    (assignment : Nat → F) (params : Params)
    (satisfied : Satisfies (rows params) assignment) :
    Satisfies (Compiler.boundaryRows params) assignment := by
  have rawSatisfied := rawRows_satisfied_of_rows assignment params satisfied
  intro row member
  apply rawSatisfied row
  unfold rawRows
  exact List.mem_append_right _ member

private theorem operationProductRows_satisfied
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool)
    (satisfied : Satisfies (operationRows params slot) assignment) :
    Satisfies (operationProductRows params slot write) assignment := by
  intro row member
  apply satisfied row
  cases write <;>
    simp [operationRows, operationCoreRows, member]

private theorem scanRowsForLane_satisfied
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (final : Bool)
    (satisfied : Satisfies (scanRows params slot) assignment) :
    Satisfies (scanRowsForLane params final slot) assignment := by
  intro row member
  apply satisfied row
  cases final <;> simp [scanRows, member]

theorem operationProduct_sound_of_rows
    (assignment : Nat → F) (params : Params)
    (satisfied : Satisfies (rows params) assignment)
    (slot : Nat) (slotBound : slot < params.operationSlots)
    (write : Bool) :
    operationProduct assignment params slot write =
      K.mul (previousOperationProductValue assignment params slot write)
        (operationGate assignment params slot write) := by
  exact operationProductRows_sound assignment params slot write
    (operationProductRows_satisfied assignment params slot write
      (operationRows_satisfied_of_rows assignment params satisfied slot
        slotBound))

theorem scanProduct_sound_of_rows
    (assignment : Nat → F) (params : Params)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows params) assignment)
    (final : Bool) (slot : Nat) (slotBound : slot < params.scanSlots) :
    scanProduct assignment params final slot =
      K.mul (previousScanProductValue assignment params final slot)
        (scanFactor assignment params final slot) := by
  exact scanRowsForLane_sound assignment params final slot constantWire
    (scanRowsForLane_satisfied assignment params slot final
      (scanRows_satisfied_of_rows assignment params satisfied slot
        slotBound))

def inputProduct (assignment : Nat → F) (product : Nat) : K :=
  evaluatePair assignment
    (productInputWord product 0)
    (productInputWord product 1)

def outputProduct (assignment : Nat → F) (product : Nat) : K :=
  evaluatePair assignment
    (productOutputWord product 0)
    (productOutputWord product 1)

def operationRun
    (assignment : Nat → F) (params : Params) (write : Bool) : Nat → K
  | 0 => inputProduct assignment (if write then 1 else 0)
  | count + 1 =>
      K.mul (operationRun assignment params write count)
        (operationGate assignment params count write)

theorem operationProduct_eq_run
    (assignment : Nat → F) (params : Params)
    (satisfied : Satisfies (rows params) assignment)
    (write : Bool) (slot : Nat) (slotBound : slot < params.operationSlots) :
    operationProduct assignment params slot write =
      operationRun assignment params write (slot + 1) := by
  induction slot with
  | zero =>
      rw [operationProduct_sound_of_rows assignment params satisfied 0
        slotBound write]
      cases write <;>
        simp [previousOperationProductValue, previousOperationProduct,
          operationRun, inputProduct]
  | succ previous inductionHypothesis =>
      have previousBound : previous < params.operationSlots :=
        Nat.lt_trans (Nat.lt_succ_self previous) slotBound
      rw [operationProduct_sound_of_rows assignment params satisfied
        (previous + 1) slotBound write]
      rw [previousOperationProductValue_succ,
        inductionHypothesis previousBound]
      rfl

def scanRun
    (assignment : Nat → F) (params : Params) (final : Bool) : Nat → K
  | 0 => inputProduct assignment (if final then 3 else 2)
  | count + 1 =>
      K.mul (scanRun assignment params final count)
        (scanFactor assignment params final count)

private theorem previousScanProductValue_succ
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat) :
    previousScanProductValue assignment params final (slot + 1) =
      scanProduct assignment params final slot := by
  cases final <;>
    rfl

theorem scanProduct_eq_run
    (assignment : Nat → F) (params : Params)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows params) assignment)
    (final : Bool) (slot : Nat) (slotBound : slot < params.scanSlots) :
    scanProduct assignment params final slot =
      scanRun assignment params final (slot + 1) := by
  induction slot with
  | zero =>
      rw [scanProduct_sound_of_rows assignment params constantWire satisfied
        final 0 slotBound]
      cases final <;>
        simp [previousScanProductValue, previousScanProduct,
          scanRun, inputProduct]
  | succ previous inductionHypothesis =>
      have previousBound : previous < params.scanSlots :=
        Nat.lt_trans (Nat.lt_succ_self previous) slotBound
      rw [scanProduct_sound_of_rows assignment params constantWire satisfied
        final (previous + 1) slotBound]
      rw [previousScanProductValue_succ,
        inductionHypothesis previousBound]
      rfl

private theorem linearRow_sound
    (identifier : RowId) (left right : Lin)
    (assignment : Nat → F)
    (holds : (linearRow identifier left right).Holds assignment) :
    Rows.LinearCombination.eval assignment left =
      Rows.LinearCombination.eval assignment right := by
  rw [linearRow_holds_iff] at holds
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp
    (by simpa only [Fin.sub_eq_add_neg] using holds)

def boundarySource
    (params : Params) (product component : Nat) : Lin :=
  if product = 0 then
    operationReadProductWord params (lastOperationSlot params) component
  else if product = 1 then
    operationWriteProductWord params (lastOperationSlot params) component
  else if product = 2 then
    scanProductWord params false (lastScanSlot params) component
  else
    scanProductWord params true (lastScanSlot params) component

private theorem boundaryProductComponent_sound
    (assignment : Nat → F) (params : Params)
    (satisfied : Satisfies (rows params) assignment)
    (product component : Nat)
    (productBound : product < 4) (componentBound : component < 2) :
    Rows.LinearCombination.eval assignment
        (productOutputWord product component) =
      Rows.LinearCombination.eval assignment
        (boundarySource params product component) := by
  have boundarySatisfied :=
    boundaryRows_satisfied_of_rows assignment params satisfied
  have member :
      linearRow (id .boundaryProduct product component 0)
          (productOutputWord product component)
          (boundarySource params product component) ∈
        Compiler.boundaryRows params := by
    apply List.mem_cons_of_mem
    apply List.mem_flatMap.mpr
    refine ⟨product, List.mem_range.mpr productBound, ?_⟩
    apply List.mem_map.mpr
    exact ⟨component, List.mem_range.mpr componentBound, rfl⟩
  exact linearRow_sound _ _ _ assignment
    (boundarySatisfied _ member)

def boundaryProductValue
    (assignment : Nat → F) (params : Params) (product : Nat) : K :=
  evaluatePair assignment
    (boundarySource params product 0)
    (boundarySource params product 1)

theorem boundaryProduct_sound
    (assignment : Nat → F) (params : Params)
    (satisfied : Satisfies (rows params) assignment)
    (product : Nat) (productBound : product < 4) :
    outputProduct assignment product =
      boundaryProductValue assignment params product := by
  change K.mk _ _ = K.mk _ _
  rw [K.mk.injEq]
  exact ⟨
    boundaryProductComponent_sound assignment params satisfied product 0
      productBound (by decide),
    boundaryProductComponent_sound assignment params satisfied product 1
      productBound (by decide)⟩

/-- The complete selected row program computes all four public running
products. This is the step-level soundness statement used by segment
composition; it does not assert terminal balance inside one step. -/
theorem wasm42x6_public_products_sound
    (assignment : Nat → F)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows wasm42x6) assignment) :
    outputProduct assignment 0 =
        operationRun assignment wasm42x6 false 1 ∧
      outputProduct assignment 1 =
        operationRun assignment wasm42x6 true 1 ∧
      outputProduct assignment 2 =
        scanRun assignment wasm42x6 false 1024 ∧
      outputProduct assignment 3 =
        scanRun assignment wasm42x6 true 1024 := by
  have operationRead := operationProduct_eq_run assignment wasm42x6
    satisfied false 0 (by decide)
  have operationWrite := operationProduct_eq_run assignment wasm42x6
    satisfied true 0 (by decide)
  have initialScan := scanProduct_eq_run assignment wasm42x6 constantWire
    satisfied false 1023 (by decide)
  have finalScan := scanProduct_eq_run assignment wasm42x6 constantWire
    satisfied true 1023 (by decide)
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [boundaryProduct_sound assignment wasm42x6 satisfied 0 (by decide)]
    simpa [boundaryProductValue, boundarySource, lastOperationSlot]
      using operationRead
  · rw [boundaryProduct_sound assignment wasm42x6 satisfied 1 (by decide)]
    simpa [boundaryProductValue, boundarySource, lastOperationSlot]
      using operationWrite
  · rw [boundaryProduct_sound assignment wasm42x6 satisfied 2 (by decide)]
    simpa [boundaryProductValue, boundarySource, lastScanSlot]
      using initialScan
  · rw [boundaryProduct_sound assignment wasm42x6 satisfied 3 (by decide)]
    simpa [boundaryProductValue, boundarySource, lastScanSlot]
      using finalScan

end Nightstream.Implementation.Lowering.Nebula.ProductSemantics
