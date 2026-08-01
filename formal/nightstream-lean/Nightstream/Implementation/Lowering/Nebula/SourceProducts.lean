import Nightstream.Implementation.Lowering.Nebula.SourceSemantics

/-!
Exact list-product refinement for the Lean-owned Nebula memory compiler.

Assurance tier: model-level.

Owns the connection from the physical operation and scan recurrences to
protocol `MemTuple` list products. Operation activation is explicit: a caller
states which slots are active and proves that the physical pad field agrees.

Does not own bit decoding, WASM port binding, segment composition, terminal
balance, challenge security, Rust, or a collision probability bound.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.Nebula.SourceProducts

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceSemantics
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

private theorem k_mul_assoc (left middle right : K) :
    K.mul (K.mul left middle) right =
      K.mul left (K.mul middle right) :=
  extensionLaws.mul_assoc left middle right

private theorem k_mul_one (value : K) : K.mul value K.one = value :=
  extensionLaws.mul_one value

private theorem product_singleton
    (challengeValues : Challenges) (entry : MemTuple) :
    product challengeValues [entry] = fingerprint challengeValues entry := by
  simp only [product]
  exact k_mul_one _

/-- Ordered scan tuples processed by the first `count` slots. -/
def scanEntries
    (assignment : Nat -> F) (params : Params) (final : Bool) :
    Nat -> List MemTuple
  | 0 => []
  | count + 1 =>
      scanEntries assignment params final count ++
        [scanEntry assignment params final count]

theorem scanEntries_eq_map_range
    (assignment : Nat -> F) (params : Params) (final : Bool) :
    forall count,
      scanEntries assignment params final count =
        (List.range count).map
          (scanEntry assignment params final) := by
  intro count
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [scanEntries, inductionHypothesis, List.range_succ, List.map_append]
      rfl

/-- Ordered active operation tuples. `activeAt` is true exactly for non-pad
slots. -/
def operationEntries
    (assignment : Nat -> F) (params : Params) (write : Bool)
    (activeAt : Nat -> Bool) : Nat -> List MemTuple
  | 0 => []
  | count + 1 =>
      operationEntries assignment params write activeAt count ++
        if activeAt count then
          [operationEntry assignment params count write]
        else []

theorem operationEntries_single_active
    (assignment : Nat -> F) (params : Params) (write : Bool) :
    operationEntries assignment params write (fun _ => true) 1 =
      [operationEntry assignment params 0 write] := by
  rfl

theorem operationEntries_single_inactive
    (assignment : Nat -> F) (params : Params) (write : Bool) :
    operationEntries assignment params write (fun _ => false) 1 = [] := by
  rfl

/-- Exact agreement between a logical active-slot pattern and the physical
pad field. Active slots carry pad zero; inactive slots carry pad one. -/
def ActivationMatches
    (assignment : Nat -> F) (params : Params)
    (activeAt : Nat -> Bool) (count : Nat) : Prop :=
  forall slot, slot < count ->
    fieldValue assignment (operationPad params slot) =
      if activeAt slot then 0 else 1

theorem operationGate_eq_fingerprint_of_active
    (assignment : Nat -> F) (params : Params)
    (slot : Nat) (write : Bool)
    (constantWire : assignment 0 = 1)
    (padZero : fieldValue assignment (operationPad params slot) = 0) :
    operationGate assignment params slot write =
      fingerprint (challenges assignment)
        (operationEntry assignment params slot write) := by
  rw [show operationGate assignment params slot write =
      gatedFactor assignment
        (operationPad params slot)
        (Rows.LinearCombination.sub one (operationPad params slot))
        (operationFingerprintPrefix params slot write)
        (gammaWord 1 1) (gammaWord 0 0) (gammaWord 0 1)
        (if write then operationWriteValue params slot
          else operationReadValue params slot) from rfl]
  rw [show fingerprint (challenges assignment)
      (operationEntry assignment params slot write) =
        operationFactor assignment params slot write from
      (operationFactor_eq_fingerprint assignment params slot write).symm]
  simp only [gatedFactor, operationFactor, fingerprintFactor,
    Rows.LinearCombination.eval_sub,
    Rows.LinearCombination.eval_constant, one, fieldValue] at padZero ⊢
  rw [constantWire, Fin.one_mul, padZero]
  simp [K.add, K.mul, K.embed, Fin.zero_mul, Fin.mul_zero,
    Fin.zero_add, Fin.one_mul,
    Lean.Grind.AddCommGroup.neg_zero, Fin.add_zero]

theorem operationGate_eq_one_of_inactive
    (assignment : Nat -> F) (params : Params)
    (slot : Nat) (write : Bool)
    (constantWire : assignment 0 = 1)
    (padOne : fieldValue assignment (operationPad params slot) = 1) :
    operationGate assignment params slot write = K.one := by
  unfold operationGate gatedFactor
  simp only [Rows.LinearCombination.eval_sub,
    Rows.LinearCombination.eval_constant, one, fieldValue] at padOne ⊢
  rw [constantWire, Fin.one_mul, padOne]
  rw [show (1 : F) + -1 = 0 from
    Lean.Grind.AddCommGroup.add_neg_cancel 1]
  simp [K.add, K.mul, K.embed, K.one, Fin.mul_zero,
    Fin.zero_mul, Fin.add_zero]

/-- The scan recurrence is the prior public product multiplied by the exact
protocol fingerprints of the decoded scan tuples. -/
theorem scanRun_eq_input_mul_product
    (assignment : Nat -> F) (params : Params)
    (final : Bool) (count : Nat) :
    scanRun assignment params final count =
      K.mul (inputProduct assignment (if final then 3 else 2))
        (product (challenges assignment)
          (scanEntries assignment params final count)) := by
  induction count with
  | zero =>
      simp only [scanRun, scanEntries, product]
      exact (k_mul_one _).symm
  | succ count inductionHypothesis =>
      rw [scanRun, inductionHypothesis, scanEntries,
        product_append, product_singleton,
        scanFactor_eq_fingerprint]
      exact k_mul_assoc _ _ _

/-- The operation recurrence has the same exact list-product meaning once the
logical active-slot pattern is tied to the physical pad fields. -/
theorem operationRun_eq_input_mul_product
    (assignment : Nat -> F) (params : Params)
    (write : Bool) (activeAt : Nat -> Bool) (count : Nat)
    (constantWire : assignment 0 = 1)
    (activation : ActivationMatches assignment params activeAt count) :
    operationRun assignment params write count =
      K.mul (inputProduct assignment (if write then 1 else 0))
        (product (challenges assignment)
          (operationEntries assignment params write activeAt count)) := by
  induction count with
  | zero =>
      simp only [operationRun, operationEntries, product]
      exact (k_mul_one _).symm
  | succ count inductionHypothesis =>
      have priorActivation :
          ActivationMatches assignment params activeAt count := by
        intro slot slotBound
        exact activation slot (Nat.lt_succ_of_lt slotBound)
      rw [operationRun, inductionHypothesis priorActivation, operationEntries]
      cases activeExact : activeAt count
      · have padOne :
            fieldValue assignment (operationPad params count) = 1 := by
          simpa [activeExact] using activation count (Nat.lt_succ_self count)
        rw [operationGate_eq_one_of_inactive assignment params count write
          constantWire padOne]
        simp only [Bool.false_eq_true, ↓reduceIte,
          List.append_nil]
        exact k_mul_one _
      · have padZero :
            fieldValue assignment (operationPad params count) = 0 := by
          simpa [activeExact] using activation count (Nat.lt_succ_self count)
        rw [operationGate_eq_fingerprint_of_active assignment params count
          write constantWire padZero]
        simp only [↓reduceIte]
        rw [product_append, product_singleton]
        exact k_mul_assoc _ _ _

/-- The complete selected row program exposes four public products with exact
source-tuple meaning. The only additional premise is the active-slot pattern;
later application binding derives it from the benchmark phase. -/
theorem wasm42x6_public_products_source_bound
    (assignment : Nat -> F)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows wasm42x6) assignment)
    (activeAt : Nat -> Bool)
    (activation : ActivationMatches assignment wasm42x6 activeAt 1) :
    outputProduct assignment 0 =
        K.mul (inputProduct assignment 0)
          (product (challenges assignment)
            (operationEntries assignment wasm42x6 false activeAt 1)) ∧
      outputProduct assignment 1 =
        K.mul (inputProduct assignment 1)
          (product (challenges assignment)
            (operationEntries assignment wasm42x6 true activeAt 1)) ∧
      outputProduct assignment 2 =
        K.mul (inputProduct assignment 2)
          (product (challenges assignment)
            (scanEntries assignment wasm42x6 false 1024)) ∧
      outputProduct assignment 3 =
        K.mul (inputProduct assignment 3)
          (product (challenges assignment)
            (scanEntries assignment wasm42x6 true 1024)) := by
  obtain ⟨readRun, writeRun, initialRun, finalRun⟩ :=
    wasm42x6_public_products_sound assignment constantWire satisfied
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [readRun]
    exact operationRun_eq_input_mul_product assignment wasm42x6 false
      activeAt 1 constantWire activation
  · rw [writeRun]
    exact operationRun_eq_input_mul_product assignment wasm42x6 true
      activeAt 1 constantWire activation
  · rw [initialRun]
    exact scanRun_eq_input_mul_product assignment wasm42x6 false 1024
  · rw [finalRun]
    exact scanRun_eq_input_mul_product assignment wasm42x6 true 1024

end Nightstream.Implementation.Lowering.Nebula.SourceProducts
