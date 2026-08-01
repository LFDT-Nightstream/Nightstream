import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaBinding

/-!
Contract: compose the two physical Nebula batches of the 42-times-6 fixture.

Assurance tier: model-level.

The theorem in this file derives the four final public products from two
satisfying row programs, exact source-port bindings, and explicit public
product carry. It then applies the separately proved memory execution to
derive terminal balance.

It does not own an honest assignment, combined CCS placement, F-prime
assembly, Rust, Fiat--Shamir binding, or a collision-probability bound.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaSegment

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceProducts
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaBinding
open Nightstream.Protocol.Nebula.Fingerprint

private theorem k_mul_assoc (left middle right : K) :
    K.mul (K.mul left middle) right =
      K.mul left (K.mul middle right) :=
  extensionLaws.mul_assoc left middle right

private theorem k_one_mul (value : K) : K.mul K.one value = value :=
  extensionLaws.one_mul value

private theorem k_mul_one (value : K) : K.mul value K.one = value :=
  extensionLaws.mul_one value

/-- The public recurrence state carried from the first batch into the second.
The initial products are verifier-owned base values, not witness conclusions. -/
structure Linked
    (first second : Nat -> F) : Prop where
  challengesExact : challenges second = challenges first
  productCarry : forall product : Fin 4,
    inputProduct second product.val = outputProduct first product.val
  initialProducts : forall product : Fin 4,
    inputProduct first product.val = K.one

private theorem first_source_equations
    (first : Nat -> F)
    (ports : FirstPorts first)
    (satisfied : Satisfies (rows wasm42x6) first) :
    outputProduct first 0 =
        K.mul (inputProduct first 0)
          (product (challenges first) [Memory.readCell]) ∧
      outputProduct first 1 =
        K.mul (inputProduct first 1)
          (product (challenges first) [Memory.writeCell]) ∧
      outputProduct first 2 =
        K.mul (inputProduct first 2)
          (product (challenges first) Memory.romChunk) ∧
      outputProduct first 3 =
        K.mul (inputProduct first 3)
          (product (challenges first) Memory.romChunk) := by
  have source := wasm42x6_public_products_source_bound first
    ports.constantWire satisfied (fun _ => true)
    (first_activation first ports)
  have operations := first_operation_entries first ports satisfied
  have scans := first_scan_entries first ports
  simpa [operationEntries_single_active, operations.1, operations.2,
    scans.1, scans.2] using source

private theorem second_source_equations
    (second : Nat -> F)
    (ports : SecondPorts second)
    (satisfied : Satisfies (rows wasm42x6) second) :
    outputProduct second 0 =
        K.mul (inputProduct second 0)
          (product (challenges second) []) ∧
      outputProduct second 1 =
        K.mul (inputProduct second 1)
          (product (challenges second) []) ∧
      outputProduct second 2 =
        K.mul (inputProduct second 2)
          (product (challenges second) Memory.initialRamChunk) ∧
      outputProduct second 3 =
        K.mul (inputProduct second 3)
          (product (challenges second) Memory.finalRamChunk) := by
  have source := wasm42x6_public_products_source_bound second
    ports.constantWire satisfied (fun _ => false)
    (second_activation second ports)
  have scans := second_scan_entries second ports
  simpa [operationEntries_single_inactive, scans.1, scans.2] using source

/-- The final public product vector is exactly the product vector of the
authoritative benchmark memory execution. -/
theorem final_products_eq_protocol
    (first second : Nat -> F)
    (firstPorts : FirstPorts first)
    (secondPorts : SecondPorts second)
    (firstSatisfied : Satisfies (rows wasm42x6) first)
    (secondSatisfied : Satisfies (rows wasm42x6) second)
    (linked : Linked first second) :
    forall productIndex : Fin 4,
      outputProduct second productIndex.val =
        Nightstream.Protocol.Nebula.Memory.products
          (challenges first) Memory.initialSnapshot
          [Memory.access] Memory.finalSnapshot productIndex := by
  have firstSource := first_source_equations first firstPorts firstSatisfied
  have secondSource := second_source_equations second secondPorts secondSatisfied
  rw [linked.challengesExact] at secondSource
  intro productIndex
  have alternatives :
      productIndex.val = 0 ∨ productIndex.val = 1 ∨
        productIndex.val = 2 ∨ productIndex.val = 3 := by
    omega
  rcases alternatives with value | value | value | value
  · have indexExact : productIndex = ⟨0, by decide⟩ := Fin.ext value
    subst productIndex
    calc
      outputProduct second 0 =
          K.mul (inputProduct second 0)
            (product (challenges first) []) := secondSource.1
      _ = outputProduct first 0 := by
        rw [linked.productCarry ⟨0, by decide⟩]
        exact k_mul_one _
      _ = K.mul K.one
          (product (challenges first) [Memory.readCell]) := by
        rw [firstSource.1, linked.initialProducts ⟨0, by decide⟩]
      _ = product (challenges first) [Memory.readCell] := k_one_mul _
      _ = Nightstream.Protocol.Nebula.Memory.products
          (challenges first) Memory.initialSnapshot
          [Memory.access] Memory.finalSnapshot ⟨0, by decide⟩ := by
        rfl
  · have indexExact : productIndex = ⟨1, by decide⟩ := Fin.ext value
    subst productIndex
    calc
      outputProduct second 1 =
          K.mul (inputProduct second 1)
            (product (challenges first) []) := secondSource.2.1
      _ = outputProduct first 1 := by
        rw [linked.productCarry ⟨1, by decide⟩]
        exact k_mul_one _
      _ = K.mul K.one
          (product (challenges first) [Memory.writeCell]) := by
        rw [firstSource.2.1, linked.initialProducts ⟨1, by decide⟩]
      _ = product (challenges first) [Memory.writeCell] := k_one_mul _
      _ = Nightstream.Protocol.Nebula.Memory.products
          (challenges first) Memory.initialSnapshot
          [Memory.access] Memory.finalSnapshot ⟨1, by decide⟩ := by
        rfl
  · have indexExact : productIndex = ⟨2, by decide⟩ := Fin.ext value
    subst productIndex
    calc
      outputProduct second 2 =
          K.mul (inputProduct second 2)
            (product (challenges first) Memory.initialRamChunk) :=
        secondSource.2.2.1
      _ = K.mul (outputProduct first 2)
          (product (challenges first) Memory.initialRamChunk) := by
        rw [linked.productCarry ⟨2, by decide⟩]
      _ = K.mul
          (K.mul K.one (product (challenges first) Memory.romChunk))
          (product (challenges first) Memory.initialRamChunk) := by
        rw [firstSource.2.2.1, linked.initialProducts ⟨2, by decide⟩]
      _ = K.mul K.one
          (K.mul (product (challenges first) Memory.romChunk)
            (product (challenges first) Memory.initialRamChunk)) :=
        k_mul_assoc _ _ _
      _ = product (challenges first) Memory.initialSnapshot := by
        rw [k_one_mul, ← product_append]
        rfl
      _ = Nightstream.Protocol.Nebula.Memory.products
          (challenges first) Memory.initialSnapshot
          [Memory.access] Memory.finalSnapshot ⟨2, by decide⟩ := by
        rfl
  · have indexExact : productIndex = ⟨3, by decide⟩ := Fin.ext value
    subst productIndex
    calc
      outputProduct second 3 =
          K.mul (inputProduct second 3)
            (product (challenges first) Memory.finalRamChunk) :=
        secondSource.2.2.2
      _ = K.mul (outputProduct first 3)
          (product (challenges first) Memory.finalRamChunk) := by
        rw [linked.productCarry ⟨3, by decide⟩]
      _ = K.mul
          (K.mul K.one (product (challenges first) Memory.romChunk))
          (product (challenges first) Memory.finalRamChunk) := by
        rw [firstSource.2.2.2, linked.initialProducts ⟨3, by decide⟩]
      _ = K.mul K.one
          (K.mul (product (challenges first) Memory.romChunk)
            (product (challenges first) Memory.finalRamChunk)) :=
        k_mul_assoc _ _ _
      _ = product (challenges first) Memory.finalSnapshot := by
        rw [k_one_mul, ← product_append]
        rfl
      _ = Nightstream.Protocol.Nebula.Memory.products
          (challenges first) Memory.initialSnapshot
          [Memory.access] Memory.finalSnapshot ⟨3, by decide⟩ := by
        rfl

/-- Terminal balance is derived from physical row satisfaction and the exact
benchmark execution. It is not a prover-carried acceptance flag. -/
theorem final_products_balanced
    (first second : Nat -> F)
    (firstPorts : FirstPorts first)
    (secondPorts : SecondPorts second)
    (firstSatisfied : Satisfies (rows wasm42x6) first)
    (secondSatisfied : Satisfies (rows wasm42x6) second)
    (linked : Linked first second) :
    Nightstream.Protocol.Nebula.Memory.Balanced
      (fun productIndex => outputProduct second productIndex.val) := by
  have exactProducts := final_products_eq_protocol first second firstPorts
    secondPorts firstSatisfied secondSatisfied linked
  unfold Nightstream.Protocol.Nebula.Memory.Balanced
  change
    K.mul (outputProduct second 2) (outputProduct second 1) =
      K.mul (outputProduct second 0) (outputProduct second 3)
  rw [exactProducts ⟨2, by decide⟩, exactProducts ⟨1, by decide⟩,
    exactProducts ⟨0, by decide⟩, exactProducts ⟨3, by decide⟩]
  exact Memory.balanced (challenges first)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaSegment
