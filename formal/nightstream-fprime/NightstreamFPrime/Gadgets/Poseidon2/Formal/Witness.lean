import NightstreamFPrime.Gadgets.Poseidon2.Support
import NightstreamFPrime.Circuit.WitnessSupport

/-! Child-owned witness read support of the exact Poseidon2 sponge compiler. -/

namespace NightstreamFPrime.Gadgets.Poseidon2.Formal

open NightstreamFPrime.Circuit

@[simp] theorem witnesses_main (interface : Interface) (offset : Nat) :
    witnesses (Circuit.ops (main interface) offset) =
      [WitnessBatch.arithmetic offset
        (Hash.compile offset (interface.input offset)).recipes] := by
  simp [main_ops, opsAt, assertions, witnesses, Op.witnesses]

theorem witnesses_main_readsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (inputSupported : ∀ expression ∈ interface.input offset,
      expression.VarsSatisfy allowed)
    (localSupported : ∀ index,
      index < (Hash.compile offset (interface.input offset)).recipes.length →
      allowed (offset + index)) :
    ∀ batch ∈ witnesses (Circuit.ops (main interface) offset),
      batch.ReadsSatisfy allowed := by
  rw [witnesses_main]
  intro batch member
  rw [List.mem_singleton] at member
  subst batch
  rw [WitnessBatch.readsSatisfy_arithmetic]
  exact (Support.hashCompile_supported offset (interface.input offset) allowed
    inputSupported localSupported).1

end NightstreamFPrime.Gadgets.Poseidon2.Formal
