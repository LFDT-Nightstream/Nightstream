import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerWords
import NightstreamFPrime.Circuit.WitnessSupport

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerWords

open NightstreamFPrime.Circuit NightstreamFPrime.Gadgets.Sampling

theorem witnesses_main_below (rangeOffset offset : Nat) :
    ∀ batch ∈ witnesses (Circuit.ops (circuit rangeOffset).main offset),
      batch.ReadsSatisfy (fun column => column < rangeOffset + WideReduction.Program.privateCount) := by
  intro batch member
  change batch ∈ witnesses (operations rangeOffset offset) at member
  simp only [operations, witnesses, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, Op.witnesses, List.mem_singleton] at member
  subst batch
  rw [WitnessBatch.readsSatisfy_arithmetic]
  intro expression member
  obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
  exact (Expr.varsSatisfy_lt_iff_varsBelow _ _).mpr (recipe_below rangeOffset index)

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerWords
