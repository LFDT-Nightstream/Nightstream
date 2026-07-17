import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.Exactness

/-!
Independent countermodels for the three retained aggregate-acceptance families.

Owns: one concrete invalid transition admitted when each of output bitness,
the weighted aggregate, or final root binding is removed.

Does not own: production placement, row removal authorization, generated
artifacts, Rust conformance, or any claim that these are production rows.

Emits constraints: no.

| Exact Rust stage path omitted | Other families retained | Invalid behavior |
|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.tree_bit_pairs` | aggregate and root binding | modular cancellation fakes the aggregate |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.product_aggregate` | output bitness and root binding | a Boolean false tree edge survives |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.root_binding` | output bitness and correct tree | the wrong accept bit survives |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance

open Nightstream.Implementation.R1CS
open Mod5

/-! ## Independent per-family necessity countermodels -/

private def zeroSourceBits : Fin 16 → GateField := fun _ => 0

/-- Non-Boolean outputs whose base-three values differ by exactly three
Goldilocks moduli. The surviving aggregate is therefore a genuine modular
cancellation countermodel. -/
private def outputBitRowsCounterexample : ProductTreeOutputs
  | ⟨0, _⟩ => fieldResidue 3
  | ⟨1, _⟩ => fieldResidue (goldilocksP - 1)
  | ⟨4, _⟩ => fieldResidue (goldilocksP - 3)
  | _ => 0

theorem productTreeOutputBitRows_are_necessary
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue) :
    ∃ bits : Fin 16 → GateField,
      ∃ outputs : ProductTreeOutputs,
        ∃ accept : GateField,
          (∀ index, FieldBit (bits index)) ∧
            ProductTreeAggregateRow bits outputs ∧
            FinalAcceptanceRow outputs accept ∧
            ¬ ProductTreeOutputBitRows outputs ∧
            ¬ ProductTreeMeaning bits outputs := by
  refine ⟨zeroSourceBits, outputBitRowsCounterexample, 1, ?_, ?_, ?_, ?_, ?_⟩
  · intro index
    exact Or.inl rfl
  · unfold ProductTreeAggregateRow
    native_decide
  · unfold FinalAcceptanceRow
    native_decide
  · intro outputRows
    have outputBoolean :=
      (productTreeOutputBitRows_iff prime nonresidue
        outputBitRowsCounterexample).mp outputRows
    have outputZeroNotBit : ¬ FieldBit (outputBitRowsCounterexample 0) := by
      unfold FieldBit
      native_decide
    exact outputZeroNotBit (outputBoolean 0)
  · intro meaning
    have first := meaning (0 : Fin 14)
    have firstFalse :
        outputBitRowsCounterexample 0 ≠
          productTreeLeft zeroSourceBits outputBitRowsCounterexample 0 *
            productTreeRight zeroSourceBits outputBitRowsCounterexample 0 := by
      native_decide
    exact firstFalse first

/-- Boolean but false first edge used when the aggregate family is absent. -/
private def aggregateRowCounterexample : ProductTreeOutputs
  | ⟨0, _⟩ => 1
  | _ => 0

theorem productTreeAggregateRow_is_necessary
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue) :
    ∃ bits : Fin 16 → GateField,
      ∃ outputs : ProductTreeOutputs,
        ∃ accept : GateField,
          (∀ index, FieldBit (bits index)) ∧
            ProductTreeOutputBitRows outputs ∧
            FinalAcceptanceRow outputs accept ∧
            ¬ ProductTreeAggregateRow bits outputs ∧
            ¬ ProductTreeMeaning bits outputs := by
  refine ⟨zeroSourceBits, aggregateRowCounterexample, 1, ?_, ?_, ?_, ?_, ?_⟩
  · intro index
    exact Or.inl rfl
  · apply (productTreeOutputBitRows_iff prime nonresidue
      aggregateRowCounterexample).mpr
    exact fin14_all (predicate := fun index =>
      FieldBit (aggregateRowCounterexample index))
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
      (by unfold FieldBit; native_decide)
  · unfold FinalAcceptanceRow
    native_decide
  · unfold ProductTreeAggregateRow
    native_decide
  · intro meaning
    have first := meaning (0 : Fin 14)
    have firstFalse :
        aggregateRowCounterexample 0 ≠
          productTreeLeft zeroSourceBits aggregateRowCounterexample 0 *
            productTreeRight zeroSourceBits aggregateRowCounterexample 0 := by
      native_decide
    exact firstFalse first

theorem finalAcceptanceRow_is_necessary
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue) :
    ∃ bits : Fin 16 → GateField,
      ∃ outputs : ProductTreeOutputs,
        ∃ accept : GateField,
          (∀ index, FieldBit (bits index)) ∧
            ProductTreeOutputBitRows outputs ∧
            ProductTreeAggregateRow bits outputs ∧
            ProductTreeMeaning bits outputs ∧
            ¬ FinalAcceptanceRow outputs accept ∧
            ¬ SourceAcceptanceMeaning bits accept := by
  refine ⟨zeroSourceBits, fun _ => 0, 0, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro index
    exact Or.inl rfl
  · apply (productTreeOutputBitRows_iff prime nonresidue (fun _ => 0)).mpr
    intro index
    exact Or.inl rfl
  · unfold ProductTreeAggregateRow
    native_decide
  · exact (productTreeMeaning_iff_equations zeroSourceBits (fun _ => 0)).mpr
      (by unfold ProductTreeEquations; native_decide)
  · unfold FinalAcceptanceRow
    native_decide
  · unfold SourceAcceptanceMeaning
    native_decide

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance
