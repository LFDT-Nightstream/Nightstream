/-!
Model-level exactness of the first-accepted selection product substitution.

Owns: the existential source product columns, their three binding equations,
the three aggregate equations obtained by substitution, and the bidirectional
equivalence between those relations.

Does not own: one-hotness, sampler semantics, concrete R1CS rows, gadget-native
matrix roles, generated artifacts, Rust trace validation, or row-removal
authorization.

Emits constraints: no.

Authority boundary: this theorem justifies the algebraic substitution. A
separate artifact refinement must prove that production emits these aggregate
equations over the intended decoded columns.

| Predicate/theorem | Mathematical obligation | Assurance tier | Open boundary |
|---|---|---|---|
| `SelectionProductDefinitions` | each temporary equals selector times source | model-level | concrete source-row decoding |
| `CurrentSelectionBlock` | three products per index and three bindings; fixed profile `33 + 3` | model-level | fixed-width instantiation |
| `AggregateSelectionBlock` | three product-sum equations after substitution | model-level | gadget-native gate refinement |
| `currentSelectionBlock_iff_aggregate` | product temporaries are exactly eliminable | model-level | production row ownership and removal |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection

universe u v

/-- Fixed-order sum over a finite production window. The substitution theorem
does not require commutativity or any other algebraic law. -/
def finiteSum {count : Nat} {value : Type v}
    [Add value] [OfNat value 0]
    (values : Fin count → value) : value :=
  (List.ofFn values).foldl (fun total value => total + value) 0

private theorem finiteSum_congr
    {count : Nat} {value : Type v}
    [Add value] [OfNat value 0]
    {left right : Fin count → value}
    (same : ∀ index, left index = right index) :
    finiteSum left = finiteSum right := by
  have functionsEqual : left = right := funext same
  rw [functionsEqual]

/-- The three families of multiplication temporaries in the source block. -/
structure SelectionProducts (index : Type u) (value : Type v) where
  accepted : index → value
  prefixWeighted : index → value
  symbol : index → value

/-- Every source product column equals its selector/source product. -/
def SelectionProductDefinitions {index : Type u} {value : Type v} [Mul value]
    (selectors accepts prefixes symbols : index → value)
    (products : SelectionProducts index value) : Prop :=
  ∀ candidate,
    products.accepted candidate = selectors candidate * accepts candidate ∧
      products.prefixWeighted candidate =
        selectors candidate * prefixes candidate ∧
      products.symbol candidate = selectors candidate * symbols candidate

/-- Source form: existential product columns followed by three bindings. -/
def CurrentSelectionBlock {count : Nat} {value : Type v}
    [Add value] [OfNat value 0] [Mul value] [OfNat value 1]
    (selectors accepts prefixes symbols : Fin count → value)
    (position output : value) : Prop :=
  ∃ products : SelectionProducts (Fin count) value,
    SelectionProductDefinitions selectors accepts prefixes symbols products ∧
      finiteSum products.accepted = 1 ∧
      finiteSum products.prefixWeighted = position ∧
      output = finiteSum products.symbol

/-- Lowered form: the three bindings after substituting product definitions. -/
def AggregateSelectionBlock {count : Nat} {value : Type v}
    [Add value] [OfNat value 0] [Mul value] [OfNat value 1]
    (selectors accepts prefixes symbols : Fin count → value)
    (position output : value) : Prop :=
  finiteSum (fun candidate => selectors candidate * accepts candidate) = 1 ∧
    finiteSum (fun candidate => selectors candidate * prefixes candidate) = position ∧
    output = finiteSum (fun candidate => selectors candidate * symbols candidate)

/-- Eliminating all product temporaries preserves and reflects acceptance.
The reverse direction constructs the exact source-product extension. -/
theorem currentSelectionBlock_iff_aggregate
    {count : Nat} {value : Type v}
    [Add value] [OfNat value 0] [Mul value] [OfNat value 1]
    (selectors accepts prefixes symbols : Fin count → value)
    (position output : value) :
    CurrentSelectionBlock selectors accepts prefixes symbols position output ↔
      AggregateSelectionBlock selectors accepts prefixes symbols position output := by
  constructor
  · rintro ⟨products, definitions, acceptedBinding, prefixBinding, symbolBinding⟩
    refine ⟨?_, ?_, ?_⟩
    · calc
        finiteSum (fun candidate => selectors candidate * accepts candidate) =
            finiteSum products.accepted := by
          apply finiteSum_congr
          intro candidate
          exact (definitions candidate).1.symm
        _ = 1 := acceptedBinding
    · calc
        finiteSum (fun candidate => selectors candidate * prefixes candidate) =
            finiteSum products.prefixWeighted := by
          apply finiteSum_congr
          intro candidate
          exact (definitions candidate).2.1.symm
        _ = position := prefixBinding
    · calc
        output = finiteSum products.symbol := symbolBinding
        _ = finiteSum (fun candidate => selectors candidate * symbols candidate) := by
          apply finiteSum_congr
          intro candidate
          exact (definitions candidate).2.2
  · rintro ⟨acceptedBinding, prefixBinding, symbolBinding⟩
    let products : SelectionProducts (Fin count) value :=
      { accepted := fun candidate => selectors candidate * accepts candidate
        prefixWeighted := fun candidate => selectors candidate * prefixes candidate
        symbol := fun candidate => selectors candidate * symbols candidate }
    refine ⟨products, ?_, ?_, ?_, ?_⟩
    · intro candidate
      exact ⟨rfl, rfl, rfl⟩
    · simpa [products] using acceptedBinding
    · simpa [products] using prefixBinding
    · simpa [products] using symbolBinding

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection
