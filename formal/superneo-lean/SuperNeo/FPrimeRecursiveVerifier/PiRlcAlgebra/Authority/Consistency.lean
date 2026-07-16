import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Parameters

/-!
Owns: the authority reduction from repeated `s_col`/fold-digest comparisons to
one parent-to-common-value comparison.

Does not own: proof that Pi_CCS outputs share that common value or concrete
row deletion.

Emits constraints: no. It compares the current and candidate authority
relations.

Authority boundary: `CommonInputAuthority` is an explicit upstream premise;
without it the row reduction theorem does not apply.

Rust emits both equality families in `pi_rlc_circuit/consistency.rs`; their
NIFS orchestration call site is `nifs/circuit/pi_rlc/consistency.rs`.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `CommonInputAuthority` | `consistency` | Every Pi_CCS output carries one reconstructed common value | Concrete Pi_CCS composition premise | No — Rust refinement open |
| `RepeatedAuthorityBindings`, `ParentAuthorityBinding` | `consistency.s_col`, `consistency.fold_digest` | States the repeated source bindings and single parent target | Shared value carrier | No — Rust refinement open |
| `repeatedAuthorityBindings_iff_parentBinding` | `consistency` | Reduces repeated comparisons to one parent comparison | `CommonInputAuthority` | No — concrete Pi_CCS/Rust refinement open |

This theorem does not assert that Pi_CCS supplies the premise; the concrete
composition refinement must prove that separately before deleting rows.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

universe u

def CommonInputAuthority {Value : Type u}
    (inputs : Fin inputCount → Value) (common : Value) : Prop :=
  ∀ inputIndex, inputs inputIndex = common

def RepeatedAuthorityBindings {Value : Type u}
    (inputs : Fin inputCount → Value) (parent : Value) : Prop :=
  ∀ inputIndex, inputs inputIndex = parent

def ParentAuthorityBinding {Value : Type u}
    (parent common : Value) : Prop :=
  parent = common

/-- Under one upstream common value, all repeated comparisons reduce to one. -/
theorem repeatedAuthorityBindings_iff_parentBinding
    {Value : Type u}
    (inputs : Fin inputCount → Value)
    (parent common : Value)
    (hCommon : CommonInputAuthority inputs common) :
    RepeatedAuthorityBindings inputs parent ↔
      ParentAuthorityBinding parent common := by
  constructor
  · intro hRepeated
    let first : Fin inputCount := ⟨0, by decide⟩
    unfold ParentAuthorityBinding
    calc
      parent = inputs first := (hRepeated first).symm
      _ = common := hCommon first
  · intro hParent inputIndex
    calc
      inputs inputIndex = common := hCommon inputIndex
      _ = parent := hParent.symm

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
