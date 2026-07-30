import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: closed arithmetic for the exact 42-times-6 benchmark Step cost.

Assurance tier: model-level.

Owns: only the small fifteen-item cost fold after every physical component
has been proved equal to its exact `Cost`.

Does not own: rows, columns, recipes, setup data, Rust, or any protocol
semantics.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CostArithmetic

open Nightstream.Implementation.Lowering.Typed

/-- Exact costs of the fifteen non-input Step receipts in typed program
order. -/
def bodyCosts : List Cost :=
  [ ⟨11, 7, 0, 4⟩,
    ⟨3, 0, 0, 3⟩,
    ⟨1, 0, 0, 1⟩,
    ⟨1, 0, 0, 1⟩,
    ⟨21, 0, 0, 21⟩,
    ⟨1, 0, 0, 0⟩,
    ⟨40_440, 40_440, 0, 0⟩,
    ⟨2503, 0, 0, 2499⟩,
    ⟨5, 0, 0, 5⟩,
    ⟨5, 0, 0, 5⟩,
    ⟨15, 0, 0, 15⟩,
    ⟨1, 0, 0, 0⟩,
    ⟨19_773_612, 40_440, 0, 19_720_201⟩,
    ⟨40_440, 40_440, 0, 0⟩,
    ⟨2503, 0, 5, 2494⟩ ]

/-- Closed arithmetic after the public one column, exact input allocation,
and exact body component costs are known. -/
theorem total :
    Cost.oneColumn .publicColumn +
        ⟨0, 122_731, 0, 0⟩ +
      Cost.sum bodyCosts =
        ⟨19_859_562, 244_058, 6, 19_725_249⟩ := by
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CostArithmetic
