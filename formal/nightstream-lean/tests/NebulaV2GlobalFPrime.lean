import Nightstream.Protocol.NebulaV2.GlobalFPrime

set_option autoImplicit false

namespace tests.NebulaV2GlobalFPrime

open Nightstream.Protocol.NebulaV2.GlobalFPrime
open Nightstream.Protocol.NebulaV2.Lifecycle

/-- The schedule uses complete claim values. It cannot replace one claim by a
different claim that happens to have the same memory suffix. -/
theorem recursive_view_uses_exact_list_elements
    (first second : Nat) :
    consumedClaimAt [first, second]
        (⟨1, by decide⟩ : InvocationIndex 2) = some first ∧
      producedClaimAt [first, second]
        (⟨1, by decide⟩ : InvocationIndex 2) = some second := by
  constructor <;> rfl

theorem terminal_view_consumes_second
    (first second : Nat) :
    consumedClaimAt [first, second] (terminalIndex 2) = some second ∧
      producedClaimAt [first, second] (terminalIndex 2) = none := by
  constructor <;> rfl

end tests.NebulaV2GlobalFPrime
