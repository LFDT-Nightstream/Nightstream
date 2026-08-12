import Nightstream.Protocol.NebulaV2.Terminal

/-! Focused gates for the independent V2 terminal semantics. -/

set_option autoImplicit false

namespace tests.NebulaV2Terminal

open Nightstream.Protocol.NebulaV2.Terminal
open Nightstream.Protocol.NebulaV2.CommitmentBundle

#check Accepted.consumes_exact_verified_trailing_claim
#check Accepted.common_witness
#check Accepted.trailing_claim_closes_segment

example : OpensSeparately Countermodels.selectorMap Countermodels.allTrue :=
  Countermodels.separate_openings_exist

example : ¬ HasCommonOpening Countermodels.selectorMap Countermodels.allTrue :=
  Countermodels.no_common_selector_opening

example :
    ¬ ∃ witness,
      Countermodels.booleanBundle witness =
          Countermodels.booleanBundle false ∧
        Countermodels.terminalTrue () witness :=
  Countermodels.no_common_opening_and_terminal_witness

example :
    (∃ assignment,
      ∀ component,
        Countermodels.selectorMap component assignment =
          Countermodels.foldedChildBundle ⟨0, by decide⟩ component) ∧
      ¬ ∃ assignments : FoldedChild → Component,
        ∀ child component,
          Countermodels.selectorMap component (assignments child) =
            Countermodels.foldedChildBundle child component :=
  Countermodels.checking_only_first_folded_child_is_insufficient

end tests.NebulaV2Terminal
