import Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-!
Focused regressions for the independent Split-NC `yZcol` terminal necessity
witnesses.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.terminal.necessity.scalar` | changing only assignment-two `yZcol` to zero flips the terminal check | omitted scalar terminal equality |
| `nifs.pi_ccs.nc.terminal.necessity.binding` | forged unit passes the cubic terminal but is not source-bound | treating a cubic image as output authority |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal.Tests

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal

/-- The two-message witness proves inclusion-necessity of the scalar check. -/
example :
    ∃ honest forged,
      honest.yRing = forged.yRing ∧
      honest.yZcol ≠ forged.yZcol ∧
      ScalarTerminalCheck dataTwo honest ∧
      ¬ ScalarTerminalCheck dataTwo forged :=
  scalarTerminalCheck_is_necessary

/-- The adaptive unit forgery proves scalar terminal equality insufficient. -/
example :
    ∃ message,
      ScalarTerminalCheck dataZero message ∧
      ¬ Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.YZcolBoundToSources
        witnessCovers dataZero (verifierPoints dataZero) message :=
  scalarTerminalCheck_is_insufficient

/-- The insufficiency witness passes only because zero and one share the
strict cubic's zero image; it still fails the independent binding predicate. -/
example : ScalarTerminalCheck dataZero forgedOne ∧
    ¬ Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.YZcolBoundToSources
      witnessCovers dataZero (verifierPoints dataZero) forgedOne :=
  ⟨forgedOne_scalarTerminalCheck, forgedOne_not_yZcolBoundToSources⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal.Tests
