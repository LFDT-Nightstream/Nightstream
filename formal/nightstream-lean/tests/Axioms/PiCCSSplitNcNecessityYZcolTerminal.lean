import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal
import tests.Axioms.Support

/-! Fail-closed dependency gate for the Split-NC `yZcol` terminal necessity witnesses. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal.scalarTerminalCheck_is_necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal.scalarTerminalCheck_is_necessary

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal.scalarTerminalCheck_is_insufficient' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal.scalarTerminalCheck_is_insufficient
