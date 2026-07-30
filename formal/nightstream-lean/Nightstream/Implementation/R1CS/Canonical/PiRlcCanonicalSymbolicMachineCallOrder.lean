import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineHonest

/-!
Contract: call-order closure for the fixed-active `Pi_RLC` transcript.

The result is independent of the absorbed lane expressions.  It supplies the
exact structural premise needed to prove that every declared transcript
temporary is emitted.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineCallOrder

open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical

theorem appendRawPair
    (base first second : Nat) (builder : SymbolicDuplex.Builder)
    (ordered : CallOrdered builder) :
    CallOrdered
      (PiRlcCanonicalSymbolicMachine.appendRawPair
        base first second builder) := by
  unfold PiRlcCanonicalSymbolicMachine.appendRawPair
  exact callOrdered_absorbMany base _ builder ordered

theorem enterScalar
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (ordered : CallOrdered builder) :
    CallOrdered
      (PiRlcCanonicalSymbolicMachine.enterScalar
        base builder coordinate) :=
  appendRawPair base 0 coordinate builder ordered

theorem digestBlock
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) (ordered : CallOrdered builder) :
    CallOrdered
      (PiRlcCanonicalSymbolicMachine.digestBlock base builder counter) := by
  unfold PiRlcCanonicalSymbolicMachine.digestBlock
  exact callOrdered_gate base _
    (appendRawPair base 1 counter builder ordered)

theorem stateBeforeBlock
    (base : Nat) (entered : SymbolicDuplex.Builder)
    (seed : Nat) (ordered : CallOrdered entered) :
    ∀ round,
      CallOrdered
        (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          base entered seed round)
  | 0 => ordered
  | round + 1 =>
      digestBlock base
        (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          base entered seed round)
        (seed + round)
        (stateBeforeBlock base entered seed ordered round)

theorem scalarBuilder
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (ordered : CallOrdered builder) :
    CallOrdered
      (PiRlcCanonicalSymbolicMachine.scalarBuilder
        base builder coordinate) := by
  unfold PiRlcCanonicalSymbolicMachine.scalarBuilder
  exact stateBeforeBlock base _ coordinate
    (enterScalar base builder coordinate ordered) 4

theorem stateAt
    (base : Nat) (initial : SymbolicDuplex.Builder)
    (ordered : CallOrdered initial) :
    ∀ coordinate,
      CallOrdered
        (PiRlcCanonicalSymbolicMachine.stateAt
          base initial coordinate)
  | 0 => ordered
  | coordinate + 1 =>
      scalarBuilder base
        (PiRlcCanonicalSymbolicMachine.stateAt
          base initial coordinate)
        coordinate
        (stateAt base initial ordered coordinate)

theorem fixedBuilder (base : Nat) (lanes : Poseidon2Core.State) :
    CallOrdered
      (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder base lanes) := by
  unfold PiRlcCanonicalSymbolicMachineHonest.fixedBuilder
    PiRlcCanonicalSymbolicMachineHonest.initialBuilder
  exact stateAt base (SymbolicDuplex.start lanes 1)
    (callOrdered_start lanes 1) 15

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineCallOrder
