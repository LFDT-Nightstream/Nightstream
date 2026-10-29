import NightstreamFPrime.Lifecycle.Nebula.PortApp

/-! Owns the free application of a plan (spec §10.1): any number of state
words, no private words, no rows, and a model that takes every step of `B_ops`
ports to any state. The memory rows are its only checks, so an execution of it
is a replay of its active ports (`MemoryTrace.executes_accessAll`), and with a
binding it gives `MemoryConsistent` for a catalog
(`MemoryTrace.memoryConsistent_of_executes`). It owns the free app's Ob7 and
its half of Ob5. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula

/-- The free application with `width` state words. Its model does not check
port validity: the access rule of `Executes` does, and `PortApp.sound` gives
the model only rows O1, O3, and O7, not O5 and O6. -/
def freeApp (p : Plan) (width : ℕ) (small : width < goldilocksModulus) : PortApp p where
  width := width
  aux := 0
  model := { Step := fun _ ports _ => ports.length = p.bOps, idle := fun _ => by simp [idlePorts] }
  rows := []
  width_lt := small
  portsLength := id
  sound := fun _ _ _ _ => List.length_ofFn
  complete := by
    intros
    exact ⟨Fin.elim0, by simp [Circuit.ConstraintsHold]⟩

end NightstreamFPrime.Lifecycle.Nebula
