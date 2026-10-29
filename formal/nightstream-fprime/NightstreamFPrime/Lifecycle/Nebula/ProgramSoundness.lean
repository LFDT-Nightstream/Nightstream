import NightstreamFPrime.Lifecycle.Nebula.MemoryProgram
import NightstreamFPrime.Lifecycle.Nebula.RunLink

/-! Owns the join of the memory program's contract to the run link: each step
that the Stage 1 relation accepts for `MemoryApp.program` is a
`StepWitness.Holds` step. `EndToEnd.bad_subset` uses it with the run link. It
does not own the Stage 1 fold that authenticates the chain. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing

variable {p : Geometry} {app : PortApp p}

/-- A valid program step with the program's output state is a run-link step. -/
theorem MemoryApp.holds_of_valid {input witness output : List F}
    (valid : MemoryApp.valid p app input witness) (out : output = MemoryApp.step p app input witness) :
    (MemoryApp.decode p app witness).Holds app input output :=
  ⟨valid.1, valid.2, out⟩

end NightstreamFPrime.Lifecycle.Nebula
