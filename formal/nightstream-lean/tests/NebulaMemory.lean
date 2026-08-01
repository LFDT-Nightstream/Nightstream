import Nightstream.Protocol.Nebula

set_option autoImplicit false

namespace tests.NebulaMemory

open Nightstream.SuperNeo.Concrete
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.Memory

def challenges : Challenges :=
  ⟨K.one, K.zero⟩

def initialCell : MemTuple :=
  ⟨0, 0, 42⟩

def finalCell : MemTuple :=
  ⟨1, 0, 42⟩

def access : Access :=
  ⟨initialCell, finalCell⟩

def honestExecution :
    Executes [initialCell] 0 [access] [finalCell] 1 :=
  .cons
    { left := []
      right := []
      beforeExact := rfl
      afterExact := rfl
      sameCell := rfl
      previousTimestamp := by decide
      writeTimestamp := rfl
      timestampExact := rfl }
    (.nil [finalCell] 1)

theorem honest_balance :
    Balanced (products challenges [initialCell] [access] [finalCell]) :=
  executes_balanced challenges honestExecution

/-- A final cell with a changed value is not accepted by this concrete
challenge when the read/write access still carries value 42. -/
example :
    ¬ Balanced
      (products challenges [initialCell] [access] [⟨1, 0, 43⟩]) := by
  unfold Balanced
  decide

end tests.NebulaMemory
