import NightstreamFPrime.Export.Stage1.PerApplicationTerminal

/-!
Owns the canonical initial HyperNova statement and empty proof envelope.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaInitial

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

def initialStatement (z0 : AppState) : TerminalStatement AppState where
  iteration := 0
  z0 := z0
  zi := z0

def initialProof (application : Lifecycle.Stage1.Application.Program) :
    PerApplicationTerminal.ProofEnvelope application := .bottom

theorem initial_accepted
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationTerminal.FitsTwoPow28 application)
    (commitmentSetup : PerApplicationTerminal.CommitmentSetup application)
    (z0 : AppState)
    (stateWidth : z0.length = Lifecycle.Stage1.Application.stateWordCount) :
    PerApplicationTerminal.Holds application fits commitmentSetup
      (initialStatement z0) (initialProof application) := by
  apply (PerApplicationTerminal.holds_bottom_iff application fits
    commitmentSetup (initialStatement z0)).2
  exact ⟨⟨by change 0 < goldilocksModulus; decide, stateWidth, stateWidth⟩, rfl, rfl⟩

end NightstreamFPrime.Export.Stage1.HyperNovaInitial
