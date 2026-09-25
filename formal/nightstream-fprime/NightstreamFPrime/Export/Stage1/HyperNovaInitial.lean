import NightstreamFPrime.Export.Stage1.PerApplicationTerminal

/-!
Owns the canonical initial HyperNova statement and empty proof envelope.
The selected terminal predicate accepts this pair for every initial state
with the application's public state width. No recursive payload is created.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaInitial

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

/-- The initial public statement advertises the same state at both endpoints
and fixes the iteration to zero. -/
def initialStatement (z0 : AppState) : TerminalStatement AppState where
  iteration := 0
  z0 := z0
  zi := z0

/-- The initial proof is the existing empty constructor and carries no
running claims, fresh claim, openings, or local NIFS proof. -/
def initialProof (application : Lifecycle.Stage1.Application.Program) :
    PerApplicationTerminal.ProofEnvelope application := .bottom

/-- The verifier-selected terminal predicate accepts the constructed initial
statement and empty proof. The sole state premise is its fixed public width;
counter admissibility and equality of the two endpoints follow by construction. -/
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
