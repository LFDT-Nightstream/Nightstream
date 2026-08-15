import Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards.Generated.Names

/-!
Stable facade for the Rust-generated verifier-native terminal guard ledger.

Assurance tier: artifact-checked structure only. The Rust drift owner fixes
the exact names and order. This module does not assign semantic meaning to a
guard and does not prove Rust refinement.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards

def schema : Nat := Generated.Names.schema

def names : List String := Generated.Names.values

theorem names_length : names.length = 18 := by
  rfl

end Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards
