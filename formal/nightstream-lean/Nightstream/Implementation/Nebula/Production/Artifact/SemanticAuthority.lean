import Nightstream.Protocol.Nebula.WasmStatement

/-!
Contract: verifier-owned semantic authority for one Nebula V2 statement.

The generated verifier selects the application transition function, program,
snapshot-root function, and the four public identifiers that name those typed
objects. A theorem caller cannot select a different machine or root function
after the verifier context is fixed.

The digest fields are identifiers. They are not authority and this module does
not claim that a digest can reconstruct a function. A deployed artifact must
construct this typed package and prove that its canonical statement decoder
selects the same identifiers.

Assurance tier: typed generated-artifact boundary.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionSemanticAuthority

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Soundness
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStatement

/-- Typed semantic objects selected by one verifier artifact. The functions are
authority-bearing. Their digests only identify the selected artifact in the
external statement. -/
structure Artifact (Program : Type) where
  machine : Machine Program
  program : Program
  snapshotRoot : Snapshot -> Digest.Value
  applicationRelationDigest : Digest.Value
  programDigest : Digest.Value
  memoryPlanDigest : Digest.Value
  applicationStateSchemaDigest : Digest.Value

/-- Exact static link from the public statement to the typed semantic artifact.
No hash collision or parser claim is hidden in this relation. -/
structure MatchesStatement
    {Program : Type}
    (artifact : Artifact Program)
    (statement : ProductionStatement Program) : Prop where
  programExact : statement.base.program = artifact.program
  applicationRelationDigestExact :
    statement.base.identity.applicationRelationDigest =
      artifact.applicationRelationDigest
  programDigestExact :
    statement.base.identity.programDigest = artifact.programDigest
  memoryPlanDigestExact :
    statement.base.identity.memoryPlanDigest = artifact.memoryPlanDigest
  applicationStateSchemaDigestExact :
    statement.base.identity.verifierKey.applicationStateSchemaDigest =
      artifact.applicationStateSchemaDigest

namespace MatchesStatement

theorem identityDigestsExact
    {Program : Type}
    {artifact : Artifact Program}
    {statement : ProductionStatement Program}
    (matched : MatchesStatement artifact statement) :
    statement.base.identity.applicationRelationDigest =
          artifact.applicationRelationDigest /\
      statement.base.identity.programDigest = artifact.programDigest /\
      statement.base.identity.memoryPlanDigest = artifact.memoryPlanDigest /\
      statement.base.identity.verifierKey.applicationStateSchemaDigest =
        artifact.applicationStateSchemaDigest :=
  ⟨matched.applicationRelationDigestExact,
    matched.programDigestExact,
    matched.memoryPlanDigestExact,
    matched.applicationStateSchemaDigestExact⟩

end MatchesStatement

/-! ## Necessity countermodel -/

/-- Digest equality alone says nothing about the selected transition function.
The theorem is parameterized by a concrete state and row, so it needs no
manufactured WASM witness. -/
def RejectMachine (Program : Type) : Machine Program where
  step := fun _ _ _ => none

def ReturnMachine {Program : Type} (next : AppStateVector) : Machine Program where
  step := fun _ _ _ => some next

theorem equal_identifiers_do_not_bind_machine
    {Program : Type} (program : Program)
    (state next : AppStateVector) (row : Ports.NormalizedRow) :
    (RejectMachine Program).step program state row = none /\
      (ReturnMachine next).step program state row = some next := by
  exact ⟨rfl, rfl⟩

end Nightstream.Implementation.Nebula.ProductionSemanticAuthority
