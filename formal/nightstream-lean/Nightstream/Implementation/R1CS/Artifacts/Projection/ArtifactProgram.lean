import Nightstream.Implementation.R1CS.Artifacts.Projection.IndexedRows
import Nightstream.Implementation.R1CS.Core.Projection.Interpretation

/-!
Profile-neutral certificate schema for a reconstructed projection program.

Owns: the data boundary joining retained physical rows to reconstructed SSA
definitions and assertion rows, set-level trace coverage, and exact-index
embedding into a larger row list.

Does not own: row satisfaction transport, a concrete artifact, trace layout,
assignment canonicality, semantic projection soundness, costs, security
bounds, or row removal.

Emits constraints: no.

| Certificate branch | Mathematical obligation | Evidence |
|---|---|---|
| definitions | source rows equal reconstructed builder equations | `ExactRows.definitionsMatch` |
| checks | source rows equal reconstructed assertions | `ExactRows.checksMatch` |
| canonicality | reconstructed subtraction rows denote their equations | `ExactRows.definitionsCanonical` |
| coverage | every trace equation/check occurs in the certificate | `Covers` |
| embedding | selected rows occur at their advertised full-program indices | `EmbeddedIn` |
-/

namespace Nightstream.Implementation.R1CS.ProjectionArtifactProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram

/-- Independent physical and reconstructed views of one selected row block. -/
structure Certificate where
  definitionSourceRows : List (Nat × Row)
  indexedDefinitions : List (Nat × Program.Definition)
  checkSourceRows : List (Nat × Row)
  indexedChecks : List (Nat × Row)

namespace Certificate

def sourceRows (certificate : Certificate) : List (Nat × Row) :=
  certificate.definitionSourceRows ++ certificate.checkSourceRows

def sourceRowValues (certificate : Certificate) : List Row :=
  certificate.sourceRows.map Prod.snd

def definitions (certificate : Certificate) : List Program.Definition :=
  certificate.indexedDefinitions.map Prod.snd

def checks (certificate : Certificate) : List Row :=
  certificate.indexedChecks.map Prod.snd

/-- Exact physical rows plus canonical reconstructed definitions. -/
structure ExactRows (certificate : Certificate) : Prop where
  definitionsMatch : ProjectionIndexedRows.indexedRowsMatch
    certificate.definitionSourceRows certificate.indexedDefinitions = true
  checksMatch : ProjectionIndexedRows.indexedRowsMatchRows
    certificate.checkSourceRows certificate.indexedChecks = true
  definitionsCanonical : forall definition,
    definition ∈ certificate.definitions -> definition.Canonical

/-- Set-level equation coverage. Shared definitions may occur in multiple
trace views while retaining one physical row owner. -/
structure Covers (certificate : Certificate)
    (traces : List ProjectionTrace) : Prop where
  definitionsIff : forall definition,
    definition ∈ traces.flatMap ProjectionTrace.definitions ↔
      definition ∈ certificate.definitions
  checksIff : forall row,
    row ∈ traces.flatMap ProjectionTrace.checks ↔
      row ∈ certificate.checks

def EmbeddedIn (certificate : Certificate) (fullRows : List Row) : Prop :=
  ProjectionIndexedRows.SourceRowsEmbedded certificate.sourceRows fullRows

end Certificate

end Nightstream.Implementation.R1CS.ProjectionArtifactProgram
