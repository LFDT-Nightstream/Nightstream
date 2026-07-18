import Nightstream.Implementation.R1CS.Correspondence.Projection.IndexedRows
import Nightstream.Implementation.R1CS.Artifacts.Projection.ArtifactProgram

/-!
Satisfaction transport for an exact reconstructed projection certificate.

Owns: transport from source/full-program satisfaction to every reconstructed
trace definition and check.

Does not own: the certificate schema, a concrete artifact, trace layout,
assignment canonicality, constant-one enforcement, semantic projection
soundness, row multiplicity, costs, security bounds, or row removal.

Emits constraints: no.

| Transport branch | Mathematical obligation | Required evidence |
|---|---|---|
| definitions | satisfying exact physical rows imply reconstructed SSA equations | `ExactRows` |
| checks | satisfying exact physical rows imply reconstructed assertions | `ExactRows` |
| coverage | certificate consequences reach every advertised trace | `Covers` |
| embedding | full-row satisfaction reaches the selected source rows | `EmbeddedIn` |
-/

namespace Nightstream.Implementation.R1CS.ProjectionArtifactProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram

namespace Certificate

private theorem definitionRows_satisfied
    {certificate : Certificate} {assignment : Nat -> Nat}
    (sourceSatisfies : Satisfies certificate.sourceRowValues assignment) :
    Satisfies
      (certificate.definitionSourceRows.map Prod.snd) assignment := by
  intro row member
  apply sourceSatisfies row
  simp only [sourceRowValues, sourceRows, List.map_append,
    List.mem_append]
  exact Or.inl member

private theorem checkRows_satisfied
    {certificate : Certificate} {assignment : Nat -> Nat}
    (sourceSatisfies : Satisfies certificate.sourceRowValues assignment) :
    Satisfies (certificate.checkSourceRows.map Prod.snd) assignment := by
  intro row member
  apply sourceSatisfies row
  simp only [sourceRowValues, sourceRows, List.map_append,
    List.mem_append]
  exact Or.inr member

/-- Exact physical definition rows imply all advertised SSA equations.
Canonical assignment values and the constant-one wire are explicit because
they are global R1CS invariants, not artifact facts. -/
theorem definitionsHold_of_sourceRows
    {certificate : Certificate} {assignment : Nat -> Nat}
    (exact : certificate.ExactRows)
    (assignmentCanonical : forall column,
      assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies certificate.sourceRowValues assignment) :
    DefinitionsHold assignment certificate.definitions := by
  have builderRows : Satisfies
      (certificate.indexedDefinitions.map
        fun entry => entry.2.builderRow) assignment :=
    ProjectionIndexedRows.builderRows_satisfied_of_indexedRowsMatch
      certificate.definitionSourceRows certificate.indexedDefinitions
      exact.definitionsMatch
      (definitionRows_satisfied sourceSatisfies)
  have normalizedBuilderRows : Satisfies
      (certificate.definitions.map Program.Definition.builderRow)
      assignment := by
    simpa [definitions, List.map_map] using builderRows
  exact Program.builderDefinitions_sound assignmentCanonical constantOne
    exact.definitionsCanonical normalizedBuilderRows

/-- Exact physical assertion rows imply every reconstructed check. -/
theorem checksHold_of_sourceRows
    {certificate : Certificate} {assignment : Nat -> Nat}
    (exact : certificate.ExactRows)
    (sourceSatisfies : Satisfies certificate.sourceRowValues assignment) :
    Satisfies certificate.checks assignment := by
  exact ProjectionIndexedRows.rows_satisfied_of_indexedRowsMatchRows
    certificate.checkSourceRows certificate.indexedChecks
    exact.checksMatch (checkRows_satisfied sourceSatisfies)

/-- One exact certificate discharges the definition and check premises for a
complete trace census. -/
theorem traceRowsHold_of_sourceRows
    {certificate : Certificate} {traces : List ProjectionTrace}
    {assignment : Nat -> Nat}
    (exact : certificate.ExactRows)
    (coverage : certificate.Covers traces)
    (assignmentCanonical : forall column,
      assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies certificate.sourceRowValues assignment) :
    DefinitionsHold assignment
        (traces.flatMap ProjectionTrace.definitions) ∧
      Satisfies (traces.flatMap ProjectionTrace.checks) assignment := by
  have definitionsHold := definitionsHold_of_sourceRows exact
    assignmentCanonical constantOne sourceSatisfies
  have checksHold := checksHold_of_sourceRows exact sourceSatisfies
  constructor
  · intro definition member
    exact definitionsHold definition
      ((coverage.definitionsIff definition).mp member)
  · intro row member
    exact checksHold row ((coverage.checksIff row).mp member)

/-- Full-program satisfaction reaches every covered trace only through an
exact absolute-index embedding of the selected source rows. -/
theorem traceRowsHold_of_embedded
    {certificate : Certificate} {traces : List ProjectionTrace}
    {fullRows : List Row} {assignment : Nat -> Nat}
    (exact : certificate.ExactRows)
    (coverage : certificate.Covers traces)
    (assignmentCanonical : forall column,
      assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (embedded : certificate.EmbeddedIn fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    DefinitionsHold assignment
        (traces.flatMap ProjectionTrace.definitions) ∧
      Satisfies (traces.flatMap ProjectionTrace.checks) assignment := by
  apply traceRowsHold_of_sourceRows exact coverage assignmentCanonical
    constantOne
  exact ProjectionIndexedRows.sourceRows_satisfied_of_embedded
    embedded fullSatisfies

end Certificate

end Nightstream.Implementation.R1CS.ProjectionArtifactProgram
