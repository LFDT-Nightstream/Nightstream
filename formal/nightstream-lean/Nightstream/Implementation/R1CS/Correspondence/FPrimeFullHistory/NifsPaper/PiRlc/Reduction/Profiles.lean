import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryProjectionSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RecursiveCarrierArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.TerminalCarrierArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.DiagnosticProfile
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.Reduction.Trace

/-!
Generated recursive and terminal projection rows refine the independent
Phi81 reduction artifact, modulo the named public-trace bad-root event.

Assurance tier: artifact-checked. The generated trace census is checked inside
Lean; the result is not yet Rust-conformant or security-reduced end to end.

Owns: the exact 29-trace row boundary for each fixed profile; membership of
those traces in the generated census; static reduction shape; restriction of
batch exactness to typed public roles; and the final
`ReductionArtifact ∨ BatchBadRoot` result.

Does not own: the two delayed-NC `y_zcol` traces, challenge sampling,
transcript probability, carrier or parent authority, full NIFS composition,
Rust conformance, costs, or row removal.

Emits constraints: no.

Authority boundary: only generated rows for `tree.flatten` are premises. The
bad-root alternative ranges over those same 29 public identities; it does not
silently absorb the two delayed-NC identities from the 31-trace generated
profile. Static widths and degree come from generated layout facts, while the
quotient-ring interpretation comes only from `Reduction.Trace`.

| Rust stage | Lean obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.public` | exactly the typed 29 public traces satisfy their generated rows | checked | `PublicHolds` |
| `nifs.pi_rlc.verify.identities.public` | each typed public trace is a member of the generated recursive/terminal census | direct dataflow | `recursivePublicTrace_mem`, `terminalPublicTrace_mem` |
| `nifs.pi_rlc.verify.identities.public` | rho/input/output widths, quotient width 53, and degree 106 | checked | `recursiveTraceShape`, `terminalTraceShape` |
| `nifs.pi_rlc.verify.identities.public` | public batch exactness constructs the independent reduction artifact | derived | `recursiveReduction_of_exact`, `terminalReduction_of_exact` |
| `nifs.pi_rlc.verify.identities.public` | generated rows imply exact reduction or a public nonzero-polynomial root | security boundary | `recursiveReduction_or_badRoot`, `terminalReduction_or_badRoot` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction.Profiles

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.ProjectionCheck
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

set_option maxRecDepth 16384
set_option maxHeartbeats 1000000

/-! ## Exact public row boundary -/

/-- Rows owned by the paper-public projection subtree. Delayed-NC rows cannot
enter through this type because they are absent from `TraceTree.flatten`. -/
def PublicHolds
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) (assignment : Nat → Nat) : Prop :=
  ∀ trace ∈ tree.flatten, Satisfies (traceRows trace) assignment

private theorem publicRole_mem {matrixCount : Nat}
    (role : PublicRole matrixCount) : role ∈ publicOrder matrixCount := by
  cases role with
  | commitment lane =>
      unfold publicOrder
      apply List.mem_append.mpr
      apply Or.inl
      apply List.mem_append.mpr
      exact Or.inl (List.mem_ofFn.mpr ⟨lane, rfl⟩)
  | x column =>
      unfold publicOrder
      apply List.mem_append.mpr
      apply Or.inl
      apply List.mem_append.mpr
      exact Or.inr (List.mem_ofFn.mpr ⟨column, rfl⟩)
  | yRing row limb =>
      unfold publicOrder
      apply List.mem_append.mpr
      apply Or.inr
      apply List.mem_flatten.mpr
      refine ⟨List.ofFn (fun candidate : Fin 2 =>
        PublicRole.yRing row candidate), ?_, ?_⟩
      · exact List.mem_ofFn.mpr ⟨row, rfl⟩
      · exact List.mem_ofFn.mpr ⟨limb, rfl⟩

private theorem publicTrace_mem_flatten
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) (role : PublicRole matrixCount) :
    tree.publicTrace role ∈ tree.flatten := by
  unfold TraceTree.flatten
  exact List.mem_map.mpr ⟨role, publicRole_mem role, rfl⟩

private theorem recursivePublicTrace_mem
    (role : PublicRole DiagnosticProfile.matrixCount) :
    RecursiveSamplerArtifact.tree.publicTrace role ∈ recursiveTraces := by
  change RecursiveSamplerArtifact.recursivePublicTrace role ∈ recursiveTraces
  exact List.get_mem recursiveTraces
    ⟨RecursiveSamplerArtifact.publicRoleIndex role,
      RecursiveSamplerArtifact.publicRoleIndex_lt role⟩

private theorem terminalPublicTrace_mem
    (role : PublicRole DiagnosticProfile.matrixCount) :
    TerminalSamplerArtifact.tree.publicTrace role ∈ terminalTraces := by
  change TerminalSamplerArtifact.terminalPublicTrace role ∈ terminalTraces
  exact List.get_mem terminalTraces
    ⟨TerminalSamplerArtifact.publicRoleIndex role,
      TerminalSamplerArtifact.publicRoleIndex_lt role⟩

private theorem recursiveFlatten_mem_global :
    ∀ trace ∈ RecursiveSamplerArtifact.tree.flatten, trace ∈ traces := by
  intro trace member
  unfold TraceTree.flatten at member
  rcases List.mem_map.mp member with ⟨role, _, rfl⟩
  exact List.mem_append_left terminalTraces (recursivePublicTrace_mem role)

private theorem terminalFlatten_mem_global :
    ∀ trace ∈ TerminalSamplerArtifact.tree.flatten, trace ∈ traces := by
  intro trace member
  unfold TraceTree.flatten at member
  rcases List.mem_map.mp member with ⟨role, _, rfl⟩
  exact List.mem_append_right recursiveTraces (terminalPublicTrace_mem role)

/-! ## Generated layout to minimal reduction shape -/

private theorem reductionShape_of_layout {trace : ProjectionTrace}
    (layout : trace.LayoutValid) :
    trace.quotientColumns.length = 53 ∧ trace.maxDegree = 106 := by
  rcases layout with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, quotientWidth, maxDegree⟩
  exact ⟨quotientWidth, maxDegree⟩

private theorem recursivePublicLayout
    (role : PublicRole DiagnosticProfile.matrixCount) :
    (RecursiveSamplerArtifact.tree.publicTrace role).LayoutValid :=
  trace_layouts _
    (List.mem_append_left terminalTraces (recursivePublicTrace_mem role))

private theorem terminalPublicLayout
    (role : PublicRole DiagnosticProfile.matrixCount) :
    (TerminalSamplerArtifact.tree.publicTrace role).LayoutValid :=
  trace_layouts _
    (List.mem_append_right recursiveTraces (terminalPublicTrace_mem role))

theorem recursiveTraceShape :
    TraceShapeArtifact RecursiveSamplerArtifact.tree where
  challengeWidth role index := by
    rw [RecursiveSamplerArtifact.publicShared role index]
    exact RecursiveSamplerArtifact.challengeColumns_length index
  inputWidth := RecursiveCarrierArtifact.inputWidth
  outputWidth := RecursiveCarrierArtifact.outputWidth
  quotientWidth role :=
    (reductionShape_of_layout (recursivePublicLayout role)).1
  maxDegree role :=
    (reductionShape_of_layout (recursivePublicLayout role)).2

theorem terminalTraceShape :
    TraceShapeArtifact TerminalSamplerArtifact.tree where
  challengeWidth role index := by
    rw [TerminalSamplerArtifact.publicShared role index]
    exact TerminalSamplerArtifact.challengeColumns_length index
  inputWidth := TerminalCarrierArtifact.inputWidth
  outputWidth := TerminalCarrierArtifact.outputWidth
  quotientWidth role :=
    (reductionShape_of_layout (terminalPublicLayout role)).1
  maxDegree role :=
    (reductionShape_of_layout (terminalPublicLayout role)).2

/-! ## Public exactness to independent reduction -/

theorem recursiveReduction_of_exact
    {assignment : Nat → Nat}
    (exact : BatchExact (BatchIdentity
      RecursiveSamplerArtifact.tree.flatten assignment)) :
    ReductionArtifact assignment RecursiveSamplerArtifact.tree := by
  apply reductionArtifact_of_exact recursiveTraceShape
  intro role
  apply exact
  exact List.mem_map.mpr
    ⟨_, publicTrace_mem_flatten RecursiveSamplerArtifact.tree role, rfl⟩

theorem terminalReduction_of_exact
    {assignment : Nat → Nat}
    (exact : BatchExact (BatchIdentity
      TerminalSamplerArtifact.tree.flatten assignment)) :
    ReductionArtifact assignment TerminalSamplerArtifact.tree := by
  apply reductionArtifact_of_exact terminalTraceShape
  intro role
  apply exact
  exact List.mem_map.mpr
    ⟨_, publicTrace_mem_flatten TerminalSamplerArtifact.tree role, rfl⟩

/-! ## Exact 29-trace generated-row soundness -/

private theorem publicBatchAccepted
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    {tree : TraceTree arity matrixCount} {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (globalMember : ∀ trace ∈ tree.flatten, trace ∈ traces)
    (holds : PublicHolds tree assignment) :
    BatchAccepted K.ops (BatchIdentity tree.flatten assignment) := by
  apply ProjectionTrace.census_batchAccepted tree.flatten assignment
    constantOne
  · intro trace member
    exact trace_layouts trace (globalMember trace member)
  · intro trace member
    exact trace_pairs_nonempty trace (globalMember trace member)
  · intro trace member pair pairMember
    exact trace_pair_widths trace (globalMember trace member) pair pairMember
  · intro definition definitionMember
    rcases List.mem_flatMap.mp definitionMember with
      ⟨trace, traceMember, member⟩
    apply builderDefinitions_sound assignmentCanonical constantOne
      (definitions_canonical trace (globalMember trace traceMember))
    intro row rowMember
    apply holds trace traceMember row
    unfold traceRows
    exact List.mem_append_left _ rowMember
    exact member
  · intro row rowMember
    rcases List.mem_flatMap.mp rowMember with ⟨trace, traceMember, member⟩
    apply holds trace traceMember row
    unfold traceRows
    exact List.mem_append_right _ member

/-- Exact recursive public rows imply the independent Phi81 reduction, or one
of those same 29 sampled identities exposes the named bad-root event. -/
theorem recursiveReduction_or_badRoot
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (holds : PublicHolds RecursiveSamplerArtifact.tree assignment) :
    ReductionArtifact assignment RecursiveSamplerArtifact.tree ∨
      BatchBadRoot K.ops (BatchIdentity
        RecursiveSamplerArtifact.tree.flatten assignment) := by
  have accepted := publicBatchAccepted assignmentCanonical constantOne
    recursiveFlatten_mem_global holds
  rcases batchAccepted_implies_exact_or_badRoot K.ops _ accepted with
    exact | badRoot
  · exact Or.inl (recursiveReduction_of_exact exact)
  · exact Or.inr badRoot

/-- Terminal counterpart of `recursiveReduction_or_badRoot`, with the same
exact 29-trace authority boundary. -/
theorem terminalReduction_or_badRoot
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (holds : PublicHolds TerminalSamplerArtifact.tree assignment) :
    ReductionArtifact assignment TerminalSamplerArtifact.tree ∨
      BatchBadRoot K.ops (BatchIdentity
        TerminalSamplerArtifact.tree.flatten assignment) := by
  have accepted := publicBatchAccepted assignmentCanonical constantOne
    terminalFlatten_mem_global holds
  rcases batchAccepted_implies_exact_or_badRoot K.ops _ accepted with
    exact | badRoot
  · exact Or.inl (terminalReduction_of_exact exact)
  · exact Or.inr badRoot

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction.Profiles
