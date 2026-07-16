import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources

/-!
Terminal-profile first-accepted field semantics for all fifteen `Pi_RLC`
scalar tails.

Assurance tier: implementation/R1CS correspondence. For each terminal scalar,
this file instantiates the profile-independent 54-of-64 theorem with the exact
terminal lane semantics, source bindings, and readable tail rows.

Owns: the terminal scalar-indexed semantic field output; exact equality of one
or all 54 terminal output wires with the independent first-accepted output;
and proof that the semantic list has length 54.

Does not own: Poseidon2 transcript provenance for the scalar field columns,
state chaining across the fifteen scalars, coefficient assembly into a ring
element, Rust trace conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: these theorems establish first-accepted semantics only for
canonical chunks of terminal field columns. They must not be described as rho
challenges until a separate verifier-owned Poseidon2 schedule theorem proves
the origin and chaining of every field column.

| Protocol | Phase | Constraint family | Terminal input | Proven result |
|---|---|---|---|---|
| `Pi_RLC` | scalar `rho` | generic tail view | exact terminal tail mapping | readable terminal rows instantiate the generic tail interface |
| `Pi_RLC` | bounded sampler | enough accepted | lane/source/tail refinement | independent field-derived output has length 54 |
| `Pi_RLC` | output selection | one position | accepted terminal rows | output wire equals the corresponding first-accepted centered symbol |
| `Pi_RLC` | output selection | all 54 positions | accepted terminal rows | complete terminal output equals independent field-derived output |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Independent first-accepted output for one terminal scalar's canonical field
columns. This definition makes no transcript-provenance claim. -/
def semanticOutput
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : List ProductionAlphabet.Coefficient :=
  TailFirstAccepted.semanticOutput (TailSources.layout rho) assignment canonical

/-- Exact terminal output wires in coefficient order. -/
def productionOutput
    (assignment : Nat -> Nat)
    (rho : Fin ScalarRows.scalarCount) : List Nat :=
  List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
    TailRows.localAssignment assignment rho
      (SelectionRows.outputCol position.val)

/-- Centered-field view of the independent terminal semantic output. -/
def semanticFieldOutput
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : List Nat :=
  TailFirstAccepted.semanticFieldOutput (TailSources.layout rho) assignment
    canonical

/-- Exact terminal rows inhabit the profile-independent readable-tail view. -/
theorem accepted_genericTailSatisfies
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment (TailSources.layout rho)
        assignment) := by
  simpa [TailSources.layout, TailCandidateSemantics.localAssignment,
    TailRows.localAssignment] using TailRows.accepted_satisfies accepted rho

/-- Every accepted terminal scalar has exactly 54 field-derived semantic
coefficients. -/
theorem semanticOutput_length
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    (semanticOutput assignment canonical rho).length =
      ProductionAlphabet.coefficientCount := by
  let layout := TailSources.layout rho
  have lanes : ScalarLanes.Refines assignment canonical layout.lanes := by
    simpa [layout, TailSources.layout] using
      ScalarSemantics.accepted_refines_lanes prime canonical one accepted rho
  have bindings : TailCandidateSemantics.SourceBindings layout assignment := by
    simpa [layout] using
      TailSources.accepted_sourceBindings prime canonical one accepted rho
  have tailSatisfies : Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment layout assignment) := by
    simpa [layout] using accepted_genericTailSatisfies accepted rho
  exact TailFirstAccepted.semanticOutput_length prime canonical one layout lanes
    bindings tailSatisfies

/-- One accepted terminal output wire equals the matching position of the
independent first-accepted field output. -/
theorem outputAt_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount)
    (position : Fin ProductionAlphabet.coefficientCount) :
    TailRows.localAssignment assignment rho
        (SelectionRows.outputCol position.val) =
      CandidateOrder.centeredField
        ((semanticOutput assignment canonical rho).getD position.val
          TailFirstAccepted.defaultCoefficient) := by
  let layout := TailSources.layout rho
  have lanes : ScalarLanes.Refines assignment canonical layout.lanes := by
    simpa [layout, TailSources.layout] using
      ScalarSemantics.accepted_refines_lanes prime canonical one accepted rho
  have bindings : TailCandidateSemantics.SourceBindings layout assignment := by
    simpa [layout] using
      TailSources.accepted_sourceBindings prime canonical one accepted rho
  have tailSatisfies : Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment layout assignment) := by
    simpa [layout] using accepted_genericTailSatisfies accepted rho
  have refined := TailFirstAccepted.outputAt_refines prime canonical one layout
    lanes bindings tailSatisfies position
  simpa [layout, semanticOutput, TailSources.layout,
    TailCandidateSemantics.localAssignment, TailRows.localAssignment] using
      refined

/-- Complete terminal scalar closure at the field-derived boundary. Poseidon2
provenance and scalar assembly remain explicit later obligations. -/
theorem accepted_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    productionOutput assignment rho =
      semanticFieldOutput assignment canonical rho := by
  let layout := TailSources.layout rho
  have lanes : ScalarLanes.Refines assignment canonical layout.lanes := by
    simpa [layout, TailSources.layout] using
      ScalarSemantics.accepted_refines_lanes prime canonical one accepted rho
  have bindings : TailCandidateSemantics.SourceBindings layout assignment := by
    simpa [layout] using
      TailSources.accepted_sourceBindings prime canonical one accepted rho
  have tailSatisfies : Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment layout assignment) := by
    simpa [layout] using accepted_genericTailSatisfies accepted rho
  have refined :=
    TailFirstAccepted.productionOutput_eq_semanticFieldOutput prime canonical
      one layout lanes bindings tailSatisfies
  simpa [productionOutput, semanticFieldOutput, layout, TailSources.layout,
    TailFirstAccepted.productionOutput,
    TailCandidateSemantics.localAssignment, TailRows.localAssignment] using
      refined

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted
