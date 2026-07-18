import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Sampler.TailSources
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted

/-!
Active field-derived first-accepted semantics for all fifteen PiRLC scalars.

Owns: bounded-sampler success; the independent first 54 accepted symbols from
each 64-candidate field prefix; and equality of all 810 active output columns
with their centered semantic coefficients.

Does not own: Poseidon2 provenance or chaining of the field columns, ring
assembly, PiRLC algebra, Rust source identity beyond explicit embedding,
costs, or row removal.

Emits constraints: no.

Authority boundary: selection witnesses only route candidates whose
acceptance, symbol, and accepted-prefix count were independently derived from
canonical field chunks. These are not yet rho challenges; a later transcript
theorem must bind every field column to the verifier-owned Poseidon2 machine.

| Branch | Multiplicity | Proven obligation |
|---|---:|---|
| bounded success | `15` | at least 54 of each 64-candidate prefix are accepted |
| selected output | `15 x 54` | physical output equals independent centered symbol |
| scalar output | `15` | complete vector equals `firstAccepted` semantic output |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Independent first-accepted output over one scalar's canonical field
chunks. It intentionally carries no transcript-provenance claim. -/
def semanticOutput
    (assignment : Nat → Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin scalarCount) : List ProductionAlphabet.Coefficient :=
  PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.semanticOutput
    (TailSources.layout rho) assignment canonical

/-- Exact active physical output columns in coefficient order. -/
def productionOutput
    (assignment : Nat → Nat) (rho : Fin scalarCount) : List Nat :=
  List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
    assignment (outputColumn rho position)

/-- Centered-field view of the independent semantic output. -/
def semanticFieldOutput
    (assignment : Nat → Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin scalarCount) : List Nat :=
  PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.semanticFieldOutput
    (TailSources.layout rho) assignment canonical

/-- Active tail rows inhabit the profile-independent readable hierarchy. -/
theorem accepted_genericTailSatisfies
    {fullRows : List Row} {assignment : Nat → Nat}
    (accepted : Rows.EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    Satisfies SelectionRows.rows
      (PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
        (TailSources.layout rho) assignment) := by
  simpa [TailSources.layout,
    PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment]
    using Rows.accepted_readableTail accepted rho

/-- The accepted rows prove the bounded-sampler success premise itself, not
only an output equality conditional on success. -/
theorem enoughAccepted
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Rows.EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    Nightstream.SuperNeo.Sampling.FirstAccepted.Enough
      ProductionAlphabet.verifier ProductionAlphabet.coefficientCount
      (PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.candidates
        (TailSources.layout rho) assignment canonical) := by
  let activeLayout := TailSources.layout rho
  have lanes :
      PiRlcChallenge.Sampler.Refinement.ScalarLanes.Refines
        assignment canonical activeLayout.lanes := by
    simpa [activeLayout, TailSources.layout] using
      ScalarSemantics.accepted_refines_lanes
        prime canonical one accepted rho
  have bindings :
      PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.SourceBindings
        activeLayout assignment := by
    simpa [activeLayout] using
      TailSources.accepted_sourceBindings
        prime canonical one accepted rho
  have tailSatisfies :
      Satisfies SelectionRows.rows
        (PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
          activeLayout assignment) := by
    simpa [activeLayout] using accepted_genericTailSatisfies accepted rho
  simpa [activeLayout] using
    PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.enoughAccepted
      prime canonical one activeLayout lanes bindings tailSatisfies

/-- One active physical output equals the corresponding independent centered
field coefficient. -/
theorem outputAt_refines
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Rows.EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount)
    (position : Fin ProductionAlphabet.coefficientCount) :
    assignment (outputColumn rho position) =
      PiRlcChallenge.Sampler.CandidateOrder.centeredField
        ((semanticOutput assignment canonical rho).getD position.val
          PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.defaultCoefficient) := by
  let activeLayout := TailSources.layout rho
  have lanes :
      PiRlcChallenge.Sampler.Refinement.ScalarLanes.Refines
        assignment canonical activeLayout.lanes := by
    simpa [activeLayout, TailSources.layout] using
      ScalarSemantics.accepted_refines_lanes
        prime canonical one accepted rho
  have bindings :
      PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.SourceBindings
        activeLayout assignment := by
    simpa [activeLayout] using
      TailSources.accepted_sourceBindings
        prime canonical one accepted rho
  have tailSatisfies :
      Satisfies SelectionRows.rows
        (PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics.localAssignment
          activeLayout assignment) := by
    simpa [activeLayout] using accepted_genericTailSatisfies accepted rho
  have refined :=
    PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.outputAt_refines
      prime canonical one activeLayout lanes bindings tailSatisfies position
  rw [TailSources.local_output_eq_physical assignment rho position] at refined
  simpa [activeLayout, semanticOutput] using refined

/-- Complete active scalar closure at the field-derived boundary. -/
theorem accepted_refines
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Rows.EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    productionOutput assignment rho =
      semanticFieldOutput assignment canonical rho := by
  apply List.ext_getElem
  · simp [productionOutput, semanticFieldOutput,
      PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.semanticFieldOutput]
  · intro index leftLt rightLt
    have indexLt : index < ProductionAlphabet.coefficientCount := by
      simpa [productionOutput] using leftLt
    let position : Fin ProductionAlphabet.coefficientCount := ⟨index, indexLt⟩
    have refined := outputAt_refines prime canonical one accepted rho position
    simpa [productionOutput, semanticFieldOutput,
      PiRlcChallenge.Sampler.Refinement.TailFirstAccepted.semanticFieldOutput,
      position] using refined

/-- Explicit result package retained by later transcript and ring-assembly
bridges. -/
structure FieldFirstAcceptedRefines
    (assignment : Nat → Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin scalarCount) : Prop where
  enough : Nightstream.SuperNeo.Sampling.FirstAccepted.Enough
    ProductionAlphabet.verifier ProductionAlphabet.coefficientCount
    (PiRlcChallenge.Sampler.Refinement.TailPrefixCounts.candidates
      (TailSources.layout rho) assignment canonical)
  outputs : productionOutput assignment rho =
    semanticFieldOutput assignment canonical rho

/-- Exact normalized source-row satisfaction plus explicit active embeddings
refine to independent 54-of-64 field semantics. -/
theorem embeddedRows_refine_firstAccepted
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Rows.EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    FieldFirstAcceptedRefines assignment canonical rho :=
  { enough := enoughAccepted prime canonical one accepted rho
    outputs := accepted_refines prime canonical one accepted rho }

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted
