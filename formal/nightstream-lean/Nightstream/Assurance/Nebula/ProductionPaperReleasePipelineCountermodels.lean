import Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline

/-!
Countermodels for the staged deployed-refinement surface.

The first model stops at the exact decode event. The second model ignores
every external artifact and returns a pre-existing exact local lifetime. These
models show why a concrete inhabitant of `StagedExtraction` must come from the
canonical parser, terminal backend, extractor, generated rows, and application
compiler. The abstract composition theorem is not implementation evidence.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.Nebula.ProductionPaperReleasePipelineCountermodels

open Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline
open Nightstream.Implementation.Nebula.ProductionPaperExactLifetime
open Nightstream.Protocol.Nebula.Soundness

/-- A staged boundary can stop at its exact decode event without constructing
any downstream witness. This is valid only when that event is declared to
occur. -/
def decodeFailureStages
    {Bytes Parsed Program : Type}
    (decode : Bytes → Option Parsed)
    (terminalAccepts : Parsed → Prop)
    (context : Context Program) :
    StagedExtraction decode terminalAccepts (fun _ => True) context where
  CanonicalProof := Unit
  TerminalWitness := Unit
  FoldWitness := Unit
  GeneratedRows := Unit
  decodedCanonical := fun _ _ _ => False
  terminalExtracted := fun _ _ _ => False
  foldExtracted := fun _ _ => False
  rowsRefined := fun _ _ => False
  lifetimeRefined := fun _ _ => False
  decodeRefinement := by
    intro _proof _parsed _decoded
    exact Or.inl trivial
  terminalBackendExtraction := by
    intro _proof _parsed _canonical impossible _accepted
    exact False.elim impossible
  foldKnowledgeExtraction := by
    intro _parsed _canonical _terminal impossible
    exact False.elim impossible
  generatedRelationRefinement := by
    intro _terminal _fold impossible
    exact False.elim impossible
  applicationPortRefinement := by
    intro _fold _rows impossible
    exact False.elim impossible

/-- The staged-boundary type is inhabited without a lifetime when its exact
decode event is allowed. A release claim must bound or exclude that event. -/
theorem stages_type_does_not_prove_extraction
    {Bytes Parsed Program : Type}
    (decode : Bytes → Option Parsed)
    (terminalAccepts : Parsed → Prop)
    (context : Context Program) :
    Nonempty (StagedExtraction decode terminalAccepts (fun _ => True) context) :=
  ⟨decodeFailureStages decode terminalAccepts context⟩

/-- Every relation in this boundary ignores its input. The last relation
returns one pre-existing exact local lifetime. No bad event occurs. -/
def ignoresAllStages
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {context : Context Program}
    (lifetime : ExtractedLifetime context) :
    StagedExtraction decode terminalAccepts (fun _ => False) context where
  CanonicalProof := Unit
  TerminalWitness := Unit
  FoldWitness := Unit
  GeneratedRows := Unit
  decodedCanonical := fun _ _ _ => True
  terminalExtracted := fun _ _ _ => True
  foldExtracted := fun _ _ => True
  rowsRefined := fun _ _ => True
  lifetimeRefined := fun _ candidate => candidate = lifetime
  decodeRefinement := by
    intro _proof _parsed _decoded
    exact Or.inr ⟨(), trivial⟩
  terminalBackendExtraction := by
    intro _proof _parsed _canonical _decoded _accepted
    exact Or.inr ⟨(), trivial⟩
  foldKnowledgeExtraction := by
    intro _parsed _canonical _terminal _extracted
    exact Or.inr ⟨(), trivial⟩
  generatedRelationRefinement := by
    intro _terminal _fold _extracted
    exact Or.inr ⟨(), trivial⟩
  applicationPortRefinement := by
    intro _fold _rows _refined
    exact Or.inr ⟨lifetime, rfl⟩

/-- The manufactured boundary has a complete trace for every proof and parsed
value. It reads neither. This is the direct counterexample to treating the
abstract staged structure as deployed refinement evidence. -/
theorem ignoresAllStages_has_trace
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {context : Context Program}
    (lifetime : ExtractedLifetime context)
    (proof : Bytes) (parsed : Parsed) :
    Nonempty
      (ExtractionTrace
        (ignoresAllStages (decode := decode) (terminalAccepts := terminalAccepts)
          lifetime)
        proof parsed lifetime) := by
  exact ⟨
    { canonical := ()
      decodedCanonical := trivial
      terminal := ()
      terminalExtracted := trivial
      fold := ()
      foldExtracted := trivial
      rows := ()
      rowsRefined := trivial
      lifetimeRefined := rfl }⟩

/-- Even terminal acceptance is not inspected by the manufactured boundary.
The concrete release proof must replace every stage with artifact-specific
refinement. -/
theorem ignoresAllStages_refines_every_accepted
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {context : Context Program}
    (lifetime : ExtractedLifetime context)
    (parsed : Parsed) (_accepted : terminalAccepts parsed) :
    ∃ rows,
      (ignoresAllStages (decode := decode) (terminalAccepts := terminalAccepts)
        lifetime).lifetimeRefined rows lifetime := by
  exact ⟨(), rfl⟩

end Nightstream.Assurance.Nebula.ProductionPaperReleasePipelineCountermodels
