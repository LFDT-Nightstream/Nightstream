import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTraceRefinement
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Poseidon2PermutationSound
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Sbox7OutputLayout

/-!
Contract: refinement from the executable 600-definition Poseidon2 artifact to
the canonical Lean Poseidon2 reference permutation.

Owns the semantic bridge between
`Poseidon2PermutationSound.permute`, which is extracted from the exact Rust
SSA artifact, and `Poseidon2Reference.referencePermutation`, which is defined
from the Lean-owned round schedule and constants.

Does not own call-site row inclusion, transcript framing, collision security,
or a protocol operation schedule.

The proof does not compare one captured witness. It proves that the SSA
interpreter satisfies all 86 exact S-box equations and all eight exact output
forms for every canonical input.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.Poseidon2ExtractedReference

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

namespace Artifact

abbrev definitions := Poseidon2Permutation.definitions
abbrev inputColumns := Poseidon2Permutation.inputColumns
abbrev sites := Poseidon2Sbox7OutputLayout.sites

end Artifact

def sourceAssignment (lanes : Nat → Nat) : Nat → Nat :=
  Poseidon2PermutationSound.inputOnly
    (Poseidon2PermutationSound.permutationAssignment lanes)

def execution (lanes : Nat → Nat) : Nat → Nat :=
  Program.run (sourceAssignment lanes) Artifact.definitions

theorem source_canonical
    {lanes : Nat → Nat}
    (canonical : ∀ lane, lane < 8 → lanes lane < goldilocksP) :
    ∀ column, sourceAssignment lanes column < goldilocksP := by
  intro column
  unfold sourceAssignment Poseidon2PermutationSound.inputOnly
  by_cases columnLt : column < 9
  · simp only [columnLt, ↓reduceIte,
      Poseidon2PermutationSound.permutationAssignment]
    by_cases columnZero : column = 0
    · simp [columnZero, goldilocksP]
    · simp only [columnZero, ↓reduceIte]
      exact canonical (column - 1) (by omega)
  · simp [columnLt, goldilocksP]

theorem execution_canonical
    {lanes : Nat → Nat}
    (canonical : ∀ lane, lane < 8 → lanes lane < goldilocksP) :
    ∀ column, execution lanes column < goldilocksP := by
  exact Program.run_canonical (source_canonical canonical)

private theorem getElem?_eq_some_mem
    {alpha : Type} {entries : List alpha} {index : Nat} {value : alpha}
    (lookup : entries[index]? = some value) : value ∈ entries := by
  rcases getElem?_eq_some_iff.mp lookup with ⟨bounded, equal⟩
  rw [← equal]
  exact List.getElem_mem (l := entries) bounded

private theorem definition_holds_of_lookup
    (lanes : Nat → Nat) {index : Nat} {definition : Definition}
    (lookup : Artifact.definitions[index]? = some definition) :
    Definition.Holds (execution lanes) definition := by
  exact Program.run_definitions_hold Poseidon2Permutation.definitions_wellFormed
    (sourceAssignment lanes) definition (getElem?_eq_some_mem lookup)

private theorem singleton_lc
    {lanes : Nat → Nat}
    (canonical : ∀ lane, lane < 8 → lanes lane < goldilocksP)
    (column : Nat) :
    lcEval (execution lanes) [(column, 1)] = execution lanes column := by
  simp [lcEval, Nat.mod_eq_of_lt (execution_canonical canonical column)]

/-! ## Exact artifact views -/

private def siteFor (index : Fin sboxCount) : Poseidon2Sbox7OutputLayout.Site :=
  Artifact.sites.get ⟨index.val, by
    have bounded : index.val < 86 := by
      simpa only [sboxCount_eq] using index.isLt
    simpa only [Poseidon2Sbox7OutputLayout.family_census.2.2.2] using bounded⟩

private theorem siteFor_mem (index : Fin sboxCount) :
    siteFor index ∈ Artifact.sites := by
  exact List.get_mem Artifact.sites _

/-- The compact trace and the topological-layout audit enumerate the same 86
S-box sites in the same order. -/
private theorem siteFor_output_exact :
    ∀ index : Fin sboxCount,
      (siteFor index).outputColumn = traceSboxOutput index := by
  native_decide

/-! ## Exact symbolic elimination of materialized linear state wires -/

private abbrev ImageTable := List (Nat × Poseidon2Core.LinComb)

/-- Read the newest symbolic image of a column. Columns without a derived
image are retained as atomic source columns. -/
private def imageOf : ImageTable → Nat → Poseidon2Core.LinComb
  | [], column => [(column, 1)]
  | entry :: rest, column =>
      if column = entry.1 then entry.2 else imageOf rest column

/-- Substitute the current images into one linear form, merge duplicate
columns, reduce coefficients, and remove zero terms. -/
private def expandedForm (table : ImageTable)
    (form : Poseidon2Core.LinComb) : Poseidon2Core.LinComb :=
  fieldNormalize (LinearSubstitution.terms (imageOf table) form)

/-- Linear definitions receive their exact symbolic image. Product outputs
remain atomic because a compact Poseidon2 trace retains the final `x^7`
outputs and does not linearize nonlinear products. -/
private def addDefinition (table : ImageTable)
    (definition : Definition) : ImageTable :=
  let image := match definition.rhs with
    | .linear terms => expandedForm table terms
    | .product _ _ => [(definition.output, 1)]
  (definition.output, image) :: table

private def compileImages : List Definition → ImageTable → ImageTable
  | [], table => table
  | definition :: rest, table =>
      compileImages rest (addDefinition table definition)

private def artifactImages : ImageTable :=
  compileImages Artifact.definitions []

private def ImagesSound (z : Nat → Nat) (table : ImageTable) : Prop :=
  ∀ column, lcEval z (imageOf table column) = z column

private theorem singleton_lc_of_canonical
    {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (column : Nat) :
    lcEval z [(column, 1)] = z column := by
  simp [lcEval, Nat.mod_eq_of_lt (canonical column)]

private theorem empty_images_sound
    {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP) :
    ImagesSound z [] := by
  intro column
  exact singleton_lc_of_canonical canonical column

private theorem expandedForm_eval
    {z : Nat → Nat} {table : ImageTable}
    (sound : ImagesSound z table)
    (form : Poseidon2Core.LinComb) :
    lcEval z (expandedForm table form) = lcEval z form := by
  calc
    lcEval z (expandedForm table form) =
        lcEval z (LinearSubstitution.terms (imageOf table) form) :=
      lcEval_fieldNormalize z _
    _ = lcEval (LinearSubstitution.assignment (imageOf table) z) form :=
      LinearSubstitution.lcEval_terms (imageOf table) z form
    _ = lcEval z form := by
      congr 1
      funext column
      exact sound column

private theorem addDefinition_sound
    {z : Nat → Nat} {table : ImageTable}
    (canonical : ∀ column, z column < goldilocksP)
    (sound : ImagesSound z table)
    (definition : Definition)
    (holds : Definition.Holds z definition) :
    ImagesSound z (addDefinition table definition) := by
  intro column
  cases definition with
  | mk output rhs =>
      cases rhs with
      | linear terms =>
          by_cases same : column = output
          · subst column
            simpa [addDefinition, imageOf, Definition.Holds, Rhs.eval] using
              (expandedForm_eval sound terms).trans holds.symm
          · simpa [addDefinition, imageOf, same] using sound column
      | product left right =>
          by_cases same : column = output
          · subst column
            simpa [addDefinition, imageOf] using
              singleton_lc_of_canonical canonical output
          · simpa [addDefinition, imageOf, same] using sound column

private theorem compileImages_sound
    {z : Nat → Nat} {definitions : List Definition} {table : ImageTable}
    (canonical : ∀ column, z column < goldilocksP)
    (sound : ImagesSound z table)
    (holds : ∀ definition ∈ definitions,
      Definition.Holds z definition) :
    ImagesSound z (compileImages definitions table) := by
  induction definitions generalizing table with
  | nil => exact sound
  | cons head tail hypothesis =>
      apply hypothesis
      · exact addDefinition_sound canonical sound head
          (holds head (by simp))
      · intro definition member
        exact holds definition (by simp [member])

private theorem artifactImages_sound
    {lanes : Nat → Nat}
    (canonical : ∀ lane, lane < 8 → lanes lane < goldilocksP) :
    ImagesSound (execution lanes) artifactImages := by
  exact compileImages_sound (execution_canonical canonical)
    (empty_images_sound (execution_canonical canonical))
    (Program.run_definitions_hold Poseidon2Permutation.definitions_wellFormed
      (sourceAssignment lanes))

/-- The compact trace is exactly the full SSA trace after eliminating every
materialized linear state wire. This is a bounded artifact check, not a
witness sample. -/
private theorem siteFor_input_expansion_exact :
    ∀ index : Fin sboxCount,
      traceTerms (expandedForm artifactImages (siteFor index).affineInput) =
        traceTerms (traceSboxInput index) := by
  native_decide

private def finalIntermediateColumn (lane : Fin width) : Nat :=
  593 + lane.val

private def finalRawForm (lane : Fin width) : Poseidon2Core.LinComb :=
  match Artifact.definitions[584 + lane.val]? with
  | some ⟨_, .linear terms⟩ => terms
  | _ => []

private theorem final_intermediate_definition_exact :
    ∀ lane : Fin width,
      Artifact.definitions[584 + lane.val]? =
        some ⟨finalIntermediateColumn lane, .linear (finalRawForm lane)⟩ := by
  native_decide

private theorem final_output_definition_exact :
    ∀ lane : Fin width,
      Artifact.definitions[592 + lane.val]? =
        some ⟨traceOutputColumn lane,
          .linear [(finalIntermediateColumn lane, 1)]⟩ := by
  native_decide

private theorem final_raw_form_exact :
    ∀ lane : Fin width,
      traceTerms (finalRawForm lane) = traceTerms (traceFinalForm lane) := by
  native_decide

private theorem trace_input_column_exact :
    ∀ lane : Fin width, traceInputColumn lane = lane.val + 1 := by
  native_decide

private theorem trace_output_column_exact :
    ∀ lane : Fin width, traceOutputColumn lane = 601 + lane.val := by
  native_decide

/-! ## The extracted execution satisfies the compact semantic trace -/

private theorem execution_one (lanes : Nat → Nat) :
    execution lanes 0 = 1 := by
  have preserved := Program.run_preserves_known
    Poseidon2Permutation.definitions_wellFormed (sourceAssignment lanes)
    0 (by simp [Poseidon2Permutation.inputColumns])
  simpa [execution, sourceAssignment,
    Poseidon2PermutationSound.inputOnly,
    Poseidon2PermutationSound.permutationAssignment] using preserved

private theorem execution_input
    (lanes : Nat → Nat) (lane : Fin width) :
    execution lanes (traceInputColumn lane) = lanes lane.val := by
  have member : traceInputColumn lane ∈ Artifact.inputColumns := by
    rw [trace_input_column_exact]
    have laneLt : lane.val < 8 := by
      simpa only [width] using lane.isLt
    simp only [Poseidon2Permutation.inputColumns, List.mem_cons,
      List.not_mem_nil, or_false]
    omega
  have preserved := Program.run_preserves_known
    Poseidon2Permutation.definitions_wellFormed (sourceAssignment lanes)
    (traceInputColumn lane) member
  rw [trace_input_column_exact] at preserved ⊢
  have laneLt : lane.val < 8 := by
    simpa only [width] using lane.isLt
  have columnLt : lane.val + 1 < 9 := by omega
  simpa [execution, sourceAssignment,
    Poseidon2PermutationSound.inputOnly,
    Poseidon2PermutationSound.permutationAssignment, columnLt] using preserved

private theorem execution_sbox
    {lanes : Nat → Nat}
    (canonical : ∀ lane, lane < 8 → lanes lane < goldilocksP)
    (index : Fin sboxCount) :
    execution lanes (traceSboxOutput index) =
      Poseidon2Reference.sbox7
        (lcEval (execution lanes) (traceSboxInput index)) := by
  let site := siteFor index
  have exact := Poseidon2Sbox7OutputLayout.topologicalDefinitions_exact
    site (siteFor_mem index)
  rcases exact with ⟨_affine, x2Lookup, x4Lookup, x6Lookup, outputLookup⟩
  have x2Holds := definition_holds_of_lookup lanes x2Lookup
  have x4Holds := definition_holds_of_lookup lanes x4Lookup
  have x6Holds := definition_holds_of_lookup lanes x6Lookup
  have outputHolds := definition_holds_of_lookup lanes outputLookup
  have x2 :
      execution lanes site.x2Column =
        lcEval (execution lanes) site.affineInput *
          lcEval (execution lanes) site.affineInput % goldilocksP := by
    simpa [Definition.Holds, Rhs.eval] using x2Holds
  have x4 :
      execution lanes site.x4Column =
        execution lanes site.x2Column * execution lanes site.x2Column %
          goldilocksP := by
    simpa [Definition.Holds, Rhs.eval, singleton_lc canonical] using x4Holds
  have x6 :
      execution lanes site.x6Column =
        execution lanes site.x2Column * execution lanes site.x4Column %
          goldilocksP := by
    simpa [Definition.Holds, Rhs.eval, singleton_lc canonical] using x6Holds
  have output :
      execution lanes site.outputColumn =
        lcEval (execution lanes) site.affineInput *
          execution lanes site.x6Column % goldilocksP := by
    simpa [Definition.Holds, Rhs.eval, singleton_lc canonical] using outputHolds
  have inputEqual :
      lcEval (execution lanes) site.affineInput =
        lcEval (execution lanes) (traceSboxInput index) := by
    calc
      lcEval (execution lanes) site.affineInput =
          lcEval (execution lanes)
            (expandedForm artifactImages site.affineInput) :=
        (expandedForm_eval (artifactImages_sound canonical)
          site.affineInput).symm
      _ = lcEval (execution lanes)
          (traceTerms (expandedForm artifactImages site.affineInput)) :=
        (lcEval_traceTerms (execution lanes)
          (expandedForm artifactImages site.affineInput)).symm
      _ = lcEval (execution lanes)
          (traceTerms (traceSboxInput index)) := by
        rw [siteFor_input_expansion_exact index]
      _ = lcEval (execution lanes) (traceSboxInput index) :=
        lcEval_traceTerms (execution lanes) (traceSboxInput index)
  rw [← siteFor_output_exact index, ← inputEqual]
  calc
    execution lanes site.outputColumn =
        lcEval (execution lanes) site.affineInput *
          execution lanes site.x6Column % goldilocksP := output
    _ = lcEval (execution lanes) site.affineInput *
          (execution lanes site.x2Column *
            execution lanes site.x4Column % goldilocksP) % goldilocksP := by
        rw [x6]
    _ = lcEval (execution lanes) site.affineInput *
          (execution lanes site.x2Column *
            (execution lanes site.x2Column *
              execution lanes site.x2Column % goldilocksP) % goldilocksP) %
            goldilocksP := by
        rw [x4]
    _ = Poseidon2Reference.sbox7
          (lcEval (execution lanes) site.affineInput) := by
        rw [x2]
        rfl

private theorem execution_output
    {lanes : Nat → Nat}
    (canonical : ∀ lane, lane < 8 → lanes lane < goldilocksP)
    (lane : Fin width) :
    execution lanes (traceOutputColumn lane) =
      lcEval (execution lanes) (traceFinalForm lane) := by
  have intermediateHolds := definition_holds_of_lookup lanes
    (final_intermediate_definition_exact lane)
  have outputHolds := definition_holds_of_lookup lanes
    (final_output_definition_exact lane)
  have intermediate :
      execution lanes (finalIntermediateColumn lane) =
        lcEval (execution lanes) (finalRawForm lane) := by
    simpa [Definition.Holds, Rhs.eval] using intermediateHolds
  have output :
      execution lanes (traceOutputColumn lane) =
        execution lanes (finalIntermediateColumn lane) := by
    simpa [Definition.Holds, Rhs.eval, singleton_lc canonical] using outputHolds
  calc
    execution lanes (traceOutputColumn lane) =
        execution lanes (finalIntermediateColumn lane) := output
    _ = lcEval (execution lanes) (finalRawForm lane) := intermediate
    _ = lcEval (execution lanes) (traceTerms (finalRawForm lane)) :=
      (lcEval_traceTerms (execution lanes) (finalRawForm lane)).symm
    _ = lcEval (execution lanes) (traceTerms (traceFinalForm lane)) := by
      rw [final_raw_form_exact lane]
    _ = lcEval (execution lanes) (traceFinalForm lane) :=
      lcEval_traceTerms (execution lanes) (traceFinalForm lane)

private theorem execution_trace_holds
    {lanes : Nat → Nat}
    (canonical : ∀ lane, lane < 8 → lanes lane < goldilocksP) :
    TraceHolds (execution lanes) where
  constantWire := execution_one lanes
  sboxes := execution_sbox canonical
  outputs := execution_output canonical

/-- The executable full SSA artifact computes the canonical Lean reference
permutation for every canonical input. -/
theorem execution_computes_reference
    {lanes : Nat → Nat}
    (canonical : ∀ lane, lane < 8 → lanes lane < goldilocksP)
    (lane : Fin width) :
    execution lanes (traceOutputColumn lane) =
      referencePermutation Poseidon2CanonicalConstants.selected
        (fun inputLane => lanes inputLane.val) lane := by
  have computes := trace_computes_reference (execution lanes)
    (execution_canonical canonical) (execution_trace_holds canonical) lane
  apply computes.trans
  congr 2
  funext inputLane
  exact execution_input lanes inputLane

/-- Headline semantic bridge: the function extracted from the exact Rust
600-row artifact is the Lean-owned selected Poseidon2 permutation. -/
theorem permute_eq_reference
    {lanes : Nat → Nat}
    (canonical : ∀ lane, lane < 8 → lanes lane < goldilocksP)
    (lane : Fin width) :
    Poseidon2PermutationSound.permute lanes lane.val =
      referencePermutation Poseidon2CanonicalConstants.selected
        (fun inputLane => lanes inputLane.val) lane := by
  rw [Poseidon2PermutationSound.permute_eq]
  change execution lanes (601 + lane.val) = _
  rw [← trace_output_column_exact lane]
  exact execution_computes_reference canonical lane

end Nightstream.Implementation.R1CS.Poseidon2ExtractedReference
