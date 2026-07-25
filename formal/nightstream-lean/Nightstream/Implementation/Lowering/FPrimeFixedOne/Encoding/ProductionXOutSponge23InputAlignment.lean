import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement

/-!
Contract: exact coordinate alignment from the Rust-emitted plain/stateless
XOut source program to the selected 23-field Poseidon2 sponge recipe.

Assurance tier: artifact-checked.

Owns:
- selection of the generated plain/stateless XOut source program;
- explicit availability and canonicality obligations for its four encoded
  four-field values;
- equality with the independent protocol-shaped XOut encoder;
- exact 23-field width and index-by-index placement in the sponge input
  bundle;
- transport of active sponge soundness and honest completeness to that exact
  ordered source vector.

Does not own: optional-digest presence or alignment checks, either hash
`CallRecipe`, state/running codecs, generated call-site placement, equality
with a native Poseidon2 implementation, or collision resistance.

Emits constraints: no. This module binds a source vector to an existing
nonoptional sponge receipt without adding rows or columns.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open Nightstream.Protocol.FPrime

namespace Source

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement

/-- The exact Rust-emitted source program selected before inspecting sponge
rows or assignments. -/
def program : Program :=
  GeneratedProgram.select false false

/-- Ordered raw fields emitted by the selected source program. -/
def fields
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) :
    List RawField :=
  execute program table preimage

/-- Load-bearing availability contract for the selected source vector.

`RawEncodingTableWellFormed` alone is insufficient: it constrains entries
that exist but a missing lookup still returns the empty list. These four
width premises name exactly the values read by the plain/stateless program.
Canonicality prevents residue reduction from changing any exported field. -/
structure Available
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) : Prop where
  semanticState_absent :
    preimage.semanticState = none
  nebula_absent :
    preimage.nebula = none
  verifierDigest_width :
    (lookupRawFields table (.digest preimage.vkFsDigest)).length = 4
  piCcsHeader_width :
    (lookupRawFields table (.header preimage.piCcsHeader)).length = 4
  currentBoundary_width :
    (lookupRawFields table (.digest preimage.currentBoundary)).length = 4
  construction2Accumulator_width :
    (lookupRawFields table
      (.digest preimage.construction2Accumulator)).length = 4
  canonical :
    ∀ value, value ∈ fields table preimage ->
      value <
        Nightstream.Implementation.R1CS.goldilocksP

/-- Exact obstruction showing why table well-formedness cannot replace the
four availability premises: the empty table is well formed but every typed
lookup is absent, so this program emits only its domain and six u64 limbs. -/
theorem emptyTable_wellFormed_but_sourceWidth_seven
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) :
    RawEncodingTableWellFormed [] = true ∧
      (fields [] preimage).length = 7 := by
  constructor
  · decide
  · simp [fields, program, GeneratedProgram.select, execute,
      Generated.StateXOutProgram.statelessPlain, instructionFields,
      u64Halves, lookupRawFields]

/-- The selected generated program is extensionally the independent
protocol-shaped encoder once the two omitted optional lanes are known absent. -/
theorem fields_eq_encodeStateXOutPreimage
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest)
    (available : Available table preimage) :
    fields table preimage =
      encodeStateXOutPreimage table preimage := by
  unfold fields program
  have generated :=
    StateXOutProgramRefinement.generated_execute_eq_encodeStateXOutPreimage
      table preimage
  simpa [available.semanticState_absent, available.nebula_absent] using
    generated

/-- The source width is derived from typed component widths, not measured
from a generated sponge trace. -/
theorem fields_length
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest)
    (available : Available table preimage) :
    (fields table preimage).length = 23 := by
  rw [fields_eq_encodeStateXOutPreimage table preimage available]
  rw [encodeStateXOutPreimage_expansion]
  simp [available.semanticState_absent, available.nebula_absent,
    available.verifierDigest_width, available.piCcsHeader_width,
    available.currentBoundary_width,
    available.construction2Accumulator_width, u64Halves]

end Source

open ProductionPoseidon2Sponge23Recipe

/-- Index-by-index alignment between the typed 23-column input bundle and
the selected Rust-emitted source vector. -/
def InputsAligned
    (frame : ProductionPoseidon2Sponge23Recipe.Frame)
    (assignment : ColumnId -> Field)
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) : Prop :=
  ∀ index, index < ProductionPoseidon2Sponge23Recipe.inputWidth ->
    assignment
        (ProductionPoseidon2Sponge23Recipe.inputColumn frame index) =
      residue ((Source.fields table preimage).getD index 0)

private theorem source_getD_canonical
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest)
    (available : Source.Available table preimage)
    (index : Nat)
    (indexLt : index < ProductionPoseidon2Sponge23Recipe.inputWidth) :
    (Source.fields table preimage).getD index 0 <
      Nightstream.Implementation.R1CS.goldilocksP := by
  apply available.canonical
  have fieldsLt : index < (Source.fields table preimage).length := by
    rw [Source.fields_length table preimage available]
    simpa [ProductionPoseidon2Sponge23Recipe.inputWidth] using indexLt
  have member :=
    List.getElem_mem (l := Source.fields table preimage) fieldsLt
  rwa [List.getElem_eq_getD 0] at member

private theorem numericInput_coordinate
    (frame : ProductionPoseidon2Sponge23Recipe.Frame)
    (assignment : ColumnId -> Field)
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest)
    (available : Source.Available table preimage)
    (aligned : InputsAligned frame assignment table preimage)
    (index : Nat)
    (indexLt : index < ProductionPoseidon2Sponge23Recipe.inputWidth) :
    ProductionPoseidon2Sponge23Recipe.initialNumeric frame assignment
        (index + 1) =
      (Source.fields table preimage).getD index 0 := by
  have canonical :=
    source_getD_canonical table preimage available index indexLt
  have canonicalConcrete :
      (Source.fields table preimage).getD index 0 <
        Nightstream.SuperNeo.Concrete.goldilocksModulus := by
    simpa [Nightstream.Implementation.R1CS.goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using canonical
  calc
    ProductionPoseidon2Sponge23Recipe.initialNumeric frame assignment
          (index + 1) =
        (assignment
          (ProductionPoseidon2Sponge23Recipe.inputColumn frame index)).val :=
      ProductionPoseidon2Sponge23Recipe.initialNumeric_input
        frame assignment index indexLt
    _ = (residue ((Source.fields table preimage).getD index 0)).val :=
      congrArg Fin.val (aligned index indexLt)
    _ = (Source.fields table preimage).getD index 0 := by
      exact Nat.mod_eq_of_lt canonicalConcrete

/-- The normalized source columns `1..23` read exactly the Rust-emitted
plain/stateless XOut fields, in the same order and without reduction. -/
theorem numericInputs_eq_sourceFields
    (frame : ProductionPoseidon2Sponge23Recipe.Frame)
    (assignment : ColumnId -> Field)
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest)
    (available : Source.Available table preimage)
    (aligned : InputsAligned frame assignment table preimage) :
    ProductionPoseidon2Sponge23Recipe.NumericSponge.trace.inputColumns.map
        (ProductionPoseidon2Sponge23Recipe.initialNumeric frame assignment) =
      Source.fields table preimage := by
  apply List.ext_get
  · simp [ProductionPoseidon2Sponge23Recipe.NumericSponge.trace,
      Source.fields_length table preimage available]
  · intro index leftLt rightLt
    have indexLt :
        index < ProductionPoseidon2Sponge23Recipe.inputWidth := by
      simpa [ProductionPoseidon2Sponge23Recipe.inputWidth] using leftLt
    simp only [List.get_eq_getElem, List.getElem_map]
    rw [show
      ProductionPoseidon2Sponge23Recipe.NumericSponge.trace.inputColumns[index] =
          index + 1 by
        simp [ProductionPoseidon2Sponge23Recipe.NumericSponge.trace]
        omega]
    rw [List.getElem_eq_getD 0]
    exact
      numericInput_coordinate frame assignment table preimage available
        aligned index indexLt

/-- Pure digest lane of the selected sponge on the exact source vector. -/
def sourceLane
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest)
    (lane : Nat) : Field :=
  residue
    (Nightstream.Implementation.R1CS.Poseidon2Sponge.runValueRounds
      ProductionPoseidon2Sponge23Recipe.NumericSponge.trace.rounds
      (Source.fields table preimage)
      (fun _ => 0) lane)

/-- Once inputs align, the recipe's executable semantic lane is definitionally
the pure sponge lane on the generated source vector. -/
theorem semanticLane_eq_sourceLane
    (frame : ProductionPoseidon2Sponge23Recipe.Frame)
    (assignment : ColumnId -> Field)
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest)
    (available : Source.Available table preimage)
    (aligned : InputsAligned frame assignment table preimage)
    (lane : Nat) :
    ProductionPoseidon2Sponge23Recipe.semanticLane frame assignment lane =
      sourceLane table preimage lane := by
  unfold ProductionPoseidon2Sponge23Recipe.semanticLane sourceLane
  rw [numericInputs_eq_sourceFields frame assignment table preimage
    available aligned]

/-- Active satisfaction binds every visible digest lane to the exact
Rust-emitted plain/stateless XOut source vector. -/
theorem active_sound
    (frame : ProductionPoseidon2Sponge23Recipe.Frame)
    (assignment : ColumnId -> Field)
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest)
    (available : Source.Available table preimage)
    (aligned : InputsAligned frame assignment table preimage)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (holds :
      Satisfies
        (ProductionPoseidon2Sponge23Recipe.rows frame) assignment)
    (lane : Nat)
    (laneLt : lane < ProductionPoseidon2Sponge23Recipe.outputWidth) :
    assignment
        (ProductionPoseidon2Sponge23Recipe.outputColumn frame lane) =
      sourceLane table preimage lane := by
  rw [
    ProductionPoseidon2Sponge23Recipe.active_sound frame assignment
      constantOne activeOne holds lane laneLt,
    semanticLane_eq_sourceLane frame assignment table preimage available
      aligned lane
  ]

/-- If the four visible outputs already equal the exact source-vector digest,
the existing deterministic temporary completion satisfies the active recipe
without changing any visible coordinate. -/
theorem active_complete
    (frame : ProductionPoseidon2Sponge23Recipe.Frame)
    (assignment : ColumnId -> Field)
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest)
    (available : Source.Available table preimage)
    (aligned : InputsAligned frame assignment table preimage)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (outputsCorrect :
      ∀ lane,
        lane < ProductionPoseidon2Sponge23Recipe.outputWidth ->
          assignment
              (ProductionPoseidon2Sponge23Recipe.outputColumn frame lane) =
            sourceLane table preimage lane) :
    Satisfies
      (ProductionPoseidon2Sponge23Recipe.rows frame)
      (ProductionPoseidon2Sponge23Recipe.complete frame assignment) := by
  apply ProductionPoseidon2Sponge23Recipe.active_complete frame assignment
    constantOne activeOne
  intro lane laneLt
  rw [semanticLane_eq_sourceLane frame assignment table preimage available
    aligned lane]
  exact outputsCorrect lane laneLt

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment
