import Lean.Data.Json
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifest
import Nightstream.Implementation.R1CS.Core.SeededAjtai

/-!
Stable JSON encoding for the Lean-owned native CCS F-prime manifest.

Owns: schema version 3 and deterministic JSON names for the complete native
Step program, the unchanged outer Terminal program, the compact direct
terminal-R1CS descriptor, the exact seeded Ajtai setup identity, profile
metadata, codec segments, selectors, activations, and exact costs.

Does not own: deployment selection, file I/O, Rust parsing, witness
generation, or a claim that operator-selected bytes are authoritative.

Emits constraints: no. It encodes the proof-free Lean manifest.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifestJson

open Lean
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRustManifest
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifest
open Nightstream.Implementation.R1CS

def schemaVersion : Nat := 3

def formatName : String := "nightstream/fprime-native-ccs-manifest"

def ajtaiAlgorithm : String := "chacha8_phi81_rejection_v1"

private def array (values : List Json) : Json :=
  Json.arr values.toArray

private def stringArray (values : List String) : Json :=
  array (values.map Json.str)

private def ownerPathSteps : OwnerPath → List String
  | .root => []
  | .rest parent => ownerPathSteps parent ++ ["rest"]
  | .trueArm parent => ownerPathSteps parent ++ ["true_arm"]
  | .falseArm parent => ownerPathSteps parent ++ ["false_arm"]
  | .continuation parent => ownerPathSteps parent ++ ["continuation"]

private def ownerPathJson (path : OwnerPath) : Json :=
  stringArray (ownerPathSteps path)

private def typedOwnerJson (owner : Owner) : Json :=
  match owner with
  | .input slot =>
      Json.mkObj [
        ("kind", "input"),
        ("slot", slot)
      ]
  | .instruction path =>
      Json.mkObj [
        ("kind", "instruction"),
        ("path", ownerPathJson path)
      ]
  | .branch path =>
      Json.mkObj [
        ("kind", "branch"),
        ("path", ownerPathJson path)
      ]

private def physicalOwnerJson (owner : PhysicalOwner) : Json :=
  match owner with
  | .prelude =>
      Json.mkObj [("kind", "prelude")]
  | .typed owner =>
      Json.mkObj [
        ("kind", "typed"),
        ("owner", typedOwnerJson owner)
      ]
  | .branchActivation path selected =>
      Json.mkObj [
        ("kind", "branch_activation"),
        ("path", ownerPathJson path),
        ("selected", selected)
      ]

private def ownershipName : Ownership → String
  | .committedColumn => "committed"
  | .publicColumn => "public"
  | .auxiliaryColumn => "auxiliary"

private def instructionKindName : InstructionKind → String
  | .prelude => "prelude"
  | .input => "input"
  | .literal => "literal"
  | .affine => "affine"
  | .product => "product"
  | .bit => "bit"
  | .call => "call"
  | .assertion => "assertion"
  | .branchControl => "branch_control"
  | .branchJoin => "branch_join"

private def columnIdJson (column : ColumnId) : Json :=
  Json.mkObj [
    ("owner", physicalOwnerJson column.owner),
    ("bundle_index", column.bundleIndex),
    ("coordinate_index", column.coordinateIndex)
  ]

private def rowIdJson (row : RowId) : Json :=
  Json.mkObj [
    ("owner", physicalOwnerJson row.owner),
    ("ordinal", row.ordinal)
  ]

private def ownedColumnJson (column : OwnedColumn) : Json :=
  Json.mkObj [
    ("id", columnIdJson column.id),
    ("ownership", ownershipName column.ownership)
  ]

private def termJson (term : ManifestTerm) : Json :=
  Json.mkObj [
    ("column", columnIdJson term.column),
    ("coefficient", term.coefficient)
  ]

private def combinationJson (combination : ManifestCombination) : Json :=
  array (combination.map termJson)

private def rowJson (row : ManifestRow) : Json :=
  Json.mkObj [
    ("id", rowIdJson row.id),
    ("a", combinationJson row.a),
    ("b", combinationJson row.b),
    ("c", combinationJson row.c)
  ]

private def nativeReceiptJson
    (receipt : NativeCcsManifest.ManifestReceipt) : Json :=
  Json.mkObj [
    ("owner", physicalOwnerJson receipt.owner),
    ("kind", instructionKindName receipt.kind),
    ("allocations", array (receipt.allocations.map ownedColumnJson)),
    ("selector", columnIdJson receipt.selector),
    ("rows", array (receipt.rows.map rowJson))
  ]

private def signName : NativeCcsManifest.Sign → String
  | .positive => "positive"
  | .negative => "negative"

private def polynomialTermJson
    (term : NativeCcsManifest.PolynomialTerm) : Json :=
  Json.mkObj [
    ("sign", signName term.sign),
    ("exponents",
      array (term.exponents.map fun (exponent : Nat) =>
        Json.num (exponent : JsonNumber)))
  ]

private def nativeProgramJson
    (program : NativeCcsManifest.Program) : Json :=
  Json.mkObj [
    ("one", columnIdJson program.one),
    ("matrix_count", program.matrixCount),
    ("polynomial_degree", program.polynomialDegree),
    ("polynomial", array (program.polynomial.map polynomialTermJson)),
    ("receipts", array (program.receipts.map nativeReceiptJson))
  ]

private def canonicalReceiptJson
    (receipt : CanonicalManifest.ManifestReceipt) : Json :=
  Json.mkObj [
    ("owner", physicalOwnerJson receipt.owner),
    ("kind", instructionKindName receipt.kind),
    ("allocations", array (receipt.allocations.map ownedColumnJson)),
    ("rows", array (receipt.rows.map rowJson))
  ]

private def canonicalProgramJson
    (program : CanonicalManifest.Program) : Json :=
  Json.mkObj [
    ("one", columnIdJson program.one),
    ("receipts", array (program.receipts.map canonicalReceiptJson))
  ]

private def costJson (cost : Cost) : Json :=
  Json.mkObj [
    ("recurring_rows", cost.recurringRows),
    ("committed_columns", cost.committedColumns),
    ("public_columns", cost.publicColumns),
    ("auxiliary_columns", cost.auxiliaryColumns)
  ]

private def terminalR1csJson
    (descriptor : TerminalR1cs.Descriptor) : Json :=
  Json.mkObj [
    ("row_variables", descriptor.rowVariables),
    ("logical_width", descriptor.logicalWidth),
    ("recursive_rows", descriptor.recursiveRows),
    ("matrix_count", descriptor.matrixCount),
    ("public_ring_columns", descriptor.publicRingColumns),
    ("verifier_rows", descriptor.verifierRows),
    ("cost", costJson descriptor.cost)
  ]

private def ajtaiSetupJson (identity : SeededAjtai.Identity) : Json :=
  Json.mkObj [
    ("algorithm", ajtaiAlgorithm),
    ("seed",
      array (identity.seed.map fun byte =>
        Json.num (JsonNumber.fromNat byte))),
    ("rejection_fuel", identity.rejectionFuel)
  ]

private def profileName : ProfileName → String
  | .fixedOnePlain270 => "fixed_one_plain_270"

private def profileJson (profile : ProfileIdentifier) : Json :=
  Json.mkObj [
    ("name", profileName profile.name),
    ("matrix_count", profile.matrixCount),
    ("fresh_source_count", profile.freshSourceCount),
    ("running_source_count", profile.runningSourceCount),
    ("public_carrier_width", profile.publicCarrierWidth),
    ("fresh_legacy_width", profile.freshLegacyWidth),
    ("fresh_completion_width", profile.freshCompletionWidth),
    ("running_carrier_width", profile.runningCarrierWidth),
    ("poseidon_width", profile.poseidonWidth),
    ("poseidon_rate", profile.poseidonRate),
    ("poseidon_capacity", profile.poseidonCapacity),
    ("poseidon_digest_width", profile.poseidonDigestWidth),
    ("binding_preimage_width", profile.bindingPreimageWidth),
    ("decomposition_base", profile.decompositionBase),
    ("decomposition_children", profile.decompositionChildren)
  ]

private def widthsJson (widths : Widths) : Json :=
  Json.mkObj [
    ("iteration", widths.iteration),
    ("state", widths.state),
    ("witness", widths.witness),
    ("running", widths.running),
    ("fresh", widths.fresh),
    ("nifs_proof", widths.nifsProof),
    ("digest", widths.digest),
    ("encoded", widths.encoded),
    ("running_witness", widths.runningWitness),
    ("fresh_witness", widths.freshWitness),
    ("bit", widths.bit)
  ]

private def segmentRoleName : SegmentRole → String
  | .iteration => "iteration"
  | .initialState => "initial_state"
  | .currentState => "current_state"
  | .running => "running"
  | .fresh => "fresh"
  | .witness => "witness"
  | .nifsProof => "nifs_proof"
  | .nextState => "next_state"
  | .nextRunning => "next_running"
  | .digest => "digest"
  | .runningWitness => "running_witness"
  | .freshWitness => "fresh_witness"

private def segmentJson (segment : CodecSegment) : Json :=
  Json.mkObj [
    ("role", segmentRoleName segment.role),
    ("width", segment.width),
    ("ownership", ownershipName segment.ownership),
    ("offset", segment.offset)
  ]

/-- Schema-versioned JSON value containing every proof-free native manifest
field. No legacy activated Step rows are included. -/
def toJson
    (ajtaiSetup : SeededAjtai.Identity)
    (manifest : ConcreteNifsNativeCcsManifest.Manifest) : Json :=
  Json.mkObj [
    ("schema", schemaVersion),
    ("format", formatName),
    ("goldilocks_modulus",
      Nightstream.SuperNeo.Concrete.goldilocksModulus),
    ("ajtai_setup", ajtaiSetupJson ajtaiSetup),
    ("profile", profileJson manifest.profile),
    ("widths", widthsJson manifest.widths),
    ("step_input", array (manifest.stepInput.map segmentJson)),
    ("step_result", array (manifest.stepResult.map segmentJson)),
    ("terminal_input", array (manifest.terminalInput.map segmentJson)),
    ("step_program", nativeProgramJson manifest.stepProgram),
    ("terminal_program", canonicalProgramJson manifest.terminalProgram),
    ("terminal_r1cs", terminalR1csJson manifest.terminalR1cs),
    ("step_result_columns",
      array (manifest.stepResultColumns.map ownedColumnJson)),
    ("step_selector", columnIdJson manifest.stepSelector),
    ("terminal_selector", columnIdJson manifest.terminalSelector),
    ("step_activations",
      array (manifest.stepActivations.map columnIdJson)),
    ("terminal_activations",
      array (manifest.terminalActivations.map columnIdJson)),
    ("step_cost", costJson manifest.stepCost),
    ("terminal_cost", costJson manifest.terminalCost)
  ]

/-- Deterministic compact wire representation. -/
def render
    (ajtaiSetup : SeededAjtai.Identity)
    (manifest : ConcreteNifsNativeCcsManifest.Manifest) : String :=
  (toJson ajtaiSetup manifest).compress ++ "\n"

theorem render_exact
    (ajtaiSetup : SeededAjtai.Identity)
    (manifest : ConcreteNifsNativeCcsManifest.Manifest) :
    render ajtaiSetup manifest =
      (toJson ajtaiSetup manifest).compress ++ "\n" :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifestJson
