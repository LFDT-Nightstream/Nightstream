import Lean.Data.Json
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.Module
import Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsCompiler
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest

/-!
Contract: compact deployment manifest for one Lean-owned WASM application
proof program.

Assurance tier: model-level.

Owns: schema version 1, exact module bytes, ordered physical columns,
normalized native-CCS rows, selector columns, and derived density metrics.

Does not own: an application relation, file I/O, Rust parsing, Spartan, WHIR,
or a security reduction. A caller supplies the proof-carrying program and a
role for each of its already-allocated columns.

Emits constraints: none. It serializes an existing program.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ApplicationProofManifest

open Lean
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm

def schemaVersion : Nat := 1

def formatName : String := "nightstream/wasm-application-proof"

/-- Verifier meaning of one column in the ordered Lean allocation stream. -/
inductive ColumnRole where
  | one
  | moduleByte (index : Nat)
  | privateWitness (index : Nat)
  | output (index : Nat)
deriving DecidableEq, Repr

structure ColumnBinding where
  column : ColumnId
  role : ColumnRole
deriving DecidableEq, Repr

private def array (values : List Json) : Json :=
  Json.arr values.toArray

private def columnIndex
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (column : ColumnId) : Nat :=
  (NativeCcsCompiler.ColumnIndex.index program valid column).val

private def roleName : ColumnRole → String
  | .one => "one"
  | .moduleByte _ => "module_byte"
  | .privateWitness _ => "private_witness"
  | .output _ => "output"

private def roleIndex : ColumnRole → Nat
  | .one => 0
  | .moduleByte index => index
  | .privateWitness index => index
  | .output index => index

private def bindingJson
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (binding : ColumnBinding) : Json :=
  Json.mkObj
    [ ("index", columnIndex program valid binding.column)
    , ("role", roleName binding.role)
    , ("role_index", roleIndex binding.role)
    ]

private def termJson
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (term : ManifestTerm) : Json :=
  Json.mkObj
    [ ("column", columnIndex program valid term.column)
    , ("coefficient", term.coefficient)
    ]

private def combinationJson
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (combination : ManifestCombination) : Json :=
  array (combination.map (termJson program valid))

private def rowJson
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (ordinal : Nat)
    (row : NativeCcsSelector.SelectedRow) : Json :=
  let normalized := ManifestRow.ofOwnedRow row.source
  Json.mkObj
    [ ("ordinal", ordinal)
    , ("selector", columnIndex program valid row.selector)
    , ("a", combinationJson program valid normalized.a)
    , ("b", combinationJson program valid normalized.b)
    , ("c", combinationJson program valid normalized.c)
    ]

private def indexedRowsFrom :
    Nat → List NativeCcsSelector.SelectedRow →
      List (Nat × NativeCcsSelector.SelectedRow)
  | _, [] => []
  | index, row :: rest =>
      (index, row) :: indexedRowsFrom (index + 1) rest

private def indexedRows
    (rows : List NativeCcsSelector.SelectedRow) :
    List (Nat × NativeCcsSelector.SelectedRow) :=
  indexedRowsFrom 0 rows

private def rowsJson
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program) : Json :=
  array (indexedRows program.rows |>.map fun pair =>
    rowJson program valid pair.1 pair.2)

private def polynomialTermJson
    (term : NativeCcsManifest.PolynomialTerm) : Json :=
  let sign := match term.sign with
    | .positive => "positive"
    | .negative => "negative"
  Json.mkObj
    [ ("sign", sign)
    , ("exponents", array (term.exponents.map fun (exponent : Nat) =>
        Json.num (exponent : JsonNumber)))
    ]

def normalizedRows
    (program : NativeCcsProgram.Program) : List ManifestRow :=
  program.rows.map fun row => ManifestRow.ofOwnedRow row.source

def r1csNonzeroCoefficients
    (program : NativeCcsProgram.Program) : Nat :=
  (normalizedRows program).foldl
    (fun total row => total + row.support) 0

def nativeCcsNonzeroCoefficients
    (program : NativeCcsProgram.Program) : Nat :=
  r1csNonzeroCoefficients program + program.rows.length

def maximumR1csRowDensity
    (program : NativeCcsProgram.Program) : Nat :=
  (normalizedRows program).foldl
    (fun largest row => max largest row.support) 0

def maximumNativeCcsRowDensity
    (program : NativeCcsProgram.Program) : Nat :=
  (normalizedRows program).foldl
    (fun largest row => max largest (row.support + 1)) 0

private def costJson (cost : Cost) : Json :=
  Json.mkObj
    [ ("rows", cost.recurringRows)
    , ("committed_columns", cost.committedColumns)
    , ("lean_public_columns", cost.publicColumns)
    , ("auxiliary_columns", cost.auxiliaryColumns)
    ]

private def metricsJson
    (program : NativeCcsProgram.Program)
    (poseidon2Calls maximumLiveWitnessColumns : Nat) : Json :=
  Json.mkObj
    [ ("r1cs_nonzero_coefficients", r1csNonzeroCoefficients program)
    , ("native_ccs_nonzero_coefficients",
        nativeCcsNonzeroCoefficients program)
    , ("maximum_r1cs_row_density", maximumR1csRowDensity program)
    , ("maximum_native_ccs_row_density",
        maximumNativeCcsRowDensity program)
    , ("poseidon2_calls", poseidon2Calls)
    , ("maximum_live_witness_columns", maximumLiveWitnessColumns)
    ]

/-- Deterministic proof manifest. The program is the authority for all rows,
costs, selectors, and density values. -/
def toJson
    (identifier : String)
    (module : CertifiedModule)
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (bindings : List ColumnBinding)
    (poseidon2Calls maximumLiveWitnessColumns : Nat) : Json :=
  Json.mkObj
    [ ("schema", schemaVersion)
    , ("format", formatName)
    , ("module_id", identifier)
    , ("module_hex", bytesHex module.bytes)
    , ("goldilocks_modulus",
        Nightstream.SuperNeo.Concrete.goldilocksModulus)
    , ("matrix_count", NativeCcsSelector.matrixCount)
    , ("polynomial_degree", NativeCcsSelector.polynomialDegree)
    , ("polynomial", array
        (NativeCcsManifest.selectorPolynomial.map polynomialTermJson))
    , ("columns", array (bindings.map (bindingJson program valid)))
    , ("rows", rowsJson program valid)
    , ("cost", costJson program.cost)
    , ("metrics", metricsJson program poseidon2Calls
        maximumLiveWitnessColumns)
    ]

def render
    (identifier : String)
    (module : CertifiedModule)
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (bindings : List ColumnBinding)
    (poseidon2Calls maximumLiveWitnessColumns : Nat) : String :=
  (toJson identifier module program valid bindings poseidon2Calls
    maximumLiveWitnessColumns).compress ++ "\n"

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ApplicationProofManifest
