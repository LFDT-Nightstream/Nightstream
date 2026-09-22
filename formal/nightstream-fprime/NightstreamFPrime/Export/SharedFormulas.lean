import Lean.Data.Json
import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Layout.MatrixProgram.Phi81Product
import NightstreamFPrime.Layout.MatrixProgram.Poseidon

/-!
Exports the existing fixed Poseidon2 and Phi81 row formulas with local ports.
Linear-form registers retain sharing across Poseidon2 steps. Registers are
export temporaries; they allocate no circuit variables or constraint rows.
The existing matrix blocks still own invocation order and physical geometry.
-/

namespace NightstreamFPrime.Export.SharedFormulas

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec

structure Port where
  name : String
  role : String
  start : Nat
  count : Nat

structure Variant where
  linearForms : List WireForm
  rows : List (List WireForm)
  outputRegisters : List Nat

structure Component where
  id : String
  inputCount : Nat
  ports : List Port
  variantCount : Nat
  variant : Fin variantCount → Variant
  definitions : List String
  contracts : List String

private def wireRow {width : Nat} (row : RowForms width) : List WireForm :=
  List.ofFn fun port => WireForm.ofSemantic (row port)

private def wireState {width : Nat} (state : SparseLayer.State width) :
    List WireForm :=
  List.ofFn fun lane => WireForm.ofSemantic (state lane)

private def poseidonInputs : Nat := 1 + 8 + PoseidonRetainedSlots.rows.length

private def poseidonWidth : Nat :=
  poseidonInputs + 8 * Permutation.schedule.length

private def poseidonInterface : PoseidonSboxPlan.Interface poseidonWidth :=
  let sboxOutput := fun slot : Fin PoseidonRetainedSlots.rows.length =>
    SparseForm.singleton
      ⟨9 + slot.val, by
        have bound := slot.isLt
        simp only [poseidonWidth, poseidonInputs]
        omega⟩ 1
  { oneColumn := ⟨0, by simp [poseidonWidth, poseidonInputs]⟩
    input := fun lane => SparseForm.singleton
      ⟨1 + lane.val, by
        have bound := lane.isLt
        simp only [poseidonWidth, poseidonInputs]
        omega⟩ 1
    sboxOutput
    output := SparseLayer.external fun lane => sboxOutput
      ⟨78 + lane.val, by
        have bound := lane.isLt
        rw [PoseidonRetainedSlots.rows_length]
        omega⟩ }

private def registerState (step : Fin Permutation.schedule.length) :
    SparseLayer.State poseidonWidth :=
  fun lane => SparseForm.singleton
    ⟨poseidonInputs + 8 * step.val + lane.val, by
      have stepBound := step.isLt
      have laneBound := lane.isLt
      unfold poseidonWidth
      omega⟩ 1

private structure PoseidonAccumulator where
  nextSbox : Nat
  state : SparseLayer.State poseidonWidth
  linearForms : List WireForm
  rows : List (List WireForm)

private def exportPoseidonStep (accumulated : PoseidonAccumulator)
    (index : Fin Permutation.schedule.length) : PoseidonAccumulator :=
  let result := PoseidonSboxPlan.compileStep poseidonInterface
    accumulated.nextSbox accumulated.state (Permutation.schedule.get index)
  { nextSbox := result.nextSbox
    state := registerState index
    linearForms := accumulated.linearForms ++ wireState result.state
    rows := accumulated.rows ++
      result.rows.map (fun row => wireRow row.meaningfulForm) }

/-- Export the existing final pins without constructing the preceding trace. -/
def directOutputRows {width : Nat} (interface : PoseidonSboxPlan.Interface width) :
    List (PinRow.Forms width) :=
  List.ofFn fun lane =>
    { selector := PoseidonSboxPlan.selector interface
      value := SparseForm.add (interface.output lane)
        (SparseForm.scale (-1) (PoseidonSboxPlan.directOutput interface lane)) }

theorem directOutputRows_eq {width : Nat} (interface : PoseidonSboxPlan.Interface width) :
    directOutputRows interface = PoseidonSboxPlan.outputRows interface := by
  simp only [directOutputRows, PoseidonSboxPlan.outputRows,
    PoseidonSboxPlan.outputDifference, PoseidonSboxPlan.trace_state_eq_directOutput]

/-- Serialize each existing step against local state ports before substituting
its predecessor. This keeps the partial-round linear sums as a DAG. -/
def poseidonVariant (_ : Unit) : Variant :=
  let initial : PoseidonAccumulator :=
    { nextSbox := 0, state := poseidonInterface.input
      linearForms := [], rows := [] }
  let compiled := (List.finRange Permutation.schedule.length).foldl
    exportPoseidonStep initial
  { linearForms := compiled.linearForms
    rows := compiled.rows ++
      (directOutputRows poseidonInterface).map
        (fun row => wireRow row.meaningfulForm)
    outputRegisters := (List.range 8).map fun lane =>
      poseidonInputs + 8 * (Permutation.schedule.length - 1) + lane }

def poseidonComponent (_ : Unit) : Component where
  id := "poseidon2-permutation-v1"
  inputCount := poseidonInputs
  ports := [⟨"one", "constant", 0, 1⟩, ⟨"input", "input", 1, 8⟩,
    ⟨"sbox_output", "witness", 9, PoseidonRetainedSlots.rows.length⟩]
  variantCount := 1
  variant := fun _ => poseidonVariant ()
  definitions := [
    "NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan.compileStep",
    "NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan.outputRows",
    "NightstreamFPrime.Gadgets.Poseidon2.Permutation.schedule"]
  contracts := [
    "NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan.compileStep_sound",
    "NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan.rowsZero_implies_permute",
    "NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan.rowsZero_of_equations",
    "NightstreamFPrime.Export.SharedFormulas.directOutputRows_eq",
    "NightstreamFPrime.Layout.MatrixProgram.Poseidon.Block.rowWithInput?_ofSemantic"]

def externalVariant (_ : Unit) : Variant :=
  let input : SparseLayer.State 8 := fun lane => SparseForm.singleton lane 1
  { linearForms := wireState (SparseLayer.external input)
    rows := []
    outputRegisters := (List.range 8).map (8 + ·) }

def externalComponent (_ : Unit) : Component where
  id := "poseidon2-external-v1"
  inputCount := 8
  ports := [⟨"input", "input", 0, 8⟩]
  variantCount := 1
  variant := fun _ => externalVariant ()
  definitions := ["NightstreamFPrime.Layout.ProductionRelation.SparseLayer.external"]
  contracts := ["NightstreamFPrime.Layout.ProductionRelation.SparseLayer.eval_external"]

private def phi81Interface : Phi81ProductPlan.Interface 271 :=
  let oneColumn : Fin 271 := 0
  let challenge : Phi81ProductPlan.State 271 := fun index =>
    SparseForm.add (SparseForm.singleton
      ⟨1 + index.val, by have bound : index.val < 54 := index.isLt; omega⟩ 1)
      (SparseForm.singleton oneColumn (-2))
  { oneColumn
    left := challenge
    right := fun index => SparseForm.singleton
      ⟨55 + index.val, by have bound : index.val < 54 := index.isLt; omega⟩ 1
    quotient := fun index => SparseForm.singleton
      ⟨109 + index.val, by have bound : index.val < 54 := index.isLt; omega⟩ 1
    prior := fun index => SparseForm.singleton
      ⟨163 + index.val, by have bound : index.val < 54 := index.isLt; omega⟩ 1
    output := fun index => SparseForm.singleton
      ⟨217 + index.val, by have bound : index.val < 54 := index.isLt; omega⟩ 1 }

def phi81Variant (_ : Fin 1) : Variant where
  linearForms := []
  rows := (Phi81ProductPlan.rows phi81Interface).map
    (fun row => wireRow row.meaningfulForm)
  outputRegisters := (List.range ringDegree).map (217 + ·)

def phi81Component (_ : Unit) : Component where
  id := "phi81-product-v1"
  inputCount := 271
  ports := [⟨"one", "constant", 0, 1⟩,
    ⟨"challenge", "input", 1, ringDegree⟩,
    ⟨"input", "input", 55, ringDegree⟩,
    ⟨"quotient", "witness", 109, ringDegree⟩,
    ⟨"prior", "input", 163, ringDegree⟩,
    ⟨"output", "output", 217, ringDegree⟩]
  variantCount := 1
  variant := phi81Variant
  definitions := [
    "NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan.rows",
    "NightstreamFPrime.Layout.MatrixProgram.Phi81Product.Block.interface?"]
  contracts := [
    "NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan.rowsZero_implies_ringProduct",
    "NightstreamFPrime.Spec.Phi81Relation.QuotientProduct.complete",
    "NightstreamFPrime.Layout.MatrixProgram.Phi81Product.Block.row?_of_loaded"]

def Variant.validate (inputCount : Nat) (variant : Variant) : Except String Unit := do
  let mut available := inputCount
  for form in variant.linearForms do
    if !form.entries.all (fun entry =>
        entry.column < available && entry.coefficient < goldilocksModulus) then
      throw "noncanonical or forward linear-form reference"
    available := available + 1
  for row in variant.rows do
    if row.length != Spec.ProductionRelation.meaningfulPortCount then
      throw "invalid meaningful matrix port count"
    for form in row do
      if !form.entries.all (fun entry =>
          entry.column < available && entry.coefficient < goldilocksModulus) then
        throw "noncanonical or invalid row-form reference"
  if !variant.outputRegisters.all (· < available) then
    throw "invalid output register"

private def formJson (form : WireForm) : Lean.Json :=
  Lean.toJson (form.entries.map fun entry => (entry.column, entry.coefficient))

def Variant.json (variant : Variant) : Lean.Json :=
  Lean.Json.mkObj [
    ("linear_forms", Lean.toJson (variant.linearForms.map formJson)),
    ("rows", Lean.toJson (variant.rows.map fun row => row.map formJson)),
    ("output_registers", Lean.toJson variant.outputRegisters)]

private def Port.json (port : Port) : Lean.Json :=
  Lean.Json.mkObj [("name", .str port.name), ("role", .str port.role),
    ("start", Lean.toJson port.start), ("count", Lean.toJson port.count)]

private def writeComponent (handle : IO.FS.Handle) (component : Component) : IO Unit := do
  let mut endPort := 0
  for port in component.ports do
    unless port.start = endPort do
      throw (IO.userError "component ports are not contiguous")
    endPort := endPort + port.count
  unless endPort = component.inputCount do
    throw (IO.userError "component ports do not cover the inputs")
  let metadata := Lean.Json.mkObj [
    ("id", .str component.id), ("input_count", Lean.toJson component.inputCount),
    ("ports", Lean.toJson (component.ports.map Port.json)),
    ("definitions", Lean.toJson component.definitions),
    ("contracts", Lean.toJson component.contracts)]
  let text := metadata.compress
  handle.putStr (text.dropEnd 1).toString
  handle.putStr ",\"variants\":["
  for index in List.finRange component.variantCount do
    if index.val != 0 then handle.putStr ","
    let variant := component.variant index
    match variant.validate component.inputCount with
    | .error error => throw (IO.userError s!"{component.id}: {error}")
    | .ok () => handle.putStr variant.json.compress
  handle.putStr "]}"

/-- Stream fixed component variants; do not construct expanded matrices. -/
def write (path : System.FilePath) : IO Unit := do
  let handle ← IO.FS.Handle.mk path .write
  let profile := [goldilocksModulus, productionGlobalParams.b,
    productionGlobalParams.k, productionGlobalParams.bigB, ringDegree,
    NightstreamFPrime.Lifecycle.cubeVariables,
    Spec.ProductionRelation.matrixCount,
    Spec.ProductionRelation.meaningfulPortCount]
  let header := Lean.Json.mkObj [
    ("format", .str "nightstream.matrix-templates"), ("version", Lean.toJson (1 : Nat)),
    ("profile", Lean.toJson profile)]
  handle.putStr (header.compress.dropEnd 1).toString
  handle.putStr ",\"components\":["
  writeComponent handle (poseidonComponent ())
  handle.putStr ","
  writeComponent handle (externalComponent ())
  handle.putStr ","
  writeComponent handle (phi81Component ())
  handle.putStr "]}\n"

end NightstreamFPrime.Export.SharedFormulas
