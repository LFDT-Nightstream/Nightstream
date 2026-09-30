import tests.AxiomAudit
import NightstreamFPrime.Export.SharedFormulas

/-! Checks the fixed exported templates against their existing Lean owners.
These execution checks do not prove the Rust interpreter or generic assembly. -/

namespace NightstreamFPrime.Tests.SharedFormulas

open NightstreamFPrime.Export.SharedFormulas
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec

private def normalized {width : Nat} (form : SparseForm width) : SparseForm width :=
  ⟨(List.ofFn fun column =>
    ({ column, coefficient := form.coefficient column } : SparseEntry width)).filter
      (fun entry => entry.coefficient != 0)⟩

private def normalizedRow {width : Nat} (row : RowForms width) : List WireForm :=
  List.ofFn fun port => WireForm.ofSemantic (normalized (row port))

private def substitute {width : Nat} (registers : Array (SparseForm width))
    (form : WireForm) : Except String (SparseForm width) := do
  let mut result : SparseForm width := .empty
  for entry in form.entries do
    let some register := registers[entry.column]?
      | throw "unknown template register"
    result := SparseForm.add result
      (SparseForm.scale (Spec.Poseidon2.ofNat entry.coefficient) register)
  pure (normalized result)

private def expanded (width : Nat) (variant : Variant) :
    Except String (List (List WireForm) × List WireForm) := do
  variant.validate width
  let mut registers := (List.ofFn fun column : Fin width =>
    SparseForm.singleton column 1).toArray
  for form in variant.linearForms do
    registers := registers.push (← substitute registers form)
  let rows ← variant.rows.mapM fun row =>
    row.mapM fun form => return WireForm.ofSemantic (← substitute registers form)
  let outputs ← variant.outputRegisters.mapM fun index => do
    let some form := registers[index]? | throw "unknown output register"
    pure (WireForm.ofSemantic form)
  pure (rows, outputs)

private def referencePoseidonInterface : PoseidonSboxPlan.Interface 95 :=
  let outputs := fun slot : Fin PoseidonRetainedSlots.rows.length =>
    SparseForm.singleton ⟨9 + slot.val, by
      have bound : slot.val < 86 := by
        simpa only [PoseidonRetainedSlots.rows_length] using slot.isLt
      omega⟩ 1
  { oneColumn := 0
    input := fun lane => SparseForm.singleton
      ⟨1 + lane.val, by have bound := lane.isLt; omega⟩ 1
    sboxOutput := outputs
    output := SparseLayer.external fun lane => outputs
      ⟨78 + lane.val, by
        have bound := lane.isLt
        rw [PoseidonRetainedSlots.rows_length]
        omega⟩ }

/-- Keep the reference state in its original local coordinates. Combining
repeated entries after each existing step prevents exponential list growth. -/
private def referencePoseidon (_ : Unit) : List (List WireForm) × List WireForm := Id.run do
  let interface := referencePoseidonInterface
  let mut nextSbox := 0
  let mut state := interface.input
  let mut rows := []
  for step in Permutation.schedule do
    let result := PoseidonSboxPlan.compileStep interface nextSbox state step
    nextSbox := result.nextSbox
    let stateValues : Vector (SparseForm 95) 8 :=
      Vector.ofFn fun lane => normalized (result.state lane)
    state := stateValues.get
    rows := rows ++ result.rows.map (fun row => normalizedRow row.meaningfulForm)
  pure (rows, List.ofFn fun lane => WireForm.ofSemantic (normalized (state lane)))

private def ensure (condition : Bool) (message : String) : Except String Unit :=
  if condition then .ok () else .error message

def check : IO Unit := do
  let checks : Except String Unit := do
    let poseidon := poseidonVariant ()
    ensure (poseidon.rows.length == 86) "Poseidon2 row footprint changed"
    ensure ((← expanded 95 poseidon) == referencePoseidon ())
      "Poseidon2 template differs from the existing step formulas"
    let external := externalVariant ()
    let input : SparseLayer.State 8 := fun lane => SparseForm.singleton lane 1
    let expected := List.ofFn fun lane =>
      WireForm.ofSemantic (normalized (SparseLayer.external input lane))
    ensure ((← expanded 8 external) == ([], expected))
      "external-layer template differs from its Lean owner"
    let variant := phi81Variant 0
    variant.validate 271
    ensure (variant.rows.length == 108) "Phi81 row footprint changed"
    ensure (variant.linearForms.isEmpty) "Phi81 has unexpected registers"
    let forward : Variant :=
      { linearForms := [⟨[⟨1, 1⟩]⟩], rows := [], outputRegisters := [] }
    ensure (!(forward.validate 1).isOk) "forward reference was accepted"
    let noncanonical : Variant :=
      { linearForms := [⟨[⟨0, goldilocksModulus⟩]⟩], rows := [], outputRegisters := [] }
    ensure (!(noncanonical.validate 1).isOk) "noncanonical coefficient was accepted"
    let wrongPorts : Variant :=
      { linearForms := [], rows := [[]], outputRegisters := [] }
    ensure (!(wrongPorts.validate 1).isOk) "wrong matrix port count was accepted"
  match checks with
  | .error error => throw (IO.userError error)
  | .ok () => IO.println "shared formula export checks passed"

#audit_axioms NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan.compileStep_sound
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan.rowsZero_implies_permute
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan.rowsZero_of_equations
#audit_axioms NightstreamFPrime.Layout.MatrixProgram.Poseidon.Block.rowWithInput?_ofSemantic
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.SparseLayer.eval_external
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.Phi81ProductPlan.rowsZero_implies_ringProduct
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.ProductSumPlan.rowsZero_iff_equations
#audit_axioms NightstreamFPrime.Layout.MatrixProgram.Phi81Product.Block.row?_of_loaded

#eval check

end NightstreamFPrime.Tests.SharedFormulas
