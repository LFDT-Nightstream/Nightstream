import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment

/-! Complete the application's physical witness and transport every application row.
The caller owns state and message words; the hash circuit owns all local values. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationWitness

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.Stage1
open Poseidon2HashChainV1Package (application)

private def inputs (target : Env) (message : Fin 4 → F) : Env := fun column =>
  if inside : Spartan.privateColumnCount ≤ column ∧ column < ApplicationInputs.localStart application then
    message ⟨column - Spartan.privateColumnCount, by
      change _ ∧ column < Spartan.privateColumnCount + 4 at inside
      omega⟩
  else target column

private theorem inputs_before (target : Env) (message : Fin 4 → F) (column : Nat)
    (before : column < Spartan.privateColumnCount) : inputs target message column = target column := by
  unfold inputs
  rw [dif_neg (fun inside => (Nat.not_le_of_lt before) inside.1)]

private theorem inputs_message (target : Env) (message : Fin 4 → F) (lane : Fin 4) :
    inputs target message (ApplicationInputs.witnessColumn lane) = message lane := by
  have bound := lane.isLt
  change inputs target message (Spartan.privateColumnCount + lane.val) = message lane
  unfold inputs
  rw [dif_pos (by change _ ∧ _ < Spartan.privateColumnCount + 4; omega)]
  exact congrArg message (Fin.ext (Nat.add_sub_cancel_left Spartan.privateColumnCount lane.val))

private def hashInterface := Lifecycle.Stage1.Poseidon2HashChainV1.hashInterface
  (ApplicationInputs.interface application)

private def completed (target : Env) (message : Fin 4 → F) : Env :=
  Gadgets.Poseidon2.Formal.witness hashInterface (inputs target message) (ApplicationInputs.localStart application)

private theorem completed_before (target : Env) (message : Fin 4 → F) (column : Nat)
    (before : column < ApplicationInputs.localStart application) :
    completed target message column = inputs target message column :=
  Gadgets.Poseidon2.Formal.witness_agreesOutside hashInterface (inputs target message)
    (ApplicationInputs.localStart application) column (Or.inl before)

/-- All physical message and local words, including the constant tag prefix. -/
def privateSuffix (target : Env) (message : Fin 4 → F) :
    Fin (PerApplicationPackage.addedPrivateColumnCount application) → F := fun index =>
  completed target message (Spartan.privateColumnCount + index.val)

def priorValues (target : Env) (lane : Fin 4) : F := target (ApplicationInputs.inputColumn lane)

def raw (target : Env) (message : Fin 4 → F) : PerApplicationCanonicalAssignment.RawValues application :=
  PerApplicationAssignmentTransportExecution.canonicalRawValues application
    (PerApplicationSourceAssignment.ofCompleted application target (privateSuffix target message))

private theorem source_eq_completed (target : Env) (message : Fin 4 → F)
    (packet : PerApplicationCanonicalAssignment.RawValues application)
    (baseEq : packet.base = (raw target message).base)
    (column : Fin (ApplicationRetainedBlocks.sourceWidth application)) :
    packet.applicationSource column = completed target message column.val := by
  change packet.base ⟨column.val, _⟩ = _
  rw [baseEq]
  change PerApplicationSourceAssignment.ofCompleted application target
    (privateSuffix target message) ⟨column.val, _⟩ = _
  have constant : PerApplicationPackage.basePackage.layout.constantColumn = Spartan.privateColumnCount := by
    rw [Spartan.privateColumnCount_eq]
    exact Package.circuitPackage_layout_values.2.2.1
  by_cases before : column.val < Spartan.privateColumnCount
  · unfold PerApplicationSourceAssignment.ofCompleted
    rw [dif_pos (by rw [constant]; exact before)]
    rw [completed_before target message _ (by
      change column.val < Spartan.privateColumnCount + 4
      omega), inputs_before target message _ before]
  · have width : ApplicationRetainedBlocks.sourceWidth application =
        Spartan.privateColumnCount + 4 + 7696 := Poseidon2HashChainV1Package.sourceWidth
    have bound := Nat.lt_of_lt_of_le column.isLt (Nat.le_of_eq width)
    let index : Fin (PerApplicationPackage.addedPrivateColumnCount application) :=
      ⟨column.val - Spartan.privateColumnCount, by
        rw [Poseidon2HashChainV1Package.addedPrivateColumnCount]
        omega⟩
    have stored := PerApplicationSourceAssignment.application_ofCompleted application target
      (privateSuffix target message) index
    have position : Spartan.privateColumnCount + index.val = column.val := by
      dsimp only [index]
      omega
    simpa only [privateSuffix, position] using stored

theorem witnessValue (target : Env) (message : Fin 4 → F) :
    Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
      (ApplicationInputs.localStart application) (SourceCompiler.sourceEnv (raw target message).base) =
        List.ofFn message := by
  apply congrArg List.ofFn
  funext lane
  have stored := source_eq_completed target message (raw target message) rfl
    ((ApplicationRetainedBlocks.witnessBlock application).source lane)
  have before : ApplicationInputs.witnessColumn lane <
      ApplicationInputs.localStart application := by
    have bound : lane.val < 4 := lane.isLt
    change Spartan.privateColumnCount + lane.val < Spartan.privateColumnCount + 4
    omega
  change (raw target message).applicationSource _ =
    completed target message (ApplicationInputs.witnessColumn lane) at stored
  exact (DirectApplicationPrefixPlan.applicationSource_eq_sourceEnv application
    (raw target message).base ((ApplicationRetainedBlocks.witnessBlock application).source lane)).symm.trans
      (stored.trans ((completed_before target message _ before).trans (inputs_message target message lane)))

/-- The canonical physical hash witness satisfies every row of the shared layout. -/
theorem complete_of_base (target : Env) (message : Fin 4 → F)
    (packet : PerApplicationCanonicalAssignment.RawValues application)
    (baseEq : packet.base = (raw target message).base)
    (step : (List.ofFn fun lane : Fin 4 => target (ApplicationInputs.outputColumn lane)) =
      application.step (List.ofFn (priorValues target)) (List.ofFn message)) :
    (ApplicationDirectPlan.plan Poseidon2HashChainV1Package.fits.package
      (PerApplicationFixedPoint.geometry application)).RowsZero packet.assignment := by
  have inputValues (lane : Fin 4) :
      inputs target message (ApplicationInputs.inputColumn lane) = target (ApplicationInputs.inputColumn lane) := by
    apply inputs_before
    rw [ApplicationInputs.inputColumn_value, Spartan.privateColumnCount_eq]
    simp only [ApplicationInputs.currentWordStart]
    have bound := lane.isLt
    omega
  have outputValues (lane : Fin 4) :
      inputs target message (ApplicationInputs.outputColumn lane) = target (ApplicationInputs.outputColumn lane) := by
    apply inputs_before
    rw [ApplicationInputs.outputColumn_value, Spartan.privateColumnCount_eq]
    have bound := lane.isLt
    omega
  have specification : Gadgets.Poseidon2.Formal.SpecHolds hashInterface
      (ApplicationInputs.localStart application) (inputs target message) := by
    apply (Lifecycle.Stage1.Poseidon2HashChainV1.spec_iff _ _ _).mpr
    change (List.ofFn fun lane => inputs target message (ApplicationInputs.outputColumn lane)) =
      application.step (List.ofFn fun lane => inputs target message (ApplicationInputs.inputColumn lane))
        (List.ofFn fun lane => inputs target message (ApplicationInputs.witnessColumn lane))
    exact (congrArg List.ofFn (funext outputValues)).trans
      (step.trans (congrArg₂ application.step
        (congrArg List.ofFn (funext inputValues)).symm
        (congrArg List.ofFn (funext (inputs_message target message))).symm))
  have completedRows := (Gadgets.Poseidon2.Formal.witness_complete hashInterface
    (inputs target message) (ApplicationInputs.localStart application)
    (application.assumptions (ApplicationInputs.interface application)
      (ApplicationInputs.localStart application) (inputs target message) (ApplicationInputs.externalBelow application)) specification).2
  have physical : R1CS.RowsHold (completed target message) (ApplicationDirectSource.sourceRows application) := by
    rw [ApplicationDirectSource.sourceRows, ApplicationPackage.ofProgram_compiledRows_toR1CS,
      Poseidon2HashChainV1Package.constraints_eq_hashConstraints]
    exact Layout.Poseidon2.hashPhysical_complete hashInterface _ _ _
      Poseidon2HashChainV1Package.hashInterface_affine completedRows
  apply (ApplicationOrdinaryPlan.rowsZero_iff_rowsHold Poseidon2HashChainV1Package.fits.package
    (PerApplicationFixedPoint.geometry application) packet.assignment packet.applicationSource
    (PerApplicationCanonicalEncodes.encodes packet).applicationEncoding
    (PerApplicationCanonicalAssignment.assignment_one packet)).mpr
  apply R1CS.rowsHold_of_agree _ (ApplicationDirectSource.SourceAllowed application)
    (completed target message) _ (ApplicationDirectSource.sourceRows_varsSatisfy application) _ physical
  intro index supported
  have bound : index < ApplicationRetainedBlocks.sourceWidth application := by
    rcases supported with ⟨lane, rfl⟩ | ⟨lane, rfl⟩ | ⟨lane, rfl⟩ | ⟨_, upper⟩
    · exact ((ApplicationRetainedBlocks.inputBlock application).source lane).isLt
    · exact ((ApplicationRetainedBlocks.witnessBlock application).source lane).isLt
    · exact ((ApplicationRetainedBlocks.outputBlock application).source lane).isLt
    · exact upper
  rw [ApplicationOrdinaryPlan.sourceEnv, dif_pos bound]
  exact source_eq_completed target message packet baseEq ⟨index, bound⟩

theorem complete (target : Env) (message : Fin 4 → F)
    (step : (List.ofFn fun lane : Fin 4 => target (ApplicationInputs.outputColumn lane)) =
      application.step (List.ofFn (priorValues target)) (List.ofFn message)) :
    (ApplicationDirectPlan.plan Poseidon2HashChainV1Package.fits.package
      (PerApplicationFixedPoint.geometry application)).RowsZero (raw target message).assignment :=
  complete_of_base target message (raw target message) rfl step

end NightstreamFPrime.Export.Stage1.ApplicationWitness
