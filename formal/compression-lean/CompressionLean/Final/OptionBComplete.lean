import CompressionLean.Final.OptionBParent
import CompressionLean.Spec.OptionBIvc
import CompressionLean.Spec.OptionBMatrix
import NightstreamFPrime.Export.OptionB.FinishCompleteness
import NightstreamFPrime.Layout.OptionB.StepCircuits
import NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.OutputBinding
import NightstreamFPrime.Lifecycle.Nebula.FactorWords
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightKnowledge
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshKnowledge
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Completeness
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.ZeroRunningPolynomial
import NightstreamFPrime.Lifecycle.RandomOracle.Budgeted
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import NightstreamFPrime.Spec.Nebula

/-! Claim-level completeness of option B's final verifier (CW-10.6), the converse of
`parentHolds_of_holds`. F′'s `FinishCompleteness.FinishAccepts` names finish columns that the
final program's checks accept for an application statement, with witnesses of both next slots.
On the emitted finish package these columns satisfy the final program's terminal for the
statement's words (`Spec.OptionB.terminalWords`): column 0 is one, the iteration and both states
are on their words, the key words are the state keys when these are the setup's digests
(`Spec.OptionB.KeysAre`), and the package's rows and blocks hold. Each next slot's words are in
their layer-1 relation when the slot's layer-1 key covers the honest witness
(`slot_complete`). So the claim of `Claims.option_b_bound` holds for the statement
(`holds_of_honest`). Nothing here says that the honest layer-1 or shrink provers' messages are
accepted. -/

set_option autoImplicit false

namespace CompressionLean.Final.OptionB

open CompressionLean.Spec CompressionLean.Spec.OptionB CompressionLean.OptionB
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper (TerminalStatement)
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.OptionB.StepRelation
open NightstreamFPrime.Export.OptionB.FinishWalk (finishStatement)
open NightstreamFPrime.Export.OptionB.FinishCompleteness (FinishAccepts)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial (modulus)

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

/-- The logical width of a step package that the emitter writes (`StepPackage.packageOf`) is that of
its slot runs, as `Complete.packageWidth` states it. The arguments stay abstract, so that no
evaluation of the pieces' slot runs starts. -/
theorem logicalWidth_packageOf (circuit columns : ℕ) (pieces : List Export.Generic.Pieces.Piece)
    (next width : ℕ) (batches : List WitnessBatch) (application : ℕ × ℕ × ℕ × ℕ × ℕ) :
    Layout.ProductionAssignment.logicalWidth
        (slotsOf (Export.OptionB.StepPackage.packageOf circuit columns pieces next width batches application)) =
      Layout.ProductionAssignment.logicalWidth (Export.Generic.SlotLayout.runSlots width
        (Export.OptionB.StepPackage.slotRuns columns pieces next batches)) :=
  rfl

/-- The converse of `decodedKeys_of_keys`: finish columns whose key words decode to the setup's
keys carry the state keys when these are the setup's digests. -/
theorem keyWords_of_decoded (program : Stage1.Application.Program)
    {circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf} {stateKeys : List Replay.Word}
    (keys : KeysAre stateKeys circuitKeys) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    {env : Env}
    (decoded : decodedKeys circuitKeys (refreshInterface (logicalWidth := lw) (publicFits := pf) program).verifierKeys
      env = circuitKeys) :
    ∀ k (inside : k < stateKeys.length), env (keysWord + k) = Replay.Word.cast stateKeys[k] := by
  intro k inside
  have mapped : (stateKeys.map (Replay.Word.cast (F := NightstreamFPrime.Spec.F))).length = 8 := by
    rw [keys, List.length_append, keysShaped, keysShaped]
  have length : stateKeys.length = 8 := by simpa using mapped
  have baseEq : base = 270 := rfl
  -- Each key's digest is the words of its key columns.
  have digestAt : ∀ (key : Fin 2) (i : ℕ) (lane : i < 4),
      (circuitKeys key).digest[i]'(by rw [keysShaped]; exact lane) =
        env (LightAssembler.keysStart base + 4 * key.val + i) := by
    intro key i lane
    have words := congrArg Lifecycle.OptionB.CircuitKey.digest (congrFun decoded key)
    have entry := List.getElem_of_eq words (show i < (decodedKeys circuitKeys
      (refreshInterface (logicalWidth := lw) (publicFits := pf) program).verifierKeys env key).digest.length by
        simp [decodedKeys, lane])
    rw [← entry]
    simp only [decodedKeys, List.getElem_ofFn]
    rfl
  have entry := List.getElem_of_eq keys (show k < (stateKeys.map
    (Replay.Word.cast (F := NightstreamFPrime.Spec.F))).length by rw [mapped]; omega)
  rw [List.getElem_map] at entry
  rw [entry, List.getElem_append]
  split
  · rename_i low
    rw [keysShaped] at low
    rw [digestAt 0 k low]
    rfl
  · rename_i high
    rw [keysShaped] at high
    simp only [keysShaped]
    rw [digestAt 1 (k - 4) (by omega)]
    congr 1
    simp only [LightAssembler.keysStart, keysWord, publicWords, baseEq]
    omega

/-- The converse of `parentHolds_of_holds` (CW-10.6): finish columns that F′'s finish checks
accept for an application statement of at least one step, on the emitted finish package, with
both next slots' layer-1 keys covering their honest witnesses, satisfy the final program's
terminal for the statement's words, and both next slots' words are in their layer-1 relations. -/
theorem holds_of_honest (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) (next width : ℕ)
    (batches : List NightstreamFPrime.Circuit.WitnessBatch) (rest : ℕ × ℕ × ℕ × ℕ)
    (s : Fin 2 → Layer1.Shape) (S : Fin 2 → Layer1.Setup Gl) (bar : ℕ → ℕ → Gl)
    (barEq : ∀ p l : Fin ringDegree, bar p l = gl (Phi81CoefficientKernel.nativeBarEntry p l))
    (shape : ∀ slot, ParentShape (s slot) lw)
    (honest : ∀ slot, HonestSetup (s slot) (S slot) (circuitKeys slot).relation ajtai)
    (packageWidths : Fin 2 → ℕ)
    (normCovers : ∀ slot,
      productionGlobalParams.accumulatorBound Folding.Schedule.productionLightFoldCap ≤ (s slot).normBound)
    (blocksCover : ∀ slot, packageWidths slot ≤ ringDegree * (s slot).blocks)
    {stateKeys : List Replay.Word} (keys : KeysAre stateKeys circuitKeys)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    {statement : TerminalStatement AppState} (positive : 0 < statement.iteration)
    (accepts : FinishAccepts program relation ajtai circuitKeys next width batches
      (program.witnessWordCount, rest) packageWidths statement) :
    Holds (s 0) (s 1) modulus bar (S 0) (S 1)
      (FinishTerminal (finishPackage program relation next width batches rest) stateKeys)
      (slotClaim (finishPackage program relation next width batches rest) 0)
      (slotClaim (finishPackage program relation next width batches rest) 1) (terminalWords statement) := by
  obtain ⟨env, witnesses, one, rows, blocks, decoded, statementEq, -, slots⟩ := accepts
  subst statementEq
  have iteration : (finishStatement (refreshInput (logicalWidth := lw) (publicFits := pf) program env)).iteration =
      (env iterationWord).val := rfl
  have slotHolds : ∀ slot : Fin 2, ∃ z, Layer1.Relation (s slot) modulus bar (S slot)
      (slotClaim (finishPackage program relation next width batches rest) slot env) z := fun slot =>
    slot_complete program _ rfl (s slot) bar (S slot) ajtai (circuitKeys slot) (shape slot) barEq (honest slot)
      env slot (slots slot).1 (slots slot).2.1 (normCovers slot) (slots slot).2.2 (blocksCover slot)
  refine ⟨env, ⟨fun zero => ?_, one, ?_, fun lane => ⟨?_, ?_⟩, keyWords_of_decoded program keys keysShaped decoded,
    rows, blocks⟩, slotHolds 0, slotHolds 1⟩
  · -- The iteration is positive and below the field, so its word is not zero.
    have word := congrArg Fin.val zero
    change (finishStatement (refreshInput (logicalWidth := lw) (publicFits := pf) program env)).iteration %
      goldilocksModulus = 0 at word
    rw [iteration, Nat.mod_eq_of_lt (env iterationWord).isLt] at word
    rw [iteration, word] at positive
    exact Nat.lt_irrefl 0 positive
  · apply Fin.ext
    change (env iterationWord).val =
      (finishStatement (refreshInput (logicalWidth := lw) (publicFits := pf) program env)).iteration % goldilocksModulus
    rw [iteration, Nat.mod_eq_of_lt (env iterationWord).isLt]
  · have laneBound := lane.isLt
    unfold terminalWords
    rw [ite_eq_right (show ¬ (1 + lane.val = 0) by omega), ite_eq_left (show 1 + lane.val < 5 by omega),
      show 1 + lane.val - 1 = lane.val by omega]
    change env (z0Word + lane.val) =
      (List.ofFn fun lane : Fin 4 => env (LightAssembler.statesStart base + lane.val)).getD lane.val 0
    rw [List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]
    congr 1
  · have laneBound := lane.isLt
    unfold terminalWords
    rw [ite_eq_right (show ¬ (5 + lane.val = 0) by omega), ite_eq_right (show ¬ (5 + lane.val < 5) by omega),
      show 5 + lane.val - 5 = lane.val by omega]
    change env (currentWord + lane.val) =
      (List.ofFn fun lane : Fin 4 => env (LightAssembler.statesStart base + 4 + lane.val)).getD lane.val 0
    rw [List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]
    congr 1

end CompressionLean.Final.OptionB
