import CompressionLean.Final.OptionBFinish
import CompressionLean.Final.Parent
import CompressionLean.Final.ParentComplete
import CompressionLean.Bridge.Field
import CompressionLean.Spec.OptionBParent
import CompressionLean.Final.OptionB

/-! Option B's slots as CE claims (CW-6.5). The words that the final program reads for a next slot
are that slot's claim in the refresh step's words (`slotClaim_eq`), so a slot statement in its
layer-1 relation gives the slot claim under its circuit's relation, with layer 1's witness padded
with zeros past the key's blocks (`slot_holds`, through `Final.parent_holds`). -/

namespace CompressionLean.Final.OptionB

open CompressionLean.Spec CompressionLean.Spec.OptionB CompressionLean.OptionB
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.OptionB.StepRelation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial (modulus)

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

private theorem range_get {β : Type} [Inhabited β] (n : ℕ) (f : ℕ → β) (i : ℕ) :
    ((Array.range n).map f)[i]! = if i < n then f i else default := by
  split_ifs with inside
  · rw [getElem!_pos _ _ (by simpa using inside)]
    simp
  · rw [getElem!_neg _ _ (by simpa using inside)]

private theorem toF_ofF (x : NightstreamFPrime.Spec.F) : (Replay.Word.ofF x).toF = gl x := by
  rw [Replay.Word.ofF, CompressionLean.Replay.Word.toF_ofNat]
  exact ZMod.natCast_zmod_val (n := goldilocksModulus) x

private theorem toF_default : (default : Replay.Word).toF = 0 := CompressionLean.Replay.Word.toF_zero

/-- The slot's words in the finish columns are its claim in the refresh step's columns. -/
theorem slotClaim_eq (program : Stage1.Application.Program)
    (finish : NightstreamFPrime.Export.OptionB.StepPackage.Package)
    (words : finish.application.1 = program.witnessWordCount) (system : PaperAlgebra.Structure lw)
    (env completed : Env)
    (agree : ∀ k, k < RefreshAssembler.sharedEnd base program.witnessWordCount → completed k = env k)
    (slot : Fin 2) :
    slotClaim finish slot env =
      Final.statementOf (lw := lw) (pf := pf) (Lifecycle.OptionB.toClaim system
        (((refreshInterface (logicalWidth := lw) (publicFits := pf) program).next slot).value
          completed).1) := by
  have start : slotStart finish.application.1 (5 + slot) =
      base + 25 + program.witnessWordCount + (5 + slot.val) * 2055 := by
    rw [words]; rfl
  have slots := slot.isLt
  -- A word of the slot is the same in the program's columns and the refresh step's.
  have word : ∀ i, slotStart finish.application.1 (5 + slot) ≤ i → i < slotStart finish.application.1 (5 + slot) + 2055 →
      (Replay.Word.ofF (env i)).toF = gl (completed i) := by
    intro i low high
    rw [toF_ofF, ← agree _ (by rw [start] at high; rw [RefreshAssembler.sharedEnd_eq]; omega)]
  have empty : ∀ (l : ℕ), ((default : Array Replay.Word)[l]!).toF = 0 := fun l => by
    rw [getElem!_neg _ _ (by show ¬ l < (#[] : Array Replay.Word).size; simp)]
    exact toF_default
  simp only [slotClaim, Spec.statementOf, Final.statementOf, Circuit.ClaimView.decode, claimAt, Array.map_map,
    Function.comp_def, id]
  congr 1
  · funext ρ l
    by_cases row : ρ < 22
    · by_cases lane : l < 54
      · rw [range_get, ite_eq_left row, range_get, ite_eq_left lane, dite_eq_left ⟨row, lane⟩,
          word _ (by omega) (by omega)]
        show gl (completed _) = gl (completed (base + 25 + program.witnessWordCount + (5 + slot.val) * 2055 +
          ρ * 54 + l))
        rw [start]
      · rw [range_get, ite_eq_left row, range_get, ite_eq_right lane, dite_eq_right fun h => lane h.2]
        exact toF_default
    · rw [range_get, ite_eq_right row, dite_eq_right fun h => row h.1]
      exact empty l
  · funext j l
    by_cases block : j < 5
    · by_cases lane : l < 54
      · rw [range_get, ite_eq_left block, range_get, ite_eq_left lane,
          dite_eq_left (show l < ringDegree ∧ ringDegree * j + l < ringDegree * PaperAlgebra.publicRingColumns by
            change l < 54 ∧ 54 * j + l < 54 * 5; omega), word _ (by omega) (by omega)]
        show gl (completed _) = gl (completed (base + 25 + program.witnessWordCount + (5 + slot.val) * 2055 +
          1784 + (54 * j + l)))
        rw [start]
        congr 2
        omega
      · rw [range_get, ite_eq_left block, range_get, ite_eq_right lane,
          dite_eq_right fun h => lane h.1]
        exact toF_default
    · rw [range_get, ite_eq_right block,
        dite_eq_right fun h => block (by change l < 54 ∧ 54 * j + l < 54 * 5 at h; omega)]
      exact empty l
  · apply List.ext_getElem (by
      simp [NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption.evalPoint,
        Lifecycle.OptionB.toClaim, Lifecycle.OptionB.SlotWords.SlotExpr.value, productionShape]
      rfl)
    intro k first second
    simp only [List.length_map, Array.length_toList, Array.size_map, Array.size_range] at first
    simp only [List.getElem_map, Array.getElem_toList, Array.getElem_map, Array.getElem_range, Prod.map,
      Lifecycle.OptionB.toClaim, Lifecycle.OptionB.SlotWords.SlotExpr.value,
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption.evalPoint, List.getElem_ofFn]
    rw [word _ (by omega) (by omega), word _ (by omega) (by omega)]
    show (gl (completed _), gl (completed _)) = (gl (completed (base + 25 + program.witnessWordCount +
      (5 + slot.val) * 2055 + 1728 + 2 * k)), gl (completed (base + 25 + program.witnessWordCount +
      (5 + slot.val) * 2055 + 1728 + 2 * k + 1)))
    rw [start]
  · funext l
    by_cases lane : l < 54
    · rw [range_get, ite_eq_left lane, dite_eq_left (show l < ringDegree from lane)]
      simp only [Prod.map]
      rw [word _ (by omega) (by omega), word _ (by omega) (by omega)]
      show (gl (completed _), gl (completed _)) = (gl (completed (base + 25 + program.witnessWordCount +
        (5 + slot.val) * 2055 + 1188 + 2 * l)), gl (completed (base + 25 + program.witnessWordCount +
        (5 + slot.val) * 2055 + 1188 + 2 * l + 1)))
      rw [start]
    · rw [range_get, ite_eq_right lane, dite_eq_right (show ¬ l < ringDegree from lane)]
      exact Prod.ext toF_default toF_default
  · funext j l
    by_cases family : j < 4
    · by_cases lane : l < 54
      · rw [range_get, ite_eq_left family, range_get, ite_eq_left lane,
          dite_eq_left (show j < productionProfile.ccsMatrices ∧ l < ringDegree from ⟨family, lane⟩)]
        simp only [Prod.map]
        rw [word _ (by omega) (by omega), word _ (by omega) (by omega)]
        show (gl (completed _), gl (completed _)) = (gl (completed (base + 25 + program.witnessWordCount +
          (5 + slot.val) * 2055 + 1296 + 2 * (j * 54 + l))), gl (completed (base + 25 + program.witnessWordCount +
          (5 + slot.val) * 2055 + 1296 + 2 * (j * 54 + l) + 1)))
        rw [start]
        congr 3 <;> omega
      · rw [range_get, ite_eq_left family, range_get, ite_eq_right lane,
          dite_eq_right fun h => lane h.2]
        exact Prod.ext toF_default toF_default
    · rw [range_get, ite_eq_right family, dite_eq_right fun h => family h.1]
      exact Prod.ext (empty l) (empty l)

/-- A circuit key's constraint system is its relation's source. -/
theorem system_eq (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (key : Lifecycle.OptionB.CircuitKey lw pf) :
    key.system ajtai = PiRLC.v1_2.InputBinding.relationSource key.relation :=
  (Lifecycle.OptionB.withRefreshTranscript_relation _).1

/-- A slot claim whose statement is in its layer-1 relation holds under the key's relation, with
layer 1's witness zero past the key's blocks. -/
theorem slot_holds (s : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (shape : ParentShape s lw) (blocks_le : s.blocks ≤ 2 ^ s.blockVars)
    (barEq : ∀ p l : Fin ringDegree, bar p l = gl (Phi81CoefficientKernel.nativeBarEntry p l))
    (honest : HonestSetup s S relation ajtai) (claim : Lifecycle.OptionB.SlotClaim lw pf) {z : ℕ → Gl}
    (holds : Layer1.Relation s modulus bar S
      (Final.statementOf (Lifecycle.OptionB.toClaim (PiRLC.v1_2.InputBinding.relationSource relation) claim)) z) :
    Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (Lifecycle.OptionB.toClaim (PiRLC.v1_2.InputBinding.relationSource relation) claim)
      (assignmentOf (lw := lw) (pf := pf) s z) :=
  parent_holds s bar S relation ajtai _ shape blocks_le barEq honest rfl rfl rfl holds

/-- The converse of `slot_holds` (CW-10.6), on finish columns `env`: when slot `slot`'s claim
under a circuit key holds with a witness below the accumulator bound of the light cap plus one and
zero on every ring block from `width`, and the layer-1 key's norm bound and blocks cover them, the
slot's words are in their layer-1 relation (through `Final.parent_complete`). -/
theorem slot_complete (program : Stage1.Application.Program)
    (finish : NightstreamFPrime.Export.OptionB.StepPackage.Package)
    (words : finish.application.1 = program.witnessWordCount) (s : Layer1.Shape) (bar : ℕ → ℕ → Gl)
    (S : Layer1.Setup Gl) (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (key : Lifecycle.OptionB.CircuitKey lw pf) (shape : ParentShape s lw)
    (barEq : ∀ p l : Fin ringDegree, bar p l = gl (Phi81CoefficientKernel.nativeBarEntry p l))
    (honest : HonestSetup s S key.relation ajtai) (env : Env) (slot : Fin 2)
    {w : PaperAlgebra.Assignment (logicalWidth := lw) (publicFits := pf)}
    (holds : Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (Lifecycle.OptionB.toClaim (key.system ajtai)
        ((refreshOutput (logicalWidth := lw) (publicFits := pf) program 1 env).runningNext slot).1) w)
    (norm : (PaperAlgebra.semantics ajtai).normBounded
      (productionGlobalParams.accumulatorBound Folding.Schedule.productionLightFoldCap + 1) w)
    (normCovers : productionGlobalParams.accumulatorBound Folding.Schedule.productionLightFoldCap ≤ s.normBound)
    {width : ℕ}
    (support : ∀ column : Fin (Phi81CarrierLayout.carrierWidth lw),
      width ≤ ringDegree * (column.val / ringDegree) → w column = 0)
    (blocksCover : width ≤ ringDegree * s.blocks) :
    ∃ z, Layer1.Relation s modulus bar S (slotClaim finish slot env) z := by
  rw [slotClaim_eq (lw := lw) (pf := pf) program finish words
    (PiRLC.v1_2.InputBinding.relationSource key.relation) env env (fun _ _ => rfl) slot]
  rw [system_eq] at holds
  obtain ⟨z, relation_holds, -⟩ := parent_complete s bar S key.relation ajtai _ shape barEq honest rfl holds
    (fun column => (Nat.le_of_lt_succ (norm column)).trans normCovers)
    (fun column outside => support column (blocksCover.trans (Nat.mul_le_mul_left ringDegree outside)))
  exact ⟨z, relation_holds⟩

/-- The final terminal on the emitted finish package, with both slot statements in their layer-1
relations, gives option B's parent claim. -/
theorem parentHolds_of_holds (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) (next width : ℕ)
    (batches : List NightstreamFPrime.Circuit.WitnessBatch) (rest : ℕ × ℕ × ℕ × ℕ)
    (packed : Export.Generic.CircuitRelation.packable Export.Generic.CircuitRelation.publicColumns
      (Export.OptionB.FinishRelation.pieces program relation) = true)
    (starts : Export.Generic.CircuitRelation.startsFrom (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (Export.OptionB.FinishRelation.pieces program relation) = true)
    (s : Fin 2 → Layer1.Shape) (S : Fin 2 → Layer1.Setup Gl) (bar : ℕ → ℕ → Gl)
    (barEq : ∀ p l : Fin ringDegree, bar p l = gl (Phi81CoefficientKernel.nativeBarEntry p l))
    (shape : ∀ slot, ParentShape (s slot) lw) (blocks_le : ∀ slot, (s slot).blocks ≤ 2 ^ (s slot).blockVars)
    (honest : ∀ slot, HonestSetup (s slot) (S slot) (circuitKeys slot).relation ajtai)
    {stateKeys : List Replay.Word} {x : StateWords}
    (holds : Holds (s 0) (s 1) modulus bar (S 0) (S 1)
      (FinishTerminal (finishPackage program relation next width batches rest) stateKeys)
      (slotClaim (finishPackage program relation next width batches rest) 0)
      (slotClaim (finishPackage program relation next width batches rest) 1) x) :
    ParentHolds program ajtai circuitKeys x := by
  obtain ⟨env, terminal, ⟨z0, rel0⟩, z1, rel1⟩ := holds
  obtain ⟨completed, agree, iteration, start, current, finish⟩ :=
    finishTerminal_holds program relation ajtai circuitKeys next width batches (program.witnessWordCount, rest)
      packed starts terminal
  have slotHolds : ∀ (slot : Fin 2) (z : ℕ → Gl),
      Layer1.Relation (s slot) modulus bar (S slot) (slotClaim (finishPackage program relation next width batches rest) slot env) z →
      ∃ witness, Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (Lifecycle.OptionB.toClaim ((circuitKeys slot).system ajtai)
          ((refreshOutput (logicalWidth := lw) (publicFits := pf) program 1 completed).runningNext slot).1) witness := by
    intro slot z relation_holds
    rw [slotClaim_eq (lw := lw) (pf := pf) program _ rfl (PiRLC.v1_2.InputBinding.relationSource (circuitKeys slot).relation) env
      completed agree slot] at relation_holds
    rw [system_eq]
    exact ⟨_, slot_holds (s slot) bar (S slot) (circuitKeys slot).relation ajtai (shape slot) (blocks_le slot) barEq
      (honest slot) _ relation_holds⟩
  refine ⟨completed, iteration, start, current, finish, fun slot => ?_⟩
  fin_cases slot
  · exact slotHolds 0 z0 rel0
  · exact slotHolds 1 z1 rel1

end CompressionLean.Final.OptionB
