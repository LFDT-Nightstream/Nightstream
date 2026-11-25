import NightstreamFPrime.Lifecycle.Nebula.MemorySoundness
import NightstreamFPrime.Lifecycle.Nebula.MemoryEval
import NightstreamFPrime.Lifecycle.Nebula.RowMeaning

/-! Owns the completeness of the memory-application circuit: when the caller's
wires lie below the call offset and the step is valid with its output state,
every flattened row holds in an environment that has the rows of the nine sponge
children. `MemoryProgram` executes the children. It does not own soundness. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

variable (p : Geometry) (app : PortApp p) (i : AppInterface p app) (offset : ℕ)

section Assumptions

variable {p app i offset}

theorem stateIn_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (stateIn p app i offset) offset env :=
  absorbing_assumptions env le_rfl (stateBlocks_supported (wiresSupported_below inputs) _ _)

theorem stateOut_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (stateOut p app i offset) (stateOutStart p app i offset) env :=
  absorbing_assumptions env (by unfold stateOutStart; omega)
    (stateBlocks_supported (wiresSupported_below inputs) _ _)

theorem chainOps_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (chainOps p app i offset) (chainOpsStart p app i offset) env :=
  absorbing_assumptions env (by unfold chainOpsStart stateOutStart; omega)
    (chainBlocks_supported (wiresSupported_below inputs) _ _ _)

theorem chainInitial_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (chainInitial p app i offset) (chainInitialStart p app i offset) env :=
  absorbing_assumptions env (by unfold chainInitialStart chainOpsStart stateOutStart; omega)
    (chainBlocks_supported (wiresSupported_below inputs) _ _ _)

theorem chainFinal_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (chainFinal p app i offset) (chainFinalStart p app i offset) env :=
  absorbing_assumptions env
    (by unfold chainFinalStart chainInitialStart chainOpsStart stateOutStart; omega)
    (chainBlocks_supported (wiresSupported_below inputs) _ _ _)

theorem eta_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (eta p app i offset) (etaStart p app i offset) env :=
  absorbing_assumptions env
    (by unfold etaStart chainFinalStart chainInitialStart chainOpsStart stateOutStart; omega)
    (etaBlocks_supported (wiresSupported_below inputs))

theorem noBlocks (start : ℕ) : Hash.BlocksBelow start [[]] := by simp [Hash.BlocksBelow]

theorem squeeze1_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (squeeze1 p app i offset) (squeeze1Start p app i offset) env := by
  refine ⟨fun lane => ?_, noBlocks _⟩
  have scope := Sponge.output_varsBelow (eta p app i offset) (etaStart p app i offset) env
    (eta_assumptions inputs env) lane
  have bound : etaStart p app i offset + ((eta p app i offset).chunks (etaStart p app i offset)).length * 1096 =
      squeeze1Start p app i offset := by
    rw [etaChunks_length]
    rfl
  rw [Sponge.localLength_eq, bound, ← etaState_output] at scope
  exact scope

theorem squeeze2_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (squeeze2 p app i offset) (squeeze2Start p app i offset) env := by
  refine ⟨fun lane => ?_, noBlocks _⟩
  have scope := Sponge.output_varsBelow (squeeze1 p app i offset) (squeeze1Start p app i offset) env
    (squeeze1_assumptions inputs env) lane
  have bound : squeeze1Start p app i offset +
      ((squeeze1 p app i offset).chunks (squeeze1Start p app i offset)).length * 1096 =
      squeeze2Start p app i offset := by
    simp only [squeeze2Start, squeeze1, permuting, List.length_singleton, one_mul]
  rw [Sponge.localLength_eq, bound] at scope
  exact scope

theorem squeeze3_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (squeeze3 p app i offset) (squeeze3Start p app i offset) env := by
  refine ⟨fun lane => ?_, noBlocks _⟩
  have scope := Sponge.output_varsBelow (squeeze2 p app i offset) (squeeze2Start p app i offset) env
    (squeeze2_assumptions inputs env) lane
  have bound : squeeze2Start p app i offset +
      ((squeeze2 p app i offset).chunks (squeeze2Start p app i offset)).length * 1096 =
      squeeze3Start p app i offset := by
    simp only [squeeze3Start, squeeze2, permuting, List.length_singleton, one_mul]
  rw [Sponge.localLength_eq, bound] at scope
  exact scope

end Assumptions

theorem rows_of_children (env completed : Env) (inputs : Application.InputsBelow i offset)
    (specification : Application.Holds (step p app) i offset env ∧
      Application.Valid (valid p app) i offset env)
    (agreesBelow : ∀ index, index < offset → completed index = env index)
    (flatRows : holdsFlat completed (childOps p app i offset)) :
    holdsFlat completed (opsAt p app i offset) := by
  obtain ⟨holdsStep, rowsHold, appRows, auxHonest⟩ := specification
  have childRows : holds completed (childOps p app i offset) :=
    holdsFlat_implies_holds _ _ flatRows
  have specOf : ∀ {name : String} {child : FormalCircuit} {start : ℕ},
      Sequence.childOp name child start ∈ childOps p app i offset →
        child.assumptions start completed → child.spec start completed :=
    fun member assumptions => childRows _ member assumptions
  have witnessEq : Application.witnessValue i offset completed =
      Application.witnessValue i offset env := by
    unfold Application.witnessValue
    congr 1
    funext index
    exact (i.witness offset index).eval_eq_of_agree_below offset _ env (inputs.witness index)
      agreesBelow
  have valuesEq : values p app i offset completed = values p app i offset env := by
    unfold values
    rw [witnessEq]
  have wireFinal : ∀ k, (wire p app i offset k).eval completed = values p app i offset env k :=
    fun k => by rw [wire_eval, valuesEq]
  have carryOut : ∀ (start : ℕ) (k : Fin 4), start + 3 < 39 →
      StepWitness.carryDigest (StepWitness.ofWords p app.toAppShape (values p app i offset env)).carryOut start k =
        values p app i offset env (Words.carryOut (start + k.val)) := by
    intro start k small
    simp [StepWitness.carryDigest, StepWitness.carryWord, StepWitness.ofWords,
      show start + k.val < 39 by omega]
  have fresh := rowsHold.freshEta
  rw [etaChallenges_eq] at fresh
  dsimp only at fresh
  have fresh0 := congrArg (fun pair : K × K => pair.1.c0) fresh
  have fresh1 := congrArg (fun pair : K × K => pair.1.c1) fresh
  have fresh2 := congrArg (fun pair : K × K => pair.2.c0) fresh
  have fresh3 := congrArg (fun pair : K × K => pair.2.c1) fresh
  dsimp only at fresh0 fresh1 fresh2 fresh3
  have etaAbsorbed := absorbing_state completed
    (specOf (name := "nebula.eta") (child := Sponge.circuit (eta p app i offset))
        (start := etaStart p app i offset) (by simp [childOps])
      (eta_assumptions inputs completed))
  rw [etaBlocks_eval, valuesEq] at etaAbsorbed
  have s1 : Sponge.evalState completed (squeeze1State p app i offset) =
      Spec.Poseidon2.permute (Sponge.evalState completed (etaState p app i offset)) :=
    permuting_state completed (specOf (name := "nebula.eta_squeeze_1")
      (child := Sponge.circuit (squeeze1 p app i offset))
        (start := squeeze1Start p app i offset) (by simp [childOps])
      (squeeze1_assumptions inputs completed))
  have s2 : Sponge.evalState completed (squeeze2State p app i offset) =
      Spec.Poseidon2.permute (Sponge.evalState completed (squeeze1State p app i offset)) :=
    permuting_state completed (specOf (name := "nebula.eta_squeeze_2")
      (child := Sponge.circuit (squeeze2 p app i offset))
        (start := squeeze2Start p app i offset) (by simp [childOps])
      (squeeze2_assumptions inputs completed))
  have s3 : Sponge.evalState completed (squeeze3State p app i offset) =
      Spec.Poseidon2.permute (Sponge.evalState completed (squeeze2State p app i offset)) :=
    permuting_state completed (specOf (name := "nebula.eta_squeeze_3")
      (child := Sponge.circuit (squeeze3 p app i offset))
        (start := squeeze3Start p app i offset) (by simp [childOps])
      (squeeze3_assumptions inputs completed))
  have etaState_eq : Sponge.evalState completed (etaState p app i offset) =
      absorbed [textWords "Nightstream/Nebula/v3/eta", digestWords (planDigest p),
        [natWord ((StepWitness.ofWords p app.toAppShape (values p app i offset env)).cIn 2).val],
        digestWords ((StepWitness.ofWords p app.toAppShape (values p app i offset env)).proposalDigest 0) ++
          digestWords (StepWitness.carryDigest
            (StepWitness.ofWords p app.toAppShape (values p app i offset env)).carryIn 35) ++
          digestWords ((StepWitness.ofWords p app.toAppShape (values p app i offset env)).proposalDigest 4)] := by
    rw [etaState_output]
    exact etaAbsorbed
  have flatAssertions : ∀ l : List Circuit.Expr, flatConstraints (l.map Op.assertZero) = l := by
    intro l
    induction l with
    | nil => rfl
    | cons e rest ih =>
      change [e] ++ flatConstraints (rest.map Op.assertZero) = e :: rest
      rw [ih]
      rfl
  rw [holdsFlat, opsAt, flatConstraints_append, flatAssertions]
  refine (constraintsHold_append _ _ _).mpr ⟨flatRows, ?_⟩
  intro e member
  simp only [assertions, hashChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append,
    List.mem_append, List.mem_map, List.mem_finRange, true_and,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with (((((⟨k, rfl⟩ | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) |
    (rfl | rfl | rfl | rfl)) | ⟨e', poly, rfl⟩
  · rw [Circuit.Expr.eval_sub, sub_eq_zero]
    have lanes := absorbing_lane completed (specOf (name := "nebula.state_in")
      (child := Sponge.circuit (stateIn p app i offset))
      (start := offset) (by simp [childOps])
      (stateIn_assumptions inputs completed)) k
    rw [stateBlocks_eval, valuesEq] at lanes
    exact lanes.trans ((congrFun (List.ofFn_injective rowsHold.stateIn) k).trans
      ((i.input offset k).eval_eq_of_agree_below offset _ env (inputs.input k) agreesBelow).symm)
  · rw [Circuit.Expr.eval_sub, sub_eq_zero]
    have lanes := absorbing_lane completed (specOf (name := "nebula.state_out")
      (child := Sponge.circuit (stateOut p app i offset))
      (start := stateOutStart p app i offset) (by simp [childOps])
      (stateOut_assumptions inputs completed)) k
    rw [stateBlocks_eval, valuesEq] at lanes
    exact lanes.trans ((congrFun (List.ofFn_injective holdsStep) k).symm.trans
      ((i.output offset k).eval_eq_of_agree_below offset _ env (inputs.output k) agreesBelow).symm)
  · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal]
    have lanes := absorbing_lane completed (specOf (name := "nebula.chain_ops")
      (child := Sponge.circuit (chainOps p app i offset))
      (start := chainOpsStart p app i offset) (by simp [childOps])
      (chainOps_assumptions inputs completed)) k
    rw [chainBlocks_eval p app i offset _ .ops 0 (by norm_num), opsLanes_eval, valuesEq] at lanes
    exact lanes.trans ((congrFun rowsHold.chainOps k).symm.trans (carryOut 23 k (by norm_num)))
  · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal]
    have lanes := absorbing_lane completed (specOf (name := "nebula.chain_is")
      (child := Sponge.circuit (chainInitial p app i offset))
      (start := chainInitialStart p app i offset) (by simp [childOps])
      (chainInitial_assumptions inputs completed)) k
    rw [chainBlocks_eval p app i offset _ .mem 4 (by norm_num),
      scanLanes_eval p app i offset _ _ _ fun j => Sym.eval_initialLane p app.toAppShape _ j, valuesEq] at lanes
    exact lanes.trans ((congrFun rowsHold.chainInitial k).symm.trans (carryOut 27 k (by norm_num)))
  · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal]
    have lanes := absorbing_lane completed (specOf (name := "nebula.chain_fs")
      (child := Sponge.circuit (chainFinal p app i offset))
      (start := chainFinalStart p app i offset) (by simp [childOps])
      (chainFinal_assumptions inputs completed)) k
    rw [chainBlocks_eval p app i offset _ .mem 8 (by norm_num),
      scanLanes_eval p app i offset _ _ _ fun j => Sym.eval_finalLane p app.toAppShape _ j, valuesEq] at lanes
    exact lanes.trans ((congrFun rowsHold.chainFinal k).symm.trans (carryOut 31 k (by norm_num)))
  · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal, lane, lane_eval, etaState_eq]
    exact fresh0
  · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal, lane, lane_eval, s1, etaState_eq]
    exact fresh1
  · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal, lane, lane_eval, s2, s1, etaState_eq]
    exact fresh2
  · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal, lane, lane_eval, s3, s2, s1, etaState_eq]
    exact fresh3
  · rw [sub_eval, valuesEq]
    exact (Rows.polyRows_iff p _ app).mpr ⟨rowsHold.toPolyRows, auxHonest,
      (congrArg (fun e => ConstraintsHold e app.rows) (view_env p app i offset env)).mp appRows⟩
      e' poly

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
