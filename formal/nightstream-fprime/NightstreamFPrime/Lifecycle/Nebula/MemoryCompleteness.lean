import NightstreamFPrime.Lifecycle.Nebula.MemorySoundness

/-! Owns the completeness of the memory-application circuit: when the caller's
wires lie below the call offset and the step is valid with its output state,
honest execution of the nine sponge children satisfies every flattened row and
changes only the circuit's own variables. It does not own soundness. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

/-- Append one honestly executed sponge child at the end of a prefix. -/
theorem appendSponge {initial : Env} {base : ℕ} (before : Sequence.Prefix initial base)
    (name : String) (child : Sponge.Interface) (start : ℕ)
    (startEq : base + localLength before.operations = start)
    (assumptions : ∀ env, Sponge.Assumptions child start env) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++ [Sequence.childOp name (Sponge.circuit child) start] ∧
      base + localLength after.operations = start + (child.chunks start).length * 1096 := by
  obtain ⟨completed, agrees, rows⟩ := Sponge.completeness child before.current start (assumptions _)
  obtain ⟨after, operations, endEq, -, -⟩ := Sequence.appendBuiltAt before name (Sponge.circuit child)
    start startEq (Sponge.flatConstraints_varsBelow child start (fun _ => 0) (assumptions _))
    completed agrees rows
  refine ⟨after, operations, ?_⟩
  rw [endEq]
  exact congrArg (start + ·) (Sponge.localLength_eq child start)

variable (p : Plan) (i : AppInterface p) (offset : ℕ)

section Assumptions

variable {p i offset}

theorem stateIn_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (stateIn p i offset) offset env :=
  absorbing_assumptions env le_rfl (stateBlocks_supported (wiresSupported_below inputs) _ _)

theorem stateOut_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (stateOut p i offset) (stateOutStart p i offset) env :=
  absorbing_assumptions env (by unfold stateOutStart; omega)
    (stateBlocks_supported (wiresSupported_below inputs) _ _)

theorem chainOps_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (chainOps p i offset) (chainOpsStart p i offset) env :=
  absorbing_assumptions env (by unfold chainOpsStart stateOutStart; omega)
    (chainBlocks_supported (wiresSupported_below inputs) _ _ _)

theorem chainInitial_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (chainInitial p i offset) (chainInitialStart p i offset) env :=
  absorbing_assumptions env (by unfold chainInitialStart chainOpsStart stateOutStart; omega)
    (chainBlocks_supported (wiresSupported_below inputs) _ _ _)

theorem chainFinal_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (chainFinal p i offset) (chainFinalStart p i offset) env :=
  absorbing_assumptions env
    (by unfold chainFinalStart chainInitialStart chainOpsStart stateOutStart; omega)
    (chainBlocks_supported (wiresSupported_below inputs) _ _ _)

theorem eta_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (eta p i offset) (etaStart p i offset) env :=
  absorbing_assumptions env
    (by unfold etaStart chainFinalStart chainInitialStart chainOpsStart stateOutStart; omega)
    (etaBlocks_supported (wiresSupported_below inputs))

theorem noBlocks (start : ℕ) : Hash.BlocksBelow start [[]] := by simp [Hash.BlocksBelow]

theorem squeeze1_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (squeeze1 p i offset) (squeeze1Start p i offset) env := by
  refine ⟨fun lane => ?_, noBlocks _⟩
  have scope := Sponge.output_varsBelow (eta p i offset) (etaStart p i offset) env
    (eta_assumptions inputs env) lane
  have bound : etaStart p i offset + ((eta p i offset).chunks (etaStart p i offset)).length * 1096 =
      squeeze1Start p i offset := by
    rw [etaChunks_length]
    rfl
  rw [Sponge.localLength_eq, bound, ← etaState_output] at scope
  exact scope

theorem squeeze2_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (squeeze2 p i offset) (squeeze2Start p i offset) env := by
  refine ⟨fun lane => ?_, noBlocks _⟩
  have scope := Sponge.output_varsBelow (squeeze1 p i offset) (squeeze1Start p i offset) env
    (squeeze1_assumptions inputs env) lane
  have bound : squeeze1Start p i offset +
      ((squeeze1 p i offset).chunks (squeeze1Start p i offset)).length * 1096 =
      squeeze2Start p i offset := by
    simp only [squeeze2Start, squeeze1, permuting, List.length_singleton, one_mul]
  rw [Sponge.localLength_eq, bound] at scope
  exact scope

theorem squeeze3_assumptions (inputs : Application.InputsBelow i offset) (env : Env) :
    Sponge.Assumptions (squeeze3 p i offset) (squeeze3Start p i offset) env := by
  refine ⟨fun lane => ?_, noBlocks _⟩
  have scope := Sponge.output_varsBelow (squeeze2 p i offset) (squeeze2Start p i offset) env
    (squeeze2_assumptions inputs env) lane
  have bound : squeeze2Start p i offset +
      ((squeeze2 p i offset).chunks (squeeze2Start p i offset)).length * 1096 =
      squeeze3Start p i offset := by
    simp only [squeeze3Start, squeeze2, permuting, List.length_singleton, one_mul]
  rw [Sponge.localLength_eq, bound] at scope
  exact scope

end Assumptions

/-- Honest execution of the nine children: a prefix whose operations are the
children, in order. -/
theorem children_complete (inputs : Application.InputsBelow i offset) (env : Env) :
    ∃ children : Sequence.Prefix env offset, children.operations = childOps p i offset := by
  obtain ⟨c1, ops1, end1⟩ := appendSponge (Sequence.empty env offset) "nebula.state_in"
    (stateIn p i offset) offset (by simp [Sequence.empty, localLength]) (stateIn_assumptions inputs)
  obtain ⟨c2, ops2, end2⟩ := appendSponge c1 "nebula.state_out" (stateOut p i offset)
    (stateOutStart p i offset) end1 (stateOut_assumptions inputs)
  obtain ⟨c3, ops3, end3⟩ := appendSponge c2 "nebula.chain_ops" (chainOps p i offset)
    (chainOpsStart p i offset) end2 (chainOps_assumptions inputs)
  obtain ⟨c4, ops4, end4⟩ := appendSponge c3 "nebula.chain_is" (chainInitial p i offset)
    (chainInitialStart p i offset) end3 (chainInitial_assumptions inputs)
  obtain ⟨c5, ops5, end5⟩ := appendSponge c4 "nebula.chain_fs" (chainFinal p i offset)
    (chainFinalStart p i offset) end4 (chainFinal_assumptions inputs)
  obtain ⟨c6, ops6, end6⟩ := appendSponge c5 "nebula.eta" (eta p i offset)
    (etaStart p i offset) end5 (eta_assumptions inputs)
  obtain ⟨c7, ops7, end7⟩ := appendSponge c6 "nebula.eta_squeeze_1" (squeeze1 p i offset)
    (squeeze1Start p i offset) (end6.trans (by rw [etaChunks_length]; rfl))
    (squeeze1_assumptions inputs)
  obtain ⟨c8, ops8, end8⟩ := appendSponge c7 "nebula.eta_squeeze_2" (squeeze2 p i offset)
    (squeeze2Start p i offset) (end7.trans (by
      simp only [squeeze2Start, squeeze1, permuting, List.length_singleton, one_mul]))
    (squeeze2_assumptions inputs)
  obtain ⟨c9, ops9, -⟩ := appendSponge c8 "nebula.eta_squeeze_3" (squeeze3 p i offset)
    (squeeze3Start p i offset) (end8.trans (by
      simp only [squeeze3Start, squeeze2, permuting, List.length_singleton, one_mul]))
    (squeeze3_assumptions inputs)
  refine ⟨c9, ?_⟩
  rw [ops9, ops8, ops7, ops6, ops5, ops4, ops3, ops2, ops1]
  rfl

/-- Completeness: honest execution satisfies every flattened row of a valid
step and changes only the circuit's own variables. -/
theorem completeness (two : p.bOps = 2) (env : Env) (inputs : Application.InputsBelow i offset)
    (specification : Application.Holds step i offset env ∧ Application.Valid (valid p two) i offset env) :
    ∃ completed, AgreesOutside env completed offset (localLength (opsAt p i offset)) ∧
      holdsFlat completed (opsAt p i offset) := by
  obtain ⟨holdsStep, rowsHold, machine⟩ := specification
  obtain ⟨children, operations⟩ := children_complete p i offset inputs env
  have agreesBelow : ∀ index, index < offset → children.current index = env index :=
    fun index below => children.agrees index (Or.inl below)
  have childRows : holds children.current (childOps p i offset) := by
    rw [← operations]
    exact holdsFlat_implies_holds _ _ children.rows
  have specOf : ∀ {name : String} {child : FormalCircuit} {start : ℕ},
      Sequence.childOp name child start ∈ childOps p i offset →
        child.assumptions start children.current → child.spec start children.current :=
    fun member assumptions => childRows _ member assumptions
  have witnessEq : Application.witnessValue i offset children.current =
      Application.witnessValue i offset env := by
    unfold Application.witnessValue
    congr 1
    funext index
    exact (i.witness offset index).eval_eq_of_agree_below offset _ env (inputs.witness index)
      agreesBelow
  have valuesEq : values p i offset children.current = values p i offset env := by
    unfold values
    rw [witnessEq]
  have wireFinal : ∀ k, (wire p i offset k).eval children.current = values p i offset env k :=
    fun k => by rw [wire_eval, valuesEq]
  have carryOut : ∀ (start : ℕ) (k : Fin 4), start + 3 < 39 →
      StepWitness.carryDigest (StepWitness.ofWords p (values p i offset env)).carryOut start k =
        values p i offset env (Words.carryOut (start + k.val)) := by
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
  have etaAbsorbed := absorbing_state children.current
    (specOf (name := "nebula.eta") (child := Sponge.circuit (eta p i offset))
        (start := etaStart p i offset) (by simp [childOps])
      (eta_assumptions inputs children.current))
  rw [etaBlocks_eval, valuesEq] at etaAbsorbed
  have s1 : Sponge.evalState children.current (squeeze1State p i offset) =
      Spec.Poseidon2.permute (Sponge.evalState children.current (etaState p i offset)) :=
    permuting_state children.current (specOf (name := "nebula.eta_squeeze_1")
      (child := Sponge.circuit (squeeze1 p i offset))
        (start := squeeze1Start p i offset) (by simp [childOps])
      (squeeze1_assumptions inputs children.current))
  have s2 : Sponge.evalState children.current (squeeze2State p i offset) =
      Spec.Poseidon2.permute (Sponge.evalState children.current (squeeze1State p i offset)) :=
    permuting_state children.current (specOf (name := "nebula.eta_squeeze_2")
      (child := Sponge.circuit (squeeze2 p i offset))
        (start := squeeze2Start p i offset) (by simp [childOps])
      (squeeze2_assumptions inputs children.current))
  have s3 : Sponge.evalState children.current (squeeze3State p i offset) =
      Spec.Poseidon2.permute (Sponge.evalState children.current (squeeze2State p i offset)) :=
    permuting_state children.current (specOf (name := "nebula.eta_squeeze_3")
      (child := Sponge.circuit (squeeze3 p i offset))
        (start := squeeze3Start p i offset) (by simp [childOps])
      (squeeze3_assumptions inputs children.current))
  have etaState_eq : Sponge.evalState children.current (etaState p i offset) =
      absorbed [textWords "Nightstream/Nebula/v3/eta", digestWords (planDigest p),
        [natWord ((StepWitness.ofWords p (values p i offset env)).cIn 2).val],
        digestWords ((StepWitness.ofWords p (values p i offset env)).proposalDigest 0) ++
          digestWords (StepWitness.carryDigest
            (StepWitness.ofWords p (values p i offset env)).carryIn 35) ++
          digestWords ((StepWitness.ofWords p (values p i offset env)).proposalDigest 4)] := by
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
  have assertionsLength : ∀ l : List Circuit.Expr, localLength (l.map Op.assertZero) = 0 := by
    intro l
    induction l with
    | nil => rfl
    | cons e rest ih =>
      change 0 + localLength (rest.map Op.assertZero) = 0
      rw [ih]
  refine ⟨children.current, ?_, ?_⟩
  · have agrees := children.agrees
    rw [operations] at agrees
    rw [opsAt, Sequence.localLength_append, assertionsLength, Nat.add_zero]
    exact agrees
  · rw [holdsFlat, opsAt, flatConstraints_append, flatAssertions]
    refine (constraintsHold_append _ _ _).mpr ⟨by rw [← operations]; exact children.rows, ?_⟩
    intro e member
    simp only [assertions, List.mem_append, List.mem_map, List.mem_finRange, true_and,
      List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with (((((⟨k, rfl⟩ | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) |
      (rfl | rfl | rfl | rfl)) | ⟨e', poly, rfl⟩
    · rw [Circuit.Expr.eval_sub, sub_eq_zero]
      have lanes := absorbing_lane children.current (specOf (name := "nebula.state_in")
        (child := Sponge.circuit (stateIn p i offset))
        (start := offset) (by simp [childOps])
        (stateIn_assumptions inputs children.current)) k
      rw [stateBlocks_eval, valuesEq] at lanes
      exact lanes.trans ((congrFun (List.ofFn_injective rowsHold.stateIn) k).trans
        ((i.input offset k).eval_eq_of_agree_below offset _ env (inputs.input k) agreesBelow).symm)
    · rw [Circuit.Expr.eval_sub, sub_eq_zero]
      have lanes := absorbing_lane children.current (specOf (name := "nebula.state_out")
        (child := Sponge.circuit (stateOut p i offset))
        (start := stateOutStart p i offset) (by simp [childOps])
        (stateOut_assumptions inputs children.current)) k
      rw [stateBlocks_eval, valuesEq] at lanes
      exact lanes.trans ((congrFun (List.ofFn_injective holdsStep) k).symm.trans
        ((i.output offset k).eval_eq_of_agree_below offset _ env (inputs.output k) agreesBelow).symm)
    · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal]
      have lanes := absorbing_lane children.current (specOf (name := "nebula.chain_ops")
        (child := Sponge.circuit (chainOps p i offset))
        (start := chainOpsStart p i offset) (by simp [childOps])
        (chainOps_assumptions inputs children.current)) k
      rw [chainBlocks_eval p i offset _ .ops 0 (by norm_num), opsLanes_eval, valuesEq] at lanes
      exact lanes.trans ((congrFun rowsHold.chainOps k).symm.trans (carryOut 23 k (by norm_num)))
    · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal]
      have lanes := absorbing_lane children.current (specOf (name := "nebula.chain_is")
        (child := Sponge.circuit (chainInitial p i offset))
        (start := chainInitialStart p i offset) (by simp [childOps])
        (chainInitial_assumptions inputs children.current)) k
      rw [chainBlocks_eval p i offset _ .mem 4 (by norm_num),
        scanLanes_eval p i offset _ _ _ fun j => Sym.eval_initialLane p _ j, valuesEq] at lanes
      exact lanes.trans ((congrFun rowsHold.chainInitial k).symm.trans (carryOut 27 k (by norm_num)))
    · rw [Circuit.Expr.eval_sub, sub_eq_zero, wireFinal]
      have lanes := absorbing_lane children.current (specOf (name := "nebula.chain_fs")
        (child := Sponge.circuit (chainFinal p i offset))
        (start := chainFinalStart p i offset) (by simp [childOps])
        (chainFinal_assumptions inputs children.current)) k
      rw [chainBlocks_eval p i offset _ .mem 8 (by norm_num),
        scanLanes_eval p i offset _ _ _ fun j => Sym.eval_finalLane p _ j, valuesEq] at lanes
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
      exact (Rows.polyRows_iff p _ two).mpr ⟨rowsHold.toPolyRows, machine⟩ e' poly

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
