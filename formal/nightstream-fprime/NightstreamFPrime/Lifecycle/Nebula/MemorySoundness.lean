import NightstreamFPrime.Lifecycle.Nebula.MemorySupport

/-! Owns the soundness of the memory-application circuit: when its rows hold
and the caller's wires lie below the call offset, the output state is the step
of the input and witness, and the decoded witness satisfies the memory rows on
the input state and the application's rows. It does not own completeness. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

variable (p : Geometry) (app : PortApp p) (i : AppInterface p app) (offset : ℕ) (env : Env)

local notation "W" => StepWitness.ofWords p app.toAppShape (values p app i offset env)

/-! ### Child facts -/

/-- An absorbing child's state is the transcript state of its evaluated
blocks. -/
theorem absorbing_state {blocks : List (List Circuit.Expr)} {start : ℕ}
    (spec : Sponge.SpecHolds (absorbing blocks) start env) :
    Sponge.evalState env (Sponge.output (absorbing blocks) start) =
      absorbed (blocks.map (Hash.evalList env)) := by
  rw [spec]
  change List.foldl Spec.Poseidon2.absorbBlock (Sponge.evalState env Hash.zeroE)
    ((transcriptChunks blocks).map (Hash.evalList env)) = _
  rw [fold_transcriptChunks, evalState_zero]
  rfl

/-- An absorbing child's digest lanes are the transcript digest of its
evaluated blocks. -/
theorem absorbing_lane {blocks : List (List Circuit.Expr)} {start : ℕ}
    (spec : Sponge.SpecHolds (absorbing blocks) start env) (k : Fin 4) :
    (lane (Sponge.output (absorbing blocks) start) k).eval env =
      squeezeDigest (absorbed (blocks.map (Hash.evalList env))) k := by
  rw [lane, lane_eval, spec]
  change (List.foldl Spec.Poseidon2.absorbBlock (Sponge.evalState env Hash.zeroE)
    ((transcriptChunks blocks).map (Hash.evalList env))).getD k.val 0 = _
  rw [fold_transcriptChunks, evalState_zero]
  rfl

/-- A permuting child's state is one permutation of its input state. -/
theorem permuting_state {state : Sponge.EState} {start : ℕ}
    (spec : Sponge.SpecHolds (permuting state) start env) :
    Sponge.evalState env (Sponge.output (permuting state) start) =
      Spec.Poseidon2.permute (Sponge.evalState env state) := by
  rw [spec]
  change List.foldl Spec.Poseidon2.absorbBlock (Sponge.evalState env state) [[]] = _
  rw [List.foldl_cons, List.foldl_nil]
  exact absorbBlock_nil (evalState_length env state)

/-- The challenge transcript after its absorbs: two extension squeezes. -/
theorem etaChallenges_eq (inp : EtaInput Digest) :
    etaChallenges inp =
      let s := absorbed [textWords "Nightstream/Nebula/v3/eta", digestWords inp.planDigest,
        [natWord inp.ts],
        digestWords inp.opsRoot ++ digestWords inp.memRoot ++ digestWords inp.finalRoot]
      ((⟨s.getD 0 0, (Spec.Poseidon2.permute s).getD 0 0⟩ : K),
        (⟨(Spec.Poseidon2.permute (Spec.Poseidon2.permute s)).getD 0 0,
          (Spec.Poseidon2.permute (Spec.Poseidon2.permute (Spec.Poseidon2.permute s))).getD 0 0⟩ :
            K)) :=
  rfl

/-! ### Rows of the operation list -/

section Rows

variable {p app i offset env}

theorem child_spec (rows : holds env (opsAt p app i offset)) {name : String} {child : FormalCircuit}
    {start : ℕ} (member : Sequence.childOp name child start ∈ childOps p app i offset)
    (assumptions : child.assumptions start env) : child.spec start env :=
  rows _ (List.mem_append_left _ member) assumptions

theorem assertion_holds (rows : holds env (opsAt p app i offset)) {e : Circuit.Expr}
    (member : e ∈ assertions p app i offset) : e.eval env = 0 :=
  rows (Op.assertZero e) (List.mem_append_right _ (List.mem_map_of_mem member))

theorem lane_eq (rows : holds env (opsAt p app i offset)) {a b : Circuit.Expr}
    (member : a - b ∈ assertions p app i offset) : a.eval env = b.eval env := by
  have zero := assertion_holds rows member
  rwa [Circuit.Expr.eval_sub, sub_eq_zero] at zero

end Rows

/-! ### Assertion membership -/

section Members

variable {p app i offset}

theorem mem_stateIn (k : Fin 4) :
    lane (Sponge.output (stateIn p app i offset) offset) k - i.input offset k ∈ assertions p app i offset := by
  simp only [assertions, hashChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append, List.mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (List.mem_map_of_mem (List.mem_finRange k)))))))

theorem mem_stateOut (k : Fin 4) :
    lane (Sponge.output (stateOut p app i offset) (stateOutStart p app i offset)) k - i.output offset k ∈
      assertions p app i offset := by
  simp only [assertions, hashChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append, List.mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr (List.mem_map_of_mem (List.mem_finRange k)))))))

theorem mem_chainOps (k : Fin 4) :
    lane (Sponge.output (chainOps p app i offset) (chainOpsStart p app i offset)) k -
      wire p app i offset (Words.carryOut (23 + k.val)) ∈ assertions p app i offset := by
  simp only [assertions, hashChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append, List.mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inr (List.mem_map_of_mem (List.mem_finRange k))))))

theorem mem_chainInitial (k : Fin 4) :
    lane (Sponge.output (chainInitial p app i offset) (chainInitialStart p app i offset)) k -
      wire p app i offset (Words.carryOut (27 + k.val)) ∈ assertions p app i offset := by
  simp only [assertions, hashChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append, List.mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inr (List.mem_map_of_mem (List.mem_finRange k)))))

theorem mem_chainFinal (k : Fin 4) :
    lane (Sponge.output (chainFinal p app i offset) (chainFinalStart p app i offset)) k -
      wire p app i offset (Words.carryOut (31 + k.val)) ∈ assertions p app i offset := by
  simp only [assertions, hashChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append, List.mem_append]
  exact Or.inl (Or.inl (Or.inr (List.mem_map_of_mem (List.mem_finRange k))))

theorem mem_eta :
    lane (etaState p app i offset) 0 - wire p app i offset (Words.etaFresh 0) ∈ assertions p app i offset ∧
    lane (squeeze1State p app i offset) 0 - wire p app i offset (Words.etaFresh 1) ∈ assertions p app i offset ∧
    lane (squeeze2State p app i offset) 0 - wire p app i offset (Words.etaFresh 2) ∈ assertions p app i offset ∧
    lane (squeeze3State p app i offset) 0 - wire p app i offset (Words.etaFresh 3) ∈ assertions p app i offset := by
  simp only [assertions, hashChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append, List.mem_append, List.mem_cons, true_or, or_true, and_self]

theorem mem_poly {e : Circuit.Expr} (member : e ∈ Rows.polyRows p app) :
    sub p app i offset e ∈ assertions p app i offset :=
  List.mem_append_right _ (List.mem_map_of_mem member)

end Members

/-! ### Soundness -/

theorem soundness (inputs : Application.InputsBelow i offset)
    (rows : holds env (opsAt p app i offset)) :
    Application.Holds (step p app) i offset env ∧ Application.Valid (valid p app) i offset env := by
  have below := wiresSupported_below (p := p) inputs
  have stateInSpec : Sponge.SpecHolds (stateIn p app i offset) offset env :=
    child_spec (name := "nebula.state_in") (child := Sponge.circuit (stateIn p app i offset)) rows (by simp [childOps])
      (absorbing_assumptions env le_rfl (stateBlocks_supported below _ _))
  have stateOutSpec : Sponge.SpecHolds (stateOut p app i offset) (stateOutStart p app i offset) env :=
    child_spec (name := "nebula.state_out") (child := Sponge.circuit (stateOut p app i offset)) rows (by simp [childOps])
      (absorbing_assumptions env (by unfold stateOutStart; omega) (stateBlocks_supported below _ _))
  have chainOpsSpec : Sponge.SpecHolds (chainOps p app i offset) (chainOpsStart p app i offset) env :=
    child_spec (name := "nebula.chain_ops") (child := Sponge.circuit (chainOps p app i offset)) rows (by simp [childOps])
      (absorbing_assumptions env (by unfold chainOpsStart stateOutStart; omega)
        (chainBlocks_supported below _ _ _))
  have chainInitialSpec :
      Sponge.SpecHolds (chainInitial p app i offset) (chainInitialStart p app i offset) env :=
    child_spec (name := "nebula.chain_is") (child := Sponge.circuit (chainInitial p app i offset)) rows (by simp [childOps])
      (absorbing_assumptions env (by unfold chainInitialStart chainOpsStart stateOutStart; omega)
        (chainBlocks_supported below _ _ _))
  have chainFinalSpec : Sponge.SpecHolds (chainFinal p app i offset) (chainFinalStart p app i offset) env :=
    child_spec (name := "nebula.chain_fs") (child := Sponge.circuit (chainFinal p app i offset)) rows (by simp [childOps])
      (absorbing_assumptions env
        (by unfold chainFinalStart chainInitialStart chainOpsStart stateOutStart; omega)
        (chainBlocks_supported below _ _ _))
  have etaAssumptions : Sponge.Assumptions (eta p app i offset) (etaStart p app i offset) env :=
    absorbing_assumptions env
      (by unfold etaStart chainFinalStart chainInitialStart chainOpsStart stateOutStart; omega)
      (etaBlocks_supported below)
  have etaSpec : Sponge.SpecHolds (eta p app i offset) (etaStart p app i offset) env :=
    child_spec (name := "nebula.eta") (child := Sponge.circuit (eta p app i offset)) rows (by simp [childOps]) etaAssumptions
  have noBlocks : ∀ start, Hash.BlocksBelow start [[]] := fun _ => by simp [Hash.BlocksBelow]
  have squeeze1Assumptions :
      Sponge.Assumptions (squeeze1 p app i offset) (squeeze1Start p app i offset) env := by
    refine ⟨fun lane => ?_, noBlocks _⟩
    have scope := Sponge.output_varsBelow (eta p app i offset) (etaStart p app i offset) env etaAssumptions lane
    have bound : etaStart p app i offset + ((eta p app i offset).chunks (etaStart p app i offset)).length * 1096 =
        squeeze1Start p app i offset := by
      rw [etaChunks_length]
      rfl
    rw [Sponge.localLength_eq, bound, ← etaState_output] at scope
    exact scope
  have squeeze1Spec : Sponge.SpecHolds (squeeze1 p app i offset) (squeeze1Start p app i offset) env :=
    child_spec (name := "nebula.eta_squeeze_1") (child := Sponge.circuit (squeeze1 p app i offset)) rows (by simp [childOps])
      squeeze1Assumptions
  have squeeze2Assumptions :
      Sponge.Assumptions (squeeze2 p app i offset) (squeeze2Start p app i offset) env := by
    refine ⟨fun lane => ?_, noBlocks _⟩
    have scope := Sponge.output_varsBelow (squeeze1 p app i offset) (squeeze1Start p app i offset) env
      squeeze1Assumptions lane
    have bound : squeeze1Start p app i offset +
        ((squeeze1 p app i offset).chunks (squeeze1Start p app i offset)).length * 1096 =
        squeeze2Start p app i offset := by
      simp only [squeeze2Start, squeeze1, permuting, List.length_singleton, one_mul]
    rw [Sponge.localLength_eq, bound] at scope
    exact scope
  have squeeze2Spec : Sponge.SpecHolds (squeeze2 p app i offset) (squeeze2Start p app i offset) env :=
    child_spec (name := "nebula.eta_squeeze_2") (child := Sponge.circuit (squeeze2 p app i offset)) rows (by simp [childOps])
      squeeze2Assumptions
  have squeeze3Assumptions :
      Sponge.Assumptions (squeeze3 p app i offset) (squeeze3Start p app i offset) env := by
    refine ⟨fun lane => ?_, noBlocks _⟩
    have scope := Sponge.output_varsBelow (squeeze2 p app i offset) (squeeze2Start p app i offset) env
      squeeze2Assumptions lane
    have bound : squeeze2Start p app i offset +
        ((squeeze2 p app i offset).chunks (squeeze2Start p app i offset)).length * 1096 =
        squeeze3Start p app i offset := by
      simp only [squeeze3Start, squeeze2, permuting, List.length_singleton, one_mul]
    rw [Sponge.localLength_eq, bound] at scope
    exact scope
  have squeeze3Spec : Sponge.SpecHolds (squeeze3 p app i offset) (squeeze3Start p app i offset) env :=
    child_spec (name := "nebula.eta_squeeze_3") (child := Sponge.circuit (squeeze3 p app i offset)) rows (by simp [childOps])
      squeeze3Assumptions
  -- Polynomial rows.
  have poly : ConstraintsHold (values p app i offset env) (Rows.polyRows p app) := fun e member => by
    have zero := assertion_holds rows (mem_poly (i := i) (offset := offset) member)
    rwa [sub_eval] at zero
  obtain ⟨polyRows, appRows⟩ := (Rows.polyRows_iff p _ app).mp poly
  -- Digest rows.
  have inputEq : stateWords (List.ofFn (W).appIn) (List.ofFn (W).carryIn) =
      Application.inputState i offset env := by
    unfold Application.inputState stateWords digestWords
    congr 1
    funext k
    have lanes := absorbing_lane env stateInSpec k
    rw [stateBlocks_eval] at lanes
    exact lanes.symm.trans (lane_eq rows (mem_stateIn k))
  have outputEq : Application.outputState i offset env =
      step p app (Application.inputState i offset env) (Application.witnessValue i offset env) := by
    unfold Application.outputState step stateWords digestWords
    congr 1
    funext k
    have lanes := absorbing_lane env stateOutSpec k
    rw [stateBlocks_eval] at lanes
    exact (lane_eq rows (mem_stateOut k)).symm.trans lanes
  have carryOut : ∀ (start : ℕ) (k : Fin 4), start + 3 < 39 →
      StepWitness.carryDigest (W).carryOut start k =
        values p app i offset env (Words.carryOut (start + k.val)) := by
    intro start k small
    simp [StepWitness.carryDigest, StepWitness.carryWord, StepWitness.ofWords,
      show start + k.val < 39 by omega]
  have chainOpsEq : StepWitness.carryDigest (W).carryOut 23 =
      chainLink .ops (W).idxEff ((W).previousDigest 0) (packWords (W).opsLaneBits) := by
    funext k
    have lanes := absorbing_lane env chainOpsSpec k
    rw [chainBlocks_eval p app i offset env .ops 0 (by norm_num), opsLanes_eval] at lanes
    have wireEq := lane_eq rows (mem_chainOps k)
    rw [wire_eval] at wireEq
    rw [carryOut 23 k (by norm_num), ← wireEq]
    exact lanes
  have chainInitialEq : StepWitness.carryDigest (W).carryOut 27 =
      chainLink .mem (W).idxEff ((W).previousDigest 4)
        (packWords (StepWitness.scanLaneBits (W).initial)) := by
    funext k
    have lanes := absorbing_lane env chainInitialSpec k
    rw [chainBlocks_eval p app i offset env .mem 4 (by norm_num),
      scanLanes_eval p app i offset env (W).initial _ fun j => Sym.eval_initialLane p app.toAppShape _ j] at lanes
    have wireEq := lane_eq rows (mem_chainInitial k)
    rw [wire_eval] at wireEq
    rw [carryOut 27 k (by norm_num), ← wireEq]
    exact lanes
  have chainFinalEq : StepWitness.carryDigest (W).carryOut 31 =
      chainLink .mem (W).idxEff ((W).previousDigest 8)
        (packWords (StepWitness.scanLaneBits (W).final)) := by
    funext k
    have lanes := absorbing_lane env chainFinalSpec k
    rw [chainBlocks_eval p app i offset env .mem 8 (by norm_num),
      scanLanes_eval p app i offset env (W).final _ fun j => Sym.eval_finalLane p app.toAppShape _ j] at lanes
    have wireEq := lane_eq rows (mem_chainFinal k)
    rw [wire_eval] at wireEq
    rw [carryOut 31 k (by norm_num), ← wireEq]
    exact lanes
  have freshEq : etaChallenges ⟨planDigest p, ((W).cIn 2).val, (W).proposalDigest 0,
      StepWitness.carryDigest (W).carryIn 35, (W).proposalDigest 4⟩ = (W).freshEta := by
    have etaAbsorbed := absorbing_state env etaSpec
    rw [etaBlocks_eval] at etaAbsorbed
    have etaAbsorbed' :=
      (congrArg (Sponge.evalState env) (etaState_output p app i offset)).trans etaAbsorbed
    have s1 := permuting_state env squeeze1Spec
    have s2 := permuting_state env squeeze2Spec
    have s3 := permuting_state env squeeze3Spec
    obtain ⟨m0, m1, m2, m3⟩ := mem_eta (p := p) (i := i) (offset := offset)
    have e0 := lane_eq rows m0
    have e1 := lane_eq rows m1
    have e2 := lane_eq rows m2
    have e3 := lane_eq rows m3
    rw [wire_eval, lane, lane_eval] at e0 e1 e2 e3
    have s1' : Sponge.evalState env (squeeze1State p app i offset) =
        Spec.Poseidon2.permute (Sponge.evalState env (etaState p app i offset)) := s1
    have s2' : Sponge.evalState env (squeeze2State p app i offset) =
        Spec.Poseidon2.permute (Sponge.evalState env (squeeze1State p app i offset)) := s2
    have s3' : Sponge.evalState env (squeeze3State p app i offset) =
        Spec.Poseidon2.permute (Sponge.evalState env (squeeze2State p app i offset)) := s3
    rw [s1'] at e1
    rw [s2', s1'] at e2
    rw [s3', s2', s1'] at e3
    rw [etaChallenges_eq]
    dsimp only
    rw [← etaAbsorbed']
    exact Prod.ext (congrArg₂ K.mk e0 e1) (congrArg₂ K.mk e2 e3)
  refine ⟨outputEq, ?_, (congrArg (fun e => ConstraintsHold e app.rows)
    (view_env p app i offset env)).mpr appRows⟩
  exact {
    toPolyRows := polyRows, chainOps := chainOpsEq, chainInitial := chainInitialEq,
    chainFinal := chainFinalEq, freshEta := freshEq, stateIn := inputEq }

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
