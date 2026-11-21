import NightstreamFPrime.Lifecycle.Nebula.MemoryLiveRows
import NightstreamFPrime.Lifecycle.Nebula.MemoryCompleteness

/-! Owns a memory application as a closed Stage 1 `Application.Program`: the
circuit, its exact step and validity predicate, and the support contract, for a
geometry. It reads no image, so one program serves every plan of a geometry. It
does not select a geometry, an application, or a package. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

variable (p : Geometry) (app : PortApp p) (i : AppInterface p app) (offset : ℕ)

theorem assertions_localLength (l : List Circuit.Expr) : localLength (l.map Op.assertZero) = 0 := by
  induction l with
  | nil => rfl
  | cons e rest ih =>
    change 0 + localLength (rest.map Op.assertZero) = 0
    rw [ih]

theorem localLength_cons (o : Op) (l : List Op) : localLength (o :: l) = o.localLength + localLength l :=
  rfl

theorem child_localLength (child : Sponge.Interface) (name : String) (start : ℕ) :
    (Sequence.childOp name (Sponge.circuit child) start).localLength =
      (child.chunks start).length * 1096 :=
  (Sequence.childOp_localLength _ _ _).trans (Sponge.localLength_eq child start)

/-- The circuit allocates exactly the children's variables. -/
theorem localLength_opsAt : offset + localLength (opsAt p app i offset) = endOffset p app i offset := by
  obtain ⟨e1, e2, e3, e4, e5, e6, e7, e8, e9⟩ := child_ends p app i offset
  rw [opsAt, Sequence.localLength_append, assertions_localLength, Nat.add_zero, childOps]
  simp only [localLength_cons, child_localLength, show localLength ([] : List Op) = 0 from rfl,
    Nat.add_zero, ← Nat.add_assoc, e1, e2, e3, e4, e5, e6, e7, e8, e9]

theorem witnesses_opsAt :
    witnesses (opsAt p app i offset) = witnesses (childOps p app i offset) := by
  rw [opsAt, Witness.witnesses_append, Witness.witnesses_assertions, List.append_nil]

theorem ordered (inputs : Application.InputsBelow i offset) :
    Witness.Ordered offset (witnesses (opsAt p app i offset))
      (offset + localLength (opsAt p app i offset)) := by
  obtain ⟨e1, e2, e3, e4, e5, e6, e7, e8, e9⟩ := child_ends p app i offset
  have child : ∀ (child : Sponge.Interface) (start next : ℕ),
      Sponge.Assumptions child start (fun _ => 0) →
        start + (child.chunks start).length * 1096 = next →
          Witness.Ordered start (witnesses (Sponge.opsAt child start)) next := by
    intro child start next assumptions finish
    have ordered := Sponge.ordered child start _ assumptions
    rwa [show localLength (Sponge.opsAt child start) = _ from Sponge.localLength_eq child start,
      finish] at ordered
  have chain :=
    (child _ _ _ (stateIn_assumptions inputs _) e1).append <|
    (child _ _ _ (stateOut_assumptions inputs _) e2).append <|
    (child _ _ _ (chainOps_assumptions inputs _) e3).append <|
    (child _ _ _ (chainInitial_assumptions inputs _) e4).append <|
    (child _ _ _ (chainFinal_assumptions inputs _) e5).append <|
    (child _ _ _ (eta_assumptions inputs _) e6).append <|
    (child _ _ _ (squeeze1_assumptions inputs _) e7).append <|
    (child _ _ _ (squeeze2_assumptions inputs _) e8).append
      (child _ _ _ (squeeze3_assumptions inputs _) e9)
  rw [witnesses_opsAt, localLength_opsAt]
  exact chain

private theorem child_holdsFlat (env : Env)
    (computed : Witness.Computed env (witnesses (childOps p app i offset)))
    {name : String} {child : Sponge.Interface} {start : ℕ}
    (member : Sequence.childOp name (Sponge.circuit child) start ∈ childOps p app i offset) :
    ConstraintsHold env (Sequence.childOp name (Sponge.circuit child) start).flatConstraints :=
  Sponge.computed_holdsFlat child env start fun batch inside =>
    computed batch (List.mem_flatMap.mpr ⟨_, member, inside⟩)

theorem children_holdsFlat (env : Env)
    (computed : Witness.Computed env (witnesses (childOps p app i offset))) :
    holdsFlat env (childOps p app i offset) := by
  intro expression member
  obtain ⟨operation, operationMember, rowMember⟩ := List.mem_flatMap.mp member
  simp only [childOps, List.mem_cons, List.not_mem_nil, or_false] at operationMember
  rcases operationMember with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact child_holdsFlat p app i offset env computed (by simp [childOps]) expression rowMember

theorem computed_holdsFlat (env : Env) (inputs : Application.InputsBelow i offset)
    (specification : Application.Holds (step p app) i offset env ∧
      Application.Valid (valid p app) i offset env)
    (computed : Witness.Computed env (witnesses (opsAt p app i offset))) :
    holdsFlat env (opsAt p app i offset) :=
  rows_of_children p app i offset env env inputs specification (fun _ _ => rfl)
    (children_holdsFlat p app i offset env (by rwa [witnesses_opsAt] at computed))

/-- Completeness: honest execution satisfies every flattened row of a valid
step and changes only the circuit's own variables. -/
theorem completeness (env : Env) (inputs : Application.InputsBelow i offset)
    (specification : Application.Holds (step p app) i offset env ∧
      Application.Valid (valid p app) i offset env) :
    ∃ completed, AgreesOutside env completed offset (localLength (opsAt p app i offset)) ∧
      holdsFlat completed (opsAt p app i offset) :=
  Witness.complete (ordered p app i offset inputs) fun completed below computed =>
    rows_of_children p app i offset env completed inputs specification below
      (children_holdsFlat p app i offset completed (by rwa [witnesses_opsAt] at computed))

/-- Every flattened row reads only supported caller wires or the circuit's own
variables. -/
theorem supported (allowed : ℕ → Prop) (inputs : Application.InputsSupported i offset allowed)
    (localSupport : ∀ index, offset ≤ index →
      index < offset + localLength (opsAt p app i offset) → allowed index) :
    ∀ expression ∈ flatConstraints (opsAt p app i offset), expression.VarsSatisfy allowed := by
  have wires : WiresSupported p app i offset allowed := inputs.witness
  have total := localLength_opsAt p app i offset
  obtain ⟨e1, e2, e3, e4, e5, e6, e7, e8, e9⟩ := child_ends p app i offset
  have within : ∀ (child : Sponge.Interface) (start : ℕ), offset ≤ start →
      start + (child.chunks start).length * 1096 ≤ endOffset p app i offset →
      ∀ index, start ≤ index → index < start + localLength (Circuit.ops (Sponge.main child) start) →
        allowed index := by
    intro child start lower upper index low high
    rw [Sponge.localLength_eq] at high
    exact localSupport index (by omega) (by omega)
  have stateInS := Sponge.supported (stateIn p app i offset) offset allowed (fun _ => trivial)
    (transcriptChunks_supported (stateBlocks_supported wires _ _)) (within _ _ le_rfl (by omega))
  have stateOutS := Sponge.supported (stateOut p app i offset) (stateOutStart p app i offset) allowed
    (fun _ => trivial) (transcriptChunks_supported (stateBlocks_supported wires _ _))
    (within _ _ (by omega) (by omega))
  have chainOpsS := Sponge.supported (chainOps p app i offset) (chainOpsStart p app i offset) allowed
    (fun _ => trivial) (transcriptChunks_supported (chainBlocks_supported wires _ _ _))
    (within _ _ (by omega) (by omega))
  have chainInitialS := Sponge.supported (chainInitial p app i offset) (chainInitialStart p app i offset)
    allowed (fun _ => trivial) (transcriptChunks_supported (chainBlocks_supported wires _ _ _))
    (within _ _ (by omega) (by omega))
  have chainFinalS := Sponge.supported (chainFinal p app i offset) (chainFinalStart p app i offset)
    allowed (fun _ => trivial) (transcriptChunks_supported (chainBlocks_supported wires _ _ _))
    (within _ _ (by omega) (by omega))
  have etaS := Sponge.supported (eta p app i offset) (etaStart p app i offset) allowed (fun _ => trivial)
    (transcriptChunks_supported (etaBlocks_supported wires)) (within _ _ (by omega) (by omega))
  have noChunks : ∀ chunk ∈ ([[]] : List (List Circuit.Expr)), ∀ e ∈ chunk, e.VarsSatisfy allowed := by
    simp
  have etaStateS : ∀ lane, (etaState p app i offset lane).VarsSatisfy allowed := fun lane => by
    rw [etaState_output]
    exact etaS.2 lane
  have squeeze1S := Sponge.supported (squeeze1 p app i offset) (squeeze1Start p app i offset) allowed
    etaStateS noChunks (within _ _ (by omega) (by omega))
  have squeeze2S := Sponge.supported (squeeze2 p app i offset) (squeeze2Start p app i offset) allowed
    squeeze1S.2 noChunks (within _ _ (by omega) (by omega))
  have squeeze3S := Sponge.supported (squeeze3 p app i offset) (squeeze3Start p app i offset) allowed
    squeeze2S.2 noChunks (within _ _ (by omega) (by omega))
  have flatAssertions : ∀ l : List Circuit.Expr, flatConstraints (l.map Op.assertZero) = l := by
    intro l
    induction l with
    | nil => rfl
    | cons e rest ih =>
      change [e] ++ flatConstraints (rest.map Op.assertZero) = e :: rest
      rw [ih]
      rfl
  have laneOf : ∀ {state : Sponge.EState}, (∀ lane, (state lane).VarsSatisfy allowed) →
      ∀ k, (lane state k).VarsSatisfy allowed := fun each k => each _
  intro expression member
  rw [opsAt, flatConstraints_append, flatAssertions, List.mem_append] at member
  rcases member with member | member
  · obtain ⟨op, opMember, member⟩ := List.mem_flatMap.mp member
    simp only [childOps, List.mem_cons, List.not_mem_nil, or_false] at opMember
    rcases opMember with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact stateInS.1 _ member
    · exact stateOutS.1 _ member
    · exact chainOpsS.1 _ member
    · exact chainInitialS.1 _ member
    · exact chainFinalS.1 _ member
    · exact etaS.1 _ member
    · exact squeeze1S.1 _ member
    · exact squeeze2S.1 _ member
    · exact squeeze3S.1 _ member
  · simp only [assertions, hashChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append,
      List.mem_append, List.mem_map, List.mem_finRange, true_and,
      List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with (((((⟨k, rfl⟩ | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) |
      (rfl | rfl | rfl | rfl)) | ⟨e', -, rfl⟩
    · exact Circuit.Expr.VarsSatisfy.sub _ _ _ (laneOf stateInS.2 k) (inputs.input k)
    · exact Circuit.Expr.VarsSatisfy.sub _ _ _ (laneOf stateOutS.2 k) (inputs.output k)
    · exact Circuit.Expr.VarsSatisfy.sub _ _ _ (laneOf chainOpsS.2 k) (wire_supported wires _)
    · exact Circuit.Expr.VarsSatisfy.sub _ _ _ (laneOf chainInitialS.2 k) (wire_supported wires _)
    · exact Circuit.Expr.VarsSatisfy.sub _ _ _ (laneOf chainFinalS.2 k) (wire_supported wires _)
    · exact Circuit.Expr.VarsSatisfy.sub _ _ _ (laneOf etaStateS 0) (wire_supported wires _)
    · exact Circuit.Expr.VarsSatisfy.sub _ _ _ (laneOf squeeze1S.2 0) (wire_supported wires _)
    · exact Circuit.Expr.VarsSatisfy.sub _ _ _ (laneOf squeeze2S.2 0) (wire_supported wires _)
    · exact Circuit.Expr.VarsSatisfy.sub _ _ _ (laneOf squeeze3S.2 0) (wire_supported wires _)
    · exact sub_supported wires e'

/-- The memory-application circuit. -/
def circuit (i : AppInterface p app) : FormalCircuit where
  main := main p app i
  assumptions := fun offset _ => Application.InputsBelow i offset
  spec := fun offset env =>
    Application.Holds (step p app) i offset env ∧ Application.Valid (valid p app) i offset env
  soundness := fun env offset inputs rows => soundness p app i offset env inputs rows
  completeness := fun env offset inputs specification =>
    completeness p app i offset env inputs specification

/-- The memory application as a closed Stage 1 program. -/
def program : Application.Program where
  witnessWordCount := Words.count p app.toAppShape
  step := step p app
  step_length := fun _ _ _ => stateWords_length _ _
  valid := valid p app
  circuit := circuit p app
  spec_iff := fun _ _ _ => Iff.rfl
  assumptions_of_inputsBelow := fun _ _ _ inputs => inputs
  constraintsSupported := fun i offset _ allowed _ inputs localSupport =>
    supported p app i offset allowed inputs localSupport
  marksPlaced := fun i offset => marksPlaced p app i offset
  marksLiveRows := fun i offset live affine inputs own =>
    marksLiveRows p app i offset live affine inputs own

/-- One setup per geometry (SPEC.md section 7.1): the memory program takes a geometry and
no image, so two plans with one geometry have the same memory program. -/
theorem program_eq_of_geometry (appOf : (g : Geometry) → PortApp g) {plan plan' : Plan}
    (same : plan.toGeometry = plan'.toGeometry) :
    program plan.toGeometry (appOf plan.toGeometry) =
      program plan'.toGeometry (appOf plan'.toGeometry) := by
  rw [same]

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
