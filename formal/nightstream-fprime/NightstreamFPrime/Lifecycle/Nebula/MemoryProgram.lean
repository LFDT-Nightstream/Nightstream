import NightstreamFPrime.Lifecycle.Nebula.MemoryCompleteness

/-! Owns the first memory application as a closed Stage 1 `Application.Program`
for a plan with two ports: the circuit, its exact step and validity predicate,
and the support contract. It does not select a plan or a package. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

variable (p : Plan) (i : AppInterface p) (offset : ℕ)

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

/-- Each child ends where the next starts. -/
theorem child_ends :
    offset + ((stateIn p i offset).chunks offset).length * 1096 = stateOutStart p i offset ∧
    stateOutStart p i offset + ((stateOut p i offset).chunks (stateOutStart p i offset)).length * 1096 =
      chainOpsStart p i offset ∧
    chainOpsStart p i offset +
      ((chainOps p i offset).chunks (chainOpsStart p i offset)).length * 1096 =
      chainInitialStart p i offset ∧
    chainInitialStart p i offset +
      ((chainInitial p i offset).chunks (chainInitialStart p i offset)).length * 1096 =
      chainFinalStart p i offset ∧
    chainFinalStart p i offset +
      ((chainFinal p i offset).chunks (chainFinalStart p i offset)).length * 1096 =
      etaStart p i offset ∧
    etaStart p i offset + ((eta p i offset).chunks (etaStart p i offset)).length * 1096 =
      squeeze1Start p i offset ∧
    squeeze1Start p i offset +
      ((squeeze1 p i offset).chunks (squeeze1Start p i offset)).length * 1096 =
      squeeze2Start p i offset ∧
    squeeze2Start p i offset +
      ((squeeze2 p i offset).chunks (squeeze2Start p i offset)).length * 1096 =
      squeeze3Start p i offset ∧
    squeeze3Start p i offset +
      ((squeeze3 p i offset).chunks (squeeze3Start p i offset)).length * 1096 =
      endOffset p i offset := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, by rw [etaChunks_length]; rfl, ?_, ?_, ?_⟩
  · simp only [squeeze2Start, squeeze1, permuting, List.length_singleton, one_mul]
  · simp only [squeeze3Start, squeeze2, permuting, List.length_singleton, one_mul]
  · simp only [endOffset, squeeze3, permuting, List.length_singleton, one_mul]

/-- The circuit allocates exactly the children's variables. -/
theorem localLength_opsAt : offset + localLength (opsAt p i offset) = endOffset p i offset := by
  obtain ⟨e1, e2, e3, e4, e5, e6, e7, e8, e9⟩ := child_ends p i offset
  rw [opsAt, Sequence.localLength_append, assertions_localLength, Nat.add_zero, childOps]
  simp only [localLength_cons, child_localLength, show localLength ([] : List Op) = 0 from rfl,
    Nat.add_zero, ← Nat.add_assoc, e1, e2, e3, e4, e5, e6, e7, e8, e9]

/-- Every flattened row reads only supported caller wires or the circuit's own
variables. -/
theorem supported (allowed : ℕ → Prop) (inputs : Application.InputsSupported i offset allowed)
    (localSupport : ∀ index, offset ≤ index →
      index < offset + localLength (opsAt p i offset) → allowed index) :
    ∀ expression ∈ flatConstraints (opsAt p i offset), expression.VarsSatisfy allowed := by
  have wires : WiresSupported p i offset allowed := inputs.witness
  have total := localLength_opsAt p i offset
  obtain ⟨e1, e2, e3, e4, e5, e6, e7, e8, e9⟩ := child_ends p i offset
  have within : ∀ (child : Sponge.Interface) (start : ℕ), offset ≤ start →
      start + (child.chunks start).length * 1096 ≤ endOffset p i offset →
      ∀ index, start ≤ index → index < start + localLength (Circuit.ops (Sponge.main child) start) →
        allowed index := by
    intro child start lower upper index low high
    rw [Sponge.localLength_eq] at high
    exact localSupport index (by omega) (by omega)
  have stateInS := Sponge.supported (stateIn p i offset) offset allowed (fun _ => trivial)
    (transcriptChunks_supported (stateBlocks_supported wires _ _)) (within _ _ le_rfl (by omega))
  have stateOutS := Sponge.supported (stateOut p i offset) (stateOutStart p i offset) allowed
    (fun _ => trivial) (transcriptChunks_supported (stateBlocks_supported wires _ _))
    (within _ _ (by omega) (by omega))
  have chainOpsS := Sponge.supported (chainOps p i offset) (chainOpsStart p i offset) allowed
    (fun _ => trivial) (transcriptChunks_supported (chainBlocks_supported wires _ _ _))
    (within _ _ (by omega) (by omega))
  have chainInitialS := Sponge.supported (chainInitial p i offset) (chainInitialStart p i offset)
    allowed (fun _ => trivial) (transcriptChunks_supported (chainBlocks_supported wires _ _ _))
    (within _ _ (by omega) (by omega))
  have chainFinalS := Sponge.supported (chainFinal p i offset) (chainFinalStart p i offset)
    allowed (fun _ => trivial) (transcriptChunks_supported (chainBlocks_supported wires _ _ _))
    (within _ _ (by omega) (by omega))
  have etaS := Sponge.supported (eta p i offset) (etaStart p i offset) allowed (fun _ => trivial)
    (transcriptChunks_supported (etaBlocks_supported wires)) (within _ _ (by omega) (by omega))
  have noChunks : ∀ chunk ∈ ([[]] : List (List Circuit.Expr)), ∀ e ∈ chunk, e.VarsSatisfy allowed := by
    simp
  have etaStateS : ∀ lane, (etaState p i offset lane).VarsSatisfy allowed := fun lane => by
    rw [etaState_output]
    exact etaS.2 lane
  have squeeze1S := Sponge.supported (squeeze1 p i offset) (squeeze1Start p i offset) allowed
    etaStateS noChunks (within _ _ (by omega) (by omega))
  have squeeze2S := Sponge.supported (squeeze2 p i offset) (squeeze2Start p i offset) allowed
    squeeze1S.2 noChunks (within _ _ (by omega) (by omega))
  have squeeze3S := Sponge.supported (squeeze3 p i offset) (squeeze3Start p i offset) allowed
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
  · simp only [assertions, List.mem_append, List.mem_map, List.mem_finRange, true_and,
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

variable (two : p.bOps = 2)

/-- The memory-application circuit. -/
def circuit (i : AppInterface p) : FormalCircuit where
  main := main p i
  assumptions := fun offset _ => Application.InputsBelow i offset
  spec := fun offset env => Application.Holds step i offset env ∧ Application.Valid (valid p two) i offset env
  soundness := fun env offset inputs rows => soundness p i offset env two inputs rows
  completeness := fun env offset inputs specification =>
    completeness p i offset two env inputs specification

/-- The first memory application as a closed Stage 1 program. -/
def program : Application.Program where
  witnessWordCount := Words.count p
  step := step
  valid := valid p two
  circuit := circuit p two
  spec_iff := fun _ _ _ => Iff.rfl
  assumptions_of_inputsBelow := fun _ _ _ inputs => inputs
  constraintsSupported := fun i offset _ allowed _ inputs localSupport =>
    supported p i offset allowed inputs localSupport

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
