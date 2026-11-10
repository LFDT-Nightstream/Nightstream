import NightstreamFPrime.Lifecycle.BoundedDepth.RefreshStepSoundness

/-! The finish circuit (CW-6.3): the last fold of a compressed bounded-depth proof. It is the refresh
step circuit (`RefreshStepLayout`) without the application, the next state's hash, the new slot's
hash and the digest select: the prior slots' and prior state's hashes, today's Π_DEC of the
selected slot, the refresh Π_CCS, today's Π_RLC and the slot select, which writes both next slots'
words for the final program. A refresh may run at any depth, so the last
fold of every active proof is a refresh. With words that carry the step's input, the rows give
`FinishHolds`: the fresh public input is the hash of the prior state with both slots, and the
refresh fold of the fresh instance into the selected slot gives the selected next slot at depth 0
while the other slot keeps its claim (`stepHolds_iff`'s recursive branch for a refresh). -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.FinishStep

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
open NightstreamFPrime.Lifecycle.BoundedDepth.RefreshStepLayout

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
  (interface : RefreshStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
  (offset : Nat)

/-- The slot select over the refresh step's inner and next slots, with the new slot. -/
def slotSelectInterface : SlotSelect.Interface logicalWidth publicFits where
  iteration := fun _ => interface.iteration
  inverse := fun _ => interface.inverse
  recursive := fun _ => interface.recursive
  priorPc := fun _ => interface.priorPc
  selector := fun _ => interface.selector
  initialState := fun _ => interface.initialState
  currentState := fun _ => interface.currentState
  prior := fun _ => interface.prior
  active := fun _ => interface.active
  new := fun _ => newSlot program relation circuit interface offset
  inner := fun _ => interface.inner
  next := fun _ => interface.next

def slotSelectCircuit : FormalCircuit :=
  SlotSelect.circuit (slotSelectInterface program relation circuit interface offset)

/-- The slot select allocates no variables; it follows Π_RLC. -/
def slotSelectOffset : Nat :=
  newSlotOffset program relation circuit interface offset

/-- The prior hashes, at the refresh step's offsets. -/
def hashOps : List Op :=
  [Formal.childOp "refresh.prior_slot_0" (priorSlotCircuit program interface 0)
      (priorSlotZeroOffset program interface offset),
    Formal.childOp "refresh.prior_slot_1" (priorSlotCircuit program interface 1)
      (priorSlotOneOffset program interface offset),
    Formal.childOp "refresh.prior_state" (priorStateCircuit program interface)
      (priorStateOffset program interface offset)]

def opsAt : List Op :=
  hashOps program interface offset ++
    Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation (decInterface program interface))
      (decOffset program circuit interface offset) ++
    Circuit.ops (RefreshPiCCSLayout.main relation (piccsInterface program interface))
      (piccsOffset program circuit interface offset) ++
    Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation
        (rlcInterface program relation circuit interface offset))
      (rlcOffset program relation circuit interface offset) ++
    [Formal.childOp "refresh.slot_select"
      (slotSelectCircuit program relation circuit interface offset)
      (slotSelectOffset program relation circuit interface offset)]

def main : Circuit Unit :=
  fun offset => ((), slotSelectOffset program relation circuit interface offset,
    opsAt program relation circuit interface offset)

/-- What the finish rows give for an active proof: the recursive branch of `stepHolds_iff` for a
refresh, without the application and the next state. -/
def FinishHolds (cap : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits) :
    Prop :=
  ∃ priorPcValid : Paper.InRange 2 input.priorPc,
    let selected := Paper.selectedIndex priorPcValid
    let key := circuitKeys selected
    input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState
        (Paper.priorHashPreimage ((setup cap ajtai circuitKeys).forCircuit circuit) input))) ∧
    BoundedDepthFolds.folds .refresh (keys key.relation ajtai)
        (toClaim (key.system ajtai) (input.running selected).1) input.fresh input.nifsProof =
      some (toClaim (key.system ajtai) (output.runningNext selected).1) ∧
    (output.runningNext selected).2 = 0 ∧
    ∀ slot, slot ≠ selected → output.runningNext slot = input.running slot

private theorem childSpec_of_rows (name : String) (child : FormalCircuit)
    (childOffset : Nat) (env : Env) (ops : List Op) (rows : holds env ops)
    (member : Formal.childOp name child childOffset ∈ ops)
    (assumptions : child.assumptions childOffset env) :
    child.spec childOffset env := by
  have callHolds := rows (Formal.childOp name child childOffset) member
  change child.assumptions childOffset env → child.spec childOffset env at callHolds
  exact callHolds assumptions

private theorem slotPreimage_eval (words : List Expr) (env : Env) :
    Hash.evalList env (slotDomainChunk.map Expr.const ++ words) =
      slotDomainChunk ++ Hash.evalList env words := by
  simp [Hash.evalList, List.map_map, Function.comp_def]

/-- Soundness of the finish circuit: its rows, words that carry an active proof's input and next
slots that the next slots' words carry give `FinishHolds` at the production cap. The output's
other fields are not read. -/
theorem rows_implies_finish
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (env : Env)
    (assumptions : RefreshStepLayout.Assumptions program relation circuit interface offset env)
    (rows : holds env (Circuit.ops (main program relation circuit interface) offset))
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (represents : RefreshStepSoundness.StepRepresents program interface ajtai circuitKeys circuit
      input output proof env)
    (nextSlots : ∀ slot,
      Hash.evalList env (interface.next slot).words = serializeSlot (output.runningNext slot))
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (active : input.iteration ≠ 0) :
    FinishHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys circuit input output := by
  change holds env (opsAt program relation circuit interface offset) at rows
  have member := fun op (inside : op ∈ hashOps program interface offset) =>
    (show op ∈ opsAt program relation circuit interface offset by
      simp only [opsAt, List.mem_append]
      exact Or.inl (Or.inl (Or.inl (Or.inl inside))))
  have priorZeroSpec := childSpec_of_rows "refresh.prior_slot_0"
    (priorSlotCircuit program interface 0) (priorSlotZeroOffset program interface offset) env _
    rows (member _ (by simp [hashOps])) assumptions.priorSlotZero
  have priorOneSpec := childSpec_of_rows "refresh.prior_slot_1"
    (priorSlotCircuit program interface 1) (priorSlotOneOffset program interface offset) env _
    rows (member _ (by simp [hashOps])) assumptions.priorSlotOne
  have priorStateSpec := childSpec_of_rows "refresh.prior_state"
    (priorStateCircuit program interface) (priorStateOffset program interface offset) env _
    rows (member _ (by simp [hashOps])) assumptions.priorState
  have decRows : holds env (Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation
      (decInterface program interface)) (decOffset program circuit interface offset)) :=
    fun op inside => rows op (by
      simp only [opsAt, List.mem_append]
      exact Or.inl (Or.inl (Or.inl (Or.inr inside))))
  have piccsRows : holds env (Circuit.ops (RefreshPiCCSLayout.main relation
      (piccsInterface program interface)) (piccsOffset program circuit interface offset)) :=
    fun op inside => rows op (by
      simp only [opsAt, List.mem_append]
      exact Or.inl (Or.inl (Or.inr inside)))
  have rlcRows : holds env (Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation
      (rlcInterface program relation circuit interface offset))
      (rlcOffset program relation circuit interface offset)) :=
    fun op inside => rows op (by
      simp only [opsAt, List.mem_append]
      exact Or.inl (Or.inr inside))
  have selectRows := childSpec_of_rows "refresh.slot_select"
    (slotSelectCircuit program relation circuit interface offset)
    (slotSelectOffset program relation circuit interface offset) env _ rows (by simp [opsAt])
    trivial
  obtain ⟨select, activeValue⟩ := SlotSelect.rows_implies_select
    (slotSelectInterface program relation circuit interface offset)
    (slotSelectOffset program relation circuit interface offset) env input output
    selectRows represents.iteration represents.iterationBound represents.priorPc
    represents.priorPcBound represents.initialState represents.currentState represents.priorSlots
    represents.priorDepth nextSlots nextDepth
  rw [show ((slotSelectInterface program relation circuit interface offset).new
      (slotSelectOffset program relation circuit interface offset)).value env =
      _ from RefreshStepSoundness.newSlot_value program interface relation circuit offset env]
    at select
  generalize SlotSelect.selected (slotSelectInterface program relation circuit interface offset)
    (slotSelectOffset program relation circuit interface offset) env = selected
    at select activeValue
  obtain ⟨pcEq, claimEq, depthEq, others⟩ := select.recursive active
  have priorPcValid : Paper.InRange 2 input.priorPc := by
    have bound := selected.isLt
    unfold Paper.InRange
    omega
  have selectedEq : Paper.selectedIndex priorPcValid = selected :=
    Fin.ext (by simp [Paper.selectedIndex, pcEq])
  refine ⟨priorPcValid, ?_⟩
  dsimp only
  rw [selectedEq]
  refine ⟨?_, ?_, depthEq, others⟩
  · exact builder_implies_prior (cap := Folding.Schedule.productionLightFoldCap) (ajtai := ajtai)
      (circuitKeys := circuitKeys) (step := fun state _ => state) (control := fun _ _ => circuit)
      (circuit := circuit) (input := input)
      (prior := (leaves program relation circuit interface offset).prior)
      (offset := (leaves program relation circuit interface offset).priorOffset)
      (digests := fun slot => gadgetDigest
        ((leaves program relation circuit interface offset).priorSlots slot)
        ((leaves program relation circuit interface offset).priorSlotOffsets slot) env)
      priorStateSpec
      (fun slot => gadgetDigest_eq
        (match slot with
          | 0 => priorZeroSpec
          | 1 => priorOneSpec)
        ((slotPreimage_eval _ env).trans
          (congrArg (slotDomainChunk ++ ·) (represents.priorSlots slot))))
      (by
        unfold StateRepresents
        change Hash.evalList env (statePreimageExpr program interface interface.iteration
          interface.currentState interface.priorPc interface.priorDigest) = _
        rw [RefreshStepSoundness.statePreimageExpr_eval, represents.iteration,
          represents.initialState, represents.currentState, represents.priorPc,
          represents.verifierKeys 0, represents.verifierKeys 1]
        rfl)
      (fun column => by
        rw [← represents.fresh]
        rfl)
  · rw [claimEq]
    exact represents.proof active ▸ RefreshFold.spec_implies_fold (circuitKeys selected).relation
      ajtai (input.running selected).1 input.fresh proof (leaves program relation circuit interface
        offset).dec (leaves program relation circuit interface offset).decOffset
      (leaves program relation circuit interface offset).piccs
      (leaves program relation circuit interface offset).rlc
      (leaves program relation circuit interface offset).rlcOffset env
      (decSpec_any (Lifecycle.PiDEC.v1_2.Formal.soundness relation
        (decInterface program interface) (decOffset program circuit interface offset) env
        assumptions.dec decRows))
      (RefreshStepSoundness.decWired program interface ajtai circuitKeys circuit input output proof
        env represents selected activeValue offset)
      (RefreshPiCCSLayout.soundness relation (piccsOffset program circuit interface offset)
        (piccsInterface program interface) env assumptions.piccs piccsRows
        (circuitKeys selected).relation)
      (RefreshPiCCSLayout.wired relation (piccsOffset program circuit interface offset)
        (circuitKeys selected).relation ajtai
        (RefreshFold.children (circuitKeys selected).relation ajtai (input.running selected).1
          proof.split) input.fresh proof.nifs (piccsInterface program interface) env
        (RefreshStepSoundness.piccsRepresents program interface ajtai circuitKeys circuit input
          output proof env represents selected activeValue (circuitKeys selected).relation
          (piccsOffset program circuit interface offset)))
      (rlcSpec_any (Lifecycle.PiRLC.v1_2.Formal.soundness relation
        (rlcInterface program relation circuit interface offset)
        (rlcOffset program relation circuit interface offset) env assumptions.rlc rlcRows))
      (RefreshStepSoundness.rlcWired program interface ajtai circuitKeys circuit input output proof
        env represents relation offset)

end NightstreamFPrime.Lifecycle.BoundedDepth.FinishStep
