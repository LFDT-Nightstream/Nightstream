import NightstreamFPrime.Lifecycle.OptionB.LightRlcLayout
import NightstreamFPrime.Lifecycle.OptionB.DigestSelect
import NightstreamFPrime.Lifecycle.OptionB.LightStep

/-! The light step circuit's layout (CW-2.5d, d5a). The step's shared words (the counters and
selector, the key digests, the states and the application witness, the prior and active slots,
the slot digests, the fresh instance and the light proof) are expressions that the assembler
allocates. From a parent offset the layout places today's application gadget, the depth guard on
the active slot's depth word, the prior state hash with its two slot hashes, the next state hash,
the light Π_CCS and Π_RLC layouts on the active slot, the new slot's hash and the digest select.
`soundness` gives the gadget specs of `LightStep.SpecHolds` and the digest-select rows from the
step's rows. -/

namespace NightstreamFPrime.Lifecycle.OptionB.LightStepLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords

/-- The step's shared words. -/
structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) (witnessWordCount : Nat) where
  iteration : Expr
  inverse : Expr
  recursive : Expr
  priorPc : Expr
  selector : Expr
  verifierKeys : Fin 2 → Fin 4 → Expr
  initialState : Fin 4 → Expr
  currentState : Fin 4 → Expr
  nextState : Fin 4 → Expr
  witness : Fin witnessWordCount → Expr
  prior : Fin 2 → SlotExpr logicalWidth publicFits
  active : SlotExpr logicalWidth publicFits
  priorDigest : Fin 2 → Fin 4 → Expr
  nextDigest : Fin 2 → Fin 4 → Expr
  outputDigest : Fin 4 → Expr
  newDigest : Fin 4 → Expr
  innerDigest : Fin 2 → Fin 4 → Expr
  fresh : StatementAbsorption.FreshExpr logicalWidth publicFits
  round : Fin productionShape.cubeVariables → RoundTranscript.Message 8
  output : OutputBinding.ShapeOutputExpr LightShape

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- A slot hash gadget over a slot's words, with its expected digest words. -/
def slotHashInterface (words : List Expr) (digest : Fin 4 → Expr) : OutputHash.Interface where
  preimage := fun _ => slotDomainChunk.map Expr.const ++ words
  digest := fun _ => digest

/-- A slot hash gadget over a slot's words gives the digest of the slot's value. -/
theorem slotHash_digest (slot : SlotExpr logicalWidth publicFits) (digest : Fin 4 → Expr)
    (offset : Nat) (env : Env)
    (spec : OutputHash.SpecHolds (slotHashInterface slot.words digest) offset env) :
    digestValue digest env = slotDigest (slot.value env) :=
  gadgetDigest_eq spec (by
    unfold SlotRepresents slotHashInterface
    rw [← slot.words_eval env]
    simp [Hash.evalList, List.map_map, Function.comp_def])

variable (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
  (interface : Interface logicalWidth publicFits program.witnessWordCount) (offset : Nat)

/-- The words of a state preimage (`statePreimage`) with the given iteration, current state, pc
and slot digests. -/
def statePreimageExpr (iteration : Expr) (current : Fin 4 → Expr) (pc : Expr)
    (digests : Fin 2 → Fin 4 → Expr) : List Expr :=
  stateDomainChunk.map Expr.const ++ List.ofFn (interface.verifierKeys 0) ++
    List.ofFn (interface.verifierKeys 1) ++ [iteration] ++ List.ofFn interface.initialState ++
    List.ofFn current ++ [pc] ++ List.ofFn (digests 0) ++ List.ofFn (digests 1)

def applicationInterface : Stage1.Application.Interface program.witnessWordCount where
  input := fun _ => interface.currentState
  witness := fun _ => interface.witness
  output := fun _ => interface.nextState

def applicationCircuit : FormalCircuit :=
  program.circuit (applicationInterface program interface)

def rangeOffset : Nat :=
  Formal.nextOffset (applicationCircuit program interface) offset

def rangeInterface : CanonicalU64.Interface where
  source := fun _ => interface.active.depth

def rangeCircuit : FormalCircuit :=
  CanonicalU64.circuit (rangeInterface program interface)

def highBitsOffset : Nat :=
  Formal.nextOffset (rangeCircuit program interface) (rangeOffset program interface offset)

def highBitsCircuit : FormalCircuit :=
  DepthGuard.highBitsCircuit (rangeOffset program interface offset)

def priorSlotInterface (slot : Fin 2) : OutputHash.Interface :=
  slotHashInterface (interface.prior slot).words (interface.priorDigest slot)

def priorSlotCircuit (slot : Fin 2) : FormalCircuit :=
  OutputHash.circuit (priorSlotInterface program interface slot)

def priorSlotZeroOffset : Nat :=
  Formal.nextOffset (highBitsCircuit program interface offset)
    (highBitsOffset program interface offset)

def priorSlotOneOffset : Nat :=
  Formal.nextOffset (priorSlotCircuit program interface 0)
    (priorSlotZeroOffset program interface offset)

def priorSlotOffset (slot : Fin 2) : Nat :=
  if slot.val = 0 then priorSlotZeroOffset program interface offset
  else priorSlotOneOffset program interface offset

def priorStateOffset : Nat :=
  Formal.nextOffset (priorSlotCircuit program interface 1)
    (priorSlotOneOffset program interface offset)

/-- The prior state hash: its preimage is the prior state's words and its public input is the
fresh instance's public input. -/
def priorStateInterface : PriorStateHash.Interface where
  preimage := fun _ => statePreimageExpr program interface interface.iteration
    interface.currentState interface.priorPc interface.priorDigest
  publicInput := fun _ column => interface.fresh.publicInput Formal.freshIndex column

def priorStateCircuit : FormalCircuit :=
  PriorStateHash.circuit (priorStateInterface program interface)

def nextStateOffset : Nat :=
  Formal.nextOffset (priorStateCircuit program interface)
    (priorStateOffset program interface offset)

/-- The next state hash (HyperNova Construction 2, step 5): iteration plus one, the application's
output state, and the one-based pc of this circuit. -/
def nextStateInterface : OutputHash.Interface where
  preimage := fun _ => statePreimageExpr program interface (interface.iteration + 1)
    interface.nextState (Expr.const (natWord (circuit.val + 1))) interface.nextDigest
  digest := fun _ => interface.outputDigest

def nextStateCircuit : FormalCircuit :=
  OutputHash.circuit (nextStateInterface program circuit interface)

def piccsOffset : Nat :=
  Formal.nextOffset (nextStateCircuit program circuit interface)
    (nextStateOffset program interface offset)

def piccsInterface : LightPiCCSLayout.Interface logicalWidth 8 publicFits where
  runningPoint := fun _ => interface.active.point
  runningEvaluation := fun _ => interface.active.evaluation
  fresh := fun _ => interface.fresh
  round := fun _ => interface.round
  output := fun _ => interface.output

def piccsLeaves : LightPiCCSPhase.Leaves logicalWidth 8 publicFits :=
  LightPiCCSLayout.leaves relation (piccsInterface program interface)
    (piccsOffset program circuit interface offset)

def rlcOffset : Nat :=
  LightPiCCSLayout.finalOffset relation (piccsInterface program interface)
    (piccsOffset program circuit interface offset)

def rlcInterface : LightRlcLayout.Interface logicalWidth publicFits where
  runningCommitment := fun _ => interface.active.commitment
  runningPublicInput := fun _ => interface.active.publicInput
  fresh := fun _ => interface.fresh

def rlcLeaves : LightFold.RlcLeaves logicalWidth publicFits :=
  LightRlcLayout.leaves (piccsLeaves program relation circuit interface offset)
    (rlcInterface program interface) (rlcOffset program relation circuit interface offset)

def newSlotOffset : Nat :=
  LightRlcLayout.finalOffset (piccsLeaves program relation circuit interface offset)
    (rlcInterface program interface) (rlcOffset program relation circuit interface offset)

/-- The new slot: the Π_RLC output claim at the SumCheck point, and the active depth plus one. -/
def newSlot : SlotExpr logicalWidth publicFits where
  commitment := LightCommitment.output
    (rlcLeaves program relation circuit interface offset).commitmentOffset
  evaluation :=
    ⟨fun coefficient => LightEvaluation.output
        (rlcLeaves program relation circuit interface offset).evaluationOffset
        ⟨0, by decide⟩ coefficient,
      fun matrix coefficient => LightEvaluation.output
        (rlcLeaves program relation circuit interface offset).evaluationOffset
        ⟨matrix.val + 1, Nat.succ_lt_succ matrix.isLt⟩ coefficient⟩
  point := LightPiCCSLayout.roundPoint (piccsInterface program interface)
    (piccsOffset program circuit interface offset)
  publicInput := LightPublicInput.output (publicFits := publicFits)
    (rlcLeaves program relation circuit interface offset).publicInputOffset
  depth := interface.active.depth + 1

/-- The new slot's hash: its preimage is the new slot's words and its digest the new digest. -/
def newSlotInterface : OutputHash.Interface :=
  slotHashInterface (newSlot program relation circuit interface offset).words interface.newDigest

def newSlotCircuit : FormalCircuit :=
  OutputHash.circuit (newSlotInterface program relation circuit interface offset)

def selectOffset : Nat :=
  Formal.nextOffset (newSlotCircuit program relation circuit interface offset)
    (newSlotOffset program relation circuit interface offset)

def selectInterface : DigestSelect.Interface logicalWidth publicFits where
  iteration := fun _ => interface.iteration
  inverse := fun _ => interface.inverse
  recursive := fun _ => interface.recursive
  priorPc := fun _ => interface.priorPc
  selector := fun _ => interface.selector
  initialState := fun _ => interface.initialState
  currentState := fun _ => interface.currentState
  prior := fun _ => interface.prior
  active := fun _ => interface.active
  priorDigest := fun _ => interface.priorDigest
  newDigest := fun _ => interface.newDigest
  innerDigest := fun _ => interface.innerDigest
  nextDigest := fun _ => interface.nextDigest

def selectCircuit : FormalCircuit :=
  DigestSelect.circuit (selectInterface program interface)

/-- The leaves of `LightStep` that this layout places. -/
def leaves : LightStep.Leaves logicalWidth publicFits program.witnessWordCount where
  application := applicationInterface program interface
  applicationOffset := offset
  depth := rangeInterface program interface
  depthOffset := rangeOffset program interface offset
  prior := priorStateInterface program interface
  priorOffset := priorStateOffset program interface offset
  priorSlots := priorSlotInterface program interface
  priorSlotOffsets := priorSlotOffset program interface offset
  next := nextStateInterface program circuit interface
  nextOffset := nextStateOffset program interface offset
  nextDigest := interface.nextDigest
  newSlot := newSlotInterface program relation circuit interface offset
  newSlotOffset := newSlotOffset program relation circuit interface offset
  piccs := piccsLeaves program relation circuit interface offset
  rlc := rlcLeaves program relation circuit interface offset

def gadgetOps : List Op :=
  [Formal.childOp "light.application" (applicationCircuit program interface) offset,
    Formal.childOp "light.depth_range" (rangeCircuit program interface)
      (rangeOffset program interface offset),
    Formal.childOp "light.depth_high_bits" (highBitsCircuit program interface offset)
      (highBitsOffset program interface offset),
    Formal.childOp "light.prior_slot_0" (priorSlotCircuit program interface 0)
      (priorSlotZeroOffset program interface offset),
    Formal.childOp "light.prior_slot_1" (priorSlotCircuit program interface 1)
      (priorSlotOneOffset program interface offset),
    Formal.childOp "light.prior_state" (priorStateCircuit program interface)
      (priorStateOffset program interface offset),
    Formal.childOp "light.next_state" (nextStateCircuit program circuit interface)
      (nextStateOffset program interface offset)]

def opsAt : List Op :=
  gadgetOps program circuit interface offset ++
    LightPiCCSLayout.opsAt relation (piccsInterface program interface)
      (piccsOffset program circuit interface offset) ++
    LightRlcLayout.opsAt (piccsLeaves program relation circuit interface offset)
      (rlcInterface program interface) (rlcOffset program relation circuit interface offset) ++
    [Formal.childOp "light.new_slot" (newSlotCircuit program relation circuit interface offset)
        (newSlotOffset program relation circuit interface offset),
      Formal.childOp "light.digest_select" (selectCircuit program interface)
        (selectOffset program relation circuit interface offset)]

def main : Circuit Unit :=
  fun offset => ((), selectOffset program relation circuit interface offset,
    opsAt program relation circuit interface offset)

/-- Each child's input scoping, which its rows need to give its spec. -/
structure Assumptions (env : Env) : Prop where
  application : (applicationCircuit program interface).assumptions offset env
  range : (rangeCircuit program interface).assumptions (rangeOffset program interface offset) env
  priorSlotZero : (priorSlotCircuit program interface 0).assumptions
    (priorSlotZeroOffset program interface offset) env
  priorSlotOne : (priorSlotCircuit program interface 1).assumptions
    (priorSlotOneOffset program interface offset) env
  priorState : (priorStateCircuit program interface).assumptions
    (priorStateOffset program interface offset) env
  nextState : (nextStateCircuit program circuit interface).assumptions
    (nextStateOffset program interface offset) env
  piccs : LightPiCCSLayout.Assumptions relation (piccsInterface program interface)
    (piccsOffset program circuit interface offset) env
  rlc : LightRlcLayout.Assumptions (piccsLeaves program relation circuit interface offset)
    (rlcInterface program interface) (rlcOffset program relation circuit interface offset) env
  newSlot : (newSlotCircuit program relation circuit interface offset).assumptions
    (newSlotOffset program relation circuit interface offset) env

private theorem childSpec_of_rows (name : String) (child : FormalCircuit)
    (childOffset : Nat) (env : Env) (ops : List Op) (rows : holds env ops)
    (member : Formal.childOp name child childOffset ∈ ops)
    (assumptions : child.assumptions childOffset env) :
    child.spec childOffset env := by
  have callHolds := rows (Formal.childOp name child childOffset) member
  change child.assumptions childOffset env → child.spec childOffset env at callHolds
  exact callHolds assumptions

/-- The step's rows give every gadget spec of `LightStep.SpecHolds`, for the relation of any
circuit key, and the digest-select rows. The application program computes `step`, and its validity
predicate implies the dispatch `control zi w = circuit`. -/
theorem soundness (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2) (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness →
      control state witness = circuit)
    (env : Env) (assumptions : Assumptions program relation circuit interface offset env)
    (rows : holds env (Circuit.ops (main program relation circuit interface) offset))
    (target : ProductionKey.LogicalRelation logicalWidth publicFits) :
    LightStep.SpecHolds target step control circuit
        (leaves program relation circuit interface offset) env ∧
      ConstraintsHold env (DigestSelect.rows (selectInterface program interface)
        (selectOffset program relation circuit interface offset)) := by
  change holds env (opsAt program relation circuit interface offset) at rows
  have member := fun op (inside : op ∈ gadgetOps program circuit interface offset) =>
    (show op ∈ opsAt program relation circuit interface offset by
      simp only [opsAt, List.mem_append]
      exact Or.inl (Or.inl (Or.inl inside)))
  have applicationSpec := childSpec_of_rows "light.application"
    (applicationCircuit program interface) offset env _ rows
    (member _ (by simp [gadgetOps])) assumptions.application
  obtain ⟨holdsStep, validStep⟩ :=
    (program.spec_iff (applicationInterface program interface) offset env).mp applicationSpec
  have rangeSpec := childSpec_of_rows "light.depth_range" (rangeCircuit program interface)
    (rangeOffset program interface offset) env _ rows (member _ (by simp [gadgetOps]))
    assumptions.range
  have highBitsSpec := childSpec_of_rows "light.depth_high_bits"
    (highBitsCircuit program interface offset) (highBitsOffset program interface offset) env _
    rows (member _ (by simp [gadgetOps])) trivial
  have priorZeroSpec := childSpec_of_rows "light.prior_slot_0"
    (priorSlotCircuit program interface 0) (priorSlotZeroOffset program interface offset) env _
    rows (member _ (by simp [gadgetOps])) assumptions.priorSlotZero
  have priorOneSpec := childSpec_of_rows "light.prior_slot_1"
    (priorSlotCircuit program interface 1) (priorSlotOneOffset program interface offset) env _
    rows (member _ (by simp [gadgetOps])) assumptions.priorSlotOne
  have priorStateSpec := childSpec_of_rows "light.prior_state"
    (priorStateCircuit program interface) (priorStateOffset program interface offset) env _
    rows (member _ (by simp [gadgetOps])) assumptions.priorState
  have nextStateSpec := childSpec_of_rows "light.next_state"
    (nextStateCircuit program circuit interface) (nextStateOffset program interface offset) env
    _ rows (member _ (by simp [gadgetOps])) assumptions.nextState
  have piccsRows : holds env (Circuit.ops (LightPiCCSLayout.main relation
      (piccsInterface program interface)) (piccsOffset program circuit interface offset)) :=
    fun op inside => rows op (by
      simp only [opsAt, List.mem_append]
      exact Or.inl (Or.inl (Or.inr inside)))
  have rlcRows : holds env (Circuit.ops (LightRlcLayout.main
      (piccsLeaves program relation circuit interface offset) (rlcInterface program interface))
      (rlcOffset program relation circuit interface offset)) :=
    fun op inside => rows op (by
      simp only [opsAt, List.mem_append]
      exact Or.inl (Or.inr inside))
  have newSlotSpec := childSpec_of_rows "light.new_slot"
    (newSlotCircuit program relation circuit interface offset)
    (newSlotOffset program relation circuit interface offset) env _ rows (by simp [opsAt])
    assumptions.newSlot
  have selectSpec := childSpec_of_rows "light.digest_select" (selectCircuit program interface)
    (selectOffset program relation circuit interface offset) env _ rows (by simp [opsAt]) trivial
  refine ⟨{
    application := stepEq ▸ holdsStep
    dispatch := validControl _ _ validStep
    depth := rangeSpec
    depthHigh := highBitsSpec
    prior := priorStateSpec
    priorSlots := fun slot => ?_
    next := nextStateSpec
    newSlot := newSlotSpec
    piccs := LightPiCCSLayout.soundness relation (piccsOffset program circuit interface offset)
      (piccsInterface program interface) env assumptions.piccs piccsRows target
    rlc := LightRlcLayout.soundness (piccsLeaves program relation circuit interface offset)
      (rlcInterface program interface) (rlcOffset program relation circuit interface offset) env
      assumptions.rlc rlcRows }, selectSpec⟩
  match slot with
  | 0 => exact priorZeroSpec
  | 1 => exact priorOneSpec

end NightstreamFPrime.Lifecycle.OptionB.LightStepLayout
