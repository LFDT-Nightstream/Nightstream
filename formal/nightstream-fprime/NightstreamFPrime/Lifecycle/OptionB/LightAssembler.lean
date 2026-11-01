import NightstreamFPrime.Lifecycle.OptionB.LightStepSoundness

/-! The light step's source layout (assembler A1). Every shared word of
`LightStepLayout.Interface` is one variable in a source region that starts at `base`: the counters
and selector (5), the key digests (8), the start, current and next states (12), the application
witness, the prior, active, inner and next slots (7 × 2,055, each in the order of
`SlotExpr.words`), the slot and output digests (20), the fresh instance (1,458), the round
messages (504) and the Π_CCS output of both sources (1,080). The step's gadgets start at
`sharedEnd`. -/

namespace NightstreamFPrime.Lifecycle.OptionB.LightAssembler

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Two adjacent columns as a `K` value. -/
def kAt (column : Nat) : KExpr := ⟨.var column, .var (column + 1)⟩

/-- One slot's 2,055 words from column `start`, in the order of `SlotExpr.words`: the commitment
(1,188), `Eval_K` (108), `Eval_A` (432), the point (56), the public input (270) and the depth. -/
def slotAt (start : Nat) : SlotExpr logicalWidth publicFits where
  commitment := fun row lane => .var (start + row.val * ringDegree + lane.val)
  evaluation :=
    ⟨fun coefficient => kAt (start + 1188 + 2 * coefficient.val),
      fun matrix coefficient =>
        kAt (start + 1296 + 2 * (matrix.val * ringDegree + coefficient.val))⟩
  point := fun coordinate => kAt (start + 1728 + 2 * coordinate.val)
  publicInput := fun column => .var (start + 1784 + column.val)
  depth := .var (start + 2054)

def slotWordCount : Nat := 2055

/-- Where each group of shared words starts. -/
def keysStart (base : Nat) : Nat := base + 5
def statesStart (base : Nat) : Nat := base + 13
def witnessStart (base : Nat) : Nat := base + 25
def slotsStart (base witnessWordCount : Nat) : Nat := witnessStart base + witnessWordCount
def digestsStart (base witnessWordCount : Nat) : Nat :=
  slotsStart base witnessWordCount + 7 * slotWordCount
def freshStart (base witnessWordCount : Nat) : Nat := digestsStart base witnessWordCount + 20
def roundsStart (base witnessWordCount : Nat) : Nat := freshStart base witnessWordCount + 1458
def outputStart (base witnessWordCount : Nat) : Nat := roundsStart base witnessWordCount + 504

/-- The first column after the shared words, where the step's gadgets start. -/
def sharedEnd (base witnessWordCount : Nat) : Nat := outputStart base witnessWordCount + 1080

theorem sharedEnd_eq (base witnessWordCount : Nat) :
    sharedEnd base witnessWordCount = base + witnessWordCount + 17472 := by
  simp only [sharedEnd, outputStart, roundsStart, freshStart, digestsStart, slotsStart,
    witnessStart, slotWordCount]
  omega

/-- The step's shared words as source columns from `base`. -/
def interface (base witnessWordCount : Nat) :
    LightStepLayout.Interface logicalWidth publicFits witnessWordCount where
  iteration := .var base
  inverse := .var (base + 1)
  recursive := .var (base + 2)
  priorPc := .var (base + 3)
  selector := .var (base + 4)
  verifierKeys := fun key lane => .var (keysStart base + 4 * key.val + lane.val)
  initialState := fun lane => .var (statesStart base + lane.val)
  currentState := fun lane => .var (statesStart base + 4 + lane.val)
  nextState := fun lane => .var (statesStart base + 8 + lane.val)
  witness := fun index => .var (witnessStart base + index.val)
  prior := fun slot => slotAt (slotsStart base witnessWordCount + slot.val * slotWordCount)
  active := slotAt (slotsStart base witnessWordCount + 2 * slotWordCount)
  inner := fun slot => slotAt (slotsStart base witnessWordCount + (3 + slot.val) * slotWordCount)
  next := fun slot => slotAt (slotsStart base witnessWordCount + (5 + slot.val) * slotWordCount)
  priorDigest := fun slot lane =>
    .var (digestsStart base witnessWordCount + 4 * slot.val + lane.val)
  nextDigest := fun slot lane =>
    .var (digestsStart base witnessWordCount + 8 + 4 * slot.val + lane.val)
  outputDigest := fun lane => .var (digestsStart base witnessWordCount + 16 + lane.val)
  fresh :=
    ⟨fun _ row lane => .var (freshStart base witnessWordCount + row.val * ringDegree + lane.val),
      fun _ column => .var (freshStart base witnessWordCount + 1188 + column.val)⟩
  round := fun roundIndex => ⟨fun coefficient =>
    kAt (roundsStart base witnessWordCount + 2 * (roundIndex.val * 9 + coefficient.val))⟩
  output := ⟨fun source coefficient =>
      kAt (outputStart base witnessWordCount + 2 * (source.val * 270 + coefficient.val)),
    fun source matrix coefficient =>
      kAt (outputStart base witnessWordCount +
        2 * (source.val * 270 + 54 + matrix.val * 54 + coefficient.val))⟩

end NightstreamFPrime.Lifecycle.OptionB.LightAssembler
