import NightstreamFPrime.Layout.BoundedDepth.LightAssembler
import NightstreamFPrime.Lifecycle.BoundedDepth.LightStepClean

/-! The light step's shared words for the exporter's blocks (SPEC.md section 4.4). Each shared
word of the source layout is a column before `sharedEnd`, so it is affine, and it is live before
the step's gadgets when every column before `sharedEnd` is live (`inputsLive`). -/

namespace NightstreamFPrime.Layout.BoundedDepth.LightAssembler

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
open NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal (WordFacts)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-- An affine word below `bound` is live before `bound` when every column before `bound` is
live. -/
theorem wordFacts_of_below {live : Nat → Prop} {bound : Nat} {word : Expr} (affine : word.Affine)
    (below : word.VarsBelow bound) (shared : ∀ column, column < bound → live column) :
    WordFacts live bound word :=
  ⟨affine, ((Expr.varsSatisfy_lt_iff_varsBelow word bound).mpr below).mono _
    fun column inside => ⟨inside, shared column inside⟩⟩

/-- A slot's words from column `start` are columns. -/
theorem slotAt_affine (start : Nat) :
    ∀ word ∈ (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits) start).words,
      word.Affine :=
  LightStepLayout.slotWords_forall (P := Expr.Affine) _ (fun _ _ => .var _)
    (fun _ => ⟨.var _, .var _⟩) (fun _ _ => ⟨.var _, .var _⟩)
    (fun _ => ⟨.var _, .var _⟩) (fun _ => .var _) (.var _)

/-- A slot placed before `bound` has words that are live before `bound`. -/
theorem slotAt_facts {live : Nat → Prop} (start bound : Nat)
    (le : start + slotWordCount ≤ bound) (shared : ∀ column, column < bound → live column) :
    ∀ word ∈ (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits) start).words,
      WordFacts live bound word :=
  fun word member => wordFacts_of_below (slotAt_affine start word member)
    (slotAt_below start bound le word member) shared

/-- The shared words of the light step's source layout are affine and live before `sharedEnd`
when every column before `sharedEnd` is live. -/
theorem inputsLive (program : Stage1.Application.Program) (base : Nat) {live : Nat → Prop}
    (shared : ∀ column, column < sharedEnd base program.witnessWordCount → live column) :
    LightStepLayout.InputsLive program
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount)
      (sharedEnd base program.witnessWordCount) live := by
  have below := inputsBelow (logicalWidth := logicalWidth) (publicFits := publicFits) base
    program.witnessWordCount
  have endEq := sharedEnd_eq base program.witnessWordCount
  have slotsEq : slotsStart base program.witnessWordCount = base + 25 + program.witnessWordCount :=
    rfl
  exact {
    iteration := wordFacts_of_below (.var _) below.iteration shared
    inverse := wordFacts_of_below (.var _)
      (show base + 1 < sharedEnd base program.witnessWordCount by rw [endEq]; omega) shared
    recursive := wordFacts_of_below (.var _)
      (show base + 2 < sharedEnd base program.witnessWordCount by rw [endEq]; omega) shared
    priorPc := wordFacts_of_below (.var _) below.priorPc shared
    selector := wordFacts_of_below (.var _)
      (show base + 4 < sharedEnd base program.witnessWordCount by rw [endEq]; omega) shared
    verifierKeys := fun key lane => wordFacts_of_below (.var _) (below.verifierKeys key lane) shared
    initialState := fun lane => wordFacts_of_below (.var _) (below.initialState lane) shared
    currentState := fun lane => wordFacts_of_below (.var _) (below.currentState lane) shared
    nextState := fun lane => wordFacts_of_below (.var _) (below.nextState lane) shared
    witness := fun index => wordFacts_of_below (.var _) (below.witness index) shared
    active := slotAt_facts _ _ (by
      rw [endEq, slotsEq]
      simp only [slotWordCount]
      omega) shared
    priorDigest := fun slot lane => wordFacts_of_below (.var _) (below.priorDigest slot lane) shared
    nextDigest := fun slot lane => wordFacts_of_below (.var _) (below.nextDigest slot lane) shared
    outputDigest := fun lane => wordFacts_of_below (.var _) (below.outputDigest lane) shared
    newDigest := fun lane => wordFacts_of_below (.var _) (below.newDigest lane) shared
    innerDigest := fun slot lane => wordFacts_of_below (.var _) (below.innerDigest slot lane) shared
    activeDigest := fun lane => wordFacts_of_below (.var _) (below.activeDigest lane) shared
    freshCommitment := fun index row lane =>
      wordFacts_of_below (.var _) (below.freshCommitment index row lane) shared
    freshPublicInput := fun index column =>
      wordFacts_of_below (.var _) (below.freshPublicInput index column) shared
    round := fun roundIndex coefficient =>
      ⟨wordFacts_of_below (.var _) (below.round roundIndex coefficient).1 shared,
        wordFacts_of_below (.var _) (below.round roundIndex coefficient).2 shared⟩
    outputPad := fun source coefficient =>
      ⟨wordFacts_of_below (.var _) (below.outputPad source coefficient).1 shared,
        wordFacts_of_below (.var _) (below.outputPad source coefficient).2 shared⟩
    outputMatrix := fun source matrix coefficient =>
      ⟨wordFacts_of_below (.var _) (below.outputMatrix source matrix coefficient).1 shared,
        wordFacts_of_below (.var _) (below.outputMatrix source matrix coefficient).2 shared⟩ }

end NightstreamFPrime.Layout.BoundedDepth.LightAssembler
