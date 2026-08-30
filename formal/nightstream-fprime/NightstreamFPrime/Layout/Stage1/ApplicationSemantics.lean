import NightstreamFPrime.Layout.Stage1.ApplicationInputs

/-!
Owns the typed value bridge for the zero-copy Stage 1 application columns.

The proofs select only the fixed four-word `current` slice from each already
represented pilot preimage. They do not evaluate the complete state hash or
reconstruct the running instances.
-/

namespace NightstreamFPrime.Layout.Stage1.ApplicationInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem drop_take_ofFn {Alpha : Type} {count start width : Nat}
    (values : Fin count → Alpha) (fits : start + width ≤ count) :
    ((List.ofFn values).drop start).take width =
      List.ofFn fun index : Fin width =>
        values ⟨start + index.val, by omega⟩ := by
  apply List.ext_getElem <;> simp
  omega

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem serializePreimage_current_slice
    (preimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage preimage) :
    ((serializePreimage (publicFits := publicFits) preimage).drop
      currentWordStart).take Lifecycle.Stage1.Application.stateWordCount =
        preimage.current := by
  rcases fixed with ⟨keyLength, z0Length, currentLength⟩
  let headWords : List F := stateDomainTag ++
    block (preimage.verifierKeys functionIndex) ++
    [natWord preimage.iteration] ++ block preimage.z0 ++
    [natWord preimage.current.length]
  let tailWords : List F := serializeRunning (publicFits := publicFits)
    (preimage.running functionIndex) ++ [natWord preimage.pc]
  have decomposition :
      serializePreimage (publicFits := publicFits) preimage =
        headWords ++ preimage.current ++ tailWords := by
    simp [serializePreimage, block, headWords, tailWords, List.append_assoc]
  have headLength : headWords.length = currentWordStart := by
    simp [headWords, block, stateDomainTag_length, keyLength, z0Length,
      currentWordStart]
    norm_num [PilotProduction.digestWords, PilotValues.digestWords]
  rw [decomposition, ← headLength]
  have currentLengthFour : preimage.current.length = 4 := by
    simpa [PilotProduction.digestWords, PilotValues.digestWords] using
      currentLength
  rw [show Lifecycle.Stage1.Application.stateWordCount =
      preimage.current.length by
        rw [currentLengthFour]
        rfl]
  simp

private theorem inputState_eq_prior_slice
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.Stage1.Application.inputState (interface program)
        (localStart program) env =
      List.take Lifecycle.Stage1.Application.stateWordCount
        (List.drop currentWordStart
          (Hash.evalList (Spartan.pullback env)
            (PilotProduction.priorPreimage 0))) := by
  simp only [Lifecycle.Stage1.Application.inputState, interface,
    Hash.evalList, PilotProduction.priorPreimage,
    PilotProduction.variableExprs, List.map_ofFn, Expr.eval_var]
  let values : Fin PilotProduction.stateHashWords → F :=
    Expr.eval (Spartan.pullback env) ∘
      fun index => Expr.var (PilotProduction.priorPreimageStart + index.val)
  have fits : currentWordStart +
      Lifecycle.Stage1.Application.stateWordCount ≤
        PilotProduction.stateHashWords := by
    norm_num [currentWordStart,
      Lifecycle.Stage1.Application.stateWordCount,
      PilotProduction.stateHashWords_eq]
  calc
    List.ofFn (fun index : Fin 4 => env (inputColumn index)) =
        List.ofFn (fun index : Fin 4 =>
          values ⟨currentWordStart + index.val, by
            have indexBound := index.isLt
            norm_num [currentWordStart,
              Lifecycle.Stage1.Application.stateWordCount,
              PilotProduction.stateHashWords_eq] at indexBound ⊢
            omega⟩) := by
      apply congrArg (@List.ofFn F 4)
      funext index
      simp only [values, Spartan.pullback, Function.comp_apply,
        Expr.eval_var, inputColumn, inputSourceColumn, currentWordStart]
      apply congrArg env
      apply congrArg Spartan.sourceToSpartan
      omega
    _ = List.take Lifecycle.Stage1.Application.stateWordCount
          (List.drop currentWordStart (List.ofFn values)) :=
      (drop_take_ofFn values fits).symm

private theorem outputState_eq_output_slice
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    Lifecycle.Stage1.Application.outputState (interface program)
        (localStart program) env =
      List.take Lifecycle.Stage1.Application.stateWordCount
        (List.drop currentWordStart
          (Hash.evalList (Spartan.pullback env)
            (PilotProduction.outputPreimage 0))) := by
  simp only [Lifecycle.Stage1.Application.outputState, interface,
    Hash.evalList, PilotProduction.outputPreimage,
    PilotProduction.variableExprs, List.map_ofFn, Expr.eval_var]
  let values : Fin PilotProduction.stateHashWords → F :=
    Expr.eval (Spartan.pullback env) ∘
      fun index => Expr.var (PilotProduction.outputPreimageStart + index.val)
  have fits : currentWordStart +
      Lifecycle.Stage1.Application.stateWordCount ≤
        PilotProduction.stateHashWords := by
    norm_num [currentWordStart,
      Lifecycle.Stage1.Application.stateWordCount,
      PilotProduction.stateHashWords_eq]
  calc
    List.ofFn (fun index : Fin 4 => env (outputColumn index)) =
        List.ofFn (fun index : Fin 4 =>
          values ⟨currentWordStart + index.val, by
            have indexBound := index.isLt
            norm_num [currentWordStart,
              Lifecycle.Stage1.Application.stateWordCount,
              PilotProduction.stateHashWords_eq] at indexBound ⊢
            omega⟩) := by
      apply congrArg (@List.ofFn F 4)
      funext index
      simp only [values, Spartan.pullback, Function.comp_apply,
        Expr.eval_var, outputColumn, outputSourceColumn, currentWordStart]
      apply congrArg env
      apply congrArg Spartan.sourceToSpartan
      omega
    _ = List.take Lifecycle.Stage1.Application.stateWordCount
          (List.drop currentWordStart (List.ofFn values)) :=
      (drop_take_ofFn values fits).symm

/-- The application input wires are exactly the typed prior state `z_i`. -/
theorem inputState_eq_current
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (preimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage preimage)
    (represented : PriorStateHash.RepresentsPreimage
      PilotProduction.priorInterface PilotProduction.witnessOffset
      (Spartan.pullback env) preimage) :
    Lifecycle.Stage1.Application.inputState (interface program)
      (localStart program) env = preimage.current := by
  rw [inputState_eq_prior_slice]
  unfold PriorStateHash.RepresentsPreimage at represented
  have representedDirect :
      Hash.evalList (Spartan.pullback env) (PilotProduction.priorPreimage 0) =
        serializePreimage (publicFits := publicFits) preimage := by
    simpa [PilotProduction.priorInterface,
      PilotProduction.makePriorInterface, PilotProduction.priorPreimage] using
        represented
  rw [representedDirect]
  exact serializePreimage_current_slice preimage fixed

/-- The application output wires are exactly the typed next state
`z_{i+1}`. -/
theorem outputState_eq_current
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (preimage : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage preimage)
    (represented : OutputHash.RepresentsPreimage
      PilotProduction.outputInterface PilotProduction.lifecycleOutputOffset
      (Spartan.pullback env) preimage) :
    Lifecycle.Stage1.Application.outputState (interface program)
      (localStart program) env = preimage.current := by
  rw [outputState_eq_output_slice]
  unfold OutputHash.RepresentsPreimage at represented
  have representedDirect :
      Hash.evalList (Spartan.pullback env) (PilotProduction.outputPreimage 0) =
        serializePreimage (publicFits := publicFits) preimage := by
    simpa [PilotProduction.outputInterface,
      PilotProduction.makeOutputInterface,
      PilotProduction.outputPreimage] using represented
  rw [representedDirect]
  exact serializePreimage_current_slice preimage fixed

end NightstreamFPrime.Layout.Stage1.ApplicationInputs
