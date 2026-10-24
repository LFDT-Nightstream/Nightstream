import NightstreamFPrime.Export.Stage1.WitnessProgram
import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics
import CompressionLean.Terminal.Data

/-!
Owns the output map of the terminal package: the package expressions of the
PiRLC parent (commitment, public input, point, `Eval_K`, `Eval_A`) and of the
sponge state after the PiRLC sampler. Layer 1 reads its claim from them.

Each list is the F′ interface expression moved to package columns, so its
value on a package assignment is the value that the soundness theorem names.
-/

namespace CompressionLean.Terminal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

/-- The PiRLC output interface of the Stage 1 layout. -/
abbrev outputInterface :=
  Formal.outputBindingInterface
    (Formal.atOffset
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset)
    NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset

abbrev outputOffset :=
  Formal.outputBindingOffset NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset

/-- The sampler output state of the Stage 1 layout. -/
abbrev samplerState : NightstreamFPrime.Gadgets.Poseidon2.Layer.EState :=
  SamplerChain.outputState
    (Formal.samplerInterface
      (Formal.atOffset
        (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset))
    (Formal.samplerOffset NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset)

/-- An F′ logical expression on package columns. -/
abbrev column (expression : Expr) : Expr := WitnessProgram.remapExpr expression

def extension (value : NightstreamFPrime.Circuit.Quadratic.KExpr) : List Expr :=
  [column value.c0, column value.c1]

/-- The output map, in the order layer 1 reads its claim. -/
structure Outputs where
  /-- Row-major: 22 rows of 54 lanes. -/
  commitment : List Expr
  publicInput : List Expr
  /-- Two words per coordinate. -/
  point : List Expr
  eval_K : List Expr
  /-- Matrix-major. -/
  eval_A : List Expr
  state : List Expr

def outputs : Outputs where
  commitment := (List.ofFn fun row => List.ofFn fun lane =>
    column (outputInterface.commitment outputOffset row lane)).flatten
  publicInput := List.ofFn fun coordinate =>
    column (outputInterface.publicInput outputOffset coordinate)
  point := (List.ofFn fun coordinate =>
    extension (outputInterface.point outputOffset coordinate)).flatten
  eval_K := (List.ofFn fun coefficient =>
    extension (outputInterface.eval_K outputOffset coefficient)).flatten
  eval_A := (List.ofFn fun matrix => (List.ofFn fun coefficient =>
    extension (outputInterface.eval_A outputOffset matrix coefficient)).flatten).flatten
  state := List.ofFn fun lane => column (samplerState lane)

/-- The canonical codec value of the output map. -/
def Outputs.value (map : Outputs) : NightstreamFPrime.Export.Codec.Value :=
  .array ([map.commitment, map.publicInput, map.point, map.eval_K, map.eval_A,
    map.state].map (NightstreamFPrime.Export.Codec.list
      NightstreamFPrime.Export.Package.exprFormat).encode)

/-- The terminal artifact: schema 1, the package, the output map. -/
def envelope (_unit : Unit) : NightstreamFPrime.Export.Codec.Value :=
  .array [.atom 1,
    NightstreamFPrime.Export.Package.CircuitPackage.format.encode (circuitPackage ()),
    outputs.value]

theorem column_eval (env : Env) (expression : Expr) :
    (column expression).eval env =
      expression.eval (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) :=
  WitnessProgram.remapExpr_eval env expression

section

variable {lw : Nat} {pf : NightstreamFPrime.Spec.ringDegree *
    NightstreamFPrime.Lifecycle.PaperAlgebra.publicRingColumns ≤
  NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth lw}
  (relation : NightstreamFPrime.Lifecycle.ProductionKey.LogicalRelation lw pf) (env : Env)

/-- The PiRLC output instance on the package assignment `env`, for the
relation of an application of logical width `lw`. The output map is the same
at every width. -/
abbrev parent :=
  Semantics.evalOutput relation
    (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
      (logicalWidth := lw) (publicFits := pf))
    NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)

theorem commitment_eval :
    outputs.commitment.map (fun expression => expression.eval env) =
      (List.ofFn fun row => List.ofFn fun lane =>
        (parent relation env).commitment row lane).flatten := by
  simp [outputs, List.map_flatten, List.map_ofFn, Function.comp_def, column_eval]
  rfl

theorem publicInput_eval :
    outputs.publicInput.map (fun expression => expression.eval env) =
      List.ofFn fun coordinate => (parent relation env).publicInput coordinate := by
  simp [outputs, List.map_ofFn, Function.comp_def, column_eval]
  rfl

/-- The single evaluation family of `parent`. -/
abbrev parentEvaluation :=
  OutputBinding.evalEvaluation outputInterface outputOffset
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)

theorem parent_evaluations :
    (parent relation env).evaluations = #[parentEvaluation env] := by
  rfl

theorem point_eval :
    outputs.point.map (fun expression => expression.eval env) =
      ((parent relation env).point.coordinates.map fun value =>
        [value.c0, value.c1]).flatten := by
  rw [show outputs.point = (List.ofFn fun coordinate => extension
      ((Formal.outputBindingInterface
        (Formal.atOffset
          (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface (logicalWidth := lw) (publicFits := pf))
          NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset)
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset).point outputOffset coordinate)).flatten from rfl]
  simp [extension, List.map_flatten, List.map_ofFn, Function.comp_def, column_eval]
  rfl

theorem eval_K_eval :
    outputs.eval_K.map (fun expression => expression.eval env) =
      (List.ofFn fun coefficient =>
        [((parentEvaluation env).pad coefficient).c0,
          ((parentEvaluation env).pad coefficient).c1]).flatten := by
  simp [outputs, extension, List.map_flatten, List.map_ofFn, Function.comp_def,
    column_eval]
  rfl

theorem eval_A_eval :
    outputs.eval_A.map (fun expression => expression.eval env) =
      (List.ofFn fun matrix => (List.ofFn fun coefficient =>
        [((parentEvaluation env).matrix matrix coefficient).c0,
          ((parentEvaluation env).matrix matrix coefficient).c1]).flatten).flatten := by
  simp [outputs, extension, List.map_flatten, List.map_ofFn, Function.comp_def,
    column_eval]
  rfl

theorem state_eval :
    outputs.state.map (fun expression => expression.eval env) =
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) samplerState := by
  change (List.ofFn fun lane => column (samplerState lane)).map
    (fun expression => expression.eval env) = _
  rw [List.map_ofFn]
  apply congrArg List.ofFn
  funext lane
  exact column_eval env (samplerState lane)

end

end CompressionLean.Terminal
