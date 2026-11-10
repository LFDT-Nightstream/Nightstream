import CompressionLean.Ports.BoundedDepth.Program
import Mathlib.Data.List.Basic

/-! Faster compiled forms of the finish rows (`BoundedDepth.finishRows`). Each `@[csimp]` theorem proves a
fast form equal to its definition, so compiled code after the theorem runs the fast form and the proofs
keep reading the definition. `Ports/` holds no theorem and compiles before these theorems, so
`compile_again` compiles the definitions on the path from the executables to the fast forms again,
here. -/

namespace CompressionLean.Lowering.Compiled

open NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Generic NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.PieceRows NightstreamFPrime.Spec.Phi81Relation

/-- `Gl`'s field with `Fin`'s ring operations, equal to `ZMod.instField` (`glField_eq`). `ZMod.commRing`
matches on the modulus inside each operation, so compiled code over Mathlib's instance builds `Fin`'s
ring for each operation; over this structure the compiler calls `Fin`'s operations. The executables
make it their local instance. -/
@[instance_reducible] def glField : Field CompressionLean.Gl :=
  { Fin.instCommRing goldilocksModulus, ZMod.instField goldilocksModulus with }

theorem glField_eq : glField = ZMod.instField goldilocksModulus := rfl

/-- `evaluation` with the powers of its point read from `powers`. -/
def evaluationWith (powers : Array F) (lanes : Fin ringDegree → R1CS.LinearCombination) :
    R1CS.LinearCombination :=
  sumCombinations (List.ofFn fun lane => .scale (powers.getD lane.val 0) (lanes lane))

/-- What the ring rows of one node share across the cells: the powers of the node's point, the value
of Φ81 there, and the challenge side `a`, which reads the same lanes for every cell. -/
structure Node where
  powers : Array F
  modulus : F
  challenge : R1CS.LinearCombination

def node (lane : ℕ → R1CS.LinearCombination) (index : Fin 108) : Node :=
  let point := QuotientProduct.node index
  let powers := Array.ofFn fun k : Fin ringDegree => point ^ k.val
  ⟨powers, QuotientProduct.modulusValue point, evaluationWith powers fun k => lane k.val⟩

/-- `ringRows` with the work that does not depend on the cell done once: each lane's combination
(`ringRow` walks `lanes` and decodes the lane for each term) and each node's `Node` (`ringRow` computes
it for each cell). -/
def ringRowsFast (cells start quotient : ℕ) (lanes : List Expr) : List R1CS.Row :=
  let combinations := (lanes.map laneCombination).toArray
  let lane (k : ℕ) : R1CS.LinearCombination := combinations.getD k (laneCombination 0)
  let nodes := Array.ofFn (node lane)
  (List.range cells).flatMap fun cell => List.ofFn fun index : Fin 108 =>
    let n := nodes[index.val]'(by rw [Array.size_ofFn]; exact index.isLt)
    { a := n.challenge
      b := evaluationWith n.powers fun k => lane (ringValue cells k cell)
      c := .add (evaluationWith n.powers fun k => .ofVar (outputColumn cells start k cell))
        (.add (.scale (-1) (evaluationWith n.powers fun k => lane (ringPrior cells k cell)))
          (.scale n.modulus (evaluationWith n.powers fun k => .ofVar (quotientColumn quotient k cell)))) }

theorem evaluation_eq (point : F) (lanes : Fin ringDegree → R1CS.LinearCombination) :
    evaluation point lanes = evaluationWith (Array.ofFn fun k : Fin ringDegree => point ^ k.val) lanes := by
  unfold evaluation evaluationWith
  apply congrArg sumCombinations
  apply List.ofFn_inj.mpr
  funext k
  simp [Array.getD_eq_getD_getElem?]

theorem lane_eq (lanes : List Expr) (k : ℕ) :
    (lanes.map laneCombination).toArray.getD k (laneCombination 0) = laneCombination (lanes.getD k 0) := by
  rw [Array.getD_eq_getD_getElem?, List.getElem?_toArray, ← List.getD_eq_getElem?_getD, List.getD_map]

@[csimp] theorem ringRows_eq_ringRowsFast : @ringRows = @ringRowsFast := by
  funext cells start quotient lanes
  unfold ringRows ringRowsFast
  refine List.flatMap_congr fun cell _ => List.ofFn_inj.mpr (funext fun index => ?_)
  simp only [Array.getElem_ofFn, node, ringRow, lane_eq, evaluation_eq]

/-- `Replay.hashLeaf` with each chunk's lanes read from `values` in place, without the list of
chunks (`hashLeaf_eq_hashLeafFast`). -/
def hashLeafFast (values : Array Replay.Word) : Array Replay.Word :=
  ((List.range ((values.size + Replay.rate - 1) / Replay.rate)).foldl (fun state c =>
      Replay.permute ((List.range (min (Replay.rate * c + Replay.rate) values.size - Replay.rate * c)).foldl
        (fun s i => s.set! i values[Replay.rate * c + i]!) state))
    (Array.replicate Replay.width 0)).extract 0 4

@[csimp] theorem hashLeaf_eq_hashLeafFast : @Replay.hashLeaf = @hashLeafFast := by
  funext values
  unfold Replay.hashLeaf hashLeafFast
  simp only [List.foldl_map]
  refine congrArg (fun state : Array Replay.Word => state.extract 0 4) ?_
  refine List.foldl_ext _ _ _ fun state c _ => ?_
  rw [Array.size_extract]
  refine congrArg Replay.permute (List.foldl_ext _ _ _ fun s i inside => ?_)
  have small := List.mem_range.mp inside
  rw [getElem!_pos _ i (by rw [Array.size_extract]; exact small), Array.getElem_extract,
    getElem!_pos values (Replay.rate * c + i) (by omega)]

open NightstreamFPrime.Export.BoundedDepth.StepPackage in
/-- The count of a piece's relation rows, without the rows (`pieceRowCount_eq`). -/
def pieceRowCount : PackagePiece → ℕ
  | .rows chunk => chunk.length
  | .ring cells .. => cells * 108
  | .block .. => 0

open NightstreamFPrime.Export.BoundedDepth.StepPackage in
theorem pieceRowCount_eq (piece : PackagePiece) : pieceRowCount piece = piece.relationRows.length := by
  cases piece with
  | rows chunk => rfl
  | ring cells start quotient lanes =>
      simp only [pieceRowCount, PackagePiece.relationRows, ringRows, List.length_flatMap, List.length_ofFn,
        List.map_const', List.sum_replicate, List.length_range, smul_eq_mul]
  | block start lanes => rfl

/-- The count of finish rows, without the rows (`finishRowCount_eq`). -/
def finishRowCount (finish : NightstreamFPrime.Export.BoundedDepth.StepPackage.Package) : ℕ :=
  (finish.pieces.map pieceRowCount).sum

theorem finishRowCount_eq (finish : NightstreamFPrime.Export.BoundedDepth.StepPackage.Package) :
    finishRowCount finish = (BoundedDepth.finishRows finish).length := by
  rw [finishRowCount, BoundedDepth.finishRows, List.length_map, List.length_flatMap]
  exact congrArg List.sum (List.map_congr_left fun piece _ => pieceRowCount_eq piece)

open Lean Elab Command Meta in
/-- `compile_again f …` adds, for each definition `f` in order, a copy `fCompiled` with the value of
`f`, compiled with the `@[csimp]` theorems now in scope, and `@[csimp] theorem f_eq_compiled :
@f = @fCompiled` by `rfl`. Compiled code after this point that calls `f` runs the copy. -/
local elab "compile_again " ids:ident+ : command => liftTermElabM do
  for id in ids do
    let name ← realizeGlobalConstNoOverloadWithInfo id
    let .defnInfo info ← getConstInfo name | throwError "{name} is not a definition"
    let copy := name.appendAfter "Compiled"
    addAndCompile (.defnDecl { info with name := copy, all := [copy] })
    let levels := info.levelParams.map mkLevelParam
    let equal := name.appendAfter "_eq_compiled"
    let statement ← mkEq (mkConst name levels) (mkConst copy levels)
    addDecl (.thmDecl (mkTheoremValEx equal info.levelParams statement (← mkEqRefl (mkConst name levels)) []))
    Compiler.CSimp.add equal .global

compile_again NightstreamFPrime.Export.BoundedDepth.StepPackage.PackagePiece.relationRows
  CompressionLean.BoundedDepth.finishRows CompressionLean.BoundedDepth.finishPrefix CompressionLean.BoundedDepth.finalProgram

end CompressionLean.Lowering.Compiled
