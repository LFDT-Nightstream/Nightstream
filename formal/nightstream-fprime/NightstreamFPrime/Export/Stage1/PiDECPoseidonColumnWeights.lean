import NightstreamFPrime.Export.Stage1.PiDECMatrixWeightedRange
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange
import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericRows
import Mathlib.Tactic.Ring
import Mathlib.Data.ZMod.Defs

/-!
Column weights of Poseidon S-box invocations. Every port of an S-box row is a
linear function of the column reads: the selector reads the one column, the
output reads a retained S-box output, and the input is a state lane plus a round
constant times the one column, where the state is the image of earlier retained
outputs (or of the input state) under the linear layers. The weighted sum of the
rows of an invocation is therefore `Σ_j W_j read_j`, and one reverse pass through
the transposed layers gives the column weights `W` of the input port. The
weights do not depend on the read, so a range prepares them once for every
source, child and lane, and then reads each column once per lane.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECPoseidonColumnWeights

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.ProductionRelation.RowSemantics (PortValues)
open NightstreamFPrime.Export.Stage1.PiDECMatrixWeightedRange (scale)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

/-- An adjoint value for each state lane. -/
abbrev Adjoint := Vector K 16

private def zeroK : K := K.zero

/-- One lane of an adjoint, or zero past the lanes. -/
def getK (state : Fin 16 → K) (index : Nat) : K :=
  if h : index < 16 then state ⟨index, h⟩ else K.zero

/-- `M₄ᵀ` for one lane of the block at `base`. -/
def mat4T (state : Fin 16 → K) (base lane : Nat) : K :=
  match lane with
  | 0 => K.add (K.add (K.add (scale (getK state base) 2) (getK state (base + 1)))
      (getK state (base + 2))) (scale (getK state (base + 3)) 3)
  | 1 => K.add (K.add (K.add (scale (getK state base) 3) (scale (getK state (base + 1)) 2))
      (getK state (base + 2))) (getK state (base + 3))
  | 2 => K.add (K.add (K.add (getK state base) (scale (getK state (base + 1)) 3))
      (scale (getK state (base + 2)) 2)) (getK state (base + 3))
  | _ => K.add (K.add (K.add (getK state base) (getK state (base + 1)))
      (scale (getK state (base + 2)) 3)) (scale (getK state (base + 3)) 2)

/-- The sum of the lanes congruent to `index` modulo four. -/
def columnT (state : Fin 16 → K) (index : Nat) : K :=
  ((List.range 4).map fun block => getK state (4 * block + index % 4)).foldl K.add K.zero

/-- The transpose of `Layer.externalF`: `(I + J)` across blocks, then `M₄ᵀ` per block. -/
def externalT (state : Fin 16 → K) : Fin 16 → K :=
  fun lane => mat4T (fun l => K.add (state l) (columnT state l.val)) (4 * (lane.val / 4))
    (lane.val % 4)

/-- The sum of all lanes. -/
def sumT (state : Fin 16 → K) : K :=
  ((List.range 16).map (getK state)).foldl K.add K.zero

/-- The transpose of `Layer.internalF`, which is symmetric. -/
def internalT (state : Fin 16 → K) : Fin 16 → K :=
  fun lane => K.add (scale (state lane) (Spec.Poseidon2.ofNat
    (Spec.Poseidon2.internalDiagonal.getD lane.val 0))) (sumT state)

/-- `Σ_lane term lane`. -/
def sum16 (term : Fin 16 → K) : K :=
  (List.finRange 16).foldl (fun total lane => K.add total (term lane)) K.zero

private def nextIndex (next : Nat) : Permutation.Step → Nat
  | .initialLayer => next
  | .initialFullRound _ => next + 16
  | .partialRound _ => next + 1
  | .terminalFullRound _ => next + 16

private def rowCount : Permutation.Step → Nat
  | .initialLayer => 0
  | .initialFullRound _ => 16
  | .partialRound _ => 1
  | .terminalFullRound _ => 16

/-- Each step with its first retained S-box index and its first row. -/
private def steps : List Permutation.Step → Nat → Nat → List (Permutation.Step × Nat × Nat)
  | [], _, _ => []
  | step :: rest, next, row => (step, next, row) :: steps rest (nextIndex next step) (row + rowCount step)

/-- The weights of the selector, output and input ports of one column. -/
abbrev Weights := K × K × K

/-- Add `weight` to one port: `0` selector, `1` output, `2` input. -/
def Weights.bump (weights : Weights) (port : Fin 3) (weight : K) : Weights :=
  match port with
  | 0 => (K.add weights.1 weight, weights.2)
  | 1 => (weights.1, K.add weights.2.1 weight, weights.2.2)
  | 2 => (weights.1, weights.2.1, K.add weights.2.2 weight)

/-- Column weights of all three ports: the distinct columns in order of first
use. `slots` only suggests where a column is; an update checks the slot, and
otherwise appends the column, so the weights never depend on the map. -/
structure Merged (columns : Nat) where
  keys : Array (Fin columns)
  weights : Array Weights
  slots : Std.HashMap Nat Nat

def Merged.push {columns : Nat} (merged : Merged columns) (port : Fin 3) (column : Fin columns)
    (weight : K) : Merged columns :=
  -- The slot is read before the push, so that both arrays stay unshared and grow in place.
  let slot := merged.keys.size
  { keys := merged.keys.push column
    weights := merged.weights.push (Weights.bump (zeroK, zeroK, zeroK) port weight)
    slots := merged.slots.insert column.val slot }

/-- Add `weight` to the weight of `column` in `port`. -/
def Merged.add {columns : Nat} (merged : Merged columns) (port : Fin 3) (column : Fin columns)
    (weight : K) : Merged columns :=
  match merged.slots[column.val]? with
  | some slot =>
      if bound : slot < merged.weights.size then
        if merged.keys[slot]? = some column then
          { merged with weights := merged.weights.set slot (merged.weights[slot].bump port weight) }
        else merged.push port column weight
      else merged.push port column weight
  | none => merged.push port column weight

private def addForm {columns : Nat} (merged : Merged columns) (port : Fin 3)
    (form : SparseForm columns) (weight : K) : Merged columns :=
  form.entries.foldl (fun merged entry =>
    merged.add port entry.column (scale weight entry.coefficient)) merged

/-- The weights of one lane of a full round: the selector and output of its
row, and the input port through the retained output and the round constant. -/
private def fullLane {columns : Nat} (interface : PoseidonSboxPlan.Interface columns)
    (weight outputAdjoint : K) (constant : F) (form : SparseForm columns)
    (merged : Merged columns) : Merged columns :=
  let merged := merged.add 0 interface.oneColumn weight
  let merged := addForm merged 1 form weight
  let merged := addForm merged 2 form outputAdjoint
  merged.add 2 interface.oneColumn (scale weight constant)

/-- The rows of one full round, and the adjoint of the state before it. -/
private def fullRound {columns arity : Nat} (interface : PoseidonSboxPlan.Interface columns)
    (point : CubePoint K arity) (start : Nat) (constants : List (List Nat))
    (round next row : Nat) (after : Adjoint) (merged : Merged columns) :
    Adjoint × Merged columns :=
  let outputs := Vector.ofFn (externalT after.get)
  let merged := (List.finRange 16).foldl (fun merged lane =>
      fullLane interface (PiDECEvaluationWeights.weight point (start + row + lane.val))
        (outputs.get lane) (Spec.Poseidon2.constantAt constants round lane.val)
        (PoseidonSboxPlan.sboxOutputAt interface (next + lane.val)) merged)
    merged
  (Vector.ofFn fun lane : Fin 16 => PiDECEvaluationWeights.weight point (start + row + lane.val),
    merged)

/-- Reverse one step: the adjoint of the state after it gives the adjoint before it. -/
private def reverseStep {columns arity : Nat} (interface : PoseidonSboxPlan.Interface columns)
    (point : CubePoint K arity) (start : Nat) :
    Permutation.Step × Nat × Nat → Adjoint × Merged columns → Adjoint × Merged columns
  | (.initialLayer, _, _), (after, merged) => (Vector.ofFn (externalT after.get), merged)
  | (.initialFullRound round, next, row), (after, merged) =>
      fullRound interface point start Spec.Poseidon2.initialConstants round next row after merged
  | (.terminalFullRound round, next, row), (after, merged) =>
      fullRound interface point start Spec.Poseidon2.terminalConstants round next row after merged
  | (.partialRound round, next, row), (after, merged) =>
      let mixed := Vector.ofFn (internalT after.get)
      let weight := PiDECEvaluationWeights.weight point (start + row)
      (Vector.ofFn fun lane : Fin 16 => if lane.val = 0 then weight else mixed.get lane,
        fullLane interface weight (mixed.get 0)
          (Spec.Poseidon2.ofNat (Spec.Poseidon2.internalConstants.getD round 0))
          (PoseidonSboxPlan.sboxOutputAt interface next) merged)

/-- Add the column weights of one invocation whose first row is `start`. -/
def addInvocation {columns arity : Nat} (point : CubePoint K arity) (start : Nat)
    (interface : PoseidonSboxPlan.Interface columns) (merged : Merged columns) :
    Merged columns :=
  let reversed := (steps Permutation.schedule 0 0).foldr
    (reverseStep interface point start) (Vector.replicate 16 zeroK, merged)
  (List.finRange 16).foldl (fun merged lane =>
    addForm merged 2 (interface.input lane) (reversed.1.get lane)) reversed.2

/-- The merged column weights of consecutive invocations. -/
def mergedOf {columns arity count : Nat} (firstRow : Nat) (point : CubePoint K arity)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count) : Merged columns :=
  Nat.fold count (fun index live merged =>
      addInvocation point (firstRow + 150 * index) (interfaces.get ⟨index, live⟩) merged)
    ⟨#[], #[], {}⟩

/-- One entry per merged column with the weights of its three ports. -/
def entriesOf {columns : Nat} (keys : List (Fin columns)) (weights : List Weights) :
    List (Fin columns × (F × F) × (F × F) × (F × F)) :=
  List.zipWith (fun column (weights : Weights) =>
      (column, (weights.1.c0, weights.1.c1), (weights.2.1.c0, weights.2.1.c1),
        (weights.2.2.c0, weights.2.2.c1)))
    keys weights

/-- The column weights of consecutive invocations with all three ports per column. -/
def prepare {columns arity count : Nat} (firstRow : Nat) (point : CubePoint K arity)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count) :
    List (Fin columns × (F × F) × (F × F) × (F × F)) :=
  let merged := mergedOf firstRow point interfaces
  entriesOf merged.keys.toList merged.weights.toList

/-- The selector, output and input ports of a matrix port, if it is one of them. -/
def portIndex? (port : Fin matrixCount) : Option (Fin 3) :=
  match port.val with
  | 1 => some 0
  | 4 => some 1
  | 5 => some 2
  | _ => none

/-- The two coordinates of one port of an entry. -/
def entryPair {columns : Nat} (entry : Fin columns × (F × F) × (F × F) × (F × F)) :
    Fin 3 → F × F
  | 0 => entry.2.1
  | 1 => entry.2.2.1
  | 2 => entry.2.2.2

/-- The range sum of one read from prepared column weights, reading each column
once per lane for the three ports. -/
@[specialize] def evaluate {columns : Nat}
    (prepared : List (Fin columns × (F × F) × (F × F) × (F × F)))
    (read : Fin ringDegree → Fin columns → F) : Vector MaterializedRingK matrixCount :=
  -- The coefficients become words once, for the reads of all lanes.
  let words := (prepared.map PiDECNativeSparseEvaluation.TripleEntry.ofEntry).toArray
  let lanes := Vector.ofFn fun output : Fin ringDegree =>
    PiDECNativeSparseEvaluation.nativeEvalTripleWords words (read output)
  Vector.ofFn fun port : Fin matrixCount =>
    match portIndex? port with
    | some 0 => MaterializedRingK.ofRing fun output =>
        ⟨(lanes.get output).1.1, (lanes.get output).1.2⟩
    | some 1 => MaterializedRingK.ofRing fun output =>
        ⟨(lanes.get output).2.1.1, (lanes.get output).2.1.2⟩
    | some 2 => MaterializedRingK.ofRing fun output =>
        ⟨(lanes.get output).2.2.1, (lanes.get output).2.2.2⟩
    | none => MaterializedRingK.ofRing fun _ => zeroK

/-! ### Values of merged weights -/

/-- One port of the weights of a column. -/
def Weights.get (weights : Weights) : Fin 3 → K
  | 0 => weights.1
  | 1 => weights.2.1
  | 2 => weights.2.2

private theorem bump_get (weights : Weights) (port target : Fin 3) (weight : K) :
    (weights.bump port weight).get target =
      if target = port then extensionOps.add (weights.get target) weight
      else weights.get target := by
  fin_cases port <;> fin_cases target <;> rfl

private theorem zero_get (target : Fin 3) :
    Weights.get (zeroK, zeroK, zeroK) target = extensionOps.zero := by
  fin_cases target <;> rfl

/-- The weighted sum of the values of the merged columns in one port. -/
def Merged.value {columns : Nat} (merged : Merged columns) (port : Fin 3)
    (values : Fin columns → K) : K :=
  PiDECMatrixWeightedRange.dot (merged.weights.toList.map (·.get port))
    (merged.keys.toList.map values)

/-- The change of one port when `weight` is added to `column` in `port`. -/
private def change (port target : Fin 3) (weight value : K) (total : K) : K :=
  if target = port then extensionOps.add total (extensionOps.mul weight value) else total

private theorem push_value {columns : Nat} (merged : Merged columns) (port : Fin 3)
    (column : Fin columns) (weight : K) (values : Fin columns → K)
    (sizes : merged.keys.size = merged.weights.size) :
    (merged.push port column weight).keys.size = (merged.push port column weight).weights.size ∧
      ∀ target, (merged.push port column weight).value target values =
        change port target weight (values column) (merged.value target values) := by
  refine ⟨by simp [Merged.push, sizes], fun target => ?_⟩
  simp only [Merged.value, Merged.push, Array.toList_push, List.map_append, List.map_cons,
    List.map_nil]
  rw [PiDECMatrixWeightedRange.dot_append_single _ _ _ _ (by simp [sizes]), bump_get, zero_get]
  unfold change
  split
  · rw [extensionLaws.zero_add]
  · rw [show extensionOps.mul extensionOps.zero (values column) = extensionOps.zero by
      rw [extensionLaws.mul_comm, extensionLaws.mul_zero], extensionLaws.add_zero]

private theorem add_value {columns : Nat} (merged : Merged columns) (port : Fin 3)
    (column : Fin columns) (weight : K) (values : Fin columns → K)
    (sizes : merged.keys.size = merged.weights.size) :
    (merged.add port column weight).keys.size = (merged.add port column weight).weights.size ∧
      ∀ target, (merged.add port column weight).value target values =
        change port target weight (values column) (merged.value target values) := by
  unfold Merged.add
  split
  · rename_i slot _
    by_cases bound : slot < merged.weights.size
    · rw [dif_pos bound]
      by_cases key : merged.keys[slot]? = some column
      · rw [if_pos key]
        refine ⟨by simp [sizes], fun target => ?_⟩
        have keyBound : slot < merged.keys.size := by omega
        have keyValue : merged.keys[slot] = column := by
          rw [Array.getElem?_eq_getElem keyBound] at key
          exact Option.some.inj key
        simp only [Merged.value, Array.toList_set, List.map_set]
        rw [bump_get]
        unfold change
        split
        · rename_i same
          subst same
          have set := PiDECMatrixWeightedRange.dot_set_add (merged.weights.toList.map (·.get target))
            (merged.keys.toList.map values) slot weight (by simpa using bound) (by simp [sizes])
          simp only [List.getElem_map, Array.getElem_toList] at set
          rw [set, keyValue]
        · have same : merged.weights[slot].get target =
              (merged.weights.toList.map (·.get target))[slot]'(by simpa using bound) := by simp
          rw [same, List.set_getElem_self]
      · rw [if_neg key]
        exact push_value merged port column weight values sizes
    · rw [dif_neg bound]
      exact push_value merged port column weight values sizes
  · exact push_value merged port column weight values sizes

/-! ### Transposes -/

private theorem vget {Alpha : Type} {size : Nat} (values : Vector Alpha size) (index : Fin size) :
    values.get index = values[index.val] := rfl

open Fin.CommRing in
/-- `externalT` is the transpose of `Layer.externalF` for every adjoint and state. -/
theorem externalT_dot (adjoint : Fin 16 → K) (state : Fin 16 → F) :
    sum16 (fun lane => scale (externalT adjoint lane) (state lane)) =
      sum16 (fun lane => scale (adjoint lane) (Layer.externalF state lane)) := by
  simp [sum16, List.finRange, externalT, Layer.externalF, Layer.blockF, Layer.columnF, mat4T,
    Layer.mat4F, columnT, getK, Layer.getF, scale, K.add, K.zero, List.range_succ]
  constructor <;> ring

set_option maxRecDepth 20000 in -- fixed-size: the sum of the sixteen Poseidon lanes
open Fin.CommRing in
/-- `internalT` is the transpose of `Layer.internalF` for every adjoint and state. -/
theorem internalT_dot (adjoint : Fin 16 → K) (state : Fin 16 → F) :
    sum16 (fun lane => scale (internalT adjoint lane) (state lane)) =
      sum16 (fun lane => scale (adjoint lane) (Layer.internalF state lane)) := by
  simp [sum16, List.finRange, internalT, Layer.internalF, Layer.sumF, sumT, getK, Layer.getF,
    scale, K.add, K.zero, List.range_succ]
  constructor <;> ring

/-! ### Adding forms and lanes -/

private def sizes {columns : Nat} (merged : Merged columns) : Prop :=
  merged.keys.size = merged.weights.size

/-- Adding the weighted entries of a form adds the weighted form in one port. -/
private theorem addForm_value {columns : Nat} (port : Fin 3) (weight : K)
    (read : Fin columns → F) :
    ∀ (entries : List (SparseEntry columns)) (merged : Merged columns), sizes merged →
      sizes (addForm merged port ⟨entries⟩ weight) ∧
      ∀ target, (addForm merged port ⟨entries⟩ weight).value target (fun column =>
          K.embed (read column)) =
        change port target weight (K.embed ((SparseForm.mk entries).evalSparse read))
          (merged.value target fun column => K.embed (read column))
  | [], merged, fits => by
      refine ⟨fits, fun target => ?_⟩
      unfold change
      split
      · change merged.value target _ = extensionOps.add _ (extensionOps.mul weight (K.embed 0))
        rw [show K.embed (0 : F) = extensionOps.zero from rfl, extensionLaws.mul_zero,
          extensionLaws.add_zero]
      · rfl
  | entry :: entries, merged, fits => by
      have step := add_value merged port entry.column (scale weight entry.coefficient)
        (fun column => K.embed (read column)) fits
      have rest := addForm_value port weight read entries _ step.1
      unfold addForm at rest ⊢
      simp only [List.foldl_cons]
      refine ⟨rest.1, fun target => ?_⟩
      rw [rest.2 target, step.2 target]
      have embed := PiDECMatrixWeightedRange.embed_evalSparse (SparseForm.mk (entry :: entries)) read
      have embedRest := PiDECMatrixWeightedRange.embed_evalSparse (SparseForm.mk entries) read
      simp only [List.map_cons] at embed
      unfold change
      split
      · rw [embedRest, embed, ← PiDECMatrixWeightedRange.mul_embed]
        change extensionOps.add (extensionOps.add _ (extensionOps.mul (extensionOps.mul weight
            (K.embed entry.coefficient)) (K.embed (read entry.column)))) _ =
          extensionOps.add _ (extensionOps.mul weight (extensionOps.add
            (extensionOps.mul (K.embed entry.coefficient) (K.embed (read entry.column))) _))
        rw [extensionLaws.left_distrib, extensionLaws.mul_assoc, extensionLaws.add_assoc]
      · rfl

private theorem fullLane_value {columns : Nat} (interface : PoseidonSboxPlan.Interface columns)
    (weight outputAdjoint : K) (constant : F) (form : SparseForm columns)
    (read : Fin columns → F) (merged : Merged columns) (fits : sizes merged) :
    sizes (fullLane interface weight outputAdjoint constant form merged) ∧
      ∀ target, (fullLane interface weight outputAdjoint constant form merged).value target
          (fun column => K.embed (read column)) =
        extensionOps.add (merged.value target fun column => K.embed (read column))
          (match target with
            | 0 => extensionOps.mul weight (K.embed (read interface.oneColumn))
            | 1 => extensionOps.mul weight (K.embed (form.evalSparse read))
            | 2 => extensionOps.add (extensionOps.mul outputAdjoint (K.embed (form.evalSparse read)))
                (extensionOps.mul (scale weight constant) (K.embed (read interface.oneColumn)))) := by
  let values := fun column => K.embed (read column)
  have first := add_value merged 0 interface.oneColumn weight values fits
  have second := addForm_value 1 weight read form.entries _ first.1
  have third := addForm_value 2 outputAdjoint read form.entries _ second.1
  have fourth := add_value _ 2 interface.oneColumn (scale weight constant) values third.1
  refine ⟨fourth.1, fun target => ?_⟩
  unfold fullLane
  rw [fourth.2 target, third.2 target, second.2 target, first.2 target]
  fin_cases target <;> simp [change, extensionLaws.add_assoc] <;> rfl

/-- Each lane adds its contribution to every port. -/
private theorem foldl_lanes {columns : Nat} (update : Fin 16 → Merged columns → Merged columns)
    (values : Fin columns → K) (delta : Fin 3 → Fin 16 → K)
    (step : ∀ lane merged, sizes merged → sizes (update lane merged) ∧
      ∀ target, (update lane merged).value target values =
        extensionOps.add (merged.value target values) (delta target lane)) :
    ∀ (lanes : List (Fin 16)) (merged : Merged columns), sizes merged →
      sizes (lanes.foldl (fun merged lane => update lane merged) merged) ∧
      ∀ target, (lanes.foldl (fun merged lane => update lane merged) merged).value target values =
        lanes.foldl (fun total lane => extensionOps.add total (delta target lane))
          (merged.value target values)
  | [], merged, fits => ⟨fits, fun _ => rfl⟩
  | lane :: lanes, merged, fits => by
      have now := step lane merged fits
      have rest := foldl_lanes update values delta step lanes _ now.1
      refine ⟨rest.1, fun target => ?_⟩
      simp only [List.foldl_cons]
      rw [rest.2 target, now.2 target]

/-! ### Rows and the forward pass -/

/-- The selector, output and input of a row, by port. -/
def portOf (values : PortValues) : Fin 3 → F
  | 0 => values.generalSelector
  | 1 => values.c
  | 2 => values.sboxInput

/-- The weighted rows from `row` on, in one port. -/
def rowTotal (weight : Nat → K) (port : Fin 3) : List PortValues → Nat → K
  | [], _ => K.zero
  | values :: rows, row => extensionOps.add (extensionOps.mul (weight row)
      (K.embed (portOf values port))) (rowTotal weight port rows (row + 1))

private theorem rowTotal_append (weight : Nat → K) (port : Fin 3) :
    ∀ (first second : List PortValues) (row : Nat),
      rowTotal weight port (first ++ second) row =
        extensionOps.add (rowTotal weight port first row)
          (rowTotal weight port second (row + first.length))
  | [], second, row => (extensionLaws.zero_add _).symm
  | values :: first, second, row => by
      simp only [List.cons_append, rowTotal, List.length_cons]
      rw [rowTotal_append weight port first second (row + 1), Nat.add_right_comm,
        Nat.add_assoc]
      exact (extensionLaws.add_assoc _ _ _).symm

/-- `⟨adjoint, state⟩`. -/
def dotState (adjoint : Adjoint) (state : Vector F 16) : K :=
  sum16 fun lane => scale (adjoint.get lane) (state.get lane)

private theorem dotState_zero (state : Vector F 16) :
    dotState (Vector.replicate 16 zeroK) state = K.zero := by
  simp [dotState, sum16, List.finRange, scale, zeroK, K.zero, K.add, vget]

private theorem steps_cons (step : Permutation.Step) (rest : List Permutation.Step)
    (next row : Nat) :
    steps (step :: rest) next row =
      (step, next, row) :: steps rest (nextIndex next step) (row + rowCount step) := rfl

private theorem rowsWithState_cons {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat) (state : Vector F 16)
    (step : Permutation.Step) (rest : List Permutation.Step) :
    (PiDECPoseidonNumericRows.rowsWithState read interface next state (step :: rest)).1 =
      (PiDECPoseidonNumericStep.stepValues read interface next state step).1 ++
        (PiDECPoseidonNumericRows.rowsWithState read interface (nextIndex next step)
          (PiDECPoseidonNumericStep.stepValues read interface next state step).2 rest).1 := by
  cases step <;> rfl

private theorem stepRows_length {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat) (state : Vector F 16)
    (step : Permutation.Step) :
    (PiDECPoseidonNumericStep.stepValues read interface next state step).1.length =
      rowCount step := by
  cases step <;> rfl

/-- The rows and next state of a full round, as `stepValues` computes them. -/
private def fullForward {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (constants : List (List Nat))
    (round next : Nat) (state : Vector F 16) : List PortValues × Vector F 16 :=
  let selector := read interface.oneColumn
  let outputs := Vector.ofFn fun lane : Fin 16 =>
    PiDECPoseidonNumericStep.retainedValue read interface (next + lane.val)
  (List.ofFn fun lane : Fin 16 =>
      RowSemantics.sbox selector
        (state.get lane + Spec.Poseidon2.constantAt constants round lane.val * selector)
        (outputs.get lane),
    Vector.ofFn (Layer.externalF outputs.get))

private theorem stepValues_initialFull {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat) (state : Vector F 16)
    (round : Nat) :
    PiDECPoseidonNumericStep.stepValues read interface next state (.initialFullRound round) =
      fullForward read interface Spec.Poseidon2.initialConstants round next state := rfl

private theorem stepValues_terminalFull {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat) (state : Vector F 16)
    (round : Nat) :
    PiDECPoseidonNumericStep.stepValues read interface next state (.terminalFullRound round) =
      fullForward read interface Spec.Poseidon2.terminalConstants round next state := rfl

private abbrev rowWeight {arity : Nat} (point : CubePoint K arity) (start : Nat) : Nat → K :=
  fun index => PiDECEvaluationWeights.weight point (start + index)

/-- What one reverse step must add: the weighted rows of the step, in the input
port together with the change of the adjoint, and directly in the other ports. -/
private def StepValue {columns arity : Nat} (point : CubePoint K arity) (start : Nat)
    (read : Fin columns → F) (row : Nat) (state : Vector F 16)
    (forward : List PortValues × Vector F 16) (after : Adjoint) (merged : Merged columns)
    (result : Adjoint × Merged columns) : Prop :=
  sizes result.2 ∧
    (∀ target : Fin 3, target ≠ 2 → result.2.value target (fun column => K.embed (read column)) =
      extensionOps.add (merged.value target (fun column => K.embed (read column)))
        (rowTotal (rowWeight point start) target forward.1 row)) ∧
    extensionOps.add (result.2.value 2 (fun column => K.embed (read column))) (dotState result.1 state) =
      extensionOps.add (extensionOps.add (merged.value 2 (fun column => K.embed (read column)))
          (dotState after forward.2))
        (rowTotal (rowWeight point start) 2 forward.1 row)

private theorem initialLayer_value {columns arity : Nat} (point : CubePoint K arity)
    (start : Nat) (read : Fin columns → F) (row : Nat) (state : Vector F 16) (after : Adjoint)
    (merged : Merged columns) (fits : sizes merged) :
    StepValue point start read row state ([], Vector.ofFn (Layer.externalF state.get)) after merged
      (Vector.ofFn (externalT after.get), merged) := by
  have transposed : dotState (Vector.ofFn (externalT after.get)) state =
      dotState after (Vector.ofFn (Layer.externalF state.get)) := by
    simp only [dotState, vget, Vector.getElem_ofFn]
    exact externalT_dot after.get state.get
  refine ⟨fits, fun target _ => (extensionLaws.add_zero _).symm, ?_⟩
  change extensionOps.add _ _ = extensionOps.add _ extensionOps.zero
  rw [extensionLaws.add_zero, transposed]

/-- The contribution of one lane of a full round to each port. -/
private def fullDelta {columns arity : Nat} (interface : PoseidonSboxPlan.Interface columns)
    (point : CubePoint K arity) (start : Nat) (read : Fin columns → F)
    (constants : List (List Nat)) (round next row : Nat) (after : Adjoint)
    (target : Fin 3) (lane : Fin 16) : K :=
  let weight := PiDECEvaluationWeights.weight point (start + row + lane.val)
  let form := PoseidonSboxPlan.sboxOutputAt interface (next + lane.val)
  match target with
  | 0 => extensionOps.mul weight (K.embed (read interface.oneColumn))
  | 1 => extensionOps.mul weight (K.embed (form.evalSparse read))
  | 2 => extensionOps.add (extensionOps.mul ((Vector.ofFn (externalT after.get)).get lane)
        (K.embed (form.evalSparse read)))
      (extensionOps.mul (scale weight (Spec.Poseidon2.constantAt constants round lane.val))
        (K.embed (read interface.oneColumn)))

open Fin.CommRing in
private theorem fullRound_value {columns arity : Nat}
    (interface : PoseidonSboxPlan.Interface columns) (point : CubePoint K arity) (start : Nat)
    (read : Fin columns → F) (constants : List (List Nat)) (round next row : Nat)
    (state : Vector F 16) (after : Adjoint) (merged : Merged columns) (fits : sizes merged) :
    StepValue point start read row state (fullForward read interface constants round next state)
      after merged (fullRound interface point start constants round next row after merged) := by
  have lanes := foldl_lanes (fun lane merged => fullLane interface
      (PiDECEvaluationWeights.weight point (start + row + lane.val))
      ((Vector.ofFn (externalT after.get)).get lane)
      (Spec.Poseidon2.constantAt constants round lane.val)
      (PoseidonSboxPlan.sboxOutputAt interface (next + lane.val)) merged)
    (fun column => K.embed (read column)) (fullDelta interface point start read constants round next row after)
    (fun lane merged fits => by
      have value := fullLane_value interface
        (PiDECEvaluationWeights.weight point (start + row + lane.val))
        ((Vector.ofFn (externalT after.get)).get lane)
        (Spec.Poseidon2.constantAt constants round lane.val)
        (PoseidonSboxPlan.sboxOutputAt interface (next + lane.val)) read merged fits
      exact ⟨value.1, fun target => by rw [value.2 target]; fin_cases target <;> rfl⟩)
    (List.finRange 16) merged fits
  let outputs := Vector.ofFn fun lane : Fin 16 =>
    PiDECPoseidonNumericStep.retainedValue read interface (next + lane.val)
  have transposed : dotState after (fullForward read interface constants round next state).2 =
      sum16 (fun lane => scale (externalT after.get lane) (outputs.get lane)) := by
    simp only [fullForward, dotState, vget, Vector.getElem_ofFn]
    exact (externalT_dot after.get outputs.get).symm
  have result : fullRound interface point start constants round next row after merged =
      (Vector.ofFn fun lane : Fin 16 =>
          PiDECEvaluationWeights.weight point (start + row + lane.val),
        (List.finRange 16).foldl (fun merged lane => fullLane interface
          (PiDECEvaluationWeights.weight point (start + row + lane.val))
          ((Vector.ofFn (externalT after.get)).get lane)
          (Spec.Poseidon2.constantAt constants round lane.val)
          (PoseidonSboxPlan.sboxOutputAt interface (next + lane.val)) merged) merged) := by
    simp only [fullRound]
  rw [result]
  beta_reduce at lanes
  unfold StepValue
  dsimp only
  refine ⟨?_, ?_, ?_⟩
  · exact lanes.1
  · intro target other
    rw [lanes.2 target]
    fin_cases target
    · simp [fullDelta, fullForward, rowTotal, List.finRange, List.ofFn_succ, portOf,
        RowSemantics.sbox, RowSemantics.general, PiDECPoseidonNumericStep.retainedValue,
        vget, Vector.getElem_ofFn, extensionOps, K.add, K.mul, K.embed, K.zero, Nat.add_assoc]
      constructor <;> ring
    · simp [fullDelta, fullForward, rowTotal, List.finRange, List.ofFn_succ, portOf,
        RowSemantics.sbox, RowSemantics.general, PiDECPoseidonNumericStep.retainedValue,
        vget, Vector.getElem_ofFn, extensionOps, K.add, K.mul, K.embed, K.zero, Nat.add_assoc]
      constructor <;> ring
    · exact absurd rfl other
  · rw [lanes.2 2, transposed]
    simp [fullDelta, fullForward, rowTotal, dotState, sum16, List.finRange, outputs,
      List.ofFn_succ, portOf, RowSemantics.sbox, RowSemantics.general, vget,
      Vector.getElem_ofFn, PiDECPoseidonNumericStep.retainedValue, scale, extensionOps, K.add,
      K.mul, K.embed, K.zero, Nat.add_assoc]
    constructor <;> ring

/-- The row and next state of a partial round, as `stepValues` computes them. -/
private def partialForward {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (round next : Nat) (state : Vector F 16) :
    List PortValues × Vector F 16 :=
  let selector := read interface.oneColumn
  let output := PiDECPoseidonNumericStep.retainedValue read interface next
  let replaced := Vector.ofFn fun lane : Fin 16 =>
    if lane.val = 0 then output else state.get lane
  ([RowSemantics.sbox selector
      (state.get 0 + Spec.Poseidon2.ofNat
        (Spec.Poseidon2.internalConstants.getD round 0) * selector)
      output],
    Vector.ofFn (Layer.internalF replaced.get))

private theorem stepValues_partial {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat) (state : Vector F 16)
    (round : Nat) :
    PiDECPoseidonNumericStep.stepValues read interface next state (.partialRound round) =
      partialForward read interface round next state := rfl

open Fin.CommRing in
private theorem partialRound_value {columns arity : Nat}
    (interface : PoseidonSboxPlan.Interface columns) (point : CubePoint K arity) (start : Nat)
    (read : Fin columns → F) (round next row : Nat) (state : Vector F 16) (after : Adjoint)
    (merged : Merged columns) (fits : sizes merged) :
    StepValue point start read row state (partialForward read interface round next state)
      after merged (reverseStep interface point start (.partialRound round, next, row)
        (after, merged)) := by
  let weight := PiDECEvaluationWeights.weight point (start + row)
  let mixed := Vector.ofFn (internalT after.get)
  let constant := Spec.Poseidon2.ofNat (Spec.Poseidon2.internalConstants.getD round 0)
  let form := PoseidonSboxPlan.sboxOutputAt interface next
  have lane := fullLane_value interface weight (mixed.get 0) constant form read merged fits
  let replaced := Vector.ofFn fun lane : Fin 16 =>
    if lane.val = 0 then PiDECPoseidonNumericStep.retainedValue read interface next
    else state.get lane
  have transposed : dotState after (partialForward read interface round next state).2 =
      sum16 (fun lane => scale (internalT after.get lane) (replaced.get lane)) := by
    simp only [partialForward, dotState, vget, Vector.getElem_ofFn]
    exact (internalT_dot after.get replaced.get).symm
  have result : reverseStep interface point start (.partialRound round, next, row)
      (after, merged) =
      (Vector.ofFn fun lane : Fin 16 => if lane.val = 0 then weight else mixed.get lane,
        fullLane interface weight (mixed.get 0) constant form merged) := rfl
  rw [result]
  unfold StepValue
  dsimp only
  refine ⟨lane.1, ?_, ?_⟩
  · intro target other
    rw [lane.2 target]
    fin_cases target
    · simp [partialForward, rowTotal, portOf, RowSemantics.sbox, RowSemantics.general,
        extensionOps, K.add, K.mul, K.embed, K.zero, weight]
    · simp [partialForward, rowTotal, portOf, RowSemantics.sbox, RowSemantics.general,
        PiDECPoseidonNumericStep.retainedValue, form, extensionOps, K.add, K.mul,
        K.embed, K.zero, weight]
    · exact absurd rfl other
  · rw [lane.2 2, transposed]
    simp [partialForward, rowTotal, dotState, sum16, List.finRange, portOf, replaced, mixed,
      RowSemantics.sbox, RowSemantics.general, vget, Vector.getElem_ofFn,
      PiDECPoseidonNumericStep.retainedValue, form, constant, weight, scale,
      extensionOps, K.add, K.mul, K.embed, K.zero]
    constructor <;> ring

private theorem step_value {columns arity : Nat} (interface : PoseidonSboxPlan.Interface columns)
    (point : CubePoint K arity) (start : Nat) (read : Fin columns → F)
    (step : Permutation.Step) (next row : Nat) (state : Vector F 16) (after : Adjoint)
    (merged : Merged columns) (fits : sizes merged) :
    StepValue point start read row state
      (PiDECPoseidonNumericStep.stepValues read interface next state step) after merged
      (reverseStep interface point start (step, next, row) (after, merged)) := by
  cases step with
  | initialLayer => exact initialLayer_value point start read row state after merged fits
  | initialFullRound round =>
      rw [stepValues_initialFull]
      exact fullRound_value interface point start read _ round next row state after merged fits
  | partialRound round =>
      rw [stepValues_partial]
      exact partialRound_value interface point start read round next row state after merged fits
  | terminalFullRound round =>
      rw [stepValues_terminalFull]
      exact fullRound_value interface point start read _ round next row state after merged fits

private theorem rowsWithState_cons_state {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (next : Nat) (state : Vector F 16)
    (step : Permutation.Step) (rest : List Permutation.Step) :
    (PiDECPoseidonNumericRows.rowsWithState read interface next state (step :: rest)).2 =
      (PiDECPoseidonNumericRows.rowsWithState read interface (nextIndex next step)
        (PiDECPoseidonNumericStep.stepValues read interface next state step).2 rest).2 := by
  cases step <;> rfl

open Fin.CommRing in
/-- The reverse pass over any suffix of the schedule adds the weighted rows of the
suffix, starting from a zero adjoint after its last step. -/
private theorem suffix_value {columns arity : Nat} (interface : PoseidonSboxPlan.Interface columns)
    (point : CubePoint K arity) (start : Nat) (read : Fin columns → F) :
    ∀ (stepList : List Permutation.Step) (next row : Nat) (state : Vector F 16)
      (merged : Merged columns), sizes merged →
      StepValue point start read row state
        (PiDECPoseidonNumericRows.rowsWithState read interface next state stepList)
        (Vector.replicate 16 zeroK) merged
        ((steps stepList next row).foldr (reverseStep interface point start)
          (Vector.replicate 16 zeroK, merged))
  | [], next, row, state, merged, fits => by
      refine ⟨fits, fun target _ => (extensionLaws.add_zero _).symm, ?_⟩
      change extensionOps.add _ (dotState (Vector.replicate 16 zeroK) state) =
        extensionOps.add (extensionOps.add _ (dotState (Vector.replicate 16 zeroK) state))
          extensionOps.zero
      rw [extensionLaws.add_zero]
      rfl
  | step :: rest, next, row, state, merged, fits => by
      have before := suffix_value interface point start read rest (nextIndex next step)
        (row + rowCount step) (PiDECPoseidonNumericStep.stepValues read interface next state step).2
        merged fits
      generalize hRest : (steps rest (nextIndex next step) (row + rowCount step)).foldr
        (reverseStep interface point start) (Vector.replicate 16 zeroK, merged) = restResult
        at before
      obtain ⟨after, middle⟩ := restResult
      have now := step_value interface point start read step next row state after middle before.1
      have result : (steps (step :: rest) next row).foldr (reverseStep interface point start)
          (Vector.replicate 16 zeroK, merged) =
          reverseStep interface point start (step, next, row) (after, middle) := by
        rw [steps_cons, List.foldr_cons, hRest]
      rw [result]
      have length := stepRows_length read interface next state step
      refine ⟨now.1, fun target other => ?_, ?_⟩
      · rw [now.2.1 target other, before.2.1 target other, rowsWithState_cons, rowTotal_append,
          length]
        simp only [extensionOps, K.add, K.mk.injEq]
        constructor <;> ring
      · rw [now.2.2, before.2.2, rowsWithState_cons, rowTotal_append, length,
          rowsWithState_cons_state]
        simp only [extensionOps, K.add, K.mk.injEq]
        constructor <;> ring

private theorem foldl_add_start {Alpha : Type} (term : Alpha → K) :
    ∀ (items : List Alpha) (initial : K),
      items.foldl (fun total item => extensionOps.add total (term item)) initial =
        extensionOps.add initial
          (items.foldl (fun total item => extensionOps.add total (term item)) extensionOps.zero)
  | [], initial => (extensionLaws.add_zero _).symm
  | item :: items, initial => by
      simp only [List.foldl_cons]
      rw [foldl_add_start term items, foldl_add_start term items
        (extensionOps.add extensionOps.zero (term item)), extensionLaws.zero_add,
        extensionLaws.add_assoc]

/-- The rows of one invocation, as the numeric evaluation computes them. -/
private theorem values_eq {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) :
    PiDECPoseidonNumericRows.values read interface =
      (PiDECPoseidonNumericRows.rowsWithState read interface 0
        (PiDECPoseidonNumericStep.stateValues read interface.input) Permutation.schedule).1 := rfl

open Fin.CommRing in
/-- Adding one invocation adds its weighted rows in every port. -/
private theorem addInvocation_value {columns arity : Nat} (point : CubePoint K arity)
    (start : Nat) (interface : PoseidonSboxPlan.Interface columns) (read : Fin columns → F)
    (merged : Merged columns) (fits : sizes merged) :
    sizes (addInvocation point start interface merged) ∧
      ∀ target, (addInvocation point start interface merged).value target
          (fun column => K.embed (read column)) =
        extensionOps.add (merged.value target fun column => K.embed (read column))
          (rowTotal (rowWeight point start) target
            (PiDECPoseidonNumericRows.values read interface) 0) := by
  let state := PiDECPoseidonNumericStep.stateValues read interface.input
  have suffix := suffix_value interface point start read Permutation.schedule 0 0 state merged
    fits
  generalize hReversed : (steps Permutation.schedule 0 0).foldr
    (reverseStep interface point start) (Vector.replicate 16 zeroK, merged) = reversed at suffix
  have result : addInvocation point start interface merged =
      (List.finRange 16).foldl (fun merged lane =>
        addForm merged 2 (interface.input lane) (reversed.1.get lane)) reversed.2 := by
    rw [← hReversed]
    rfl
  have lanes := foldl_lanes (fun lane merged =>
      addForm merged 2 (interface.input lane) (reversed.1.get lane))
    (fun column => K.embed (read column))
    (fun target lane => if target = 2 then extensionOps.mul (reversed.1.get lane)
        (K.embed ((interface.input lane).evalSparse read)) else extensionOps.zero)
    (fun lane merged fits => by
      have form := addForm_value 2 (reversed.1.get lane) read (interface.input lane).entries
        merged fits
      refine ⟨form.1, fun target => ?_⟩
      rw [show addForm merged 2 (interface.input lane) (reversed.1.get lane) =
        addForm merged 2 ⟨(interface.input lane).entries⟩ (reversed.1.get lane) from rfl,
        form.2 target]
      by_cases same : target = 2
      · subst same
        simp [change]
      · simp only [change, same, if_false]
        exact (extensionLaws.add_zero _).symm)
    (List.finRange 16) reversed.2 suffix.1
  rw [result]
  refine ⟨lanes.1, fun target => ?_⟩
  rw [lanes.2 target, values_eq]
  rw [foldl_add_start]
  by_cases input : target = 2
  · subst input
    have total := suffix.2.2
    rw [show dotState (Vector.replicate 16 zeroK) _ = K.zero from dotState_zero _] at total
    have inputs : (List.finRange 16).foldl (fun total lane => extensionOps.add total
        (if (2 : Fin 3) = 2 then extensionOps.mul (reversed.1.get lane)
          (K.embed ((interface.input lane).evalSparse read)) else extensionOps.zero))
        extensionOps.zero = dotState reversed.1 state := by
      simp only [if_true, PiDECMatrixWeightedRange.mul_embed, dotState, sum16, state,
        PiDECPoseidonNumericStep.stateValues, vget, Vector.getElem_ofFn]
      rfl
    rw [inputs, total, show K.zero = extensionOps.zero from rfl, extensionLaws.add_zero]
  · have zeros : (List.finRange 16).foldl (fun total lane => extensionOps.add total
        (if target = 2 then extensionOps.mul (reversed.1.get lane)
          (K.embed ((interface.input lane).evalSparse read)) else extensionOps.zero))
        extensionOps.zero = extensionOps.zero := by
      simp only [input, if_false, extensionLaws.add_zero]
      simp [List.finRange]
    rw [zeros, extensionLaws.add_zero, suffix.2.1 target input]

/-! ### Ranges of invocations -/

private theorem fold_value {columns : Nat} (values : Fin columns → K) :
    ∀ (count : Nat) (update : (index : Nat) → index < count → Merged columns → Merged columns)
      (delta : Fin 3 → Nat → K),
      (∀ index live merged, sizes merged → sizes (update index live merged) ∧
        ∀ target, (update index live merged).value target values =
          extensionOps.add (merged.value target values) (delta target index)) →
      ∀ merged, sizes merged → sizes (Nat.fold count update merged) ∧
        ∀ target, (Nat.fold count update merged).value target values =
          extensionOps.add (merged.value target values)
            (NumericCompletionSum.numericSum extensionOps count (delta target))
  | 0, _, _, _, merged, fits => ⟨fits, fun _ => (extensionLaws.add_zero _).symm⟩
  | count + 1, update, delta, step, merged, fits => by
      have before := fold_value values count (fun index live => update index (by omega)) delta
        (fun index live => step index (by omega)) merged fits
      have now := step count (by omega) _ before.1
      rw [Nat.fold_succ]
      refine ⟨now.1, fun target => ?_⟩
      rw [now.2 target, before.2 target, PiDECMatrixWeightedRange.numericSum_succ,
        extensionLaws.add_assoc]

private theorem numericSum_shift (term : Nat → K) :
    ∀ count : Nat, NumericCompletionSum.numericSum extensionOps (count + 1) term =
      extensionOps.add (term 0)
        (NumericCompletionSum.numericSum extensionOps count fun index => term (index + 1))
  | 0 => by
      rw [PiDECMatrixWeightedRange.numericSum_succ]
      change extensionOps.add extensionOps.zero (term 0) =
        extensionOps.add (term 0) extensionOps.zero
      rw [extensionLaws.zero_add, extensionLaws.add_zero]
  | count + 1 => by
      rw [PiDECMatrixWeightedRange.numericSum_succ, numericSum_shift term count,
        PiDECMatrixWeightedRange.numericSum_succ, extensionLaws.add_assoc]

/-- The weighted rows from `row` on as an indexed sum. -/
private theorem rowTotal_eq (weight : Nat → K) (port : Fin 3) :
    ∀ (rows : List PortValues) (row : Nat),
      rowTotal weight port rows row =
        NumericCompletionSum.numericSum extensionOps rows.length (fun index =>
          extensionOps.mul (weight (row + index)) (K.embed (portOf (rows.getD index {}) port)))
  | [], _ => rfl
  | values :: rows, row => by
      rw [List.length_cons, numericSum_shift, rowTotal, rowTotal_eq weight port rows (row + 1)]
      simp only [Nat.add_zero, List.getD_cons_zero, List.getD_cons_succ, Nat.add_assoc,
        Nat.add_comm 1]

/-! ### Evaluation -/

open Fin.CommRing in
/-- The entries of each port evaluate as the dot product of its weights. -/
private theorem entries_eval {columns : Nat} (read : Fin columns → F) (target : Fin 3) :
    ∀ (keys : List (Fin columns)) (weights : List Weights), keys.length = weights.length →
      (⟨(SparseForm.mk ((entriesOf keys weights).map fun entry =>
            ⟨entry.1, (entryPair entry target).1⟩)).evalSparse read,
          (SparseForm.mk ((entriesOf keys weights).map fun entry =>
            ⟨entry.1, (entryPair entry target).2⟩)).evalSparse read⟩ : K) =
        PiDECMatrixWeightedRange.dot (weights.map (·.get target))
          (keys.map fun column => K.embed (read column))
  | [], [], _ => rfl
  | column :: keys, weights :: rest, same => by
      have tail := entries_eval read target keys rest (by simpa using same)
      simp only [entriesOf, List.zipWith_cons_cons, List.map_cons] at tail ⊢
      rw [PiDECMatrixWeightedRange.evalSparse_cons, PiDECMatrixWeightedRange.evalSparse_cons]
      change _ = extensionOps.add (extensionOps.mul _ _) (PiDECMatrixWeightedRange.dot _ _)
      rw [← tail]
      fin_cases target <;> simp [entryPair, Weights.get, extensionOps, K.add, K.mul, K.embed]
  | [], _ :: _, same => by simp at same
  | _ :: _, [], same => by simp at same

/-- Every row of the numeric evaluation is an S-box row. -/
private theorem rows_sbox {columns : Nat} (read : Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) :
    ∀ (stepList : List Permutation.Step) (next : Nat) (state : Vector F 16),
      ∀ values ∈ (PiDECPoseidonNumericRows.rowsWithState read interface next state stepList).1,
        ∃ selector input output, values = RowSemantics.sbox selector input output
  | [], _, _ => by
      intro values member
      simp [PiDECPoseidonNumericRows.rowsWithState] at member
  | step :: rest, next, state => by
      intro values member
      rw [rowsWithState_cons, List.mem_append] at member
      rcases member with now | later
      · cases step with
        | initialLayer => simp [PiDECPoseidonNumericStep.stepValues] at now
        | initialFullRound round =>
            rw [stepValues_initialFull] at now
            simp only [fullForward, List.mem_ofFn] at now
            obtain ⟨lane, same⟩ := now
            exact ⟨_, _, _, same.symm⟩
        | partialRound round =>
            rw [stepValues_partial] at now
            simp only [partialForward, List.mem_singleton] at now
            exact ⟨_, _, _, now⟩
        | terminalFullRound round =>
            rw [stepValues_terminalFull] at now
            simp only [fullForward, List.mem_ofFn] at now
            obtain ⟨lane, same⟩ := now
            exact ⟨_, _, _, same.symm⟩
      · exact rows_sbox read interface rest _ _ values later

private theorem get_port (values : PortValues) (port : Fin matrixCount) (target : Fin 3)
    (index : portIndex? port = some target) : values.get port = portOf values target := by
  fin_cases port <;> simp_all [portIndex?] <;> subst index <;> rfl

private theorem sbox_other (selector input output : F) (port : Fin matrixCount)
    (index : portIndex? port = none) : (RowSemantics.sbox selector input output).get port = 0 := by
  fin_cases port <;> simp_all [portIndex?] <;> rfl

private theorem numericSum_congr (count : Nat) (first second : Nat → K)
    (same : ∀ index, index < count → first index = second index) :
    NumericCompletionSum.numericSum extensionOps count first =
      NumericCompletionSum.numericSum extensionOps count second := by
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [PiDECMatrixWeightedRange.numericSum_succ, PiDECMatrixWeightedRange.numericSum_succ,
        ih (fun index bound => same index (by omega)), same count (by omega)]

private theorem numericSum_zero (count : Nat) :
    NumericCompletionSum.numericSum extensionOps count (fun _ => extensionOps.zero) =
      extensionOps.zero := by
  induction count with
  | zero => rfl
  | succ count ih => rw [PiDECMatrixWeightedRange.numericSum_succ, ih, extensionLaws.add_zero]

/-- One stored row of the existing invocation evaluation. -/
private theorem prepared_get {columns : Nat} (read : Fin ringDegree → Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (output : Fin ringDegree) (row : Nat)
    (bound : row < 150) :
    ((PiDECMatrixInvocation.prepare read interface).get output).get ⟨row, bound⟩ =
      (PiDECPoseidonNumericRows.values (read output) interface).getD row {} := by
  have length := PiDECPoseidonNumericRows.values_length (read output) interface
  rw [List.getD_eq_getElem _ _ (by omega)]
  simp only [PiDECMatrixInvocation.prepare, PiDECPoseidonNumericRows.stored, vget,
    Vector.getElem_ofFn, Vector.getElem_mk, List.getElem_toArray]

/-- The existing invocation sum of one port and lane is the weighted row total. -/
private theorem invocation_value {columns arity : Nat} (start : Nat) (point : CubePoint K arity)
    (read : Fin ringDegree → Fin columns → F) (interface : PoseidonSboxPlan.Interface columns)
    (port : Fin matrixCount) (target : Fin 3) (index : portIndex? port = some target)
    (output : Fin ringDegree) :
    ((PiDECMatrixInvocation.sum start point (PiDECMatrixInvocation.prepare read interface)).get
        port).toRing output =
      rowTotal (rowWeight point start) target
        (PiDECPoseidonNumericRows.values (read output) interface) 0 := by
  rw [PiDECMatrixInvocation.sum_value, rowTotal_eq, PiDECPoseidonNumericRows.values_length]
  apply numericSum_congr
  intro row bound
  rw [dif_pos bound, prepared_get read interface output row bound, get_port _ port target index,
    Nat.zero_add]

/-- Ports outside the selector, output and input have zero sums. -/
private theorem invocation_other {columns arity : Nat} (start : Nat) (point : CubePoint K arity)
    (read : Fin ringDegree → Fin columns → F) (interface : PoseidonSboxPlan.Interface columns)
    (port : Fin matrixCount) (index : portIndex? port = none) (output : Fin ringDegree) :
    ((PiDECMatrixInvocation.sum start point (PiDECMatrixInvocation.prepare read interface)).get
        port).toRing output = extensionOps.zero := by
  rw [PiDECMatrixInvocation.sum_value]
  have length := PiDECPoseidonNumericRows.values_length (read output) interface
  rw [numericSum_congr 150 _ (fun _ => extensionOps.zero) (fun row bound => ?_), numericSum_zero]
  rw [dif_pos bound, prepared_get read interface output row bound,
    List.getD_eq_getElem _ _ (by omega)]
  have member := List.getElem_mem (l := PiDECPoseidonNumericRows.values (read output) interface)
    (n := row) (by omega)
  obtain ⟨selector, input, value, same⟩ := rows_sbox (read output) interface
    Permutation.schedule 0 (PiDECPoseidonNumericStep.stateValues (read output) interface.input) _
    member
  rw [same, sbox_other selector input value port index]
  change extensionOps.mul _ extensionOps.zero = _
  rw [extensionLaws.mul_zero]

/-- Each prepared port evaluates to the merged value of that port. -/
private theorem evaluate_get {columns : Nat} (merged : Merged columns) (fits : sizes merged)
    (read : Fin ringDegree → Fin columns → F) (port : Fin matrixCount) (target : Fin 3)
    (index : portIndex? port = some target) (output : Fin ringDegree) :
    ((evaluate (entriesOf merged.keys.toList merged.weights.toList) read).get port).toRing output =
      merged.value target fun column => K.embed (read output column) := by
  have length : merged.keys.toList.length = merged.weights.toList.length := by
    rw [Array.length_toList, Array.length_toList]
    exact fits
  have entries := entries_eval (read output) target merged.keys.toList merged.weights.toList
    length
  fin_cases target <;>
    simp only [evaluate, vget, Vector.getElem_ofFn, index, MaterializedRingK.toRing_ofRing,
      PiDECNativeSparseEvaluation.nativeEvalTripleWords_ofEntry,
      PiDECNativeSparseEvaluation.nativeEvalTriple_eq_spec] <;>
    simp only [entryPair] at entries <;>
    exact entries

open Fin.CommRing in
/-- Prepared column weights give the existing numeric invocation range for every
read, port, lane and both field coordinates. No interface, read or row premise
is needed. -/
theorem evaluate_prepare_toRing {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity) (read : Fin ringDegree → Fin columns → F)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count) (port : Fin matrixCount) :
    ((evaluate (prepare firstRow point interfaces) read).get port).toRing =
      ((PiDECMatrixInvocationRange.sum firstRow point read interfaces).get port).toRing := by
  funext output
  unfold PiDECMatrixInvocationRange.sum
  rw [PiDECEvaluationBatch.sum_value]
  cases index : portIndex? port with
  | none =>
      rw [numericSum_congr count _ (fun _ => extensionOps.zero) (fun row live => by
        rw [dif_pos live, invocation_other _ point read _ port index output]), numericSum_zero]
      simp only [evaluate, vget, Vector.getElem_ofFn, index, MaterializedRingK.toRing_ofRing]
      rfl
  | some target =>
      have fold := fold_value (fun column => K.embed (read output column)) count
        (fun index live merged => addInvocation point (firstRow + 150 * index)
          (interfaces.get ⟨index, live⟩) merged)
        (fun target index => if live : index < count then
          rowTotal (rowWeight point (firstRow + 150 * index)) target
            (PiDECPoseidonNumericRows.values (read output) (interfaces.get ⟨index, live⟩)) 0
          else extensionOps.zero)
        (fun index live merged fits => by
          have now := addInvocation_value point (firstRow + 150 * index)
            (interfaces.get ⟨index, live⟩) (read output) merged fits
          exact ⟨now.1, fun target => by rw [now.2 target, dif_pos live]⟩)
        ⟨#[], #[], {}⟩ rfl
      have merged := fold.2 target
      rw [show (Merged.value (⟨#[], #[], {}⟩ : Merged columns) target
          fun column => K.embed (read output column)) = extensionOps.zero from rfl,
        extensionLaws.zero_add] at merged
      rw [numericSum_congr count _ (fun index => if live : index < count then
          rowTotal (rowWeight point (firstRow + 150 * index)) target
            (PiDECPoseidonNumericRows.values (read output) (interfaces.get ⟨index, live⟩)) 0
          else extensionOps.zero) (fun row live => by
        rw [dif_pos live, dif_pos live, invocation_value _ point read _ port target index output]),
        ← merged]
      exact evaluate_get (mergedOf firstRow point interfaces) fold.1 read port target index output

end NightstreamFPrime.Export.Stage1.PiDECPoseidonColumnWeights