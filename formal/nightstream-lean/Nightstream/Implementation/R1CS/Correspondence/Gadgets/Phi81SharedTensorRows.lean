import Mathlib.Tactic.Ring
import Nightstream.Implementation.R1CS.Correspondence.TerminalR1cs.Extension
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-!
Contract: one shared dynamic equality-weight tensor for terminal Phi81
evaluation.

The verifier point is read from circuit columns.  Each internal Boolean-cube
node owns one quadratic-extension multiplication.  Its high child is the
parent weight times the current point coordinate; its low child is the
allocation-free difference between the parent and that high child.  Thus the
complete tensor uses exactly `2^variableCount - 1` extension multiplications,
independent of the number of matrices and Phi81 lanes that later consume it.

The main theorem derives every leaf as the independent paper equality weight
`eq(vertex, point)` from the multiplication rows.  It does not accept a
weight table, an evaluation equation, or a validity bit.

This module does not own matrix-vector rows, claimed-evaluation rows, physical
artifact placement, frame disjointness, Rust refinement, or a compact proof
backend.  Frame disjointness is a completeness and ownership obligation; it
is not needed for row soundness.

Assurance tier: model-level shared-tensor soundness.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Phi81SharedTensorRows

open Nightstream.Implementation.R1CS.TerminalR1cs
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Dynamic point columns and the exact frame selected for every internal
tensor node.  `tensorFrame coordinate tailIndex` is structural placement; it
does not carry a value or an equation. -/
structure Frame (variableCount : Nat) where
  one : ColumnId
  pointLow : Fin variableCount -> ColumnId
  pointHigh : Fin variableCount -> ColumnId
  owner : PhysicalOwner
  tensorFirstOrdinal : Nat -> Nat -> Nat
  tensorFrame : Nat -> Nat -> Extension.Frame

/-- One extension value read directly from two circuit columns. -/
def pointValue {variableCount : Nat} (frame : Frame variableCount)
    (coordinate : Fin variableCount) : Extension.Value where
  low := [⟨frame.pointLow coordinate, 1⟩]
  high := [⟨frame.pointHigh coordinate, 1⟩]

/-- The constant extension-field unit, represented by the verifier-owned
constant-one column. -/
def oneValue {variableCount : Nat} (frame : Frame variableCount) : Extension.Value where
  low := [⟨frame.one, 1⟩]
  high := []

/-- Negate one sparse linear combination without allocating a column. -/
def negateCombination (terms : LinearCombination) : LinearCombination :=
  terms.map fun term => { term with coefficient := -term.coefficient }

/-- Subtract two carried extension values without allocating a column. -/
def subValue (left right : Extension.Value) : Extension.Value where
  low := left.low ++ negateCombination right.low
  high := left.high ++ negateCombination right.high

/-- Decode one carried extension value in the typed Goldilocks assignment. -/
def value (assignment : ColumnId -> F) (carried : Extension.Value) : K :=
  ⟨carried.low.eval assignment, carried.high.eval assignment⟩

theorem eval_append
    (assignment : ColumnId -> F)
    (left right : LinearCombination) :
    (left ++ right).eval assignment =
      left.eval assignment + right.eval assignment := by
  induction left with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, LinearCombination.eval_cons,
        inductionHypothesis]
      exact (Lean.Grind.Fin.add_assoc _ _ _).symm

private theorem eval_negateCombination
    (assignment : ColumnId -> F) (terms : LinearCombination) :
    (negateCombination terms).eval assignment =
      -terms.eval assignment := by
  induction terms with
  | nil =>
      simp only [negateCombination, List.map_nil,
        LinearCombination.eval_nil]
      exact Lean.Grind.AddCommGroup.neg_zero.symm
  | cons term tail inductionHypothesis =>
      change
        (-term.coefficient) * assignment term.column +
            (negateCombination tail).eval assignment =
          -(term.coefficient * assignment term.column +
            LinearCombination.eval assignment tail)
      rw [inductionHypothesis, Lean.Grind.AddCommGroup.neg_add,
        Lean.Grind.Fin.neg_mul]

@[simp] theorem value_oneValue
    {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F)
    (constantOne : assignment frame.one = 1) :
    value assignment (oneValue frame) = K.one := by
  simp only [value, oneValue, LinearCombination.eval_cons,
    LinearCombination.eval_nil, constantOne, Fin.one_mul, Fin.add_zero,
    K.one]

@[simp] theorem value_pointValue
    {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F) (coordinate : Fin variableCount) :
    value assignment (pointValue frame coordinate) =
      ⟨assignment (frame.pointLow coordinate),
       assignment (frame.pointHigh coordinate)⟩ := by
  simp only [value, pointValue, LinearCombination.eval_cons,
    LinearCombination.eval_nil, Fin.one_mul, Fin.add_zero]

theorem value_subValue
    (assignment : ColumnId -> F) (left right : Extension.Value) :
    value assignment (subValue left right) =
      K.sub (value assignment left) (value assignment right) := by
  simp only [value, subValue, K.sub, K.mk.injEq]
  rw [eval_append, eval_append, eval_negateCombination,
    eval_negateCombination]
  simp only [Fin.sub_eq_add_neg]
  exact ⟨trivial, trivial⟩

/-- Symbolic equality weight for one suffix vertex.  The recursion processes
the point in head-first paper order.  False and true siblings share the same
`tensorFrame`; only the high child is multiplied and the low child is a
linear difference. -/
def chiFrom {variableCount : Nat} (frame : Frame variableCount) :
    (start : Nat) -> {remaining : Nat} ->
      BooleanVertex remaining -> Extension.Value
  | _, 0, .nil => oneValue frame
  | start, remaining + 1, .cons bit tail =>
      let parent := chiFrom frame (start + 1) tail
      let high := Extension.output
        (frame.tensorFrame start (NumericBooleanDomain.index tail))
      if bit then high else subValue parent high

/-- Complete symbolic equality weight at one Boolean vertex. -/
def chi {variableCount : Nat} (frame : Frame variableCount)
    (vertex : BooleanVertex variableCount) : Extension.Value :=
  chiFrom frame 0 vertex

/-- One internal node of the exact full tensor.  The equation prevents a
caller from pairing a tail vertex with the wrong point coordinate. -/
structure Node (variableCount : Nat) where
  start : Nat
  remaining : Nat
  exact : start + remaining + 1 = variableCount
  tail : BooleanVertex remaining

def Node.coordinate {variableCount : Nat} (node : Node variableCount) : Fin variableCount :=
  ⟨node.start, by have exact := node.exact; omega⟩

def Node.tailIndex {variableCount : Nat} (node : Node variableCount) : Nat :=
  NumericBooleanDomain.index node.tail

def Node.parent {variableCount : Nat} (frame : Frame variableCount)
    (node : Node variableCount) : Extension.Value :=
  chiFrom frame (node.start + 1) node.tail

def Node.frame {variableCount : Nat} (layout : Frame variableCount)
    (node : Node variableCount) : Extension.Frame :=
  layout.tensorFrame node.start node.tailIndex

/-- The three actual Karatsuba rows for one shared tensor node. -/
def nodeRows {variableCount : Nat} (frame : Frame variableCount)
    (node : Node variableCount) : List OwnedRow :=
  Extension.rows frame.owner
    (frame.tensorFirstOrdinal node.start node.tailIndex)
    (node.parent frame) (pointValue frame node.coordinate) (node.frame frame)

/-- Canonical construction of an internal tensor node. Enumeration uses the
remaining suffix length because that leaves the dependent tail type exact. -/
def nodeAtRemaining {variableCount : Nat}
    (remaining : Fin variableCount)
    (tail : BooleanVertex remaining.val) : Node variableCount where
  start := variableCount - remaining.val - 1
  remaining := remaining.val
  exact := by omega
  tail := tail

/-- Canonical finite enumeration of every internal tensor node. The outer
order is increasing suffix length. The inner order is the independent
low/high Boolean-vertex order. -/
def nodes (variableCount : Nat) : List (Node variableCount) :=
  (canonicalFinIndices variableCount).flatMap fun remaining =>
    (BooleanVertex.all remaining.val).map fun tail =>
      nodeAtRemaining remaining tail

private theorem nodeAtRemaining_self
    {variableCount : Nat} (node : Node variableCount) :
    nodeAtRemaining
        (⟨node.remaining, by have exact := node.exact; omega⟩ :
          Fin variableCount)
        node.tail = node := by
  cases node with
  | mk start remaining exact tail =>
      simp only [nodeAtRemaining]
      have startExact : variableCount - remaining - 1 = start := by omega
      subst start
      rfl

/-- The canonical list contains every well-shaped internal node. -/
theorem mem_nodes {variableCount : Nat} (node : Node variableCount) :
    node ∈ nodes variableCount := by
  let remaining : Fin variableCount :=
    ⟨node.remaining, by have exact := node.exact; omega⟩
  apply List.mem_flatMap.mpr
  refine ⟨remaining, by simp [canonicalFinIndices], ?_⟩
  apply List.mem_map.mpr
  refine ⟨node.tail, BooleanVertex.mem_all node.tail, ?_⟩
  exact nodeAtRemaining_self node

/-- The exact finite shared-tensor row list. No caller supplies its nodes. -/
def rows {variableCount : Nat} (frame : Frame variableCount) :
    List OwnedRow :=
  (nodes variableCount).flatMap (nodeRows frame)

private theorem satisfies_flatMap_member
    {Index : Type} {parts : List Index}
    {rowsOf : Index -> List OwnedRow}
    {assignment : ColumnId -> F}
    (satisfied : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (parts.flatMap rowsOf) assignment)
    {part : Index} (member : part ∈ parts) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rowsOf part) assignment := by
  induction parts with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, satisfies_append_iff] at satisfied
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact satisfied.1
      · exact inductionHypothesis satisfied.2 tailMember

private theorem satisfies_flatMap_of_forall
    {Index : Type} (parts : List Index)
    (rowsOf : Index -> List OwnedRow)
    (assignment : ColumnId -> F)
    (satisfied : forall part, part ∈ parts ->
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (rowsOf part) assignment) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (parts.flatMap rowsOf) assignment := by
  induction parts with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, satisfies_append_iff]
      exact ⟨satisfied head (by simp),
        inductionHypothesis (fun part member =>
          satisfied part (by simp [member]))⟩

/-- Satisfaction of every actual shared-tensor row. -/
def RowsSatisfied {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F) : Prop :=
  forall node : Node variableCount,
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (nodeRows frame node) assignment

/-- Finite row satisfaction is exactly the prior universal node interface.
This theorem rules out a manifest that silently omits one tensor node. -/
theorem rows_satisfied_iff
    {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (rows frame) assignment ↔
      RowsSatisfied frame assignment := by
  constructor
  · intro satisfied node
    exact satisfies_flatMap_member satisfied (mem_nodes node)
  · intro satisfied
    apply satisfies_flatMap_of_forall
    intro node _
    exact satisfied node

/-- Total point lookup used only for the recursive semantic reference.  A
well-shaped tensor never takes the zero fallback. -/
def decodedCoordinateAt {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F) (index : Nat) : K :=
  if bounded : index < variableCount then
    ⟨assignment (frame.pointLow ⟨index, bounded⟩),
     assignment (frame.pointHigh ⟨index, bounded⟩)⟩
  else
    K.zero

/-- Independent recursive equality-weight semantics used before connecting
the rows to `BooleanVertex.equalityWeight`. -/
def semanticChiFrom {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F) :
    (start : Nat) -> {remaining : Nat} -> BooleanVertex remaining -> K
  | _, 0, .nil => K.one
  | start, remaining + 1, .cons bit tail =>
      let parent := semanticChiFrom frame assignment (start + 1) tail
      let point := decodedCoordinateAt frame assignment start
      if bit then K.mul parent point
      else K.sub parent (K.mul parent point)

private theorem sub_mul_self (left point : K) :
    K.sub left (K.mul left point) =
      K.mul (K.sub K.one point) left := by
  calc
    K.sub left (K.mul left point) =
        ConcreteCarrier.extensionOps.sub left
          (ConcreteCarrier.extensionOps.mul left point) := by
      simpa only [ConcreteCarrier.extensionOps] using
        (ConcreteCarrier.derived_sub_eq_concrete_sub
          left (K.mul left point)).symm
    _ = ConcreteCarrier.extensionOps.mul
          (ConcreteCarrier.extensionOps.sub
            ConcreteCarrier.extensionOps.one point) left := by
      unfold InterpolationOps.sub
      rw [ConcreteCarrier.extensionLaws.right_distrib,
        ConcreteCarrier.extensionLaws.one_mul,
        ConcreteCarrier.extensionLaws.neg_mul,
        ConcreteCarrier.extensionLaws.mul_comm point left]
    _ = K.mul (K.sub K.one point) left := by
      simpa only [ConcreteCarrier.extensionOps] using
        congrArg (fun value : K => K.mul value left)
          (ConcreteCarrier.derived_sub_eq_concrete_sub K.one point)

private theorem node_output_sound
    {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F)
    (satisfied : RowsSatisfied frame assignment)
    (node : Node variableCount) :
    value assignment (Extension.output (node.frame frame)) =
      K.mul (value assignment (node.parent frame))
        (decodedCoordinateAt frame assignment node.start) := by
  have rows := Extension.rows_sound frame.owner
    (frame.tensorFirstOrdinal node.start node.tailIndex)
    (node.parent frame) (pointValue frame node.coordinate) (node.frame frame)
    assignment (satisfied node)
  have pointExact :
      value assignment (pointValue frame node.coordinate) =
        decodedCoordinateAt frame assignment node.start := by
    rw [value_pointValue]
    have startBound : node.start < variableCount := by
      have exact := node.exact
      omega
    simp [decodedCoordinateAt, startBound, Node.coordinate]
  have rowsExact :
      value assignment (Extension.output (node.frame frame)) =
        K.mul (value assignment (node.parent frame))
          (value assignment (pointValue frame node.coordinate)) := by
    simpa [value, Extension.output] using rows
  rw [rowsExact, pointExact]

/-- Every satisfying shared-tensor program computes the independently
recursive equality weight. -/
theorem chiFrom_sound
    {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F)
    (constantOne : assignment frame.one = 1)
    (satisfied : RowsSatisfied frame assignment) :
    forall {remaining : Nat} (vertex : BooleanVertex remaining)
      (start : Nat), start + remaining = variableCount ->
      value assignment (chiFrom frame start vertex) =
        semanticChiFrom frame assignment start vertex
  | 0, .nil, start, _ => value_oneValue frame assignment constantOne
  | remaining + 1, .cons bit tail, start, shape => by
      let node : Node variableCount :=
        { start := start
          remaining := remaining
          exact := by omega
          tail := tail }
      have parentExact :=
        chiFrom_sound frame assignment constantOne satisfied tail
          (start + 1) (by omega)
      have highExact := node_output_sound frame assignment satisfied node
      have highExact' :
          value assignment
              (Extension.output
                (frame.tensorFrame start
                  (NumericBooleanDomain.index tail))) =
            K.mul
              (value assignment (chiFrom frame (start + 1) tail))
              (decodedCoordinateAt frame assignment start) := by
        simpa [node, Node.frame, Node.parent, Node.tailIndex] using highExact
      cases bit with
      | false =>
          simp only [chiFrom, semanticChiFrom, Bool.false_eq_true, if_false]
          rw [value_subValue, highExact', parentExact]
      | true =>
          simp only [chiFrom, semanticChiFrom, if_true]
          rw [highExact', parentExact]

/-- The dynamic point reconstructed from the exact point columns. -/
def decodedPoint {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F) : CubePoint K variableCount where
  coordinates := List.ofFn fun coordinate =>
    ⟨assignment (frame.pointLow coordinate),
     assignment (frame.pointHigh coordinate)⟩
  dimension := by simp

private theorem decodedCoordinateAt_of_lt
    {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F) (index : Nat) (bounded : index < variableCount) :
    decodedCoordinateAt frame assignment index =
      ⟨assignment (frame.pointLow ⟨index, bounded⟩),
       assignment (frame.pointHigh ⟨index, bounded⟩)⟩ := by
  simp [decodedCoordinateAt, bounded]

private theorem semanticChiFrom_eq_coordinates
    {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F) :
    forall {remaining : Nat} (vertex : BooleanVertex remaining)
      (start : Nat),
      semanticChiFrom frame assignment start vertex =
        BooleanVertex.equalityWeightCoordinates ConcreteCarrier.extensionOps
          vertex
          (List.ofFn fun coordinate : Fin remaining =>
            decodedCoordinateAt frame assignment
              (start + coordinate.val))
  | 0, .nil, _ => rfl
  | remaining + 1, .cons bit tail, start => by
      have tailExact := semanticChiFrom_eq_coordinates frame assignment tail
        (start + 1)
      rw [List.ofFn_succ]
      have tailCoordinates :
          (List.ofFn fun coordinate : Fin remaining =>
            decodedCoordinateAt frame assignment
              (start + Fin.succ coordinate)) =
          (List.ofFn fun coordinate : Fin remaining =>
            decodedCoordinateAt frame assignment
              (start + 1 + coordinate.val)) := by
        apply congrArg List.ofFn
        funext coordinate
        apply congrArg (decodedCoordinateAt frame assignment)
        change start + (coordinate.val + 1) = start + 1 + coordinate.val
        omega
      rw [tailCoordinates]
      cases bit with
      | false =>
          simp only [semanticChiFrom,
            BooleanVertex.equalityWeightCoordinates,
            Bool.false_eq_true, if_false, Nat.add_zero]
          rw [← tailExact]
          rw [ConcreteCarrier.derived_sub_eq_concrete_sub]
          exact sub_mul_self _ _
      | true =>
          simp only [semanticChiFrom,
            BooleanVertex.equalityWeightCoordinates, if_true, Nat.add_zero]
          rw [← tailExact]
          exact ConcreteCarrier.extensionLaws.mul_comm _ _

/-- The independently recursive reference is exactly the paper equality
weight at the point decoded from the same columns. -/
theorem semanticChiFrom_eq_equalityWeight
    {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F)
    (vertex : BooleanVertex variableCount) :
    semanticChiFrom frame assignment 0 vertex =
      vertex.equalityWeight ConcreteCarrier.extensionOps
        (decodedPoint frame assignment) := by
  unfold BooleanVertex.equalityWeight decodedPoint
  simpa [decodedCoordinateAt] using
    semanticChiFrom_eq_coordinates frame assignment vertex 0

/-- Headline soundness: every shared tensor leaf is the exact paper equality
weight at the dynamic point carried by the circuit. -/
theorem rows_sound
    {variableCount : Nat} (frame : Frame variableCount)
    (assignment : ColumnId -> F)
    (constantOne : assignment frame.one = 1)
    (satisfied : RowsSatisfied frame assignment)
    (vertex : BooleanVertex variableCount) :
    value assignment (chi frame vertex) =
      vertex.equalityWeight ConcreteCarrier.extensionOps
        (decodedPoint frame assignment) := by
  rw [chi, chiFrom_sound frame assignment constantOne satisfied vertex 0
    (by omega)]
  exact semanticChiFrom_eq_equalityWeight frame assignment vertex

/-- Exact number of internal nodes, and therefore extension
multiplications, in the shared tensor. -/
def multiplicationCount (variableCount : Nat) : Nat := 2 ^ variableCount - 1

/-- Exact shared-tensor R1CS row count. -/
def rowCount (variableCount : Nat) : Nat := 3 * multiplicationCount variableCount

end Nightstream.Implementation.R1CS.Phi81SharedTensorRows
