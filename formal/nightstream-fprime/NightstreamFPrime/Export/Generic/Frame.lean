import NightstreamFPrime.Export.Generic.PieceRows
import NightstreamFPrime.Export.Pilot

/-! The frame of packed Poseidon2 blocks (CW-5.8). A packed block keeps only its 16 output locals
live; the packed rows do not constrain its other 1,080 locals (S-box powers and linear-layer
states), which are dead. `complete` refills each block's locals with the honest trace
(`Pilot.completePermutationInvocationEnv`). When the blocks are separated, no lane reads a block's
locals or a dead local (`Framed`, checked by `checkFrame`), and each block's outputs are the permutation
of its lanes, the completed environment agrees with the original one off the dead locals and every
block's invocation holds there (`complete_sound`). -/

namespace NightstreamFPrime.Export.Generic.Frame

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Gadgets.Poseidon2

/-! ### Avoiding columns -/

/-- `expression` reads no column that `dead` marks. -/
def avoids (dead : Nat → Bool) : Expr → Bool
  | .var column => !dead column
  | .const _ => true
  | .add left right => avoids dead left && avoids dead right
  | .mul left right => avoids dead left && avoids dead right

theorem eval_congr {dead : Nat → Bool} {before after : Env}
    (agree : ∀ column, dead column = false → after column = before column) :
    ∀ expression : Expr, avoids dead expression = true →
      expression.eval after = expression.eval before
  | .var column, avoid => by
      have live : dead column = false := by simpa [avoids] using avoid
      simp only [Expr.eval_var, agree column live]
  | .const _, _ => rfl
  | .add left right, avoid => by
      simp only [avoids, Bool.and_eq_true] at avoid
      simp only [Expr.eval_add, eval_congr agree left avoid.1, eval_congr agree right avoid.2]
  | .mul left right, avoid => by
      simp only [avoids, Bool.and_eq_true] at avoid
      simp only [Expr.eval_mul, eval_congr agree left avoid.1, eval_congr agree right avoid.2]

theorem avoids_subst {dead : Nat → Bool} {bound : Nat} (wires : Nat → Expr)
    (wiresAvoid : ∀ column, column < bound → avoids dead (wires column) = true) :
    ∀ expression : Expr, expression.VarsBelow bound →
      avoids dead (subst wires expression) = true
  | .var column, below => wiresAvoid column below
  | .const _, _ => rfl
  | .add left right, below => by
      show (avoids dead (subst wires left) && avoids dead (subst wires right)) = true
      rw [avoids_subst wires wiresAvoid left below.1, avoids_subst wires wiresAvoid right below.2]
      rfl
  | .mul left right, below => by
      show (avoids dead (subst wires left) && avoids dead (subst wires right)) = true
      rw [avoids_subst wires wiresAvoid left below.1, avoids_subst wires wiresAvoid right below.2]
      rfl

/-- The columns of a linear lane's combination are columns the lane reads. -/
theorem linear?_terms_avoid {dead : Nat → Bool} :
    ∀ (expression : Expr) (combination : SparseCombination),
      linear? expression = some combination → avoids dead expression = true →
        ∀ term ∈ combination.terms, dead term.column = false
  | .var column, combination, found, avoid => by
      simp only [linear?, Option.some.injEq] at found
      subst found
      intro term member
      simp only [List.mem_singleton] at member
      subst member
      simpa [avoids] using avoid
  | .const _, combination, found, _ => by
      simp only [linear?, Option.some.injEq] at found
      subst found
      simp
  | .add left right, combination, found, avoid => by
      simp only [linear?, Option.bind_eq_some_iff, Option.map_eq_some_iff] at found
      obtain ⟨leftCombination, leftFound, rightCombination, rightFound, rfl⟩ := found
      simp only [avoids, Bool.and_eq_true] at avoid
      intro term member
      simp only [plus, List.mem_append] at member
      rcases member with member | member
      · exact linear?_terms_avoid left leftCombination leftFound avoid.1 term member
      · exact linear?_terms_avoid right rightCombination rightFound avoid.2 term member
  | .mul left right, combination, found, avoid => by
      simp only [avoids, Bool.and_eq_true] at avoid
      unfold linear? at found
      cases leftConstant : constant? left with
      | some factor =>
          rw [leftConstant] at found
          simp only [Option.map_eq_some_iff] at found
          obtain ⟨inner, innerFound, rfl⟩ := found
          intro term member
          simp only [scale, List.mem_map] at member
          obtain ⟨original, originalMember, rfl⟩ := member
          exact linear?_terms_avoid right inner innerFound avoid.2 original originalMember
      | none =>
          cases rightConstant : constant? right with
          | some factor =>
              rw [leftConstant, rightConstant] at found
              simp only [Option.map_eq_some_iff] at found
              obtain ⟨inner, innerFound, rfl⟩ := found
              intro term member
              simp only [scale, List.mem_map] at member
              obtain ⟨original, originalMember, rfl⟩ := member
              exact linear?_terms_avoid left inner innerFound avoid.1 original originalMember
          | none =>
              rw [leftConstant, rightConstant] at found
              simp at found

/-! ### Blocks -/

structure Block where
  start : Nat
  lanes : List Expr

def Block.invocation (block : Block) : PermutationInvocation :=
  Pieces.invocation 0 0 block.start block.lanes

@[simp] theorem Block.invocation_witnessStart (block : Block) :
    block.invocation.witnessStart = block.start :=
  rfl

theorem outputLocalStart_eq : (PilotData.circuitPackage ()).permutation.outputLocalStart = 1080 :=
  rfl

/-- The block's 1,096 locals. -/
def Block.covers (block : Block) (column : Nat) : Bool :=
  block.start ≤ column && column < block.start + 1096

/-- The block's dead locals: all but the 16 output lanes from local 1,080. -/
def Block.dead (block : Block) (column : Nat) : Bool :=
  block.start ≤ column && column < block.start + 1080

def deadIn (blocks : List Block) (column : Nat) : Bool :=
  blocks.any fun block => block.dead column

/-- Lane `lane` of the block's invocation, as the block's input combination reads it. -/
def Block.input (block : Block) (env : Env) (lane : Fin 16) : F :=
  (invocationInputCombination block.invocation lane.val).toR1CS.eval env

/-- The block's output lanes in `env`. -/
def Block.output (block : Block) (env : Env) (lane : Fin 16) : F :=
  env (block.start + 1080 + lane.val)

/-- Every block's outputs are the permutation of its inputs. -/
def OutputsHold (blocks : List Block) (env : Env) : Prop :=
  ∀ block ∈ blocks, block.output env = Permutation.runF Permutation.schedule (block.input env)

def Separated (first second : Block) : Prop :=
  first.start + 1096 ≤ second.start ∨ second.start + 1096 ≤ first.start

structure Framed (blocks : List Block) : Prop where
  separated : blocks.Pairwise Separated
  linear : ∀ block ∈ blocks, LinearLanes block.lanes
  lanes : ∀ block ∈ blocks, ∀ lane ∈ block.lanes,
    avoids (fun column => block.covers column || deadIn blocks column) lane = true

/-! ### The check -/

def ordered : List Block → Bool
  | first :: second :: rest =>
      decide (first.start + 1096 ≤ second.start) && ordered (second :: rest)
  | _ => true

def checkFrame (blocks : List Block) : Bool :=
  ordered blocks && blocks.all fun block => block.lanes.all fun lane =>
    (linear? lane).isSome && avoids (fun column => block.covers column || deadIn blocks column) lane

theorem ordered_pairwise : ∀ blocks : List Block, ordered blocks = true →
    blocks.Pairwise fun first second => first.start + 1096 ≤ second.start
  | [], _ => List.Pairwise.nil
  | [_], _ => List.pairwise_singleton _ _
  | first :: second :: rest, check => by
      simp only [ordered, Bool.and_eq_true, decide_eq_true_eq] at check
      have tail := ordered_pairwise (second :: rest) check.2
      refine List.Pairwise.cons ?_ tail
      intro block member
      rcases List.mem_cons.mp member with rfl | member
      · exact check.1
      · have later := (List.pairwise_cons.mp tail).1 block member
        omega

theorem checkFrame_spec {blocks : List Block} (check : checkFrame blocks = true) :
    Framed blocks := by
  simp only [checkFrame, Bool.and_eq_true, List.all_eq_true] at check
  refine ⟨(ordered_pairwise blocks check.1).imp Or.inl, ?_, ?_⟩
  · intro block member
    exact List.all_eq_true.mpr fun lane laneMember => (check.2 block member lane laneMember).1
  · intro block member lane laneMember
    exact (check.2 block member lane laneMember).2

/-! ### The completed environment -/

/-- Each block's locals from its honest trace; every other column unchanged. -/
def complete (blocks : List Block) (env : Env) : Env := fun column =>
  match blocks.find? fun block => block.covers column with
  | some block => Pilot.completePermutationInvocationEnv block.invocation env column
  | none => env column

variable {blocks : List Block}

theorem separated_of_mem (separated : blocks.Pairwise Separated) {first second : Block}
    (firstMember : first ∈ blocks) (secondMember : second ∈ blocks) (different : first ≠ second) :
    Separated first second := by
  induction blocks with
  | nil => simp at firstMember
  | cons head rest inductionHypothesis =>
      rw [List.pairwise_cons] at separated
      rcases List.mem_cons.mp firstMember with rfl | firstRest <;>
        rcases List.mem_cons.mp secondMember with rfl | secondRest
      · exact absurd rfl different
      · exact separated.1 second secondRest
      · exact Or.symm (separated.1 first firstRest)
      · exact inductionHypothesis separated.2 firstRest secondRest

theorem find?_covers (frame : Framed blocks) {block : Block} (member : block ∈ blocks)
    {column : Nat} (covered : block.covers column = true) :
    blocks.find? (fun block => block.covers column) = some block := by
  cases found : blocks.find? (fun block => block.covers column) with
  | none =>
      have := List.find?_eq_none.mp found block member
      simp [covered] at this
  | some other =>
      have otherCovers : other.covers column = true := by simpa using List.find?_some found
      have otherMember : other ∈ blocks := List.mem_of_find?_eq_some found
      by_cases same : other = block
      · rw [same]
      · have separated := separated_of_mem frame.separated otherMember member same
        simp only [Block.covers, Bool.and_eq_true, decide_eq_true_eq] at covered otherCovers
        unfold Separated at separated
        omega

theorem complete_of_uncovered {env : Env} {column : Nat}
    (uncovered : ∀ block ∈ blocks, block.covers column = false) :
    complete blocks env column = env column := by
  have found : blocks.find? (fun block => block.covers column) = none :=
    List.find?_eq_none.mpr fun block member => by simp [uncovered block member]
  simp only [complete, found]

/-- A block's input lane is its lane expression. -/
theorem input_eq (frame : Framed blocks) {block : Block} (member : block ∈ blocks) (env : Env)
    (lane : Fin 16) :
    block.input env lane = (block.lanes.getD lane.val 0).eval env := by
  show ((block.lanes.map fun lane => (linear? lane).getD zeroSparseCombination).getD lane.val
    zeroSparseCombination).toR1CS.eval env = _
  exact (lane_eval block.lanes (frame.linear block member) env lane.val).symm

private theorem getD_avoids (frame : Framed blocks) {block : Block} (member : block ∈ blocks)
    (lane : Nat) :
    avoids (fun column => block.covers column || deadIn blocks column)
      (block.lanes.getD lane 0) = true := by
  by_cases inside : lane < block.lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    exact frame.lanes block member _ (List.getElem_mem inside)
  · rw [List.getD_eq_default _ _ (by omega)]
    rfl

/-- The block's input combinations read no column of the block. -/
theorem inputsOutside (frame : Framed blocks) {block : Block} (member : block ∈ blocks) :
    ∀ lane : Fin 16,
      ∀ term ∈ (invocationInputCombination block.invocation lane.val).toR1CS.terms,
        term.1 < block.invocation.witnessStart ∨
          block.invocation.witnessStart + 1096 ≤ term.1 := by
  intro lane term member'
  simp only [invocationInputCombination, Block.invocation, Pieces.invocation,
    SparseCombination.toR1CS, List.mem_map] at member'
  obtain ⟨original, originalMember, rfl⟩ := member'
  by_cases inside : lane.val < block.lanes.length
  · rw [List.getD_eq_getElem _ _ (by simpa using inside), List.getElem_map] at originalMember
    have isSome := List.all_eq_true.mp (frame.linear block member) _ (List.getElem_mem inside)
    obtain ⟨combination, found⟩ := Option.isSome_iff_exists.mp isSome
    rw [found, Option.getD_some] at originalMember
    have avoid := getD_avoids frame member lane.val
    rw [List.getD_eq_getElem _ _ inside] at avoid
    have live := linear?_terms_avoid _ combination found avoid original originalMember
    simp only [Bool.or_eq_false_iff, Block.covers, Bool.and_eq_false_iff,
      decide_eq_false_iff_not] at live
    simp only [Block.invocation, Pieces.invocation]
    omega
  · rw [List.getD_eq_default _ _ (by simpa using inside)] at originalMember
    simp [zeroSparseCombination] at originalMember

/-- The honest trace of one block, with all other columns unchanged. -/
theorem completed_holds (frame : Framed blocks) {block : Block} (member : block ∈ blocks)
    (env : Env) :
    AgreesOutside env (Pilot.completePermutationInvocationEnv block.invocation env)
        block.start 1096 ∧
      PermutationInvocationHolds (PilotData.circuitPackage ()) block.invocation
        (Pilot.completePermutationInvocationEnv block.invocation env) :=
  Pilot.completePermutationInvocation block.invocation env (inputsOutside frame member)

/-- A block's inputs read the same values in its completed trace as in `env`. -/
theorem completed_input (frame : Framed blocks) {block : Block} (member : block ∈ blocks)
    (env : Env) :
    block.input (Pilot.completePermutationInvocationEnv block.invocation env) =
      block.input env := by
  funext lane
  rw [input_eq frame member, input_eq frame member]
  apply eval_congr (dead := fun column => block.covers column || deadIn blocks column)
  · intro column live
    simp only [Bool.or_eq_false_iff] at live
    apply (completed_holds frame member env).1 column
    simp only [Block.covers, Bool.and_eq_false_iff, decide_eq_false_iff_not] at live
    omega
  · exact getD_avoids frame member lane.val

theorem complete_live (frame : Framed blocks) {env : Env} (outputs : OutputsHold blocks env)
    {column : Nat} (live : deadIn blocks column = false) :
    complete blocks env column = env column := by
  by_cases covered : ∃ block ∈ blocks, block.covers column = true
  · obtain ⟨block, member, covers⟩ := covered
    have notDead : block.dead column = false := by
      simp only [deadIn, List.any_eq_false] at live
      simpa using live block member
    simp only [complete, find?_covers frame member covers]
    simp only [Block.covers, Block.dead, Bool.and_eq_true, Bool.and_eq_false_iff,
      decide_eq_true_eq, decide_eq_false_iff_not] at covers notDead
    obtain ⟨lane, rfl⟩ : ∃ lane : Fin 16, column = block.start + 1080 + lane.val :=
      ⟨⟨column - (block.start + 1080), by omega⟩, by simp only; omega⟩
    have completed := completed_holds frame member env
    have honest := Pilot.canonicalPermutationInvocation_sound block.invocation _ completed.2
    rw [outputLocalStart_eq] at honest
    have expected := outputs block member
    have outputEq := congrFun honest lane
    have expectedEq := congrFun expected lane
    simp only [Block.output] at expectedEq
    rw [expectedEq, ← completed_input frame member env]
    exact outputEq
  · simp only [not_exists, not_and, Bool.not_eq_true] at covered
    exact complete_of_uncovered covered

/-- The completed environment agrees with `env` off the dead locals, and every block's invocation
holds in it. -/
theorem complete_sound (frame : Framed blocks) (env : Env) (outputs : OutputsHold blocks env) :
    (∀ column, deadIn blocks column = false → complete blocks env column = env column) ∧
      ∀ block ∈ blocks, PermutationInvocationHolds (PilotData.circuitPackage ())
        block.invocation (complete blocks env) := by
  refine ⟨fun column live => complete_live frame outputs live, ?_⟩
  intro block member
  have completed := completed_holds frame member env
  apply Pilot.permutationInvocationHolds_of_agreement block.invocation _ _ ?_ ?_ completed.2
  · intro lane
    change block.input (complete blocks env) lane =
      block.input (Pilot.completePermutationInvocationEnv block.invocation env) lane
    rw [completed_input frame member env, input_eq frame member, input_eq frame member]
    apply eval_congr (dead := fun column => block.covers column || deadIn blocks column)
    · intro column live
      simp only [Bool.or_eq_false_iff] at live
      exact complete_live frame outputs live.2
    · exact getD_avoids frame member lane.val
  · intro index below
    have covers : block.covers (block.invocation.witnessStart + index) = true := by
      rw [Block.invocation_witnessStart]
      simp [Block.covers, below]
    simp only [complete, find?_covers frame member covers]

end NightstreamFPrime.Export.Generic.Frame
