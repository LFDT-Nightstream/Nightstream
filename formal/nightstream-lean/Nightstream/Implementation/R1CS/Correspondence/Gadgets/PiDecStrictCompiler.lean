import Nightstream.Implementation.R1CS.Core.CheckedProgram

/-!
Contract: semantic compiler model for the production strict-PiDEC gadget.

The input is an explicit CE-claim wire layout.  `instructions` reconstructs
the rows emitted by `enforce_dec_v_strict` for the production radix-two
profile.  `Accepted` is deliberately stated as claim-level equations and
canonicality conditions, independently of R1CS row satisfaction.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictCompiler

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

structure CommitmentLayout where
  dCol : Nat
  kappaCol : Nat
  dataCols : List Nat
deriving DecidableEq, Repr

structure AdvLayout where
  ops : CommitmentLayout
  is : CommitmentLayout
  fs : CommitmentLayout
deriving DecidableEq, Repr

structure ClaimLayout where
  commitment : CommitmentLayout
  adv : Option AdvLayout
  xActiveCols : List Nat
  xInactiveCol : Nat
  xRows : Nat
  xWidth : Nat
  xRowsCol : Nat
  xWidthCol : Nat
  mIn : Nat
  mInCol : Nat
  yRingCols : List (List Nat)
  ctCols : List (Nat × Nat)
  rCols : List (Nat × Nat)
  sColCols : List (Nat × Nat)
  foldDigestCols : List Nat
deriving DecidableEq, Repr

structure Layout where
  radix : Nat
  ringDimension : Nat
  extensionLimbs : Nat
  firstAllocatedColumn : Nat
  parent : ClaimLayout
  children : List ClaimLayout
deriving DecidableEq, Repr

def activeColumns (layout : Layout) : Nat :=
  (layout.parent.mIn + layout.ringDimension - 1) / layout.ringDimension

def equalityCheck (lhs rhs : Nat) : Instruction :=
  .check ⟨[(lhs, 1), (rhs, goldilocksP - 1)], [(0, 1)], []⟩

def zeroCheck (column : Nat) : Instruction :=
  .check ⟨[(column, 1)], [(0, 1)], []⟩

def radixPowers (radix count : Nat) : List Nat :=
  (List.range count).map (fun exponent => radix ^ exponent % goldilocksP)

def recompositionCheck
    (parent : Nat) (children powers : List Nat) : Instruction :=
  .check ⟨
    (parent, 1) :: (children.zip powers).map
      (fun pair => (pair.1, negCoeff pair.2)),
    [(0, 1)], []⟩

def dataRecomposition
    (parent : List Nat) (children : List (List Nat))
    (powers : List Nat) : List Instruction :=
  (List.range parent.length).map fun lane =>
    recompositionCheck (parent.getD lane 0)
      (children.map fun child => child.getD lane 0) powers

def advCoordinateInstructions
    (parent : CommitmentLayout) (children : List CommitmentLayout)
    (powers : List Nat) : List Instruction :=
  children.flatMap (fun child =>
      [equalityCheck parent.dCol child.dCol,
       equalityCheck parent.kappaCol child.kappaCol]) ++
    dataRecomposition parent.dataCols (children.map (·.dataCols)) powers

def advInstructions
    (parent : Option AdvLayout) (children : List (Option AdvLayout))
    (powers : List Nat) : List Instruction :=
  match parent with
  | none => []
  | some parent =>
      let children := children.filterMap id
      advCoordinateInstructions parent.ops (children.map (·.ops)) powers ++
      advCoordinateInstructions parent.is (children.map (·.is)) powers ++
      advCoordinateInstructions parent.fs (children.map (·.fs)) powers

def xColumn (layout : Layout) (claim : ClaimLayout) (row column : Nat) : Nat :=
  if column < activeColumns layout then
    claim.xActiveCols.getD
      (row * activeColumns layout + column) 0
  else claim.xInactiveCol

def activeXColumns (layout : Layout) (claim : ClaimLayout) : List Nat :=
  (List.range claim.xRows).flatMap fun row =>
    (List.range (activeColumns layout)).map fun column =>
      xColumn layout claim row column

def inactiveXColumns (layout : Layout) (claim : ClaimLayout) : List Nat :=
  (List.range claim.xRows).flatMap fun row =>
    (List.range (claim.xWidth - activeColumns layout)).map fun offset =>
      xColumn layout claim row (activeColumns layout + offset)

def uniqueAux : List Nat → List Nat → List Nat
  | _, [] => []
  | seen, head :: tail =>
      if head ∈ seen then uniqueAux seen tail
      else head :: uniqueAux (head :: seen) tail

def unique (values : List Nat) : List Nat := uniqueAux [] values

def xRecompositionInstructions
    (layout : Layout) (powers : List Nat) : List Instruction :=
  (List.range layout.parent.xRows).flatMap fun row =>
    (List.range (activeColumns layout)).map fun column =>
      recompositionCheck (xColumn layout layout.parent row column)
        (layout.children.map fun child => xColumn layout child row column) powers

def yRecompositionInstructions
    (layout : Layout) (powers : List Nat) : List Instruction :=
  (List.range layout.parent.yRingCols.length).flatMap fun row =>
    let parent := layout.parent.yRingCols.getD row []
    (List.range parent.length).map fun lane =>
      recompositionCheck (parent.getD lane 0)
        (layout.children.map fun child =>
          (child.yRingCols.getD row []).getD lane 0) powers

def shapeInstructions (layout : Layout) : List Instruction :=
  layout.children.flatMap fun child =>
    [equalityCheck layout.parent.commitment.dCol child.commitment.dCol,
     equalityCheck layout.parent.commitment.kappaCol child.commitment.kappaCol,
     equalityCheck layout.parent.xRowsCol child.xRowsCol,
     equalityCheck layout.parent.xWidthCol child.xWidthCol,
     equalityCheck layout.parent.mInCol child.mInCol]

def pairEqualityInstructions
    (parent : List (Nat × Nat)) (children : List (List (Nat × Nat))) :
    List Instruction :=
  children.flatMap fun child =>
    (parent.zip child).flatMap fun pair =>
      [equalityCheck pair.1.1 pair.2.1,
       equalityCheck pair.1.2 pair.2.2]

def inactiveInstructions (layout : Layout) : List Instruction :=
  (layout.parent :: layout.children).flatMap fun claim =>
    (unique (inactiveXColumns layout claim)).map zeroCheck

def centeredUnitInstructions (column output : Nat) : List Instruction :=
  [.define ⟨output,
      .product [(column, 1), (0, 1)] [(column, 1)]⟩,
   .check ⟨[(output, 1)],
      [(column, 1), (0, goldilocksP - 1)], []⟩]

def alphabetFrom : Nat → List Nat → List Instruction
  | _, [] => []
  | output, column :: tail =>
      centeredUnitInstructions column output ++ alphabetFrom (output + 1) tail

def alphabetInstructions (layout : Layout) : List Instruction :=
  alphabetFrom layout.firstAllocatedColumn
    (layout.children.flatMap (activeXColumns layout))

def ctInstructions (layout : Layout) : List Instruction :=
  (layout.parent :: layout.children).flatMap fun claim =>
    (claim.ctCols.zip claim.yRingCols).flatMap fun pair =>
      [equalityCheck pair.1.1 (pair.2.getD 0 0),
       equalityCheck pair.1.2 (pair.2.getD 1 0)]

def paddingInstructions (layout : Layout) : List Instruction :=
  (layout.parent :: layout.children).flatMap fun claim =>
    claim.yRingCols.flatMap fun row =>
      (row.drop (layout.ringDimension * layout.extensionLimbs)).map zeroCheck

def foldDigestInstructions (layout : Layout) : List Instruction :=
  layout.children.flatMap fun child =>
    (child.foldDigestCols.zip layout.parent.foldDigestCols).map fun pair =>
      equalityCheck pair.1 pair.2

def centeredUnitCheckRow (column output : Nat) : Row :=
  ⟨[(output, 1)], [(column, 1), (0, goldilocksP - 1)], []⟩

def alphabetCheckRowsFrom : Nat → List Nat → List Row
  | _, [] => []
  | output, column :: tail =>
      centeredUnitCheckRow column output ::
        alphabetCheckRowsFrom (output + 1) tail

def alphabetCheckRows (layout : Layout) : List Row :=
  alphabetCheckRowsFrom layout.firstAllocatedColumn
    (layout.children.flatMap (activeXColumns layout))

/-- Named-in-order compiler groups. Their flattening is the emitted program. -/
def groups (layout : Layout) : List (List Instruction) :=
  let powers := radixPowers layout.radix layout.children.length
  [dataRecomposition layout.parent.commitment.dataCols
      (layout.children.map (·.commitment.dataCols)) powers ++
      advInstructions layout.parent.adv (layout.children.map (·.adv)) powers,
   xRecompositionInstructions layout powers,
   yRecompositionInstructions layout powers,
   shapeInstructions layout,
   pairEqualityInstructions layout.parent.rCols
      (layout.children.map (·.rCols)),
   pairEqualityInstructions layout.parent.sColCols
      (layout.children.map (·.sColCols)),
   inactiveInstructions layout,
   alphabetInstructions layout,
   ctInstructions layout,
   paddingInstructions layout,
   foldDigestInstructions layout]

/-- Exact instruction schedule of production strict PiDEC at radix two. -/
def instructions (layout : Layout) : List Instruction :=
  (groups layout).flatten

def rows (layout : Layout) : List Row := CheckedProgram.rows (instructions layout)

/-- Assertion projection reconstructed independently from the mixed-program
normalizer. A concrete artifact proves this list equals `checks instructions`. -/
def checkRows (layout : Layout) : List Row :=
  let powers := radixPowers layout.radix layout.children.length
  CheckedProgram.rows
      (dataRecomposition layout.parent.commitment.dataCols
        (layout.children.map (·.commitment.dataCols)) powers ++
       advInstructions layout.parent.adv (layout.children.map (·.adv)) powers) ++
    CheckedProgram.rows (xRecompositionInstructions layout powers) ++
    CheckedProgram.rows (yRecompositionInstructions layout powers) ++
    CheckedProgram.rows (shapeInstructions layout) ++
    CheckedProgram.rows (pairEqualityInstructions layout.parent.rCols
      (layout.children.map (·.rCols))) ++
    CheckedProgram.rows (pairEqualityInstructions layout.parent.sColCols
      (layout.children.map (·.sColCols))) ++
    CheckedProgram.rows (inactiveInstructions layout) ++
    alphabetCheckRows layout ++
    CheckedProgram.rows (ctInstructions layout) ++
    CheckedProgram.rows (paddingInstructions layout) ++
    CheckedProgram.rows (foldDigestInstructions layout)

/-- Claim-level equality of one parent lane with its radix-weighted children. -/
def Recomposes
    (assignment : Nat → Nat) (parent : Nat) (children powers : List Nat) : Prop :=
  assignment parent = lcEval assignment (children.zip powers)

instance (assignment : Nat → Nat) (parent : Nat)
    (children powers : List Nat) :
    Decidable (Recomposes assignment parent children powers) := by
  unfold Recomposes
  infer_instance

def AllRecompose
    (assignment : Nat → Nat) (parent : List Nat)
    (children : List (List Nat)) (powers : List Nat) : Prop :=
  ∀ lane, lane < parent.length →
    Recomposes assignment (parent.getD lane 0)
      (children.map fun child => child.getD lane 0) powers

def CommitmentAccepted
    (assignment : Nat → Nat) (powers : List Nat)
    (parent : CommitmentLayout) (children : List CommitmentLayout) : Prop :=
  (∀ child ∈ children,
      assignment parent.dCol = assignment child.dCol ∧
      assignment parent.kappaCol = assignment child.kappaCol) ∧
    AllRecompose assignment parent.dataCols (children.map (·.dataCols)) powers

def AdvAccepted
    (assignment : Nat → Nat) (powers : List Nat)
    (parent : Option AdvLayout) (children : List (Option AdvLayout)) : Prop :=
  match parent with
  | none => ∀ child ∈ children, child = none
  | some parent =>
      ∃ concrete : List AdvLayout,
        children = concrete.map some ∧
        CommitmentAccepted assignment powers parent.ops (concrete.map (·.ops)) ∧
        CommitmentAccepted assignment powers parent.is (concrete.map (·.is)) ∧
        CommitmentAccepted assignment powers parent.fs (concrete.map (·.fs))

def EqualPairs
    (assignment : Nat → Nat) (parent child : List (Nat × Nat)) : Prop :=
  ∀ pair ∈ parent.zip child,
    assignment pair.1.1 = assignment pair.2.1 ∧
      assignment pair.1.2 = assignment pair.2.2

def CenteredUnit (value : Nat) : Prop :=
  value = 0 ∨ value = 1 ∨ value = goldilocksP - 1

instance (value : Nat) : Decidable (CenteredUnit value) := by
  unfold CenteredUnit
  infer_instance

/-- Host-side fixed-shape obligations checked before the compiler indexes the
wire arrays. They contain no verifier acceptance conclusion. -/
structure ShapeValid (layout : Layout) : Prop where
  ringPositive : 0 < layout.ringDimension
  powersCanonical : ∀ coefficient ∈
      radixPowers layout.radix layout.children.length,
    0 < coefficient ∧ coefficient < goldilocksP
  commitmentLengths : ∀ child ∈ layout.children,
    child.commitment.dataCols.length = layout.parent.commitment.dataCols.length
  xShapes : ∀ child ∈ layout.children,
    child.xRows = layout.parent.xRows ∧
    child.xWidth = layout.parent.xWidth ∧
    child.mIn = layout.parent.mIn
  activeXLengths : ∀ claim ∈ layout.parent :: layout.children,
    claim.xActiveCols.length = claim.xRows * activeColumns layout
  yShapes : ∀ child ∈ layout.children,
    child.yRingCols.length = layout.parent.yRingCols.length ∧
    ∀ row, row < layout.parent.yRingCols.length →
      (child.yRingCols.getD row []).length =
        (layout.parent.yRingCols.getD row []).length
  rShapes : ∀ child ∈ layout.children,
    child.rCols.length = layout.parent.rCols.length
  sColShapes : ∀ child ∈ layout.children,
    child.sColCols.length = layout.parent.sColCols.length
  ctShapes : ∀ claim ∈ layout.parent :: layout.children,
    claim.ctCols.length = claim.yRingCols.length ∧
    ∀ row ∈ claim.yRingCols, 2 ≤ row.length
  foldDigestShapes : ∀ child ∈ layout.children,
    child.foldDigestCols.length = layout.parent.foldDigestCols.length

/-- Independent strict-PiDEC verifier predicate for a decoded wire layout. -/
structure Accepted (layout : Layout) (assignment : Nat → Nat) : Prop where
  radixTwo : layout.radix = 2
  commitment : AllRecompose assignment layout.parent.commitment.dataCols
    (layout.children.map (·.commitment.dataCols))
    (radixPowers layout.radix layout.children.length)
  adv : AdvAccepted assignment
    (radixPowers layout.radix layout.children.length)
    layout.parent.adv (layout.children.map (·.adv))
  x : ∀ row column,
    row < layout.parent.xRows → column < activeColumns layout →
    Recomposes assignment (xColumn layout layout.parent row column)
      (layout.children.map fun child => xColumn layout child row column)
      (radixPowers layout.radix layout.children.length)
  y : ∀ row lane,
    row < layout.parent.yRingCols.length →
    lane < (layout.parent.yRingCols.getD row []).length →
    Recomposes assignment
      ((layout.parent.yRingCols.getD row []).getD lane 0)
      (layout.children.map fun child =>
        (child.yRingCols.getD row []).getD lane 0)
      (radixPowers layout.radix layout.children.length)
  shape : ∀ child ∈ layout.children,
    assignment layout.parent.commitment.dCol = assignment child.commitment.dCol ∧
    assignment layout.parent.commitment.kappaCol = assignment child.commitment.kappaCol ∧
    assignment layout.parent.xRowsCol = assignment child.xRowsCol ∧
    assignment layout.parent.xWidthCol = assignment child.xWidthCol ∧
    assignment layout.parent.mInCol = assignment child.mInCol
  sameR : ∀ child ∈ layout.children,
    EqualPairs assignment layout.parent.rCols child.rCols
  sameSCol : ∀ child ∈ layout.children,
    EqualPairs assignment layout.parent.sColCols child.sColCols
  inactiveZero : ∀ claim ∈ layout.parent :: layout.children,
    ∀ column ∈ unique (inactiveXColumns layout claim), assignment column = 0
  childCentered : ∀ child ∈ layout.children,
    ∀ column ∈ activeXColumns layout child, CenteredUnit (assignment column)
  ct : ∀ claim ∈ layout.parent :: layout.children,
    ∀ pair ∈ claim.ctCols.zip claim.yRingCols,
      assignment pair.1.1 = assignment (pair.2.getD 0 0) ∧
        assignment pair.1.2 = assignment (pair.2.getD 1 0)
  paddingZero : ∀ claim ∈ layout.parent :: layout.children,
    ∀ row ∈ claim.yRingCols,
    ∀ column ∈ row.drop (layout.ringDimension * layout.extensionLimbs),
      assignment column = 0
  foldDigest : ∀ child ∈ layout.children,
    ∀ pair ∈ child.foldDigestCols.zip layout.parent.foldDigestCols,
      assignment pair.1 = assignment pair.2

/-! ## Executable semantic verifier

This checker evaluates the decoded claim equations above.  It never evaluates
an R1CS row, so compiler soundness may use it as an independent endpoint.
-/

private def allRecomposeCheck (assignment : Nat → Nat)
    (parent : List Nat) (children : List (List Nat))
    (powers : List Nat) : Bool :=
  (List.range parent.length).all fun lane =>
    decide (Recomposes assignment (parent.getD lane 0)
      (children.map fun child => child.getD lane 0) powers)

private theorem allRecomposeCheck_eq_true_iff
    (assignment : Nat → Nat) (parent : List Nat)
    (children : List (List Nat)) (powers : List Nat) :
    allRecomposeCheck assignment parent children powers = true ↔
      AllRecompose assignment parent children powers := by
  simp [allRecomposeCheck, AllRecompose, List.all_eq_true,
    decide_eq_true_eq]

private def commitmentCheck (assignment : Nat → Nat)
    (powers : List Nat) (parent : CommitmentLayout)
    (children : List CommitmentLayout) : Bool :=
  children.all (fun child =>
    decide (assignment parent.dCol = assignment child.dCol) &&
      decide (assignment parent.kappaCol = assignment child.kappaCol)) &&
    allRecomposeCheck assignment parent.dataCols
      (children.map (·.dataCols)) powers

private theorem commitmentCheck_eq_true_iff
    (assignment : Nat → Nat) (powers : List Nat)
    (parent : CommitmentLayout) (children : List CommitmentLayout) :
    commitmentCheck assignment powers parent children = true ↔
      CommitmentAccepted assignment powers parent children := by
  simp [commitmentCheck, CommitmentAccepted, List.all_eq_true,
    Bool.and_eq_true, decide_eq_true_eq, allRecomposeCheck_eq_true_iff]

private def optionPresent {α : Type} : Option α → Bool
  | none => false
  | some _ => true

private theorem all_optionPresent_iff_exists
    {α : Type} (values : List (Option α)) :
    values.all optionPresent = true ↔
      ∃ concrete : List α, values = concrete.map some := by
  induction values with
  | nil => exact ⟨fun _ => ⟨[], rfl⟩, fun _ => rfl⟩
  | cons head tail inductionHypothesis =>
      cases head with
      | none =>
          constructor
          · intro impossible
            simp [optionPresent] at impossible
          · rintro ⟨concrete, equality⟩
            cases concrete <;> simp at equality
      | some value =>
          constructor
          · intro accepted
            have tailAccepted : tail.all optionPresent = true := by
              simpa [optionPresent] using accepted
            rcases inductionHypothesis.mp tailAccepted with ⟨concrete, rfl⟩
            exact ⟨value :: concrete, rfl⟩
          · rintro ⟨concrete, equality⟩
            cases concrete with
            | nil => simp at equality
            | cons first rest =>
                simp only [List.map_cons, List.cons.injEq] at equality
                rcases equality with ⟨headEquality, tailEquality⟩
                have valueEquality : value = first := Option.some.inj headEquality
                subst first
                have tailAll : tail.all optionPresent = true :=
                  inductionHypothesis.mpr ⟨rest, tailEquality⟩
                simpa [optionPresent] using tailAll

private def advCheck (assignment : Nat → Nat) (powers : List Nat)
    (parent : Option AdvLayout) (children : List (Option AdvLayout)) : Bool :=
  match parent with
  | none => children.all fun child => decide (child = none)
  | some parent =>
      let concrete := children.filterMap id
      children.all optionPresent &&
        (commitmentCheck assignment powers parent.ops (concrete.map (·.ops)) &&
        (commitmentCheck assignment powers parent.is (concrete.map (·.is)) &&
          commitmentCheck assignment powers parent.fs (concrete.map (·.fs))))

private theorem filterMap_id_of_map_some {α : Type}
    (values : List α) : (values.map some).filterMap id = values := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [inductionHypothesis]

private theorem advCheck_eq_true_iff
    (assignment : Nat → Nat) (powers : List Nat)
    (parent : Option AdvLayout) (children : List (Option AdvLayout)) :
    advCheck assignment powers parent children = true ↔
      AdvAccepted assignment powers parent children := by
  cases parent with
  | none =>
      simp [advCheck, AdvAccepted, List.all_eq_true, decide_eq_true_eq]
  | some parent =>
      constructor
      · intro accepted
        have present : children.all optionPresent = true := by
          have decoded : children.all optionPresent = true ∧
              (commitmentCheck assignment powers parent.ops
                ((children.filterMap id).map (·.ops)) &&
               (commitmentCheck assignment powers parent.is
                ((children.filterMap id).map (·.is)) &&
                commitmentCheck assignment powers parent.fs
                  ((children.filterMap id).map (·.fs)))) = true := by
            simpa only [advCheck, Bool.and_eq_true] using accepted
          exact decoded.1
        rcases (all_optionPresent_iff_exists children).mp present with
          ⟨concrete, rfl⟩
        have decoded :
            commitmentCheck assignment powers parent.ops
                (concrete.map (·.ops)) = true ∧
              commitmentCheck assignment powers parent.is
                (concrete.map (·.is)) = true ∧
              commitmentCheck assignment powers parent.fs
                (concrete.map (·.fs)) = true := by
          have all : (concrete.map some).all optionPresent = true ∧
              commitmentCheck assignment powers parent.ops
                    (concrete.map (·.ops)) = true ∧
                commitmentCheck assignment powers parent.is
                      (concrete.map (·.is)) = true ∧
                  commitmentCheck assignment powers parent.fs
                    (concrete.map (·.fs)) = true := by
            simpa only [advCheck, Bool.and_eq_true,
              filterMap_id_of_map_some] using accepted
          exact all.2
        exact ⟨concrete, rfl,
          (commitmentCheck_eq_true_iff assignment powers parent.ops _).mp decoded.1,
          (commitmentCheck_eq_true_iff assignment powers parent.is _).mp decoded.2.1,
          (commitmentCheck_eq_true_iff assignment powers parent.fs _).mp decoded.2.2⟩
      · rintro ⟨concrete, rfl, ops, is, fs⟩
        simp only [advCheck, Bool.and_eq_true, filterMap_id_of_map_some]
        exact ⟨by simp [List.all_map, optionPresent],
            (commitmentCheck_eq_true_iff assignment powers parent.ops _).mpr ops,
            (commitmentCheck_eq_true_iff assignment powers parent.is _).mpr is,
            (commitmentCheck_eq_true_iff assignment powers parent.fs _).mpr fs⟩

private def equalPairsCheck (assignment : Nat → Nat)
    (parent child : List (Nat × Nat)) : Bool :=
  (parent.zip child).all fun pair =>
    decide (assignment pair.1.1 = assignment pair.2.1) &&
      decide (assignment pair.1.2 = assignment pair.2.2)

private theorem equalPairsCheck_eq_true_iff
    (assignment : Nat → Nat) (parent child : List (Nat × Nat)) :
    equalPairsCheck assignment parent child = true ↔
      EqualPairs assignment parent child := by
  simp [equalPairsCheck, EqualPairs, List.all_eq_true, Bool.and_eq_true,
    decide_eq_true_eq]

private def centeredUnitCheck (value : Nat) : Bool :=
  decide (value = 0) ||
    (decide (value = 1) || decide (value = goldilocksP - 1))

private theorem centeredUnitCheck_eq_true_iff (value : Nat) :
    centeredUnitCheck value = true ↔ CenteredUnit value := by
  simp [centeredUnitCheck, CenteredUnit, decide_eq_true_eq]

/-- Executable strict-PiDEC verifier on decoded public claims. -/
def check (layout : Layout) (assignment : Nat → Nat) : Bool :=
  let powers := radixPowers layout.radix layout.children.length
  let radix := decide (layout.radix = 2)
  let commitment := allRecomposeCheck assignment
    layout.parent.commitment.dataCols
    (layout.children.map (·.commitment.dataCols)) powers
  let adv := advCheck assignment powers layout.parent.adv
    (layout.children.map (·.adv))
  let x := (List.range layout.parent.xRows).all fun row =>
      (List.range (activeColumns layout)).all fun column =>
        decide (Recomposes assignment
          (xColumn layout layout.parent row column)
          (layout.children.map fun child => xColumn layout child row column)
          powers)
  let y := (List.range layout.parent.yRingCols.length).all fun row =>
      (List.range (layout.parent.yRingCols.getD row []).length).all fun lane =>
        decide (Recomposes assignment
          ((layout.parent.yRingCols.getD row []).getD lane 0)
          (layout.children.map fun child =>
            (child.yRingCols.getD row []).getD lane 0) powers)
  let shape := layout.children.all fun child =>
      decide (assignment layout.parent.commitment.dCol =
        assignment child.commitment.dCol) &&
      decide (assignment layout.parent.commitment.kappaCol =
        assignment child.commitment.kappaCol) &&
      decide (assignment layout.parent.xRowsCol = assignment child.xRowsCol) &&
      decide (assignment layout.parent.xWidthCol = assignment child.xWidthCol) &&
      decide (assignment layout.parent.mInCol = assignment child.mInCol)
  let sameR := layout.children.all fun child =>
    equalPairsCheck assignment layout.parent.rCols child.rCols
  let sameSCol := layout.children.all fun child =>
    equalPairsCheck assignment layout.parent.sColCols child.sColCols
  let inactive := (layout.parent :: layout.children).all fun claim =>
      (unique (inactiveXColumns layout claim)).all fun column =>
        decide (assignment column = 0)
  let centered := layout.children.all fun child =>
      (activeXColumns layout child).all fun column =>
        centeredUnitCheck (assignment column)
  let ct := (layout.parent :: layout.children).all fun claim =>
      (claim.ctCols.zip claim.yRingCols).all fun pair =>
        decide (assignment pair.1.1 = assignment (pair.2.getD 0 0)) &&
        decide (assignment pair.1.2 = assignment (pair.2.getD 1 0))
  let padding := (layout.parent :: layout.children).all fun claim =>
      claim.yRingCols.all fun row =>
        (row.drop (layout.ringDimension * layout.extensionLimbs)).all
          fun column => decide (assignment column = 0)
  let foldDigest := layout.children.all fun child =>
      (child.foldDigestCols.zip layout.parent.foldDigestCols).all fun pair =>
        decide (assignment pair.1 = assignment pair.2)
  [radix, commitment, adv, x, y, shape, sameR, sameSCol, inactive,
    centered, ct, padding, foldDigest].all id

/-- The executable semantic checker accepts exactly the independent
`Accepted` predicate. -/
theorem check_eq_true_iff (layout : Layout) (assignment : Nat → Nat) :
    check layout assignment = true ↔ Accepted layout assignment := by
  simp only [check, List.all_cons, List.all_nil, id_eq, Bool.and_eq_true,
    decide_eq_true_eq, allRecomposeCheck_eq_true_iff,
    advCheck_eq_true_iff, List.all_eq_true, List.mem_range,
    equalPairsCheck_eq_true_iff, centeredUnitCheck_eq_true_iff,
    and_true]
  constructor
  · rintro ⟨radix, commitment, adv, x, y, shape, sameR, sameSCol,
      inactive, centered, ct, padding, foldDigest⟩
    exact ⟨radix, commitment, adv,
      (fun row column rowLt columnLt => x row rowLt column columnLt),
      (fun row lane rowLt laneLt => y row rowLt lane laneLt),
      (fun child childMember =>
        let facts := shape child childMember
        ⟨facts.1.1.1.1, facts.1.1.1.2, facts.1.1.2,
          facts.1.2, facts.2⟩),
      sameR, sameSCol,
      (fun claim claimMember => by
        rcases List.mem_cons.mp claimMember with rfl | childMember
        · exact inactive.1
        · exact inactive.2 claim childMember),
      centered,
      (fun claim claimMember => by
        rcases List.mem_cons.mp claimMember with rfl | childMember
        · exact ct.1
        · exact ct.2 claim childMember),
      (fun claim claimMember => by
        rcases List.mem_cons.mp claimMember with rfl | childMember
        · exact padding.1
        · exact padding.2 claim childMember),
      foldDigest⟩
  · intro accepted
    exact ⟨accepted.radixTwo, accepted.commitment, accepted.adv,
      (fun row rowLt column columnLt => accepted.x row column rowLt columnLt),
      (fun row rowLt lane laneLt => accepted.y row lane rowLt laneLt),
      (fun child childMember =>
        let facts := accepted.shape child childMember
        ⟨⟨⟨⟨facts.1, facts.2.1⟩, facts.2.2.1⟩,
          facts.2.2.2.1⟩, facts.2.2.2.2⟩),
      accepted.sameR, accepted.sameSCol,
      ⟨accepted.inactiveZero layout.parent (by simp),
        fun claim claimMember => accepted.inactiveZero claim (by simp [claimMember])⟩,
      accepted.childCentered,
      ⟨accepted.ct layout.parent (by simp),
        fun claim claimMember => accepted.ct claim (by simp [claimMember])⟩,
      ⟨accepted.paddingZero layout.parent (by simp),
        fun claim claimMember => accepted.paddingZero claim (by simp [claimMember])⟩,
      accepted.foldDigest⟩

end Nightstream.Implementation.R1CS.PiDecStrictCompiler
