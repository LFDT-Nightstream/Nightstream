import Nightstream.Assurance.FPrimeFullHistoryCircuitComplete
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalLinkDrift
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryManifest

/-!
Contract: exact obligation ownership for the captured full-history F-prime
artifact.

The generated manifest is the sole authority for physical row intervals.
This module expands its recursive and terminal NIFS owners to the deepest
generated PiCCS/PiRLC constraint-family ranges, proves exact parent costs and
unique ownership of every physical row, and attaches typed mathematical,
Rust-emitter, and Lean-evidence routes to every leaf.

Zero-cost organizational nodes remain in the hierarchy but own no physical
row. Formula-only estimates have a separate type with no row range. This file
does not regenerate the artifact, select a new encoding, or identify the
captured 257-row terminal link with the current 270-row owner.
-/

namespace Nightstream.Assurance.FPrimeFullHistoryObligationTree

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursiveManifest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest

/-- A row is physically owned by a half-open generated interval. -/
def OwnsRow (range : RowRange) (row : Nat) : Prop :=
  range.rowStart ≤ row ∧ row < range.rowEnd

theorem covers_cons_iff
    {start finish : Nat} {head : RowRange} {tail : List RowRange} :
    covers start finish (head :: tail) = true ↔
      head.rowStart = start ∧
      head.rowStart ≤ head.rowEnd ∧
      covers head.rowEnd finish tail = true := by
  simp [covers, and_assoc]

theorem covers_start_le_finish
    {start finish : Nat} {ranges : List RowRange}
    (exact : covers start finish ranges = true) :
    start ≤ finish := by
  induction ranges generalizing start with
  | nil =>
      simp only [covers, beq_iff_eq] at exact
      omega
  | cons head tail ih =>
      rcases covers_cons_iff.mp exact with
        ⟨headStart, ordered, tailExact⟩
      have tailBound := ih tailExact
      omega

theorem covers_member_start
    {start finish : Nat} {ranges : List RowRange}
    (exact : covers start finish ranges = true)
    {range : RowRange} (member : range ∈ ranges) :
    start ≤ range.rowStart := by
  induction ranges generalizing start with
  | nil => simp at member
  | cons head tail ih =>
      rcases covers_cons_iff.mp exact with
        ⟨headStart, ordered, tailExact⟩
      rcases List.mem_cons.mp member with rfl | inTail
      · omega
      · have tailStart := ih tailExact inTail
        omega

theorem covers_member_end
    {start finish : Nat} {ranges : List RowRange}
    (exact : covers start finish ranges = true)
    {range : RowRange} (member : range ∈ ranges) :
    range.rowEnd ≤ finish := by
  induction ranges generalizing start with
  | nil => simp at member
  | cons head tail ih =>
      rcases covers_cons_iff.mp exact with ⟨_, _, tailExact⟩
      rcases List.mem_cons.mp member with rfl | inTail
      · exact covers_start_le_finish tailExact
      · exact ih tailExact inTail

theorem covers_has_owner
    {start finish : Nat} {ranges : List RowRange}
    (exact : covers start finish ranges = true)
    {row : Nat} (lower : start ≤ row) (upper : row < finish) :
    ∃ range, range ∈ ranges ∧ OwnsRow range row := by
  induction ranges generalizing start with
  | nil =>
      simp only [covers, beq_iff_eq] at exact
      omega
  | cons head tail ih =>
      rcases covers_cons_iff.mp exact with ⟨headStart, _, tailExact⟩
      by_cases inHead : row < head.rowEnd
      · exact ⟨head, List.mem_cons_self, by
          simp only [OwnsRow]
          omega⟩
      · have afterHead : head.rowEnd ≤ row := Nat.le_of_not_gt inHead
        rcases ih tailExact afterHead with ⟨range, member, owned⟩
        exact ⟨range, List.mem_cons_of_mem head member, owned⟩

theorem covers_owners_equal
    {start finish : Nat} {ranges : List RowRange}
    (exact : covers start finish ranges = true)
    {row : Nat} {left right : RowRange}
    (leftMember : left ∈ ranges) (leftOwns : OwnsRow left row)
    (rightMember : right ∈ ranges) (rightOwns : OwnsRow right row) :
    left = right := by
  induction ranges generalizing start left right with
  | nil => simp at leftMember
  | cons head tail ih =>
      rcases covers_cons_iff.mp exact with ⟨_, _, tailExact⟩
      rcases List.mem_cons.mp leftMember with rfl | leftTail
      · rcases List.mem_cons.mp rightMember with rfl | rightTail
        · rfl
        · have rightStart := covers_member_start tailExact rightTail
          simp only [OwnsRow] at leftOwns rightOwns
          omega
      · rcases List.mem_cons.mp rightMember with rfl | rightTail
        · have leftStart := covers_member_start tailExact leftTail
          simp only [OwnsRow] at leftOwns rightOwns
          omega
        · exact ih tailExact leftTail leftOwns rightTail rightOwns

/-- Contiguous half-open ranges provide an exact owner, without enumerating
the rows in the covered interval. -/
theorem covers_has_exact_owner
    {start finish : Nat} {ranges : List RowRange}
    (exact : covers start finish ranges = true)
    {row : Nat} (lower : start ≤ row) (upper : row < finish) :
    ∃ range, (range ∈ ranges ∧ OwnsRow range row) ∧
      ∀ other, other ∈ ranges ∧ OwnsRow other row → other = range := by
  rcases covers_has_owner exact lower upper with ⟨range, member, owned⟩
  refine ⟨range, ⟨member, owned⟩, ?_⟩
  intro other otherFacts
  exact covers_owners_equal exact otherFacts.1 otherFacts.2 member owned

/-- Invocation distinguishes equal stage names appearing in recursive and
terminal folds. -/
inductive Invocation where
  | base
  | recursive
  | global
  | terminal
deriving DecidableEq, Repr

/-- A deepest generated owner in the selected materialized hierarchy. -/
structure Leaf where
  invocation : Invocation
  range : RowRange
deriving DecidableEq, Repr

def leavesAt (invocation : Invocation) (ranges : List RowRange) : List Leaf :=
  ranges.map fun range => { invocation, range }

/-- Recursive owner with NIFS, PiCCS, and PiRLC parents expanded. -/
def recursiveLeaves : List Leaf :=
  leavesAt .recursive (recursiveFamilies.take 2) ++
  leavesAt .recursive recursivePiCcsFamilies ++
  leavesAt .recursive recursivePiRlcFamilies ++
  leavesAt .recursive (recursiveNifsFamilies.drop 2) ++
  leavesAt .recursive (recursiveFamilies.drop 3)

/-- Terminal owner with NIFS, PiCCS, and PiRLC parents expanded. -/
def terminalLeaves : List Leaf :=
  leavesAt .terminal (terminalNifsFamilies.take 1) ++
  leavesAt .terminal terminalPiCcsFamilies ++
  leavesAt .terminal terminalPiRlcFamilies ++
  leavesAt .terminal (terminalNifsFamilies.drop 3) ++
  leavesAt .terminal (terminalFamilies.drop 1)

/-- Complete leaf sequence in physical emission order. -/
def allLeaves : List Leaf :=
  leavesAt .base (topLevelFamilies.take 1) ++
  recursiveLeaves ++
  leavesAt .global ((topLevelFamilies.drop 2).take 1) ++
  terminalLeaves ++
  leavesAt .global (topLevelFamilies.drop 4)

inductive CostKind where
  | materialized
  | zeroCost
  | formulaOnly
deriving DecidableEq, Repr

def Leaf.costKind (leaf : Leaf) : CostKind :=
  if leaf.range.rowCount = 0 then .zeroCost else .materialized

def materializedLeaves : List Leaf :=
  allLeaves.filter fun leaf => leaf.costKind = .materialized

def zeroCostLeaves : List Leaf :=
  allLeaves.filter fun leaf => leaf.costKind = .zeroCost

def materializedLeafRanges : List RowRange :=
  materializedLeaves.map Leaf.range

/-- A formula-only estimate deliberately cannot carry a physical interval. -/
structure FormulaEstimate where
  name : String
  rationale : String
deriving DecidableEq, Repr

/-- Formula-only selector arithmetic is documented outside the materialized
tree and therefore cannot own an artifact row. -/
def formulaOnlyEstimates : List FormulaEstimate :=
  [ { name := "frontends.f_prime.gadget_native.selector_gated"
      rationale := "source estimate; emits no R1CS row" } ]

/-- A generated parent and its immediate generated children. -/
structure Branch where
  parent : RowRange
  children : List RowRange
deriving DecidableEq, Repr

def rootRange : RowRange where
  name := "decider.full_history"
  rowStart := 0
  rowEnd := totalRows
  nonzeroEntries := (topLevelFamilies.map RowRange.nonzeroEntries).sum
  sha256 := totalSha256

/-- Protocol -> phase -> constraint-family hierarchy for the exact artifact. -/
def branches : List Branch :=
  [ { parent := rootRange, children := topLevelFamilies }
  , { parent := topLevelFamilies[1]!, children := recursiveFamilies }
  , { parent := recursiveFamilies[2]!, children := recursiveNifsFamilies }
  , { parent := recursiveNifsFamilies[0]!,
      children := recursivePiCcsFamilies }
  , { parent := recursiveNifsFamilies[1]!,
      children := recursivePiRlcFamilies }
  , { parent := topLevelFamilies[3]!, children := terminalFamilies }
  , { parent := terminalFamilies[0]!, children := terminalNifsFamilies }
  , { parent := terminalNifsFamilies[1]!,
      children := terminalPiCcsFamilies }
  , { parent := terminalNifsFamilies[2]!,
      children := terminalPiRlcFamilies } ]

def Branch.Exact (branch : Branch) : Prop :=
  covers branch.parent.rowStart branch.parent.rowEnd branch.children = true ∧
    (branch.children.map RowRange.rowCount).sum = branch.parent.rowCount

instance (branch : Branch) : Decidable branch.Exact := by
  unfold Branch.Exact
  infer_instance

/-- Every hierarchy parent is exactly the sum of its immediate children. -/
theorem every_parent_cost_exact :
    ∀ branch ∈ branches, branch.Exact := by
  intro branch member
  simp only [branches, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    decide

/-- The hierarchy has the one root and eight generated nested partitions. -/
theorem exact_hierarchy_shape :
    branches.map (fun branch => branch.parent.name) =
      [ "decider.full_history"
      , "decider.step.recursive"
      , "fprime.recursive.nifs"
      , "nifs.pi_ccs"
      , "nifs.pi_rlc"
      , "decider.terminal_fold"
      , "terminal.nifs"
      , "nifs.pi_ccs"
      , "nifs.pi_rlc" ] := by
  decide

/-- Removing the two zero-width organizational leaves preserves exact
physical coverage. -/
theorem materialized_leaf_ranges_cover :
    covers 0 totalRows materializedLeafRanges = true := by
  decide

theorem materialized_leaf_cost :
    (materializedLeafRanges.map RowRange.rowCount).sum = totalRows := by
  decide

theorem zero_cost_nodes_exact :
    zeroCostLeaves.map (fun leaf =>
        (leaf.invocation, leaf.range.name, leaf.range.rowCount)) =
      [ (.recursive, "nifs.pi_ccs.running_authority", 0)
      , (.recursive, "fprime.recursive.nebula", 0) ] := by
  decide

theorem zero_cost_row_counts :
    zeroCostLeaves.map (fun leaf => leaf.range.rowCount) = [0, 0] := by
  decide

theorem exact_leaf_census :
    allLeaves.length = 61 ∧
      materializedLeaves.length = 59 ∧
      zeroCostLeaves.length = 2 := by
  decide

theorem materialized_leaf_ranges_nodup :
    materializedLeafRanges.Nodup := by
  decide

/-- Every one of the 4,193,134 physical rows has exactly one deepest leaf. -/
theorem every_materialized_row_has_exactly_one_leaf
    {row : Nat} (inProgram : row < totalRows) :
    ∃ range, (range ∈ materializedLeafRanges ∧ OwnsRow range row) ∧
      ∀ other,
        other ∈ materializedLeafRanges ∧ OwnsRow other row → other = range :=
  covers_has_exact_owner materialized_leaf_ranges_cover
    (Nat.zero_le row) inProgram

/-- No owned row lies outside the generated program interval. -/
theorem no_row_outside_materialized_leaves
    {leaf : Leaf} (member : leaf ∈ materializedLeaves)
    {row : Nat} (owned : OwnsRow leaf.range row) :
    row < totalRows := by
  have rangeMember : leaf.range ∈ materializedLeafRanges :=
    List.mem_map.mpr ⟨leaf, member, rfl⟩
  have endBound :=
    covers_member_end materialized_leaf_ranges_cover rangeMember
  exact Nat.lt_of_lt_of_le owned.2 endBound

inductive ProtocolObligation where
  | fPrimeTransition
  | piCcs
  | piRlc
  | piDec
  | nifsPointBinding
  | stateContinuity
  | publicEncoding
  | terminalCe
deriving DecidableEq, Repr

def Leaf.protocolObligation (leaf : Leaf) : ProtocolObligation :=
  if leaf.range ∈ recursivePiCcsFamilies ∨
      leaf.range ∈ terminalPiCcsFamilies then
    .piCcs
  else if leaf.range ∈ recursivePiRlcFamilies ∨
      leaf.range ∈ terminalPiRlcFamilies then
    .piRlc
  else if leaf.range.name = "nifs.pi_dec" then
    .piDec
  else if leaf.range.name = "nifs.point_binding" then
    .nifsPointBinding
  else if leaf.range.name = "decider.state_link" ∨
      leaf.range.name = "decider.terminal_continuity" then
    .stateContinuity
  else if leaf.range.name = "decider.public_pins" then
    .publicEncoding
  else if leaf.range.name = "decider.terminal_ce" then
    .terminalCe
  else
    .fPrimeTransition

/-- Typed mathematical route; `family` is the exact generated constraint
family, not a prose alias. -/
structure MathematicalRoute where
  protocol : ProtocolObligation
  family : String
deriving DecidableEq, Repr

def Leaf.mathematicalRoute (leaf : Leaf) : MathematicalRoute where
  protocol := leaf.protocolObligation
  family := leaf.range.name

inductive RustOwner where
  | fullHistoryDecider
  | fPrime
  | nifs
  | nifsPiCcs
  | nifsPiRlc
  | nifsPiDec
deriving DecidableEq, Repr

def Leaf.rustOwner (leaf : Leaf) : RustOwner :=
  match leaf.protocolObligation with
  | .piCcs => .nifsPiCcs
  | .piRlc => .nifsPiRlc
  | .piDec => .nifsPiDec
  | .nifsPointBinding => .nifs
  | .fPrimeTransition => .fPrime
  | .stateContinuity => .fullHistoryDecider
  | .publicEncoding => .fullHistoryDecider
  | .terminalCe => .fullHistoryDecider

def RustOwner.sourceModule : RustOwner → String
  | .fullHistoryDecider => "engine::decider"
  | .fPrime => "paper::f_prime::r1cs"
  | .nifs => "paper::nifs::circuit"
  | .nifsPiCcs => "paper::reductions::pi_ccs_split_nc_circuit"
  | .nifsPiRlc => "paper::reductions::pi_rlc_circuit"
  | .nifsPiDec => "paper::reductions::pi_dec_circuit"

theorem RustOwner.sourceModule_ne_empty (owner : RustOwner) :
    owner.sourceModule ≠ "" := by
  cases owner <;> decide

structure RustEmitterRoute where
  owner : RustOwner
  sourceModule : String
  stagePath : String
deriving DecidableEq, Repr

def Leaf.rustEmitterRoute (leaf : Leaf) : RustEmitterRoute where
  owner := leaf.rustOwner
  sourceModule := leaf.rustOwner.sourceModule
  stagePath := leaf.range.name

inductive LeanEvidence where
  | artifactSoundness
  | artifactCompleteness
  | zeroCostNecessity
  | currentTerminalDrift
deriving DecidableEq, Repr

def LeanEvidence.theoremName : LeanEvidence → String
  | .artifactSoundness =>
      "FPrimeFullHistoryCircuit.fPrimeCircuit_sound_or_bad"
  | .artifactCompleteness =>
      "FPrimeFullHistoryCircuit.fPrimeCircuit_complete"
  | .zeroCostNecessity =>
      "FPrimeFullHistoryObligationTree.zero_cost_nodes_exact"
  | .currentTerminalDrift =>
      "FPrimeFullHistoryTerminalLinkDrift.generatedSnapshotRows_ne_currentPlainOwnerRows"

def Leaf.leanEvidence (leaf : Leaf) : List LeanEvidence :=
  [.artifactSoundness, .artifactCompleteness] ++
    if leaf.costKind = .zeroCost then [.zeroCostNecessity] else []

def LeanEvidence.Statement : LeanEvidence → Prop
  | .artifactSoundness =>
      ∀ (_prime : EuclidPrime goldilocksP)
        {assignment : Nat → Nat}
        (_canonical : ∀ column, assignment column < goldilocksP)
        (_one : assignment 0 = 1)
        (_rows : Satisfies FPrimeFullHistoryRows.fullRows assignment),
        Nightstream.Assurance.ValidExecution
            FPrimeFullHistoryCircuit.Edge
            (FPrimeFullHistoryCircuit.TerminalValid assignment _canonical)
            FPrimeFullHistoryCircuit.initialState
            (FPrimeFullHistoryCircuit.finalState assignment _canonical) 2 ∨
          FPrimeFullHistoryCircuit.BadEvent assignment
  | .artifactCompleteness =>
      ∀ (_prime : EuclidPrime goldilocksP)
        {field : CanonicalU64Complete.FieldInverse}
        {assignment : Nat → Nat}
        (_canonical : ∀ column, assignment column < goldilocksP)
        (_one : assignment 0 = 1)
        (_witness : FPrimeFullHistoryCircuit.CompilerWitness field assignment),
        Satisfies FPrimeFullHistoryRows.fullRows assignment
  | .zeroCostNecessity =>
      zeroCostLeaves.map (fun leaf => leaf.range.rowCount) = [0, 0]
  | .currentTerminalDrift =>
      FPrimeFullHistoryTerminalLink.rows ≠ FPrimeTerminalLink.rows

/-- Every cited Lean route is a theorem in the active tree. The fourth route
keeps the stale-snapshot boundary part of the checked evidence. -/
theorem every_lean_evidence_checked
    (evidence : LeanEvidence) : evidence.Statement := by
  cases evidence with
  | artifactSoundness =>
      exact FPrimeFullHistoryCircuit.fPrimeCircuit_sound_or_bad
  | artifactCompleteness =>
      exact FPrimeFullHistoryCircuit.fPrimeCircuit_complete
  | zeroCostNecessity =>
      exact zero_cost_row_counts
  | currentTerminalDrift =>
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.generatedSnapshotRows_ne_currentPlainOwnerRows

structure CrossLayerMapped (leaf : Leaf) : Prop where
  mathematicalFamily :
    leaf.mathematicalRoute.family = leaf.range.name
  rustStage : leaf.rustEmitterRoute.stagePath = leaf.range.name
  rustSourcePresent : leaf.rustEmitterRoute.sourceModule ≠ ""
  evidencePresent : leaf.leanEvidence ≠ []
  evidenceNamesPresent :
    ∀ evidence ∈ leaf.leanEvidence, evidence.theoremName ≠ ""
  evidenceChecked :
    ∀ evidence ∈ leaf.leanEvidence, evidence.Statement

/-- Every physical or zero-cost leaf has exact mathematical and Rust paths
and at least one kernel-checked Lean refinement/necessity route. -/
theorem every_leaf_cross_layer_mapped :
    ∀ leaf ∈ allLeaves, CrossLayerMapped leaf := by
  intro leaf _
  refine
    { mathematicalFamily := rfl
      rustStage := rfl
      rustSourcePresent := RustOwner.sourceModule_ne_empty leaf.rustOwner
      evidencePresent := by
        simp [Leaf.leanEvidence]
      evidenceNamesPresent := by
        intro evidence _
        cases evidence <;> decide
      evidenceChecked := ?_ }
  intro evidence _
  exact every_lean_evidence_checked evidence

/-- The tree explicitly carries the current 257-versus-270 obstruction; its
closure cannot be misread as current-production row equality. -/
theorem obligation_tree_retains_terminal_drift :
    LeanEvidence.currentTerminalDrift.Statement :=
  every_lean_evidence_checked .currentTerminalDrift

end Nightstream.Assurance.FPrimeFullHistoryObligationTree
