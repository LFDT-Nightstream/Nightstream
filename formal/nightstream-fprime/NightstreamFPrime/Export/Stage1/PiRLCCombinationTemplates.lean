import NightstreamFPrime.Export.Stage1.CompactRows
import NightstreamFPrime.Layout.PiRLC.v1_1.CombinationCost
import NightstreamFPrime.Layout.Stage1.PiRLCStarts

/-!
Owns the normalized Lean row templates for the repeated PiRLC Phi81
combination constraints.

Each lane has two templates: source zero has a constant-zero prior, while all
later sources read the prior logical output. Every concrete family invocation
is only a column renaming of one of these 108 templates.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationTemplates

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.PiRLC.v1_1

def inputCount : Nat := 110
def challengeInputStart : Nat := 0
def valueInputStart : Nat := 54
def priorInput : Nat := 108
def outputInput : Nat := 109

def challenge (lane : Fin ringDegree) : Expr :=
  Expr.var (challengeInputStart + lane.val) - 2

def value (lane : Fin ringDegree) : Expr :=
  Expr.var (valueInputStart + lane.val)

def prior (firstSource : Bool) : Expr :=
  if firstSource then 0 else Expr.var priorInput

def outputRecipe (firstSource : Bool) (lane : Fin ringDegree) : Expr :=
  prior firstSource +
    CombinationStep.mulExpr challenge value lane

private theorem renameExpr_exprCoeff (column : Nat → Nat)
    (family : Fin ringDegree → Expr) (index : Nat) :
    CompactRows.renameExpr column (CombinationStep.exprCoeff family index) =
      CombinationStep.exprCoeff
        (fun lane => CompactRows.renameExpr column (family lane)) index := by
  unfold CombinationStep.exprCoeff
  split <;> rfl

private theorem renameExpr_foldl (column : Nat → Nat)
    (indices : List Nat) (degree : Nat)
    (left right : Fin ringDegree → Expr) (initial : Expr) :
    CompactRows.renameExpr column
        (indices.foldl (fun accumulated source =>
          if source ≤ degree ∧ degree - source < ringDegree then
            accumulated + CombinationStep.exprCoeff left source *
              CombinationStep.exprCoeff right (degree - source)
          else accumulated) initial) =
      indices.foldl (fun accumulated source =>
          if source ≤ degree ∧ degree - source < ringDegree then
            accumulated +
              CombinationStep.exprCoeff
                  (fun lane => CompactRows.renameExpr column (left lane)) source *
                CombinationStep.exprCoeff
                  (fun lane => CompactRows.renameExpr column (right lane))
                  (degree - source)
          else accumulated) (CompactRows.renameExpr column initial) := by
  induction indices generalizing initial with
  | nil => rfl
  | cons source rest inductionHypothesis =>
      simp only [List.foldl_cons]
      split
      · rw [inductionHypothesis]
        apply congrArg (fun start => rest.foldl (fun accumulated source =>
          if source ≤ degree ∧ degree - source < ringDegree then
            accumulated +
              CombinationStep.exprCoeff
                  (fun lane => CompactRows.renameExpr column (left lane)) source *
                CombinationStep.exprCoeff
                  (fun lane => CompactRows.renameExpr column (right lane))
                  (degree - source)
          else accumulated) start)
        simp only [CompactRows.renameExpr]
        rw [renameExpr_exprCoeff, renameExpr_exprCoeff]
        rfl
      · exact inductionHypothesis initial

private theorem renameExpr_rawExpr (column : Nat → Nat)
    (left right : Fin ringDegree → Expr) (degree : Nat) :
    CompactRows.renameExpr column
        (CombinationStep.rawExpr left right degree) =
      CombinationStep.rawExpr
        (fun lane => CompactRows.renameExpr column (left lane))
        (fun lane => CompactRows.renameExpr column (right lane)) degree := by
  unfold CombinationStep.rawExpr
  exact renameExpr_foldl column (List.range ringDegree) degree left right 0

theorem renameExpr_mulExpr (column : Nat → Nat)
    (left right : Fin ringDegree → Expr) (lane : Fin ringDegree) :
    CompactRows.renameExpr column (CombinationStep.mulExpr left right lane) =
      CombinationStep.mulExpr
        (fun current => CompactRows.renameExpr column (left current))
        (fun current => CompactRows.renameExpr column (right current)) lane := by
  unfold CombinationStep.mulExpr
  split <;> split <;>
    simp [CompactRows.renameExpr, CompactRows.renameExpr_sub,
      renameExpr_rawExpr] <;> rfl

def template (firstSource : Bool) (lane : Fin ringDegree) :
    CompactRowTemplate :=
  CompactRows.compactTemplate inputCount outputInput
    (outputRecipe firstSource lane)

def firstTemplates : List CompactRowTemplate :=
  (List.finRange ringDegree).map (template true)

def laterTemplates : List CompactRowTemplate :=
  (List.finRange ringDegree).map (template false)

def templates : List CompactRowTemplate := firstTemplates ++ laterTemplates

def templateIndex (source lane : Nat) : Nat :=
  (if source = 0 then 0 else ringDegree) + lane

theorem constraint_varsBelow (firstSource : Bool)
    (lane : Fin ringDegree) :
    (Expr.var outputInput - outputRecipe firstSource lane).VarsBelow
      inputCount := by
  apply Expr.VarsBelow.sub
  · change outputInput < inputCount
    norm_num [outputInput, inputCount]
  · unfold outputRecipe
    apply Expr.VarsBelow.add
    · unfold prior
      split
      · trivial
      · change priorInput < inputCount
        norm_num [priorInput, inputCount]
    · apply CombinationStep.mulExpr_varsBelow
      · intro current
        unfold challenge
        apply Expr.VarsBelow.sub
        · change challengeInputStart + current.val < inputCount
          have currentBound : current.val < 54 := by
            simpa [ringDegree] using current.isLt
          norm_num [challengeInputStart, inputCount]
          omega
        · trivial
      · intro current
        unfold value
        change valueInputStart + current.val < inputCount
        have currentBound : current.val < 54 := by
          simpa [ringDegree] using current.isLt
        norm_num [valueInputStart, inputCount]
        omega

set_option maxRecDepth 100000 in -- fixed-size: 54 normalized lane templates
theorem template_localColumnCount (firstSource : Bool)
    (lane : Fin ringDegree) :
    (template firstSource lane).localColumnCount =
      CombinationStep.laneFreshCount lane := by
  fin_cases lane <;> cases firstSource <;> rfl

theorem template_rows_length (firstSource : Bool)
    (lane : Fin ringDegree) :
    (template firstSource lane).rows.length =
      CombinationStep.laneFreshCount lane + 1 := by
  have generic : (template firstSource lane).rows.length =
      (template firstSource lane).localColumnCount + 1 := by
    unfold template CompactRows.compactTemplate
    simp only [List.length_map]
    unfold R1CS.lowerGenericConstraint
    rw [List.length_append, R1CS.lowerExpression_rows_length]
    simp
  rw [generic, template_localColumnCount]

@[simp] theorem firstTemplates_length : firstTemplates.length = 54 := by
  simp [firstTemplates, ringDegree]

@[simp] theorem laterTemplates_length : laterTemplates.length = 54 := by
  simp [laterTemplates, ringDegree]

@[simp] theorem templates_length : templates.length = 108 := by
  simp [templates]

theorem laneFreshCount_sum :
    (List.ofFn CombinationStep.laneFreshCount).sum = 8100 := by
  rfl

theorem laneRowCount_sum :
    (List.ofFn fun lane : Fin ringDegree =>
      CombinationStep.laneFreshCount lane + 1).sum = 8154 := by
  rfl

end NightstreamFPrime.Export.Stage1.PiRLCCombinationTemplates
