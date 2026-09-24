import NightstreamFPrime.Export.Stage1.Wide.AssignmentProjection
import NightstreamFPrime.Export.Stage1.Wide.PiDECSource
import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
import NightstreamFPrime.Layout.Stage1.Wide.SourceOrder

/-! Construct the common retained values from the wide physical witness and
application values. Reference source addresses are a view, not a stored array.
Sampler and ring-arithmetic scratch have no source image. -/

namespace NightstreamFPrime.Export.Stage1.Wide.SourceAssignment

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev Program := RetainedLayout.Program

def prefixEnd : Nat := Layout.Stage1.PiRLCInputs.phaseOffset
def productStart : Nat := Layout.Stage1.PiRLCStarts.commitmentLogicalStart
def productEnd : Nat := Layout.Stage1.PiRLCStarts.outputLogicalStart
def suffixStart : Nat := Layout.Stage1.PiDECInputs.proofInputStart

theorem reference_bounds : prefixEnd = 19513117 ∧ productStart = 19776685 ∧
    productEnd = 19829011 ∧ suffixStart = 28421542 ∧ Layout.Stage1.Spartan.SourceColumnCount = 28785018 := by
  exact ⟨rfl, rfl, rfl, rfl, Layout.Stage1.Spartan.sourceColumnCount_eq⟩

/-- The logical product region and the complete PiDEC/transition suffix have
separate images. All other old PiRLC source intervals are discarded. -/
def source? (source : Nat) : Option Nat :=
  if source < prefixEnd then some source
  else if productStart ≤ source ∧ source < productEnd then
    some (Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart + (source - productStart))
  else if suffixStart ≤ source ∧ source < Layout.Stage1.Spartan.SourceColumnCount then
    some (Layout.Stage1.Wide.PiDECInputs.proofInputStart + (source - suffixStart))
  else none

theorem source?_lt (source target : Nat) (mapped : source? source = some target) :
    target < Layout.Stage1.Wide.SourceOrder.sourceWidth := by
  obtain ⟨prefix_eq, product, product_end_eq, suffix, sourceEnd⟩ := reference_bounds
  unfold source? at mapped
  rw [prefix_eq, product, product_end_eq, suffix, sourceEnd] at mapped
  change (if source < 19513117 then some source
    else if 19776685 ≤ source ∧ source < 19829011 then some (19568520 + (source - 19776685))
    else if 28421542 ≤ source ∧ source < 28785018 then some (27496062 + (source - 28421542))
    else none) = some target at mapped
  rw [Layout.Stage1.Wide.SourceOrder.sourceWidth_eq]
  split_ifs at mapped <;> simp only [Option.some.injEq] at mapped <;> omega

theorem source?_prefix (source : Nat) (before : source < prefixEnd) : source? source = some source := by
  exact if_pos before

theorem source?_product (source : Nat) (inside : productStart ≤ source ∧ source < productEnd) :
    source? source = some (Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart + (source - productStart)) := by
  have before : ¬source < prefixEnd := by
    obtain ⟨prefix_eq, start, _, _, _⟩ := reference_bounds
    rw [prefix_eq]; rw [start] at inside; omega
  simp only [source?, if_neg before, if_pos inside]

theorem source?_suffix (source : Nat)
    (inside : suffixStart ≤ source ∧ source < Layout.Stage1.Spartan.SourceColumnCount) :
    source? source = some (Layout.Stage1.Wide.PiDECInputs.proofInputStart + (source - suffixStart)) := by
  obtain ⟨prefix_eq, product, product_end_eq, suffix, _⟩ := reference_bounds
  have before : ¬source < prefixEnd := by rw [prefix_eq]; rw [suffix] at inside; omega
  have outsideProduct : ¬(productStart ≤ source ∧ source < productEnd) := by
    rw [product, product_end_eq]; rw [suffix] at inside; omega
  simp only [source?, if_neg before, if_neg outsideProduct, if_pos inside]

theorem source?_piDec (location : PiDECDirectPlan.Location) :
    source? location.sourceColumn = some (PiDECSource.column location) := by
  have mappedProduct (source target : Nat) (inside : 19776685 ≤ source ∧ source < 19829011)
      (same : 19568520 + (source - 19776685) = target) : source? source = some target := by
    have mapped := source?_product source (by change 19776685 ≤ source ∧ source < 19829011; exact inside)
    exact mapped.trans (congrArg some same)
  have mappedSuffix (source target : Nat) (inside : 28421542 ≤ source ∧ source < 28785018)
      (same : 27496062 + (source - 28421542) = target) : source? source = some target := by
    have mapped := source?_suffix source (by change 28421542 ≤ source ∧ source < 28785018; exact inside)
    exact mapped.trans (congrArg some same)
  cases location with
  | parentCommitment index =>
    have bound : index.val < 1188 := index.isLt
    change source? (19795693 + index.val) = some (19587528 + index.val)
    exact mappedProduct _ _ (by omega) (by omega)
  | parentPublicInput index =>
    have bound : index.val < 270 := index.isLt
    change source? (19801201 + index.val) = some (19593036 + index.val)
    exact mappedProduct _ _ (by omega) (by omega)
  | parentEvalK index =>
    have bound : index.val < 108 := index.isLt
    change source? (19803199 + index.val) = some (19595034 + index.val)
    exact mappedProduct _ _ (by omega) (by omega)
  | parentEvalA index =>
    have bound : index.val < 1512 := index.isLt
    change source? (19827499 + index.val) = some (19619334 + index.val)
    exact mappedProduct _ _ (by omega) (by omega)
  | proof index =>
    have bound : index.val < 49248 := index.isLt
    change source? (28421542 + index.val) = some (27496062 + index.val)
    exact mappedSuffix _ _ (by omega) (by omega)
  | logical index =>
    have bound : index.val < 270 := index.isLt
    change source? (28470790 + index.val) = some (27545310 + index.val)
    exact mappedSuffix _ _ (by omega) (by omega)
  | fresh index =>
    have bound : index.val < 17820 := index.isLt
    change source? (28471060 + index.val) = some (27545580 + index.val)
    exact mappedSuffix _ _ (by omega) (by omega)

/-- Missing reference values are never used as an address. -/
def sourceEnv (env : Env) : Env := fun source => (source? source).map env |>.getD 0

theorem sourceEnv_prefix (env : Env) (source : Nat) (before : source < prefixEnd) :
    sourceEnv env source = env source := by
  simp only [sourceEnv, source?_prefix source before, Option.map_some, Option.getD_some]

theorem sourceEnv_piDec (env : Env) (location : PiDECDirectPlan.Location) :
    sourceEnv env location.sourceColumn = PiDECSource.value env location := by
  simp only [sourceEnv, source?_piDec, Option.map_some, Option.getD_some, PiDECSource.value]

/-- Reuse the already proved private/constant/public permutation. -/
def targetEnv (env : Env) : Env := fun target =>
  if target = Layout.Stage1.Spartan.constantColumn then 1 else
    (Layout.Stage1.Spartan.spartanToSource target).map (sourceEnv env) |>.getD 0

theorem targetEnv_source (env : Env) (source : Nat) (bounded : source < Layout.Stage1.Spartan.SourceColumnCount) :
    targetEnv env (Layout.Stage1.Spartan.sourceToSpartan source) = sourceEnv env source := by
  unfold targetEnv
  rw [if_neg (Layout.Stage1.Spartan.sourceToSpartan_ne_constant source bounded),
    Layout.Stage1.Spartan.spartanToSource_sourceToSpartan source bounded]
  rfl

/-- The reference packet supplies common encoders only. Both obsolete derived
value families are zero; the direct PiRLC constructor owns their replacements. -/
def raw (program : Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F) :
    PerApplicationCanonicalAssignment.RawValues program where
  base := PerApplicationSourceAssignment.ofCompleted program (targetEnv env) application
  groupValue := fun _ _ => 0
  products := fun _ => 0

theorem raw_source (program : Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (source : Nat) (bounded : source < Layout.Stage1.Spartan.SourceColumnCount) :
    PiRLCFirst54DirectPlan.baseEnv program (raw program env application).base source = sourceEnv env source := by
  exact (PerApplicationSourceAssignment.source_ofCompleted program (targetEnv env) application source bounded).trans
    (targetEnv_source env source bounded)

def assignment (program : Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F) :=
  AssignmentProjection.assignment program (raw program env application).assignment

theorem piRlc_complete (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F) :
    (Stage1Plan.piRlc program compiled).RowsZero (assignment program env application) := by
  apply AssignmentProjection.piRlc_complete
  exact PerApplicationCanonicalAssignment.assignment_one (raw program env application)

end NightstreamFPrime.Export.Stage1.Wide.SourceAssignment
