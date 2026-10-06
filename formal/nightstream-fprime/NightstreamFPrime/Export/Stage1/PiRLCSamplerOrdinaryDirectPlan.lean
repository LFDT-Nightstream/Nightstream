import NightstreamFPrime.Export.Stage1.DirectPiDECPrefixPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedGeometry
import NightstreamFPrime.Layout.Stage1.SpartanValues

/-! Resolves the four source families read by the sampler ordinary rows.
Each successful decode must reproduce the exact source column. The matrix
plan reads checked field blocks and the existing Poseidon2 output forms. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCSamplerOrdinaryRetainedBlocks

inductive Location where
  | poseidon (source : Fin sourceCount) (lane : Fin 4)
  | logical (source : Fin sourceCount) (position : Fin logicalCountPerSource)
  | fresh (source : Fin sourceCount) (position : Fin freshCountPerSource)
  | word (source : Fin sourceCount) (position : Fin ringDegree)

namespace Location

def sourceColumn : Location → Nat
  | .poseidon source lane => PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val lane
  | .logical source position => logicalSource source position
  | .fresh source position => freshSource source position
  | .word source position => PiRLCStarts.challengeWordStart source.val + position.val

theorem sourceSupport (location : Location) :
    PiRLCSamplerOrdinaryDirectSource.Source location.sourceColumn := by
  cases location with
  | poseidon source lane => exact .poseidon source.val lane source.isLt
  | logical source position => exact .logical source.val position.val source.isLt position.isLt
  | fresh source position => exact .fresh source.val position.val source.isLt position.isLt
  | word source position => exact .word source.val position.val source.isLt position.isLt

theorem sourceColumn_lt (location : Location) : location.sourceColumn < Spartan.SourceColumnCount :=
  location.sourceSupport.bounded

end Location

private def logicalIndex (column : Nat) : Nat := (column - PiRLCStarts.samplerLogicalStart) / 4267
private def logicalOffset (column : Nat) : Nat := (column - PiRLCStarts.samplerLogicalStart) % 4267

private def poseidonCandidate (column : Nat) : Location :=
  .poseidon ⟨logicalIndex column % sourceCount, Nat.mod_lt _ (by decide)⟩
    ⟨(logicalOffset column - 1080) % 4, Nat.mod_lt _ (by decide)⟩

private def logicalCandidate (column : Nat) : Location :=
  .logical ⟨logicalIndex column % sourceCount, Nat.mod_lt _ (by decide)⟩
    ⟨(logicalOffset column - 2500) % logicalCountPerSource, Nat.mod_lt _ (by decide)⟩

private def freshCandidate (column : Nat) : Location :=
  .fresh ⟨((column - PiRLCStarts.samplerFreshStart) / 144) % sourceCount, Nat.mod_lt _ (by decide)⟩
    ⟨(column - PiRLCStarts.samplerFreshStart) % 144, Nat.mod_lt _ (by decide)⟩

private def wordCandidate (column : Nat) : Location :=
  .word ⟨logicalIndex column % sourceCount, Nat.mod_lt _ (by decide)⟩
    ⟨(logicalOffset column - 4213) % ringDegree, Nat.mod_lt _ (by decide)⟩

private def exactCandidate (column : Nat) (candidate : Location) : Option Location :=
  if candidate.sourceColumn = column then some candidate else none

def classifySource (column : Nat) : Option Location :=
  if PiRLCStarts.samplerFreshStart ≤ column then exactCandidate column (freshCandidate column)
  else if 1080 ≤ logicalOffset column ∧ logicalOffset column < 1084 then
    exactCandidate column (poseidonCandidate column)
  else if 2500 ≤ logicalOffset column ∧ logicalOffset column < 3117 then
    exactCandidate column (logicalCandidate column)
  else exactCandidate column (wordCandidate column)

private theorem exactCandidate_sound {column : Nat} {candidate location : Location}
    (found : exactCandidate column candidate = some location) : location.sourceColumn = column := by
  unfold exactCandidate at found
  split at found
  · rename_i same
    have selected := Option.some.inj found
    subst location
    exact same
  · contradiction

theorem classifySource_sound {column : Nat} {location : Location}
    (found : classifySource column = some location) : location.sourceColumn = column := by
  unfold classifySource at found
  split at found
  · exact exactCandidate_sound found
  · split at found
    · exact exactCandidate_sound found
    · split at found <;> exact exactCandidate_sound found

private theorem scalarIndex (source offset : Nat) (bound : offset < 4267) :
    logicalIndex (PiRLCStarts.samplerLogicalStart + source * 4267 + offset) = source := by
  unfold logicalIndex
  rw [show PiRLCStarts.samplerLogicalStart + source * 4267 + offset -
      PiRLCStarts.samplerLogicalStart = source * 4267 + offset by omega]
  rw [Nat.mul_comm source 4267, Nat.mul_add_div (by decide : 0 < 4267),
    Nat.div_eq_of_lt bound, Nat.add_zero]

private theorem scalarOffset (source offset : Nat) (bound : offset < 4267) :
    logicalOffset (PiRLCStarts.samplerLogicalStart + source * 4267 + offset) = offset := by
  unfold logicalOffset
  rw [show PiRLCStarts.samplerLogicalStart + source * 4267 + offset -
      PiRLCStarts.samplerLogicalStart = source * 4267 + offset by omega]
  exact Nat.mul_add_mod_of_lt bound

private theorem scalarBeforeFresh (source : Fin sourceCount) (offset : Nat) (bound : offset < 4267) :
    PiRLCStarts.samplerLogicalStart + source.val * 4267 + offset < PiRLCStarts.samplerFreshStart := by
  have sourceLt : source.val < 17 := source.isLt
  simp only [PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart,
    PiRLCStarts.samplerLogicalStart, Formal.samplerOffset, Formal.logicalPrivateCount_eq]
  omega

theorem poseidonColumn (source : Fin sourceCount) (lane : Fin 4) :
    (Location.poseidon source lane).sourceColumn =
      PiRLCStarts.samplerLogicalStart + source.val * 4267 + (1080 + lane.val) := by
  simp only [Location.sourceColumn, PiRLCSamplerOrdinaryDirectSource.poseidonSource,
    PiRLCStarts.samplerSourceLogicalStart, SamplerChain.sourceOffset, Sampler.counts.1]
  omega

theorem logicalColumn (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    (Location.logical source position).sourceColumn =
      PiRLCStarts.samplerLogicalStart + source.val * 4267 + (2500 + position.val) := by
  simp only [Location.sourceColumn, logicalSource, PiRLCSamplerOrdinaryDirectSource.coreStart,
    Gadgets.Sampling.WideReduction.Program.coreOffset, PiRLCStarts.rangeLogicalStart,
    PiRLCStarts.samplerSourceLogicalStart, SamplerChain.sourceOffset, Sampler.rangeOffset,
    Sampler.counts.1, Gadgets.Sampling.WideReduction.HintProgram.helperCount_eq]
  omega

theorem wordColumn (source : Fin sourceCount) (position : Fin ringDegree) :
    (Location.word source position).sourceColumn =
      PiRLCStarts.samplerLogicalStart + source.val * 4267 + (4213 + position.val) := by
  simp only [Location.sourceColumn, PiRLCStarts.challengeWordStart_eq,
    PiRLCStarts.samplerLogicalStart, Formal.samplerOffset]
  omega

theorem classifySource_poseidonEntry (source : Fin sourceCount) (lane : Fin 4) :
    classifySource (Location.poseidon source lane).sourceColumn = some (.poseidon source lane) := by
  have bound : 1080 + lane.val < 4267 := by omega
  have offset := scalarOffset source.val (1080 + lane.val) bound
  have index := scalarIndex source.val (1080 + lane.val) bound
  rw [poseidonColumn]
  unfold classifySource
  rw [if_neg (Nat.not_le_of_lt (scalarBeforeFresh source _ bound)), offset,
    if_pos (by omega)]
  have candidate : poseidonCandidate (PiRLCStarts.samplerLogicalStart + source.val * 4267 + (1080 + lane.val)) =
      .poseidon source lane := by
    unfold poseidonCandidate
    apply congrArg₂ Location.poseidon
    · apply Fin.ext
      change logicalIndex _ % sourceCount = source.val
      rw [index, Nat.mod_eq_of_lt source.isLt]
    · apply Fin.ext
      change (logicalOffset _ - 1080) % 4 = lane.val
      rw [offset, Nat.add_sub_cancel_left, Nat.mod_eq_of_lt lane.isLt]
  rw [candidate]
  unfold exactCandidate
  rw [if_pos (poseidonColumn source lane)]

theorem classifySource_logical (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    classifySource (logicalSource source position) = some (.logical source position) := by
  have positionLt : position.val < 617 := position.isLt
  have bound : 2500 + position.val < 4267 := by omega
  have offset := scalarOffset source.val (2500 + position.val) bound
  have index := scalarIndex source.val (2500 + position.val) bound
  change classifySource (Location.logical source position).sourceColumn = _
  rw [logicalColumn]
  unfold classifySource
  rw [if_neg (Nat.not_le_of_lt (scalarBeforeFresh source _ bound)), offset,
    if_neg (by omega), if_pos (by omega)]
  have candidate : logicalCandidate (PiRLCStarts.samplerLogicalStart + source.val * 4267 + (2500 + position.val)) =
      .logical source position := by
    unfold logicalCandidate
    apply congrArg₂ Location.logical
    · apply Fin.ext
      change logicalIndex _ % sourceCount = source.val
      rw [index, Nat.mod_eq_of_lt source.isLt]
    · apply Fin.ext
      change (logicalOffset _ - 2500) % logicalCountPerSource = position.val
      rw [offset, Nat.add_sub_cancel_left, Nat.mod_eq_of_lt position.isLt]
  rw [candidate]
  unfold exactCandidate
  rw [if_pos (logicalColumn source position)]

theorem classifySource_word (source : Fin sourceCount) (position : Fin ringDegree) :
    classifySource (PiRLCStarts.challengeWordStart source.val + position.val) = some (.word source position) := by
  have positionLt : position.val < 54 := position.isLt
  have bound : 4213 + position.val < 4267 := by omega
  have offset := scalarOffset source.val (4213 + position.val) bound
  have index := scalarIndex source.val (4213 + position.val) bound
  change classifySource (Location.word source position).sourceColumn = _
  rw [wordColumn]
  unfold classifySource
  rw [if_neg (Nat.not_le_of_lt (scalarBeforeFresh source _ bound)), offset,
    if_neg (by omega), if_neg (by omega)]
  have candidate : wordCandidate (PiRLCStarts.samplerLogicalStart + source.val * 4267 + (4213 + position.val)) =
      .word source position := by
    unfold wordCandidate
    apply congrArg₂ Location.word
    · apply Fin.ext
      change logicalIndex _ % sourceCount = source.val
      rw [index, Nat.mod_eq_of_lt source.isLt]
    · apply Fin.ext
      change (logicalOffset _ - 4213) % ringDegree = position.val
      rw [offset, Nat.add_sub_cancel_left, Nat.mod_eq_of_lt position.isLt]
  rw [candidate]
  unfold exactCandidate
  rw [if_pos (wordColumn source position)]

theorem classifySource_fresh (source : Fin sourceCount) (position : Fin freshCountPerSource) :
    classifySource (freshSource source position) = some (.fresh source position) := by
  have bound : position.val < 144 := position.isLt
  have column : freshSource source position =
      PiRLCStarts.samplerFreshStart + source.val * 144 + position.val := rfl
  have delta : freshSource source position - PiRLCStarts.samplerFreshStart =
      source.val * 144 + position.val := by rw [column]; omega
  have quotient : (source.val * 144 + position.val) / 144 = source.val := by
    rw [Nat.mul_comm source.val 144, Nat.mul_add_div (by decide : 0 < 144),
      Nat.div_eq_of_lt bound, Nat.add_zero]
  have candidate : freshCandidate (freshSource source position) = .fresh source position := by
    unfold freshCandidate
    apply congrArg₂ Location.fresh
    · apply Fin.ext
      change ((freshSource source position - PiRLCStarts.samplerFreshStart) / 144) % sourceCount = source.val
      rw [delta, quotient, Nat.mod_eq_of_lt source.isLt]
    · apply Fin.ext
      change (freshSource source position - PiRLCStarts.samplerFreshStart) % 144 = position.val
      rw [delta, Nat.mul_add_mod_of_lt bound]
  unfold classifySource
  rw [if_pos (by rw [column]; omega), candidate]
  simp [exactCandidate, Location.sourceColumn]

theorem classifySource_complete {column : Nat}
    (supported : PiRLCSamplerOrdinaryDirectSource.Source column) : classifySource column ≠ none := by
  cases supported with
  | poseidon source lane sourceLt =>
      rw [show classifySource (PiRLCSamplerOrdinaryDirectSource.poseidonSource source lane) =
        some (.poseidon ⟨source, sourceLt⟩ lane) from classifySource_poseidonEntry ⟨source, sourceLt⟩ lane]
      simp
  | logical source position sourceLt positionLt =>
      rw [show classifySource (PiRLCSamplerOrdinaryDirectSource.coreStart source + position) =
        some (.logical ⟨source, sourceLt⟩ ⟨position, positionLt⟩) from
          classifySource_logical ⟨source, sourceLt⟩ ⟨position, positionLt⟩]
      simp
  | fresh source position sourceLt positionLt =>
      rw [show classifySource (PiRLCStarts.rangeFreshStart source + position) =
        some (.fresh ⟨source, sourceLt⟩ ⟨position, positionLt⟩) from
          classifySource_fresh ⟨source, sourceLt⟩ ⟨position, positionLt⟩]
      simp
  | word source position sourceLt positionLt =>
      rw [classifySource_word ⟨source, sourceLt⟩ ⟨position, positionLt⟩]
      simp

def piDecGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    PiDECRetainedGeometry.Geometry program logicalWidth :=
  PiRLCSamplerOrdinaryRetainedGeometry.prefixGeometry geometry

def poseidonGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    PiCCSPoseidonPlan.Geometry program logicalWidth :=
  DirectPiDECPrefixPlan.poseidonGeometry (piDecGeometry geometry)

def piRlcGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    PiRLCRetainedGeometry.Geometry program logicalWidth :=
  DirectPrefixPlan.prefixGeometry (poseidonGeometry geometry)

namespace Location

def poseidonInvocation (source : Fin sourceCount) : Fin PiRLCSamplerPoseidonPlan.invocationCount :=
  PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩

def form {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    Location → SparseForm logicalWidth
  | .poseidon source lane =>
      (PiRLCSamplerPoseidonPlan.interface (poseidonGeometry geometry)).output
        (poseidonInvocation source) (Sampler.rateLane lane)
  | .logical source position =>
      (logicalBlock program).form
        (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart program)
        (PiRLCSamplerOrdinaryRetainedGeometry.logicalFits geometry) (logicalSlot source position)
  | .fresh source position =>
      (freshBlock program).form
        (PiRLCSamplerOrdinaryRetainedGeometry.freshStart program)
        (PiRLCSamplerOrdinaryRetainedGeometry.freshFits geometry) (freshSlot source position)
  | .word source position =>
      (PiRLCRetainedGeometry.challengeBlock program).form
        (PiRLCRetainedGeometry.challengeStart program)
        (PiRLCRetainedGeometry.challengeFits (piRlcGeometry geometry))
        (PiRLCProductSourceBlocks.challengeIndex source position)

end Location

def classifyTarget (column : Nat) : Option Location :=
  match Spartan.spartanToSource column with
  | none => none
  | some source => classifySource source

theorem classifyTarget_complete {column : Nat}
    (support : PiRLCSamplerOrdinaryDirectSource.Target column) :
    ∃ decoded, classifyTarget column = some decoded := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  have complete := classifySource_complete sourceSupport
  cases found : classifySource source with
  | none =>
      rw [found] at complete
      contradiction
  | some location =>
      have sourceBound : source < Spartan.SourceColumnCount := by
        have locationBound := location.sourceColumn_lt
        have owns := classifySource_sound found
        exact Eq.mp (congrArg (fun value => value < Spartan.SourceColumnCount)
          owns) locationBound
      have inverse := Spartan.spartanToSource_sourceToSpartan source sourceBound
      refine ⟨location, ?_⟩
      calc
        classifyTarget (Spartan.sourceToSpartan source) =
            classifySource source := by
          unfold classifyTarget
          rw [inverse]
        _ = some location := found

def resolvedForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (column : Nat) : SparseForm logicalWidth :=
  match classifyTarget column with
  | none => .empty
  | some location => location.form geometry

/-- Assignment-derived source environment for the exact canonical sampler
ordinary rows. Supported columns select a retained form; unsupported columns
are irrelevant to those rows and evaluate to zero. -/
def resolvedEnv {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth) : Env :=
  fun column => (resolvedForm geometry column).eval assignment

def sourceMap {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    SourceCompiler.SourceMap Spartan.spartanColumnCount logicalWidth where
  form := fun column => resolvedForm geometry column.val

@[simp] theorem sourceMap_form_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (column : Fin Spartan.spartanColumnCount) :
    ((sourceMap geometry).form column).eval assignment =
      resolvedEnv geometry assignment column.val := by
  rfl

private theorem preservesCombination
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded Spartan.spartanColumnCount
      combination) :
    OrdinarySourcePlan.SourceMap.PreservesCombination (sourceMap geometry)
      assignment (resolvedEnv geometry assignment) combination bounded := by
  intro term member
  exact sourceMap_form_eval geometry assignment ⟨term.1, bounded term member⟩

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def inputs
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (_relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    (PiRLCSamplerOrdinaryDirectSource.program
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits)).Inputs logicalWidth where
  oneColumn := PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry
  sourceMap := fun _ => sourceMap geometry

theorem inputs_preserve
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth) :
    ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((inputs relation geometry).sourceMap index) assignment
      (resolvedEnv geometry assignment)
      ((PiRLCSamplerOrdinaryDirectSource.program
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits)).row index)
      ((PiRLCSamplerOrdinaryDirectSource.program
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits)).bounded index) := by
  intro index
  exact ⟨
    preservesCombination geometry assignment _ _ ,
    preservesCombination geometry assignment _ _ ,
    preservesCombination geometry assignment _ _ ⟩

/-- Exact row-local preservation for one indexed canonical sampler row. -/
theorem programRow_preserve
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (index : Fin 14943) :
    OrdinarySourcePlan.SourceMap.PreservesRow (sourceMap geometry) assignment
      (resolvedEnv geometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.programRow
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index)
      (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index) := by
  exact ⟨
    preservesCombination geometry assignment _ _,
    preservesCombination geometry assignment _ _,
    preservesCombination geometry assignment _ _ ⟩

/-- Exact sparse forms for one canonical Lean-lowered sampler row. -/
def rowForms
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (_relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (index : Fin 14943) : OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (sourceMap geometry)
    (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry)
    (PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)
    (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)

/-- Canonical direct 4-matrix rows for every sampler ordinary constraint. -/
def plan
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (_relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  OrdinaryRow.planOfForms (by norm_num [Lifecycle.cubeVariables])
    (rowForms _relation geometry)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan
