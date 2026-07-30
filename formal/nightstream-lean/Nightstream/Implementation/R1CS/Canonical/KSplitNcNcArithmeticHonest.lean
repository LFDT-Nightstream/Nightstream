import Nightstream.Implementation.R1CS.Canonical.KBooleanMleSequentialHonest
import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointSupport
import Nightstream.Implementation.R1CS.Canonical.KStrictNormSequentialHonest

/-!
Contract: constructive completeness for the arithmetic prefix of the
block×lane NC endpoint.

Owns the source MLEs, strict-`b = 2` residuals, and gamma fold.  It does not
own the block/lane selectors or either authoritative endpoint equality.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcNcArithmeticHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private theorem source_positions {shape : SemanticShape} :
    (canonicalFinIndices shape.sourceCount).map
        (fun source => source.val) =
      List.range' 0 (canonicalFinIndices shape.sourceCount).length := by
  rw [canonicalFinIndices_values, canonicalFinIndices_length]
  simp [List.range'_eq_map_range]

private def sourceCoordinates
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (_ : Fin shape.sourceCount) : List Carried :=
  KSplitNcNcEndpoint.laneCoordinates input

theorem mleRows_eq_sequential
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    KSplitNcNcEndpoint.mleRows input =
      KBooleanMleSequentialHonest.rowsFrom
        (canonicalFinIndices shape.sourceCount)
        (KSplitNcNcEndpoint.sourceTable input)
        (sourceCoordinates input) input.frameBase 0 := by
  unfold KSplitNcNcEndpoint.mleRows
  symm
  rw [KBooleanMleSequentialHonest.rowsFrom_eq_flatMap
    (canonicalFinIndices shape.sourceCount)
    (KSplitNcNcEndpoint.sourceTable input)
    (sourceCoordinates input) (fun source => source.val)
    input.frameBase 0 source_positions]
  rfl

def afterMle
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KBooleanMleSequentialHonest.witnessFrom assignment
    (canonicalFinIndices shape.sourceCount)
    (KSplitNcNcEndpoint.sourceTable input)
    (sourceCoordinates input) input.frameBase 0

theorem normRows_eq_sequential
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    KSplitNcNcEndpoint.normRows input =
      KStrictNormSequentialHonest.rowsFrom
        (canonicalFinIndices shape.sourceCount)
        (KSplitNcNcEndpoint.mleOutput input)
        (KSplitNcNcEndpoint.normBase input) 0 := by
  unfold KSplitNcNcEndpoint.normRows
  symm
  rw [KStrictNormSequentialHonest.rowsFrom_eq_flatMap
    (canonicalFinIndices shape.sourceCount)
    (KSplitNcNcEndpoint.mleOutput input)
    (fun source => source.val)
    (KSplitNcNcEndpoint.normBase input) 0 source_positions]
  rfl

def afterNorm
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KStrictNormSequentialHonest.witnessFrom
    (afterMle input assignment)
    (canonicalFinIndices shape.sourceCount)
    (KSplitNcNcEndpoint.mleOutput input)
    (KSplitNcNcEndpoint.normBase input) 0

def witness
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KHornerHonest.hornerWitness (afterNorm input assignment) input.gamma
    (KSplitNcNcEndpoint.mixedBase input)
    (KSplitNcNcEndpoint.normOutputs input) 0

structure SourceBounds
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) : Prop where
  gamma : CarriedBelow input.gamma input.frameBase
  pointLane :
    ∀ coordinate, CarriedBelow (input.pointLane coordinate) input.frameBase
  message :
    ∀ source lane,
      CarriedBelow (input.messageYZcol source lane) input.frameBase

private theorem tables_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (sources : SourceBounds input) :
    ∀ source,
      KBooleanMleSupport.TableBelowBase
        (KSplitNcNcEndpoint.sourceTable input source) input.frameBase := by
  intro source
  unfold KSplitNcNcEndpoint.sourceTable
  apply paddedTable_below
  intro lane
  exact sources.message source lane

private theorem coordinates_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (sources : SourceBounds input) :
    ∀ source,
      KBooleanMleSupport.CoordinatesBelowBase
        (sourceCoordinates input source) input.frameBase := by
  intro source
  unfold sourceCoordinates KSplitNcNcEndpoint.laneCoordinates
  apply coordinates_below_ofFn
  exact sources.pointLane

theorem mleRows_honest
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    Satisfies (KSplitNcNcEndpoint.mleRows input)
      (afterMle input assignment) := by
  rw [mleRows_eq_sequential]
  exact KBooleanMleSequentialHonest.rowsFrom_honest assignment
    (KSplitNcNcEndpoint.sourceTable input) (sourceCoordinates input)
    positive (tables_below input sources) (coordinates_below input sources)
    (canonicalFinIndices shape.sourceCount) 0

theorem mleRows_below_normBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow (KSplitNcNcEndpoint.mleRows input)
      (KSplitNcNcEndpoint.normBase input) := by
  rw [mleRows_eq_sequential]
  have bounded :=
    KBooleanMleSequentialHonest.rowsFrom_below_end
      (KSplitNcNcEndpoint.sourceTable input) (sourceCoordinates input)
      (tables_below input sources) (coordinates_below input sources)
      (canonicalFinIndices shape.sourceCount) 0
  simpa [KBooleanMleSequentialHonest.blockWidth,
    KSplitNcNcEndpoint.normBase, KSplitNcNcEndpoint.rowsPerMle,
    canonicalFinIndices_length] using bounded

private theorem source_table_at_mleBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (sources : SourceBounds input)
    (source : Fin shape.sourceCount) :
    KBooleanMleSupport.TableBelowBase
      (KSplitNcNcEndpoint.sourceTable input source)
      (KSplitNcNcEndpoint.mleBase input source) :=
  KBooleanMleSequentialHonest.tableBelow_mono _
    (tables_below input sources source) (by
      unfold KSplitNcNcEndpoint.mleBase
      omega)

private theorem source_coordinates_at_mleBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (sources : SourceBounds input)
    (source : Fin shape.sourceCount) :
    KBooleanMleSupport.CoordinatesBelowBase
      (KSplitNcNcEndpoint.laneCoordinates input)
      (KSplitNcNcEndpoint.mleBase input source) :=
  KBooleanMleSequentialHonest.coordinatesBelow_mono _
    (coordinates_below input sources source) (by
      unfold KSplitNcNcEndpoint.mleBase
      omega)

private theorem mle_end_le_normBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (source : Fin shape.sourceCount) :
    KSplitNcNcEndpoint.mleBase input source +
        KBooleanMleSequentialHonest.blockWidth domain.laneVariables ≤
      KSplitNcNcEndpoint.normBase input := by
  have sourceBound : source.val + 1 ≤ shape.sourceCount :=
    Nat.succ_le_iff.mpr source.isLt
  have scaled :=
    Nat.mul_le_mul_left
      (KSplitNcNcEndpoint.rowsPerMle domain) sourceBound
  unfold KSplitNcNcEndpoint.rowsPerMle at scaled
  unfold KSplitNcNcEndpoint.mleBase KSplitNcNcEndpoint.normBase
    KSplitNcNcEndpoint.rowsPerMle
    KBooleanMleSequentialHonest.blockWidth
  simp only [Nat.mul_succ] at scaled
  omega

theorem mleOutputs_below_normBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (sources : SourceBounds input) :
    ∀ source,
      CarriedBelow (KSplitNcNcEndpoint.mleOutput input source)
        (KSplitNcNcEndpoint.normBase input) := by
  intro source
  apply boolean_output_below
    (KSplitNcNcEndpoint.sourceTable input source)
    (KSplitNcNcEndpoint.laneCoordinates input)
  · exact source_table_at_mleBase input sources source
  · exact source_coordinates_at_mleBase input sources source
  · exact mle_end_le_normBase input source

theorem normRows_honest
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    Satisfies (KSplitNcNcEndpoint.normRows input)
      (afterNorm input assignment) := by
  rw [normRows_eq_sequential]
  apply KStrictNormSequentialHonest.rowsFrom_honest
  · unfold KSplitNcNcEndpoint.normBase
    omega
  · exact mleOutputs_below_normBase input sources

private theorem mlePrefix_honest
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    Satisfies
      (KSplitNcNcEndpoint.mleRows input ++
        KSplitNcNcEndpoint.normRows input)
      (afterNorm input assignment) := by
  have mleSatisfied := mleRows_honest input assignment positive sources
  have mlePreserved :
      Satisfies (KSplitNcNcEndpoint.mleRows input)
        (afterNorm input assignment) := by
    apply KHornerSupport.satisfies_extend _
      (afterMle input assignment) (afterNorm input assignment)
    · intro row member column mentioned
      symm
      apply KStrictNormSequentialHonest.witnessFrom_off_before
      exact mleRows_below_normBase input positive sources
        row member column mentioned
    · exact mleSatisfied
  have normSatisfied := normRows_honest input assignment positive sources
  intro row member
  exact (List.mem_append.1 member).elim
    (mlePreserved row) (normSatisfied row)

private theorem normRows_below_mixedBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow (KSplitNcNcEndpoint.normRows input)
      (KSplitNcNcEndpoint.mixedBase input) := by
  rw [normRows_eq_sequential]
  have bounded :=
    KStrictNormSequentialHonest.rowsFrom_below_end
      (KSplitNcNcEndpoint.mleOutput input)
      (by unfold KSplitNcNcEndpoint.normBase; omega)
      (mleOutputs_below_normBase input sources)
      (canonicalFinIndices shape.sourceCount) 0
  simpa [KSplitNcNcEndpoint.mixedBase,
    canonicalFinIndices_length] using bounded

private theorem arithmeticPrefix_below_mixedBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow
      (KSplitNcNcEndpoint.mleRows input ++
        KSplitNcNcEndpoint.normRows input)
      (KSplitNcNcEndpoint.mixedBase input) := by
  intro row member column mentioned
  rcases List.mem_append.1 member with inMle | inNorm
  · exact Nat.lt_of_lt_of_le
      (mleRows_below_normBase input positive sources
        row inMle column mentioned)
      (by unfold KSplitNcNcEndpoint.mixedBase; omega)
  · exact normRows_below_mixedBase input positive sources
      row inNorm column mentioned

private theorem norm_output_below_mixedBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (source : Fin shape.sourceCount) :
    CarriedBelow
      (KStrictNorm.output (KSplitNcNcEndpoint.normInput input source))
      (KSplitNcNcEndpoint.mixedBase input) := by
  apply strictNorm_output_below
  change
    KSplitNcNcEndpoint.normBase input + 6 * source.val + 6 ≤
      KSplitNcNcEndpoint.mixedBase input
  have sourceBound : source.val + 1 ≤ shape.sourceCount :=
    Nat.succ_le_iff.mpr source.isLt
  have scaled := Nat.mul_le_mul_left 6 sourceBound
  unfold KSplitNcNcEndpoint.mixedBase
  simp only [Nat.mul_succ] at scaled
  omega

theorem normOutputs_below_mixedBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    ∀ output ∈ KSplitNcNcEndpoint.normOutputs input,
      CarriedBelow output (KSplitNcNcEndpoint.mixedBase input) := by
  intro output member
  rcases List.mem_map.1 member with ⟨source, _, rfl⟩
  exact norm_output_below_mixedBase input source

theorem mixedRows_honest
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    Satisfies (KSplitNcNcEndpoint.mixedRows input)
      (witness input assignment) := by
  unfold KSplitNcNcEndpoint.mixedRows witness
  apply KHornerHonest.hornerWitness_satisfies
  · exact (carried_mono sources.gamma (by
      unfold KSplitNcNcEndpoint.mixedBase
        KSplitNcNcEndpoint.normBase
      omega)).1
  · exact (carried_mono sources.gamma (by
      unfold KSplitNcNcEndpoint.mixedBase
        KSplitNcNcEndpoint.normBase
      omega)).2
  · exact normOutputs_below_mixedBase input

def rows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) : List Row :=
  KSplitNcNcEndpoint.mleRows input ++
    KSplitNcNcEndpoint.normRows input ++
      KSplitNcNcEndpoint.mixedRows input

theorem rows_honest
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    Satisfies (rows input) (witness input assignment) := by
  have prefixSatisfied :=
    mlePrefix_honest input assignment positive sources
  have prefixPreserved :
      Satisfies
        (KSplitNcNcEndpoint.mleRows input ++
          KSplitNcNcEndpoint.normRows input)
        (witness input assignment) := by
    apply KHornerSupport.satisfies_extend _
      (afterNorm input assignment) (witness input assignment)
    · intro row member column mentioned
      symm
      apply KHornerHonest.hornerWitness_off_block
      exact arithmeticPrefix_below_mixedBase input positive sources
        row member column mentioned
    · exact prefixSatisfied
  have mixedSatisfied := mixedRows_honest input assignment positive sources
  intro row member
  rcases List.mem_append.1 member with inPrefix | inMixed
  · exact prefixPreserved row inPrefix
  · exact mixedSatisfied row inMixed

theorem rows_below_equalityBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow (rows input) (KSplitNcNcEndpoint.equalityBase input) := by
  have prefixBelow :
      RowsBelow
        (KSplitNcNcEndpoint.mleRows input ++
          KSplitNcNcEndpoint.normRows input)
        (KSplitNcNcEndpoint.equalityBase input) := by
    intro row member column mentioned
    exact Nat.lt_of_lt_of_le
      (arithmeticPrefix_below_mixedBase input positive sources
        row member column mentioned)
      (by unfold KSplitNcNcEndpoint.equalityBase; omega)
  have mixedBelow :
      RowsBelow (KSplitNcNcEndpoint.mixedRows input)
        (KSplitNcNcEndpoint.equalityBase input) := by
    unfold KSplitNcNcEndpoint.mixedRows
    apply horner_rows_below
    · exact carried_mono sources.gamma (by
        unfold KSplitNcNcEndpoint.equalityBase
          KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
        omega)
    · intro output member
      exact carried_mono
        (normOutputs_below_mixedBase input output member)
        (by unfold KSplitNcNcEndpoint.equalityBase; omega)
    · rw [KSplitNcNcEndpoint.normOutputs_length]
      unfold KSplitNcNcEndpoint.equalityBase
      omega
  intro row member column mentioned
  exact (List.mem_append.1 member).elim
    (fun inPrefix => prefixBelow row inPrefix column mentioned)
    (fun inMixed => mixedBelow row inMixed column mentioned)

theorem mixedOutput_below_equalityBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    CarriedBelow (KSplitNcNcEndpoint.mixedOutput input)
      (KSplitNcNcEndpoint.equalityBase input) := by
  unfold KSplitNcNcEndpoint.mixedOutput
  apply horner_output_below
  · exact fun output member =>
      carried_mono (normOutputs_below_mixedBase input output member)
        (by unfold KSplitNcNcEndpoint.equalityBase; omega)
  · rw [KSplitNcNcEndpoint.normOutputs_length]
    unfold KSplitNcNcEndpoint.equalityBase
    omega

theorem witness_off_source
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    witness input assignment column = assignment column := by
  unfold witness
  rw [KHornerHonest.hornerWitness_off_block _ _ _ _ 0 column (by
    unfold KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
    omega)]
  unfold afterNorm
  rw [KStrictNormSequentialHonest.witnessFrom_off_before
    (afterMle input assignment)
    (KSplitNcNcEndpoint.mleOutput input)
    (canonicalFinIndices shape.sourceCount)
    (KSplitNcNcEndpoint.normBase input) 0 column (by
      unfold KSplitNcNcEndpoint.normBase
      omega)]
  unfold afterMle
  rw [KBooleanMleSequentialHonest.witnessFrom_off_before assignment
    (KSplitNcNcEndpoint.sourceTable input) (sourceCoordinates input)
    (canonicalFinIndices shape.sourceCount) input.frameBase 0
    column (by simpa using below)]

end Nightstream.Implementation.R1CS.Canonical.KSplitNcNcArithmeticHonest
