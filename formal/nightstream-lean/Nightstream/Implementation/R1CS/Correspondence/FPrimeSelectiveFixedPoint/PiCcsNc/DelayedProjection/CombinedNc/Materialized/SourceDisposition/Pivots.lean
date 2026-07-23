import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBatchIndex
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.Physical

/-!
Exact rewrite-terminal pivot ownership for the source program.

Owns: the 24-shard pivot projection, its exact 941-column schedule, source-output
membership, distinctness, and disjointness from physical compiler outputs.

Does not own: selected-row satisfaction, protocol acceptance, transcript
authority, commitment binding, costs, or permission to remove rows.

Assurance tier: artifact-checked for the fixed generated production profile
once this leaf validates.
-/

/-!
Emits constraints: none; this module classifies existing rewrite-chain pivots.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_disposition.pivots` | Identify the exact source definitions that terminate rewrite blocks. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-! ## Rewrite-terminal pivot columns -/

/-- The leading source term is the output pivot of a terminal rewrite step.
Empty source forms fail closed by producing no owner. -/
def terminalPivotColumn? (step : RawRewriteStep) : Option Nat :=
  match step.output with
  | .derivedProductSum _ => none
  | .source linear => linear.terms.head?.map fun term => term.column

def terminalPivotColumnsOf (values : List RawRewriteStep) : List Nat :=
  values.filterMap terminalPivotColumn?

def terminalPivotChunks : List (List Nat) := [
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk0.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk1.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk2.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk3.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk4.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk5.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk6.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk7.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk8.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk9.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk10.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk11.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk12.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk13.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk14.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk15.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk16.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk17.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk18.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk19.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk20.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk21.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk22.values,
  terminalPivotColumnsOf Provenance.RewriteSteps.Chunk23.values]

def terminalPivotColumns : List Nat := terminalPivotChunks.flatten

/-- The direct-shard pivot schedule is exactly the generated rewrite stream
projected through the leading-target convention. -/
theorem terminalPivotColumns_exact :
    terminalPivotColumns =
      Provenance.RewriteSteps.values.filterMap terminalPivotColumn? := by
  simp only [terminalPivotColumns, terminalPivotChunks,
    terminalPivotColumnsOf, Provenance.RewriteSteps.values,
    List.flatten_cons, List.flatten_nil, List.filterMap_append,
    List.append_nil, List.append_assoc]

structure PivotShape where
  column : Nat
  sourceOutput : Bool
  notPhysical : Bool
deriving DecidableEq, Repr

def pivotShape (column : Nat) : PivotShape :=
  { column
    sourceOutput := decide (column ∈ SourceExecution.definitionOutputs)
    notPhysical := decide (column ∉ physicalDefinitionOutputs) }

def PivotShapesValid (lower upper count : Nat)
    (values : List PivotShape) : Prop :=
  values.length = count ∧
  (values.map PivotShape.column).Pairwise (fun left right => left < right) ∧
  (∀ shape ∈ values,
    lower ≤ shape.column ∧ shape.column ≤ upper ∧
      shape.sourceOutput = true ∧ shape.notPhysical = true)

instance (lower upper count : Nat) (values : List PivotShape) :
    Decidable (PivotShapesValid lower upper count values) := by
  unfold PivotShapesValid
  infer_instance

private theorem pivotChunk0 : PivotShapesValid 3964260 3964478 64
    ((terminalPivotChunks[0]!).map pivotShape) := by native_decide
private theorem pivotChunk1 : PivotShapesValid 3964484 3971913 64
    ((terminalPivotChunks[1]!).map pivotShape) := by native_decide
private theorem pivotChunk2 : PivotShapesValid 3971919 3986553 64
    ((terminalPivotChunks[2]!).map pivotShape) := by native_decide
private theorem pivotChunk3 : PivotShapesValid 3986559 4001193 64
    ((terminalPivotChunks[3]!).map pivotShape) := by native_decide
private theorem pivotChunk4 : PivotShapesValid 4001199 4074764 64
    ((terminalPivotChunks[4]!).map pivotShape) := by native_decide
private theorem pivotChunk5 : PivotShapesValid 4074768 4074924 64
    ((terminalPivotChunks[5]!).map pivotShape) := by native_decide
private theorem pivotChunk6 : PivotShapesValid 4074928 4075084 64
    ((terminalPivotChunks[6]!).map pivotShape) := by native_decide
private theorem pivotChunk7 : PivotShapesValid 4075088 4075244 64
    ((terminalPivotChunks[7]!).map pivotShape) := by native_decide
private theorem pivotChunk8 : PivotShapesValid 4075248 4075685 49
    ((terminalPivotChunks[8]!).map pivotShape) := by native_decide
private theorem pivotChunk9 : PivotShapesValid 4075686 4076021 8
    ((terminalPivotChunks[9]!).map pivotShape) := by native_decide
private theorem pivotChunk10 : PivotShapesValid 4076022 4076690 13
    ((terminalPivotChunks[10]!).map pivotShape) := by native_decide
private theorem pivotChunk11 : PivotShapesValid 4076691 4077026 8
    ((terminalPivotChunks[11]!).map pivotShape) := by native_decide
private theorem pivotChunk12 : PivotShapesValid 4077027 4077695 13
    ((terminalPivotChunks[12]!).map pivotShape) := by native_decide
private theorem pivotChunk13 : PivotShapesValid 4077696 4078031 8
    ((terminalPivotChunks[13]!).map pivotShape) := by native_decide
private theorem pivotChunk14 : PivotShapesValid 4078032 4078700 13
    ((terminalPivotChunks[14]!).map pivotShape) := by native_decide
private theorem pivotChunk15 : PivotShapesValid 4078701 4079036 8
    ((terminalPivotChunks[15]!).map pivotShape) := by native_decide
private theorem pivotChunk16 : PivotShapesValid 4079037 4079382 12
    ((terminalPivotChunks[16]!).map pivotShape) := by native_decide
private theorem pivotChunk17 : PivotShapesValid 4079705 4080041 9
    ((terminalPivotChunks[17]!).map pivotShape) := by native_decide
private theorem pivotChunk18 : PivotShapesValid 4080042 4080420 23
    ((terminalPivotChunks[18]!).map pivotShape) := by native_decide
private theorem pivotChunk19 : PivotShapesValid 4080421 4080654 58
    ((terminalPivotChunks[19]!).map pivotShape) := by native_decide
private theorem pivotChunk20 : PivotShapesValid 4080658 4080848 64
    ((terminalPivotChunks[20]!).map pivotShape) := by native_decide
private theorem pivotChunk21 : PivotShapesValid 4080852 4081076 58
    ((terminalPivotChunks[21]!).map pivotShape) := by native_decide
private theorem pivotChunk22 : PivotShapesValid 4081077 4081268 64
    ((terminalPivotChunks[22]!).map pivotShape) := by native_decide
private theorem pivotChunk23 : PivotShapesValid 4081269 4081323 21
    ((terminalPivotChunks[23]!).map pivotShape) := by native_decide

structure BoundedNodup (lower upper : Nat) (values : List Nat) : Prop where
  nodup : values.Nodup
  bounds : ∀ value ∈ values, lower ≤ value ∧ value ≤ upper

private theorem boundedNodup_of_shapes
    {lower upper count : Nat} {values : List Nat}
    (valid : PivotShapesValid lower upper count (values.map pivotShape)) :
    BoundedNodup lower upper values := by
  constructor
  · have ordered : values.Pairwise (fun left right => left < right) := by
      simpa [List.map_map, Function.comp_def, pivotShape] using valid.2.1
    exact ordered.imp Nat.ne_of_lt
  · intro value member
    have shapeMember : pivotShape value ∈ values.map pivotShape :=
      List.mem_map.mpr ⟨value, member, rfl⟩
    have bounds := valid.2.2 (pivotShape value) shapeMember
    exact ⟨bounds.1, bounds.2.1⟩

private theorem boundedNodup_append
    {lower middleLower middleUpper upper : Nat}
    {left right : List Nat}
    (leftValid : BoundedNodup lower middleUpper left)
    (rightValid : BoundedNodup middleLower upper right)
    (separated : middleUpper < middleLower)
    (lowerLeMiddle : lower ≤ middleLower)
    (middleLeUpper : middleUpper ≤ upper) :
    BoundedNodup lower upper (left ++ right) := by
  constructor
  · rw [List.nodup_append]
    refine ⟨leftValid.nodup, rightValid.nodup, ?_⟩
    intro left leftMember right rightMember
    have leftBound := (leftValid.bounds left leftMember).2
    have rightBound := (rightValid.bounds right rightMember).1
    omega
  · intro value member
    simp only [List.mem_append] at member
    rcases member with member | member
    · have bounds := leftValid.bounds value member
      exact ⟨bounds.1, Nat.le_trans bounds.2 middleLeUpper⟩
    · have bounds := rightValid.bounds value member
      exact ⟨Nat.le_trans lowerLeMiddle bounds.1, bounds.2⟩

private theorem pivotBounded0 : BoundedNodup 3964260 3964478
    (terminalPivotChunks[0]!) := boundedNodup_of_shapes pivotChunk0

private theorem pivotBounded1 : BoundedNodup 3964260 3971913
    ((terminalPivotChunks[0]!) ++ terminalPivotChunks[1]!) :=
  boundedNodup_append pivotBounded0 (boundedNodup_of_shapes pivotChunk1)
    (by decide) (by decide) (by decide)
private theorem pivotBounded2 : BoundedNodup 3964260 3986553
    ((terminalPivotChunks[0]!) ++ terminalPivotChunks[1]! ++ terminalPivotChunks[2]!) :=
  by
    simpa [List.append_assoc] using
      boundedNodup_append pivotBounded1
        (boundedNodup_of_shapes pivotChunk2) (by decide) (by decide) (by decide)
private theorem pivotBounded3 : BoundedNodup 3964260 4001193
    ((terminalPivotChunks[0]!) ++ terminalPivotChunks[1]! ++ terminalPivotChunks[2]! ++ terminalPivotChunks[3]!) :=
  by
    simpa [List.append_assoc] using
      boundedNodup_append pivotBounded2
        (boundedNodup_of_shapes pivotChunk3) (by decide) (by decide) (by decide)

private theorem pivotTailBounded : BoundedNodup 4001199 4081323
    ((terminalPivotChunks.drop 4).flatten) := by
  have b4 := boundedNodup_of_shapes pivotChunk4
  have b5 := boundedNodup_append b4 (boundedNodup_of_shapes pivotChunk5)
    (by decide) (by decide) (by decide)
  have b6 := boundedNodup_append b5 (boundedNodup_of_shapes pivotChunk6)
    (by decide) (by decide) (by decide)
  have b7 := boundedNodup_append b6 (boundedNodup_of_shapes pivotChunk7)
    (by decide) (by decide) (by decide)
  have b8 := boundedNodup_append b7 (boundedNodup_of_shapes pivotChunk8)
    (by decide) (by decide) (by decide)
  have b9 := boundedNodup_append b8 (boundedNodup_of_shapes pivotChunk9)
    (by decide) (by decide) (by decide)
  have b10 := boundedNodup_append b9 (boundedNodup_of_shapes pivotChunk10)
    (by decide) (by decide) (by decide)
  have b11 := boundedNodup_append b10 (boundedNodup_of_shapes pivotChunk11)
    (by decide) (by decide) (by decide)
  have b12 := boundedNodup_append b11 (boundedNodup_of_shapes pivotChunk12)
    (by decide) (by decide) (by decide)
  have b13 := boundedNodup_append b12 (boundedNodup_of_shapes pivotChunk13)
    (by decide) (by decide) (by decide)
  have b14 := boundedNodup_append b13 (boundedNodup_of_shapes pivotChunk14)
    (by decide) (by decide) (by decide)
  have b15 := boundedNodup_append b14 (boundedNodup_of_shapes pivotChunk15)
    (by decide) (by decide) (by decide)
  have b16 := boundedNodup_append b15 (boundedNodup_of_shapes pivotChunk16)
    (by decide) (by decide) (by decide)
  have b17 := boundedNodup_append b16 (boundedNodup_of_shapes pivotChunk17)
    (by decide) (by decide) (by decide)
  have b18 := boundedNodup_append b17 (boundedNodup_of_shapes pivotChunk18)
    (by decide) (by decide) (by decide)
  have b19 := boundedNodup_append b18 (boundedNodup_of_shapes pivotChunk19)
    (by decide) (by decide) (by decide)
  have b20 := boundedNodup_append b19 (boundedNodup_of_shapes pivotChunk20)
    (by decide) (by decide) (by decide)
  have b21 := boundedNodup_append b20 (boundedNodup_of_shapes pivotChunk21)
    (by decide) (by decide) (by decide)
  have b22 := boundedNodup_append b21 (boundedNodup_of_shapes pivotChunk22)
    (by decide) (by decide) (by decide)
  have b23 := boundedNodup_append b22 (boundedNodup_of_shapes pivotChunk23)
    (by decide) (by decide) (by decide)
  simpa [terminalPivotChunks, List.drop, List.flatten_cons,
    List.flatten_nil, List.append_assoc] using b23

theorem terminalPivotColumns_nodup : terminalPivotColumns.Nodup := by
  have initialBounded := pivotBounded3
  have combined := boundedNodup_append initialBounded pivotTailBounded
    (by decide) (by decide) (by decide)
  simpa [terminalPivotColumns, terminalPivotChunks, List.flatten_cons,
    List.flatten_nil, List.append_assoc] using combined.nodup

private theorem pivotChunkLengths :
    terminalPivotChunks.map List.length =
      [64, 64, 64, 64, 64, 64, 64, 64, 49, 8, 13, 8,
       13, 8, 13, 8, 12, 9, 23, 58, 64, 58, 64, 21] := by
  have h0 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk0.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk0.1
  have h1 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk1.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk1.1
  have h2 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk2.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk2.1
  have h3 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk3.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk3.1
  have h4 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk4.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk4.1
  have h5 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk5.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk5.1
  have h6 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk6.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk6.1
  have h7 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk7.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk7.1
  have h8 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk8.values).length = 49 := by
    simpa [terminalPivotChunks] using pivotChunk8.1
  have h9 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk9.values).length = 8 := by
    simpa [terminalPivotChunks] using pivotChunk9.1
  have h10 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk10.values).length = 13 := by
    simpa [terminalPivotChunks] using pivotChunk10.1
  have h11 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk11.values).length = 8 := by
    simpa [terminalPivotChunks] using pivotChunk11.1
  have h12 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk12.values).length = 13 := by
    simpa [terminalPivotChunks] using pivotChunk12.1
  have h13 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk13.values).length = 8 := by
    simpa [terminalPivotChunks] using pivotChunk13.1
  have h14 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk14.values).length = 13 := by
    simpa [terminalPivotChunks] using pivotChunk14.1
  have h15 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk15.values).length = 8 := by
    simpa [terminalPivotChunks] using pivotChunk15.1
  have h16 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk16.values).length = 12 := by
    simpa [terminalPivotChunks] using pivotChunk16.1
  have h17 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk17.values).length = 9 := by
    simpa [terminalPivotChunks] using pivotChunk17.1
  have h18 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk18.values).length = 23 := by
    simpa [terminalPivotChunks] using pivotChunk18.1
  have h19 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk19.values).length = 58 := by
    simpa [terminalPivotChunks] using pivotChunk19.1
  have h20 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk20.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk20.1
  have h21 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk21.values).length = 58 := by
    simpa [terminalPivotChunks] using pivotChunk21.1
  have h22 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk22.values).length = 64 := by
    simpa [terminalPivotChunks] using pivotChunk22.1
  have h23 : (terminalPivotColumnsOf
      Provenance.RewriteSteps.Chunk23.values).length = 21 := by
    simpa [terminalPivotChunks] using pivotChunk23.1
  simp [terminalPivotChunks, h0, h1, h2, h3, h4, h5, h6, h7,
    h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18,
    h19, h20, h21, h22, h23]

theorem terminalPivotColumn_count : terminalPivotColumns.length = 941 := by
  rw [terminalPivotColumns, List.length_flatten, pivotChunkLengths]
  decide

private theorem pivotFacts_of_shapes
    {lower upper count : Nat} {values : List Nat}
    (valid : PivotShapesValid lower upper count (values.map pivotShape)) :
    ∀ column ∈ values,
      column ∈ SourceExecution.definitionOutputs ∧
        column ∉ physicalDefinitionOutputs := by
  intro column member
  have shapeMember : pivotShape column ∈ values.map pivotShape :=
    List.mem_map.mpr ⟨column, member, rfl⟩
  have facts := (valid.2.2 _ shapeMember).2.2
  exact ⟨of_decide_eq_true (by simpa only [pivotShape] using facts.1),
    of_decide_eq_true (by simpa only [pivotShape] using facts.2)⟩

private theorem terminalPivotChunk_facts :
    ∀ chunk ∈ terminalPivotChunks, ∀ column ∈ chunk,
      column ∈ SourceExecution.definitionOutputs ∧
        column ∉ physicalDefinitionOutputs := by
  intro chunk member
  simp only [terminalPivotChunks, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl
  · exact pivotFacts_of_shapes pivotChunk0
  · exact pivotFacts_of_shapes pivotChunk1
  · exact pivotFacts_of_shapes pivotChunk2
  · exact pivotFacts_of_shapes pivotChunk3
  · exact pivotFacts_of_shapes pivotChunk4
  · exact pivotFacts_of_shapes pivotChunk5
  · exact pivotFacts_of_shapes pivotChunk6
  · exact pivotFacts_of_shapes pivotChunk7
  · exact pivotFacts_of_shapes pivotChunk8
  · exact pivotFacts_of_shapes pivotChunk9
  · exact pivotFacts_of_shapes pivotChunk10
  · exact pivotFacts_of_shapes pivotChunk11
  · exact pivotFacts_of_shapes pivotChunk12
  · exact pivotFacts_of_shapes pivotChunk13
  · exact pivotFacts_of_shapes pivotChunk14
  · exact pivotFacts_of_shapes pivotChunk15
  · exact pivotFacts_of_shapes pivotChunk16
  · exact pivotFacts_of_shapes pivotChunk17
  · exact pivotFacts_of_shapes pivotChunk18
  · exact pivotFacts_of_shapes pivotChunk19
  · exact pivotFacts_of_shapes pivotChunk20
  · exact pivotFacts_of_shapes pivotChunk21
  · exact pivotFacts_of_shapes pivotChunk22
  · exact pivotFacts_of_shapes pivotChunk23

theorem terminalPivotColumns_subset_sourceOutputs :
    ∀ column ∈ terminalPivotColumns,
      column ∈ SourceExecution.definitionOutputs := by
  intro column member
  rw [terminalPivotColumns] at member
  rcases List.mem_flatten.mp member with ⟨chunk, chunkMember, columnMember⟩
  exact (terminalPivotChunk_facts chunk chunkMember column columnMember).1

theorem terminalPivotColumns_disjoint_physical :
    ∀ column ∈ terminalPivotColumns,
      column ∉ physicalDefinitionOutputs := by
  intro column member
  rw [terminalPivotColumns] at member
  rcases List.mem_flatten.mp member with ⟨chunk, chunkMember, columnMember⟩
  exact (terminalPivotChunk_facts chunk chunkMember column columnMember).2

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition
