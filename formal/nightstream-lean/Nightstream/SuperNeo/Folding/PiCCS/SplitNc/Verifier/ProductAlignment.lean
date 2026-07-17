import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTypes
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-!
Partition-preserving alignment between a production `PiCCS.InputProduct` and
one Split-NC semantic source family.

Protocol: SuperNeo `Pi_CCS`.
Phase: public-product/source indexing shared by input and output authority.
Constraint family: finite-index transport only; this file emits no rows.

Owns: equality of the fresh and running partition sizes; exact total-count
alignment; and mutually inverse transports for unified, fresh, and running
indices.

Does not own: source fields, commitments, public inputs, assignments,
transcripts, output materialization, Rust, R1CS, rows, or costs.

Emits constraints: no.

Authority boundary: equal total counts are insufficient because they permit a
fresh/running repartition. `SourceAlignment` records both partition equalities
and every later transport is derived from those two equalities.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.product.partition.fresh` | product fresh count equals semantic fresh count | checked premise | `freshCount_eq` |
| `nifs.pi_ccs.product.partition.running` | product running count equals semantic running count | checked premise | `runningCount_eq` |
| `nifs.pi_ccs.product.partition.total` | total source counts are equal | derived | `total_eq_sourceCount` |
| `nifs.pi_ccs.product.index` | transports are mutual inverses | derived | index inverse theorems |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The semantic source model and protocol product agree on each side of the
fresh/running partition. -/
structure SourceAlignment
    (shape : SemanticShape)
    (params : GlobalParams)
    (arity : BatchArity params) : Prop where
  freshCount_eq : arity.freshCount = shape.freshCount
  runningCount_eq : arity.mode.count params = shape.runningCount

namespace SourceAlignment

/-- Partition alignment implies exact total source-count alignment. -/
theorem total_eq_sourceCount
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity) :
    arity.total = shape.sourceCount := by
  simp only [BatchArity.total, SemanticShape.sourceCount]
  rw [alignment.freshCount_eq, alignment.runningCount_eq]

/-- Reindex one product source into the semantic source family. -/
def semanticIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin arity.total) : Fin shape.sourceCount :=
  Fin.cast alignment.total_eq_sourceCount source

/-- Reindex one semantic source into the protocol product. -/
def productIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin shape.sourceCount) : Fin arity.total :=
  Fin.cast alignment.total_eq_sourceCount.symm source

/-- Reindex one fresh product source without crossing the partition. -/
def semanticFreshIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin arity.freshCount) : Fin shape.freshCount :=
  Fin.cast alignment.freshCount_eq source

/-- Reindex one semantic fresh source into the product. -/
def productFreshIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin shape.freshCount) : Fin arity.freshCount :=
  Fin.cast alignment.freshCount_eq.symm source

/-- Reindex one running product source without crossing the partition. -/
def semanticRunningIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin (arity.mode.count params)) : Fin shape.runningCount :=
  Fin.cast alignment.runningCount_eq source

/-- Reindex one semantic running source into the product. -/
def productRunningIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin shape.runningCount) : Fin (arity.mode.count params) :=
  Fin.cast alignment.runningCount_eq.symm source

@[simp] theorem semanticIndex_productIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin shape.sourceCount) :
    alignment.semanticIndex (alignment.productIndex source) = source := by
  apply Fin.ext
  rfl

@[simp] theorem productIndex_semanticIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin arity.total) :
    alignment.productIndex (alignment.semanticIndex source) = source := by
  apply Fin.ext
  rfl

@[simp] theorem semanticFreshIndex_productFreshIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin shape.freshCount) :
    alignment.semanticFreshIndex (alignment.productFreshIndex source) =
      source := by
  apply Fin.ext
  rfl

@[simp] theorem productFreshIndex_semanticFreshIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin arity.freshCount) :
    alignment.productFreshIndex (alignment.semanticFreshIndex source) =
      source := by
  apply Fin.ext
  rfl

@[simp] theorem semanticRunningIndex_productRunningIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin shape.runningCount) :
    alignment.semanticRunningIndex (alignment.productRunningIndex source) =
      source := by
  apply Fin.ext
  rfl

@[simp] theorem productRunningIndex_semanticRunningIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin (arity.mode.count params)) :
    alignment.productRunningIndex (alignment.semanticRunningIndex source) =
      source := by
  apply Fin.ext
  rfl

/-- Unified reindexing preserves the fresh injection exactly. -/
@[simp] theorem semanticIndex_fresh
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin arity.freshCount) :
    alignment.semanticIndex
        (Fin.castAdd (arity.mode.count params) source) =
      SplitNc.Sources.Data.freshIndex
        (alignment.semanticFreshIndex source) := by
  apply Fin.ext
  rfl

/-- Unified reindexing preserves the running injection exactly. -/
@[simp] theorem semanticIndex_running
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (source : Fin (arity.mode.count params)) :
    alignment.semanticIndex (Fin.natAdd arity.freshCount source) =
      SplitNc.Sources.Data.runningIndex
        (alignment.semanticRunningIndex source) := by
  apply Fin.ext
  simp [semanticIndex, semanticRunningIndex,
    SplitNc.Sources.Data.runningIndex, alignment.freshCount_eq]

end SourceAlignment

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
