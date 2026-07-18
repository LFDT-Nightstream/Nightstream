import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout

/-!
Semantic dataflow bridge from typed PiCCS `y_zcol` output columns to their
PiRLC coefficient consumers.

Assurance tier: model-level representation correspondence.

Owns: a protocol → phase → family model for the producer/consumer column
boundary; the exact two-limb, source-major, 54-lane decoder; and the theorem
that typed PiCCS `y_zcol` binding plus leaf-for-leaf column equality determines
the complete PiRLC input vector.

Does not own: any concrete column numbers, PiCCS output truth, transcript
challenges, projection identities, the returned parent, Rust/R1CS artifact
recovery, costs, security bounds, or row removal.

Emits constraints: no.

Authority boundary: `ConsumerMatches` is pure dataflow. It can transport an
already-proved PiCCS source binding, but cannot create that binding. The Rust
fixed-point audit checks the concrete 15 × 54 × 2 instance separately.

| Protocol → phase → family | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.output.y_zcol.columns` | one producer column for each source/lane/limb | checked upstream | `SourceRole.yZcolLimb` |
| `pi_rlc.identities.y_zcol.inputs` | decode two physical limbs into one `RingK` source | direct dataflow | `decodedInputs` |
| `pi_ccs_to_pi_rlc.y_zcol` | every consumer column equals its typed producer column | checked refinement | `ConsumerMatches` |
| `pi_ccs_to_pi_rlc.y_zcol.semantic` | source binding determines all decoded PiRLC inputs | derived | `decodedInputs_eq_yZcol_of_bound` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout

/-- Physical extension-field limb selected by one PiRLC evaluator family. -/
inductive Limb where
  | c0
  | c1
deriving DecidableEq, Repr

/-- Consumer-side coefficient columns, indexed by the semantic source tree. -/
structure ConsumerColumns (shape : SemanticShape) where
  column : Limb -> Fin shape.sourceCount -> Fin ringDegree -> Nat

/-- Exact cross-phase dataflow: neither side may reorder, truncate, or replace
one active coefficient column. -/
structure ConsumerMatches
    {shape : SemanticShape}
    (producer : SourceRole shape -> Nat)
    (consumer : ConsumerColumns shape) : Prop where
  c0 : forall source lane,
    consumer.column .c0 source lane =
      producer (.yZcolLimb source lane .c0)
  c1 : forall source lane,
    consumer.column .c1 source lane =
      producer (.yZcolLimb source lane .c1)

/-- Complete typed PiRLC source vector decoded from the two physical consumer
families. -/
def decodedInputs
    {shape : SemanticShape}
    (assignment : Nat -> F)
    (consumer : ConsumerColumns shape) :
    Fin shape.sourceCount -> RingK :=
  fun source lane =>
    ⟨assignment (consumer.column .c0 source lane),
     assignment (consumer.column .c1 source lane)⟩

private theorem k_eq_of_limbs
    (left right : K)
    (c0 : left.c0 = right.c0)
    (c1 : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Once PiCCS has bound every typed `y_zcol` source value, exact physical
producer/consumer column equality eliminates any additional PiRLC input
authority premise. -/
theorem decodedInputs_eq_yZcol_of_bound
    {shape : SemanticShape}
    {assignment : Nat -> F}
    {producer : SourceRole shape -> Nat}
    {consumer : ConsumerColumns shape}
    {message : OutputMessage shape}
    (columnMatch : ConsumerMatches producer consumer)
    (bound : BindingsHoldFor .yZcolOutput assignment producer message) :
    decodedInputs assignment consumer = message.yZcol := by
  funext source lane
  apply k_eq_of_limbs
  · change assignment (consumer.column .c0 source lane) =
      (message.yZcol source lane).c0
    rw [columnMatch.c0 source lane]
    exact bound (.yZcolLimb source lane .c0) rfl
  · change assignment (consumer.column .c1 source lane) =
      (message.yZcol source lane).c1
    rw [columnMatch.c1 source lane]
    exact bound (.yZcolLimb source lane .c1) rfl

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer
