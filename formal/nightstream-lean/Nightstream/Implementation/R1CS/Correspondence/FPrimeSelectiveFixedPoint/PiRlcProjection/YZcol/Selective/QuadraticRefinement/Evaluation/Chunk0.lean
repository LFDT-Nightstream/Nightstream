import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Core

/-!
Executable evaluation-group certificate shard 0.

Owns: the checked `take`/`drop` slice of evaluation pairs at this stable
shard position and its direct `GroupMatches` certificate.

Does not own: whole-list coverage, symbolic semantics, product groups,
protocol authority, security events, or permission to remove rows.

Emits constraints: no.

| Certificate leaf | Mathematical obligation | Authority class |
|---|---|---|
| evaluation shard 0 | every pair in this slice matches its independent target | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Evaluation.Chunk0

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

def pairs : List EvaluationPair :=
  (evaluationPairs.drop 0).take 7

def data : List GroupMatchShape :=
  pairs.map evaluationPairShape

theorem dataLengthExact : data.length = 7 := by
  simp [data, pairs, evaluationPairsLengthExact]

theorem dataWithinCertificateLimit : data.length ≤ 256 := by
  rw [dataLengthExact]
  decide

private def check : Bool :=
  groupMatchShapesCheck data

set_option maxRecDepth 100000 in
private theorem check_true : check = true := by
  native_decide

theorem pairsMatch :
    ∀ pair ∈ pairs, GroupMatches pair.1 (evaluationExpected pair.2) := by
  intro pair member
  have shapeMember : evaluationPairShape pair ∈ data :=
    List.mem_map.mpr ⟨pair, member, rfl⟩
  have allChecked : data.all groupMatchShapeCheck = true := by
    simpa only [check, groupMatchShapesCheck] using check_true
  exact groupMatches_of_shape_check_true
    ((List.all_eq_true.mp allChecked) (evaluationPairShape pair) shapeMember)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Evaluation.Chunk0
