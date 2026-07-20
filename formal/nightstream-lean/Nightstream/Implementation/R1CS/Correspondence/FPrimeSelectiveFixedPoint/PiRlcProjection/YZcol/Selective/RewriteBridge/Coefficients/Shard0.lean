import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.Chunks

/-!
Proof-free coefficient certificate for rewrite pairs 0 through 249.

Owns: one bounded compact coefficient check and its kernel lift.

Does not own: other chunks, row satisfaction, selector truth, source
authority, security events, or permission to remove rows.

Emits constraints: no.

| Certificate leaf | Exact obligation | Authority class | Multiplicity |
|---|---|---|---|
| rewrite chunk 0 | all thirteen normalized port forms match | checked | 250 records |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.Shard0

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients

def pairs : List RewritePair := rewriteCoefficientChunk0

def data : List CoefficientMatchShape := rewriteCoefficientDataChunk0

theorem dataLengthExact : data.length = 250 :=
  rewriteCoefficientDataChunkLengthsExact.1

theorem dataWithinCertificateLimit : data.length ≤ 256 := by
  rw [dataLengthExact]
  decide

private def check : Bool := coefficientMatchShapesCheck data

set_option maxRecDepth 100000 in
private theorem check_true : check = true := by
  native_decide

theorem coefficientsExact :
    ∀ pair ∈ pairs, RewriteCoefficientsMatch pair.1 pair.2 := by
  intro pair member
  have shapeMember : rewritePairCoefficientShape pair ∈ data :=
    List.mem_map.mpr ⟨pair, member, rfl⟩
  have allChecked : data.all coefficientMatchShapeCheck = true := by
    simpa only [check, coefficientMatchShapesCheck] using check_true
  exact rewriteCoefficientsMatch_of_shape_check_true
    ((List.all_eq_true.mp allChecked) _ shapeMember)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.Shard0
