import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Core

/-!
Executable product-group certificate for the focused compact `y_zcol` slice.

Owns: direct executable checking that every scheduled two-step extension
product group matches its independent quadratic target.

Does not own: symbolic definitions, evaluation groups, source-program
execution, protocol authority, security events, or permission to remove rows.

Emits constraints: no.

| Certificate leaf | Mathematical obligation | Authority class |
|---|---|---|
| product groups | every scheduled product pair matches its independent target | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Product

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

def pairs : List ProductPair :=
  productPairs

def data : List GroupMatchShape :=
  pairs.map productPairShape

theorem dataLengthExact : data.length = 86 := by
  simpa only [data, pairs, List.length_map] using productPairsLengthExact

theorem dataWithinCertificateLimit : data.length ≤ 256 := by
  rw [dataLengthExact]
  decide

private def check : Bool :=
  groupMatchShapesCheck data

set_option maxRecDepth 100000 in
private theorem check_true : check = true := by
  native_decide

theorem pairsMatch :
    ∀ pair ∈ pairs, GroupMatches pair.1 (productExpected pair.2) := by
  intro pair member
  have shapeMember : productPairShape pair ∈ data :=
    List.mem_map.mpr ⟨pair, member, rfl⟩
  have allChecked : data.all groupMatchShapeCheck = true := by
    simpa only [check, groupMatchShapesCheck] using check_true
  exact groupMatches_of_shape_check_true
    ((List.all_eq_true.mp allChecked) (productPairShape pair) shapeMember)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Product
