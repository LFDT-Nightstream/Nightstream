import Mathlib.Data.Nat.Log
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.EncodingRows
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment

/-!
Contract: derive the selective-CCS row-domain profile from the exact
Lean-emitted row and column lists.

Assurance tier: model-level.

Owns: the least Boolean row-domain exponent and a profile constructor whose
dimensions are the exact compiler output dimensions.

Does not own: construction of a full F-prime encoding, proof that its logical
width contains the required 270 public coordinates, Rust fixed-point
iteration, generated headers, or protocol security events.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.Profile

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.SelectiveCcs

universe u

/-- Least Boolean-cube exponent that covers the emitted rows and the
protocol-required two-row minimum. -/
def rowVariables (rows : Nat) : Nat :=
  Nat.clog 2 (max rows 2)

theorem exactRowDomain (rows : Nat) :
    RelationProfile.ExactRowDomain rows (rowVariables rows) := by
  constructor
  · exact Nat.le_pow_clog (by decide) (max rows 2)
  · intro smaller smallerLt
    exact
      (Nat.lt_clog_iff_pow_lt (by decide)).mp smallerLt

/-- Any logical width of at least 270 has enough completed Phi81 carrier
coordinates for the five public rings. -/
theorem publicFits_of_alignedWidth
    {columns : Nat}
    (aligned : 270 ≤ columns) :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth columns := by
  have logicalFits :
      columns ≤ Phi81CarrierLayout.carrierWidth columns :=
    Phi81CarrierLayout.logicalWidth_le_carrierWidth columns
  change 270 ≤ Phi81CarrierLayout.carrierWidth columns
  exact Nat.le_trans aligned logicalFits

/-- Construct the exact current-program shape directly from the Lean
encoding. The only premise is the protocol-visible 270-column public prefix;
row variables are computed and proved minimal here. -/
def ofEncoding
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (publicWidth : 270 ≤ encoding.columnIds.length) :
    RelationProfile.Profile
      (EncodingRows.program encoding).length
      encoding.columnIds.length where
  rowVariables := rowVariables (EncodingRows.program encoding).length
  rowDomain := exactRowDomain (EncodingRows.program encoding).length
  publicRingColumns := publicRingColumns
  publicFits := publicFits_of_alignedWidth publicWidth

@[simp] theorem ofEncoding_rowVariables
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (publicWidth : 270 ≤ encoding.columnIds.length) :
    (ofEncoding encoding publicWidth).rowVariables =
      rowVariables (EncodingRows.program encoding).length :=
  rfl

theorem ofEncoding_rows_covered
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (publicWidth : 270 ≤ encoding.columnIds.length) :
    (EncodingRows.program encoding).length ≤
      2 ^ (ofEncoding encoding publicWidth).rowVariables :=
  RelationProfile.Profile.rows_covered (ofEncoding encoding publicWidth)

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.Profile
