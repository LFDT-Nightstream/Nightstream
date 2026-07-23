import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard0
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard1
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard2
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.CertificateShard3

/-!
Kernel composition of the bounded production combined-NC pair certificates.

Owns: certificate truth for all 1,493 rewrite and 52 retained pairs, exact
pair-to-row and pair-to-provenance coverage, and exact ownership of every one
of the 1,545 emitted rows.

Does not own: row satisfaction, eliminated source equations, selector truth,
transcript or raw-child authority, commitment binding, costs, or permission
to remove rows.

Emits constraints: none.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.selective_pairs.certificates` | Compose bounded pairing shards and prove exact certificate-input coverage. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Certificates

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

private theorem rewriteCertificate_append {left right : List RawRewritePair}
    (leftValid : RewritePairsCertificate left)
    (rightValid : RewritePairsCertificate right) :
    RewritePairsCertificate (left ++ right) := by
  intro pair member
  rcases List.mem_append.mp member with leftMember | rightMember
  · exact leftValid pair leftMember
  · exact rightValid pair rightMember

theorem retainedPairsCertified : RetainedPairsCertificate retainedPairs :=
  CertificateShard0.retained

theorem rewritePairsCertified : RewritePairsCertificate rewritePairs := by
  simp only [rewritePairs]
  simpa only [List.append_assoc] using
    (rewriteCertificate_append CertificateShard0.rewrite0 <|
    rewriteCertificate_append CertificateShard0.rewrite1 <|
    rewriteCertificate_append CertificateShard0.rewrite2 <|
    rewriteCertificate_append CertificateShard0.rewrite3 <|
    rewriteCertificate_append CertificateShard0.rewrite4 <|
    rewriteCertificate_append CertificateShard0.rewrite5 <|
    rewriteCertificate_append CertificateShard1.rewrite6 <|
    rewriteCertificate_append CertificateShard1.rewrite7 <|
    rewriteCertificate_append CertificateShard1.rewrite8 <|
    rewriteCertificate_append CertificateShard1.rewrite9 <|
    rewriteCertificate_append CertificateShard1.rewrite10 <|
    rewriteCertificate_append CertificateShard1.rewrite11 <|
    rewriteCertificate_append CertificateShard2.rewrite12 <|
    rewriteCertificate_append CertificateShard2.rewrite13 <|
    rewriteCertificate_append CertificateShard2.rewrite14 <|
    rewriteCertificate_append CertificateShard2.rewrite15 <|
    rewriteCertificate_append CertificateShard2.rewrite16 <|
    rewriteCertificate_append CertificateShard2.rewrite17 <|
    rewriteCertificate_append CertificateShard3.rewrite18 <|
    rewriteCertificate_append CertificateShard3.rewrite19 <|
    rewriteCertificate_append CertificateShard3.rewrite20 <|
    rewriteCertificate_append CertificateShard3.rewrite21 <|
    rewriteCertificate_append CertificateShard3.rewrite22
      CertificateShard3.rewrite23)

private theorem map_zipWith_rewrite_left
    (rows : List RawEmittedRow) (steps : List RawRewriteStep)
    (lengths : rows.length = steps.length) :
    (List.zipWith RawRewritePair.mk rows steps).map RawRewritePair.emitted =
      rows := by
  induction rows generalizing steps with
  | nil =>
      cases steps <;> simp_all
  | cons row rows inductionHypothesis =>
      cases steps with
      | nil => simp at lengths
      | cons step steps =>
          simp only [List.length_cons, Nat.succ.injEq] at lengths
          simp [inductionHypothesis steps lengths]

private theorem map_zipWith_rewrite_right
    (rows : List RawEmittedRow) (steps : List RawRewriteStep)
    (lengths : rows.length = steps.length) :
    (List.zipWith RawRewritePair.mk rows steps).map
        RawRewritePair.provenance = steps := by
  induction rows generalizing steps with
  | nil =>
      cases steps <;> simp_all
  | cons row rows inductionHypothesis =>
      cases steps with
      | nil => simp at lengths
      | cons step steps =>
          simp only [List.length_cons, Nat.succ.injEq] at lengths
          simp [inductionHypothesis steps lengths]

private theorem map_zipWith_retained_left
    (rows : List RawEmittedRow) (steps : List RawRetainedStep)
    (lengths : rows.length = steps.length) :
    (List.zipWith RawRetainedPair.mk rows steps).map RawRetainedPair.emitted =
      rows := by
  induction rows generalizing steps with
  | nil =>
      cases steps <;> simp_all
  | cons row rows inductionHypothesis =>
      cases steps with
      | nil => simp at lengths
      | cons step steps =>
          simp only [List.length_cons, Nat.succ.injEq] at lengths
          simp [inductionHypothesis steps lengths]

private theorem map_zipWith_retained_right
    (rows : List RawEmittedRow) (steps : List RawRetainedStep)
    (lengths : rows.length = steps.length) :
    (List.zipWith RawRetainedPair.mk rows steps).map
        RawRetainedPair.provenance = steps := by
  induction rows generalizing steps with
  | nil =>
      cases steps <;> simp_all
  | cons row rows inductionHypothesis =>
      cases steps with
      | nil => simp at lengths
      | cons step steps =>
          simp only [List.length_cons, Nat.succ.injEq] at lengths
          simp [inductionHypothesis steps lengths]

/- Each length proof evaluates two proof-free natural-number scalars, not a
row or decoded-record list. -/
private theorem retainedLengths : retainedRows.length = retainedSteps.length := by
  native_decide
private theorem rewriteLengths0 : rewriteRows0.length = rewriteSteps0.length := by native_decide
private theorem rewriteLengths1 : rewriteRows1.length = rewriteSteps1.length := by native_decide
private theorem rewriteLengths2 : rewriteRows2.length = rewriteSteps2.length := by native_decide
private theorem rewriteLengths3 : rewriteRows3.length = rewriteSteps3.length := by native_decide
private theorem rewriteLengths4 : rewriteRows4.length = rewriteSteps4.length := by native_decide
private theorem rewriteLengths5 : rewriteRows5.length = rewriteSteps5.length := by native_decide
private theorem rewriteLengths6 : rewriteRows6.length = rewriteSteps6.length := by native_decide
private theorem rewriteLengths7 : rewriteRows7.length = rewriteSteps7.length := by native_decide
private theorem rewriteLengths8 : rewriteRows8.length = rewriteSteps8.length := by native_decide
private theorem rewriteLengths9 : rewriteRows9.length = rewriteSteps9.length := by native_decide
private theorem rewriteLengths10 : rewriteRows10.length = rewriteSteps10.length := by native_decide
private theorem rewriteLengths11 : rewriteRows11.length = rewriteSteps11.length := by native_decide
private theorem rewriteLengths12 : rewriteRows12.length = rewriteSteps12.length := by native_decide
private theorem rewriteLengths13 : rewriteRows13.length = rewriteSteps13.length := by native_decide
private theorem rewriteLengths14 : rewriteRows14.length = rewriteSteps14.length := by native_decide
private theorem rewriteLengths15 : rewriteRows15.length = rewriteSteps15.length := by native_decide
private theorem rewriteLengths16 : rewriteRows16.length = rewriteSteps16.length := by native_decide
private theorem rewriteLengths17 : rewriteRows17.length = rewriteSteps17.length := by native_decide
private theorem rewriteLengths18 : rewriteRows18.length = rewriteSteps18.length := by native_decide
private theorem rewriteLengths19 : rewriteRows19.length = rewriteSteps19.length := by native_decide
private theorem rewriteLengths20 : rewriteRows20.length = rewriteSteps20.length := by native_decide
private theorem rewriteLengths21 : rewriteRows21.length = rewriteSteps21.length := by native_decide
private theorem rewriteLengths22 : rewriteRows22.length = rewriteSteps22.length := by native_decide
private theorem rewriteLengths23 : rewriteRows23.length = rewriteSteps23.length := by native_decide

theorem retainedPairRowsExact :
    retainedPairs.map RawRetainedPair.emitted = retainedRows := by
  exact map_zipWith_retained_left retainedRows retainedSteps retainedLengths

theorem retainedPairStepsExact :
    retainedPairs.map RawRetainedPair.provenance = retainedSteps := by
  exact map_zipWith_retained_right retainedRows retainedSteps retainedLengths

theorem rewritePairRowsExact :
    rewritePairs.map RawRewritePair.emitted = rewriteRows := by
  simp only [rewritePairs, rewriteRows, rewritePairs0, rewritePairs1,
    rewritePairs2, rewritePairs3, rewritePairs4, rewritePairs5,
    rewritePairs6, rewritePairs7, rewritePairs8, rewritePairs9,
    rewritePairs10, rewritePairs11, rewritePairs12, rewritePairs13,
    rewritePairs14, rewritePairs15, rewritePairs16, rewritePairs17,
    rewritePairs18, rewritePairs19, rewritePairs20, rewritePairs21,
    rewritePairs22, rewritePairs23, pairRewrite, List.map_append]
  simp only [map_zipWith_rewrite_left rewriteRows0 rewriteSteps0 rewriteLengths0,
    map_zipWith_rewrite_left rewriteRows1 rewriteSteps1 rewriteLengths1,
    map_zipWith_rewrite_left rewriteRows2 rewriteSteps2 rewriteLengths2,
    map_zipWith_rewrite_left rewriteRows3 rewriteSteps3 rewriteLengths3,
    map_zipWith_rewrite_left rewriteRows4 rewriteSteps4 rewriteLengths4,
    map_zipWith_rewrite_left rewriteRows5 rewriteSteps5 rewriteLengths5,
    map_zipWith_rewrite_left rewriteRows6 rewriteSteps6 rewriteLengths6,
    map_zipWith_rewrite_left rewriteRows7 rewriteSteps7 rewriteLengths7,
    map_zipWith_rewrite_left rewriteRows8 rewriteSteps8 rewriteLengths8,
    map_zipWith_rewrite_left rewriteRows9 rewriteSteps9 rewriteLengths9,
    map_zipWith_rewrite_left rewriteRows10 rewriteSteps10 rewriteLengths10,
    map_zipWith_rewrite_left rewriteRows11 rewriteSteps11 rewriteLengths11,
    map_zipWith_rewrite_left rewriteRows12 rewriteSteps12 rewriteLengths12,
    map_zipWith_rewrite_left rewriteRows13 rewriteSteps13 rewriteLengths13,
    map_zipWith_rewrite_left rewriteRows14 rewriteSteps14 rewriteLengths14,
    map_zipWith_rewrite_left rewriteRows15 rewriteSteps15 rewriteLengths15,
    map_zipWith_rewrite_left rewriteRows16 rewriteSteps16 rewriteLengths16,
    map_zipWith_rewrite_left rewriteRows17 rewriteSteps17 rewriteLengths17,
    map_zipWith_rewrite_left rewriteRows18 rewriteSteps18 rewriteLengths18,
    map_zipWith_rewrite_left rewriteRows19 rewriteSteps19 rewriteLengths19,
    map_zipWith_rewrite_left rewriteRows20 rewriteSteps20 rewriteLengths20,
    map_zipWith_rewrite_left rewriteRows21 rewriteSteps21 rewriteLengths21,
    map_zipWith_rewrite_left rewriteRows22 rewriteSteps22 rewriteLengths22,
    map_zipWith_rewrite_left rewriteRows23 rewriteSteps23 rewriteLengths23,
    List.append_assoc]

theorem rewritePairStepsExact :
    rewritePairs.map RawRewritePair.provenance = rewriteSteps := by
  simp only [rewritePairs, rewriteSteps, rewritePairs0, rewritePairs1,
    rewritePairs2, rewritePairs3, rewritePairs4, rewritePairs5,
    rewritePairs6, rewritePairs7, rewritePairs8, rewritePairs9,
    rewritePairs10, rewritePairs11, rewritePairs12, rewritePairs13,
    rewritePairs14, rewritePairs15, rewritePairs16, rewritePairs17,
    rewritePairs18, rewritePairs19, rewritePairs20, rewritePairs21,
    rewritePairs22, rewritePairs23, pairRewrite, List.map_append]
  simp only [map_zipWith_rewrite_right rewriteRows0 rewriteSteps0 rewriteLengths0,
    map_zipWith_rewrite_right rewriteRows1 rewriteSteps1 rewriteLengths1,
    map_zipWith_rewrite_right rewriteRows2 rewriteSteps2 rewriteLengths2,
    map_zipWith_rewrite_right rewriteRows3 rewriteSteps3 rewriteLengths3,
    map_zipWith_rewrite_right rewriteRows4 rewriteSteps4 rewriteLengths4,
    map_zipWith_rewrite_right rewriteRows5 rewriteSteps5 rewriteLengths5,
    map_zipWith_rewrite_right rewriteRows6 rewriteSteps6 rewriteLengths6,
    map_zipWith_rewrite_right rewriteRows7 rewriteSteps7 rewriteLengths7,
    map_zipWith_rewrite_right rewriteRows8 rewriteSteps8 rewriteLengths8,
    map_zipWith_rewrite_right rewriteRows9 rewriteSteps9 rewriteLengths9,
    map_zipWith_rewrite_right rewriteRows10 rewriteSteps10 rewriteLengths10,
    map_zipWith_rewrite_right rewriteRows11 rewriteSteps11 rewriteLengths11,
    map_zipWith_rewrite_right rewriteRows12 rewriteSteps12 rewriteLengths12,
    map_zipWith_rewrite_right rewriteRows13 rewriteSteps13 rewriteLengths13,
    map_zipWith_rewrite_right rewriteRows14 rewriteSteps14 rewriteLengths14,
    map_zipWith_rewrite_right rewriteRows15 rewriteSteps15 rewriteLengths15,
    map_zipWith_rewrite_right rewriteRows16 rewriteSteps16 rewriteLengths16,
    map_zipWith_rewrite_right rewriteRows17 rewriteSteps17 rewriteLengths17,
    map_zipWith_rewrite_right rewriteRows18 rewriteSteps18 rewriteLengths18,
    map_zipWith_rewrite_right rewriteRows19 rewriteSteps19 rewriteLengths19,
    map_zipWith_rewrite_right rewriteRows20 rewriteSteps20 rewriteLengths20,
    map_zipWith_rewrite_right rewriteRows21 rewriteSteps21 rewriteLengths21,
    map_zipWith_rewrite_right rewriteRows22 rewriteSteps22 rewriteLengths22,
    map_zipWith_rewrite_right rewriteRows23 rewriteSteps23 rewriteLengths23,
    List.append_assoc]

/-- Exact physical emitted-row ownership: 52 retained plus 1,493 rewrite
pairs cover the generated 1,545-row stream in its literal order. -/
theorem allPairedEmittedRowsExact :
    retainedPairs.map RawRetainedPair.emitted ++
        rewritePairs.map RawRewritePair.emitted =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.EmittedRows.values := by
  rw [retainedPairRowsExact, rewritePairRowsExact, emittedRowsExact]

/-- Provenance pairing is equally exact; zip truncation cannot hide a missing
rewrite or retained record. -/
theorem allPairedProvenanceExact :
    retainedPairs.map RawRetainedPair.provenance =
        Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.retainedSteps ∧
      rewritePairs.map RawRewritePair.provenance =
        Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.rewriteSteps := by
  constructor
  · rw [retainedPairStepsExact, retainedStepsExact]
  · rw [rewritePairStepsExact, rewriteStepsExact]

theorem everyRewritePairIndexMatches :
    ∀ pair ∈ rewritePairs,
      pair.emitted.emittedRow = pair.provenance.emittedRow := by
  intro pair member
  exact (rewritePairsCertified pair member).2.2.1

theorem everyRetainedPairIndexMatches :
    ∀ pair ∈ retainedPairs,
      pair.emitted.emittedRow = pair.provenance.emittedRow := by
  intro pair member
  exact (retainedPairsCertified pair member).2.2.1

theorem everyRewriteCoefficientShapeMatches :
    ∀ pair ∈ rewritePairs,
      coefficientShapeMatches
        (rewriteCoefficientShape pair.emitted pair.provenance) := by
  intro pair member
  exact (rewritePairsCertified pair member).2.2.2

theorem everyRetainedCoefficientShapeMatches :
    ∀ pair ∈ retainedPairs,
      coefficientShapeMatches
        (retainedCoefficientShape pair.emitted pair.provenance) := by
  intro pair member
  exact (retainedPairsCertified pair member).2.2.2

/-- Every checked rewrite record has at most five factors, exactly the
capacity of the physical evaluation row. -/
theorem everyRewriteFactorCapacity :
  ∀ pair ∈ rewritePairs, pair.provenance.factors.length ≤ 5 := by
  intro pair member
  exact (rewritePairsCertified pair member).2.1.2.2.2.2.2

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Certificates
