import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk62
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Artifact

/-!
Exact source owners for the 52 retained production combined-NC compiler
obligations.

Owns: the direct pairing of the first two source rows of each of the 25
generated SumCheck rounds and the two terminal equality rows with all 52
retained compiler obligations, exact sparse A/B/C equality, unique source-row
ownership inside this retained set, fail-closed source decoding, and transport
from every retained compiler obligation to its physical source-row equation.

Does not own: any eliminated source row, rewrite-program execution, selected
row satisfaction, selector or constant-one enforcement, transcript order,
parent or raw-child authority, commitment binding, costs, or row removal.

Emits constraints: none. The executable certificate consumes exactly 52
proof-free `(RawRetainedPair, RawSourceRow)` records. No decoded or
proof-carrying structure is evaluated.

Assurance tier: artifact-checked for the fixed generated production profile.
The certificate compares literal sparse coefficients and source-row indices;
stage labels, digests, counts, and contiguous intervals are not semantic
authority.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedSourceArtifact

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs

private def roundRows (rows : List RawSourceRow) : List RawSourceRow :=
  rows.take 2

/-- The two claimed-sum checks at the head of every exact 30-row production
round, in generated round order. -/
def roundRetainedSourceRows : List RawSourceRow :=
  roundRows RoundArtifact.round0Rows ++
  roundRows RoundArtifact.round1Rows ++
  roundRows RoundArtifact.round2Rows ++
  roundRows RoundArtifact.round3Rows ++
  roundRows RoundArtifact.round4Rows ++
  roundRows RoundArtifact.round5Rows ++
  roundRows RoundArtifact.round6Rows ++
  roundRows RoundArtifact.round7Rows ++
  roundRows RoundArtifact.round8Rows ++
  roundRows RoundArtifact.round9Rows ++
  roundRows RoundArtifact.round10Rows ++
  roundRows RoundArtifact.round11Rows ++
  roundRows RoundArtifact.round12Rows ++
  roundRows RoundArtifact.round13Rows ++
  roundRows RoundArtifact.round14Rows ++
  roundRows RoundArtifact.round15Rows ++
  roundRows RoundArtifact.round16Rows ++
  roundRows RoundArtifact.round17Rows ++
  roundRows RoundArtifact.round18Rows ++
  roundRows RoundArtifact.round19Rows ++
  roundRows RoundArtifact.round20Rows ++
  roundRows RoundArtifact.round21Rows ++
  roundRows RoundArtifact.round22Rows ++
  roundRows RoundArtifact.round23Rows ++
  roundRows RoundArtifact.round24Rows

/-- The two final terminal-equality checks. Their absolute source rows are
`4299099` and `4299100`; the certificate below checks those indices and all
sparse coefficients through the retained provenance rather than trusting the
offset. -/
def terminalRetainedSourceRows : List RawSourceRow :=
  (SourceRows.Chunk62.values.drop 83).take 2

def retainedSourceRows : List RawSourceRow :=
  roundRetainedSourceRows ++ terminalRetainedSourceRows

structure RawRetainedSourcePair where
  compiler : RawRetainedPair
  source : RawSourceRow
deriving DecidableEq, Repr

def retainedSourcePairs : List RawRetainedSourcePair :=
  List.zipWith RawRetainedSourcePair.mk
    SelectiveArtifactPairs.Chunks.retainedPairs retainedSourceRows

/-- Exact source/provenance equality for one retained row. Constants are
compared after placing them on column zero, exactly as the compiler evaluator
does. -/
def RawRetainedSourcePairValid (pair : RawRetainedSourcePair) : Prop :=
  SourceDecodeBridge.RawSourceRowValid pair.source ∧
  pair.source.rows = Metadata.sourceRelationRows ∧
  pair.source.columns = Metadata.sourceRelationColumns ∧
  pair.source.sourceRow = pair.compiler.provenance.sourceRow ∧
  SourceDecodeBridge.rawTerms pair.source.a =
    SourceAssignment.RawLinearCombination.programTerms
      pair.compiler.provenance.a ∧
  SourceDecodeBridge.rawTerms pair.source.b =
    SourceAssignment.RawLinearCombination.programTerms
      pair.compiler.provenance.b ∧
  SourceDecodeBridge.rawTerms pair.source.c =
    SourceAssignment.RawLinearCombination.programTerms
      pair.compiler.provenance.c

instance (pair : RawRetainedSourcePair) :
    Decidable (RawRetainedSourcePairValid pair) := by
  unfold RawRetainedSourcePairValid SourceDecodeBridge.RawSourceRowValid
    SourceDecodeBridge.RawTermValid
  infer_instance

/-- One bounded certificate covers the exact 50+2 schedule, lockstep compiler
and source projections, unique source owners, and every literal coefficient
match. -/
def GeneratedCertificate : Prop :=
  roundRetainedSourceRows.length = 50 ∧
  terminalRetainedSourceRows.length = 2 ∧
  retainedSourcePairs.length = 52 ∧
  retainedSourcePairs.map RawRetainedSourcePair.compiler =
    SelectiveArtifactPairs.Chunks.retainedPairs ∧
  retainedSourcePairs.map RawRetainedSourcePair.source = retainedSourceRows ∧
  (retainedSourcePairs.map
    (fun pair => pair.source.sourceRow)).Nodup ∧
  ∀ pair ∈ retainedSourcePairs, RawRetainedSourcePairValid pair

instance : Decidable GeneratedCertificate := by
  unfold GeneratedCertificate
  infer_instance

set_option maxRecDepth 100000 in
theorem generatedCertificate : GeneratedCertificate := by
  native_decide

theorem roundRetainedSourceRowCountExact :
    roundRetainedSourceRows.length = 50 :=
  generatedCertificate.1

theorem terminalRetainedSourceRowCountExact :
    terminalRetainedSourceRows.length = 2 :=
  generatedCertificate.2.1

theorem retainedSourcePairCountExact : retainedSourcePairs.length = 52 :=
  generatedCertificate.2.2.1

theorem compilerPairsExact :
    retainedSourcePairs.map RawRetainedSourcePair.compiler =
      SelectiveArtifactPairs.Chunks.retainedPairs :=
  generatedCertificate.2.2.2.1

theorem sourceRowsExact :
    retainedSourcePairs.map RawRetainedSourcePair.source =
      retainedSourceRows :=
  generatedCertificate.2.2.2.2.1

/-- No physical source check is owned twice inside the 52-row retained set. -/
theorem sourceOwnersUnique :
    (retainedSourcePairs.map
      (fun pair => pair.source.sourceRow)).Nodup :=
  generatedCertificate.2.2.2.2.2.1

theorem pairValid {pair : RawRetainedSourcePair}
    (member : pair ∈ retainedSourcePairs) :
    RawRetainedSourcePairValid pair :=
  generatedCertificate.2.2.2.2.2.2 pair member

private theorem sourceRowIndex_eq_of_decode {raw : RawSourceRow}
    {decoded : DecodedSourceRow}
    (decodes : decodeSourceRow raw = some decoded) :
    decoded.sourceRow.val = raw.sourceRow := by
  unfold decodeSourceRow at decodes
  split at decodes
  next version =>
    split at decodes
    next rowsPositive =>
      split at decodes
      next columnsPositive =>
        split at decodes
        next rowInRange =>
          cases aResult : decodeTerms raw.columns raw.a with
          | none => simp [aResult] at decodes
          | some a =>
              cases bResult : decodeTerms raw.columns raw.b with
              | none => simp [aResult, bResult] at decodes
              | some b =>
                  cases cResult : decodeTerms raw.columns raw.c with
                  | none => simp [aResult, bResult, cResult] at decodes
                  | some c =>
                      simp [aResult, bResult, cResult] at decodes
                      subst decoded
                      rfl
        next rowOutside => simp at decodes
      next columnsEmpty => simp at decodes
    next rowsEmpty => simp at decodes
  next wrongVersion => simp at decodes

private theorem sourceSideValues_eq_rawTerms {raw : RawSourceRow}
    {decoded : DecodedSourceRow}
    (decodes : decodeSourceRow raw = some decoded)
    (assignment : Nat → Nat) :
    SelectiveCompilerBridge.sourceSideValue decoded.a assignment =
        fieldResidue (lcEval assignment (SourceDecodeBridge.rawTerms raw.a)) ∧
    SelectiveCompilerBridge.sourceSideValue decoded.b assignment =
        fieldResidue (lcEval assignment (SourceDecodeBridge.rawTerms raw.b)) ∧
    SelectiveCompilerBridge.sourceSideValue decoded.c assignment =
        fieldResidue (lcEval assignment (SourceDecodeBridge.rawTerms raw.c)) := by
  have rowEquality :=
    SourceDecodeBridge.sourceRowToRow_eq_rawRow_of_decode decodes
  have aTerms : termsAsNatTerms decoded.a = SourceDecodeBridge.rawTerms raw.a := by
    simpa [sourceRowToRow, SourceDecodeBridge.rawRow] using
      congrArg (fun row : Row => row.a) rowEquality
  have bTerms : termsAsNatTerms decoded.b = SourceDecodeBridge.rawTerms raw.b := by
    simpa [sourceRowToRow, SourceDecodeBridge.rawRow] using
      congrArg (fun row : Row => row.b) rowEquality
  have cTerms : termsAsNatTerms decoded.c = SourceDecodeBridge.rawTerms raw.c := by
    simpa [sourceRowToRow, SourceDecodeBridge.rawRow] using
      congrArg (fun row : Row => row.c) rowEquality
  constructor
  · simpa [SelectiveCompilerBridge.sourceSideValue, aTerms]
  constructor
  · simpa [SelectiveCompilerBridge.sourceSideValue, bTerms]
  · simpa [SelectiveCompilerBridge.sourceSideValue, cTerms]

private theorem linearCombinationValue_eq_programTerms
    {raw : RawLinearCombination}
    {decoded : DecodedLinearCombination Metadata.sourceRelationColumns}
    (decodes : decodeLinearCombination Metadata.sourceRelationColumns raw =
      some decoded)
    (assignment : Nat → Nat) :
    linearCombinationValue decoded
        (SourceAssignment.compilerAssignment assignment) =
      fieldResidue
        (lcEval (SourceAssignment.compilerAssignment assignment)
          (SourceAssignment.RawLinearCombination.programTerms raw)) := by
  calc
    linearCombinationValue decoded
        (SourceAssignment.compilerAssignment assignment) =
        SelectiveArtifactPairs.evalLinearForm
          (fun column => fieldResidue (assignment column))
          (SelectiveArtifactPairs.sourceLinearForm raw) :=
      SelectiveArtifactPairs.linearCombinationValue_eq_evalSourceLinearForm
        decodes assignment
    _ = fieldResidue
        (lcEval (SourceAssignment.compilerAssignment assignment)
          (SourceAssignment.RawLinearCombination.programTerms raw)) :=
      SelectiveArtifactPairs.evalSourceLinearForm assignment raw

private theorem retainedStepData_eq_programTerms
    {raw : RawRetainedStep}
    {decoded : DecodedRetainedStep Metadata.sourceRelationColumns}
    (decodes : decodeRetainedStep Metadata.sourceRelationRows
      Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
        some decoded)
    (assignment : Nat → Nat) :
    decoded.sourceRow = raw.sourceRow ∧
    linearCombinationValue decoded.a
        (SourceAssignment.compilerAssignment assignment) =
      fieldResidue
        (lcEval (SourceAssignment.compilerAssignment assignment)
          (SourceAssignment.RawLinearCombination.programTerms raw.a)) ∧
    linearCombinationValue decoded.b
        (SourceAssignment.compilerAssignment assignment) =
      fieldResidue
        (lcEval (SourceAssignment.compilerAssignment assignment)
          (SourceAssignment.RawLinearCombination.programTerms raw.b)) ∧
    linearCombinationValue decoded.c
        (SourceAssignment.compilerAssignment assignment) =
      fieldResidue
        (lcEval (SourceAssignment.compilerAssignment assignment)
          (SourceAssignment.RawLinearCombination.programTerms raw.c)) := by
  unfold decodeRetainedStep at decodes
  split at decodes
  next emittedBound =>
    split at decodes
    next sourceBound =>
      cases aResult : decodeLinearCombination
          Metadata.sourceRelationColumns raw.a with
      | none => simp [aResult] at decodes
      | some a =>
          cases bResult : decodeLinearCombination
              Metadata.sourceRelationColumns raw.b with
          | none => simp [aResult, bResult] at decodes
          | some b =>
              cases cResult : decodeLinearCombination
                  Metadata.sourceRelationColumns raw.c with
              | none => simp [aResult, bResult, cResult] at decodes
              | some c =>
                  simp [aResult, bResult, cResult] at decodes
                  subst decoded
                  exact ⟨rfl,
                    linearCombinationValue_eq_programTerms aResult assignment,
                    linearCombinationValue_eq_programTerms bResult assignment,
                    linearCombinationValue_eq_programTerms cResult assignment⟩
    next sourceOutside => simp at decodes
  next emittedOutside => simp at decodes

/-- A literal source/provenance certificate gives the semantic match required
by `retainedStepHolds_implies_sourceRowHolds`. No row-satisfaction premise is
used. -/
theorem retainedSourceMatches_of_valid
    {pair : RawRetainedSourcePair}
    (valid : RawRetainedSourcePairValid pair)
    (assignment : Nat → Nat)
    {provenance : DecodedRetainedStep Metadata.sourceRelationColumns}
    {source : DecodedSourceRow}
    (provenanceDecodes :
      decodeRetainedStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows
          pair.compiler.provenance = some provenance)
    (sourceDecodes : decodeSourceRow pair.source = some source) :
    SelectiveCompilerBridge.RetainedSourceMatches
      (SourceAssignment.compilerAssignment assignment) provenance source := by
  rcases sourceSideValues_eq_rawTerms sourceDecodes
      (SourceAssignment.compilerAssignment assignment) with
    ⟨sourceA, sourceB, sourceC⟩
  rcases retainedStepData_eq_programTerms provenanceDecodes assignment with
    ⟨provenanceRow, provenanceA, provenanceB, provenanceC⟩
  constructor
  · calc
      source.sourceRow.val = pair.source.sourceRow :=
        sourceRowIndex_eq_of_decode sourceDecodes
      _ = pair.compiler.provenance.sourceRow := valid.2.2.2.1
      _ = provenance.sourceRow := provenanceRow.symm
  constructor
  · calc
      SelectiveCompilerBridge.sourceSideValue source.a
          (SourceAssignment.compilerAssignment assignment) =
          fieldResidue
            (lcEval (SourceAssignment.compilerAssignment assignment)
              (SourceDecodeBridge.rawTerms pair.source.a)) := sourceA
      _ = fieldResidue
            (lcEval (SourceAssignment.compilerAssignment assignment)
              (SourceAssignment.RawLinearCombination.programTerms
                pair.compiler.provenance.a)) := by rw [valid.2.2.2.2.1]
      _ = linearCombinationValue provenance.a
            (SourceAssignment.compilerAssignment assignment) :=
          provenanceA.symm
  constructor
  · calc
      SelectiveCompilerBridge.sourceSideValue source.b
          (SourceAssignment.compilerAssignment assignment) =
          fieldResidue
            (lcEval (SourceAssignment.compilerAssignment assignment)
              (SourceDecodeBridge.rawTerms pair.source.b)) := sourceB
      _ = fieldResidue
            (lcEval (SourceAssignment.compilerAssignment assignment)
              (SourceAssignment.RawLinearCombination.programTerms
                pair.compiler.provenance.b)) := by rw [valid.2.2.2.2.2.1]
      _ = linearCombinationValue provenance.b
            (SourceAssignment.compilerAssignment assignment) :=
          provenanceB.symm
  · calc
      SelectiveCompilerBridge.sourceSideValue source.c
          (SourceAssignment.compilerAssignment assignment) =
          fieldResidue
            (lcEval (SourceAssignment.compilerAssignment assignment)
              (SourceDecodeBridge.rawTerms pair.source.c)) := sourceC
      _ = fieldResidue
            (lcEval (SourceAssignment.compilerAssignment assignment)
              (SourceAssignment.RawLinearCombination.programTerms
                pair.compiler.provenance.c)) := by rw [valid.2.2.2.2.2.2]
      _ = linearCombinationValue provenance.c
            (SourceAssignment.compilerAssignment assignment) :=
          provenanceC.symm

private theorem compilerPairMember {pair : RawRetainedSourcePair}
    (member : pair ∈ retainedSourcePairs) :
    pair.compiler ∈ SelectiveArtifactPairs.Chunks.retainedPairs := by
  have mapped : pair.compiler ∈
      retainedSourcePairs.map RawRetainedSourcePair.compiler :=
    List.mem_map.mpr ⟨pair, member, rfl⟩
  rw [compilerPairsExact] at mapped
  exact mapped

/-- Source-row satisfaction for all 52 exact retained physical rows. Decode
failure is excluded by the artifact certificate, while the row equations are
derived only from the caller's retained compiler obligations. -/
def GeneratedRetainedSourceRowsSatisfy (assignment : Nat → Nat) : Prop :=
  ∀ raw ∈ retainedSourceRows,
    ∀ decoded, decodeSourceRow raw = some decoded →
      SourceRowHolds decoded (SourceAssignment.compilerAssignment assignment)

theorem allRetainedObligationsHold_implies_sourceRowsSatisfy
    {assignment : Nat → Nat}
    (obligations :
      SelectiveArtifactPairs.Artifact.AllRetainedObligationsHold assignment) :
    GeneratedRetainedSourceRowsSatisfy assignment := by
  intro raw rawMember decoded rawDecodes
  have mappedMember : raw ∈
      retainedSourcePairs.map RawRetainedSourcePair.source := by
    rw [sourceRowsExact]
    exact rawMember
  rcases List.mem_map.mp mappedMember with ⟨pair, pairMember, rfl⟩
  rcases obligations pair.compiler (compilerPairMember pairMember) with
    ⟨_emitted, provenance, _emittedDecodes, provenanceDecodes,
      provenanceHolds⟩
  have valid := pairValid pairMember
  rcases SourceDecodeBridge.decodeSourceRow_of_valid valid.1 with
    ⟨source, sourceDecodes⟩
  have decodedEquality : decoded = source :=
    Option.some.inj (rawDecodes.symm.trans sourceDecodes)
  subst decoded
  exact SelectiveCompilerBridge.retainedStepHolds_implies_sourceRowHolds
    provenanceHolds
    (retainedSourceMatches_of_valid valid assignment provenanceDecodes
      sourceDecodes)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedSourceArtifact
