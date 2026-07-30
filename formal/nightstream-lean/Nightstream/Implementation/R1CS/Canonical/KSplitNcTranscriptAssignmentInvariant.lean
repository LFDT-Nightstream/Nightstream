import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPlacement
import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases
import Nightstream.Implementation.R1CS.Canonical.KSplitNcOperational

/-!
Contract: prove that the value-level Split-NC transcript input is unchanged
when an assignment extension preserves every caller-owned source below the
transcript base.

This is the completeness-side bridge used after installing the honest
Poseidon2 replay witness. It changes no row, challenge, certificate, or
verifier relation.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1800000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptAssignmentInvariant

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev KColumns :=
  Nightstream.Implementation.R1CS.ProjectionProgram.KColumns

private theorem fieldValues_eq
    {base : Nat} (left right : Nat → Nat)
    (agree : ∀ column, column < base → left column = right column)
    (fields : List LinComb)
    (below :
      ∀ field ∈ fields,
        SymbolicDuplexPlacement.ValueInPrefix base field) :
    KSplitNcTranscriptSemantics.fieldValues left fields =
      KSplitNcTranscriptSemantics.fieldValues right fields := by
  unfold KSplitNcTranscriptSemantics.fieldValues
  apply List.map_congr_left
  intro field member
  exact KMulHonest.lcEval_congr left right field
    (fun column mentioned => agree column (below field member column mentioned))

private theorem columns_value_eq
    {base : Nat} (left right : Nat → Nat)
    (agree : ∀ column, column < base → left column = right column)
    (columns : KColumns)
    (low : columns.c0 < base) (high : columns.c1 < base) :
    columns.value left = columns.value right := by
  rcases columns with ⟨c0, c1⟩
  simp only [
    Nightstream.Implementation.R1CS.ProjectionProgram.KColumns.value,
    Nightstream.Implementation.R1CS.ProjectionProgram.baseAt]
  rw [agree c0 low, agree c1 high]

private theorem fixedPolynomial_eq_of_coefficients_eq
    {Field : Type} {degree : Nat}
    {left right : FixedPolynomial Field degree}
    (equal : left.coefficients = right.coefficients) :
    left = right := by
  cases left with
  | mk leftCoefficients leftLength =>
      cases right with
      | mk rightCoefficients rightLength =>
          simp only at equal
          subst rightCoefficients
          rfl

private theorem round_eq
    {degree base : Nat} (left right : Nat → Nat)
    (agree : ∀ column, column < base → left column = right column)
    (round : RoundColumns degree)
    (below :
      ∀ field ∈ KSplitNcTranscript.roundFields round,
        SymbolicDuplexPlacement.ValueInPrefix base field) :
    round.paperPolynomial left = round.paperPolynomial right := by
  apply fixedPolynomial_eq_of_coefficients_eq
  simp only [RoundColumns.paperPolynomial]
  apply List.map_congr_left
  intro columns member
  apply congrArg KConcreteFixedPhaseBridge.ofProjection
  apply columns_value_eq left right agree columns
  · exact
      below (carried columns).low
          (by
            unfold KSplitNcTranscript.roundFields
            apply List.mem_flatMap.mpr
            exact ⟨columns, member, by
              simp [KSplitNcTranscript.carriedFields]⟩)
        columns.c0 (by simp [carried, Mentions])
  · exact
      below (carried columns).high
          (by
            unfold KSplitNcTranscript.roundFields
            apply List.mem_flatMap.mpr
            exact ⟨columns, member, by
              simp [KSplitNcTranscript.carriedFields]⟩)
        columns.c1 (by simp [carried, Mentions])

private theorem serialization_eq_of_fields_eq
    {VerifierKey : Type} {Input : Type} {shape : SemanticShape}
    (left right :
      KSplitNcPoseidonSchedule.Serialization VerifierKey Input shape)
    (statement :
      left.statementFields = right.statementFields)
    (output :
      left.outputFields = right.outputFields) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem ncCertificate_eq_of_rounds
    {domain : BlockNcDomain}
    (left right : Transcript.Nc.BlockLane.Certificate domain)
    (rounds : left.rounds = right.rounds) :
    left = right := by
  cases left
  cases right
  simp_all

/-- The selected value schedule reads only caller-owned fields. -/
theorem valueSchedule_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (left right : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix input)
    (agree :
      ∀ column, column < input.transcriptBase →
        left column = right column) :
    KSplitNcTranscriptSemantics.valueSchedule constants left input =
      KSplitNcTranscriptSemantics.valueSchedule constants right input := by
  unfold KSplitNcTranscriptSemantics.valueSchedule
  apply congrArg (KSplitNcPoseidonSchedule.schedule constants)
  apply serialization_eq_of_fields_eq
  · funext statement
    exact fieldValues_eq left right agree input.statementFields
      sources.statement
  · funext output
    exact fieldValues_eq left right agree input.outputFields
      sources.output

/-- The caller-owned prior duplex state is unchanged by the extension. -/
theorem priorState_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix input)
    (agree :
      ∀ column, column < input.transcriptBase →
        left column = right column) :
    KSplitNcTranscriptSemantics.priorState left input =
      KSplitNcTranscriptSemantics.priorState right input := by
  unfold KSplitNcTranscriptSemantics.priorState
    SymbolicDuplexSemantics.decodedBuilder
    KSplitNcTranscript.initialBuilder SymbolicDuplex.start
    SymbolicDuplexSemantics.evalState
  congr 1
  · funext lane
    exact KMulHonest.lcEval_congr left right (input.priorLanes lane)
      (fun column mentioned =>
        agree column (sources.prior lane column mentioned))

/-- FE messages decode to the same certificate after extending the
assignment above the transcript base. -/
theorem feCertificate_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix input)
    (agree :
      ∀ column, column < input.transcriptBase →
        left column = right column) :
    KSplitNcTranscriptPhases.feCertificate left input =
      KSplitNcTranscriptPhases.feCertificate right input := by
  unfold KSplitNcTranscriptPhases.feCertificate
  congr 1
  · funext index
    apply round_eq left right agree
    exact sources.feRow (input.fe.rowRounds index)
      (List.mem_ofFn.2 ⟨index, rfl⟩)
  · funext index
    apply round_eq left right agree
    exact sources.feLane (input.fe.laneRounds index)
      (List.mem_ofFn.2 ⟨index, rfl⟩)

/-- NC messages decode to the same certificate after extending the
assignment above the transcript base. -/
theorem ncCertificate_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix input)
    (agree :
      ∀ column, column < input.transcriptBase →
        left column = right column) :
    KSplitNcTranscriptPhases.ncCertificate left input =
    KSplitNcTranscriptPhases.ncCertificate right input := by
  apply ncCertificate_eq_of_rounds
  funext index
  refine Fin.addCases (motive := fun index =>
    (KSplitNcTranscriptPhases.ncCertificate left input).rounds index =
      (KSplitNcTranscriptPhases.ncCertificate right input).rounds index)
    ?_ ?_ index
  · intro block
    simpa only [KSplitNcTranscriptPhases.ncCertificate,
      Fin.addCases_left] using
      round_eq left right agree (input.nc.blockRounds block)
        (sources.ncBlock (input.nc.blockRounds block)
          (List.mem_ofFn.2 ⟨block, rfl⟩))
  · intro lane
    simpa only [KSplitNcTranscriptPhases.ncCertificate,
      Fin.addCases_right] using
      round_eq left right agree (input.nc.laneRounds lane)
        (sources.ncLane (input.nc.laneRounds lane)
          (List.mem_ofFn.2 ⟨lane, rfl⟩))

/-- The complete operational certificate is unchanged by an assignment
extension above the transcript base. -/
theorem certificate_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix input)
    (agree :
      ∀ column, column < input.transcriptBase →
        left column = right column)
    (message : OutputMessage shape) :
    KSplitNcOperational.certificate left input message =
      KSplitNcOperational.certificate right input message := by
  unfold KSplitNcOperational.certificate
  rw [feCertificate_eq left right input sources agree,
    ncCertificate_eq left right input sources agree]

/-- Selected operational acceptance depends only on caller-owned transcript
sources, so installing replay witnesses above the transcript base preserves
it exactly. -/
theorem accepted_iff
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (left right : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix input)
    (agree :
      ∀ column, column < input.transcriptBase →
        left column = right column)
    (message : OutputMessage shape) :
    Protocol.BlockLane.Accepted
        (fun _ : Unit => polynomialInput)
        (KSplitNcTranscriptSemantics.valueSchedule constants left input)
        (KSplitNcTranscriptSemantics.priorState left input)
        profile KSplitNcTranscriptSemantics.unitStatement
        (KSplitNcOperational.certificate left input message) ↔
      Protocol.BlockLane.Accepted
        (fun _ : Unit => polynomialInput)
        (KSplitNcTranscriptSemantics.valueSchedule constants right input)
        (KSplitNcTranscriptSemantics.priorState right input)
        profile KSplitNcTranscriptSemantics.unitStatement
        (KSplitNcOperational.certificate right input message) := by
  rw [valueSchedule_eq constants left right input sources agree,
    priorState_eq left right input sources agree,
    certificate_eq left right input sources agree message]

/-- Pre-SumCheck transcript derivation is extension-invariant. -/
theorem semanticPre_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (left right : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix input)
    (agree :
      ∀ column, column < input.transcriptBase →
        left column = right column) :
    KSplitNcTranscriptPhases.semanticPre constants left input =
      KSplitNcTranscriptPhases.semanticPre constants right input := by
  unfold KSplitNcTranscriptPhases.semanticPre
  rw [valueSchedule_eq constants left right input sources agree,
    priorState_eq left right input sources agree]

/-- FE transcript execution is extension-invariant. -/
theorem semanticFeExecution_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (left right : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix input)
    (agree :
      ∀ column, column < input.transcriptBase →
        left column = right column) :
    KSplitNcTranscriptPhases.semanticFeExecution
        profile constants left input =
      KSplitNcTranscriptPhases.semanticFeExecution
        profile constants right input := by
  unfold KSplitNcTranscriptPhases.semanticFeExecution
    KSplitNcTranscriptPhases.semanticFeMachine
    KSplitNcTranscriptPhases.semanticFeInitial
  rw [valueSchedule_eq constants left right input sources agree,
    semanticPre_eq constants left right input sources agree,
    feCertificate_eq left right input sources agree]

/-- NC transcript execution is extension-invariant. -/
theorem semanticNcExecution_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (left right : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix input)
    (agree :
      ∀ column, column < input.transcriptBase →
        left column = right column) :
    KSplitNcTranscriptPhases.semanticNcExecution
        profile constants left input =
      KSplitNcTranscriptPhases.semanticNcExecution
        profile constants right input := by
  unfold KSplitNcTranscriptPhases.semanticNcExecution
  rw [valueSchedule_eq constants left right input sources agree,
    semanticFeExecution_eq profile constants left right input sources agree,
    ncCertificate_eq left right input sources agree]

end Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptAssignmentInvariant
