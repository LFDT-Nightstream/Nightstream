import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Parameters
import Nightstream.SuperNeo.SumCheck.FixedPhase

/-!
Physical mixed-width SumCheck interface for the independent Split-NC FE
polynomial.

Owns: exact row/lane message arities, physical serialization, the semantic-
only uniform proof view, and the verifier-visible claimed-chain checker.

Does not own: honest message construction, FE completeness, transcript
derivation, raw decoding, Poseidon2, Rust, R1CS, rows, removals, or costs.

Emits constraints: no.

Authority boundary: `rawRounds` is the only serialization projection. It
contains row messages at `Drow + 1` slots and lane messages at exactly three
slots. `uniformRounds` only appends verifier-known high zeros for generic
fixed-phase reasoning and must never be absorbed, allocated, or counted.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.sumcheck.certificate.row` | one `Drow + 1`-slot message per row variable | checked by type | `Certificate.rowRounds` |
| `nifs.pi_ccs.fe.sumcheck.certificate.lane` | one three-slot message per lane variable | checked by type | `Certificate.laneRounds` |
| `nifs.pi_ccs.fe.sumcheck.serialization` | serialize row messages before lane messages at their physical widths | direct dataflow | `Certificate.rawRounds` |
| `nifs.pi_ccs.fe.sumcheck.proof_view` | widen lane messages with verifier-known high zeros only | derived | `Certificate.uniformRounds`, `lane_evaluate_uniform` |
| `nifs.pi_ccs.fe.sumcheck.chain` | one claimed chain crosses the row/lane cut | checked | `Accepted`, `check` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev RawMessage :=
  Nightstream.SuperNeo.SumCheck.Finite.Message K

/-- Syntax-derived row degree owned by the authoritative FE public input. -/
abbrev Drow
    {shape : SemanticShape}
    (input : PublicInput shape) : Nat :=
  rowSumcheckDegreeBound input

/-- Physical FE row-round message. -/
abbrev RowMessage
    {shape : SemanticShape}
    (input : PublicInput shape) :=
  FixedPolynomial K (Drow input)

/-- Physical FE lane-round message. Its degree is independently two. -/
abbrev LaneMessage :=
  FixedPolynomial K laneSumcheckDegreeBound

/-- The independently proved lane degree fits the syntax-derived row ceiling. -/
theorem laneDegree_le_Drow
    {shape : SemanticShape}
    (input : PublicInput shape) :
    laneSumcheckDegreeBound <= Drow input := by
  unfold laneSumcheckDegreeBound Drow rowSumcheckDegreeBound
  exact Nat.le_max_right _ _

/-- Append verifier-known high zeros to one lane polynomial for algebraic
reuse of the uniform-degree reduction. -/
def laneToUniform
    {shape : SemanticShape}
    (input : PublicInput shape)
    (message : LaneMessage) : RowMessage input :=
  FixedPolynomial.widen ops.toOps (laneDegree_le_Drow input) message

/-- Semantic widening leaves the physical three-coefficient lane evaluator
unchanged at every challenge. -/
theorem lane_evaluate_uniform
    {shape : SemanticShape}
    (input : PublicInput shape)
    (message : LaneMessage)
    (point : K) :
    (laneToUniform input message).evaluate ops.toOps point =
      message.evaluate ops.toOps point := by
  exact FixedPolynomial.evaluate_widen ops.toOps polynomialLaws
    (laneDegree_le_Drow input) message point

/-- Prover-visible mixed-width FE certificate. Exact phase counts are part of
the type; challenges, transcript states, and semantic expected polynomials
cannot be represented as fields. -/
structure Certificate
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain) where
  rowRounds : Fin shape.rowVariables -> RowMessage input
  laneRounds : Fin domain.laneVariables -> LaneMessage

namespace Certificate

/-- Physical row serialization, still separate from the lane phase so width
ownership remains mechanically visible. -/
def rowRawRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) : List RawMessage :=
  (List.ofFn certificate.rowRounds).map FixedPolynomial.toMessage

/-- Physical lane serialization. These messages always contain exactly three
extension-field coefficients, even when `Drow` is larger than two. -/
def laneRawRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) : List RawMessage :=
  (List.ofFn certificate.laneRounds).map FixedPolynomial.toMessage

/-- The sole physical serialization order: all row rounds, then all lane
rounds, each at its own exact width. -/
def rawRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) : List RawMessage :=
  certificate.rowRawRounds ++ certificate.laneRawRounds

/-- Algebraic proof view used only by the uniform generic reduction. Lane
high zeros are computed constants, not certificate data. -/
def uniformRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) :
    List (RowMessage input) :=
  List.ofFn certificate.rowRounds ++
    (List.ofFn certificate.laneRounds).map (laneToUniform input)

/-- Physical serialization has one message per FE coordinate. -/
@[simp] theorem rawRounds_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) :
    certificate.rawRounds.length =
      shape.rowVariables + domain.laneVariables := by
  simp [rawRounds, rowRawRounds, laneRawRounds]

/-- The physical row prefix has the verifier-owned row-round count. -/
@[simp] theorem rowRawRounds_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) :
    certificate.rowRawRounds.length = shape.rowVariables := by
  simp [rowRawRounds]

/-- The physical lane suffix has the verifier-owned lane-round count. -/
@[simp] theorem laneRawRounds_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) :
    certificate.laneRawRounds.length = domain.laneVariables := by
  simp [laneRawRounds]

/-- Splitting physical serialization at the verifier-owned row count recovers
the row prefix exactly. -/
@[simp] theorem rawRounds_take_rowVariables
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) :
    certificate.rawRounds.take shape.rowVariables =
      certificate.rowRawRounds := by
  calc
    certificate.rawRounds.take shape.rowVariables =
        (certificate.rowRawRounds ++ certificate.laneRawRounds).take
          certificate.rowRawRounds.length := by
      rw [rawRounds, rowRawRounds_length]
    _ = certificate.rowRawRounds := List.take_left

/-- The same authoritative split recovers the physical lane suffix exactly. -/
@[simp] theorem rawRounds_drop_rowVariables
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) :
    certificate.rawRounds.drop shape.rowVariables =
      certificate.laneRawRounds := by
  rw [rawRounds]
  simp

/-- Every physical row message has the syntax-derived exact width. -/
theorem rowRawRounds_width
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain)
    (message : RawMessage)
    (member : message ∈ certificate.rowRawRounds) :
    message.coefficients.length = Drow input + 1 := by
  simp only [rowRawRounds, List.mem_map] at member
  rcases member with ⟨typed, _, rfl⟩
  exact typed.toMessage_coefficients_length

/-- Every physical lane message has exactly three coefficient slots. -/
theorem laneRawRounds_width
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain)
    (message : RawMessage)
    (member : message ∈ certificate.laneRawRounds) :
    message.coefficients.length = 3 := by
  simp only [laneRawRounds, List.mem_map] at member
  rcases member with ⟨typed, _, rfl⟩
  simp [laneSumcheckDegreeBound]

/-- The semantic proof view has the same round count as physical
serialization. -/
@[simp] theorem uniformRounds_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Certificate input domain) :
    certificate.uniformRounds.length =
      shape.rowVariables + domain.laneVariables := by
  simp [uniformRounds]

end Certificate

/-- Verifier-visible FE acceptance is one claimed chain across both physical
phases. The terminal remains an explicit verifier-owned input. -/
def Accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (initial terminal : K)
    (point : Point shape domain)
    (certificate : Certificate input domain) : Prop :=
  FixedPhase.Chain ops.toOps initial certificate.uniformRounds
    point.coordinates terminal

/-- Executable mixed-width FE claimed-chain checker. Semantic lane widening
uses only fixed high-zero constants. -/
def check
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (initial terminal : K)
    (point : Point shape domain)
    (certificate : Certificate input domain) : Bool :=
  FixedPhase.checkChain ops.toOps initial certificate.uniformRounds
    point.coordinates terminal

/-- Executable and logical FE claimed-chain acceptance coincide exactly. -/
theorem check_eq_true_iff_accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (initial terminal : K)
    (point : Point shape domain)
    (certificate : Certificate input domain) :
    check initial terminal point certificate = true <->
      Accepted initial terminal point certificate :=
  FixedPhase.checkChain_eq_true_iff ops.toOps initial terminal
    certificate.uniformRounds point.coordinates

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe
