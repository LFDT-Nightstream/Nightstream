import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicCarrier

/-!
Production public-input boundary for the fixed F' SuperNeo profile.

Owns: kernel-checked dimension facts comparing the production `X` carrier
with SuperNeo's paper requirement `n_F,in = d * n_R,in`, and one concrete
non-injectivity witness for the current 270-to-257 projection.

Does not own: a repaired public-input representation, a proof that the current
Rust carrier instantiates the paper CE relation, or permission to remove any
constraint. The counterexample is a refinement blocker, not a replacement
semantic model. It proves that the scalar projection alone cannot authorize
the packed carrier; it does not claim that the particular tail-only witness is
reachable from an honest CE execution. Reachability and closure under the
actual ring action remain separate relation theorems.

| Fact | Mathematical obligation | Result | Consequence |
|---|---|---:|---|
| `productionPublicWidth` | scalar CCS public width | 257 | implementation datum |
| `productionRingDegree` | SuperNeo coefficient degree `d` | 54 | paper parameter |
| `productionActiveRingColumns` | `ceil(257 / 54)` | 5 | implementation carrier width |
| `productionPackedWidth` | `54 * 5` | 270 | full Π_RLC/Π_DEC carrier |
| `productionPublicRemainder` | `257 mod 54` | 41 | paper alignment fails |
| `productionExtraCoefficients` | `270 - 257` | 13 | not authorized by a 257-field projection |
| `publicProjection_not_injective` | full carrier determined by scalar public input | false | an explicit refinement is required |
| `ringAction_enters_extra_coefficient` | closure of rows 0..40 in the fifth ring column under multiplication | false | scalar truncation is not Π_RLC-closed |
| `DistinguishedFreshOutputBinding` | current fresh/output equality surface | 257 projected coordinates only | does not establish CE carrier authority |
| `distinguishedBinding_not_sufficient_for_ceCarrier` | current binding implies `X = L_in(z)` | false | full-carrier authority is a necessary additional obligation |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PublicInputBoundary

open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper

def productionPublicWidth : Nat := layout.parent.mIn

def productionRingDegree : Nat := layout.ringDimension

def productionActiveRingColumns : Nat := activeColumns layout

def productionPackedWidth : Nat :=
  productionRingDegree * productionActiveRingColumns

def productionExtraCoefficients : Nat :=
  productionPackedWidth - productionPublicWidth

/-- SuperNeo's paper profile requires the field public width to be an exact
multiple of the ring degree. -/
def PaperPublicWidthAligned : Prop :=
  productionPublicWidth % productionRingDegree = 0

theorem productionPublicWidth_value : productionPublicWidth = 257 := by
  decide

theorem productionRingDegree_value : productionRingDegree = 54 := by
  decide

theorem productionActiveRingColumns_value : productionActiveRingColumns = 5 := by
  decide

theorem productionPackedWidth_value : productionPackedWidth = 270 := by
  decide

theorem productionPublicRemainder :
    productionPublicWidth % productionRingDegree = 41 := by
  decide

theorem productionExtraCoefficients_value : productionExtraCoefficients = 13 := by
  decide

/-- The current `m_in = 257` profile does not satisfy the paper's exact
`n_F,in = d * n_R,in` dimension precondition. -/
theorem productionPublicWidth_not_aligned : ¬ PaperPublicWidthAligned := by
  unfold PaperPublicWidthAligned
  decide

/-- All-zero representative of the full production Π_RLC/Π_DEC public-input
carrier. -/
def zeroPackedInput : PackedPublicInput :=
  ⟨List.replicate productionPackedWidth (0 : Scalar)⟩

/-- A distinct carrier whose only non-zero coordinate is the final one. That
coordinate is outside the 257 scalar positions selected by
`unpackPublicInput`. -/
def lastTailPackedInput : PackedPublicInput :=
  ⟨zeroPackedInput.data.set (productionPackedWidth - 1) (1 : Scalar)⟩

set_option maxRecDepth 524288 in
theorem publicProjection_not_injective :
    zeroPackedInput ≠ lastTailPackedInput ∧
      unpackPublicInput zeroPackedInput = unpackPublicInput lastTailPackedInput := by
  decide

/-- Coefficient 40 is the final scalar-public row in the fifth active ring
column (`256 = 4 * 54 + 40`). -/
def finalScalarPublicMonomial : Nightstream.SuperNeo.Concrete.RingF :=
  Nightstream.SuperNeo.Concrete.ringFMonomial 40 1

/-- Multiplication by `X`. This is an algebraic closure witness, not yet a
claim that the concrete Fiat-Shamir sampler can emit this challenge; strong-set
membership belongs to the Π_RLC sampler refinement. -/
def shiftByOneMonomial : Nightstream.SuperNeo.Concrete.RingF :=
  Nightstream.SuperNeo.Concrete.ringFMonomial 1 1

set_option maxRecDepth 524288 in
/-- The actual Φ81 quotient-ring action moves the last scalar-public
coefficient into row 41, the first extra coefficient in the fifth ring
column. Thus the 257 selected coordinates are not closed under the operation
used by Π_RLC. -/
theorem ringAction_enters_extra_coefficient :
    finalScalarPublicMonomial ⟨41, by decide⟩ = 0 ∧
      Nightstream.SuperNeo.Concrete.ringFMul
        shiftByOneMonomial finalScalarPublicMonomial
        ⟨41, by decide⟩ = 1 := by
  decide

/-! ## Necessity of full-carrier authority -/

/-- The mathematical content of the current fresh-output wiring: the output
carrier agrees with the fresh statement at the 257 verifier-visible scalar
positions. This deliberately says nothing about the other 13 coefficients. -/
def DistinguishedFreshOutputBinding
    (fresh : Nightstream.SuperNeo.Concrete.PublicInput)
    (output : PackedPublicInput) : Prop :=
  unpackPublicInput output = fresh

/-- The CE obligation omitted by distinguished-coordinate equality: the
claimed output carrier is the actual `L_in(z)` projection of the opening. -/
def CeCarrierAuthority
    (actualProjection claimedOutput : PackedPublicInput) : Prop :=
  claimedOutput = actualProjection

/-- A concrete false-CE-carrier witness accepted by the current mathematical
binding surface. Both packed values project to the same 257-field fresh input,
but the claimed output is not the actual full carrier.

This is a protocol-obligation counterexample, not a claim about one generated
R1CS trace. Any verifier whose only fresh/output `X` obligation is
`DistinguishedFreshOutputBinding` admits this ambiguity. -/
theorem currentBinding_accepts_falseCeCarrier :
    DistinguishedFreshOutputBinding
        (unpackPublicInput zeroPackedInput) lastTailPackedInput ∧
      ¬ CeCarrierAuthority zeroPackedInput lastTailPackedInput := by
  constructor
  · exact publicProjection_not_injective.2.symm
  · intro authority
    exact publicProjection_not_injective.1 authority.symm

/-- Per-obligation necessity result: distinguished fresh equality cannot imply
the full CE projection equation, even when the honest projection has the same
257-field public view. A sound Π_CCS refinement must therefore either:

1. use an aligned public carrier whose full value is already authoritative, or
2. add and prove a heterogeneous reduction obligation binding every extra
   coefficient to the same opening.
-/
theorem distinguishedBinding_not_sufficient_for_ceCarrier :
    ¬ ∀ (actualProjection claimedOutput : PackedPublicInput),
        DistinguishedFreshOutputBinding
            (unpackPublicInput actualProjection) claimedOutput →
          CeCarrierAuthority actualProjection claimedOutput := by
  intro claimedSufficient
  exact currentBinding_accepts_falseCeCarrier.2
    (claimedSufficient zeroPackedInput lastTailPackedInput
      currentBinding_accepts_falseCeCarrier.1)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PublicInputBoundary
