import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Binding

/-!
Pre-SumCheck verifier challenge derivation for the production-shaped
`Pi_CCS` transcript.

Assurance tier: executable implementation semantics. Challenge cardinalities,
domain tags, squeeze counts, and bundle slicing are stated independently of
the Rust verifier and generated R1CS trace.

Owns: the dimension profile needed by this phase; the `[2]` engine-challenge
domain; the ordered `alpha`, `beta_a`, `beta_r`, and `gamma` bundle; the `[3]`
NC-column domain; `beta_m`; and the shared successor state.

Does not own: why these production split coins refine the paper joint
`Pi_CCS` sample, field distribution, SumCheck truth, binding-prefix authority,
native/gadget/R1CS correspondence, costs, or row removal.

Emits constraints: no.

Authority boundary: challenge values are projections of `squeezeN`; they are
not fields of an input structure. The same execution that returns the values
also returns the only successor state accepted by later phases.

| Protocol | Phase | Constraint family | Exact obligation |
|---|---|---|---|
| `Pi_CCS` | engine domain | `engineDomain` | raw append `[2]` |
| `Pi_CCS` | engine response | `engineScalarCount` | squeeze `2 * (2*ellD + ellN + 1)` base fields |
| `Pi_CCS` | engine partition | `Output` | split K values as `alpha`, `beta_a`, `beta_r`, `gamma` |
| `Pi_CCS` | engine shape | `run_alpha_length`, `run_betaA_length`, `run_betaR_length` | every semantic point receives its exact verifier-owned dimension |
| `Pi_CCS` | NC domain | `betaMDomain` | raw append `[3]` |
| `Pi_CCS` | NC response | `betaM` | squeeze `2 * ellM` base fields and pair them |
| `Pi_CCS` | NC shape | `run_betaM_length` | the NC column point receives exactly `ellM` coordinates |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives

set_option maxHeartbeats 1000000

/-- Verifier-fixed logarithmic dimensions used by transcript sampling. -/
structure Shape where
  ellD : Nat
  ellN : Nat
  ellM : Nat
  degreeBound : Nat
deriving DecidableEq

/-- Number of extension elements in `[alpha | beta_a | beta_r | gamma]`. -/
def engineScalarCount (shape : Shape) : Nat :=
  2 * shape.ellD + shape.ellN + 1

/-- Exact raw domain payload for the first challenge batch. -/
def engineDomain : List Field :=
  [wordField 2]

/-- Exact raw domain payload for the NC column challenge. -/
def betaMDomain : List Field :=
  [wordField 3]

/-- All pre-SumCheck verifier responses and their jointly computed successor
state. -/
structure Output where
  alpha : List Extension
  betaA : List Extension
  betaR : List Extension
  gamma : Extension
  betaM : List Extension
  state : State

/-- Execute both domain-separated challenge batches. -/
def run (initial : State) (shape : Shape) : Output :=
  let engineStart := appendRaw initial engineDomain
  let engineFields := squeezeN engineStart (2 * engineScalarCount shape)
  let engineValues := pairFields engineFields.2
  let alpha := engineValues.take shape.ellD
  let betaA := (engineValues.drop shape.ellD).take shape.ellD
  let betaR := (engineValues.drop (2 * shape.ellD)).take shape.ellN
  let gamma := engineValues.getD (2 * shape.ellD + shape.ellN)
    Extension.zero
  let betaMStart := appendRaw engineFields.1 betaMDomain
  let betaMFields := squeezeN betaMStart (2 * shape.ellM)
  let betaM := pairFields betaMFields.2
  { alpha, betaA, betaR, gamma, betaM, state := betaMFields.1 }

/-- State-only decomposition of the first response batch, useful for mapping
the semantic phase to concrete Poseidon2 call families. -/
def afterEngine (initial : State) (shape : Shape) : State :=
  (squeezeN (appendRaw initial engineDomain)
    (2 * engineScalarCount shape)).1

@[simp] theorem run_state (initial : State) (shape : Shape) :
    (run initial shape).state =
      (squeezeN (appendRaw (afterEngine initial shape) betaMDomain)
        (2 * shape.ellM)).1 := by
  rfl

@[simp] theorem run_alpha (initial : State) (shape : Shape) :
    (run initial shape).alpha =
      (pairFields (squeezeN (appendRaw initial engineDomain)
        (2 * engineScalarCount shape)).2).take shape.ellD := by
  rfl

/-- `gamma` is the unique final extension element of the engine batch by
construction; it is not an independently supplied mixing scalar. -/
theorem run_gamma (initial : State) (shape : Shape) :
    (run initial shape).gamma =
      (pairFields (squeezeN (appendRaw initial engineDomain)
        (2 * engineScalarCount shape)).2).getD
          (2 * shape.ellD + shape.ellN) Extension.zero := by
  rfl

/-- The first engine slice has exactly the lane/Ajtai dimension. -/
@[simp] theorem run_alpha_length (initial : State) (shape : Shape) :
    (run initial shape).alpha.length = shape.ellD := by
  have engineLength :
      (pairFields
        (squeezeN (appendRaw initial engineDomain)
          (2 * engineScalarCount shape)).2).length =
        engineScalarCount shape :=
    pairFields_squeezeN_even_length _ _
  simp only [run, List.length_take]
  rw [engineLength]
  apply Nat.min_eq_left
  unfold engineScalarCount
  omega

/-- The second engine slice has exactly the lane/Ajtai dimension. -/
@[simp] theorem run_betaA_length (initial : State) (shape : Shape) :
    (run initial shape).betaA.length = shape.ellD := by
  have engineLength :
      (pairFields
        (squeezeN (appendRaw initial engineDomain)
          (2 * engineScalarCount shape)).2).length =
        engineScalarCount shape :=
    pairFields_squeezeN_even_length _ _
  simp only [run, List.length_take, List.length_drop]
  rw [engineLength]
  apply Nat.min_eq_left
  unfold engineScalarCount
  omega

/-- The third engine slice has exactly the FE row dimension. -/
@[simp] theorem run_betaR_length (initial : State) (shape : Shape) :
    (run initial shape).betaR.length = shape.ellN := by
  have engineLength :
      (pairFields
        (squeezeN (appendRaw initial engineDomain)
          (2 * engineScalarCount shape)).2).length =
        engineScalarCount shape :=
    pairFields_squeezeN_even_length _ _
  simp only [run, List.length_take, List.length_drop]
  rw [engineLength]
  apply Nat.min_eq_left
  unfold engineScalarCount
  omega

/-- The independently domain-separated NC response has exactly the flat
column dimension. -/
@[simp] theorem run_betaM_length (initial : State) (shape : Shape) :
    (run initial shape).betaM.length = shape.ellM := by
  simp only [run]
  exact pairFields_squeezeN_even_length _ _

end Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges
