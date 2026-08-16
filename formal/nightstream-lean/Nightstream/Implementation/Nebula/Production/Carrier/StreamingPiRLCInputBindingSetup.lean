import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputResidual
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingSetup

/-!
Contract: concrete Phi81 representation for the complete production PiRLC
input residual.

Assurance tier: implementation-to-security-reduction bridge.

Owns specialization of the 89,100-field binding to one explicit seeded Ajtai
setup, canonical flattening of its two degree-54 outputs to 108 Goldilocks
fields, the inverse representation, local family binding fields, and recovery
of equal PiRLC inputs or one named Module-SIS failure.

Does not own the fixed production seed, Rust sampler conformance, generated
rows, recursive state placement, local-transition telescoping, or Module-SIS
hardness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 262144

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.AjtaiBinding

abbrev RingF := Nightstream.SuperNeo.Concrete.RingF

abbrev Phi81Ring :=
  ProductionStreamingPiCcsCoordinateBindingSetup.ExecutablePhi81.Ring

/-- The coefficient embedding is definitionally the same degree-54 map used
by the existing PiCCS coordinate binding. -/
abbrev coefficientMap : CoefficientVector shape →+ Phi81Ring :=
  ProductionStreamingPiCcsCoordinateBindingSetup.coefficientMap

/-- Exact rank-two matrix selected by an explicit verifier-owned setup. -/
def seededMatrix
    (setup : SeededAjtai.Setup verifierRows messageColumnCount) :
    Matrix Phi81Ring shape :=
  fun row column => ⟨setup.verifierKey row column⟩

@[simp] theorem seededMatrix_coefficients
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (row : Fin shape.rows) (column : Fin shape.columns) :
    (seededMatrix setup row column).coefficients =
      setup.verifierKey row column := rfl

/-! ## Exact 108-field carried representation -/

abbrev ResidualFields :=
  Fin (shape.rows * shape.degree) → Nightstream.SuperNeo.Concrete.F

def outputPair
    (output : Fin (shape.rows * shape.degree)) :
    Fin shape.rows × Fin shape.degree :=
  (finProdFinEquiv (m := shape.rows) (n := shape.degree)).symm output

def outputIndex
    (row : Fin shape.rows) (lane : Fin shape.degree) :
    Fin (shape.rows * shape.degree) :=
  finProdFinEquiv (row, lane)

@[simp] theorem outputPair_outputIndex
    (row : Fin shape.rows) (lane : Fin shape.degree) :
    outputPair (outputIndex row lane) = (row, lane) := by
  exact Equiv.symm_apply_apply _ _

@[simp] theorem outputIndex_outputPair
    (output : Fin (shape.rows * shape.degree)) :
    outputIndex (outputPair output).1 (outputPair output).2 = output := by
  exact Equiv.apply_symm_apply _ output

theorem outputIndex_val
    (row : Fin shape.rows) (lane : Fin shape.degree) :
    (outputIndex row lane).val = row.val * shape.degree + lane.val := by
  unfold outputIndex
  change lane.val + shape.degree * row.val =
    row.val * shape.degree + lane.val
  ac_rfl

/-- Canonical row-major field view of the two Phi81 outputs. -/
def flattenCommitment
    (commitment : Commitment Phi81Ring shape) : ResidualFields :=
  fun output =>
    let pair := outputPair output
    (commitment pair.1).coefficients pair.2

/-- Exact inverse of the 108-field carried representation. -/
def unflattenCommitment
    (fields : ResidualFields) : Commitment Phi81Ring shape :=
  fun row => ⟨fun lane => fields (outputIndex row lane)⟩

@[simp] theorem flatten_unflatten (fields : ResidualFields) :
    flattenCommitment (unflattenCommitment fields) = fields := by
  funext output
  simp [flattenCommitment, unflattenCommitment]

@[simp] theorem unflatten_flatten
    (commitment : Commitment Phi81Ring shape) :
    unflattenCommitment (flattenCommitment commitment) = commitment := by
  funext row
  apply ProductionStreamingPiCcsCoordinateBindingSetup.ExecutablePhi81.Ring.ext
  funext lane
  simp [unflattenCommitment, flattenCommitment]

theorem flattenCommitment_injective :
    Function.Injective flattenCommitment := by
  intro left right equal
  calc
    left = unflattenCommitment (flattenCommitment left) :=
      (unflatten_flatten left).symm
    _ = unflattenCommitment (flattenCommitment right) :=
      congrArg unflattenCommitment equal
    _ = right := unflatten_flatten right

theorem exact_output_width : shape.rows * shape.degree = 108 := by
  decide

/-! ## Concrete full and local bindings -/

def concreteBinding
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (inputs : InputRings) : ResidualFields :=
  flattenCommitment
    (inputBindingMap (seededMatrix setup) coefficientMap inputs)

def concretePhaseBinding
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (family : Family) (inputs : Source → RingF) : ResidualFields :=
  flattenCommitment
    (ProductionStreamingPiRlcInputResidual.phaseBinding
      (seededMatrix setup) coefficientMap family inputs)

def ConcreteBindingFailure
    (setup : SeededAjtai.Setup verifierRows messageColumnCount) : Prop :=
  BindingFailure (seededMatrix setup) coefficientMap

theorem equal_concrete_binding_recovers_inputs_or_failure
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (left right : InputRings)
    (equal : concreteBinding setup left = concreteBinding setup right) :
    Or (left = right) (ConcreteBindingFailure setup) := by
  apply equal_input_binding_recovers_inputs_or_failure
    (seededMatrix setup) coefficientMap left right
  exact flattenCommitment_injective equal

def addResidualFields
    (left right : ResidualFields) : ResidualFields :=
  fun output => left output + right output

def zeroResidualFields : ResidualFields := fun _ => 0

/-- One output coefficient as an additive map. This keeps finite-sum proofs
symbolic and prevents expansion of the 110-family index. -/
def coefficientAt (lane : Fin shape.degree) :
    Phi81Ring →+ Nightstream.SuperNeo.Concrete.F where
  toFun value := value.coefficients lane
  map_zero' := rfl
  map_add' := by
    intro left right
    rfl

theorem flatten_addResidual
    (left right : Commitment Phi81Ring shape) :
    flattenCommitment
        (ProductionStreamingPiRlcInputResidual.addResidual left right) =
      addResidualFields (flattenCommitment left) (flattenCommitment right) := by
  funext output
  simp [flattenCommitment,
    ProductionStreamingPiRlcInputResidual.addResidual, addResidualFields,
    ProductionStreamingPiCcsCoordinateBindingSetup.ExecutablePhi81.add_coefficients,
    Nightstream.SuperNeo.Concrete.ringFAdd]

theorem flatten_zeroResidual :
    flattenCommitment
        (ProductionStreamingPiRlcInputResidual.zeroResidual :
          Commitment Phi81Ring shape) =
      zeroResidualFields := by
  funext output
  simp [flattenCommitment,
    ProductionStreamingPiRlcInputResidual.zeroResidual, zeroResidualFields,
    Nightstream.SuperNeo.Concrete.ringFZero]

/-- Exact field-level local transition used by recursive state glue. -/
def ConcreteResidualTransition
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (before after : ResidualFields)
    (family : Family) (inputs : Source → RingF) : Prop :=
  before = addResidualFields
    (concretePhaseBinding setup family inputs) after

/-! ## Complete concrete run -/

/-- The 110 local field commitments sum to the one full input commitment. -/
theorem concretePhaseBindings_sum
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (inputs : InputRings) :
    (fun output =>
      ∑ ordinal : Fin familyCount,
        concretePhaseBinding setup (familyAtOrdinal ordinal)
          (fun source => inputs source (familyAtOrdinal ordinal)) output) =
      concreteBinding setup inputs := by
  funext output
  have atRow := congrFun
    (ProductionStreamingPiRlcInputResidual.phaseBindings_sum
      (seededMatrix setup) coefficientMap inputs) (outputPair output).1
  have atCoefficient := congrArg
    (coefficientAt (outputPair output).2) atRow
  rw [map_sum] at atCoefficient
  change
    (∑ ordinal : Fin familyCount,
      ((ProductionStreamingPiRlcInputResidual.phaseBinding
        (seededMatrix setup) coefficientMap (familyAtOrdinal ordinal)
        (fun source => inputs source (familyAtOrdinal ordinal)))
          (outputPair output).1).coefficients (outputPair output).2) =
      ((inputBindingMap (seededMatrix setup) coefficientMap inputs)
        (outputPair output).1).coefficients (outputPair output).2
  exact atCoefficient

/-- Aggregate field equation that a complete sequence of local transitions
must derive by telescoping. -/
def ConcreteCompleteResidualRun
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (start finish : ResidualFields) (inputs : InputRings) : Prop :=
  start = addResidualFields
    (fun output =>
      ∑ ordinal : Fin familyCount,
        concretePhaseBinding setup (familyAtOrdinal ordinal)
          (fun source => inputs source (familyAtOrdinal ordinal)) output)
    finish

theorem honest_concreteCompleteResidualRun
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (inputs : InputRings) :
    ConcreteCompleteResidualRun setup
      (concreteBinding setup inputs) zeroResidualFields inputs := by
  unfold ConcreteCompleteResidualRun
  rw [concretePhaseBindings_sum]
  funext output
  simp [addResidualFields, zeroResidualFields]

/-- A complete field-level run that starts at the authoritative commitment
and finishes at zero recovers all supplied inputs, or exposes the named
Module-SIS failure. -/
theorem concrete_complete_zero_recovers_inputs_or_failure
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (authoritative supplied : InputRings)
    (start : ResidualFields)
    (startAuthoritative : start = concreteBinding setup authoritative)
    (run : ConcreteCompleteResidualRun setup start zeroResidualFields
      supplied) :
    Or (supplied = authoritative) (ConcreteBindingFailure setup) := by
  apply equal_concrete_binding_recovers_inputs_or_failure setup
    supplied authoritative
  calc
    concreteBinding setup supplied =
        (fun output =>
          ∑ ordinal : Fin familyCount,
            concretePhaseBinding setup (familyAtOrdinal ordinal)
              (fun source => supplied source (familyAtOrdinal ordinal))
              output) :=
      (concretePhaseBindings_sum setup supplied).symm
    _ = start := by
      have exactRun := run
      unfold ConcreteCompleteResidualRun at exactRun
      funext output
      have atOutput := congrFun exactRun output
      simpa [addResidualFields, zeroResidualFields] using atOutput.symm
    _ = concreteBinding setup authoritative := startAuthoritative

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup
