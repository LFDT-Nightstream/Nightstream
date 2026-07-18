import Nightstream.Implementation.R1CS.Artifacts.Poseidon2
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.CarrierCodec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage

/-!
Raw-R1CS Poseidon2 cost leaf for the reduced accumulator carriers.

Assurance tier: artifact-checked primitive profile plus model-level sponge
composition. The isolated permutation census is checked against the generated
Rust artifact. Exact call-site/preimage conformance is still open.

Owns: the exact product/linear census of one generated width-eight Poseidon2
permutation; the rate-four sponge permutation formula; wrapper, permutation,
product, linear, total-row, and fresh-column formulas; and specialization to
the two injective public-statement carrier encodings.

Does not own: allocation of domain constants, reuse versus copying of carrier
wires, an emitted hash call, Poseidon2 native parity, collision resistance,
gadget-native lowering, encoded columns, or row removal. The canonical-parent
candidate still has no approved concrete domain message.

Emits constraints: no. This file accounts for a candidate invocation of the
existing field-valued R1CS hash gadget.

Authority boundary: the ordered-commitment input length comes from the exact
injective message codec. `spongeRows` counts only `enforce_poseidon2_hash`
after all input variables already exist. Therefore it must not be reported as
the complete accumulator-binding stage cost or as call-site conformance.

| Stage path | Cost owner | Exact raw-R1CS formula | Assurance |
|---|---|---|---|
| `fprime.accumulator.hash.preimage.domain` | exact ten-field ordered-commitment tag | outside this leaf | model-level |
| `fprime.accumulator.hash.preimage.carrier` | injective codec | zero rows when existing wires are reused | refinement open |
| `fprime.accumulator.hash.sponge.wrapper.zero` | zero-state definition | `1` linear row / fresh column | source-model |
| `fprime.accumulator.hash.sponge.wrapper.absorb` | absorb additions | `inputFields` linear rows / fresh columns | source-model |
| `fprime.accumulator.hash.sponge.wrapper.pad` | final `+1` | `1` linear row / fresh column | source-model |
| `fprime.accumulator.hash.sponge.permutation.sbox` | 86 four-row S-boxes | `344 * permutations` product rows | artifact-checked |
| `fprime.accumulator.hash.sponge.permutation.linear` | materialized linear layers/outputs | `256 * permutations` linear rows | artifact-checked |
| `fprime.accumulator.hash.sponge.total` | complete field-valued hash gadget | `inputFields + 2 + 600 * permutations` rows and fresh columns | composed model |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost

open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Permutation
open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec
open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage
open Nightstream.SuperNeo.Concrete.Phi81Relation

/-! ## Generated permutation census -/

def permutationRate : Nat := 4

def permutationRows : Nat :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.rowCount

def permutationFreshColumns : Nat :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.colCount -
    Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns.length

private def productDefinitionCount : List Definition -> Nat
  | [] => 0
  | definition :: definitions =>
      (match definition.rhs with
        | .linear _ => 0
        | .product _ _ => 1) + productDefinitionCount definitions

def permutationProductRows : Nat :=
  productDefinitionCount
    Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions

def permutationLinearRows : Nat :=
  permutationRows - permutationProductRows

theorem permutation_rate_eq : permutationRate = 4 := by
  rfl

theorem permutation_rows_eq : permutationRows = 600 := by
  rfl

theorem permutation_fresh_columns_eq : permutationFreshColumns = 600 := by
  decide

/-- Exact classification of all generated permutation definitions. This is
artifact census evidence and intentionally uses trusted native evaluation; it
is not a semantic security theorem. -/
theorem permutation_product_rows_eq : permutationProductRows = 344 := by
  native_decide

theorem permutation_linear_rows_eq : permutationLinearRows = 256 := by
  rw [permutationLinearRows, permutation_rows_eq,
    permutation_product_rows_eq]

/-! ## Sponge composition -/

/-- Number of rate-four input chunks. Empty input has no absorb chunk. -/
def absorbPermutationCount (inputFields : Nat) : Nat :=
  (inputFields + permutationRate - 1) / permutationRate

/-- One permutation per absorb chunk plus the mandatory padding permutation. -/
def permutationCount (inputFields : Nat) : Nat :=
  absorbPermutationCount inputFields + 1

/-- Zero-state row, one addition per absorbed field, and final padding row. -/
def wrapperLinearRows (inputFields : Nat) : Nat :=
  inputFields + 2

def spongeProductRows (inputFields : Nat) : Nat :=
  permutationCount inputFields * permutationProductRows

def spongePermutationLinearRows (inputFields : Nat) : Nat :=
  permutationCount inputFields * permutationLinearRows

def spongeLinearRows (inputFields : Nat) : Nat :=
  wrapperLinearRows inputFields + spongePermutationLinearRows inputFields

def spongeRows (inputFields : Nat) : Nat :=
  spongeProductRows inputFields + spongeLinearRows inputFields

/-- The current SSA builder allocates one fresh output variable per wrapper or
permutation row. Input variables themselves are pre-existing and excluded. -/
def spongeFreshColumns (inputFields : Nat) : Nat :=
  wrapperLinearRows inputFields +
    permutationCount inputFields * permutationFreshColumns

theorem sponge_rows_formula (inputFields : Nat) :
    spongeRows inputFields =
      inputFields + 2 + 600 * permutationCount inputFields := by
  rw [spongeRows, spongeProductRows, spongeLinearRows,
    spongePermutationLinearRows, wrapperLinearRows,
    permutation_product_rows_eq, permutation_linear_rows_eq]
  omega

theorem sponge_fresh_columns_eq_rows (inputFields : Nat) :
    spongeFreshColumns inputFields = spongeRows inputFields := by
  rw [spongeFreshColumns, sponge_rows_formula, wrapperLinearRows,
    permutation_fresh_columns_eq]
  omega

theorem sponge_gate_partition (inputFields : Nat) :
    spongeRows inputFields =
      spongeProductRows inputFields + spongeLinearRows inputFields := by
  rfl

/-! ## Reduced-carrier specializations -/

/-- Fixed-profile ordered-child carrier fields at one row-point dimension. -/
def commitmentFamilyCarrierFields (rowVariables : Nat) : Nat :=
  2 * rowVariables + 13608

/-- Fixed-profile canonical-parent carrier fields at one row-point dimension. -/
def canonicalParentCarrierFields (rowVariables : Nat) : Nat :=
  2 * rowVariables + 972

/-- Exact ordered-commitment Poseidon2 input length: ten domain fields followed
by the fixed-profile carrier. -/
def commitmentFamilyPreimageFields (rowVariables : Nat) : Nat :=
  fixedPreimageFieldCount rowVariables

/-- Hash rows as a direct function of row-point dimension. Domain length is
fixed by `OrderedCommitmentMessage`, not caller supplied. -/
def commitmentFamilyHashRowsFor (rowVariables : Nat) : Nat :=
  spongeRows (commitmentFamilyPreimageFields rowVariables)

/-- Hash rows as a direct function of row-point dimension and the separately
materialized domain-field count. -/
def canonicalParentHashRowsFor
    (rowVariables domainFields : Nat) : Nat :=
  spongeRows (domainFields + canonicalParentCarrierFields rowVariables)

/-- Hash-gadget rows after the caller has materialized the exact ten domain
lanes and the complete ordered-child carrier lanes. -/
def commitmentFamilyHashRows (shape : Shape) : Nat :=
  commitmentFamilyHashRowsFor shape.rowVariables

/-- Hash-gadget rows after a caller has materialized `domainFields` domain
lanes and the canonical-parent carrier lanes. -/
def canonicalParentHashRows
    (shape : Shape) (domainFields : Nat) : Nat :=
  canonicalParentHashRowsFor shape.rowVariables domainFields

theorem commitment_family_hash_rows_formula
    (shape : Shape) :
    commitmentFamilyHashRows shape =
      spongeRows (10 + (2 * shape.rowVariables + 13608)) := by
  rfl

theorem canonical_parent_hash_rows_formula
    (shape : Shape) (domainFields : Nat) :
    canonicalParentHashRows shape domainFields =
      spongeRows (domainFields + (2 * shape.rowVariables + 972)) := by
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost
