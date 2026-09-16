import NightstreamFPrime.Export.Stage1.StoredPhysicalRowCheck
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-!
Transport final physical row checks from the runtime pilot header to the exact
selected source package. Only the metadata read by each checker is compared.
The source package occurs in proofs; no executable package builder is added.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredPhysicalRowContext

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package

local notation "sourcePackage" =>
  PerApplicationPackage.package Poseidon2HashChainV1Package.application

private theorem header_poseidon (templates : Array CompactRowTemplate) :
    ({ PilotData.circuitPackage () with compactRowTemplates := templates.toList } :
      CircuitPackage).poseidon = (sourcePackage).poseidon := by
  rfl

private theorem header_permutation (templates : Array CompactRowTemplate) :
    ({ PilotData.circuitPackage () with compactRowTemplates := templates.toList } :
      CircuitPackage).permutation = (sourcePackage).permutation := by
  rw [PerApplicationPackage.package_permutation]
  change (PilotData.circuitPackage ()).permutation =
    (Data.circuitPackage ()).permutation
  rw [Data.circuitPackage_permutation]
  exact PilotData.circuitPackageOf_permutation _ _ _

private theorem header_compactTemplates (templates : Array CompactRowTemplate)
    (canonical : templates.toList = Data.compactRowTemplates ()) :
    ({ PilotData.circuitPackage () with compactRowTemplates := templates.toList } :
      CircuitPackage).compactRowTemplates = (sourcePackage).compactRowTemplates := by
  rw [PerApplicationPackage.package_compactRowTemplates]
  change templates.toList = (Data.circuitPackage ()).compactRowTemplates
  rw [Data.circuitPackage_compactRowTemplates]
  exact canonical

private theorem hashInvocation_eq_of_metadata (left right : CircuitPackage)
    (poseidonEqual : left.poseidon = right.poseidon)
    (permutationEqual : left.permutation = right.permutation)
    (chain : HashChain) (ordinal : Nat) (env : Env) :
    StoredPhysicalRowCheck.hashInvocation left chain ordinal env =
      StoredPhysicalRowCheck.hashInvocation right chain ordinal env := by
  have columns : ∀ column,
      instantiateColumn left chain ordinal column =
        instantiateColumn right chain ordinal column := by
    intro column
    cases column <;>
      simp only [instantiateColumn, invocationInput, invocationLocalStart,
        poseidonEqual, permutationEqual]
  simp only [StoredPhysicalRowCheck.hashInvocation, permutationEqual, columns]

/-- All hash ordinals and all environments use the exact source-package rows. -/
theorem hashInvocation_iff (header : CircuitPackage)
    (templates : Array CompactRowTemplate)
    (headerEqual : header =
      { PilotData.circuitPackage () with compactRowTemplates := templates.toList })
    (chain : HashChain) (ordinal : Nat) (env : Env) :
    StoredPhysicalRowCheck.hashInvocation header chain ordinal env = true ↔
      TemplateInvocationHolds sourcePackage chain ordinal env := by
  rw [headerEqual, hashInvocation_eq_of_metadata _ sourcePackage
    (header_poseidon templates) (header_permutation templates)]
  exact StoredPhysicalRowCheck.hashInvocation_iff sourcePackage chain ordinal env

/-- Explicit permutation checks read only the unchanged permutation template. -/
theorem permutationInvocation_iff (header : CircuitPackage)
    (templates : Array CompactRowTemplate)
    (headerEqual : header =
      { PilotData.circuitPackage () with compactRowTemplates := templates.toList })
    (invocation : PermutationInvocation) (env : Env) :
    StoredPhysicalRowCheck.permutationInvocation header invocation env = true ↔
      PermutationInvocationHolds sourcePackage invocation env := by
  have equal : StoredPhysicalRowCheck.permutationInvocation header invocation env =
      StoredPhysicalRowCheck.permutationInvocation sourcePackage invocation env := by
    rw [headerEqual]
    unfold StoredPhysicalRowCheck.permutationInvocation
    rw [header_permutation templates]
  rw [equal]
  exact StoredPhysicalRowCheck.permutationInvocation_iff sourcePackage invocation env

/-- Canonical templates preserve both successful checks and missing-template rejection. -/
theorem compactInvocation_iff (header : CircuitPackage)
    (templates : Array CompactRowTemplate)
    (headerEqual : header =
      { PilotData.circuitPackage () with compactRowTemplates := templates.toList })
    (canonical : templates.toList = Data.compactRowTemplates ())
    (invocation : CompactRowInvocation) (env : Env) :
    StoredPhysicalRowCheck.compactInvocation header invocation env = true ↔
      CompactRowInvocationHolds sourcePackage invocation env := by
  have equal : StoredPhysicalRowCheck.compactInvocation header invocation env =
      StoredPhysicalRowCheck.compactInvocation sourcePackage invocation env := by
    rw [headerEqual]
    unfold StoredPhysicalRowCheck.compactInvocation
    rw [header_compactTemplates templates canonical]
  rw [equal]
  exact StoredPhysicalRowCheck.compactInvocation_iff sourcePackage invocation env

end NightstreamFPrime.Export.Stage1.StoredPhysicalRowContext
