import NightstreamFPrime.Export.RowSemantics
import NightstreamFPrime.Export.Stage1.PackagePlan
import NightstreamFPrime.Export.Stage1.PerApplicationCachedShift
import NightstreamFPrime.Export.Stage1.PermutationPlan
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-!
Compose final-state checks of the selected canonical component sources into
physical package row satisfaction. This proof does not construct event arrays,
justify IO origins, or make a claim about witness execution order.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PreparedPhysicalPackageRows

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package

local notation "selectedApplication" => Poseidon2HashChainV1Package.application
local notation "selectedShift" =>
  PerApplicationCachedShift.Context.ofProgram Poseidon2HashChainV1Package.application
local notation "sourcePackage" =>
  PerApplicationPackage.package Poseidon2HashChainV1Package.application

/-- Every premise concerns the same completed environment and an existing
canonical source. No schedule-preservation or IO provenance premise is used. -/
theorem rowsHold (env : Env)
    (hashes : ∀ chain ∈ [Data.priorChain, Data.outputChain],
      ∀ ordinal,
        ordinal ≤ (PerApplicationCachedShift.shiftHashChain selectedShift chain).absorbCount →
        TemplateInvocationHolds sourcePackage
          (PerApplicationCachedShift.shiftHashChain selectedShift chain) ordinal env)
    (permutations : ∀ block ∈ PermutationPlan.canonicalBlocks (),
      ∀ invocation ∈ block.expand,
        PermutationInvocationHolds sourcePackage
          (PerApplicationCachedShift.shiftPermutationInvocation selectedShift invocation) env)
    (compacts : ∀ block ∈ PackagePlan.canonicalCompactBlocks,
      ∀ invocation ∈ block.expand,
        CompactRowInvocationHolds sourcePackage
          (PerApplicationCachedShift.shiftCompactRowInvocation selectedShift invocation) env)
    (pilotRows :
      (∀ instruction ∈ Data.liftPilotInstructions (PilotData.witnessInstructions ()),
        (PerApplicationCachedShift.shiftWitnessInstruction selectedShift instruction).Holds env) ∧
      (∀ row ∈ Data.liftPilotRows (PilotData.assertionRows ()),
        (PerApplicationCachedShift.shiftSparseRow selectedShift row).Holds env))
    (arithmeticRows :
      (∀ instruction ∈ Rows.witnessInstructionsTR (Data.arithmeticRows ()),
        (PerApplicationCachedShift.shiftWitnessInstruction selectedShift instruction).Holds env) ∧
      (∀ row ∈ Rows.assertionRowsTR (Data.arithmeticRows ()),
        (PerApplicationCachedShift.shiftSparseRow selectedShift row).Holds env))
    (applicationRows :
      (∀ instruction ∈
        (PerApplicationPackage.directApplicationPlan selectedApplication).witnessInstructions,
        instruction.Holds env) ∧
      (∀ row ∈
        (PerApplicationPackage.directApplicationPlan selectedApplication).assertionRows,
        row.Holds env))
    (nextPreimageRows : ∀ row ∈ NextPreimagePackage.assertionRows
      (PerApplicationPackage.nextPreimageRowStart selectedApplication), row.Holds env) :
    (Poseidon2HashChainV1Package.package ()).RowsHold env := by
  apply (PerApplicationCanonicalPackage.rowsHold_iff_sourcePackage
    selectedApplication Poseidon2HashChainV1Package.fits env).mpr
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro chain member
    rw [PerApplicationPackage.package_hashChains] at member
    change chain ∈ (Data.circuitPackage ()).hashChains.map
      (PerApplicationPackage.shiftHashChain selectedApplication) at member
    rw [Data.circuitPackage_hashChains] at member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    simpa only [HashChainHolds, PerApplicationCachedShift.shiftHashChain_eq] using
      hashes source sourceMember
  · intro invocation member
    rw [PerApplicationPackage.package_permutationInvocations] at member
    change invocation ∈ (Data.circuitPackage ()).permutationInvocations.map
      (PerApplicationPackage.shiftPermutationInvocation selectedApplication) at member
    rw [Data.circuitPackage_permutationInvocations,
      Data.components_permutationInvocations,
      ← PermutationPlan.canonicalBlocks_expand] at member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    rcases List.mem_flatMap.mp sourceMember with ⟨block, blockMember, invocationMember⟩
    simpa only [PerApplicationCachedShift.shiftPermutationInvocation_eq] using
      permutations block blockMember source invocationMember
  · intro invocation member
    rw [PerApplicationPackage.package_compactRowInvocations] at member
    change invocation ∈ (Data.circuitPackage ()).compactRowInvocations.map
      (PerApplicationPackage.shiftCompactRowInvocation selectedApplication) at member
    rw [Data.circuitPackage_compactRowInvocations,
      ← PackagePlan.canonicalCompactBlocks_expand] at member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    rcases List.mem_flatMap.mp sourceMember with ⟨block, blockMember, invocationMember⟩
    simpa only [PerApplicationCachedShift.shiftCompactRowInvocation_eq] using
      compacts block blockMember source invocationMember
  · intro instruction member
    rw [PerApplicationPackage.package_witnessInstructions] at member
    rcases List.mem_append.mp member with baseMember | applicationMember
    · rcases List.mem_map.mp baseMember with ⟨source, sourceMember, rfl⟩
      change source ∈ (Data.circuitPackage ()).witnessInstructions at sourceMember
      rw [Data.circuitPackage_witnessInstructions] at sourceMember
      change source ∈ Data.liftPilotInstructions (PilotData.witnessInstructions ()) ++
        Rows.witnessInstructionsTR (Data.components ()).arithmeticRows at sourceMember
      rw [Data.components_arithmeticRows] at sourceMember
      rcases List.mem_append.mp sourceMember with pilotMember | arithmeticMember
      · simpa only [PerApplicationCachedShift.shiftWitnessInstruction_eq] using
          pilotRows.1 source pilotMember
      · simpa only [PerApplicationCachedShift.shiftWitnessInstruction_eq] using
          arithmeticRows.1 source arithmeticMember
    · rw [← PerApplicationPackage.directApplicationPlan_eq_applicationPlan
        selectedApplication] at applicationMember
      exact applicationRows.1 instruction applicationMember
  · intro row member
    rw [PerApplicationPackage.package_assertionRows] at member
    rcases List.mem_append.mp member with prefixMember | nextMember
    · rcases List.mem_append.mp prefixMember with baseMember | applicationMember
      · rcases List.mem_map.mp baseMember with ⟨source, sourceMember, rfl⟩
        change source ∈ (Data.circuitPackage ()).assertionRows at sourceMember
        rw [Data.circuitPackage_assertionRows] at sourceMember
        change source ∈ Data.liftPilotRows (PilotData.assertionRows ()) ++
          Rows.assertionRowsTR (Data.components ()).arithmeticRows at sourceMember
        rw [Data.components_arithmeticRows] at sourceMember
        rcases List.mem_append.mp sourceMember with pilotMember | arithmeticMember
        · simpa only [PerApplicationCachedShift.shiftSparseRow_eq] using
            pilotRows.2 source pilotMember
        · simpa only [PerApplicationCachedShift.shiftSparseRow_eq] using
            arithmeticRows.2 source arithmeticMember
      · rw [← PerApplicationPackage.directApplicationPlan_eq_applicationPlan
          selectedApplication] at applicationMember
        exact applicationRows.2 row applicationMember
    · exact nextPreimageRows row nextMember

end NightstreamFPrime.Export.Stage1.PreparedPhysicalPackageRows
