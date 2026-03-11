import SuperNeo.ProtocolTheorem

/-!
Contract interface for `SuperNeo.ProtocolTheorem`.

Spec: `./formal/superneo-lean/specs/ProtocolTheorem.spec.md`

Paper anchors (Source: `./formal/superneo-lean/SuperNeo.pdf.md`):
- Section 7.6 (implied final theorem): Composition of Π_CCS, Π_RLC, Π_DEC with knowledge-soundness
- Section 7, lines 447–596: Neo's folding scheme for CCS
- Appendix B/C/D: Assumption accounting, lattice security (MSIS, Ajtai binding)
-/

namespace SuperNeo

namespace ProtocolTheoremInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `schwartzZippelFailureEvent`. -/
abbrev schwartzZippelFailureEvent := SuperNeo.schwartzZippelFailureEvent

/-- [Role: Theorem-Target] Curated re-export of `SchwartzZippelAdvantage`. -/
abbrev SchwartzZippelAdvantage := SuperNeo.SchwartzZippelAdvantage

/-- [Role: Theorem-Target] Curated re-export of `SchwartzZippelAdvantageBound`. -/
abbrev SchwartzZippelAdvantageBound := SuperNeo.SchwartzZippelAdvantageBound

/-- [Role: Theorem-Target] Curated re-export of `LatticeParams`. -/
abbrev LatticeParams := SuperNeo.LatticeParams

/-- [Role: Theorem-Target] Curated re-export of `FinalTheoremShape`. -/
abbrev FinalTheoremShape := SuperNeo.FinalTheoremShape

/-- [Role: Theorem-Target] Canonical constructor for final error packages from component boundaries. -/
def finalErrorPackageOfComponentBoundaries :=
  @SuperNeo.FinalErrorPackage.ofComponentBoundaries

/-- [Role: Theorem-Target] Canonical constructor for aligned final error packages. -/
def finalErrorPackageOfAlignedComponents :=
  @SuperNeo.FinalErrorPackage.ofAlignedComponents

/-- [Role: Theorem-Target] Canonical constructor for aligned final error packages on the proved `paperCarrier` path, deriving Ajtai reduction data from the MSIS boundary. -/
def finalErrorPackageOfAlignedPaperCarrierFromThreeDLe :=
  @SuperNeo.FinalErrorPackage.ofAlignedPaperCarrierFromThreeDLe

/-- [Role: Theorem-Target] Canonical constructor for aligned final error packages on the Goldilocks Appendix B.2 paper-parameter family. -/
def finalErrorPackageOfGoldilocksPaperCarrier :=
  @SuperNeo.FinalErrorPackage.ofGoldilocksPaperCarrier

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions from boundary packages. -/
def finalTheoremAssumptionsOfBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofBoundaryPackages

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the proved `paperCarrier` path, deriving Ajtai reduction data from the MSIS boundary. -/
def finalTheoremAssumptionsOfAlignedPaperCarrierBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofAlignedPaperCarrierBoundaryPackages

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the paper-facing `paperCarrier`-difference path, deriving Ajtai reduction data from the MSIS boundary. -/
def finalTheoremAssumptionsOfAlignedPaperCarrierDiffBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofAlignedPaperCarrierDiffBoundaryPackages

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the proved `paperCarrier` path from a stronger strict low-norm invertibility theorem with threshold at least `5`. -/
def finalTheoremAssumptionsOfAlignedPaperCarrierLowNormBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofAlignedPaperCarrierLowNormBoundaryPackages

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family. -/
def finalTheoremAssumptionsOfGoldilocksPaperCarrierBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family, deriving the witness-level SumCheck and local Schwartz-Zippel boundaries directly from the carried transition witness and reduction arithmetic and reconstructing the internal MSIS boundary from the theorem-level hardness assumption. -/
noncomputable def finalTheoremAssumptionsOfGoldilocksPaperCarrierDerivedSumcheck :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierDerivedSumcheck

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family and active `paperCarrier`-difference path. -/
def finalTheoremAssumptionsOfGoldilocksPaperCarrierDiffBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierDiffBoundaryPackages

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family and active native-bar `paperCarrier`-difference path, discharging the generic Theorem-3 boundary from `thm3CoreAssumption_native`, deriving the witness-level SumCheck and local Schwartz-Zippel boundaries internally, and keeping only the theorem-level MSIS hardness assumption explicit. -/
noncomputable def finalTheoremAssumptionsOfGoldilocksNativePaperCarrierDiffBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family from a stronger strict low-norm invertibility theorem with threshold at least `5`. -/
def finalTheoremAssumptionsOfGoldilocksPaperCarrierLowNormBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierLowNormBoundaryPackages

/-- [Role: Theorem-Target] Canonical final theorem specialized to the proved `paperCarrier` path. -/
def finalTheoremShapeOfAlignedPaperCarrierBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_alignedPaperCarrierBoundaryPackages

/-- [Role: Theorem-Target] Canonical final theorem specialized to the paper-facing `paperCarrier`-difference path. -/
def finalTheoremShapeOfAlignedPaperCarrierDiffBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_alignedPaperCarrierDiffBoundaryPackages

/-- [Role: Theorem-Target] Canonical final theorem specialized to the proved `paperCarrier` path from a stronger strict low-norm invertibility theorem with threshold at least `5`. -/
def finalTheoremShapeOfAlignedPaperCarrierLowNormBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_alignedPaperCarrierLowNormBoundaryPackages

/-- [Role: Theorem-Target] Canonical final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family. -/
def finalTheoremShapeOfGoldilocksPaperCarrierBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_goldilocksPaperCarrierBoundaryPackages

/-- [Role: Theorem-Target] Canonical final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family and active `paperCarrier`-difference path. -/
def finalTheoremShapeOfGoldilocksPaperCarrierDiffBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_goldilocksPaperCarrierDiffBoundaryPackages

/-- [Role: Theorem-Target] Canonical final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family and active native-bar `paperCarrier`-difference path. -/
def finalTheoremShapeOfGoldilocksNativePaperCarrierDiffBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_goldilocksNativePaperCarrierDiffBoundaryPackages

/-- [Role: Theorem-Target] Canonical final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family from a stronger strict low-norm invertibility theorem with threshold at least `5`. -/
def finalTheoremShapeOfGoldilocksPaperCarrierLowNormBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_goldilocksPaperCarrierLowNormBoundaryPackages

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Context-local Schwartz-Zippel boundary surface. -/
abbrev SchwartzZippelBoundary := SuperNeo.SchwartzZippelBoundary

/-- [Role: Boundary] Theorem-level MSIS hardness assumption surface; this is the intended explicit security assumption on the active native Goldilocks final route. -/
abbrev msisHardnessAssumption := SuperNeo.msisHardnessAssumption

/-- [Role: Boundary] Boundary surface `ajtaiBindingAssumption` requiring closure. -/
abbrev ajtaiBindingAssumption := SuperNeo.ajtaiBindingAssumption

/-- [Role: Boundary] Boundary surface `ajtaiRelaxedBindingAssumption` requiring closure. -/
abbrev ajtaiRelaxedBindingAssumption := SuperNeo.ajtaiRelaxedBindingAssumption

/-- [Role: Boundary] Faithful prefix-dependent SumCheck Lund package for protocols. Retained as a local replay boundary, not as an active-route final-theorem requirement. -/
abbrev SumcheckPrefixLundBoundary := SuperNeo.SumcheckPrefixLundBoundary

/-- [Role: Boundary] Named Goldilocks/full-field Lund setup boundary. Retained for local replay surfaces, not required on the active native Goldilocks final-theorem route. -/
abbrev GoldilocksFullFieldLundBoundary :=
  SuperNeo.GoldilocksFullFieldLundBoundary

/-- [Role: Boundary] Witness-level SumCheck failure-advantage bound surface. -/
abbrev sumcheckFailureAdvantageBound :=
  SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantageBound

/-- [Role: Boundary] Canonical final error package surface. -/
abbrev FinalErrorPackage := SuperNeo.FinalErrorPackage

/-- [Role: Boundary] Boundary surface `FinalTheoremAssumptions` requiring closure. -/
abbrev FinalTheoremAssumptions := SuperNeo.FinalTheoremAssumptions

end ProtocolTheoremInterface

end SuperNeo
