import SuperNeo.FoldingProtocol.ProtocolTheorem

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
def finalErrorPackageOfComponentBoundaries
  {ctx : ProtocolTargetContext}
  {params : SuperNeo.LatticeParams}
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SuperNeo.SchwartzZippelBoundary ctx)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (msisToAjtai : SuperNeo.ProofSystem.MSISToAjtaiReductions params) :
  SuperNeo.FinalErrorPackage ctx params :=
  SuperNeo.FinalErrorPackage.ofComponentBoundaries
    sumcheckError schwartzZippelBoundary msisBoundary msisToAjtai

/-- [Role: Theorem-Target] Canonical constructor for final error packages on the Goldilocks Appendix B.2 paper-parameter family. -/
def finalErrorPackageOfGoldilocksPaperCarrier
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SuperNeo.SchwartzZippelBoundary ctx)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  SuperNeo.FinalErrorPackage ctx
    (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength) :=
  SuperNeo.FinalErrorPackage.ofGoldilocksPaperCarrier
    messageLength sumcheckError schwartzZippelBoundary msisBoundary

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions from boundary packages. -/
def finalTheoremAssumptionsOfBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : SuperNeo.LatticeParams}
  (reduction : SuperNeo.InteractiveReductionAssumptions ctx)
  (errorPackage : SuperNeo.FinalErrorPackage ctx params) :
  SuperNeo.FinalTheoremAssumptions ctx :=
  SuperNeo.FinalTheoremAssumptions.ofBoundaryPackages reduction errorPackage

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family. -/
def finalTheoremAssumptionsOfGoldilocksPaperCarrierBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (reduction : SuperNeo.InteractiveReductionAssumptions ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SuperNeo.SchwartzZippelBoundary ctx)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  SuperNeo.FinalTheoremAssumptions ctx :=
  SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages
    messageLength reduction sumcheckError schwartzZippelBoundary msisBoundary

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family, deriving the witness-level SumCheck and local Schwartz-Zippel boundaries directly from the carried transition witness and reduction arithmetic and reconstructing the internal MSIS boundary from the theorem-level hardness assumption. -/
noncomputable def finalTheoremAssumptionsOfGoldilocksPaperCarrierDerivedSumcheck
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (reduction : SuperNeo.InteractiveReductionAssumptions ctx)
  (hMsis :
    SuperNeo.msisHardnessAssumption
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  SuperNeo.FinalTheoremAssumptions ctx :=
  SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierDerivedSumcheck
    messageLength reduction hMsis

/-- [Role: Theorem-Target] Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family and active native-bar `paperCarrier`-difference path, discharging the generic Theorem-3 boundary from `thm3CoreAssumption_native`, deriving the witness-level SumCheck and local Schwartz-Zippel boundaries internally, and keeping only the theorem-level MSIS hardness assumption explicit. -/
noncomputable def finalTheoremAssumptionsOfGoldilocksNativePaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hBarNative : ctx.bar = nativeBarMatrix)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx)
  (hMsis :
    SuperNeo.msisHardnessAssumption
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  SuperNeo.FinalTheoremAssumptions ctx :=
  SuperNeo.FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages
    messageLength hBarNative hArithmetic hDiff hNe hWitness hMsis

/-- [Role: Theorem-Target] Canonical final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family and active native-bar `paperCarrier`-difference path. -/
theorem finalTheoremShapeOfGoldilocksNativePaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hBarNative : ctx.bar = nativeBarMatrix)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx)
  (hMsis :
    SuperNeo.msisHardnessAssumption
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  SuperNeo.FinalTheoremShape ctx
    (SuperNeo.FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages
      messageLength hBarNative hArithmetic hDiff hNe hWitness hMsis) :=
  SuperNeo.finalTheoremShape_of_goldilocksNativePaperCarrierDiffBoundaryPackages
    messageLength hBarNative hArithmetic hDiff hNe hWitness hMsis

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
