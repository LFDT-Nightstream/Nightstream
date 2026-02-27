import SuperNeo.ProtocolReduction
import SuperNeo.Sumcheck

/-! Pi_CCS reduction statement and bridge lemmas. -/


namespace SuperNeo

open F

/--
Pi_CCS soundness-side binding from accepted SumCheck transcripts to arithmetic validity.
This is where paper-specific encoding details are threaded into the reduction.
-/
def PiCCSArithmeticLinkAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (inst : SumcheckInstance) (tr : SumcheckTranscript),
    SumcheckAcceptedProp inst tr →
    SumcheckClaimTrue inst →
    ClaimArithmeticValid ctx claim

/--
Stronger Pi_CCS soundness-side binding that consumes the stronger accepted
transcript predicate (including round-consistency constraints).
-/
def PiCCSArithmeticStrongLinkAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (inst : SumcheckInstance) (tr : SumcheckTranscript),
    SumcheckAcceptedStrongProp inst tr →
    SumcheckClaimTrue inst →
    ClaimArithmeticValid ctx claim

/--
Concrete theorem-native link surface:
an accepted SumCheck transcript is enough to produce the full `P20` arithmetic bundle.
-/
def PiCCSP20LinkAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (inst : SumcheckInstance) (tr : SumcheckTranscript),
    SumcheckAcceptedProp inst tr →
    p20ForClaim ctx claim

/--
Strong variant of the concrete link surface using round-consistent accepted transcripts.
-/
def PiCCSP20StrongLinkAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (inst : SumcheckInstance) (tr : SumcheckTranscript),
    SumcheckAcceptedStrongProp inst tr →
    p20ForClaim ctx claim

/--
Concrete check-level encoding boundary for Pi_CCS:
accepted transcripts produce the exact protocol check obligations used by
`protocol_checks_imply_props`.
-/
def PiCCSCheckLinkAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (inst : SumcheckInstance) (tr : SumcheckTranscript),
    SumcheckAcceptedProp inst tr →
    p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    p20InvertibilityWindowProp claim.invDelta ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true

/--
Strong check-level encoding boundary (round-consistent accepted transcripts).
-/
def PiCCSStrongCheckLinkAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (inst : SumcheckInstance) (tr : SumcheckTranscript),
    SumcheckAcceptedStrongProp inst tr →
    p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    p20InvertibilityWindowProp claim.invDelta ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true

/--
Named bundle for the concrete protocol-check obligations consumed by
`protocol_checks_imply_props`.
-/
structure PiCCSProtocolChecks (ctx : ProtocolCtx) (claim : CEClaim) : Prop where
  hP10 : p10CoreCheck ctx.bar claim.a claim.b = true
  hP6 : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true
  hP12 : matrixTransformIdentity ctx.bar claim.m claim.z = true
  hP14 : evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true
  hVecAdd : preservesAddVec ctx.hVec claim.z1 claim.z2 = true
  hVecScale : preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true
  hScalAdd : preservesAddScalar ctx.hScal claim.z1 claim.z2 = true
  hScalScale : preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true
  hP16Win : p20InvertibilityWindowProp claim.invDelta
  hP17 : samplingSetBoundCheck claim.cset claim.samples = true
  hP18Eq : eqLiftAllBoolean claim.qVals ctx.ell = true
  hP18SZ : schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true
  hP19 : interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true

def PiCCSCheckBundleLinkAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (inst : SumcheckInstance) (tr : SumcheckTranscript),
    SumcheckAcceptedProp inst tr →
    PiCCSProtocolChecks ctx claim

def PiCCSStrongCheckBundleLinkAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (inst : SumcheckInstance) (tr : SumcheckTranscript),
    SumcheckAcceptedStrongProp inst tr →
    PiCCSProtocolChecks ctx claim

theorem piCCSProtocolChecks_of_checkConj
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hChecks :
    p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    p20InvertibilityWindowProp claim.invDelta ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true) :
  PiCCSProtocolChecks ctx claim := by
  rcases hChecks with ⟨hP10, hP6, hP12, hP14, hVecAdd, hVecScale, hScalAdd, hScalScale, hP16Win, hP17, hP18Eq, hP18SZ, hP19⟩
  exact ⟨hP10, hP6, hP12, hP14, hVecAdd, hVecScale, hScalAdd, hScalScale, hP16Win, hP17, hP18Eq, hP18SZ, hP19⟩

theorem piCCSCheckConj_of_protocolChecks
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hChecks : PiCCSProtocolChecks ctx claim) :
    p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    p20InvertibilityWindowProp claim.invDelta ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
  exact ⟨
    hChecks.hP10, hChecks.hP6, hChecks.hP12, hChecks.hP14,
    hChecks.hVecAdd, hChecks.hVecScale, hChecks.hScalAdd, hChecks.hScalScale,
    hChecks.hP16Win, hChecks.hP17, hChecks.hP18Eq, hChecks.hP18SZ, hChecks.hP19
  ⟩

theorem p20ForClaim_of_piCCSProtocolChecks
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hChecks : PiCCSProtocolChecks ctx claim) :
  p20ForClaim ctx claim := by
  exact (protocol_checks_imply_props
    (ctx := ctx) (claim := claim)
    hChecks.hP10 hChecks.hP6 hChecks.hP12 hChecks.hP14
    hChecks.hVecAdd hChecks.hVecScale hChecks.hScalAdd hChecks.hScalScale
    hChecks.hP16Win hChecks.hP17 hChecks.hP18Eq hChecks.hP18SZ hChecks.hP19).2

theorem piCCSCheckBundleLinkAssumption_of_checkLinkAssumption
  (hCheckLink : PiCCSCheckLinkAssumption) :
  PiCCSCheckBundleLinkAssumption := by
  intro ctx claim inst tr hAccepted
  exact piCCSProtocolChecks_of_checkConj (hCheckLink ctx claim inst tr hAccepted)

theorem piCCSCheckLinkAssumption_of_checkBundleLinkAssumption
  (hCheckLink : PiCCSCheckBundleLinkAssumption) :
  PiCCSCheckLinkAssumption := by
  intro ctx claim inst tr hAccepted
  exact piCCSCheckConj_of_protocolChecks (hCheckLink ctx claim inst tr hAccepted)

theorem piCCSStrongCheckBundleLinkAssumption_of_strongCheckLinkAssumption
  (hCheckLink : PiCCSStrongCheckLinkAssumption) :
  PiCCSStrongCheckBundleLinkAssumption := by
  intro ctx claim inst tr hAccepted
  exact piCCSProtocolChecks_of_checkConj (hCheckLink ctx claim inst tr hAccepted)

theorem piCCSStrongCheckLinkAssumption_of_strongCheckBundleLinkAssumption
  (hCheckLink : PiCCSStrongCheckBundleLinkAssumption) :
  PiCCSStrongCheckLinkAssumption := by
  intro ctx claim inst tr hAccepted
  exact piCCSCheckConj_of_protocolChecks (hCheckLink ctx claim inst tr hAccepted)

/--
Pi_CCS completeness-side encoding: arithmetic-valid claims induce a true SumCheck root claim.
-/
def PiCCSSumcheckEncodingAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (inst : SumcheckInstance),
    ClaimArithmeticValid ctx claim →
    SumcheckClaimTrue inst

/--
First strong-IR assumption bundle for Pi_CCS, parameterized over the SumCheck layer.
-/
def PiCCSProtocolAssumption : Prop :=
  SumcheckProtocolAssumption ∧
  PiCCSArithmeticLinkAssumption ∧
  PiCCSSumcheckEncodingAssumption

/--
Stronger protocol bundle variant that tracks strong SumCheck soundness and
round-consistent accepted transcripts.
-/
def PiCCSStrongProtocolAssumption : Prop :=
  SumcheckStrongSoundnessAssumption ∧
  SumcheckCompletenessAssumption ∧
  PiCCSArithmeticStrongLinkAssumption ∧
  PiCCSSumcheckEncodingAssumption

/--
Protocol bundle variant where the Pi_CCS link is supplied through the named
protocol-check bundle surface.
-/
def PiCCSCheckBundleProtocolAssumption : Prop :=
  SumcheckProtocolAssumption ∧
  PiCCSCheckBundleLinkAssumption ∧
  PiCCSSumcheckEncodingAssumption

/--
Strong protocol bundle variant where the Pi_CCS link is supplied through the
strong named protocol-check bundle surface.
-/
def PiCCSStrongCheckBundleProtocolAssumption : Prop :=
  SumcheckStrongSoundnessAssumption ∧
  SumcheckCompletenessAssumption ∧
  PiCCSStrongCheckBundleLinkAssumption ∧
  PiCCSSumcheckEncodingAssumption

theorem claimArithmeticValid_of_p20ForClaim
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP20 : p20ForClaim ctx claim) :
  ClaimArithmeticValid ctx claim := by
  rcases hP20 with
    ⟨hP6, hRows, hMat, hEval, _hVec, _hScal, hInvPre, hInvWin, hSampling, hPoly, hInterp⟩
  exact ⟨hP6, hRows, hMat, hEval, hInvPre, hInvWin, hSampling, hPoly, hInterp⟩

theorem piCCSArithmeticLink_of_p20LinkAssumption
  (hP20Link : PiCCSP20LinkAssumption) :
  PiCCSArithmeticLinkAssumption := by
  intro ctx claim inst tr hAccepted _hClaim
  exact claimArithmeticValid_of_p20ForClaim (hP20Link ctx claim inst tr hAccepted)

theorem piCCSArithmeticLink_of_checkBundleLinkAssumption
  (hCheckLink : PiCCSCheckBundleLinkAssumption) :
  PiCCSArithmeticLinkAssumption := by
  exact piCCSArithmeticLink_of_p20LinkAssumption (fun ctx claim inst tr hAccepted =>
    p20ForClaim_of_piCCSProtocolChecks (hCheckLink ctx claim inst tr hAccepted))

theorem piCCSArithmeticStrongLink_of_p20StrongLinkAssumption
  (hP20Link : PiCCSP20StrongLinkAssumption) :
  PiCCSArithmeticStrongLinkAssumption := by
  intro ctx claim inst tr hAccepted _hClaim
  exact claimArithmeticValid_of_p20ForClaim (hP20Link ctx claim inst tr hAccepted)

theorem piCCSArithmeticStrongLink_of_strongCheckBundleLinkAssumption
  (hCheckLink : PiCCSStrongCheckBundleLinkAssumption) :
  PiCCSArithmeticStrongLinkAssumption := by
  exact piCCSArithmeticStrongLink_of_p20StrongLinkAssumption (fun ctx claim inst tr hAccepted =>
    p20ForClaim_of_piCCSProtocolChecks (hCheckLink ctx claim inst tr hAccepted))

theorem piCCSArithmeticStrongLinkAssumption_of_linkAssumption
  (hLink : PiCCSArithmeticLinkAssumption) :
  PiCCSArithmeticStrongLinkAssumption := by
  intro ctx claim inst tr hAcceptedStrong hClaim
  exact hLink ctx claim inst tr
    (sumcheckAcceptedStrong_implies_accepted hAcceptedStrong)
    hClaim

theorem piCCSP20Link_of_checkLinkAssumption
  (hCheckLink : PiCCSCheckLinkAssumption) :
  PiCCSP20LinkAssumption := by
  intro ctx claim inst tr hAccepted
  exact p20ForClaim_of_piCCSProtocolChecks
    (piCCSProtocolChecks_of_checkConj (hCheckLink ctx claim inst tr hAccepted))

theorem piCCSP20Link_of_checkBundleLinkAssumption
  (hCheckLink : PiCCSCheckBundleLinkAssumption) :
  PiCCSP20LinkAssumption := by
  intro ctx claim inst tr hAccepted
  exact p20ForClaim_of_piCCSProtocolChecks (hCheckLink ctx claim inst tr hAccepted)

theorem piCCSP20StrongLink_of_strongCheckLinkAssumption
  (hCheckLink : PiCCSStrongCheckLinkAssumption) :
  PiCCSP20StrongLinkAssumption := by
  intro ctx claim inst tr hAccepted
  exact p20ForClaim_of_piCCSProtocolChecks
    (piCCSProtocolChecks_of_checkConj (hCheckLink ctx claim inst tr hAccepted))

theorem piCCSP20StrongLink_of_strongCheckBundleLinkAssumption
  (hCheckLink : PiCCSStrongCheckBundleLinkAssumption) :
  PiCCSP20StrongLinkAssumption := by
  intro ctx claim inst tr hAccepted
  exact p20ForClaim_of_piCCSProtocolChecks (hCheckLink ctx claim inst tr hAccepted)

theorem piCCSStrongCheckLinkAssumption_of_checkLinkAssumption
  (hCheckLink : PiCCSCheckLinkAssumption) :
  PiCCSStrongCheckLinkAssumption := by
  intro ctx claim inst tr hAcceptedStrong
  exact hCheckLink ctx claim inst tr (sumcheckAcceptedStrong_implies_accepted hAcceptedStrong)

theorem piCCSStrongCheckBundleLinkAssumption_of_checkBundleLinkAssumption
  (hCheckLink : PiCCSCheckBundleLinkAssumption) :
  PiCCSStrongCheckBundleLinkAssumption := by
  intro ctx claim inst tr hAcceptedStrong
  exact hCheckLink ctx claim inst tr (sumcheckAcceptedStrong_implies_accepted hAcceptedStrong)

theorem piCCSProtocolAssumption_of_checkBundleProtocolAssumption
  (hProto : PiCCSCheckBundleProtocolAssumption) :
  PiCCSProtocolAssumption := by
  exact ⟨
    hProto.1,
    piCCSArithmeticLink_of_checkBundleLinkAssumption hProto.2.1,
    hProto.2.2
  ⟩

theorem piCCSStrongCheckBundleProtocolAssumption_of_checkBundleProtocolAssumption
  (hProto : PiCCSCheckBundleProtocolAssumption) :
  PiCCSStrongCheckBundleProtocolAssumption := by
  exact ⟨
    sumcheckStrongSoundnessAssumption_of_soundnessAssumption hProto.1.1,
    hProto.1.2,
    piCCSStrongCheckBundleLinkAssumption_of_checkBundleLinkAssumption hProto.2.1,
    hProto.2.2
  ⟩

theorem piCCSStrongProtocolAssumption_of_strongCheckBundleProtocolAssumption
  (hProto : PiCCSStrongCheckBundleProtocolAssumption) :
  PiCCSStrongProtocolAssumption := by
  exact ⟨
    hProto.1,
    hProto.2.1,
    piCCSArithmeticStrongLink_of_strongCheckBundleLinkAssumption hProto.2.2.1,
    hProto.2.2.2
  ⟩

theorem piCCSStrongProtocolAssumption_of_checkBundleProtocolAssumption
  (hProto : PiCCSCheckBundleProtocolAssumption) :
  PiCCSStrongProtocolAssumption := by
  exact piCCSStrongProtocolAssumption_of_strongCheckBundleProtocolAssumption
    (piCCSStrongCheckBundleProtocolAssumption_of_checkBundleProtocolAssumption hProto)

theorem piCCSStrongProtocolAssumption_of_protocolAssumption
  (hProto : PiCCSProtocolAssumption) :
  PiCCSStrongProtocolAssumption := by
  exact ⟨
    sumcheckStrongSoundnessAssumption_of_soundnessAssumption hProto.1.1,
    hProto.1.2,
    piCCSArithmeticStrongLinkAssumption_of_linkAssumption hProto.2.1,
    hProto.2.2
  ⟩

/--
Strong-IR soundness skeleton for Pi_CCS:
an accepted transcript plus CCS-side obligations yields both paper relations.
-/
theorem piCCSStrongIR_relations_of_assumptions
  (hSumSound : SumcheckSoundnessAssumption)
  (hLink : PiCCSArithmeticLinkAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAccepted : SumcheckAcceptedProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  have hClaimTrue : SumcheckClaimTrue inst :=
    sumcheckAccepted_implies_claim_of_assumption hSumSound hAccepted
  have hArith : ClaimArithmeticValid ctx claim :=
    hLink ctx claim inst tr hAccepted hClaimTrue
  have hEval : EvalClaimValid ctx claim := ⟨hShape, hArith⟩
  exact ⟨
    ⟨hBar, hA, hB, hP10⟩,
    ⟨hEval, hWitness, hNorm⟩
  ⟩

/--
Strong-IR soundness corollary in CE-valid form.
-/
theorem piCCSStrongIR_ceValid_of_assumptions
  (hSumSound : SumcheckSoundnessAssumption)
  (hLink : PiCCSArithmeticLinkAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAccepted : SumcheckAcceptedProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  rcases piCCSStrongIR_relations_of_assumptions
      hSumSound hLink hShape hBar hA hB hP10 hAccepted hWitness hNorm with
    ⟨hCCS, hCE⟩
  exact ceValid_of_relations hCCS hCE

theorem piCCSStrongIR_relations_of_p20LinkAssumptions
  (hSumSound : SumcheckSoundnessAssumption)
  (hP20Link : PiCCSP20LinkAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAccepted : SumcheckAcceptedProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  exact piCCSStrongIR_relations_of_assumptions
    hSumSound
    (piCCSArithmeticLink_of_p20LinkAssumption hP20Link)
    hShape hBar hA hB hP10 hAccepted hWitness hNorm

theorem piCCSStrongIR_ceValid_of_p20LinkAssumptions
  (hSumSound : SumcheckSoundnessAssumption)
  (hP20Link : PiCCSP20LinkAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAccepted : SumcheckAcceptedProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact piCCSStrongIR_ceValid_of_assumptions
    hSumSound
    (piCCSArithmeticLink_of_p20LinkAssumption hP20Link)
    hShape hBar hA hB hP10 hAccepted hWitness hNorm

theorem piCCSStrongIR_relations_of_protocolAssumption_with_strongAccepted
  (hProto : PiCCSProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : SumcheckAcceptedStrongProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  exact piCCSStrongIR_relations_of_assumptions hProto.1.1 hProto.2.1
    hShape hBar hA hB hP10
    (sumcheckAcceptedStrong_implies_accepted hAcceptedStrong)
    hWitness hNorm

theorem piCCSStrongIR_ceValid_of_protocolAssumption_with_strongAccepted
  (hProto : PiCCSProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : SumcheckAcceptedStrongProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact piCCSStrongIR_ceValid_of_assumptions hProto.1.1 hProto.2.1
    hShape hBar hA hB hP10
    (sumcheckAcceptedStrong_implies_accepted hAcceptedStrong)
    hWitness hNorm

/--
Strong-IR soundness skeleton variant that consumes a round-consistent accepted
transcript and strong SumCheck soundness.
-/
theorem piCCSStrongIR_relations_of_strongAssumptions
  (hSumStrongSound : SumcheckStrongSoundnessAssumption)
  (hLinkStrong : PiCCSArithmeticStrongLinkAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : SumcheckAcceptedStrongProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  have hClaimTrue : SumcheckClaimTrue inst :=
    (sumcheckAcceptedStrong_implies_result_of_assumption hSumStrongSound hAcceptedStrong).1
  have hArith : ClaimArithmeticValid ctx claim :=
    hLinkStrong ctx claim inst tr hAcceptedStrong hClaimTrue
  have hEval : EvalClaimValid ctx claim := ⟨hShape, hArith⟩
  exact ⟨
    ⟨hBar, hA, hB, hP10⟩,
    ⟨hEval, hWitness, hNorm⟩
  ⟩

theorem piCCSStrongIR_ceValid_of_strongAssumptions
  (hSumStrongSound : SumcheckStrongSoundnessAssumption)
  (hLinkStrong : PiCCSArithmeticStrongLinkAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : SumcheckAcceptedStrongProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  rcases piCCSStrongIR_relations_of_strongAssumptions
      hSumStrongSound hLinkStrong hShape hBar hA hB hP10 hAcceptedStrong hWitness hNorm with
    ⟨hCCS, hCE⟩
  exact ceValid_of_relations hCCS hCE

theorem piCCSStrongIR_relations_of_p20StrongLinkAssumptions
  (hSumStrongSound : SumcheckStrongSoundnessAssumption)
  (hP20Link : PiCCSP20StrongLinkAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : SumcheckAcceptedStrongProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  exact piCCSStrongIR_relations_of_strongAssumptions
    hSumStrongSound
    (piCCSArithmeticStrongLink_of_p20StrongLinkAssumption hP20Link)
    hShape hBar hA hB hP10 hAcceptedStrong hWitness hNorm

theorem piCCSStrongIR_ceValid_of_p20StrongLinkAssumptions
  (hSumStrongSound : SumcheckStrongSoundnessAssumption)
  (hP20Link : PiCCSP20StrongLinkAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : SumcheckAcceptedStrongProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact piCCSStrongIR_ceValid_of_strongAssumptions
    hSumStrongSound
    (piCCSArithmeticStrongLink_of_p20StrongLinkAssumption hP20Link)
    hShape hBar hA hB hP10 hAcceptedStrong hWitness hNorm

/--
Strong-IR completeness skeleton for Pi_CCS:
if arithmetic validity encodes a true SumCheck claim, completeness returns an accepted transcript.
-/
theorem piCCSStrongIR_completeness_of_assumptions
  (hSumComp : SumcheckCompletenessAssumption)
  (hEncode : PiCCSSumcheckEncodingAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {inst : SumcheckInstance}
  (hArith : ClaimArithmeticValid ctx claim) :
  ∃ tr : SumcheckTranscript, SumcheckAcceptedProp inst tr := by
  have hClaimTrue : SumcheckClaimTrue inst := hEncode ctx claim inst hArith
  exact sumcheckCompleteness_of_assumption hSumComp hClaimTrue

/--
Protocol-assumption wrapper: expose both soundness and completeness skeleton theorems.
-/
theorem piCCSStrongIR_relations_of_protocolAssumption
  (hProto : PiCCSProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAccepted : SumcheckAcceptedProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  exact piCCSStrongIR_relations_of_assumptions hProto.1.1 hProto.2.1
    hShape hBar hA hB hP10 hAccepted hWitness hNorm

/--
Protocol-assumption wrapper for completeness-side existence of accepted transcripts.
-/
theorem piCCSStrongIR_completeness_of_protocolAssumption
  (hProto : PiCCSProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {inst : SumcheckInstance}
  (hArith : ClaimArithmeticValid ctx claim) :
  ∃ tr : SumcheckTranscript, SumcheckAcceptedProp inst tr := by
  exact piCCSStrongIR_completeness_of_assumptions hProto.1.2 hProto.2.2 hArith

theorem piCCSStrongIR_relations_of_checkBundleProtocolAssumption
  (hProto : PiCCSCheckBundleProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAccepted : SumcheckAcceptedProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  exact piCCSStrongIR_relations_of_protocolAssumption
    (piCCSProtocolAssumption_of_checkBundleProtocolAssumption hProto)
    hShape hBar hA hB hP10 hAccepted hWitness hNorm

theorem piCCSStrongIR_completeness_of_checkBundleProtocolAssumption
  (hProto : PiCCSCheckBundleProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {inst : SumcheckInstance}
  (hArith : ClaimArithmeticValid ctx claim) :
  ∃ tr : SumcheckTranscript, SumcheckAcceptedProp inst tr := by
  exact piCCSStrongIR_completeness_of_protocolAssumption
    (piCCSProtocolAssumption_of_checkBundleProtocolAssumption hProto)
    hArith

theorem piCCSStrongIR_relations_of_strongProtocolAssumption
  (hProto : PiCCSStrongProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : SumcheckAcceptedStrongProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  exact piCCSStrongIR_relations_of_strongAssumptions
    hProto.1 hProto.2.2.1
    hShape hBar hA hB hP10 hAcceptedStrong hWitness hNorm

theorem piCCSStrongIR_completeness_of_strongProtocolAssumption
  (hProto : PiCCSStrongProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {inst : SumcheckInstance}
  (hArith : ClaimArithmeticValid ctx claim) :
  ∃ tr : SumcheckTranscript, SumcheckAcceptedProp inst tr := by
  exact piCCSStrongIR_completeness_of_assumptions hProto.2.1 hProto.2.2.2 hArith

theorem piCCSStrongIR_relations_of_strongCheckBundleProtocolAssumption
  (hProto : PiCCSStrongCheckBundleProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : SumcheckAcceptedStrongProp inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  exact piCCSStrongIR_relations_of_strongProtocolAssumption
    (piCCSStrongProtocolAssumption_of_strongCheckBundleProtocolAssumption hProto)
    hShape hBar hA hB hP10 hAcceptedStrong hWitness hNorm

theorem piCCSStrongIR_completeness_of_strongCheckBundleProtocolAssumption
  (hProto : PiCCSStrongCheckBundleProtocolAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {inst : SumcheckInstance}
  (hArith : ClaimArithmeticValid ctx claim) :
  ∃ tr : SumcheckTranscript, SumcheckAcceptedProp inst tr := by
  exact piCCSStrongIR_completeness_of_strongProtocolAssumption
    (piCCSStrongProtocolAssumption_of_strongCheckBundleProtocolAssumption hProto)
    hArith

/--
Bridge from the existing protocol skeleton to Pi_CCS relations (sumcheck-independent shell).
-/
theorem piCCSRelations_of_protocolSkeleton
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP20 : p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CCSRelation ctx claim ∧ CERelation ctx claim witness := by
  have hCEValid : CEValid ctx claim witness :=
    superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm
  exact ceValid_iff_relations.mp hCEValid

end SuperNeo
