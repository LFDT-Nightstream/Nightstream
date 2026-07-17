import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc

/-! Narrow compile-time checks for the model-level Π_CCS NC direct-packing,
mixed-polynomial, and conditional terminal-identity surface. -/

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial
open Nightstream.Implementation.R1CS.PiCcsNc.Terminal
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.ProjectionNecessity
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.SuperNeo.Concrete

example (value : F) :
    directDiagonal [value] 0 0 = K.embed value := by
  simp [directDiagonal, ringDegree]

example (value : F) :
    directDiagonal [value] 0 1 = K.zero := by
  simp [directDiagonal, ringDegree]

example
    (prime : EuclidPrime goldilocksP)
    {assignments : List (List F)}
    (norms : InputsNormBoundedTwo assignments)
    (shape : Shape) (betaM betaA : List K) (gamma : K) :
    trueInitial shape betaM betaA gamma assignments = K.zero :=
  trueInitial_eq_zero_of_normBounded
    prime norms shape betaM betaA gamma

example
    (shape : Shape) (betaM betaA : List K) (gamma : K)
    (assignments : List (List F)) {column lane : Nat}
    (betaMLength : betaM.length = shape.ellM)
    (betaALength : betaA.length = shape.ellD)
    (columnLt : column < shape.columnDomain)
    (laneLt : lane < shape.laneDomain) :
    qNc shape betaM betaA gamma assignments
        (cubePoint shape.ellM column)
        (cubePoint shape.ellD lane) =
      qNcOnCube betaM betaA gamma assignments column lane :=
  qNc_cubePoint_eq_qNcOnCube shape betaM betaA gamma assignments
    betaMLength betaALength columnLt laneLt

example
    {shape : Shape} {assignments : List (List F)}
    {s : List K} {outputs : List YZcol}
    (bound : YZcolBound shape assignments s outputs)
    (betaM betaA : List K) (gamma : K) (alpha : List K) :
    terminalRhs shape betaM betaA gamma outputs s alpha =
      qNc shape betaM betaA gamma assignments s alpha :=
  terminalRhs_eq_qNc_of_yZcolBound
    bound betaM betaA gamma alpha

example :
    ¬ YZcolBound counterexampleShape counterexampleAssignments []
      erasedOutputs :=
  erasedOutputs_not_yZcolBound

example :
    TerminalMismatch counterexampleShape [] [] K.one
      counterexampleAssignments erasedOutputs [] [] :=
  erasedOutputs_terminalMismatch

example
    {shape : Shape} {radix : F} {parentAssignment : List F}
    {rawChildren : List (List F)}
    (decomposition : PointwiseRadixDecomposition
      shape radix parentAssignment rawChildren)
    (sCol : List K) {lane : Nat}
    (laneLt : lane < shape.laneDomain) :
    authoritativeYZcol shape parentAssignment sCol lane =
      radixWeightedChildProjection shape radix rawChildren sCol lane :=
  authoritativeYZcol_eq_radixWeightedChildProjection
    decomposition sCol laneLt

example
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (sCol : List K) {lane : Nat}
    (laneLt : lane < shape.laneDomain) :
    radixWeightedChildProjection shape radix rawChildren sCol lane =
      radixWeightedAuthoritativeYZcol
        shape radix rawChildren sCol lane :=
  radixWeightedChildProjection_eq_weightedAuthoritativeYZcol
    shape radix rawChildren sCol laneLt

example
    {shape : Shape} {radix : F} {parent : DelayedParent}
    {rawChildren : List (List F)}
    (bound : DelayedParentProjectionBound
      shape radix parent rawChildren)
    {lane : Nat} (laneLt : lane < shape.laneDomain) :
    parent.yZcol lane =
      radixWeightedChildProjection
        shape radix rawChildren parent.sCol lane :=
  exactLane_of_delayedParentProjectionBound bound laneLt

example
    {shape : Shape} {radix : F} {parent : DelayedParent}
    {rawChildren : List (List F)}
    (bound : DelayedParentProjectionBound
      shape radix parent rawChildren)
    {lane : Nat} (laneLt : lane < shape.laneDomain) :
    parent.yZcol lane =
      radixWeightedAuthoritativeYZcol
        shape radix rawChildren parent.sCol lane :=
  exactWeightedAuthoritativeYZcolLane_of_bound bound laneLt

example
    {shape : Shape} {radix : F}
    {stateParent checkedOld : DelayedParent}
    {rawRunningChildren nextAssignments : List (List F)}
    {nextSCol : List K} {nextOutputs : List YZcol}
    (step : DelayedProjectionStep shape radix stateParent checkedOld
      rawRunningChildren nextAssignments nextSCol nextOutputs) :
    DelayedParentProjectionBound
        shape radix stateParent rawRunningChildren ∧
      YZcolBound shape nextAssignments nextSCol nextOutputs :=
  delayedProjectionStep_transfer step

example
    {shape : Shape} {radix : F}
    {stateParent checkedOld : DelayedParent}
    {rawRunningChildren nextAssignments : List (List F)}
    {nextSCol : List K} {nextOutputs : List YZcol}
    (step : DelayedProjectionStep shape radix stateParent checkedOld
      rawRunningChildren nextAssignments nextSCol nextOutputs) :
    ¬ DelayedParentProjectionMismatch
      shape radix stateParent rawRunningChildren :=
  not_delayedParentProjectionMismatch_of_step step

example
    (producerBeta : K) (width lane : Nat)
    (laneLt : lane < 2 ^ width) :
    betaPowerSelector producerBeta (cubePoint width lane) =
      powK producerBeta lane :=
  betaPowerSelector_cubePoint producerBeta width lane laneLt

example
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (wellShaped : DelayedResidualShape shape oldS rawChildren) :
    delayedResidualCubeSum shape radix rawChildren
        producerBeta batchWeight oldS =
      K.mul batchWeight
        (activeRawProjectionAtProducerBeta
          shape radix rawChildren producerBeta oldS) :=
  delayedResidualCubeSum_eq_weightedCompactOldProjection
    shape radix rawChildren producerBeta batchWeight oldS wellShaped

example
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (terminalS terminalAlpha : List K)
    (wellShaped : DelayedResidualShape shape oldS rawChildren)
    (terminalShape : TerminalPointShape shape terminalS terminalAlpha) :
    delayedResidualPolynomial shape radix rawChildren
        producerBeta batchWeight oldS terminalS terminalAlpha =
      delayedResidualTerminalRhs shape radix rawChildren
        producerBeta batchWeight oldS terminalS terminalAlpha :=
  delayedResidualPolynomial_eq_terminalRhs
    shape radix rawChildren producerBeta batchWeight oldS
      terminalS terminalAlpha wellShaped terminalShape

example
    {shape : Shape} {oldS : List K} {rawChildren : List (List F)}
    (wellShaped : DelayedResidualShape shape oldS rawChildren)
    {assignment : List F} (assignmentMem : assignment ∈ rawChildren)
    {column : Nat} (columnLt : column < assignment.length) :
    column < shape.columnDomain :=
  rawChildCoordinate_lt_columnDomain wellShaped assignmentMem columnLt

example
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta : K) (s : List K) :
    compactOldPointEvaluation
        shape radix rawChildren producerBeta s =
      K.add
        (childC0ProjectionEvaluation
          shape radix rawChildren producerBeta s)
        (K.mul extensionGenerator
          (childC1ProjectionEvaluation
            shape radix rawChildren producerBeta s)) :=
  compactOldPointEvaluation_eq_childLimbEvaluations
    shape radix rawChildren producerBeta s

example
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) :
    (paddedRawChildProjectionCoefficients
      shape radix rawChildren s).drop 54 =
      List.replicate 10 K.zero := by
  simpa [ringDegree] using
    paddedRawChildProjectionCoefficients_drop_active
      shape radix rawChildren s

example
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) (claimedParentCoefficients : List K)
    (producerBeta : K)
    (accepted : Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
      (projectionIdentity shape radix rawChildren s
        claimedParentCoefficients producerBeta)) :
    (projectionIdentity shape radix rawChildren s
        claimedParentCoefficients producerBeta).Exact ∨
      Nightstream.SuperNeo.ProjectionCheck.BadRoot projectionOps
        (projectionIdentity shape radix rawChildren s
          claimedParentCoefficients producerBeta) :=
  acceptedProjectionIdentity_implies_exact_or_badRoot
    shape radix rawChildren s claimedParentCoefficients producerBeta accepted

example
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) (producerBeta : K) :
    Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
      (projectionIdentity shape radix rawChildren s
        (rawChildProjectionCoefficients shape radix rawChildren s)
        producerBeta) :=
  projectionIdentity_accepted_of_exact shape radix rawChildren s
    (rawChildProjectionCoefficients shape radix rawChildren s) producerBeta rfl

example
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (claimedParentCoefficients : List K)
    (wellShaped : DelayedResidualShape shape oldS rawChildren)
    (accepted : Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
      (projectionIdentity shape radix rawChildren oldS
        claimedParentCoefficients producerBeta)) :
    delayedResidualCubeSum shape radix rawChildren
        producerBeta batchWeight oldS =
      K.mul batchWeight
        (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
          claimedParentCoefficients producerBeta) ∨
      Nightstream.SuperNeo.ProjectionCheck.BadRoot projectionOps
        (projectionIdentity shape radix rawChildren oldS
          claimedParentCoefficients producerBeta) :=
  acceptedProjectionIdentity_implies_cubeSum_eq_claimed_or_badRoot
    shape radix rawChildren producerBeta batchWeight oldS
      claimedParentCoefficients wellShaped accepted
