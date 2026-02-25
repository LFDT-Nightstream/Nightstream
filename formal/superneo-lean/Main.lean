import SuperNeo

open SuperNeo


def main : IO UInt32 := do
  let okSuper := checkSuperNeoCases
  let okRing := checkRingMulCases
  let okNorm := checkNormCases
  let okSplit := checkSplitCases
  let okEq := checkEqCases
  let okMle := checkMleCases
  let okEmbeddingVec := checkEmbeddingVecCases
  let okEmbeddingMatrix := checkEmbeddingMatrixCases
  let okBarLiftVec := checkBarLiftVecCases
  let okBarLiftMatrix := checkBarLiftMatrixCases
  let okMatrixTransform := checkMatrixTransformCases
  let okEvalLink := checkEvalLinkCases
  let okEvalHom := checkEvalHomCases
  let okModuleHom := checkModuleHomCases
  let okInvertibility := checkInvertibilityCases
  let okSampling := checkSamplingCases
  let okEqLift := checkEqLiftCases
  let okPolyLemmas := checkPolyLemmaCases
  let okCoeffMaps := checkCoeffMapCases
  let okParams := checkParameterCases
  let okInterp := checkInterpCases
  let allOk :=
    okSuper && okRing && okNorm && okSplit && okEq && okMle &&
      okEmbeddingVec && okEmbeddingMatrix &&
      okBarLiftVec && okBarLiftMatrix && okMatrixTransform &&
      okEvalLink && okEvalHom && okModuleHom && okInvertibility && okSampling && okEqLift && okPolyLemmas &&
      okCoeffMaps && okParams && okInterp
  IO.println s!"superneo_cases={okSuper}"
  IO.println s!"ring_mul_cases={okRing}"
  IO.println s!"norm_cases={okNorm}"
  IO.println s!"split_cases={okSplit}"
  IO.println s!"eq_cases={okEq}"
  IO.println s!"mle_cases={okMle}"
  IO.println s!"embedding_vec_cases={okEmbeddingVec}"
  IO.println s!"embedding_matrix_cases={okEmbeddingMatrix}"
  IO.println s!"bar_lift_vec_cases={okBarLiftVec}"
  IO.println s!"bar_lift_matrix_cases={okBarLiftMatrix}"
  IO.println s!"matrix_transform_cases={okMatrixTransform}"
  IO.println s!"eval_link_cases={okEvalLink}"
  IO.println s!"eval_hom_cases={okEvalHom}"
  IO.println s!"module_hom_cases={okModuleHom}"
  IO.println s!"invertibility_cases={okInvertibility}"
  IO.println s!"sampling_cases={okSampling}"
  IO.println s!"eq_lift_cases={okEqLift}"
  IO.println s!"poly_lemma_cases={okPolyLemmas}"
  IO.println s!"coeff_map_cases={okCoeffMaps}"
  IO.println s!"parameter_cases={okParams}"
  IO.println s!"interp_cases={okInterp}"
  if allOk then
    IO.println "all_checks=true"
    pure 0
  else
    IO.println "all_checks=false"
    pure 1
