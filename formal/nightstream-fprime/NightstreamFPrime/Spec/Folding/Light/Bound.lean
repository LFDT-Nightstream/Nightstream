import NightstreamFPrime.Spec.Profile

/-! Cyclo (ePrint 2026/359) Theorem 3 on SuperNeo v1.2 Definition 22: the honest norm of
an accumulator that skips its norm check. -/

namespace NightstreamFPrime.Spec.GlobalParams

/-- A refresh parent combines `1 + k` sources, and each light fold adds one more. -/
def accumulatorBound (p : GlobalParams) (lightFolds : Nat) : Nat :=
  (1 + lightFolds + p.k) * p.expansionT * (p.b - 1)

theorem accumulatorBound_lt_bigB (p : GlobalParams) {lightFolds : Nat}
    (depth : 1 + lightFolds ≤ p.maxFresh) :
    p.accumulatorBound lightFolds < p.bigB :=
  p.rlc_bound_for depth

theorem production_accumulatorBound_lt_bigB {lightFolds : Nat}
    (depth : lightFolds ≤ 286) :
    productionGlobalParams.accumulatorBound lightFolds <
      productionGlobalParams.bigB :=
  productionGlobalParams.accumulatorBound_lt_bigB (by
    change 1 + lightFolds ≤ 287
    omega)

end NightstreamFPrime.Spec.GlobalParams
