import SuperNeo.BarLift

namespace SuperNeo

/-- Coefficient map cf : R_q -> F_q^d (concrete encoding). -/
def cf (a : Coeffs) : Array F := a

/-- Inverse coefficient map cf^{-1} : F_q^d -> R_q (concrete encoding). -/
def cfInv (v : Array F) : Coeffs := v

theorem cfInv_cf (a : Coeffs) : cfInv (cf a) = a := rfl

theorem cf_cfInv (v : Array F) : cf (cfInv v) = v := rfl

theorem cf_size (a : Coeffs) : (cf a).size = a.size := rfl

theorem cfInv_size (v : Array F) : (cfInv v).size = v.size := rfl

theorem cf_vecAdd (a b : Coeffs) : cf (vecAdd a b) = vecAdd (cf a) (cf b) := rfl

theorem cfInv_vecAdd (v w : Array F) :
  cfInv (vecAdd v w) = vecAdd (cfInv v) (cfInv w) := rfl

theorem cf_vecScale (s : F) (a : Coeffs) :
  cf (vecScale s a) = vecScale s (cf a) := rfl

theorem cfInv_vecScale (s : F) (v : Array F) :
  cfInv (vecScale s v) = vecScale s (cfInv v) := rfl

theorem ct_cf (a : Coeffs) : ct (cf a) = ct a := rfl

theorem ct_cfInv (v : Array F) : ct (cfInv v) = ct v := rfl

theorem hasRingDegreeShape_cf_iff (a : Coeffs) :
  hasRingDegreeShape (cf a) ↔ hasRingDegreeShape a := by
  rfl

theorem hasRingDegreeShape_cfInv_iff (v : Array F) :
  hasRingDegreeShape (cfInv v) ↔ v.size = D := by
  unfold hasRingDegreeShape cfInv
  rfl

theorem ringMulShapeProp_cf_iff (a b : Coeffs) :
  ringMulShapeProp (cf a) (cf b) ↔ ringMulShapeProp a b := by
  rfl

theorem cfInv_mulRq_cf (a b : Coeffs) :
  cfInv (cf (mulRq a b)) = mulRq a b := rfl

theorem cf_mulRq_cfInv (v w : Array F) :
  cf (mulRq (cfInv v) (cfInv w)) = mulRq v w := rfl

theorem cfInv_mulRq_cfInv (v w : Array F) :
  cfInv (mulRq v w) = mulRq (cfInv v) (cfInv w) := rfl

theorem ct_mulRq_cfInv (v w : Array F) :
  ct (mulRq (cfInv v) (cfInv w)) = ct (mulRq v w) := rfl

def coeffMapRoundTripProp (v : Coeffs) : Prop :=
  cfInv (cf v) = v ∧ cf (cfInv v) = v

instance coeffMapRoundTripProp_decidable (v : Coeffs) : Decidable (coeffMapRoundTripProp v) := by
  unfold coeffMapRoundTripProp
  infer_instance

def coeffMapRoundTrip (v : Coeffs) : Bool :=
  decide (coeffMapRoundTripProp v)

theorem coeffMapRoundTrip_sound
  {v : Coeffs}
  (hOk : coeffMapRoundTrip v = true) :
  coeffMapRoundTripProp v := by
  unfold coeffMapRoundTrip at hOk
  exact decide_eq_true_eq.mp hOk

theorem coeffMapRoundTrip_theorem (v : Coeffs) : coeffMapRoundTripProp v := by
  exact ⟨cfInv_cf v, cf_cfInv v⟩

theorem coeffMapRoundTrip_true (v : Coeffs) : coeffMapRoundTrip v = true := by
  unfold coeffMapRoundTrip
  exact decide_eq_true (coeffMapRoundTrip_theorem v)

theorem ct_cfInv_cf (v : Coeffs) : ct (cfInv (cf v)) = ct v := by
  simpa [cfInv_cf] using congrArg ct (cfInv_cf v)

end SuperNeo
