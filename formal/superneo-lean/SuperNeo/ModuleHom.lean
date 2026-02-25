import SuperNeo.EvalHom

namespace SuperNeo

open F

private def dotF (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    Id.run do
      let mut acc : F := 0
      for i in [0:a.size] do
        acc := acc + a[i]! * b[i]!
      return acc

/-- Lightweight computational interface for R-module homomorphisms on vectors. -/
structure VecModuleHom where
  map : Array F -> Array F

/-- Lightweight computational interface for scalar-valued module homomorphisms. -/
structure ScalarModuleHom where
  map : Array F -> F

def preservesAddVec (h : VecModuleHom) (x y : Array F) : Bool :=
  if x.size != y.size then
    false
  else
    decide (h.map (vecAdd x y) = vecAdd (h.map x) (h.map y))

def preservesScaleVec (h : VecModuleHom) (s : F) (x : Array F) : Bool :=
  decide (h.map (vecScale s x) = vecScale s (h.map x))

def preservesAddScalar (h : ScalarModuleHom) (x y : Array F) : Bool :=
  if x.size != y.size then
    false
  else
    decide (h.map (vecAdd x y) = h.map x + h.map y)

def preservesScaleScalar (h : ScalarModuleHom) (s : F) (x : Array F) : Bool :=
  decide (h.map (vecScale s x) = s * h.map x)

def idHom : VecModuleHom := { map := fun x => x }

def scaleHom (s : F) : VecModuleHom := { map := fun x => vecScale s x }

def dotHom (w : Array F) : ScalarModuleHom := { map := fun x => dotF x w }

def moduleHomSanity : Bool :=
  let x := #[1, 2, 3, 4]
  let y := #[5, 6, 7, 8]
  let s : F := 9
  let h1 := idHom
  let h2 := scaleHom 3
  let h3 := dotHom #[2, 1, 0, 4]
  preservesAddVec h1 x y &&
    preservesScaleVec h1 s x &&
    preservesAddVec h2 x y &&
    preservesScaleVec h2 s x &&
    preservesAddScalar h3 x y &&
    preservesScaleScalar h3 s x

theorem preservesAddVec_sound
  {h : VecModuleHom} {x y : Array F}
  (hOk : preservesAddVec h x y = true) :
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) := by
  unfold preservesAddVec at hOk
  by_cases hsz : x.size != y.size
  · simp [hsz] at hOk
  · simp [hsz] at hOk
    exact hOk

theorem preservesAddVec_complete
  {h : VecModuleHom} {x y : Array F}
  (hSize : x.size = y.size)
  (hProp : h.map (vecAdd x y) = vecAdd (h.map x) (h.map y)) :
  preservesAddVec h x y = true := by
  unfold preservesAddVec
  simp [hSize, decide_eq_true hProp]

theorem preservesScaleVec_sound
  {h : VecModuleHom} {s : F} {x : Array F}
  (hOk : preservesScaleVec h s x = true) :
  h.map (vecScale s x) = vecScale s (h.map x) := by
  unfold preservesScaleVec at hOk
  exact decide_eq_true_eq.mp hOk

theorem preservesScaleVec_complete
  {h : VecModuleHom} {s : F} {x : Array F}
  (hProp : h.map (vecScale s x) = vecScale s (h.map x)) :
  preservesScaleVec h s x = true := by
  unfold preservesScaleVec
  exact decide_eq_true hProp

theorem preservesAddScalar_sound
  {h : ScalarModuleHom} {x y : Array F}
  (hOk : preservesAddScalar h x y = true) :
  h.map (vecAdd x y) = h.map x + h.map y := by
  unfold preservesAddScalar at hOk
  by_cases hsz : x.size != y.size
  · simp [hsz] at hOk
  · simp [hsz] at hOk
    exact hOk

theorem preservesAddScalar_complete
  {h : ScalarModuleHom} {x y : Array F}
  (hSize : x.size = y.size)
  (hProp : h.map (vecAdd x y) = h.map x + h.map y) :
  preservesAddScalar h x y = true := by
  unfold preservesAddScalar
  simp [hSize, decide_eq_true hProp]

theorem preservesScaleScalar_sound
  {h : ScalarModuleHom} {s : F} {x : Array F}
  (hOk : preservesScaleScalar h s x = true) :
  h.map (vecScale s x) = s * h.map x := by
  unfold preservesScaleScalar at hOk
  exact decide_eq_true_eq.mp hOk

theorem preservesScaleScalar_complete
  {h : ScalarModuleHom} {s : F} {x : Array F}
  (hProp : h.map (vecScale s x) = s * h.map x) :
  preservesScaleScalar h s x = true := by
  unfold preservesScaleScalar
  exact decide_eq_true hProp

end SuperNeo
