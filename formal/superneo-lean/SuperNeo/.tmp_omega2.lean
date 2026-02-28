import SuperNeo.Decomp

example (a q0 bi r0 r : Int)
  (hA : a = q0 * bi + r0)
  (hr : r = r0 - bi) :
  a = (q0 + 1) * bi + r := by
  omega

example (a q0 bi r0 r : Int)
  (hA : a = q0 * bi + r0)
  (hr : r = r0 + bi) :
  a = (q0 - 1) * bi + r := by
  omega
