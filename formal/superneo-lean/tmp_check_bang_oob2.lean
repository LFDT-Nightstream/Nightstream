import SuperNeo.Ring

namespace SuperNeo
open F

example {arr : Array F} {i : Nat} (h : ¬ i < arr.size) : arr[i]! = (0 : F) := by
  simp [h, (rfl : (default : F) = 0)]

end SuperNeo
