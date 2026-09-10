import NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit

/-! Production allocation values for the public-input split. Core composition
uses the symbolic counts and does not import this module. -/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle.PaperAlgebra

theorem exactPrivateCount_eq : exactPrivateCount = 270 := by
  norm_num [exactPrivateCount, exactCoordinateCount,
    SignedSplitScalar.exactPrivateCount, ringDegree, publicRingColumns]

theorem exactRowCount_eq : exactRowCount = 4860 := by
  norm_num [exactRowCount, exactCoordinateCount,
    SignedSplitScalar.exactRowCount, ringDegree, publicRingColumns]

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit
