import Nightstream.Implementation.R1CS.Core.SevenBytePacking
import tests.Axioms.Support

/-! Fail-closed dependency gate for shared seven-byte packing. -/

/-- info: 'Nightstream.Implementation.R1CS.SevenBytePacking.packBytesAsNats_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SevenBytePacking.packBytesAsNats_length
