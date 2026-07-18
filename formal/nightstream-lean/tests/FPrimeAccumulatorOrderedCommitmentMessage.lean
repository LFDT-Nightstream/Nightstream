import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage

/-! Focused interface gate for the exact ordered-commitment hash message. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage

#check domainNats_eq
#check domainNats_canonical
#check serialize_injective
#check digest_eq_or_fieldHashCollision
#check claimDigest_eq_payloadDigest
#check fixed_serialize_length

#guard domainBytes.length = 63
#guard domainNats.length = 10
#guard fixedPreimageFieldCount 12 = 13642
