import Nightstream.Implementation.R1CS.Canonical.KZeroCheck
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKZeroCheck

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.zeroRows_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KZeroCheck.zeroRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.zeroColumns_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KZeroCheck.zeroColumns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.zeroColumns_nodup' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KZeroCheck.zeroColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.zeroRows_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KZeroCheck.zeroRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.zeroRows_honest' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KZeroCheck.zeroRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.zeroRows_conservation' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KZeroCheck.zeroRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.zeroCost_rows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KZeroCheck.zeroCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.zeroCost_columns' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KZeroCheck.zeroCost_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.carriedZeroRows_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KZeroCheck.carriedZeroRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.carriedZeroRows_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KZeroCheck.carriedZeroRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.carriedZeroRows_honest' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KZeroCheck.carriedZeroRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.carriedZeroCost_rows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KZeroCheck.carriedZeroCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.paddingRows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KZeroCheck.paddingRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.paddingRows_length_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KZeroCheck.paddingRows_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.paddingRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KZeroCheck.paddingRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.paddingRows_honest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KZeroCheck.paddingRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.paddingCost_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KZeroCheck.paddingCost_rows


/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.carriedZeroRows_conservation' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KZeroCheck.carriedZeroRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KZeroCheck.paddingRows_conservation' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KZeroCheck.paddingRows_conservation

end NightstreamTests.Axioms.CanonicalKZeroCheck
