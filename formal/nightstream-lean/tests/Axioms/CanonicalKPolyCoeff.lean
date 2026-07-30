import Nightstream.Implementation.R1CS.Canonical.KPolyCoeff
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKPolyCoeff

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.polyAdd_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyCoeff.polyAdd_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.polyScale_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPolyCoeff.polyScale_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.coeffAt_polyScale' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyCoeff.coeffAt_polyScale

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.coeffAt_polyAdd' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPolyCoeff.coeffAt_polyAdd

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.canonical_polyMul' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyCoeff.canonical_polyMul

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.coeffAt_polyMul' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyCoeff.coeffAt_polyMul

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.toList_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPolyCoeff.toList_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.coeffAt_toList' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyCoeff.coeffAt_toList

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.convolution_canonical' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPolyCoeff.convolution_canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.convolution_add_zero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPolyCoeff.convolution_add_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.coeffAt_beyond_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPolyCoeff.coeffAt_beyond_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.polyMul_length_cons' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyCoeff.polyMul_length_cons

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.polyMul_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyCoeff.polyMul_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.rawProduct_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyCoeff.rawProduct_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyCoeff.list_ext_coeffAt' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyCoeff.list_ext_coeffAt

end NightstreamTests.Axioms.CanonicalKPolyCoeff
