import Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalGoldilocksCertificate

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate.order_factorisation' does not depend on any axioms -/
#guard_msgs in
#audit_axioms GoldilocksCertificate.order_factorisation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate.fermat' does not depend on any axioms -/
#guard_msgs in
#audit_axioms GoldilocksCertificate.fermat

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate.order_not_halved' does not depend on any axioms -/
#guard_msgs in
#audit_axioms GoldilocksCertificate.order_not_halved

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate.order_not_thirded' does not depend on any axioms -/
#guard_msgs in
#audit_axioms GoldilocksCertificate.order_not_thirded

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate.order_not_fifthed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms GoldilocksCertificate.order_not_fifthed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate.order_not_seventeenthed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms GoldilocksCertificate.order_not_seventeenthed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate.order_not_two_five_seventhed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms GoldilocksCertificate.order_not_two_five_seventhed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate.order_not_sixtyfive_five_three_seventhed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms GoldilocksCertificate.order_not_sixtyfive_five_three_seventhed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate.certificate_residues_ne_one' does not depend on any axioms -/
#guard_msgs in
#audit_axioms GoldilocksCertificate.certificate_residues_ne_one

end NightstreamTests.Axioms.CanonicalGoldilocksCertificate
