"""Reproduce the SIS estimates used by the Nebula security note.

Run with malb/lattice-estimator commit 3e48ef421ec256afddb3e7d2249a77eab6e9ba12:

    sage -python scripts/estimate_nebula_sis.sage

The estimator package must be on PYTHONPATH. The protocol uses the rough
model as its conservative floor; the full model is printed for comparison
with SuperNeo Appendix D.8.
"""

from sage.all import sqrt

from estimator import SIS


Q = 2**64 - 2**32 + 1
D = 54
ESTIMATOR_COMMIT = "3e48ef421ec256afddb3e7d2249a77eab6e9ba12"


def estimate_coefficients(label, kappa, coefficient_columns, infinity_bound=2):
    params = SIS.Parameters(
        n=kappa * D,
        q=Q,
        m=coefficient_columns,
        length_bound=infinity_bound * sqrt(coefficient_columns),
        norm=2,
        tag=label,
    )
    print(f"\n{label}: {params}")
    print("rough:")
    SIS.estimate.rough(params)
    print("full:")
    SIS.estimate(params)


def estimate(label, kappa, ring_columns, infinity_bound=2):
    estimate_coefficients(label, kappa, ring_columns * D, infinity_bound)


print(f"lattice-estimator commit: {ESTIMATOR_COMMIT}")
# Rejected design: rough 59.9 bits, full 87.6 bits.
estimate("Nebula long binding, rejected rank 1", 1, 22_147)
# Adopted long-message map: rough 167.0 bits, full 190.2 bits.
estimate("Nebula long binding, adopted rank 2", 2, 22_147)
# Adopted short digest-compression map: rough 223.1 bits, full 242.1 bits.
estimate("Nebula short digest compression", 1, 82)
# SuperNeo Appendix B.2 relaxed-binding comparison. Appendix B requires
# hardness of MSIS at infinity bound 8*T*B over n_F=2^30 scalar
# coefficients, with T=216 and B=2^14.
estimate_coefficients("SuperNeo Appendix B.2 relaxed binding", 18, 2**30, infinity_bound=8 * 216 * 2**14)
