# Security parameter review

The implementation now separates evidence from policy.

For each concrete one-joint relation, `neo-params` computes the exact combined
statistical error from:

- the joint SumCheck field term;
- the paper mixing term; and
- the coordinate-fork term over the strong challenge set.

The default shape constructors bind the strongest lambda supported by this
census, capped by the SuperNeo Appendix B.2 lambda. They do not apply a
repository-invented minimum, safety margin, lifetime target, or random-oracle
query cap.

An application owner can supply an authoritative minimum through the explicit
`*_with` constructors. The remaining open task is an independent review of
the census and a deployment-specific choice of acceptable end-to-end
security.
