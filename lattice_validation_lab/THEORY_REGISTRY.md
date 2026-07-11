# Theory Registry

| ID | Theory | Operational question | Required control | Failure condition |
|---|---|---|---|---|
| LL-01 | Observer–Actor–Bridge product | Does a forward-only O×A×B score improve held-out prediction? | Additive and ordinary feature baselines | No reliable gain on held-out participants |
| LL-02 | Phi scaling | Does φ uniquely improve prediction relative to other constants? | 1.0, 1.5, π/2, 2.0 and random fixed constants | Equivalent rankings or absorbed scaling |
| LL-03 | Eight-corner state structure | Do eight states emerge without forcing k=8? | Compare k=2..12 using held-out criteria | Another k is consistently preferred |
| LL-04 | Coherence bounce | Does contradiction-triggered reorganization improve recovery? | No-bounce and random-bounce engines | Bounce is neutral or harmful |
| LL-05 | Fibonacci memory | Does Fibonacci decay outperform exponential and learned decay? | Exponential, power-law and optimized kernels | Standard kernels perform equally or better |
| LL-06 | Quantum φ signature | Do preregistered exact outcomes deviate from calibrated QM controls? | Topology-matched non-φ circuits | Signature survives angle replacement |
| LL-07 | Linguistic lattice | Do O/A/B semantic coordinates add value beyond embeddings? | Matched embedding baseline | No held-out improvement |
| LL-08 | ARC action policy | Does lattice scoring improve unseen-game performance? | Deterministic heuristics and ablations | Baselines outperform it |
| LL-09 | Periodic-table mapping | Does the lattice predict unseen chemical properties? | Standard periodic descriptors | Mapping remains descriptive only |
| LL-10 | Coherence maintenance | Can the system maintain coherent state organization under contradiction, interruption, or noisy input without collapsing into arbitrary output? | Shuffled-state control and random-warning control | LL-10 fails if coherence scores, limiting constraints, or warnings are no more stable than shuffled-state or random-warning controls under matched perturbation. |
| LL-11 | Identity persistence | Can the system preserve a recognizable self-state across time without merely copying the previous output? | Stateless diagnostic run and direct-copy baseline | LL-11 fails if identity-like persistence is indistinguishable from stateless recomputation or direct copying of prior output. |
| LL-12 | Self-model accuracy | Can the system accurately report what it can do, what it cannot do, and why a claim is unsupported? | Report-only confidence baseline and always-uncertain conservative baseline | LL-12 fails if the system's self-report does not reliably match external test outcomes or if it overclaims unsupported results. |
| LL-13 | Integration | Can the system integrate observation, action, bridge state, memory, and language into one coherent diagnostic report without losing claim discipline? | Ablated reports missing one or more of Observer, Actor, Bridge, memory, or evidence fields | LL-13 fails if integrated reports do not outperform ablated reports in predicting task outcome, limitation class, or claim-validity category. |
