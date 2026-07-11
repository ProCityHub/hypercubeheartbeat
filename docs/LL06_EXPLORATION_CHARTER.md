# LL-06 Exploration Charter

EXPLORATORY — UNREGISTERED — generates hypotheses only.

Nothing here is evidence.

Confirmation requires fresh hardware runs under a frozen preregistration.

## Purpose

This charter defines the rules for any future exploratory LL-06 report generated from the ten IBM job records intaken through the AI infrastructure.

The purpose of exploration is to describe training-side material and generate candidate predictions for future fresh hardware runs.

The purpose is not to confirm, validate, score, or upgrade any claim.

## Training-side rule

The ten IBM jobs already intaken for LL-06 are training-side forever.

No result from those ten jobs may serve as confirmatory evidence under any future preregistration.

They may be used only to describe exploratory material, identify data quality issues, test analysis scripts, estimate uncertainty, generate candidate predictions, and design future fresh hardware tests.

## Required report header

Every LL-06 exploration report must begin with:

EXPLORATORY — UNREGISTERED — generates hypotheses only.

Nothing here is evidence.

Confirmation requires fresh hardware runs under a frozen prereg.

## Allowed analysis

An LL-06 exploration report may include metadata inventory, descriptive count statistics, count uncertainty, ratio uncertainty, multiple-comparisons pricing, entropy, normalized entropy, hardware versus noiseless simulation comparison, rank-stability checks, repeated-job stability when circuit assignment is known, backend comparison with limitations, and duplicate detection.

## Count uncertainty

Every count N must be reported as:

N ± sqrt(N)

## Ratio uncertainty

Ratios may be reported only as descriptive quantities.

If R = a / b, uncertainty must be:

sigma_R = R * sqrt((1/a) + (1/b))

Ratios must be displayed as:

R ± sigma_R

## Multiple-comparisons pricing

Each histogram must state how many possible outcome pairs exist.

If K outcomes are observed:

pair_count = K * (K - 1) / 2

For a complete 5-bit histogram:

K = 32

pair_count = 496

If using only the top m outcomes:

pair_count_top_m = m * (m - 1) / 2

The report must state:

With many possible pairs, some ratios may land near phi by arithmetic chance. These observations are hypothesis-generating only.

## Entropy

Shannon entropy:

H = -sum(p_i * log2(p_i))

Normalized entropy:

H_norm = H / log2(K_possible)

where:

K_possible = 2 ** number_of_bits

## Hardware versus noiseless simulation

If Qiskit is available and QASM parsing succeeds, the report may simulate the committed circuit noiselessly, produce a predicted probability vector, and compare hardware count distribution to simulation.

Required descriptive statistic:

TVD = 0.5 * sum(abs(p_hardware(i) - p_simulation(i)))

Top-k overlap may also be computed:

top_k_overlap = size(top_k_hardware intersect top_k_simulation) / k

Recommended:

k = 4

## Rank stability

Top-outcome rankings must be treated as unstable when counts are too close.

For two counts a and b:

sigma_diff = sqrt(a + b)

If:

abs(a - b) < 2 * sqrt(a + b)

then the rank is unstable and must not be treated as a clean top-k fact.

## JSON-counts-first rule

The JSON result records are the data source.

PNG files are artifacts.

PNG files may be listed, viewed, or linked, but they must not drive any statistic.

## Bit-order law

Before any candidate prediction is frozen, bit order must be declared.

The report must state whether bitstrings are read as raw classical-register order, reversed display order, Qiskit display convention, or custom decoded order.

The report must explain how the bit-order convention was proven from QASM plus result metadata.

If bit-order mapping cannot be proven, no exact bitstring prediction may be promoted into future preregistration.

## Researcher Degrees of Freedom

Every LL-06 exploration report must include a section titled:

Researcher Degrees of Freedom

That section must list every metric computed.

It must also state that candidate predictions were selected after seeing training-side data and therefore require fresh hardware runs.

## Forbidden analysis and language

An LL-06 exploration report may not include p-values, verdict language, claim upgrades, outcome-log edits, theory-file edits, frozen-file edits, quantum validation language, consciousness proof language, AGI proof language, physics proof language, restoration of April 2026 analysis as evidence, use of the ten IBM jobs as confirmation data, or claims that formulas need revision based on these exploratory jobs.

## Candidate predictions

Every LL-06 exploration report must end with:

Candidate Predictions for Future Fresh LL-06 Preregistration

It must include one to three candidate predictions.

Each candidate prediction must include training-side source, job IDs inspected, circuit hash, backend or backends, reason the candidate was generated, fresh-test requirement, exact bitstrings, exact statistic, exact margin, required controls, and failure condition.

## Future registered LL-06 requirements

A later registered LL-06 test must use fresh hardware runs.

It must define before execution exact circuit files, exact control bank, exact predicted bitstrings, exact statistic, exact margin, shot count, backend plan, execution-order policy, simulation method, bit-order convention, and failure condition.

## Control-bank note

A future registered LL-06 design should prefer a full control bank.

Minimum viable bank:

- flat-angle control
- angle-placement control
- three seeded random-angle controls

Preferred full bank:

- flat-angle control
- rational-angle control
- angle-placement control
- five seeded random-angle controls

The final count must be fixed at preregistration time with a hardware-cost note.

## Standing boundary

Exploration may generate sharper hypotheses.

Only a future frozen preregistration using fresh hardware runs may test them.
