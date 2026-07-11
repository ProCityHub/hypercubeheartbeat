# Claims Ledger

The claims ledger maps public claims to evidence.

Every public claim should have:

- claim_id
- plain_language_claim
- scope
- status
- evidence_file
- evidence_commit
- allowed_public_language
- forbidden_public_language
- next_required_step

The ledger exists so the system can separate:

- symbolic generation
- pilot outcomes
- confirmatory outcomes
- replications
- retractions
- infrastructure claims
- theory interpretations

A future PR will extend the guard to check new documents against this ledger automatically.

## Guard scoping

This PR scopes reserved empirical-verdict language checks to claim-controlled files, including the claims ledger and claim/protocol documents.

Charters, maps, and exploratory notes may discuss vocabulary without being treated as empirical outcome records.

## Rule

If a claim has no ledger entry, it should not be promoted as public evidence.

If a claim is retracted, the forbidden language must remain visible so the system knows what may never be said again.
