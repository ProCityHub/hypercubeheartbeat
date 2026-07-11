# Decisions

This folder stores human-approved next-step declarations.

A decision file should state:

- what is being approved
- what is forbidden
- what files may be touched
- what commands may be run
- whether scoring is allowed
- whether a preregistered test may run
- when the agent must stop

Rules:

- No agent may treat an idea as approved unless it has a decision record or explicit human instruction.
- Decision records should be created before risky work begins.
- Empirical runs require extra caution and explicit one-run language.
