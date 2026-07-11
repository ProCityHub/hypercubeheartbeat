# App Infrastructure

## Status

This is the app-facing infrastructure skeleton for GARVIS.

It is not a working app.

It is not an autonomous agent.

It does not add external actions.

It does not call an LLM.

It does not activate sensors.

It does not commit runtime sensor data.

## Purpose

App Infrastructure defines the future interface layer between Adrien D Thomas, the local Termux body, the Stage 1 senses ledger, and future GARVIS app surfaces.

It exists so future app work has boundaries before capability.

The app layer may later provide:

- mobile-first interface
- local status display
- ledger inspection
- approval prompts
- notification surface
- sensor status surface
- human-readable GARVIS reports
- bridge to Termux scripts
- bridge to SQLite ledger
- future bridge to Lattice Thinker
- future bridge to approved LLM calls

## Current rule

This skeleton may define folders, boundaries, and contracts.

It may not create autonomous action.

It may not create a production app.

It may not start a background service.

It may not send messages.

It may not contact people.

It may not commit secrets.

It may not commit camera, microphone, notification, or other runtime outputs.

## Architecture position

The current organism is:

- GitHub repository as long-term memory
- Termux as local execution body
- Stage 1 senses loop as local sensory scaffold
- SQLite decision ledger as local append-only memory
- GARVIS constitution as behavior law
- app infrastructure as future human-facing interface layer

## Human approval gate

The app layer must preserve human approval.

A future app may show recommended actions.

A future app may request approval.

A future app may display ledger state.

A future app may prepare an action directive.

A future app may not silently execute sensitive actions.

## Standing boundary

The app may make GARVIS easier to see.

The app may make GARVIS easier to approve.

The app may make GARVIS easier to audit.

The app may not make GARVIS harder to control.
