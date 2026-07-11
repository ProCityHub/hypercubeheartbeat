# Manifests

This folder stores provenance records.

A manifest should record:

- filename
- source URL or source description
- date received
- sha256 hash
- file size
- intended use
- related decision document
- whether the file is raw, converted, derived, or report-only

Rules:

- Every external file used in a test should have a manifest.
- Hashes must be recorded before analysis whenever possible.
- Manifests are records, not results.
