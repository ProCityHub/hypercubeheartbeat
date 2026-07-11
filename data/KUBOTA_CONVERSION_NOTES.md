# Phase 4 Step 2b-1C — Kubota MAT Conversion Notes

Source URL: https://osf.io/ugdcz

This PR performs conversion only.

No scoring was performed.

No preregistration_test.py execution was performed.

No empirical outcome label was produced.

## Raw input

- Raw archive: `UltimatumRawData.zip`
- Raw archive sha256: `268d42416e249e83e1303f5dadab63bc26dd9076e003ce1e7f69e139d8a74207`

## Loader used

- `scipy.io.loadmat`: 49 file(s)

## Converter invocation

```bash
/data/data/com.termux/files/usr/bin/python data/convert_kubota_mat.py --raw-zip /data/data/com.termux/files/home/kubota_conversion/raw/UltimatumRawData.zip --output data/ug_probe.csv --notes data/KUBOTA_CONVERSION_NOTES.md
```

## Frozen mapping applied

- `offer = pTOffer[:, 1]`
- `stake = 10`
- `rt_ms = utDecision[:, 0] - pTOffer[:, 0]`
- `accept = utDecision[:, 1]`, normalized as `1 -> 1`, `2 -> 0`

## Timing unit decision

- Median raw latency before unit conversion: `0.9376724759993635`
- Unit decision: `seconds_to_milliseconds`
- Multiplier applied to raw latency: `1000.0`

The unit decision used timing magnitude only.

No offer-vs-RT, accept-vs-RT, or offer-vs-accept statistic was computed.

## Row counts

- Total raw trials: `7840`
- Excluded no-response trials: `0`
- Excluded bad-parse rows: `20`
- Excluded nonpositive-latency rows: `0`
- Excluded unexpected-decision rows: `0`
- Excluded rt_ms below 200.0: `3`
- Final output rows: `7817`

## Unexpected decision codes

- None observed.

## Per-file trial counts

| File | Total | Included | No response | Bad parse | Nonpositive latency | Unexpected decision | RT floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| UltimatumData/012701.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/013001.mat | 160 | 158 | 0 | 2 | 0 | 0 | 0 |
| UltimatumData/020101.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/020801.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/020803.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/020902.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/021501.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/021701.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/022101.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/022301.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/022401.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/022402.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/022801.mat | 160 | 157 | 0 | 3 | 0 | 0 | 0 |
| UltimatumData/072201.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/072801.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/072802.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/072901.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/080101.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/080102.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/080301.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/080401.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/081001.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/081102.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/081201.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/081501.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/081601.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/081701.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/081801.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/081901.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/081902.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/081903.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/082201.mat | 160 | 159 | 0 | 0 | 0 | 0 | 1 |
| UltimatumData/082401.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/082501.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/082502.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/083101.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/090101.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/090201.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/090601.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/090701.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/091501.mat | 160 | 158 | 0 | 1 | 0 | 0 | 1 |
| UltimatumData/091901.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |
| UltimatumData/091902.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/092001.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/092601.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/101301.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/101701.mat | 160 | 160 | 0 | 0 | 0 | 0 | 0 |
| UltimatumData/101801.mat | 160 | 156 | 0 | 3 | 0 | 0 | 1 |
| UltimatumData/102001.mat | 160 | 159 | 0 | 1 | 0 | 0 | 0 |

## Output

- Output file: `data/ug_probe.csv`
- Output sha256: `96cfd31b7bff9ea765fc24825b59787d90fea999c28e6a9524614bd1f95ea92d`

## Claim discipline

This conversion is not an empirical result.

This conversion does not score the Lattice Law.

This conversion does not prove consciousness.

This conversion does not prove AGI.

The only purpose of this file is to document the reproducible data conversion before the single preregistered run.
