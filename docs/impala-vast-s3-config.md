# Wiring the Impala Virtual Warehouse to VAST S3 (`mschuler-cloudera`)

> **Superseded for the app's own data (2026-07-08):** the demo no longer needs
> any of this to put its training splits on VAST — the Data dialog's **VAST
> S3** backend writes Parquet objects to the bucket directly with boto3
> (`tfm_demo/vast.py`), sidestepping the unresolved s3a `400` documented at the
> bottom of this file. Keep this doc for the day Impala tables themselves must
> live on VAST.

Goal: let the TFM demo's Impala tables live physically in the VAST S3 store at
`https://s3.previewhub.dev`, bucket `mschuler-cloudera`. Everything below is done
in the **Cloudera Data Warehouse (CDW) web UI** — no base-cluster changes.

## Why each piece is needed

An s3a URI (`s3a://mschuler-cloudera/...`) names only the **bucket and path**.
The endpoint, addressing style, and credentials come from `fs.s3a.*`
configuration properties that each service reads at startup. Two independent
services touch the bucket, so the properties go in two places:

| Service | Lives in | What it does with the bucket |
|---|---|---|
| Hive Metastore (HMS) | **Database Catalog** | Validates table/database `LOCATION` on `CREATE` (this is what threw `NoSuchBucketException`) |
| Impala (coordinator / executor / catalogd) | **Virtual Warehouse** | Reads and writes the actual Parquet data; catalogd drives DDL through HMS |

## The four properties (identical everywhere)

All four are scoped to the bucket name — they affect nothing else on the
warehouse:

```
fs.s3a.bucket.mschuler-cloudera.endpoint=https://s3.previewhub.dev
fs.s3a.bucket.mschuler-cloudera.path.style.access=true
fs.s3a.bucket.mschuler-cloudera.access.key=<ACCESS KEY>
fs.s3a.bucket.mschuler-cloudera.secret.key=<SECRET KEY>
```

Notes:
- `path.style.access=true` is required for VAST (same as boto3's
  `addressing_style: "path"`).
- The property prefix is `fs.s3a.bucket.<bucket-name>.` — the bucket name must
  appear in the key, spelled exactly (`mschuler-cloudera`, hyphen not underscore).
- Never put the keys in SQL (`SET ...`) — Impala's `SET` only handles query
  options, and statements land in query logs. Config properties do not.

## Where to enter them

### 1. Virtual Warehouse (type: Impala)

CDW UI → Virtual Warehouses → your VW → **⋮ → Edit → CONFIGURATIONS** tab.
Add all four properties (as key/value pairs) to the `hadoop-core-site` (or
`core-site`) configuration file of **each** of these components:

| Component (dropdown) | Configuration file | Why |
|---|---|---|
| Impala coordinator | `hadoop-core-site` | Plans queries, brokers small results, runs `REFRESH` |
| Impala executor | `hadoop-core-site` | Reads/writes the Parquet files (the data path) |
| Impala catalogd | `hadoop-core-site` | Executes DDL, talks to HMS, loads file metadata |

Do **not** put them in `flagfile` — that file is for impalad startup flags,
not Hadoop filesystem properties.

Click **Apply** — the control plane restarts the VW (a few minutes; running
queries are interrupted).

### 2. Database Catalog (the metastore)

CDW UI → Database Catalogs → the catalog your VW is attached to →
**⋮ → Edit → CONFIGURATIONS** tab:

| Component (dropdown) | Configuration file | Why |
|---|---|---|
| Metastore | `hive-site` (use `hadoop-core-site` if offered) | HMS does the `getFileStatus` location check on every `CREATE DATABASE/TABLE ... LOCATION` |

Same four properties. **Apply** restarts the metastore — this briefly affects
every VW attached to the catalog, but the bucket-scoped properties themselves
are invisible to anything not using `mschuler-cloudera`.

## Validate (Hue, against the Impala VW)

URI anatomy for this environment — the endpoint hostname must NOT appear in it:

```
s3a://  mschuler-cloudera  /mschuler-bucket/fsi_demo
scheme  └─ bucket ─┘       └─ folder path inside the bucket ─┘
```

```sql
CREATE DATABASE fsi_demo LOCATION 's3a://mschuler-cloudera/mschuler-bucket/fsi_demo';
CREATE EXTERNAL TABLE fsi_demo.loc_probe (i INT) STORED AS PARQUET;
INSERT INTO fsi_demo.loc_probe VALUES (1);
SELECT * FROM fsi_demo.loc_probe;
DROP TABLE fsi_demo.loc_probe;
```

Then run the boto3 connectivity script and list `mschuler-cloudera` — a
`mschuler-bucket/fsi_demo/loc_probe/` prefix (before the DROP) proves bytes
physically landed on VAST.

## Hook the app up

1. Have a **CML data connection** created for the new VW (CML workspace →
   Site Administration → Data Connections, or ask the workspace admin) —
   connection type **Impala**, pointing at the VW's coordinator endpoint.
2. In the demo UI, open **Data**, enter that connection name and `fsi_demo`,
   **Save & test**, then **Load TabFormer → Impala**.

## Known failure modes

| Symptom | Meaning | Action |
|---|---|---|
| `NoSuchBucketException ... s3a://s3.previewhub.dev/...` | Endpoint hostname crept back into a `LOCATION` URI | Fix the URI: `s3a://mschuler-cloudera/...` |
| `403 / AccessDenied` on the probe | Wrong keys, or key properties missing on one of the components | Re-check all four properties in **both** the VW components and the DBC |
| `SSL/PKIX handshake` errors | `s3.previewhub.dev` serves an internal-CA certificate the warehouse JVM doesn't trust (the boto3 test skipped verification) | Truststore changes are outside VW config — platform team item; use the managed-table fallback meanwhile |
| `UnknownHostException s3.previewhub.dev` | The warehouse pods can't resolve/reach the VAST endpoint on the network | Unlikely here — the boto3 test succeeded from inside a CML session (2026-07-07), so the endpoint is reachable from the platform's pod network; if it still appears, it's a CDW-namespace networking item |

## Fallback (no config at all)

The app does not require any of the above to function: `CREATE DATABASE
fsi_demo;` with no `LOCATION`, point the Data dialog at it, and the loader
falls back to managed tables in the warehouse's default storage automatically
(it logs "external table rejected → falling back to a managed table"). This
document is only about making the data land on VAST.
