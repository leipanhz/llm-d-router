# KV-Cache File Prefetch Plugin (Experimental)

This package implements an experimental pre-request plugin that proactively prefetches
KV-cache blocks across different storage tiers before inference requests are processed
by the GPU pod. The goal is to promote KV-cache blocks to a closer storage tier ahead
of time to reduce inference latency.

## Overview

The plugin implements the `PreRequest` interface and is invoked after a routing decision
is made but before the request is dispatched to the GPU pod. It determines the storage
locations (file paths) of KV-cache blocks that will be needed for the request and
arranges for them to be promoted to a closer storage tier.

The current implementation targets a **shared file system with transparent access to a
remote storage tier**, such as IBM Storage Scale configured to offload cold data to
remote object storage. A simple sequential read of the beginning of a KV-cache file
is sufficient to trigger the underlying storage system to promote (prefetch) the full
file from remote storage to the local file system tier.

In a future version, this could be extended to prefetch KV-cache blocks from the file
system to CPU memory on the worker node that the request is being routed to.

## How It Works

```
                                  ┌──────────────────────────┐
                                  │ Inference Request         │
                                  └────────────┬──────────────┘
                                               │
                                               ▼
                          ┌──────────────────────────────────────┐
                          │ PreRequest (this plugin)             │
                          ├──────────────────────────────────────┤
                          │ 1. GetEngineKeysForRequest()         │
                          │     ← precise-prefix-cache scorer    │
                          │                                      │
                          │ 2. discoveryCache.discover()         │
                          │     ← lazy on first request          │
                          │     ← globs <model>_*_r0/ once       │
                          │     → caches (base, group_idx)       │
                          │                                      │
                          │ 3. Build paths for ranks [0..N)      │
                          │     <base>_r<rank>/hhh/hh_g<G>/      │
                          │       <hash>.bin                     │
                          │                                      │
                          │ 4. Submit each path to work queue    │
                          └────────────┬─────────────────────────┘
                                       │
                                       ▼
              ┌─────────── Worker Pool (M concurrent goroutines) ──────────┐
              │                                                            │
              │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ...           │
              │   │ worker 0 │  │ worker 1 │  │ worker 2 │                │
              │   └─────┬────┘  └─────┬────┘  └─────┬────┘                │
              │         │             │             │                     │
              │     read first BlockSize×BlockCount bytes from each file  │
              │         │             │             │                     │
              │     on os.IsNotExist → cache.invalidate()                 │
              │     (next request re-discovers <base>)                    │
              │         │             │             │                     │
              └─────────┼─────────────┼─────────────┼─────────────────────┘
                        ▼             ▼             ▼
              ┌──────────────────────────────────────────────────┐
              │ Shared File System Mount (e.g. IBM Storage Scale) │
              │                                                   │
              │   <rootDir>/<safeModel>_<12hex>_r<rank>/          │
              │     <hhh>/<hh>_g<group_idx>/<hash>.bin            │
              │                                                   │
              │   reading the head of each file triggers the      │
              │   storage system to pull the full file from       │
              │   cold object storage to the local SSD tier       │
              └──────────────────────────────────────────────────┘
```

1. The plugin calls `GetEngineKeysForRequest()` on the configured
   `precise-prefix-cache` scorer to obtain the engine keys (block hashes) for the
   incoming request. Each key is a 64-bit content hash of one block's tokens —
   identical across all ranks for the same logical block.
2. **Path discovery** runs lazily on the first request. The plugin globs
   `<rootDir>/<safeModelName>_*_r0` to anchor the shared `<base>` prefix written
   by vLLM, reads one shard subfolder to learn the `_g<group_idx>` segment, and
   caches the result in an in-memory `discoveryCache` for the process lifetime.
   Subsequent requests reuse the cached `(base, group_idx)` with no filesystem
   inspection.
3. For each engine key, the plugin builds one path per rank in the deployment
   (`Tp × Pp × Pcp × Dcp` ranks) using the discovered base. Each rank's path
   points at a different on-disk file holding that rank's KV shard for the same
   logical block.
4. Each file path is submitted to a worker pool. A worker reads a configurable
   number of bytes (`BlockSize × BlockCount`) from the head of the file, which
   triggers the storage system's transparent prefetch from the cold tier to the
   local tier.
5. **Cache invalidation:** if a worker's `os.Open` returns `os.ErrNotExist` (vLLM
   has restarted with new hash inputs and the digest folder has changed), the
   worker invalidates the discovery cache. The current in-flight prefetch for
   that file is lost, but the next `PreRequest` re-discovers the new `<base>` and
   subsequent prefetches succeed.

## On-disk layout

The plugin assumes the layout written by vLLM's `llmd_fs_backend.FileMapper`:

```
<rootDir>/<safeModelName>_<12hex>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash>.bin
```

Where:
- `safeModelName` = `model_name` with `/` replaced by `_` (HuggingFace IDs).
- `<12hex>` = the first 12 chars of a SHA-256 over a JSON-canonicalized dict of
  vLLM-internal fields (`kv_cache_groups`, `dtype`, `block_size`, parallel
  sizes, etc.). The router does not reproduce this hash; it discovers the
  folder by globbing.
- `<rank>` = `parallel_config.rank` for the worker that wrote the file.
- `<hhh>/<hh>` = first 5 hex chars of the block hash, sharded into two
  subdirectories.
- `<group_idx>` = KV cache group index. Currently single-group only — discovery
  picks one and reuses it.
- `<hash>` = full 16-hex block hash. **The same filename appears under every
  `_r<rank>` folder, but each rank's file holds a different byte content
  (that rank's local KV shard).**

## Architecture

The plugin uses a **concurrent worker thread pool** to prefetch multiple files in
parallel. Workers are long-lived goroutines that read from a shared work queue. A
configurable queue timeout prevents slow queues from blocking the request path.

## Configuration

The plugin is registered as `prefetch-prerequest-handler` and configured via JSON
parameters in the EPP config.

### Parameters

| Field | Type | Description |
|---|---|---|
| `engineKeysProviderPluginName` | string | Name of the `precise-prefix-cache` scorer plugin instance to use for engine key retrieval |
| `kvFilePathBase` | object | KV-cache file path parameters (see below) |
| `prefetchConfig` | object | Prefetch worker pool configuration (see below) |

### `kvFilePathBase`

| Field | Type | Default | Description |
|---|---|---|---|
| `rootDir` | string | — | Root directory of the KV-cache file system mount |
| `modelName` | string | — | Model name (e.g. `meta-llama/Llama-3.1-8B-Instruct`); `/` is replaced by `_` for the on-disk folder |
| `gpuBlocksPerFile` | int | `1` | Number of GPU blocks vLLM stores per file; the plugin emits one path per file (every Nth engine key) |
| `tpSize` | int | `1` | Tensor-parallel size |
| `ppSize` | int | `1` | Pipeline-parallel size |
| `pcpSize` | int | `1` | Prefill-context-parallel size |
| `dcpSize` | int | `1` | Decode-context-parallel size |

**Removed in this version (no longer needed):** `gpuBlockSize`, `dtype`,
`modelParentDir`, `rank`. Those fields were used by the previous
hash-compute path-derivation approach; under glob-based discovery the
plugin doesn't need them. `rank` is iterated `[0, Tp×Pp×Pcp×Dcp)` at
runtime from the operator-supplied parallel sizes.

### `prefetchConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable or disable the prefetch worker pool |
| `blockSize` | int64 | `4194304` (4 MiB) | Bytes to read per block from each file |
| `blockCount` | int | `3` | Number of blocks to read per file |
| `maxConcurrentFiles` | int | `16` | Number of parallel worker goroutines |
| `workQueueSize` | int | `256` | Capacity of the work queue channel |
| `queueTimeout` | int | `0` | Milliseconds to wait before skipping a file when the queue is full (0 = block indefinitely) |

### Example EPP Config Snippet

```yaml
plugins:
  - name: precise-prefix-cache-scorer
    type: precise-prefix-cache-scorer
    parameters:
      tokenProcessorConfig:
        hashAlgorithm: sha256-cbor
        hashSeed: "10"
        blockSizeTokens: 16

  - name: kv-prefetch
    type: prefetch-prerequest-handler
    parameters:
      engineKeysProviderPluginName: precise-prefix-cache-scorer
      kvFilePathBase:
        rootDir: /mnt/kv-cache
        modelName: meta-llama/Llama-3.1-8B-Instruct
        gpuBlocksPerFile: 1
        tpSize: 1
        ppSize: 1
        pcpSize: 1
        dcpSize: 1
      prefetchConfig:
        enabled: true
        blockSize: 4194304
        blockCount: 3
        maxConcurrentFiles: 16
        workQueueSize: 256
        queueTimeout: 100
```

## Hash Algorithm Requirement

For engine keys to match the KV-cache files written by vLLM, the `precise-prefix-cache`
scorer must be configured to use the **SHA256-CBOR** hash algorithm (`hashAlgorithm: sha256-cbor`),
which matches vLLM's engine key computation. The default FNV64a algorithm will produce
different keys and cause a mismatch.

This plugin only works with the **llm-d-fs-connector** file naming convention. File
path generation is not compatible with other KV-cache storage backends.

## Behavior notes & limitations

- **No-op until vLLM writes its first block.** Discovery requires at least one
  `<base>_r0/<hhh>/<hh>_g<N>/` directory to exist. Until then, every
  `PreRequest` returns no paths and a V(1) "discovery deferred" log line; the
  plugin retries on the next request. There is no plugin-init dependency on
  vLLM readiness.
- **Single-group only.** The plugin picks one `_g<N>` directory at discovery
  and emits paths only for that group. Models with multiple KV cache groups
  (sliding-window + full-attention, mamba hybrids) will silently prefetch
  only one group. Multi-group support is a follow-up.
- **Single vLLM deployment per plugin instance.** The discovery cache holds
  one `(base, group_idx)` pair. Routing to multiple vLLM deployments with
  different parallelism or model configs from a single plugin instance is
  not supported. Multi-pod support is a follow-up.
- **In-flight prefetch loss on cache invalidation.** When a worker hits
  `os.ErrNotExist`, the in-flight file open is lost; only the next request's
  prefetches benefit from the re-discovered base.
- **Ambiguous matches fail loudly.** If the glob finds multiple
  `<safeModelName>_*_r0` candidates (e.g. two deployments sharing the same
  `rootDir`), discovery returns an error listing all candidates.

## Testing

```bash
go test ./pkg/epp/framework/plugins/scheduling/scorer/prefetch/ -count=1
```

The path test file (`prefetch_prerequest_experimental_path_test.go`) builds
synthetic on-disk vLLM layouts under `t.TempDir()` and exercises discovery,
invalidation, ambiguity handling, and path formatting end-to-end.
