package prefetch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/preciseprefixcache"
)

const (
	// PrefetchPrerequestHandlerType is the type of the PrefetchPrerequestHandler.
	PrefetchPrerequestHandlerType = "prefetch-prerequest-handler"
)

// engineKeyToFilenamePathSuffix converts an engine-key (uint64 block hash) to
// the path suffix used by llm-d-fs-connector: hhh/hh_g<groupIdx>/<16hex>.bin.
func engineKeyToFilenamePathSuffix(engineKey uint64, groupIdx int) string {
	blockHashHex := fmt.Sprintf("%016x", engineKey)
	sub1, sub2 := blockHashHex[:3], blockHashHex[3:5]
	return fmt.Sprintf("%s/%s_g%d/%s.bin", sub1, sub2, groupIdx, blockHashHex)
}

// KVFilePathBaseParams holds operator-supplied parameters for KV-cache file
// paths. All fields are deserialized from the plugin's JSON config; this
// struct does not hold runtime state. Discovery state lives on a separate
// in-memory discoveryCache owned by the handler.
//
// On-disk layout written by vLLM (one identical filename per rank, with
// different byte contents — each rank stores only its own KV shard):
//
//	<rootDir>/<safeModelName>_<12hex>_r<rank>/<hhh>/<hh>_g<groupIdx>/<hash>.bin
//
// The router cannot recompute the <12hex> digest because vLLM's hash inputs
// (kv_cache_groups, layer_names, dtype, etc.) are not reliably reproducible.
// Instead, discovery globs "<safeModelName>_*_r0" to anchor the shared
// "<base>" prefix; per-rank paths are built by appending "_r<rank>" for each
// rank in [0, Tp*Pp*Pcp*Dcp). _r0 is used purely as the glob anchor — it is
// the only rank guaranteed to exist whenever any worker has flushed — and
// the resulting <base> applies to all ranks because every worker shares the
// same base_path (file_mapper.py:117).
type KVFilePathBaseParams struct {
	RootDir          string `json:"rootDir"`
	ModelName        string `json:"modelName"`
	GpuBlocksPerFile int    `json:"gpuBlocksPerFile"`
	TpSize           int    `json:"tpSize"`
	PpSize           int    `json:"ppSize"`
	PcpSize          int    `json:"pcpSize"`
	DcpSize          int    `json:"dcpSize"`
}

// discoveryCache memoizes the (base, groupIdx) pair resolved from globbing
// the filesystem once per process. Populated lazily on the first PreRequest
// and reset by invalidate() when a prefetch open hits os.ErrNotExist (vLLM
// restarted with new hash inputs).
type discoveryCache struct {
	mu    sync.RWMutex
	base  string // <rootDir>/<safeModelName>_<12hex>
	group int    // typically 0
	done  bool
}

// IsSet returns true if base path can be built.
func (b *KVFilePathBaseParams) IsSet() bool {
	return b != nil && b.RootDir != "" && b.ModelName != ""
}

// SetDefaults applies default values to unset KVFilePathBaseParams fields.
func (b *KVFilePathBaseParams) SetDefaults() {
	if b.GpuBlocksPerFile < 1 {
		b.GpuBlocksPerFile = 1
	}
	if b.TpSize < 1 {
		b.TpSize = 1
	}
	if b.PpSize < 1 {
		b.PpSize = 1
	}
	if b.PcpSize < 1 {
		b.PcpSize = 1
	}
	if b.DcpSize < 1 {
		b.DcpSize = 1
	}
}

// discover resolves (base, group) by inspecting the filesystem. Idempotent
// and safe to call concurrently — only the first caller actually scans
// (RWMutex with double-check). On failure, returns an error and leaves the
// cache unset; the caller skips this round and the next PreRequest will
// retry.
func (c *discoveryCache) discover(ctx context.Context, params *KVFilePathBaseParams) error {
	c.mu.RLock()
	if c.done {
		c.mu.RUnlock()
		return nil
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.done {
		return nil
	}

	safeModelName := strings.ReplaceAll(params.ModelName, "/", "_")

	// Glob "_r0" directories — rank 0 is guaranteed to exist whenever any
	// worker has flushed, and every rank shares the same <base> prefix.
	pattern := filepath.Join(params.RootDir, safeModelName+"_*_r0")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("discover: glob %q: %w", pattern, err)
	}
	if len(matches) == 0 {
		return fmt.Errorf("discover: no folder matches %q (has vLLM initialized and written any blocks?)", pattern)
	}
	if len(matches) > 1 {
		return fmt.Errorf("discover: ambiguous — %d folders match %q: %v", len(matches), pattern, matches)
	}
	rank0 := matches[0]

	// Find any "<hhh>/<hh>_g<N>" subfolder to learn the group index. We
	// only need one sample — group_idx is the same for every block in a
	// given vLLM deployment (single-group simplification, A3a).
	groupIdx := -1
	shards, err := os.ReadDir(rank0)
	if err != nil {
		return fmt.Errorf("discover: read %q: %w", rank0, err)
	}
	for _, shard := range shards {
		if !shard.IsDir() {
			continue
		}
		shardPath := filepath.Join(rank0, shard.Name())
		groups, err := os.ReadDir(shardPath)
		if err != nil {
			continue
		}
		for _, g := range groups {
			if !g.IsDir() {
				continue
			}
			if idx := parseGroupSuffix(g.Name()); idx >= 0 {
				groupIdx = idx
				break
			}
		}
		if groupIdx >= 0 {
			break
		}
	}
	if groupIdx < 0 {
		return fmt.Errorf("discover: no \"<hh>_g<N>\" group subfolder found under %q (no blocks written yet?)", rank0)
	}

	c.base = strings.TrimSuffix(rank0, "_r0")
	c.group = groupIdx
	c.done = true

	log.FromContext(ctx).Info("prefetch: resolved KV cache base path",
		"rootDir", params.RootDir, "modelName", params.ModelName,
		"basePath", c.base, "groupIdx", c.group)
	return nil
}

// parseGroupSuffix extracts N from "<prefix>_g<N>". Returns -1 on mismatch.
func parseGroupSuffix(name string) int {
	i := strings.LastIndex(name, "_g")
	if i < 0 {
		return -1
	}
	n, err := strconv.Atoi(name[i+2:])
	if err != nil || n < 0 {
		return -1
	}
	return n
}

// invalidate clears the discovery cache. Called when prefetch observes a
// path no longer exists — vLLM may have restarted with new hash inputs,
// shifting <base> to a new digest. Next PreRequest re-discovers.
func (c *discoveryCache) invalidate() {
	c.mu.Lock()
	c.done = false
	c.base = ""
	c.group = 0
	c.mu.Unlock()
}

// engineKeyToFullPath returns the complete file path for an engine key on
// the given rank. The first call performs lazy discovery; on discovery
// failure it logs at V(1) and returns ("", false) so the caller can skip
// prefetch for this round.
func (c *discoveryCache) engineKeyToFullPath(ctx context.Context, params *KVFilePathBaseParams, rank int, engineKey uint64) (string, bool) {
	if err := c.discover(ctx, params); err != nil {
		log.FromContext(ctx).V(1).Info("prefetch: discovery deferred", "reason", err.Error())
		return "", false
	}
	c.mu.RLock()
	base := c.base
	group := c.group
	c.mu.RUnlock()

	suffix := engineKeyToFilenamePathSuffix(engineKey, group)
	return fmt.Sprintf("%s_r%d/%s", base, rank, filepath.FromSlash(suffix)), true
}

// engineKeysToFilePaths returns the file paths to prefetch for the given
// engine keys on the given rank. Returns nil if discovery hasn't succeeded
// yet — the caller should skip this round entirely rather than emit partial
// paths.
func engineKeysToFilePaths(ctx context.Context, cache *discoveryCache, params *KVFilePathBaseParams, rank int, engineKeys []uint64) []string {
	n := params.GpuBlocksPerFile
	if n <= 1 {
		paths := make([]string, 0, len(engineKeys))
		for _, ek := range engineKeys {
			p, ok := cache.engineKeyToFullPath(ctx, params, rank, ek)
			if !ok {
				return nil
			}
			paths = append(paths, p)
		}
		return paths
	}

	paths := make([]string, 0, (len(engineKeys)+n-1)/n)
	for i := n - 1; i < len(engineKeys); i += n {
		p, ok := cache.engineKeyToFullPath(ctx, params, rank, engineKeys[i])
		if !ok {
			return nil
		}
		paths = append(paths, p)
	}
	return paths
}

func prefetchFile(ctx context.Context, filePath string, buffer []byte, cache *discoveryCache) error {
	file, err := os.Open(filePath)
	if err != nil {
		// vLLM may have restarted with new hash inputs, shifting <base>
		// to a new digest. Drop the cache so the next request re-discovers.
		if cache != nil && errors.Is(err, os.ErrNotExist) {
			cache.invalidate()
		}
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	n, err := file.Read(buffer)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read file: %w", err)
	}

	if err == io.EOF || n < len(buffer) {
		log.FromContext(ctx).V(4).Info("prefetchFile: read partial file",
			"path", filePath, "requestedBytes", len(buffer), "actualBytes", n)
	} else {
		log.FromContext(ctx).V(4).Info("prefetchFile: read complete",
			"path", filePath, "bytes", n)
	}

	return nil
}

func initializeWorkerPool(ctx context.Context, handler *PrefetchPrerequestHandler) error {
	if handler.prefetchConfig == nil || !handler.prefetchConfig.Enabled {
		log.FromContext(ctx).Info("initializeWorkerPool: prefetching disabled")
		return nil
	}

	config := handler.prefetchConfig
	log.FromContext(ctx).Info("initializeWorkerPool: initializing worker pool",
		"maxConcurrentFiles", config.MaxConcurrentFiles,
		"workQueueSize", config.WorkQueueSize,
		"blockSize", config.BlockSize,
		"blockCount", config.BlockCount)

	pool := &PrefetchWorkerPool{
		workQueue:   make(chan string, config.WorkQueueSize),
		workersDone: make(chan struct{}),
	}
	pool.shutdownCtx, pool.shutdownFn = context.WithCancel(context.Background())
	handler.workerPool = pool

	bufferSize := int(config.BlockSize * int64(config.BlockCount))

	var wg sync.WaitGroup
	for i := 0; i < config.MaxConcurrentFiles; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			buffer := make([]byte, bufferSize)
			log.FromContext(ctx).V(2).Info("initializeWorkerPool: worker started",
				"workerID", workerID, "bufferSize", bufferSize)

			for {
				select {
				case filePath, ok := <-pool.workQueue:
					if !ok {
						log.FromContext(ctx).V(2).Info("initializeWorkerPool: worker exiting (channel closed)",
							"workerID", workerID)
						return
					}

					startTime := time.Now()
					err := prefetchFile(pool.shutdownCtx, filePath, buffer, handler.cache)
					duration := time.Since(startTime)

					if err != nil {
						log.FromContext(ctx).Error(err, "initializeWorkerPool: worker failed to prefetch file",
							"workerID", workerID, "path", filePath, "duration", duration)
					} else {
						log.FromContext(ctx).V(1).Info("initializeWorkerPool: worker successfully prefetched file",
							"workerID", workerID, "path", filePath, "duration", duration)
					}

				case <-pool.shutdownCtx.Done():
					log.FromContext(ctx).V(2).Info("initializeWorkerPool: worker exiting (shutdown signal)",
						"workerID", workerID)
					return
				}
			}
		}(i)
	}

	go func() {
		wg.Wait()
		close(pool.workersDone)
		log.FromContext(ctx).Info("initializeWorkerPool: all workers exited")
	}()

	log.FromContext(ctx).Info("initializeWorkerPool: worker pool initialized",
		"workerCount", config.MaxConcurrentFiles)
	return nil
}

// PrefetchConfig holds configuration for file prefetching behavior.
type PrefetchConfig struct {
	Enabled            bool  `json:"enabled"`
	BlockSize          int64 `json:"blockSize,omitempty"`
	BlockCount         int   `json:"blockCount,omitempty"`
	MaxConcurrentFiles int   `json:"maxConcurrentFiles,omitempty"`
	WorkQueueSize      int   `json:"workQueueSize,omitempty"`
	QueueTimeout       int   `json:"queueTimeout,omitempty"`
}

// SetDefaultsForFilePrefetching applies default values to unset prefetch configuration fields.
func (p *PrefetchConfig) SetDefaultsForFilePrefetching() {
	if p.BlockSize == 0 {
		p.BlockSize = 4 * 1024 * 1024
	}
	if p.BlockCount == 0 {
		p.BlockCount = 3
	}
	if p.MaxConcurrentFiles == 0 {
		p.MaxConcurrentFiles = 16
	}
	if p.WorkQueueSize == 0 {
		p.WorkQueueSize = 256
	}
}

// PrefetchWorkerPool holds the runtime state for the prefetch worker pool.
type PrefetchWorkerPool struct {
	workQueue    chan string
	workersDone  chan struct{}
	shutdownCtx  context.Context
	shutdownFn   context.CancelFunc
	shutdownOnce sync.Once
}

type prefetchPrerequestHandlerParameters struct {
	EngineKeysProviderPluginName string                `json:"engineKeysProviderPluginName"`
	KVFilePathBase               *KVFilePathBaseParams `json:"kvFilePathBase,omitempty"`
	PrefetchConfig               *PrefetchConfig       `json:"prefetchConfig,omitempty"`
}

var _ requestcontrol.PreRequest = &PrefetchPrerequestHandler{}

// PluginFactory defines the factory function for the PrefetchPrerequestHandler.
func PluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := prefetchPrerequestHandlerParameters{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' pre-request plugin - %w", PrefetchPrerequestHandlerType, err)
		}
	}

	if parameters.PrefetchConfig != nil {
		parameters.PrefetchConfig.SetDefaultsForFilePrefetching()
	}
	if parameters.KVFilePathBase != nil {
		parameters.KVFilePathBase.SetDefaults()
	}

	handler := NewPrefetchPrerequestHandler(handle, parameters.EngineKeysProviderPluginName, parameters.KVFilePathBase, parameters.PrefetchConfig).WithName(name)

	if err := initializeWorkerPool(handle.Context(), handler); err != nil {
		return nil, fmt.Errorf("failed to initialize worker pool for '%s' plugin - %w", name, err)
	}

	return handler, nil
}

// NewPrefetchPrerequestHandler initializes a new PrefetchPrerequestHandler.
func NewPrefetchPrerequestHandler(handle plugin.Handle, engineKeysProviderPluginName string, kvFilePathBase *KVFilePathBaseParams, prefetchConfig *PrefetchConfig) *PrefetchPrerequestHandler {
	h := &PrefetchPrerequestHandler{
		typedName:                    plugin.TypedName{Type: PrefetchPrerequestHandlerType},
		handle:                       handle,
		engineKeysProviderPluginName: engineKeysProviderPluginName,
		kvFilePathBase:               kvFilePathBase,
		prefetchConfig:               prefetchConfig,
	}
	if kvFilePathBase != nil {
		h.cache = &discoveryCache{}
	}
	return h
}

// PrefetchPrerequestHandler is a PreRequest plugin.
type PrefetchPrerequestHandler struct {
	typedName                    plugin.TypedName
	handle                       plugin.Handle
	engineKeysProviderPluginName string
	kvFilePathBase               *KVFilePathBaseParams
	cache                        *discoveryCache
	prefetchConfig               *PrefetchConfig
	workerPool                   *PrefetchWorkerPool
}

// TypedName returns the typed name of the plugin.
func (p *PrefetchPrerequestHandler) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *PrefetchPrerequestHandler) WithName(name string) *PrefetchPrerequestHandler {
	p.typedName.Name = name
	return p
}

// PreRequest logs engine keys and submits matching KV cache files for best-effort prefetching.
func (p *PrefetchPrerequestHandler) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	_ = schedulingResult

	if request == nil {
		return
	}

	log.FromContext(ctx).Info("prefetch-prerequest-handler PreRequest triggered", "requestId", request.RequestID,
		"engineKeysProviderPluginName", p.engineKeysProviderPluginName, "handleNil", p.handle == nil)

	if p.engineKeysProviderPluginName != "" && p.handle != nil {
		rawPlugin := p.handle.Plugin(p.engineKeysProviderPluginName)
		if rawPlugin != nil {
			if keysProvider, ok := rawPlugin.(*preciseprefixcache.Scorer); ok {
				log.FromContext(ctx).Info("PreRequest: accessing engine-keys from provider",
					"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)

				engineKeys, err := keysProvider.GetEngineKeysForRequest(ctx, request)
				if err != nil {
					log.FromContext(ctx).Error(err, "PreRequest: GetEngineKeysForRequest failed",
						"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)
					return
				}

				if len(engineKeys) == 0 {
					return
				}

				validEngineKeys := make([]uint64, 0, len(engineKeys))
				for _, ek := range engineKeys {
					if ek != 0 {
						validEngineKeys = append(validEngineKeys, ek)
					}
				}

				if emptyCount := len(engineKeys) - len(validEngineKeys); emptyCount > 0 {
					log.FromContext(ctx).Info("PreRequest: empty engine keys (0) received",
						"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName,
						"emptyCount", emptyCount, "totalEngineKeys", len(engineKeys))
				}
				if len(validEngineKeys) == 0 {
					log.FromContext(ctx).Info("PreRequest: all engine-keys are 0 (no requestKey→engineKey mapping); skipping paths",
						"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)
					return
				}

				log.FromContext(ctx).Info("PreRequest: engine-keys for request",
					"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName,
					"engineKeys", validEngineKeys)

				if p.kvFilePathBase != nil && p.kvFilePathBase.IsSet() {
					base := p.kvFilePathBase
					totalRanks := base.TpSize * base.PpSize * base.PcpSize * base.DcpSize
					filesPerRank := (len(validEngineKeys) + base.GpuBlocksPerFile - 1) / base.GpuBlocksPerFile
					allFilePaths := make([]string, 0, filesPerRank*totalRanks)

					for rank := 0; rank < totalRanks; rank++ {
						fullPaths := engineKeysToFilePaths(ctx, p.cache, base, rank, validEngineKeys)
						if fullPaths == nil {
							log.FromContext(ctx).V(1).Info("PreRequest: skipping request — KV cache base path not yet discovered",
								"requestId", request.RequestID)
							return
						}
						allFilePaths = append(allFilePaths, fullPaths...)
						log.FromContext(ctx).Info("PreRequest: KV-cache file paths for rank",
							"requestId", request.RequestID, "rank", rank,
							"gpuBlocksPerFile", base.GpuBlocksPerFile, "paths", fullPaths)
					}

					if p.workerPool != nil && p.workerPool.workQueue != nil && p.prefetchConfig != nil && p.prefetchConfig.Enabled {
						log.FromContext(ctx).Info("PreRequest: submitting files for prefetch",
							"requestId", request.RequestID, "fileCount", len(allFilePaths),
							"queueTimeout", p.prefetchConfig.QueueTimeout)

						submitted := 0
						skipped := 0

						for _, path := range allFilePaths {
							if p.prefetchConfig.QueueTimeout > 0 {
								select {
								case p.workerPool.workQueue <- path:
									submitted++
								case <-time.After(time.Duration(p.prefetchConfig.QueueTimeout) * time.Millisecond):
									skipped++
									log.FromContext(ctx).V(1).Info("PreRequest: queue timeout, skipping file",
										"requestId", request.RequestID, "path", path, "timeout", p.prefetchConfig.QueueTimeout)
								}
							} else {
								p.workerPool.workQueue <- path
								submitted++
							}
						}

						log.FromContext(ctx).Info("PreRequest: prefetch submission complete",
							"requestId", request.RequestID, "submitted", submitted, "skipped", skipped)
					}
				}
			} else {
				log.FromContext(ctx).Info("PreRequest: plugin found but is not a precise-prefix-cache scorer",
					"requestId", request.RequestID, "plugin", p.engineKeysProviderPluginName)
			}
		} else {
			registeredNames := make([]string, 0, len(p.handle.GetAllPluginsWithNames()))
			for name := range p.handle.GetAllPluginsWithNames() {
				registeredNames = append(registeredNames, name)
			}
			log.FromContext(ctx).Info("PreRequest: engine-keys provider plugin not found",
				"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName, "registeredPluginNames", registeredNames)
		}
	} else if p.engineKeysProviderPluginName != "" {
		log.FromContext(ctx).Info("PreRequest: engine-keys provider configured but handle unavailable",
			"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)
	}
}
