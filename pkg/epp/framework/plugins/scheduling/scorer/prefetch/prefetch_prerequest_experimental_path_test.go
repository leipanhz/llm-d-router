/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package prefetch

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// makeRank0Tree builds a minimal vLLM-style on-disk layout under root:
//
//	<root>/<safeModel>_<digest>_r0/<sub1>/<sub2>_g<groupIdx>/<dummy>.bin
//
// safeModel is modelName with '/' replaced by '_'. Returns the rank0 dir.
func makeRank0Tree(t *testing.T, root, modelName, digest, sub1, sub2 string, groupIdx int) string {
	t.Helper()
	safe := strings.ReplaceAll(modelName, "/", "_")
	rank0 := filepath.Join(root, fmt.Sprintf("%s_%s_r0", safe, digest))
	groupDir := filepath.Join(rank0, sub1, fmt.Sprintf("%s_g%d", sub2, groupIdx))
	require.NoError(t, os.MkdirAll(groupDir, 0o755))
	dummy := filepath.Join(groupDir, "0000000000000000.bin")
	require.NoError(t, os.WriteFile(dummy, []byte{}, 0o644))
	return rank0
}

func TestDiscover_SingleMatch(t *testing.T) {
	root := t.TempDir()
	makeRank0Tree(t, root, "meta-llama/Llama-3.1-8B", "abcdef123456", "abc", "de", 0)

	params := &KVFilePathBaseParams{
		RootDir:   root,
		ModelName: "meta-llama/Llama-3.1-8B",
	}
	cache := &discoveryCache{}
	require.NoError(t, cache.discover(context.Background(), params))

	expectedBase := filepath.Join(root, "meta-llama_Llama-3.1-8B_abcdef123456")
	assert.Equal(t, expectedBase, cache.base)
	assert.Equal(t, 0, cache.group)
	assert.True(t, cache.done)
}

func TestDiscover_NoMatch(t *testing.T) {
	root := t.TempDir()
	params := &KVFilePathBaseParams{
		RootDir:   root,
		ModelName: "no-such-model",
	}
	cache := &discoveryCache{}
	err := cache.discover(context.Background(), params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no folder matches")
	assert.Contains(t, err.Error(), "vLLM initialized")
	assert.False(t, cache.done)
}

func TestDiscover_AmbiguousMatch(t *testing.T) {
	root := t.TempDir()
	makeRank0Tree(t, root, "model", "aaaaaaaaaaaa", "aaa", "aa", 0)
	makeRank0Tree(t, root, "model", "bbbbbbbbbbbb", "bbb", "bb", 0)

	params := &KVFilePathBaseParams{
		RootDir:   root,
		ModelName: "model",
	}
	cache := &discoveryCache{}
	err := cache.discover(context.Background(), params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ambiguous")
	// Both candidates should appear in the message
	assert.Contains(t, err.Error(), "aaaaaaaaaaaa")
	assert.Contains(t, err.Error(), "bbbbbbbbbbbb")
}

func TestDiscover_NoBlocksYet(t *testing.T) {
	root := t.TempDir()
	// Create the _r0 dir but no group subfolders.
	rank0 := filepath.Join(root, "model_abcdef123456_r0")
	require.NoError(t, os.MkdirAll(rank0, 0o755))

	params := &KVFilePathBaseParams{
		RootDir:   root,
		ModelName: "model",
	}
	cache := &discoveryCache{}
	err := cache.discover(context.Background(), params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no \"<hh>_g<N>\" group subfolder found")
}

func TestDiscover_GroupIdxNonZero(t *testing.T) {
	root := t.TempDir()
	makeRank0Tree(t, root, "model", "abcdef123456", "abc", "de", 3)

	params := &KVFilePathBaseParams{
		RootDir:   root,
		ModelName: "model",
	}
	cache := &discoveryCache{}
	require.NoError(t, cache.discover(context.Background(), params))
	assert.Equal(t, 3, cache.group)
}

func TestEngineKeyToFullPath_FormatsCorrectly(t *testing.T) {
	root := t.TempDir()
	makeRank0Tree(t, root, "m", "abc", "abc", "ab", 0)

	params := &KVFilePathBaseParams{
		RootDir:   root,
		ModelName: "m",
	}
	cache := &discoveryCache{}
	require.NoError(t, cache.discover(context.Background(), params))

	// 0xdeadbeef00000000 → hex "deadbeef00000000"; sub1="dea", sub2="db"
	path, ok := cache.engineKeyToFullPath(context.Background(), params, 2, 0xdeadbeef00000000)
	assert.True(t, ok)
	expectedBase := filepath.Join(root, "m_abc")
	expected := fmt.Sprintf("%s_r2/dea/db_g0/deadbeef00000000.bin", expectedBase)
	assert.Equal(t, expected, path)
}

func TestEngineKeyToFullPath_DiscoveryDefers(t *testing.T) {
	root := t.TempDir()
	// No vLLM tree under root → discovery fails.
	params := &KVFilePathBaseParams{
		RootDir:   root,
		ModelName: "no-such-model",
	}
	cache := &discoveryCache{}
	path, ok := cache.engineKeyToFullPath(context.Background(), params, 0, 0x1)
	assert.False(t, ok)
	assert.Equal(t, "", path)
}

func TestInvalidate(t *testing.T) {
	root := t.TempDir()
	makeRank0Tree(t, root, "m", "first", "abc", "ab", 0)

	params := &KVFilePathBaseParams{
		RootDir:   root,
		ModelName: "m",
	}
	cache := &discoveryCache{}
	require.NoError(t, cache.discover(context.Background(), params))
	firstBase := cache.base

	// Simulate vLLM restart with new digest: replace the on-disk tree.
	require.NoError(t, os.RemoveAll(filepath.Join(root, "m_first_r0")))
	makeRank0Tree(t, root, "m", "second", "def", "ef", 0)

	// Without invalidate, cache is stale.
	require.NoError(t, cache.discover(context.Background(), params))
	assert.Equal(t, firstBase, cache.base)

	// After invalidate, the next discover picks up the new digest.
	cache.invalidate()
	require.NoError(t, cache.discover(context.Background(), params))
	assert.Equal(t, filepath.Join(root, "m_second"), cache.base)
}

func TestParseGroupSuffix(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want int
	}{
		{"zero", "ab_g0", 0},
		{"two-digit", "ab_g12", 12},
		{"missing _g", "ab", -1},
		{"non-numeric", "ab_gxx", -1},
		{"trailing chars", "ab_g3x", -1},
		{"empty index", "ab_g", -1},
		{"multiple _g, last wins", "ab_g1_g2", 2},
		{"negative not allowed", "ab_g-1", -1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			assert.Equal(t, c.want, parseGroupSuffix(c.in))
		})
	}
}

func TestEngineKeysToFilePaths_BatchedByBlocksPerFile(t *testing.T) {
	root := t.TempDir()
	makeRank0Tree(t, root, "m", "abc", "abc", "ab", 0)

	params := &KVFilePathBaseParams{
		RootDir:          root,
		ModelName:        "m",
		GpuBlocksPerFile: 4,
	}
	cache := &discoveryCache{}
	keys := []uint64{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8}
	paths := engineKeysToFilePaths(context.Background(), cache, params, 0, keys)
	require.Len(t, paths, 2)
	// With GpuBlocksPerFile=4, the loop picks indices 3 and 7
	// (zero-based), i.e. keys 0x4 and 0x8.
	assert.Contains(t, paths[0], "0000000000000004.bin")
	assert.Contains(t, paths[1], "0000000000000008.bin")
}

func TestEngineKeysToFilePaths_DiscoveryDefersReturnsNil(t *testing.T) {
	root := t.TempDir()
	params := &KVFilePathBaseParams{
		RootDir:          root,
		ModelName:        "no-such-model",
		GpuBlocksPerFile: 1,
	}
	cache := &discoveryCache{}
	paths := engineKeysToFilePaths(context.Background(), cache, params, 0, []uint64{0x1, 0x2})
	assert.Nil(t, paths)
}
