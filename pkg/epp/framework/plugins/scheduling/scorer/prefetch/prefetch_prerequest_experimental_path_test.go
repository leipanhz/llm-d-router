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
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeAnchorFile builds a vLLM-style on-disk path under root and writes
// an empty `<digestHex>.bin` file at the leaf, mirroring the layout that
// FileMapper produces:
//
//	<root>/<safeModelName>_<basePathDigest>_r<rank>/<sub1>/<sub2>_g<groupIdx>/<digestHex>.bin
//
// where sub1 = digestHex[:3], sub2 = digestHex[3:5]. Returns the digest
// bytes used to anchor discovery.
func writeAnchorFile(t *testing.T, root, safeModelName, basePathDigest string, rank int, groupIdx int, digestHex string) []byte {
	t.Helper()
	require.GreaterOrEqual(t, len(digestHex), 5, "digest hex must be at least 5 chars")
	sub1, sub2 := digestHex[:3], digestHex[3:5]
	rankDir := filepath.Join(root, fmt.Sprintf("%s_%s_r%d", safeModelName, basePathDigest, rank))
	groupDir := filepath.Join(rankDir, sub1, fmt.Sprintf("%s_g%d", sub2, groupIdx))
	require.NoError(t, os.MkdirAll(groupDir, 0o755))
	leaf := filepath.Join(groupDir, digestHex+".bin")
	require.NoError(t, os.WriteFile(leaf, []byte{}, 0o644))
	d, err := hex.DecodeString(digestHex)
	require.NoError(t, err)
	return d
}

func TestDiscover_FindsAnchorAndExtractsBaseAndGroup(t *testing.T) {
	root := t.TempDir()
	digest := writeAnchorFile(t, root, "Qwen_Qwen3-8B", "07d7b166f256", 0, 0,
		"7050ab3d42d0b5e628c4e846e90715c1e1b2ac6247ce88b5e1a944b73c04d5d1")

	params := &KVFilePathBaseParams{RootDir: root, ModelName: "Qwen/Qwen3-8B"}
	cache := &discoveryCache{}
	require.NoError(t, cache.discover(context.Background(), params, digest))

	expectedBase := filepath.Join(root, "Qwen_Qwen3-8B_07d7b166f256")
	assert.Equal(t, expectedBase, cache.base)
	assert.Equal(t, 0, cache.group)
	assert.True(t, cache.done)
}

func TestDiscover_GroupIdxNonZero(t *testing.T) {
	root := t.TempDir()
	digest := writeAnchorFile(t, root, "model", "abcdef123456", 2, 3,
		"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789")

	params := &KVFilePathBaseParams{RootDir: root, ModelName: "model"}
	cache := &discoveryCache{}
	require.NoError(t, cache.discover(context.Background(), params, digest))
	assert.Equal(t, 3, cache.group)
	assert.Equal(t, filepath.Join(root, "model_abcdef123456"), cache.base)
}

func TestDiscover_NoMatch(t *testing.T) {
	root := t.TempDir()
	params := &KVFilePathBaseParams{RootDir: root, ModelName: "no-such-model"}
	cache := &discoveryCache{}
	digest := []byte{0xde, 0xad, 0xbe, 0xef, 0x00, 0x00, 0x00, 0x00}
	err := cache.discover(context.Background(), params, digest)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no folder matches")
	assert.False(t, cache.done)
}

func TestDiscover_FolderExistsButAnchorMissing(t *testing.T) {
	root := t.TempDir()
	// Folder for the model exists but doesn't contain the anchor digest.
	otherDigestHex := "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
	writeAnchorFile(t, root, "model", "abcdef123456", 0, 0, otherDigestHex)

	params := &KVFilePathBaseParams{RootDir: root, ModelName: "model"}
	cache := &discoveryCache{}
	digest := []byte{0xde, 0xad, 0xbe, 0xef, 0x00, 0x00, 0x00, 0x00}
	err := cache.discover(context.Background(), params, digest)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "anchor file")
	assert.False(t, cache.done)
}

func TestDiscover_EmptyDigest(t *testing.T) {
	root := t.TempDir()
	params := &KVFilePathBaseParams{RootDir: root, ModelName: "model"}
	cache := &discoveryCache{}
	err := cache.discover(context.Background(), params, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty digest")
}

func TestDigestToFullPath_FormatsCorrectly(t *testing.T) {
	root := t.TempDir()
	anchor := writeAnchorFile(t, root, "m", "abcdef123456", 0, 0,
		"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")

	params := &KVFilePathBaseParams{RootDir: root, ModelName: "m"}
	cache := &discoveryCache{}
	require.NoError(t, cache.discover(context.Background(), params, anchor))

	// Build a path for a different digest on a different rank using the
	// cached prefix and group.
	other := []byte{0x12, 0x34, 0x56, 0x78, 0xaa, 0xbb, 0xcc, 0xdd}
	path := cache.digestToFullPath(2, other)
	expected := filepath.Join(root, "m_abcdef123456") + "_r2/123/45_g0/12345678aabbccdd.bin"
	assert.Equal(t, expected, path)
}

func TestInvalidate(t *testing.T) {
	root := t.TempDir()
	first := writeAnchorFile(t, root, "m", "firstdigest1", 0, 0,
		"1111111111111111111111111111111111111111111111111111111111111111")

	params := &KVFilePathBaseParams{RootDir: root, ModelName: "m"}
	cache := &discoveryCache{}
	require.NoError(t, cache.discover(context.Background(), params, first))
	firstBase := cache.base

	// Simulate vLLM restart with new base-path digest: replace the on-disk tree.
	require.NoError(t, os.RemoveAll(filepath.Join(root, "m_firstdigest1_r0")))
	second := writeAnchorFile(t, root, "m", "seconddigest", 0, 0,
		"2222222222222222222222222222222222222222222222222222222222222222")

	// Without invalidate, cache is stale (no probe happens — discover is a no-op).
	require.NoError(t, cache.discover(context.Background(), params, second))
	assert.Equal(t, firstBase, cache.base)

	// After invalidate, the next discover picks up the new base-path digest.
	cache.invalidate()
	require.NoError(t, cache.discover(context.Background(), params, second))
	assert.Equal(t, filepath.Join(root, "m_seconddigest"), cache.base)
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

func TestDigestsToFilePaths_BatchedByBlocksPerFile(t *testing.T) {
	root := t.TempDir()

	digestN := func(n byte) []byte {
		d := make([]byte, 32)
		d[31] = n
		return d
	}

	// fs-connector aggregates 4 vLLM blocks per file and names each
	// file after the last block. With 8 digests and GpuBlocksPerFile=4,
	// vLLM persists files at digests[3] and digests[7]. Discovery
	// anchors on digests[3] (the first persistable index).
	anchorHex := hex.EncodeToString(digestN(4))
	writeAnchorFile(t, root, "m", "abcdef123456", 0, 0, anchorHex)

	params := &KVFilePathBaseParams{
		RootDir:          root,
		ModelName:        "m",
		GpuBlocksPerFile: 4,
	}
	cache := &discoveryCache{}

	digests := [][]byte{
		digestN(1), digestN(2), digestN(3), digestN(4),
		digestN(5), digestN(6), digestN(7), digestN(8),
	}
	paths := digestsToFilePaths(context.Background(), cache, params, 0, digests)
	require.Len(t, paths, 2)
	assert.Contains(t, paths[0], hex.EncodeToString(digestN(4))+".bin")
	assert.Contains(t, paths[1], hex.EncodeToString(digestN(8))+".bin")
}

func TestDigestsToFilePaths_FewerDigestsThanBlocksPerFile(t *testing.T) {
	root := t.TempDir()
	// vLLM hasn't aggregated enough blocks to write any file yet, so
	// discovery must defer and the helper returns nil.
	params := &KVFilePathBaseParams{
		RootDir:          root,
		ModelName:        "m",
		GpuBlocksPerFile: 8,
	}
	cache := &discoveryCache{}
	digests := make([][]byte, 7) // 7 < 8
	for i := range digests {
		d := make([]byte, 32)
		d[31] = byte(i + 1)
		digests[i] = d
	}
	paths := digestsToFilePaths(context.Background(), cache, params, 0, digests)
	assert.Nil(t, paths)
}

func TestDigestsToFilePaths_DiscoveryDefersReturnsNil(t *testing.T) {
	root := t.TempDir()
	params := &KVFilePathBaseParams{
		RootDir:          root,
		ModelName:        "no-such-model",
		GpuBlocksPerFile: 1,
	}
	cache := &discoveryCache{}
	// No model folder under root → glob finds nothing → discover errors → nil result.
	paths := digestsToFilePaths(context.Background(), cache, params, 0,
		[][]byte{{0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 2}})
	assert.Nil(t, paths)
}
