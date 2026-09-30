/*
Copyright The Kubernetes Authors.

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

package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sfake "k8s.io/client-go/kubernetes/fake"
)

func TestDriverConfigValidate(t *testing.T) {
	tests := map[string]struct {
		config  *DriverConfig
		wantErr bool
	}{
		"mixed": {
			config: defaultDriverConfig(),
		},
		"dynamic MIG uses public MIG policy": {
			config: &DriverConfig{
				Version: driverConfigVersion,
				GPU:     &GPUDriverConfig{AdvertisedDeviceTypes: []string{MigStaticDeviceType}},
			},
		},
		"missing version": {
			config:  &DriverConfig{GPU: &GPUDriverConfig{AdvertisedDeviceTypes: []string{GpuDeviceType}}},
			wantErr: true,
		},
		"missing GPU config": {
			config:  &DriverConfig{Version: driverConfigVersion},
			wantErr: true,
		},
		"empty device types": {
			config:  &DriverConfig{Version: driverConfigVersion, GPU: &GPUDriverConfig{}},
			wantErr: true,
		},
		"internal dynamic MIG type": {
			config: &DriverConfig{
				Version: driverConfigVersion,
				GPU:     &GPUDriverConfig{AdvertisedDeviceTypes: []string{MigDynamicDeviceType}},
			},
			wantErr: true,
		},
		"unknown type": {
			config: &DriverConfig{
				Version: driverConfigVersion,
				GPU:     &GPUDriverConfig{AdvertisedDeviceTypes: []string{"unknown"}},
			},
			wantErr: true,
		},
		"duplicate type": {
			config: &DriverConfig{
				Version: driverConfigVersion,
				GPU:     &GPUDriverConfig{AdvertisedDeviceTypes: []string{GpuDeviceType, GpuDeviceType}},
			},
			wantErr: true,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			err := tc.config.validate()
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
		})
	}
}

func TestDriverConfigAdvertises(t *testing.T) {
	config := &DriverConfig{
		Version: driverConfigVersion,
		GPU:     &GPUDriverConfig{AdvertisedDeviceTypes: []string{GpuDeviceType, MigStaticDeviceType}},
	}

	assert.True(t, config.advertises(GpuDeviceType))
	assert.True(t, config.advertises(MigStaticDeviceType))
	assert.True(t, config.advertises(MigDynamicDeviceType))
	assert.False(t, config.advertises(VfioDeviceType))
	assert.True(t, (*DriverConfig)(nil).advertises(VfioDeviceType))
}

func TestResolveDriverConfig(t *testing.T) {
	writeProfile := func(t *testing.T, directory, name, contents string) {
		t.Helper()
		require.NoError(t, os.WriteFile(filepath.Join(directory, name), []byte(contents), 0600))
	}

	t.Run("empty directory preserves built-in mixed behavior", func(t *testing.T) {
		config, profile, err := resolveDriverConfig(
			context.Background(), nil, "node-a", "", defaultDriverConfigProfile, defaultDriverConfigNodeLabel,
		)
		require.NoError(t, err)
		assert.Equal(t, defaultDriverConfigProfile, profile)
		assert.ElementsMatch(t, []string{GpuDeviceType, MigStaticDeviceType, VfioDeviceType}, config.GPU.AdvertisedDeviceTypes)
	})

	t.Run("node label selects profile", func(t *testing.T) {
		directory := t.TempDir()
		writeProfile(t, directory, "container", `
version: v1alpha1
gpu:
  advertisedDeviceTypes: [gpu, mig]
`)
		client := k8sfake.NewSimpleClientset(&corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "node-a",
				Labels: map[string]string{defaultDriverConfigNodeLabel: "container"},
			},
		})

		config, profile, err := resolveDriverConfig(
			context.Background(), client, "node-a", directory, "mixed", defaultDriverConfigNodeLabel,
		)
		require.NoError(t, err)
		assert.Equal(t, "container", profile)
		assert.Equal(t, []string{GpuDeviceType, MigStaticDeviceType}, config.GPU.AdvertisedDeviceTypes)
	})

	t.Run("default profile is used when label is absent", func(t *testing.T) {
		directory := t.TempDir()
		writeProfile(t, directory, "passthrough", `
version: v1alpha1
gpu:
  advertisedDeviceTypes: [vfio]
`)
		client := k8sfake.NewSimpleClientset(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})

		config, profile, err := resolveDriverConfig(
			context.Background(), client, "node-a", directory, "passthrough", defaultDriverConfigNodeLabel,
		)
		require.NoError(t, err)
		assert.Equal(t, "passthrough", profile)
		assert.Equal(t, []string{VfioDeviceType}, config.GPU.AdvertisedDeviceTypes)
	})

	t.Run("explicit empty label fails closed", func(t *testing.T) {
		directory := t.TempDir()
		client := k8sfake.NewSimpleClientset(&corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "node-a",
				Labels: map[string]string{defaultDriverConfigNodeLabel: ""},
			},
		})

		_, _, err := resolveDriverConfig(
			context.Background(), client, "node-a", directory, "mixed", defaultDriverConfigNodeLabel,
		)
		require.Error(t, err)
		assert.ErrorContains(t, err, "must not be empty")
	})

	t.Run("unknown profile fails", func(t *testing.T) {
		directory := t.TempDir()
		client := k8sfake.NewSimpleClientset(&corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "node-a",
				Labels: map[string]string{defaultDriverConfigNodeLabel: "missing"},
			},
		})

		_, _, err := resolveDriverConfig(
			context.Background(), client, "node-a", directory, "mixed", defaultDriverConfigNodeLabel,
		)
		require.Error(t, err)
	})

	t.Run("profile cannot escape config directory", func(t *testing.T) {
		directory := t.TempDir()
		client := k8sfake.NewSimpleClientset(&corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "node-a",
				Labels: map[string]string{defaultDriverConfigNodeLabel: ".."},
			},
		})

		_, _, err := resolveDriverConfig(
			context.Background(), client, "node-a", directory, "mixed", defaultDriverConfigNodeLabel,
		)
		require.Error(t, err)
		assert.ErrorContains(t, err, "invalid driver config profile")
	})

	t.Run("unknown field fails strict decoding", func(t *testing.T) {
		directory := t.TempDir()
		writeProfile(t, directory, "mixed", `
version: v1alpha1
gpu:
  advertisedDeviceTypes: [gpu, mig, vfio]
  sharing: {}
`)
		client := k8sfake.NewSimpleClientset(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-a"}})

		_, _, err := resolveDriverConfig(
			context.Background(), client, "node-a", directory, "mixed", defaultDriverConfigNodeLabel,
		)
		require.Error(t, err)
		assert.ErrorContains(t, err, `unknown field "sharing"`)
	})
}
