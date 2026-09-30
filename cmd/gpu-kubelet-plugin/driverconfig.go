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
	"fmt"
	"os"
	"path/filepath"

	"k8s.io/apimachinery/pkg/api/validate/content"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	coreclientset "k8s.io/client-go/kubernetes"
	"sigs.k8s.io/yaml"
)

const (
	driverConfigVersion          = "v1alpha1"
	defaultDriverConfigProfile   = "mixed"
	defaultDriverConfigNodeLabel = "nvidia.com/dra-driver-gpu.config"
)

// DriverConfig is the versioned startup configuration selected for this node.
// This POC intentionally limits it to ResourceSlice publication policy.
type DriverConfig struct {
	Version string           `json:"version"`
	GPU     *GPUDriverConfig `json:"gpu,omitempty"`
}

type GPUDriverConfig struct {
	AdvertisedDeviceTypes []string `json:"advertisedDeviceTypes"`
}

func defaultDriverConfig() *DriverConfig {
	return &DriverConfig{
		Version: driverConfigVersion,
		GPU: &GPUDriverConfig{
			AdvertisedDeviceTypes: []string{
				GpuDeviceType,
				MigStaticDeviceType,
				VfioDeviceType,
			},
		},
	}
}

func (c *DriverConfig) validate() error {
	if c.Version != driverConfigVersion {
		return fmt.Errorf("unsupported version %q, expected %q", c.Version, driverConfigVersion)
	}
	if c.GPU == nil {
		return fmt.Errorf("gpu configuration is required")
	}
	if len(c.GPU.AdvertisedDeviceTypes) == 0 {
		return fmt.Errorf("gpu.advertisedDeviceTypes must not be empty")
	}

	supported := map[string]struct{}{
		GpuDeviceType:       {},
		MigStaticDeviceType: {},
		VfioDeviceType:      {},
	}
	seen := make(map[string]struct{}, len(c.GPU.AdvertisedDeviceTypes))
	for _, deviceType := range c.GPU.AdvertisedDeviceTypes {
		if _, ok := supported[deviceType]; !ok {
			return fmt.Errorf("unsupported gpu.advertisedDeviceTypes value %q", deviceType)
		}
		if _, ok := seen[deviceType]; ok {
			return fmt.Errorf("duplicate gpu.advertisedDeviceTypes value %q", deviceType)
		}
		seen[deviceType] = struct{}{}
	}
	return nil
}

func (c *DriverConfig) advertises(deviceType string) bool {
	if c == nil || c.GPU == nil {
		return true
	}
	if deviceType == MigDynamicDeviceType {
		deviceType = MigStaticDeviceType
	}
	for _, allowed := range c.GPU.AdvertisedDeviceTypes {
		if deviceType == allowed {
			return true
		}
	}
	return false
}

func resolveDriverConfig(
	ctx context.Context,
	client coreclientset.Interface,
	nodeName string,
	configDirectory string,
	defaultProfile string,
	nodeLabel string,
) (*DriverConfig, string, error) {
	if configDirectory == "" {
		return defaultDriverConfig(), defaultDriverConfigProfile, nil
	}
	if client == nil {
		return nil, "", fmt.Errorf("kubernetes client is required")
	}
	if errs := validation.IsQualifiedName(nodeLabel); len(errs) > 0 {
		return nil, "", fmt.Errorf("invalid driver config node label %q: %v", nodeLabel, errs)
	}

	node, err := client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return nil, "", fmt.Errorf("get node %q: %w", nodeName, err)
	}

	profile := defaultProfile
	if selected := node.Labels[nodeLabel]; selected != "" {
		profile = selected
	}
	if profile == "" {
		return nil, "", fmt.Errorf("no driver config profile selected and no default configured")
	}
	if errs := validation.IsConfigMapKey(profile); len(errs) > 0 {
		return nil, "", fmt.Errorf("invalid driver config profile %q: %v", profile, errs)
	}
	if errs := content.IsPathSegmentName(profile); len(errs) > 0 {
		return nil, "", fmt.Errorf("invalid driver config profile path %q: %v", profile, errs)
	}

	data, err := os.ReadFile(filepath.Join(configDirectory, profile))
	if err != nil {
		return nil, "", fmt.Errorf("read driver config profile %q: %w", profile, err)
	}

	config := &DriverConfig{}
	if err := yaml.UnmarshalStrict(data, config); err != nil {
		return nil, "", fmt.Errorf("decode driver config profile %q: %w", profile, err)
	}
	if err := config.validate(); err != nil {
		return nil, "", fmt.Errorf("validate driver config profile %q: %w", profile, err)
	}

	return config, profile, nil
}
