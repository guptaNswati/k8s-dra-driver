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
	"testing"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/stretchr/testify/require"
)

type fakeDeleteNVML struct {
	nvml.Interface
	parent nvml.Device
}

func (l *fakeDeleteNVML) InitWithFlags(uint32) nvml.Return {
	return nvml.SUCCESS
}

func (l *fakeDeleteNVML) Shutdown() nvml.Return {
	return nvml.SUCCESS
}

func (l *fakeDeleteNVML) DeviceGetHandleByUUID(string) (nvml.Device, nvml.Return) {
	return l.parent, nvml.SUCCESS
}

type fakeDeleteParent struct {
	nvml.Device
	gi nvml.GpuInstance
}

func (d *fakeDeleteParent) GetGpuInstanceById(int) (nvml.GpuInstance, nvml.Return) {
	return d.gi, nvml.SUCCESS
}

type fakeDeleteGI struct {
	nvml.GpuInstance
	ci    nvml.ComputeInstance
	ciRet nvml.Return
}

func (g *fakeDeleteGI) GetComputeInstanceById(int) (nvml.ComputeInstance, nvml.Return) {
	return g.ci, g.ciRet
}

type fakeDeleteCI struct {
	nvml.ComputeInstance
	device    nvml.Device
	destroyed bool
}

func (c *fakeDeleteCI) GetInfo() (nvml.ComputeInstanceInfo, nvml.Return) {
	return nvml.ComputeInstanceInfo{Device: c.device}, nvml.SUCCESS
}

func (c *fakeDeleteCI) Destroy() nvml.Return {
	c.destroyed = true
	return nvml.SUCCESS
}

type fakeDeleteMIGDevice struct {
	nvml.Device
	uuid string
}

func (d *fakeDeleteMIGDevice) GetUUID() (string, nvml.Return) {
	return d.uuid, nvml.SUCCESS
}

func TestResolveCreatedMigUUID(t *testing.T) {
	t.Run("direct MIG UUID", func(t *testing.T) {
		called := false
		uuid, err := resolveCreatedMigUUID("MIG-direct", "GPU-parent", 3, 4, func() (*MigLiveTuple, error) {
			called = true
			return nil, nil
		})
		require.NoError(t, err)
		require.Equal(t, "MIG-direct", uuid)
		require.False(t, called)
	})

	t.Run("parent UUID falls back to live discovery", func(t *testing.T) {
		uuid, err := resolveCreatedMigUUID("GPU-parent", "GPU-parent", 3, 4, func() (*MigLiveTuple, error) {
			return &MigLiveTuple{ParentUUID: "GPU-parent", GIID: 3, CIID: 4, MigUUID: "MIG-live"}, nil
		})
		require.NoError(t, err)
		require.Equal(t, "MIG-live", uuid)
	})

	t.Run("fallback tuple mismatch", func(t *testing.T) {
		_, err := resolveCreatedMigUUID("GPU-parent", "GPU-parent", 3, 4, func() (*MigLiveTuple, error) {
			return &MigLiveTuple{ParentUUID: "GPU-parent", GIID: 7, CIID: 0, MigUUID: "MIG-other"}, nil
		})
		require.ErrorContains(t, err, "newly created MIG tuple mismatch")
	})
}

func TestDeleteMigDeviceRejectsReusedTuple(t *testing.T) {
	liveDevice := &fakeDeleteMIGDevice{uuid: "MIG-replacement"}
	ci := &fakeDeleteCI{device: liveDevice}
	gi := &fakeDeleteGI{ci: ci, ciRet: nvml.SUCCESS}
	parent := &fakeDeleteParent{gi: gi}
	lib := &deviceLib{nvmllib: &fakeDeleteNVML{parent: parent}}

	err := lib.deleteMigDevice(&MigLiveTuple{
		ParentUUID: "GPU-parent-1",
		GIID:       3,
		CIID:       4,
		MigUUID:    "MIG-expected",
	})

	require.ErrorContains(t, err, "refusing to delete reused MIG tuple")
	require.False(t, ci.destroyed)
}

func TestDeleteMigDeviceRejectsMissingExpectedCI(t *testing.T) {
	gi := &fakeDeleteGI{ciRet: nvml.ERROR_NOT_FOUND}
	parent := &fakeDeleteParent{gi: gi}
	lib := &deviceLib{nvmllib: &fakeDeleteNVML{parent: parent}}

	err := lib.deleteMigDevice(&MigLiveTuple{
		ParentUUID: "GPU-parent-1",
		GIID:       3,
		CIID:       4,
		MigUUID:    "MIG-expected",
	})

	require.ErrorContains(t, err, "expected compute instance UUID MIG-expected was not found")
}
