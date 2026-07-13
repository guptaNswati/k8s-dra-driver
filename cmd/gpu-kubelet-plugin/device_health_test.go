/*
Copyright The Kubernetes Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"context"
	"strconv"
	"sync"
	"testing"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	resourceapi "k8s.io/api/resource/v1"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockHealthMonitor implements deviceHealthMonitor for testing healthEventToTaint.
type mockHealthMonitor struct {
	nonFatalXids map[uint64]bool
}

type fakeMigDeviceResolver struct {
	live      *MigLiveTuple
	err       error
	requested []*MigSpecTuple
}

type fakeHealthEventSet struct {
	nvml.EventSet
}

type fakeHealthDevice struct {
	nvml.Device
	registerCalls int
}

func (d *fakeHealthDevice) GetSupportedEventTypes() (uint64, nvml.Return) {
	return uint64(nvml.EventTypeXidCriticalError), nvml.SUCCESS
}

func (d *fakeHealthDevice) RegisterEvents(_ uint64, _ nvml.EventSet) nvml.Return {
	d.registerCalls++
	return nvml.SUCCESS
}

type fakeHealthNVML struct {
	nvml.Interface
	device         nvml.Device
	eventSet       nvml.EventSet
	initCalls      int
	eventSetCalls  int
	deviceGetCalls int
}

func (l *fakeHealthNVML) Init() nvml.Return {
	l.initCalls++
	return nvml.SUCCESS
}

func (l *fakeHealthNVML) Shutdown() nvml.Return {
	return nvml.SUCCESS
}

func (l *fakeHealthNVML) EventSetCreate() (nvml.EventSet, nvml.Return) {
	l.eventSetCalls++
	return l.eventSet, nvml.SUCCESS
}

func (l *fakeHealthNVML) DeviceGetHandleByUUID(_ string) (nvml.Device, nvml.Return) {
	l.deviceGetCalls++
	return l.device, nvml.SUCCESS
}

func (r *fakeMigDeviceResolver) FindMigDevBySpec(spec *MigSpecTuple) (*MigLiveTuple, error) {
	r.requested = append(r.requested, spec)
	return r.live, r.err
}

func (m *mockHealthMonitor) Start(context.Context) error          { return nil }
func (m *mockHealthMonitor) Stop()                                {}
func (m *mockHealthMonitor) Unhealthy() <-chan *DeviceHealthEvent { return nil }
func (m *mockHealthMonitor) IsEventNonFatal(e *DeviceHealthEvent) bool {
	if e.EventType == HealthEventXID {
		return m.nonFatalXids[e.EventData]
	}
	return false
}

func TestAddOrUpdateTaint_NewTaint(t *testing.T) {
	dev := &AllocatableDevice{}
	taint := &resourceapi.DeviceTaint{
		Key:    TaintKeyXID,
		Value:  "48",
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	}

	changed := dev.AddOrUpdateTaint(taint)

	require.True(t, changed)
	require.Len(t, dev.Taints(), 1)
	assert.Equal(t, TaintKeyXID, dev.Taints()[0].Key)
	assert.Equal(t, "48", dev.Taints()[0].Value)
	assert.Equal(t, resourceapi.DeviceTaintEffectNoSchedule, dev.Taints()[0].Effect)
}

func TestAddOrUpdateTaint_DuplicateNoChange(t *testing.T) {
	dev := &AllocatableDevice{}
	taint := &resourceapi.DeviceTaint{
		Key:    TaintKeyGPULost,
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	}

	dev.AddOrUpdateTaint(taint)
	changed := dev.AddOrUpdateTaint(taint)

	assert.False(t, changed, "identical taint should not count as a change")
	assert.Len(t, dev.Taints(), 1)
}

func TestAddOrUpdateTaint_UpdateValue(t *testing.T) {
	dev := &AllocatableDevice{}
	dev.AddOrUpdateTaint(&resourceapi.DeviceTaint{
		Key:    TaintKeyXID,
		Value:  "48",
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	})

	changed := dev.AddOrUpdateTaint(&resourceapi.DeviceTaint{
		Key:    TaintKeyXID,
		Value:  "63",
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	})

	require.True(t, changed)
	require.Len(t, dev.Taints(), 1)
	assert.Equal(t, "63", dev.Taints()[0].Value, "value should be overwritten to latest XID")
}

func TestAddOrUpdateTaint_UpdateEffect(t *testing.T) {
	dev := &AllocatableDevice{}
	dev.AddOrUpdateTaint(&resourceapi.DeviceTaint{
		Key:    TaintKeyXID,
		Value:  "48",
		Effect: resourceapi.DeviceTaintEffectNone,
	})

	changed := dev.AddOrUpdateTaint(&resourceapi.DeviceTaint{
		Key:    TaintKeyXID,
		Value:  "48",
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	})

	require.True(t, changed)
	assert.Equal(t, resourceapi.DeviceTaintEffectNoSchedule, dev.Taints()[0].Effect)
}

func TestAddOrUpdateTaint_DifferentKeysAppended(t *testing.T) {
	dev := &AllocatableDevice{}
	dev.AddOrUpdateTaint(&resourceapi.DeviceTaint{
		Key:    TaintKeyXID,
		Value:  "48",
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	})
	dev.AddOrUpdateTaint(&resourceapi.DeviceTaint{
		Key:    TaintKeyGPULost,
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	})

	taints := dev.Taints()
	require.Len(t, taints, 2)
	assert.Equal(t, TaintKeyXID, taints[0].Key)
	assert.Equal(t, TaintKeyGPULost, taints[1].Key)
}

func TestAddOrUpdateTaint_TimeAddedResetOnChange(t *testing.T) {
	dev := &AllocatableDevice{}
	dev.AddOrUpdateTaint(&resourceapi.DeviceTaint{
		Key:    TaintKeyXID,
		Value:  "48",
		Effect: resourceapi.DeviceTaintEffectNone,
	})

	dev.AddOrUpdateTaint(&resourceapi.DeviceTaint{
		Key:    TaintKeyXID,
		Value:  "63",
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	})

	assert.Nil(t, dev.Taints()[0].TimeAdded, "TimeAdded should be nil so the API server sets a fresh timestamp")
}

func TestTaintsConcurrentReadWrite(t *testing.T) {
	dev := &AllocatableDevice{}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := range 1000 {
			dev.AddOrUpdateTaint(&resourceapi.DeviceTaint{
				Key:    TaintKeyXID,
				Value:  strconv.Itoa(i),
				Effect: resourceapi.DeviceTaintEffectNoSchedule,
			})
		}
	}()
	go func() {
		defer wg.Done()
		for range 1000 {
			_ = dev.Taints()
		}
	}()
	wg.Wait()
	require.Len(t, dev.Taints(), 1)
}

func TestHealthEventToTaint(t *testing.T) {
	monitor := &mockHealthMonitor{
		nonFatalXids: map[uint64]bool{13: true, 31: true},
	}

	tests := []struct {
		name           string
		event          *DeviceHealthEvent
		monitor        deviceHealthMonitor
		expectedKey    string
		expectedValue  string
		expectedEffect resourceapi.DeviceTaintEffect
	}{
		{
			name: "fatal XID",
			event: &DeviceHealthEvent{
				EventType: HealthEventXID,
				EventData: 48,
			},
			monitor:        monitor,
			expectedKey:    TaintKeyXID,
			expectedValue:  "48",
			expectedEffect: resourceapi.DeviceTaintEffectNoSchedule,
		},
		{
			name: "non-fatal XID (skipped)",
			event: &DeviceHealthEvent{
				EventType: HealthEventXID,
				EventData: 13,
			},
			monitor:        monitor,
			expectedKey:    TaintKeyXID,
			expectedValue:  "13",
			expectedEffect: resourceapi.DeviceTaintEffectNone,
		},
		{
			name: "XID with nil monitor defaults to fatal",
			event: &DeviceHealthEvent{
				EventType: HealthEventXID,
				EventData: 13,
			},
			monitor:        nil,
			expectedKey:    TaintKeyXID,
			expectedValue:  "13",
			expectedEffect: resourceapi.DeviceTaintEffectNoSchedule,
		},
		{
			name: "GPU lost",
			event: &DeviceHealthEvent{
				EventType: HealthEventGPULost,
			},
			monitor:        monitor,
			expectedKey:    TaintKeyGPULost,
			expectedValue:  "",
			expectedEffect: resourceapi.DeviceTaintEffectNoSchedule,
		},
		{
			name: "unmonitored",
			event: &DeviceHealthEvent{
				EventType: HealthEventUnmonitored,
			},
			monitor:        monitor,
			expectedKey:    TaintKeyUnmonitored,
			expectedValue:  "",
			expectedEffect: resourceapi.DeviceTaintEffectNone,
		},
		{
			name: "unknown event type defaults to unmonitored",
			event: &DeviceHealthEvent{
				EventType: DeviceHealthEventType("bogus"),
			},
			monitor:        monitor,
			expectedKey:    TaintKeyUnmonitored,
			expectedValue:  "",
			expectedEffect: resourceapi.DeviceTaintEffectNone,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			taint := healthEventToTaint(tc.monitor, tc.event)
			assert.Equal(t, tc.expectedKey, taint.Key)
			assert.Equal(t, tc.expectedValue, taint.Value)
			assert.Equal(t, tc.expectedEffect, taint.Effect)
		})
	}
}

func TestIsEventNonFatal(t *testing.T) {
	m := &nvmlDeviceHealthMonitor{
		skippedXids: map[uint64]bool{
			13: true,
			31: true,
			43: true,
		},
	}

	tests := []struct {
		name     string
		event    *DeviceHealthEvent
		expected bool
	}{
		{
			name: "skipped XID is non-fatal",
			event: &DeviceHealthEvent{
				EventType: HealthEventXID,
				EventData: 13,
			},
			expected: true,
		},
		{
			name: "non-skipped XID is fatal",
			event: &DeviceHealthEvent{
				EventType: HealthEventXID,
				EventData: 48,
			},
			expected: false,
		},
		{
			name: "GPU_LOST is always fatal",
			event: &DeviceHealthEvent{
				EventType: HealthEventGPULost,
			},
			expected: false,
		},
		{
			name: "unmonitored is not an XID event",
			event: &DeviceHealthEvent{
				EventType: HealthEventUnmonitored,
			},
			expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, m.IsEventNonFatal(tc.event))
		})
	}
}

func TestRegisterUnregisterDevicePlacement(t *testing.T) {
	parent := &GpuInfo{UUID: "GPU-parent-1", minor: 0}
	dynamicDev := &AllocatableDevice{MigDynamic: &MigSpec{Parent: parent}}
	m := &nvmlDeviceHealthMonitor{deviceByPlacement: make(devicePlacementMap)}

	m.RegisterDevicePlacement(parent.UUID, 3, 4, dynamicDev)
	require.Equal(t, dynamicDev, m.lookupDevicePlacement(parent.UUID, 3, 4))
	m.UnregisterDevicePlacement(parent.UUID, 3, 4)
	require.Nil(t, m.lookupDevicePlacement(parent.UUID, 3, 4))
}

func TestGetDevicePlacementMapPreservesStaticMIGRouting(t *testing.T) {
	parent := &GpuInfo{UUID: "GPU-parent-1", minor: 0}
	fullGPU := &AllocatableDevice{Gpu: parent}
	staticMIG := &AllocatableDevice{
		MigStatic: &MigDeviceInfo{
			parent: parent,
			gIInfo: &nvml.GpuInstanceInfo{Id: 3},
			cIInfo: &nvml.ComputeInstanceInfo{Id: 4},
		},
	}

	placements := getDevicePlacementMap(AllocatableDevices{
		"gpu":    fullGPU,
		"static": staticMIG,
	})

	require.Equal(t, fullGPU, placements.get(parent.UUID, FullGPUInstanceID, FullGPUInstanceID))
	require.Equal(t, staticMIG, placements.get(parent.UUID, 3, 4))
}

func TestHealthMonitorStartRequiresRegisteredEvents(t *testing.T) {
	m := &nvmlDeviceHealthMonitor{}
	require.ErrorContains(t, m.Start(context.Background()), "events have not been registered")
}

func TestRegisterEventsRegistersEachParentBeforeStart(t *testing.T) {
	device := &fakeHealthDevice{}
	eventSet := &fakeHealthEventSet{}
	nvmllib := &fakeHealthNVML{device: device, eventSet: eventSet}
	parent := &GpuInfo{UUID: "GPU-parent-1", minor: 0}
	m := &nvmlDeviceHealthMonitor{
		nvmllib:   nvmllib,
		unhealthy: make(chan *DeviceHealthEvent, 1),
		allocatableByParent: map[string][]*AllocatableDevice{
			parent.UUID: {&AllocatableDevice{Gpu: parent}},
		},
	}

	require.NoError(t, m.RegisterEvents())
	require.Equal(t, eventSet, m.eventSet)
	require.Equal(t, 1, nvmllib.initCalls)
	require.Equal(t, 1, nvmllib.eventSetCalls)
	require.Equal(t, 1, nvmllib.deviceGetCalls)
	require.Equal(t, 1, device.registerCalls)
}

func TestRegisterHealthPlacementWithoutRegistryIsNoop(t *testing.T) {
	state := &DeviceState{}
	require.NotPanics(t, func() {
		state.registerHealthPlacement("GPU-parent-1", 3, 4, &AllocatableDevice{})
	})
}

func TestRebuildHealthPlacementsFromCheckpoint(t *testing.T) {
	parent := &GpuInfo{UUID: "GPU-parent-1", minor: 0, pciBusID: "0000:01:00.0"}
	dynamicDev := &AllocatableDevice{MigDynamic: &MigSpec{Parent: parent}}
	perGPU := &PerGPUAllocatableDevices{
		allocatablesMap: map[PCIBusID]AllocatableDevices{
			parent.pciBusID: {"dynamic-1": dynamicDev},
		},
	}
	expected := &MigLiveTuple{ParentUUID: parent.UUID, GIID: 3, CIID: 4, MigUUID: "MIG-1"}
	cp := checkpointWithDynamicMIGClaims(expected, "dynamic-1", "claim-1")
	resolver := &fakeMigDeviceResolver{live: expected}
	m := &nvmlDeviceHealthMonitor{deviceByPlacement: make(devicePlacementMap)}

	err := rebuildHealthPlacementsFromCheckpoint(m, resolver, perGPU, cp)

	require.NoError(t, err)
	require.Len(t, resolver.requested, 1)
	assert.Equal(t, dynamicDev.MigDynamic.Tuple(), resolver.requested[0])
	require.Equal(t, dynamicDev, m.lookupDevicePlacement(parent.UUID, 3, 4))
}

func TestValidateMigIdentity(t *testing.T) {
	expected := &MigLiveTuple{ParentUUID: "GPU-parent-1", GIID: 3, CIID: 4, MigUUID: "MIG-1"}
	tests := []struct {
		name string
		live *MigLiveTuple
	}{
		{name: "parent UUID mismatch", live: &MigLiveTuple{ParentUUID: "GPU-parent-2", GIID: 3, CIID: 4, MigUUID: "MIG-1"}},
		{name: "GI mismatch", live: &MigLiveTuple{ParentUUID: "GPU-parent-1", GIID: 5, CIID: 4, MigUUID: "MIG-1"}},
		{name: "CI mismatch", live: &MigLiveTuple{ParentUUID: "GPU-parent-1", GIID: 3, CIID: 5, MigUUID: "MIG-1"}},
		{name: "MIG UUID mismatch", live: &MigLiveTuple{ParentUUID: "GPU-parent-1", GIID: 3, CIID: 4, MigUUID: "MIG-2"}},
	}

	require.NoError(t, validateMigIdentity(expected, expected))
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.ErrorContains(t, validateMigIdentity(expected, tc.live), "MIG identity mismatch")
		})
	}
}

func TestRebuildHealthPlacementsRejectsInvalidState(t *testing.T) {
	parent := &GpuInfo{UUID: "GPU-parent-1", minor: 0, pciBusID: "0000:01:00.0"}
	dynamicDev := &AllocatableDevice{MigDynamic: &MigSpec{Parent: parent}}
	perGPU := &PerGPUAllocatableDevices{
		allocatablesMap: map[PCIBusID]AllocatableDevices{
			parent.pciBusID: {"dynamic-1": dynamicDev},
		},
	}
	expected := &MigLiveTuple{ParentUUID: parent.UUID, GIID: 3, CIID: 4, MigUUID: "MIG-1"}

	t.Run("missing live device", func(t *testing.T) {
		m := &nvmlDeviceHealthMonitor{deviceByPlacement: make(devicePlacementMap)}
		err := rebuildHealthPlacementsFromCheckpoint(
			m, &fakeMigDeviceResolver{}, perGPU,
			checkpointWithDynamicMIGClaims(expected, "dynamic-1", "claim-1"),
		)
		require.ErrorContains(t, err, "is missing")
		require.Nil(t, m.lookupDevicePlacement(parent.UUID, 3, 4))
	})

	t.Run("reused tuple with different UUID", func(t *testing.T) {
		m := &nvmlDeviceHealthMonitor{deviceByPlacement: make(devicePlacementMap)}
		replacement := *expected
		replacement.MigUUID = "MIG-replacement"
		err := rebuildHealthPlacementsFromCheckpoint(
			m, &fakeMigDeviceResolver{live: &replacement}, perGPU,
			checkpointWithDynamicMIGClaims(expected, "dynamic-1", "claim-1"),
		)
		require.ErrorContains(t, err, "MIG identity mismatch")
		require.Nil(t, m.lookupDevicePlacement(parent.UUID, 3, 4))
	})

	t.Run("duplicate completed claims", func(t *testing.T) {
		m := &nvmlDeviceHealthMonitor{deviceByPlacement: make(devicePlacementMap)}
		err := rebuildHealthPlacementsFromCheckpoint(
			m, &fakeMigDeviceResolver{live: expected}, perGPU,
			checkpointWithDynamicMIGClaims(expected, "dynamic-1", "claim-1", "claim-2"),
		)
		require.ErrorContains(t, err, "resolve to the same live MIG device")
		require.Nil(t, m.lookupDevicePlacement(parent.UUID, 3, 4))
	})
}

func checkpointWithDynamicMIGClaims(concrete *MigLiveTuple, deviceName string, claimUIDs ...string) *Checkpoint {
	claims := make(PreparedClaimsByUIDV2)
	for _, claimUID := range claimUIDs {
		claims[claimUID] = PreparedClaimV2{
			CheckpointState: ClaimCheckpointStatePrepareCompleted,
			PreparedDevices: PreparedDevices{
				&PreparedDeviceGroup{
					Devices: PreparedDeviceList{
						{
							Mig: &PreparedMigDevice{
								Concrete: concrete,
								Device:   &CheckpointedDevice{DeviceName: deviceName},
							},
						},
					},
				},
			},
		}
	}
	return &Checkpoint{V2: &CheckpointV2{PreparedClaims: claims}}
}

func TestGetAllocatableByParent(t *testing.T) {
	parent1 := &GpuInfo{UUID: "GPU-parent-1", minor: 0}
	parent2 := &GpuInfo{UUID: "GPU-parent-2", minor: 1}
	fullGPU := &AllocatableDevice{Gpu: parent1}
	staticMIG := &AllocatableDevice{MigStatic: &MigDeviceInfo{parent: parent1}}
	dynamic1 := &AllocatableDevice{MigDynamic: &MigSpec{Parent: parent1}}
	dynamic2 := &AllocatableDevice{MigDynamic: &MigSpec{Parent: parent1}}
	dynamicOnly := &AllocatableDevice{MigDynamic: &MigSpec{Parent: parent2}}
	vfio := &AllocatableDevice{Vfio: &VfioDeviceInfo{UUID: "VFIO-1"}}

	byParent := getAllocatableByParent(AllocatableDevices{
		"gpu":          fullGPU,
		"static":       staticMIG,
		"dynamic-1":    dynamic1,
		"dynamic-2":    dynamic2,
		"dynamic-only": dynamicOnly,
		"vfio":         vfio,
	})

	require.Len(t, byParent, 2)
	assert.ElementsMatch(t, []*AllocatableDevice{fullGPU, staticMIG, dynamic1, dynamic2}, byParent[parent1.UUID])
	assert.ElementsMatch(t, []*AllocatableDevice{dynamicOnly}, byParent[parent2.UUID])
}

func TestSendHealthEventForAllDevicesWhilePlacementsChange(t *testing.T) {
	parent := &GpuInfo{UUID: "GPU-parent-1", minor: 0}
	fullGPU := &AllocatableDevice{Gpu: parent}
	liveDynamic := &AllocatableDevice{MigDynamic: &MigSpec{Parent: parent}}
	uninstantiatedDynamic := &AllocatableDevice{MigDynamic: &MigSpec{Parent: parent}}
	all := AllocatableDevices{
		"gpu":            fullGPU,
		"live":           liveDynamic,
		"uninstantiated": uninstantiatedDynamic,
	}
	m := &nvmlDeviceHealthMonitor{
		unhealthy:           make(chan *DeviceHealthEvent, 1),
		deviceByPlacement:   make(devicePlacementMap),
		allocatableByParent: getAllocatableByParent(all),
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range 100 {
			m.RegisterDevicePlacement(parent.UUID, 3, 4, liveDynamic)
			m.UnregisterDevicePlacement(parent.UUID, 3, 4)
		}
	}()

	for range 100 {
		m.sendHealthEventForAllDevices(HealthEventGPULost)
		event := <-m.unhealthy
		assert.Equal(t, HealthEventGPULost, event.EventType)
		assert.ElementsMatch(t, []*AllocatableDevice{fullGPU, liveDynamic, uninstantiatedDynamic}, event.Devices)
	}
	wg.Wait()
}
