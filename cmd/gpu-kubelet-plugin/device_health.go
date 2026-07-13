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
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/klog/v2"
)

const (
	FullGPUInstanceID uint32 = 0xFFFFFFFF
)

const (
	TaintKeyXID         = DriverName + "/xid"
	TaintKeyGPULost     = DriverName + "/gpu-lost"
	TaintKeyUnmonitored = DriverName + "/unmonitored"
)

// DeviceHealthEventType classifies the category of health event detected by
// the NVML health monitor.
type DeviceHealthEventType string

const (
	HealthEventXID         DeviceHealthEventType = "xid"
	HealthEventGPULost     DeviceHealthEventType = "gpu-lost"
	HealthEventUnmonitored DeviceHealthEventType = "unmonitored"
)

// DeviceHealthEvent carries a typed health notification from the NVML health
// monitor to the driver's event handler, enabling the driver to set the
// appropriate DRA device taint per the Option A schema (KEP-5055).
// Devices is a batch: for GPU_LOST and unmonitored events where all affected devices
// are aggregated into a single event so the consumer applies one ResourceSlice
// update instead of N.
type DeviceHealthEvent struct {
	Devices   []*AllocatableDevice
	EventType DeviceHealthEventType
	// inspired by NVML Event type and only meaningful for xid errors.
	// may have to create a custom type based on future device-api
	EventData uint64
}

// healthEventToTaint maps a DeviceHealthEvent to the corresponding DRA
// DeviceTaint using the Option A taint key schema: one key per health
// dimension under the gpu.nvidia.com domain.
func healthEventToTaint(monitor deviceHealthMonitor, event *DeviceHealthEvent) *resourceapi.DeviceTaint {
	switch event.EventType {
	case HealthEventXID:
		effect := resourceapi.DeviceTaintEffectNoSchedule
		if monitor != nil && monitor.IsEventNonFatal(event) {
			effect = resourceapi.DeviceTaintEffectNone
		}
		return &resourceapi.DeviceTaint{
			Key:    TaintKeyXID,
			Value:  strconv.FormatUint(event.EventData, 10),
			Effect: effect,
		}
	case HealthEventGPULost:
		return &resourceapi.DeviceTaint{
			Key:    TaintKeyGPULost,
			Effect: resourceapi.DeviceTaintEffectNoSchedule,
		}
	case HealthEventUnmonitored:
		return &resourceapi.DeviceTaint{
			Key:    TaintKeyUnmonitored,
			Effect: resourceapi.DeviceTaintEffectNone,
		}
	default:
		klog.Errorf("Unknown health event type %q, defaulting to unmonitored taint", event.EventType)
		return &resourceapi.DeviceTaint{
			Key:    TaintKeyUnmonitored,
			Effect: resourceapi.DeviceTaintEffectNone,
		}
	}
}

// For a MIG device the placement is defined by the 3-tuple <parent UUID, GI, CI>.
// For a full device the returned 3-tuple is the device's uuid and (FullGPUInstanceID) 0xFFFFFFFF for the other two elements.
type devicePlacementMap map[string]map[uint32]map[uint32]*AllocatableDevice

// devicePlacementRegistry tracks live (parentUUID, GI, CI) -> allocatable device
// mappings for dynamically created MIG partitions. Static MIG and full GPU
// entries are populated at monitor startup; dynamic MIG entries are added and
// removed at prepare/unprepare time.
type devicePlacementRegistry interface {
	RegisterDevicePlacement(parentUUID string, gi, ci uint32, dev *AllocatableDevice)
	UnregisterDevicePlacement(parentUUID string, gi, ci uint32)
}

// migDeviceResolver finds the concrete MIG device currently implementing an
// abstract Dynamic MIG profile and placement.
type migDeviceResolver interface {
	FindMigDevBySpec(*MigSpecTuple) (*MigLiveTuple, error)
}

type nvmlDeviceHealthMonitor struct {
	nvmllib           nvml.Interface
	eventSet          nvml.EventSet
	unhealthy         chan *DeviceHealthEvent
	deviceByPlacement devicePlacementMap
	skippedXids       map[uint64]bool
	wg                sync.WaitGroup
	placementMu       sync.RWMutex
	// allocatableByParent is the immutable inventory of all advertised devices
	// grouped by physical GPU. Unlike deviceByPlacement, it includes dynamic MIG
	// devices which have not yet been prepared.
	allocatableByParent map[string][]*AllocatableDevice
}

func newNvmlDeviceHealthMonitor(config *Config, perGPUAllocatable *PerGPUAllocatableDevices, nvdevlib *deviceLib) (*nvmlDeviceHealthMonitor, error) {
	if nvdevlib.nvmllib == nil {
		return nil, fmt.Errorf("nvml library is nil")
	}
	if ret := nvdevlib.nvmllib.Init(); ret != nvml.SUCCESS {
		return nil, fmt.Errorf("failed to initialize NVML: %v", ret)
	}
	defer func() {
		_ = nvdevlib.nvmllib.Shutdown()
	}()

	if perGPUAllocatable == nil {
		return nil, fmt.Errorf("perGPUAllocatable is nil")
	}
	all := perGPUAllocatable.GetAllDevices()
	placementMap := getDevicePlacementMap(all)
	m := &nvmlDeviceHealthMonitor{
		nvmllib:             nvdevlib.nvmllib,
		unhealthy:           make(chan *DeviceHealthEvent, len(all)),
		deviceByPlacement:   placementMap,
		allocatableByParent: getAllocatableByParent(all),
		skippedXids:         xidsToSkip(config.flags.additionalXidsToIgnore),
	}
	return m, nil
}

// RegisterEvents creates the NVML event set and starts recording events for
// every physical parent GPU. It intentionally does not start the wait loop, so
// registration can complete before the kubelet server accepts requests.
func (m *nvmlDeviceHealthMonitor) RegisterEvents() (rerr error) {
	if ret := m.nvmllib.Init(); ret != nvml.SUCCESS {
		return fmt.Errorf("failed to initialize NVML: %v", ret)
	}

	defer func() {
		if rerr != nil {
			_ = m.nvmllib.Shutdown()
		}
	}()

	klog.V(4).Info("creating NVML events for device health monitor")
	eventSet, ret := m.nvmllib.EventSetCreate()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("failed to create event set: %w", ret)
	}

	m.eventSet = eventSet

	klog.V(4).Info("registering NVML events for device health monitor")
	m.registerEventsForDevices()
	return nil
}

// Start launches the NVML event wait loop after RegisterEvents has completed.
func (m *nvmlDeviceHealthMonitor) Start(ctx context.Context) error {
	if m.eventSet == nil {
		return fmt.Errorf("NVML events have not been registered")
	}
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		m.run(ctx)
	}()

	klog.V(4).Info("started device health monitoring")
	return nil
}

// registerEventsForDevices registers NVML events once per physical GPU. NVML
// (XID) events are delivered at the physical-GPU level (with GI/CI fields on
// the event for MIG), so registration is on the parent GPU handle and covers
// all of its (static or dynamic) partitions.
func (m *nvmlDeviceHealthMonitor) registerEventsForDevices() {
	eventMask := uint64(nvml.EventTypeXidCriticalError | nvml.EventTypeDoubleBitEccError | nvml.EventTypeSingleBitEccError)

	for parentUUID, devices := range m.allocatableByParent {
		gpu, ret := m.nvmllib.DeviceGetHandleByUUID(parentUUID)
		if ret != nvml.SUCCESS {
			klog.Warningf("Unable to get device handle from UUID[%s]: %v; marking devices as unmonitored", parentUUID, ret)
			m.sendBatchedHealthEvent(devices, HealthEventUnmonitored)
			continue
		}

		supportedEvents, ret := gpu.GetSupportedEventTypes()
		if ret != nvml.SUCCESS {
			klog.Warningf("unable to determine the supported events for %s: %v; marking devices as unmonitored", parentUUID, ret)
			m.sendBatchedHealthEvent(devices, HealthEventUnmonitored)
			continue
		}

		ret = gpu.RegisterEvents(eventMask&supportedEvents, m.eventSet)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			klog.Warningf("Device %v is too old to support healthchecking.", parentUUID)
			m.sendBatchedHealthEvent(devices, HealthEventUnmonitored)
		} else if ret != nvml.SUCCESS {
			klog.Warningf("unable to register events for %s: %v; marking devices as unmonitored", parentUUID, ret)
			m.sendBatchedHealthEvent(devices, HealthEventUnmonitored)
		}
	}
}

func (m *nvmlDeviceHealthMonitor) Stop() {
	if m == nil {
		return
	}
	klog.V(6).Info("stopping health monitor")

	m.wg.Wait()

	if ret := m.eventSet.Free(); ret != nvml.SUCCESS {
		klog.Warningf("failed to unset events: %v", ret)
	}

	if ret := m.nvmllib.Shutdown(); ret != nvml.SUCCESS {
		klog.Warningf("failed to shutdown NVML: %v", ret)
	}
	close(m.unhealthy)
}

func (m *nvmlDeviceHealthMonitor) run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			klog.V(6).Info("Stopping event-driven GPU health monitor...")
			return
		default:
			event, ret := m.eventSet.Wait(5000) // timeout in 5000 ms.
			if ret == nvml.ERROR_TIMEOUT {
				continue
			}
			// not all return errors are handled as currently there is no proper way to process these errors other than marking all devices healthy.
			// Ref doc: [https://docs.nvidia.com/deploy/nvml-api/group__nvmlEvents.html#group__nvmlEvents_1g9714b0ca9a34c7a7780f87fee16b205c].
			if ret != nvml.SUCCESS {
				if ret == nvml.ERROR_GPU_IS_LOST {
					klog.Warningf("GPU is lost error: %v; Tainting all devices with %s", ret, TaintKeyGPULost)
					m.sendHealthEventForAllDevices(HealthEventGPULost)
					continue
				}
				klog.V(6).Infof("Error waiting for NVML event: %v. Retrying...", ret)
				continue
			}

			// TODO: check why other supported types are not considered?
			eType := event.EventType
			xid := event.EventData
			gi := event.GpuInstanceId
			ci := event.ComputeInstanceId
			if eType != nvml.EventTypeXidCriticalError {
				klog.V(6).Infof("Skipping non-nvmlEventTypeXidCriticalError event: Data=%d, Type=%d, GI=%d, CI=%d", xid, eType, gi, ci)
				continue
			}

			klog.V(4).Infof("Processing event XID=%d event", xid)
			// this seems an extreme action.
			// should we just log the error and proceed anyway.
			// TODO: look into how to properly handle this error.
			eventUUID, ret := event.Device.GetUUID()
			if ret != nvml.SUCCESS {
				klog.Warningf("Failed to determine uuid for event %v: %v; Tainting all devices with %s", event, ret, TaintKeyGPULost)
				m.sendHealthEventForAllDevices(HealthEventGPULost)
				continue
			}
			affectedDevice := m.lookupDevicePlacement(eventUUID, gi, ci)
			if affectedDevice == nil {
				klog.V(6).Infof("Ignoring event for unexpected device (UUID:%s, GI:%d, CI:%d)", eventUUID, gi, ci)
				continue
			}

			klog.V(4).Infof("Sending XID=%d health event for device %s", xid, affectedDevice.CanonicalName())
			m.unhealthy <- &DeviceHealthEvent{
				Devices:   []*AllocatableDevice{affectedDevice},
				EventType: HealthEventXID,
				EventData: xid,
			}
		}
	}
}

func (m *nvmlDeviceHealthMonitor) Unhealthy() <-chan *DeviceHealthEvent {
	return m.unhealthy
}

// sendHealthEventForAllDevices aggregates every device across all GPUs into a
// single batched DeviceHealthEvent so the consumer makes one ResourceSlice
// update.
func (m *nvmlDeviceHealthMonitor) sendHealthEventForAllDevices(eventType DeviceHealthEventType) {
	var devices []*AllocatableDevice
	for _, parentDevices := range m.allocatableByParent {
		devices = append(devices, parentDevices...)
	}
	m.sendBatchedHealthEvent(devices, eventType)
}

// sendBatchedHealthEvent sends a single DeviceHealthEvent containing all
// affected devices.
func (m *nvmlDeviceHealthMonitor) sendBatchedHealthEvent(devices []*AllocatableDevice, eventType DeviceHealthEventType) {
	if len(devices) == 0 {
		return
	}
	m.unhealthy <- &DeviceHealthEvent{
		Devices:   devices,
		EventType: eventType,
	}
}

// The purpose of this function is to allow for a O(1) lookup of
// AllocatableDevice by ([parent]UUID, GI, CI) when processing health events. It
// currently assumes that this is constant for the lifetime of the healthchecker
// which does not hold for Dynamic MIG. This will have to be resolved once we
// support device health checking with dynamic MIG.
func getDevicePlacementMap(allocatable AllocatableDevices) devicePlacementMap {
	placementMap := make(devicePlacementMap)

	for _, d := range allocatable {
		var parentUUID string
		var giID, ciID uint32

		switch d.Type() {
		case GpuDeviceType:
			parentUUID = d.UUID()
			if parentUUID == "" {
				continue
			}
			giID = FullGPUInstanceID
			ciID = FullGPUInstanceID

		case MigStaticDeviceType:
			parentUUID = d.MigStatic.parent.UUID

			// Note(JP): it's unclear why we handle this case here (and why do
			// we think this can be empty?)
			if parentUUID == "" {
				continue
			}
			giID = d.MigStatic.gIInfo.Id
			ciID = d.MigStatic.cIInfo.Id

		default:
			// This may be a problem; and should be logged
			klog.V(4).Infof("getDevicePlacementMap: skipping device with type: %s", d.Type())
			continue
		}
		placementMap.addDevice(parentUUID, giID, ciID, d)
	}
	return placementMap
}

func (p devicePlacementMap) addDevice(parentUUID string, giID uint32, ciID uint32, d *AllocatableDevice) {
	if _, ok := p[parentUUID]; !ok {
		p[parentUUID] = make(map[uint32]map[uint32]*AllocatableDevice)
	}
	if _, ok := p[parentUUID][giID]; !ok {
		p[parentUUID][giID] = make(map[uint32]*AllocatableDevice)
	}
	p[parentUUID][giID][ciID] = d
}

func (p devicePlacementMap) get(uuid string, gi, ci uint32) *AllocatableDevice {
	giMap, ok := p[uuid]
	if !ok {
		return nil
	}

	ciMap, ok := giMap[gi]
	if !ok {
		return nil
	}
	return ciMap[ci]
}

// getAdditionalXids returns a list of additional Xids to skip from the specified string.
// The input is treated as a comma-separated string and all valid uint64 values are considered as Xid values.
// Invalid values are ignored.
// TODO: add list of EXPLICIT XIDs from [https://github.com/NVIDIA/k8s-device-plugin/pull/1443].
func getAdditionalXids(input string) []uint64 {
	if input == "" {
		return nil
	}

	var additionalXids []uint64
	klog.V(6).Infof("Creating a list of additional xids to ignore: [%s]", input)
	for _, additionalXid := range strings.Split(input, ",") {
		trimmed := strings.TrimSpace(additionalXid)
		if trimmed == "" {
			continue
		}
		xid, err := strconv.ParseUint(trimmed, 10, 64)
		if err != nil {
			klog.V(6).Infof("Ignoring malformed Xid value %v: %v", trimmed, err)
			continue
		}
		additionalXids = append(additionalXids, xid)
	}

	return additionalXids
}

func xidsToSkip(additionalXids string) map[uint64]bool {
	// Add the list of hardcoded disabled (ignored) XIDs:
	// http://docs.nvidia.com/deploy/xid-errors/index.html#topic_4
	// Application errors: the GPU should still be healthy.
	ignoredXids := []uint64{
		13,  // Graphics Engine Exception
		31,  // GPU memory page fault
		43,  // GPU stopped processing
		45,  // Preemptive cleanup, due to previous errors
		68,  // Video processor exception
		109, // Context Switch Timeout Error
	}

	skippedXids := make(map[uint64]bool)
	for _, id := range ignoredXids {
		skippedXids[id] = true
	}

	for _, additionalXid := range getAdditionalXids(additionalXids) {
		skippedXids[additionalXid] = true
	}
	return skippedXids
}

// IsEventNonFatal evaluates whether a hardware event is considered an application-level
// warning (None) rather than a critical hardware failure (NoSchedule).
// Currently, it only checks for XID events.
func (m *nvmlDeviceHealthMonitor) IsEventNonFatal(event *DeviceHealthEvent) bool {
	if event.EventType == HealthEventXID {
		return m.skippedXids[event.EventData]
	}
	return false
}

func getAllocatableByParent(allocatable AllocatableDevices) map[string][]*AllocatableDevice {
	byParent := make(map[string][]*AllocatableDevice)
	for _, d := range allocatable {
		var parentUUID string
		switch d.Type() {
		case GpuDeviceType:
			parentUUID = d.Gpu.UUID
		case MigStaticDeviceType:
			parentUUID = d.MigStatic.parent.UUID
		case MigDynamicDeviceType:
			parentUUID = d.MigDynamic.Parent.UUID
		default:
			continue
		}
		if parentUUID == "" {
			continue
		}
		byParent[parentUUID] = append(byParent[parentUUID], d)
	}
	return byParent
}

func (m *nvmlDeviceHealthMonitor) RegisterDevicePlacement(parentUUID string, gi, ci uint32, dev *AllocatableDevice) {
	if m == nil || dev == nil || parentUUID == "" {
		return
	}
	m.placementMu.Lock()
	defer m.placementMu.Unlock()
	m.deviceByPlacement.addDevice(parentUUID, gi, ci, dev)
}

func (m *nvmlDeviceHealthMonitor) UnregisterDevicePlacement(parentUUID string, gi, ci uint32) {
	if m == nil || parentUUID == "" {
		return
	}
	m.placementMu.Lock()
	defer m.placementMu.Unlock()
	giMap, ok := m.deviceByPlacement[parentUUID]
	if !ok {
		return
	}
	ciMap, ok := giMap[gi]
	if !ok {
		return
	}
	delete(ciMap, ci)
	if len(ciMap) == 0 {
		delete(giMap, gi)
	}
	if len(giMap) == 0 {
		delete(m.deviceByPlacement, parentUUID)
	}
}

func (m *nvmlDeviceHealthMonitor) lookupDevicePlacement(parentUUID string, gi, ci uint32) *AllocatableDevice {
	m.placementMu.RLock()
	defer m.placementMu.RUnlock()
	return m.deviceByPlacement.get(parentUUID, gi, ci)
}

// rebuildHealthPlacementsFromCheckpoint validates every completed Dynamic MIG
// claim against live NVML state before registering any placement. Validation is
// fail-closed: no checkpoint data is rewritten and no replacement device is
// adopted when profile, placement, tuple, or MIG UUID differs.
func rebuildHealthPlacementsFromCheckpoint(registry devicePlacementRegistry, resolver migDeviceResolver, perGPU *PerGPUAllocatableDevices, cp *Checkpoint) error {
	if registry == nil || cp == nil || cp.V2 == nil {
		return nil
	}
	if resolver == nil {
		return fmt.Errorf("MIG device resolver is nil")
	}
	if perGPU == nil {
		return fmt.Errorf("per-GPU allocatable device inventory is nil")
	}

	type placement struct {
		parentUUID string
		gi         uint32
		ci         uint32
		device     *AllocatableDevice
	}
	type migIdentityKey struct {
		parentUUID string
		gi         int
		ci         int
		migUUID    string
	}

	var placements []placement
	owners := make(map[migIdentityKey]string)
	for claimUID, pc := range cp.V2.PreparedClaims {
		if pc.CheckpointState != ClaimCheckpointStatePrepareCompleted {
			continue
		}
		for _, group := range pc.PreparedDevices {
			for _, pd := range group.Devices {
				if pd.Mig == nil {
					continue
				}
				if pd.Mig.Concrete == nil || pd.Mig.Device == nil {
					return fmt.Errorf("completed claim %s has incomplete MIG checkpoint data", claimUID)
				}
				allocatable := perGPU.GetAllocatableDevice(DeviceName(pd.Mig.Device.DeviceName))
				if allocatable == nil {
					return fmt.Errorf("completed claim %s references device %s but no allocatable device was found",
						claimUID, pd.Mig.Device.DeviceName)
				}
				if allocatable.Type() != MigDynamicDeviceType {
					continue
				}

				live, err := resolver.FindMigDevBySpec(allocatable.MigDynamic.Tuple())
				if err != nil {
					return fmt.Errorf("resolve live MIG device for completed claim %s (%s): %w",
						claimUID, pd.Mig.Device.DeviceName, err)
				}
				if live == nil {
					return fmt.Errorf("live MIG device for completed claim %s (%s) is missing",
						claimUID, pd.Mig.Device.DeviceName)
				}
				if err := validateMigIdentity(pd.Mig.Concrete, live); err != nil {
					return fmt.Errorf("completed claim %s (%s): %w",
						claimUID, pd.Mig.Device.DeviceName, err)
				}

				key := migIdentityKey{live.ParentUUID, live.GIID, live.CIID, live.MigUUID}
				if owner, exists := owners[key]; exists {
					return fmt.Errorf("completed claims %s and %s resolve to the same live MIG device %s at parentUUID=%s GI=%d CI=%d",
						owner, claimUID, live.MigUUID, live.ParentUUID, live.GIID, live.CIID)
				}
				owners[key] = claimUID
				placements = append(placements, placement{
					parentUUID: live.ParentUUID,
					gi:         uint32(live.GIID),
					ci:         uint32(live.CIID),
					device:     allocatable,
				})
			}
		}
	}

	for _, p := range placements {
		registry.RegisterDevicePlacement(p.parentUUID, p.gi, p.ci, p.device)
	}
	return nil
}

// validateMigIdentity requires the live device to match the checkpoint's
// physical parent, GI, CI, and MIG UUID exactly.
func validateMigIdentity(expected, live *MigLiveTuple) error {
	if expected == nil || live == nil {
		return fmt.Errorf("cannot validate nil MIG identity (expected=%v live=%v)", expected != nil, live != nil)
	}
	if expected.ParentUUID != live.ParentUUID ||
		expected.GIID != live.GIID ||
		expected.CIID != live.CIID ||
		expected.MigUUID != live.MigUUID {
		return fmt.Errorf(
			"MIG identity mismatch: expected parentUUID=%s GI=%d CI=%d migUUID=%s, got parentUUID=%s GI=%d CI=%d migUUID=%s",
			expected.ParentUUID, expected.GIID, expected.CIID, expected.MigUUID,
			live.ParentUUID, live.GIID, live.CIID, live.MigUUID,
		)
	}
	return nil
}
