---
title: 0001 — Per-node GPU publication profiles
linkTitle: Per-node GPU profiles
weight: 2
description: Startup-only per-node filtering of GPU ResourceSlice publication.
---

| Field          | Value |
|----------------|-------|
| Status         | provisional |
| Authors        | @guptaNswati |
| Created        | 2026-09-30 |
| Related issues | [#1067](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu/issues/1067) |

## Summary

Add an Alpha, startup-only mechanism for selecting which discovered `gpu`,
`mig`, and `vfio` devices the GPU kubelet plugin publishes on each node.
Named, versioned profiles come from a mounted ConfigMap and a DRA-owned node
label selects one profile. The default-off feature preserves existing mixed
publication behavior.

## Motivation

### Who is asking for this, and why?

Cluster administrators running standard containers alongside KubeVirt or Kata
passthrough workloads need to dedicate nodes to one workload type. Today,
enabling passthrough initially advertises full-GPU and VFIO representations of
the same hardware, leaving an allocation race before Prepare removes the
alternate representation.

### Goals

- Select one publication profile per node at plugin startup.
- Filter ResourceSlices without deleting internal state needed by
  Prepare, Unprepare, or checkpoint recovery.
- Preserve current behavior when the feature is disabled or the `mixed`
  profile is selected.
- Fail clearly for invalid profiles and checkpointed claims incompatible with
  the selected profile.

### Non-goals

- Live reload, sidecars, or automatic profile transitions.
- Claim-level sharing defaults, feature-gate overrides, or per-GPU rules.
- IMEX, ComputeDomain, vGPU, or confidential-container lifecycle management.
- Automatically draining nodes or proving that every API allocation has
  reached the local checkpoint.

## Why this belongs in the NVIDIA DRA driver

The NVIDIA GPU plugin owns discovery and ResourceSlice publication for the
`gpu`, `mig`, and `vfio` device types. Kubernetes DRA and CEL selectors cannot
prevent the driver from advertising conflicting representations. GPU Operator
may express workload intent, but the DRA driver must decide which of its own
devices to publish and preserve its local claim state.

## Proposal

### User-facing example

```yaml
featureGates:
  PerNodeGPUConfig: true

gpuDriverConfig:
  default: mixed
  nodeLabel: nvidia.com/dra-driver-gpu.config
  map:
    mixed: |-
      version: v1alpha1
      gpu:
        advertisedDeviceTypes: [gpu, mig, vfio]
    container: |-
      version: v1alpha1
      gpu:
        advertisedDeviceTypes: [gpu, mig]
    passthrough: |-
      version: v1alpha1
      gpu:
        advertisedDeviceTypes: [vfio]
```

After draining a node, select a profile and manually replace its plugin pod:

```bash
kubectl label node <node> --overwrite \
    nvidia.com/dra-driver-gpu.config=passthrough
kubectl -n <driver-namespace> delete pod <kubelet-plugin-pod-on-node>
```

### Affected components

- [ ] `api/` — CRDs, CRD fields, or ResourceClaim shape
- [x] `gpu-kubelet-plugin`
- [ ] `compute-domain-kubelet-plugin`
- [ ] `compute-domain-controller`
- [ ] `compute-domain-daemon`
- [ ] admission webhook
- [x] Helm chart (`deployments/helm`)
- [ ] CDI spec generation
- [ ] Metrics
- [ ] Kubelet-plugin checkpoint schema
- [x] Documentation
- [x] CI / testing

### Authoritative state owner

The mounted ConfigMap owns available profiles; the node label owns explicit
selection. The GPU kubelet plugin resolves both once at startup and owns the
effective in-memory publication policy. No component writes the label or
ConfigMap.

### Smallest valuable slice

The first slice supports only strict startup parsing and ResourceSlice
filtering. It mounts all named profiles, reads the current Node once, selects
the label value or Helm default, validates the profile, then publishes only
allowed device types.

## Design

### API changes

User-facing contracts are the Alpha `PerNodeGPUConfig` feature gate, the
`gpuDriverConfig.name`, `default`, `nodeLabel`, and `map` Helm values, the
`--driver-config-*` CLI flags and corresponding environment variables, the
`nvidia.com/dra-driver-gpu.config` node label, and the versioned profile YAML.
The ConfigMap mount path and in-memory filtering helpers are implementation
details. This proposal does not change a CRD, opaque ResourceClaim config,
ResourceSlice attribute, device name, or checkpoint schema.

### Configuration and selection

`DriverConfig` is a deployment configuration local to the GPU binary, not an
opaque `resource.nvidia.com/v1beta1.GpuConfig` claim API:

```go
type DriverConfig struct {
	Version string           `json:"version"`
	GPU     *GPUDriverConfig `json:"gpu,omitempty"`
}

type GPUDriverConfig struct {
	AdvertisedDeviceTypes []string `json:"advertisedDeviceTypes"`
}
```

Profiles use strict YAML decoding. `version` must be `v1alpha1`;
`advertisedDeviceTypes` must be non-empty and contain unique values from
`gpu`, `mig`, and `vfio`. Internal dynamic MIG devices map to public `mig`.
Unknown fields, versions, types, selected profiles, invalid label keys, and
unsafe profile paths fail startup.

When the label is absent, `gpuDriverConfig.default` is selected. An explicitly
empty label fails. With the gate disabled, the built-in exhaustive mixed
policy is used and explicit driver-config options fail validation.

### Publication and internal state

Filtering occurs only while constructing ResourceSlice input in the legacy,
combined DynamicMIG, and split DynamicMIG paths. The complete
`perGPUAllocatable` state remains unchanged for Prepare, Unprepare, health
handling, sibling rediscovery, and checkpoint recovery. An empty filtered pool
still publishes an empty slice.

### Checkpoint and transition safety

Before plugin registration or ResourceSlice publication, startup checks the
local checkpoint. Startup fails if an allocated or prepared device recorded in
the checkpoint has a type excluded by the selected profile. This is a
fail-closed guard, not a complete transition controller: allocations not yet
recorded locally remain possible.

Profile changes therefore require cordoning and draining the node, waiting for
Unprepare, changing the label or profile, manually deleting the plugin pod,
verifying ResourceSlices, and only then uncordoning. While the gate is enabled,
Helm sets the shared kubelet-plugin DaemonSet to `OnDelete`; profile and chart
changes do not roll pods automatically.

### Feature gate & graduation

`PerNodeGPUConfig` is Alpha and defaults to `false` in driver version 0.6.
Helm creates and mounts profile configuration only when enabled. Beta requires:

- agreed public configuration ownership and schema;
- tested composition with static MIG, DynamicMIG, passthrough, consumable
  shares, health taints, upgrade, downgrade, and restart recovery;
- an accepted transition design for allocated but not checkpointed claims;
- mock-NVML and real-GPU evidence across container and passthrough nodes.

### Upgrade & downgrade

Default upgrades are unchanged because the gate is off. Enabling the gate sets
`OnDelete`; administrators must drain and manually replace each plugin pod.
Downgrading or disabling requires a two-phase drained transition: select
`mixed` and replace pods while the gate remains enabled, then disable the gate
before uncordoning. No checkpoint format changes are introduced.

### Environment floor

There are no new Kubernetes, NVIDIA driver, GPU generation, or hardware
requirements. Individual device types retain their existing feature-gate,
driver, and hardware requirements.

### Test plan

- Unit: strict profile parsing, default and label selection with a fake client,
  disabled-gate option rejection, and unsafe path rejection.
- Publication: legacy, combined DynamicMIG, split DynamicMIG, empty pool, and
  non-mutating internal-state tests.
- Restart safety: compatible and incompatible checkpoint allocations and
  prepared devices.
- Helm: lint and render with the gate off and on; assert ConfigMap/mount/env
  absence by default and `OnDelete` plus profile wiring when enabled.
- Mock NVML: CPU-only multi-node profile selection and ResourceSlice inspection
  is deferred until the startup contract is accepted.
- Real GPU: passthrough binding and workload execution are deferred; this slice
  claims publication filtering only.

## Risks

- A scheduler allocation may exist before it reaches the local checkpoint.
- A wrong profile can intentionally leave a node with an empty advertised pool.
- `OnDelete` affects both containers in the shared kubelet-plugin DaemonSet and
  requires manual replacement for chart upgrades while enabled.
- Disabling the gate restores the configured update strategy and can roll pods;
  safety depends on the documented drained two-phase procedure.

## Alternatives

- Pod selectors and taints require every workload author to apply policy and do
  not remove conflicting ResourceSlice devices.
- Per-device node selection is not compatible with the current node-local pool
  model and does not express workload intent.
- NVIDIA k8s-device-plugin's named ConfigMap/default/label pattern is reused
  conceptually, but its schema and `nvidia.com/device-plugin.config` label
  remain device-plugin-specific.
- GPU Operator's `nvidia.com/gpu.workload.config` remains workload and operand
  intent; direct mapping may be added after ownership is agreed.

## Drawbacks

This Alpha slice adds a public Helm/configuration surface without live
reconciliation. Safe use is operationally heavy because every transition and
upgrade requires a drain and manual pod replacement.

## Open questions

1. Should a future stable selector map GPU Operator workload labels or remain
   DRA-specific?
2. How should transitions account for allocated claims not yet checkpointed?
3. Should the versioned schema move from the binary package into a dedicated
   importable configuration API?
4. Can a future controller or sidecar automate transitions without weakening
   Prepare/Unprepare and checkpoint guarantees?
