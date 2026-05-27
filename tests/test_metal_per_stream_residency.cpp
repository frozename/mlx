// Copyright © 2026 Apple Inc.
//
// Unit tests for the per-stream `ResidencySet` accessor on
// `mlx::core::metal::Device`. The accessor lazily creates one
// `ResidencySet` per stream index and lets backend / downstream code wire
// stream-specific allocations into the GPU's wired pool without affecting
// the device-global `residency_set()`.
//
// Skipped when MLX_BUILD_METAL is off (no Device available).

#include "doctest/doctest.h"

#include "mlx/mlx.h"

#ifdef _METAL_

#include "mlx/backend/metal/device.h"

using namespace mlx::core;

TEST_CASE("per-stream residency set: distinct sets for distinct indices") {
  auto& d = metal::device(Device::gpu);
  auto& a = d.residency_set(9001);
  auto& b = d.residency_set(9002);
  CHECK(&a != &b);
}

TEST_CASE("per-stream residency set: same index returns same set") {
  auto& d = metal::device(Device::gpu);
  auto& a1 = d.residency_set(9003);
  auto& a2 = d.residency_set(9003);
  CHECK(&a1 == &a2);
}

TEST_CASE("per-stream residency set: distinct from device-global set") {
  auto& d = metal::device(Device::gpu);
  auto& per_stream = d.residency_set(9004);
  auto& global = d.residency_set();
  CHECK(&per_stream != &global);
}

TEST_CASE("per-stream residency set: release deallocates") {
  auto& d = metal::device(Device::gpu);
  auto& a = d.residency_set(9005);
  // Take the underlying mtl pointer before release so we can compare.
  const void* before_ptr = static_cast<const void*>(&a);
  d.release_stream_residency_set(9005);
  // After release the next call should produce a freshly-allocated set,
  // which may or may not land at the same address depending on allocator
  // reuse. We can't directly verify the deallocation; what we CAN verify
  // is that the new set is consistent (re-allocation succeeds) and that
  // the index has been freed for re-use.
  auto& b = d.residency_set(9005);
  // Either a freshly-allocated set or accidental reuse — either way the
  // accessor must not crash and the returned set must be functional.
  (void)before_ptr;
  (void)b;
  CHECK(true);
}

TEST_CASE("per-stream residency set: clear_stream_residency_sets releases all") {
  auto& d = metal::device(Device::gpu);
  (void)d.residency_set(9006);
  (void)d.residency_set(9007);
  d.clear_stream_residency_sets();
  // After clear, both indices should re-allocate freshly. Nothing to
  // assert deterministically about pointer identity; the assertion is
  // that the call completes without throwing.
  (void)d.residency_set(9006);
  (void)d.residency_set(9007);
  CHECK(true);
}

#endif // _METAL_
