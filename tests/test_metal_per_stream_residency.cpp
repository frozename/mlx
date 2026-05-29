// Copyright © 2026 Apple Inc.
//
// Unit tests for the per-stream `ResidencySet` accessor on
// `mlx::core::metal::Device`. The accessor lazily creates one
// `ResidencySet` per stream index and lets backend / downstream code wire
// stream-specific allocations into the GPU's wired pool without affecting
// the device-global `residency_set()`.
//
// Compiled only when MLX_BUILD_METAL is on — gated in tests/CMakeLists.txt,
// not via an in-source `#ifdef _METAL_` (that macro is defined for downstream
// consumers via MLXConfig.cmake, not for the in-tree tests target).

#include "doctest/doctest.h"

#include "mlx/mlx.h"
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

TEST_CASE("per-stream residency set: re-allocation after release is functional") {
  auto& d = metal::device(Device::gpu);
  (void)d.residency_set(9005);
  d.release_stream_residency_set(9005);
  // After release the index re-allocates a fresh set. We can't assert
  // deallocation directly (allocator reuse may land the new set at the same
  // address), but we CAN assert the accessor stays functional: the
  // re-allocated set is stable across calls and remains distinct from the
  // device-global set.
  auto& b = d.residency_set(9005);
  CHECK(&b == &d.residency_set(9005)); // stable / idempotent after realloc
  CHECK(&b != &d.residency_set());     // still distinct from device-global
}

TEST_CASE("per-stream residency set: clear_stream_residency_sets releases all") {
  auto& d = metal::device(Device::gpu);
  (void)d.residency_set(9006);
  (void)d.residency_set(9007);
  d.clear_stream_residency_sets();
  // After clear, both indices re-allocate fresh, functional sets: distinct
  // per index, distinct from the device-global set, and stable across calls.
  auto& r6 = d.residency_set(9006);
  auto& r7 = d.residency_set(9007);
  CHECK(&r6 != &r7);                    // distinct indices -> distinct sets
  CHECK(&r6 != &d.residency_set());     // distinct from device-global
  CHECK(&r6 == &d.residency_set(9006)); // stable after re-allocation
}
