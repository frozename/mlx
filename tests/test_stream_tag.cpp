// Copyright © 2026 Apple Inc.
//
// Unit tests for the advisory per-stream tag. The tag is metadata used by
// higher layers (oMLX, mlx-lm) to record model affinity; it lives in a
// side-registry keyed by stream.index (see mlx/stream.h) rather than on the
// Stream struct, so the handle stays small and trivially-copyable on the hot
// Metal completion-handler path. Backend keying off the tag (per-stream
// MTL::CommandQueue / ResidencySet) lands as follow-up PRs.

#include <optional>
#include <string>
#include <type_traits>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/stream.h"

using namespace mlx::core;

// The point of the side-registry: Stream must stay a lightweight,
// trivially-copyable handle (it is captured by value in Metal completion
// lambdas). Guard against anyone re-introducing a non-trivial member.
static_assert(
    std::is_trivially_copyable_v<Stream>,
    "Stream must stay trivially-copyable; store per-stream metadata in a "
    "side-registry (set_stream_tag), not on the handle.");

TEST_CASE("stream tag defaults to empty") {
  Stream s(7001, default_device());
  CHECK_FALSE(stream_tag(s).has_value());
}

TEST_CASE("stream tag round-trips and is shared by index across copies") {
  Stream s(7002, default_device());
  set_stream_tag(s, "granite-4.1-3b");
  REQUIRE(stream_tag(s).has_value());
  CHECK_EQ(*stream_tag(s), "granite-4.1-3b");

  // A copy carries the same index, so it resolves to the same registry entry.
  Stream copy = s;
  REQUIRE(stream_tag(copy).has_value());
  CHECK_EQ(*stream_tag(copy), "granite-4.1-3b");

  clear_stream_tag(s);
  CHECK_FALSE(stream_tag(s).has_value());
}

TEST_CASE("stream equality ignores tag") {
  // Two streams with the same {index, device} are identical regardless of
  // tag — the tag is advisory metadata, not load-bearing for identity.
  Stream a(7003, default_device());
  Stream b(7003, default_device());
  set_stream_tag(b, "qwen3-8b");
  CHECK(a == b);
  clear_stream_tag(b);
}

TEST_CASE("stream tag is assignable and clearable post-construction") {
  Stream s(7004, default_device());
  CHECK_FALSE(stream_tag(s).has_value());
  set_stream_tag(s, "late-tagged");
  REQUIRE(stream_tag(s).has_value());
  CHECK_EQ(*stream_tag(s), "late-tagged");
  clear_stream_tag(s);
  CHECK_FALSE(stream_tag(s).has_value());
}
