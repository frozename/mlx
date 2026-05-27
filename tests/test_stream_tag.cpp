// Copyright © 2026 Apple Inc.
//
// Unit tests for the optional `Stream::tag` field. The field is advisory
// metadata used by higher layers (oMLX, mlx-lm) to record model affinity.
// Backend keying off the tag (per-stream MTL::CommandQueue / ResidencySet)
// lands as follow-up PRs.

#include <optional>
#include <string>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/stream.h"

using namespace mlx::core;

TEST_CASE("stream tag defaults to nullopt") {
  Stream s(7, default_device());
  CHECK_FALSE(s.tag.has_value());
}

TEST_CASE("stream tag round-trips through copy") {
  Stream s(7, default_device(), std::string("granite-4.1-3b"));
  REQUIRE(s.tag.has_value());
  CHECK_EQ(*s.tag, "granite-4.1-3b");

  Stream copy = s;
  REQUIRE(copy.tag.has_value());
  CHECK_EQ(*copy.tag, "granite-4.1-3b");
}

TEST_CASE("stream equality ignores tag") {
  // Two streams with the same {index, device} are considered identical
  // even if their tags differ. The tag is advisory metadata, not
  // load-bearing for stream identity.
  Stream untagged(9, default_device());
  Stream tagged(9, default_device(), std::string("qwen3-8b"));
  CHECK(untagged == tagged);
}

TEST_CASE("stream tag is assignable post-construction") {
  Stream s(11, default_device());
  CHECK_FALSE(s.tag.has_value());
  s.tag = std::string("late-tagged");
  REQUIRE(s.tag.has_value());
  CHECK_EQ(*s.tag, "late-tagged");
}
