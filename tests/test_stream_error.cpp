// Copyright © 2026 Apple Inc.
//
// Unit tests for the per-stream error stash that lets Metal completion
// handlers — which run on libdispatch and cannot throw across a dispatch
// block — record an error to be re-thrown on the caller thread at the next
// synchronous waitpoint. See mlx-explore/mlx#2670.
//
// These tests do not require a Metal device: the scheduler error stash is
// backend-agnostic. They use CPU streams with explicit high-numbered indices
// that cannot collide with streams allocated sequentially from 0 by
// new_stream() elsewhere in the same test process. Each test drains the
// errors it stashes so it leaves no global state for the next one.

#include <stdexcept>
#include <string>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/scheduler.h"

using namespace mlx::core;

namespace {

// Build a detached Stream with an explicit index, bypassing new_stream() so a
// test can target a specific index in isolation.
Stream make_stream(int index) {
  return Stream{index, Device::cpu};
}

// Run a waitpoint; report whether it threw a runtime_error whose message
// contains `needle`.
bool throws_with(const Stream& s, const std::string& needle) {
  try {
    scheduler::throw_if_stream_error(s);
  } catch (const std::runtime_error& e) {
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

// Run a waitpoint; report whether it threw anything at all.
bool throws_any(const Stream& s) {
  try {
    scheduler::throw_if_stream_error(s);
  } catch (...) {
    return true;
  }
  return false;
}

} // namespace

TEST_CASE(
    "stashed stream error is rethrown once at the next waitpoint" *
    doctest::test_suite("[stream-error]")) {
  auto s = make_stream(9'000'001);

  scheduler::stash_stream_error(
      s, std::make_exception_ptr(std::runtime_error("boom on 9000001")));

  // The next waitpoint surfaces the stashed error...
  CHECK(throws_with(s, "boom on 9000001"));
  // ...and consumes it: a second waitpoint on the same stream is a no-op.
  CHECK_FALSE(throws_any(s));
}

TEST_CASE(
    "first error wins until the pending one is consumed" *
    doctest::test_suite("[stream-error]")) {
  auto s = make_stream(9'000'002);

  scheduler::stash_stream_error(
      s, std::make_exception_ptr(std::runtime_error("first")));
  // A second error while one is still pending is dropped, not overwritten.
  scheduler::stash_stream_error(
      s, std::make_exception_ptr(std::runtime_error("second")));

  CHECK(throws_with(s, "first"));
  // Only one error was ever stored; the stream is now clean.
  CHECK_FALSE(throws_any(s));
}

TEST_CASE(
    "throw_if_stream_error on a clean stream is a no-op" *
    doctest::test_suite("[stream-error]")) {
  auto s = make_stream(9'000'003);
  CHECK_FALSE(throws_any(s));
}

TEST_CASE(
    "a null exception_ptr is not stashed" *
    doctest::test_suite("[stream-error]")) {
  auto s = make_stream(9'000'004);
  scheduler::stash_stream_error(s, std::exception_ptr{});
  CHECK_FALSE(throws_any(s));
}

TEST_CASE(
    "errors are isolated per stream index" *
    doctest::test_suite("[stream-error]")) {
  auto a = make_stream(9'000'005);
  auto b = make_stream(9'000'006);

  scheduler::stash_stream_error(
      a, std::make_exception_ptr(std::runtime_error("only on a")));

  // A waitpoint on a DIFFERENT stream must not consume or throw a's error,
  // even though the process-global sentinel is raised.
  CHECK_FALSE(throws_any(b));

  // a's error is still pending and surfaces on a's own waitpoint. This is the
  // primitive that makes the per-thread clear_streams() scope (H1) correct:
  // touching one stream's stash never affects another's.
  CHECK(throws_with(a, "only on a"));
}

TEST_CASE(
    "clear_stream_error drops a pending error without throwing" *
    doctest::test_suite("[stream-error]")) {
  auto s = make_stream(9'000'007);

  scheduler::stash_stream_error(
      s, std::make_exception_ptr(std::runtime_error("to be cleared")));
  scheduler::clear_stream_error(s);

  CHECK_FALSE(throws_any(s));
}

TEST_CASE(
    "clearing one stream leaves another stream's error intact" *
    doctest::test_suite("[stream-error]")) {
  auto a = make_stream(9'000'008);
  auto b = make_stream(9'000'009);

  scheduler::stash_stream_error(
      a, std::make_exception_ptr(std::runtime_error("keep a")));
  scheduler::stash_stream_error(
      b, std::make_exception_ptr(std::runtime_error("drop b")));

  // Clearing b must not disturb a's stashed error (the H1 invariant).
  scheduler::clear_stream_error(b);
  CHECK_FALSE(throws_any(b));
  CHECK(throws_with(a, "keep a"));
}
