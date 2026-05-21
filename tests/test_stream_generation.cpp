// Copyright © 2026 Apple Inc.
//
// Unit tests for the Stream generation counter that prevents stale
// completion-handler errors from being attributed to a new stream that
// reuses the same index. Does not require a Metal device — test streams
// use explicit high-numbered indices that cannot collide with streams
// allocated by other tests in the same process.

#include <stdexcept>
#include <string>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/scheduler.h"

using namespace mlx::core;
using namespace mlx::core::scheduler;

// ── helpers ─────────────────────────────────────────────────────────────────

// Build a detached Stream with an explicit index and generation.
// Bypasses new_stream() so we can craft two incarnations that share an index.
static Stream make_incarnation(int index, uint64_t generation) {
  Stream s{index, Device::cpu};
  s.generation = generation;
  return s;
}

// ── test ────────────────────────────────────────────────────────────────────

TEST_CASE(
    "stream generation counter prevents stale error attribution" *
    doctest::test_suite("[stream-generation]")) {
  // Use a high-numbered index to avoid collisions with streams allocated
  // elsewhere in the test suite (new_stream() allocates sequentially from 0).
  constexpr int kTestIndex = 8'000'001;
  constexpr uint64_t kOldGen = 7;
  constexpr uint64_t kNewGen = 8;

  auto old_s = make_incarnation(kTestIndex, kOldGen);
  auto new_s = make_incarnation(kTestIndex, kNewGen);

  // --- step 1: stash an error under the old generation ---

  scheduler::notify_stream_error(
      old_s,
      std::make_exception_ptr(
          std::runtime_error("stale error from old incarnation")));

  // --- step 2: new stream (same index, later generation) must NOT see it ---

  bool threw = false;
  try {
    scheduler::throw_if_stream_error(new_s);
  } catch (...) {
    threw = true;
  }
  CHECK_FALSE(threw);

  // --- step 3: the stale slot was consumed; now stash a real error ---

  // throw_if_stream_error erased the slot unconditionally, so new_s has no
  // pending error and any_stream_error_ was lowered.  A fresh notify must go
  // through.
  scheduler::notify_stream_error(
      new_s,
      std::make_exception_ptr(
          std::runtime_error("real error on new incarnation")));

  // --- step 4: new stream's matching generation MUST surface the error ---

  bool threw2 = false;
  std::string msg;
  try {
    scheduler::throw_if_stream_error(new_s);
  } catch (const std::runtime_error& e) {
    threw2 = true;
    msg = e.what();
  }
  CHECK(threw2);
  CHECK(msg.find("real error on new incarnation") != std::string::npos);

  // --- step 5: baseline — a notify + throw on old incarnation still works ---
  // (Confirms the generation-match branch is not always skipping.)

  // Slot was consumed in step 4; stash against old_s now.
  scheduler::notify_stream_error(
      old_s,
      std::make_exception_ptr(std::runtime_error("error attributed to old")));

  bool threw3 = false;
  std::string msg3;
  try {
    scheduler::throw_if_stream_error(old_s);
  } catch (const std::runtime_error& e) {
    threw3 = true;
    msg3 = e.what();
  }
  CHECK(threw3);
  CHECK(msg3.find("error attributed to old") != std::string::npos);
}
