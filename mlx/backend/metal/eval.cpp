// Copyright © 2023-2024 Apple Inc.
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

namespace {

// Parse a positive-int environment variable. Returns `fallback` when the var is
// unset, empty, malformed, out of int range, or non-positive. Uses strtol (not
// atoi, which silently yields 0 on garbage and is undefined on overflow) so a
// typo can never wedge or mis-tune the gate.
int read_positive_env_int(const char* name, int fallback) {
  const char* env = std::getenv(name);
  if (!env || !*env) {
    return fallback;
  }
  errno = 0;
  char* end = nullptr;
  long v = std::strtol(env, &end, 10);
  if (errno != 0 || end == env || *end != '\0' || v <= 0 ||
      v > std::numeric_limits<int>::max()) {
    return fallback;
  }
  return static_cast<int>(v);
}

// MLX_METAL_MAX_INFLIGHT_PER_STREAM: max command buffers in flight per stream
// before submission blocks. Unset/invalid -> INT_MAX, i.e. back-pressure is OFF
// and acquire/release are skipped entirely so the default hot path pays
// nothing. Cached once at first use via the C++11 function-local-static
// guarantee (thread-safe): this runs on the hot Metal submission path, and
// std::getenv takes a process-wide libc lock and is not async-signal-safe, so a
// single read is the correct contract for a static process-level knob.
int read_inflight_limit() {
  static const int limit = read_positive_env_int(
      "MLX_METAL_MAX_INFLIGHT_PER_STREAM", std::numeric_limits<int>::max());
  return limit;
}

// MLX_METAL_BACKPRESSURE_TIMEOUT_SECS: anti-deadlock deadline for the gate (not
// the cap itself). Unset/invalid -> 30. Cached once — same rationale as above.
int read_backpressure_timeout_secs() {
  static const int timeout =
      read_positive_env_int("MLX_METAL_BACKPRESSURE_TIMEOUT_SECS", 30);
  return timeout;
}

} // namespace

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

// Record a Metal command-buffer error into a per-stream stash instead of
// throwing. Metal completion handlers run on
// `com.Metal.CompletionQueueDispatch` (libdispatch); a C++ exception thrown
// across a dispatch block aborts the process unrecoverably. The stash is
// re-thrown on the caller thread at the next sync waitpoint (eval / finalize
// / synchronize). See mlx-explore/mlx#2670.
//
// Called before any task-completion notification so a thread that wakes on
// notify_task_completion is guaranteed to see the error already stashed when
// it next reaches a throw_if_stream_error waitpoint.
//
// noexcept by contract: this runs inside a libdispatch completion handler, so
// it must never let an exception escape — that is the exact process-abort this
// patch exists to prevent. If building or stashing the error itself throws
// (e.g. std::bad_alloc under memory pressure), the command-buffer error is
// dropped rather than crashing the process.
inline void stash_command_buffer_error(
    Stream s,
    MTL::CommandBuffer* cbuf) noexcept {
  if (cbuf->status() != MTL::CommandBufferStatusError) {
    return;
  }
  try {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed";
    // Defensive null-checks: in practice both nodes are populated when the
    // status is Error, but the Metal API does not statically guarantee it.
    if (auto* err = cbuf->error()) {
      if (auto* desc = err->localizedDescription()) {
        if (auto* utf = desc->utf8String()) {
          msg << ": " << utf;
        }
      }
    }
    scheduler::stash_stream_error(
        s, std::make_exception_ptr(std::runtime_error(msg.str())));
  } catch (...) {
    // Swallow: never throw out of a libdispatch completion handler.
  }
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  // Surface any error stashed asynchronously by a prior completion handler
  // before queueing new work on this stream.
  scheduler::throw_if_stream_error(s);
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (encoder.needs_commit()) {
    encoder.end_encoding();
    const int limit = read_inflight_limit();
    const bool backpressure_enabled = limit != std::numeric_limits<int>::max();
    if (backpressure_enabled) {
      // Throttle BEFORE notify_new_task so the in-flight slot and the
      // active-task accounting stay consistent (acquire balances the
      // completion handler's release; notify_new_task balances its
      // notify_task_completion). Skipped entirely when the env var is unset.
      scheduler::acquire_stream_slot(s, limit, read_backpressure_timeout_secs());
    }
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s, backpressure_enabled, buffers = std::move(buffers)](
            MTL::CommandBuffer* cbuf) {
          // Stash any error BEFORE waking either waiter — both the
          // back-pressure release and notify_task_completion can unblock a
          // thread, and the #2670 ordering requires the error be visible
          // before that thread reaches its next throw_if_stream_error
          // waitpoint.
          stash_command_buffer_error(s, cbuf);
          if (backpressure_enabled) {
            scheduler::release_stream_slot(s);
          }
          scheduler::notify_task_completion(s);
        });
    encoder.commit();
  } else {
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          stash_command_buffer_error(s, cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  // Surface any error stashed asynchronously by a prior completion handler
  // before committing this stream's pending work.
  scheduler::throw_if_stream_error(s);
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  encoder.end_encoding();
  const int limit = read_inflight_limit();
  const bool backpressure_enabled = limit != std::numeric_limits<int>::max();
  if (backpressure_enabled) {
    scheduler::acquire_stream_slot(s, limit, read_backpressure_timeout_secs());
  }
  cb->addCompletedHandler([s, backpressure_enabled](MTL::CommandBuffer* cbuf) {
    // Stash before releasing the slot (see eval()'s handler) so a woken
    // submitter sees the error at its next waitpoint.
    stash_command_buffer_error(s, cbuf);
    if (backpressure_enabled) {
      scheduler::release_stream_slot(s);
    }
  });
  encoder.commit();
}

void synchronize(Stream s) {
  // Surface any error stashed asynchronously by a prior completion handler
  // before driving the synchronous wait.
  scheduler::throw_if_stream_error(s);
  metal::get_command_encoder(s).synchronize();
  // CommandEncoder::synchronize() throws directly on its OWN command buffer's
  // error, but a prior buffer committed earlier on this stream can complete
  // (and stash an error) while we were blocked above. Re-check so synchronize()
  // surfaces those before returning rather than deferring them to the next
  // waitpoint.
  scheduler::throw_if_stream_error(s);
}

void clear_streams() {
  auto& encoders = metal::get_command_encoders();
  // Capture this thread's stream indices before teardown. get_command_
  // encoders() is thread_local, so we sweep only the indices this thread owns
  // — a process-global sweep would silently discard another live thread's
  // pending stream error and corrupt its bookkeeping.
  std::vector<int> indices;
  indices.reserve(encoders.size());
  for (const auto& [index, encoder] : encoders) {
    indices.push_back(index);
  }
  // Destroy the encoders FIRST: ~CommandEncoder() calls synchronize(), which
  // blocks in waitUntilCompleted() until each pending command buffer's
  // completion handler has run — so any teardown-time error is stashed before
  // we sweep. Sweeping first would let those handlers re-stash an error after
  // the sweep, orphaning it (the stream is gone, so nothing ever consumes it)
  // and leaving the global error sentinel stuck on the slow path.
  encoders.clear();
  for (int index : indices) {
    scheduler::clear_stream_error(Stream(index, Device::gpu));
  }
}

} // namespace mlx::core::gpu
