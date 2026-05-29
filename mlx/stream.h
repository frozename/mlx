// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "mlx/api.h"
#include "mlx/device.h"

namespace mlx::core {

struct MLX_API Stream {
  int index;
  Device device;
  // Per-incarnation counter assigned by new_stream(). A completion handler
  // captures the Stream by value; if the stream is later destroyed and a new
  // stream reuses the same index, the stale generation lets the scheduler
  // distinguish the old handler's error from a real error on the new stream.
  uint64_t generation{0};
  explicit Stream(int index, Device device) : index(index), device(device) {}

  bool operator==(const Stream& rhs) const {
    return index == rhs.index && device == rhs.device;
  }
  bool operator<(const Stream& rhs) const {
    return device < rhs.device || index < rhs.index;
  }
};

struct MLX_API ThreadLocalStream : public Stream {
  using Stream::Stream;
};

// Process-global counter incremented on each new_stream() call. The value
// is stamped into Stream::generation so completion handlers captured before
// a stream is destroyed carry the old generation, allowing the scheduler to
// discard their errors rather than attributing them to a new stream that
// reuses the same index. Defined in mlx/stream.cpp.
extern std::atomic<uint64_t> stream_generation_counter_;

/** Get the default stream of current thread for the given device. */
MLX_API Stream default_stream(Device d);

/** Make the stream the default for its device on current thread. */
MLX_API void set_default_stream(Stream s);

/** Make a new stream on the given device. */
MLX_API Stream new_stream(Device d);

/** Make a new stream that will be unique per thread. */
MLX_API ThreadLocalStream new_thread_local_stream(Device d);

/** Get the stream for current thread from ThreadLocalStream. */
MLX_API Stream stream_from_thread_local_stream(ThreadLocalStream tls);

/** Get all available streams. */
MLX_API std::vector<Stream> get_streams();

/* Synchronize with the default stream. */
MLX_API void synchronize();

/* Synchronize with the provided stream. */
MLX_API void synchronize(Stream);

/* Synchronize with the stream corresponding to the current thread. */
MLX_API void synchronize(ThreadLocalStream);

/* Destroy all streams created in current thread. */
MLX_API void clear_streams();

/* Advisory per-stream tag (e.g. model affinity) used by higher layers
 * (oMLX, mlx-lm) to key downstream resources. Stored in a side-registry keyed
 * by stream.index rather than on the Stream struct itself, so the handle stays
 * small and trivially-copyable — Stream is captured BY VALUE in Metal
 * completion-handler lambdas on the hot submission path, where an embedded
 * std::optional<std::string> would inflate the capture and heap-allocate on
 * copy for non-SSO tags. Not part of stream identity: equality / ordering
 * ignore the tag (two streams with the same {index, device} are identical). */
MLX_API void set_stream_tag(const Stream& s, std::string tag);
MLX_API std::optional<std::string> stream_tag(const Stream& s);
MLX_API void clear_stream_tag(const Stream& s);

} // namespace mlx::core
