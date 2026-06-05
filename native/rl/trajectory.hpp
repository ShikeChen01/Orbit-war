// A flat batch of collected transitions for the GRPO update. Leading dim N = number of valid
// (pre-terminal) transitions gathered across all grouped episodes. Stored on CPU; minibatches
// are moved to the device inside the update.
#pragma once
#include <torch/torch.h>

namespace ow {

struct TrajectoryBatch {
    torch::Tensor entities;     // (N,E,F) float32
    torch::Tensor entity_mask;  // (N,E)   float32
    torch::Tensor action_mask;  // (N,E)   float32
    torch::Tensor globals;      // (N,G)   float32
    torch::Tensor action;       // (N,E,2K) float32 squashed-Gaussian controls
    torch::Tensor old_logp;     // (N,)    behavior log-prob (under the sampling policy)
    torch::Tensor ref_logp;     // (N,)    reference (BC) log-prob, for the KL anchor
    torch::Tensor advantage;    // (N,)    group-normalized return, broadcast to each step
    long n_transitions = 0;
};

}  // namespace ow
