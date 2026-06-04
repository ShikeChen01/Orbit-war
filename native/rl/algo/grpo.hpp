// The GRPO-specific algorithm pieces, each an explicit function (the action distribution's
// sample / log_prob / entropy / kl live in model/distribution.hpp). Math: docs/rl_math.pdf
// section "GRPO update". No value network -- the baseline is the group.
#pragma once
#include <torch/torch.h>

namespace ow {

// (advantage) Group-relative, no critic:  A_i = (R_i - mu_g) / (std_g + eps), broadcast to
// the trajectory's transitions. returns (N,), group_id (N,) int64 in [0,num_groups).
// whiten=false -> mean-subtract only (A_i = R_i - mu_g).   [Eq. (8)]
torch::Tensor group_advantage(const torch::Tensor& returns, const torch::Tensor& group_id,
                              int num_groups, double eps, bool whiten);

// (policy loss) Clipped surrogate, scalar (mean over transitions).   [Eq. (10)]
//   ratio = exp(logp - old_logp); -mean(min(ratio*A, clip(ratio,1+/-eps)*A))
torch::Tensor policy_surrogate(const torch::Tensor& logp, const torch::Tensor& old_logp,
                               const torch::Tensor& adv, double clip);

// (KL penalty) k3 estimator to the reference policy on the taken action, scalar mean.  [Eq. (11)]
//   d = ref_logp - logp;  mean( exp(d) - d - 1 )  (>= 0)
torch::Tensor kl_penalty(const torch::Tensor& logp, const torch::Tensor& ref_logp);

}  // namespace ow
