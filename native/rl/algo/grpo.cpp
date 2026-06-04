#include "rl/algo/grpo.hpp"

namespace ow {

torch::Tensor group_advantage(const torch::Tensor& returns, const torch::Tensor& group_id,
                              int num_groups, double eps, bool whiten) {
    auto opts = returns.options();
    auto sum = torch::zeros({num_groups}, opts);
    auto cnt = torch::zeros({num_groups}, opts);
    sum.index_add_(0, group_id, returns);
    cnt.index_add_(0, group_id, torch::ones_like(returns));
    cnt = cnt.clamp_min(1.0);
    auto mean = sum / cnt;                                    // (num_groups,)
    auto centered = returns - mean.index_select(0, group_id); // (N,)
    if (!whiten) return centered;
    auto sqsum = torch::zeros({num_groups}, opts);
    sqsum.index_add_(0, group_id, centered * centered);
    auto std = (sqsum / cnt).index_select(0, group_id).add(eps).sqrt();
    return centered / std;
}

torch::Tensor policy_surrogate(const torch::Tensor& logp, const torch::Tensor& old_logp,
                               const torch::Tensor& adv, double clip) {
    auto ratio = torch::exp(logp - old_logp);
    auto s1 = ratio * adv;
    auto s2 = torch::clamp(ratio, 1.0 - clip, 1.0 + clip) * adv;
    return -torch::min(s1, s2).mean();
}

torch::Tensor kl_penalty(const torch::Tensor& logp, const torch::Tensor& ref_logp) {
    auto d = ref_logp - logp;                  // log(pi_ref / pi_theta)
    return (torch::exp(d) - d - 1.0).mean();   // k3 estimator, >= 0
}

}  // namespace ow
