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
    // clamp the whitened advantage: a group whose trajectories return nearly the same value (e.g.
    // stage 1, same world vs a passive opponent) has std->0, which would blow the advantage (and the
    // policy step) up. [-10,10] bounds it without affecting well-separated groups.
    return (centered / std).clamp(-10.0, 10.0);
}

torch::Tensor policy_surrogate(const torch::Tensor& logp, const torch::Tensor& old_logp,
                               const torch::Tensor& adv, double clip) {
    // clamp the log-ratio before exp: a single large policy step (common early with a deep net over
    // many summed action components) would otherwise overflow exp -> inf ratio -> NaN loss.
    auto ratio = torch::exp((logp - old_logp).clamp(-20.0, 20.0));
    auto s1 = ratio * adv;
    auto s2 = torch::clamp(ratio, 1.0 - clip, 1.0 + clip) * adv;
    return -torch::min(s1, s2).mean();
}

torch::Tensor kl_penalty(const torch::Tensor& logp, const torch::Tensor& ref_logp) {
    auto d = (ref_logp - logp).clamp(-20.0, 20.0);  // log(pi_ref / pi_theta), guarded against exp inf
    return (torch::exp(d) - d - 1.0).mean();         // k3 estimator, >= 0
}

}  // namespace ow
