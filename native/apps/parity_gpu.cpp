// Parity oracle for GpuEnv: drive the batched GPU sim and the reference C++ ow::step from the
// SAME states with the SAME ego action (opponent = noop), then compare per-env aggregates each
// tick. GpuEnv runs float32; ow::step runs float64, so exact equality is impossible -- instead we
// flag DISCRETE divergence (a planet owner flip, a fleet hit/miss, a +/-1 ship count). A logic bug
// shows as massive immediate mismatch; benign float32 boundary effects show as small, late drift.
//   native\run.cmd ow_parity_gpu --worlds runs/native/train.owp --batch 64 --steps 40
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "apps/cli.hpp"
#include "core/encode.hpp"  // decode_action_continuous
#include "core/sim.hpp"
#include "core/state.hpp"
#include "io/serialize.hpp"
#include "rl/gpu_env.hpp"

using namespace ow;

#ifdef _WIN32
// LibTorch's CUDA backend lives in torch_cuda.dll, whose static initializers register the CUDA
// device only when the DLL is loaded. MSVC won't load it unless a symbol from it is referenced
// (is_available() lives in torch_cpu, so it does NOT pull it). Load it explicitly at startup so
// tensor.to(kCUDA) works. (torch\lib must be on PATH so the dependent CUDA DLLs resolve.)
extern "C" __declspec(dllimport) void* __stdcall LoadLibraryA(const char*);
static void force_load_torch_cuda() {
    LoadLibraryA("c10_cuda.dll");
    LoadLibraryA("torch_cuda.dll");
}
#else
static void force_load_torch_cuda() {}
#endif

int main(int argc, char** argv) try {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: survive an abort mid-rollout
    force_load_torch_cuda();
    printf("cuda_available=%d device_count=%d\n", (int)torch::cuda::is_available(),
           (int)torch::cuda::device_count());
    Args a(argc, argv);
    std::string worlds = a.s("worlds", "runs/native/train.owp");
    int B = a.i("batch", 64);
    int T = a.i("steps", 40);  // keep < 50 to stay pre-comet for the first parity milestone
    int K = a.i("fleets", 5);
    int Ec = a.i("planet-cap", 40);
    double thr = a.f("act-threshold", 0.05);
    int dumpN = a.i("dump-steps", -1);  // dump env 0 detail for steps 0..dumpN (-1 = off)
    uint64_t seed = (uint64_t)a.l("seed", 0);

    torch::Device dev(torch::kCUDA);
    auto pool = read_world_pool(worlds);
    if (pool.empty()) { printf("empty world pool: %s\n", worlds.c_str()); return 1; }
    std::vector<GameState> cpu(B);
    for (int b = 0; b < B; ++b) cpu[b] = pool[b % pool.size()];

    GpuEnvConfig cfg;
    cfg.planet_cap = Ec;
    cfg.fleet_cap = a.i("fleet-cap", 2048);
    cfg.episode_steps = a.i("episode-steps", 500);
    GpuEnv env(cfg, dev);
    env.reset(cpu);
    printf("reset ok: B=%d Ec=%d Fc=%d\n", B, env.planet_cap(), env.fleet_cap());

    Config econf{cfg.episode_steps, cfg.ship_speed, cfg.comet_speed};
    const int twoK = 2 * K;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32);

    int first_div = -1;
    for (int t = 0; t < T; ++t) {
        std::vector<float> act((size_t)B * Ec * twoK);
        for (auto& v : act) v = (float)U(rng);
        auto act_t = torch::from_blob(act.data(), {B, Ec, twoK}, f32).clone().to(dev);
        env.step(act_t, /*opponent=*/2, thr);

        for (int b = 0; b < B; ++b) {
            GameState& s = cpu[b];
            std::vector<long> rid(Ec, -1), rsh(Ec, 0);
            std::vector<float> am(Ec, 0.f);
            int n = std::min((int)s.planets.size(), Ec);
            for (int e = 0; e < n; ++e) {
                rid[e] = s.planets[e].id;
                rsh[e] = s.planets[e].ships;
                am[e] = (s.planets[e].owner == 0 && s.planets[e].ships > 0) ? 1.f : 0.f;
            }
            int inv = 0;
            Action move0 = decode_action_continuous(act.data() + (size_t)b * Ec * twoK, am.data(),
                                                    rid.data(), rsh.data(), Ec, K, thr, &inv);
            std::vector<Action> acts = {move0};  // opponent absent -> player 1 noop
            ow::step(s, acts, econf);
            s.step += 1;
        }

        const EnvTensors& g = env.tensors();
        auto po = g.p_owner.to(torch::kCPU), ps = g.p_ships.to(torch::kCPU),
             pa = g.p_alive.to(torch::kCPU), pX = g.p_x.to(torch::kCPU), pY = g.p_y.to(torch::kCPU);
        auto fo = g.f_owner.to(torch::kCPU), fs = g.f_ships.to(torch::kCPU),
             fa = g.f_alive.to(torch::kCPU), fX = g.f_x.to(torch::kCPU), fY = g.f_y.to(torch::kCPU);
        auto poA = po.accessor<float, 2>(), psA = ps.accessor<float, 2>(),
             paA = pa.accessor<float, 2>(), pXA = pX.accessor<float, 2>(), pYA = pY.accessor<float, 2>();
        auto foA = fo.accessor<float, 2>(), fsA = fs.accessor<float, 2>(),
             faA = fa.accessor<float, 2>(), fXA = fX.accessor<float, 2>(), fYA = fY.accessor<float, 2>();
        int Fc = env.fleet_cap();

        if (t <= dumpN) {
            int b = 0;
            printf("\n=== DETAIL env %d at step %d ===\n", b, t);
            printf("PLANETS slot | GPU al own ships    x      y   ||  CPU own ships    x      y\n");
            int npl = (int)cpu[b].planets.size();
            for (int e = 0; e < Ec; ++e) {
                bool ga = paA[b][e] > 0.5;
                if (!ga && e >= npl) continue;
                bool diff = e < npl && (std::lround(poA[b][e]) != cpu[b].planets[e].owner ||
                                        std::abs(psA[b][e] - cpu[b].planets[e].ships) > 0.5);
                printf("%s %4d | %2d %3d %7.0f %6.1f %6.1f  ||", diff ? "!!" : "  ", e, (int)ga,
                       (int)std::lround(poA[b][e]), psA[b][e], pXA[b][e], pYA[b][e]);
                if (e < npl) {
                    const Planet& p = cpu[b].planets[e];
                    printf(" %3d %7ld %6.1f %6.1f\n", p.owner, p.ships, p.x, p.y);
                } else printf("  (none)\n");
            }
            printf("FLEETS GPU (own,ships,x,y):");
            for (int f = 0; f < Fc; ++f)
                if (faA[b][f] > 0.5)
                    printf(" [%d,%.0f,%.2f,%.2f]", (int)std::lround(foA[b][f]), fsA[b][f], fXA[b][f],
                           fYA[b][f]);
            printf("\nFLEETS CPU (own,ships,x,y):");
            for (auto& f : cpu[b].fleets)
                printf(" [%d,%ld,%.2f,%.2f]", f.owner, f.ships, f.x, f.y);
            printf("\n=== END DETAIL ===\n");
        }

        int mism = 0;
        for (int b = 0; b < B; ++b) {
            double g0 = 0, g1 = 0, gn = 0, gf0 = 0, gf1 = 0;
            int gp0 = 0, gp1 = 0, gpn = 0, gnf = 0;
            for (int e = 0; e < Ec; ++e) {
                if (paA[b][e] < 0.5) continue;
                double sh = psA[b][e];
                int ow = (int)std::lround(poA[b][e]);
                if (ow == 0) { g0 += sh; gp0++; }
                else if (ow == 1) { g1 += sh; gp1++; }
                else { gn += sh; gpn++; }
            }
            for (int f = 0; f < Fc; ++f) {
                if (faA[b][f] < 0.5) continue;
                gnf++;
                if ((int)std::lround(foA[b][f]) == 0) gf0 += fsA[b][f]; else gf1 += fsA[b][f];
            }
            double c0 = 0, c1 = 0, cn = 0, cf0 = 0, cf1 = 0;
            int cp0 = 0, cp1 = 0, cpn = 0, cnf = 0;
            for (auto& p : cpu[b].planets) {
                if (p.owner == 0) { c0 += p.ships; cp0++; }
                else if (p.owner == 1) { c1 += p.ships; cp1++; }
                else { cn += p.ships; cpn++; }
            }
            for (auto& f : cpu[b].fleets) {
                cnf++;
                if (f.owner == 0) cf0 += f.ships; else cf1 += f.ships;
            }
            bool ok = gp0 == cp0 && gp1 == cp1 && gpn == cpn && gnf == cnf &&
                      std::abs(g0 - c0) < 1e-3 && std::abs(g1 - c1) < 1e-3 &&
                      std::abs(gn - cn) < 1e-3 && std::abs(gf0 - cf0) < 1e-3 &&
                      std::abs(gf1 - cf1) < 1e-3;
            if (!ok) ++mism;
        }
        if (mism > 0 && first_div < 0) first_div = t;
        printf("step %3d | match %5.1f%%  (%d/%d envs agree)\n", t, 100.0 * (B - mism) / B,
               B - mism, B);
    }
    printf("\nfirst divergence at step %d  (-1 = perfect; small late drift = benign float32)\n",
           first_div);
    return 0;
} catch (const std::exception& e) {
    fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
}
