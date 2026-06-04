// Tiny `--key value` / `--flag` argument parser shared by the native executables.
#pragma once
#include <map>
#include <string>

namespace ow {

struct Args {
    std::map<std::string, std::string> kv;
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) continue;
            std::string key = a.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
                kv[key] = argv[++i];
            else
                kv[key] = "1";  // bare flag
        }
    }
    std::string s(const std::string& k, const std::string& d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
    long l(const std::string& k, long d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : std::stol(it->second);
    }
    int i(const std::string& k, int d) const { return (int)l(k, d); }
    double f(const std::string& k, double d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : std::stod(it->second);
    }
    bool flag(const std::string& k) const { return kv.find(k) != kv.end(); }
};

}  // namespace ow
