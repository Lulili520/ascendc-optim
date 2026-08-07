#include <cstdint>
#include <iostream>

#include "utils/tiling/platform/platform_ascendc.h"

int main() {
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        std::cerr << "PlatformAscendCManager::GetInstance returned null\n";
        return 2;
    }

    uint64_t ub = 0;
    uint64_t l1 = 0;
    uint64_t l2 = 0;
    uint64_t l0a = 0;
    uint64_t l0b = 0;
    uint64_t l0c = 0;
    uint64_t hbm = 0;
    uint64_t hbmBw = 0;
    uint64_t l2Bw = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::HBM, hbm);
    platform->GetCoreMemBw(platform_ascendc::CoreMemType::HBM, hbmBw);
    platform->GetCoreMemBw(platform_ascendc::CoreMemType::L2, l2Bw);

    std::cout << "{\n"
              << "  \"soc_version_enum\": "
              << static_cast<uint32_t>(platform->GetSocVersion()) << ",\n"
              << "  \"aic_core_count\": " << platform->GetCoreNumAic() << ",\n"
              << "  \"aiv_core_count\": " << platform->GetCoreNumAiv() << ",\n"
              << "  \"ub_bytes_per_core\": " << ub << ",\n"
              << "  \"l1_bytes_per_core\": " << l1 << ",\n"
              << "  \"l0a_bytes_per_core\": " << l0a << ",\n"
              << "  \"l0b_bytes_per_core\": " << l0b << ",\n"
              << "  \"l0c_bytes_per_core\": " << l0c << ",\n"
              << "  \"l2_bytes\": " << l2 << ",\n"
              << "  \"hbm_bytes\": " << hbm << ",\n"
              << "  \"hbm_bytes_per_cycle_per_core\": " << hbmBw << ",\n"
              << "  \"l2_bytes_per_cycle_per_core\": " << l2Bw << "\n"
              << "}\n";
    return 0;
}
