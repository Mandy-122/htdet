#include <iostream>
#include <cstring>
#include <cmath>
#include "mbconv_20_top.h"

int main() {
    std::cout << "=================================================\n";
    std::cout << "mbconv_20_top  C-simulation testbench\n";
    std::cout << "in_ch=" << MBCONV20_IN_CH << " out_ch=" << MBCONV20_OUT_CH
              << " hid=" << MBCONV20_HID << " H=W=20 stride=1\n";
    std::cout << "=================================================\n";

    static act_t    in   [MBCONV20_IN_ELEMS];
    static act_t    out  [MBCONV20_OUT_ELEMS];
    static weight_t wc   [MBCONV20_WCONV_ELEMS];
    static meta_t   wm   [MBCONV20_WMETA_ELEMS];

    // Zero input → all-zero output (residual add gives zero + zero = zero through SiLU→0.5 at BN=0)
    memset(in,  0, sizeof(in));
    memset(out, 0, sizeof(out));
    memset(wc,  0, sizeof(wc));
    memset(wm,  0, sizeof(wm));

    std::cout << "\n[Test 1] zero input, zero weights -> run without crash\n";
    mbconv_20_top(in, out, wc, wm);
    std::cout << "  PASS (completed without crash)\n";

    // Simple non-zero test
    std::cout << "\n[Test 2] non-zero weights, check output is finite\n";
    for (int i = 0; i < MBCONV20_IN_ELEMS;   i++) in[i] = (act_t)((i % 5) - 2) * 0.1f;
    for (int i = 0; i < MBCONV20_WCONV_ELEMS; i++) wc[i] = (weight_t)((i % 3) - 1);
    for (int i = 0; i < MBCONV20_WMETA_ELEMS; i++) wm[i] = 0.01f;
    memset(out, 0, sizeof(out));

    mbconv_20_top(in, out, wc, wm);

    bool finite = true;
    for (int i = 0; i < MBCONV20_OUT_ELEMS; i++)
        if (!std::isfinite(out[i])) { finite = false; break; }
    std::cout << "  max_out[0]=" << out[0] << "\n";
    std::cout << (finite ? "  PASS (all outputs finite)\n" : "  FAIL (non-finite output)\n");

    std::cout << "\n=================================================\n";
    std::cout << "Testbench complete.\n";
    return 0;
}
