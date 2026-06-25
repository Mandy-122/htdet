#include <iostream>
#include <cstring>
#include <cmath>
#include "mbconv_160_top.h"

int main() {
    std::cout << "=================================================\n";
    std::cout << "mbconv_160_top  C-simulation testbench\n";
    std::cout << "in_ch=" << MBCONV160_IN_CH << " out_ch=" << MBCONV160_OUT_CH
              << " hid=" << MBCONV160_HID << " H=W=160 stride=2\n";
    std::cout << "in_elems=" << MBCONV160_IN_ELEMS
              << "  out_elems=" << MBCONV160_OUT_ELEMS << "\n";
    std::cout << "=================================================\n";

    static act_t    in   [MBCONV160_IN_ELEMS];
    static act_t    out  [MBCONV160_OUT_ELEMS];
    static weight_t wc   [MBCONV160_WCONV_ELEMS];
    static meta_t   wm   [MBCONV160_WMETA_ELEMS];

    std::cout << "\n[Test 1] zero weights -> run without crash\n";
    memset(in,  0, sizeof(in));
    memset(out, 0, sizeof(out));
    memset(wc,  0, sizeof(wc));
    memset(wm,  0, sizeof(wm));
    mbconv_160_top(in, out, wc, wm);
    std::cout << "  PASS (completed)\n";

    std::cout << "\n[Test 2] non-zero: check output is finite\n";
    for (int i = 0; i < MBCONV160_IN_ELEMS;    i++) in[i] = (act_t)((i % 5) - 2) * 0.1f;
    for (int i = 0; i < MBCONV160_WCONV_ELEMS; i++) wc[i] = (weight_t)((i % 3) - 1);
    for (int i = 0; i < MBCONV160_WMETA_ELEMS; i++) wm[i] = 0.01f;
    memset(out, 0, sizeof(out));
    mbconv_160_top(in, out, wc, wm);

    bool finite = true;
    for (int i = 0; i < MBCONV160_OUT_ELEMS; i++)
        if (!std::isfinite(out[i])) { finite = false; break; }
    std::cout << "  out[0]=" << out[0] << "\n";
    std::cout << (finite ? "  PASS\n" : "  FAIL (non-finite)\n");

    std::cout << "\n=================================================\n";
    std::cout << "Testbench complete.\n";
    return 0;
}
