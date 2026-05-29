#pragma once
/*
 * ap_int.h  —  software-emulation stub for Xilinx ap_int<W>
 * Used by g++ C-simulation.  HLS synthesis uses the real Xilinx headers.
 *
 * W-bit signed 2's complement integer. Range: [-2^(W-1), 2^(W-1)-1]
 */
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <cmath>

template<int W>
struct ap_int {
    int64_t _v;

    static int64_t _quant(int64_t v) {
        const int64_t max_v =  (int64_t(1) << (W-1)) - 1;
        const int64_t min_v = -(int64_t(1) << (W-1));
        return std::max(min_v, std::min(max_v, v));
    }

    ap_int()           : _v(0) {}
    ap_int(int64_t v)  : _v(_quant(v)) {}
    ap_int(int     v)  : _v(_quant(int64_t(v))) {}
    ap_int(double  v)  : _v(_quant(int64_t(std::round(v)))) {}
    ap_int(float   v)  : _v(_quant(int64_t(std::round(double(v))))) {}

    operator double()  const { return double(_v); }
    operator float()   const { return float(_v); }
    operator int()     const { return int(_v); }
    operator int64_t() const { return _v; }

    ap_int& operator=(int64_t v) { _v = _quant(v); return *this; }
    ap_int& operator=(int     v) { _v = _quant(int64_t(v)); return *this; }

    friend std::ostream& operator<<(std::ostream& os, const ap_int& a) {
        return os << a._v;
    }
};
