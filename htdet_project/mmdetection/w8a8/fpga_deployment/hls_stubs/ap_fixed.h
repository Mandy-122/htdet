#pragma once
/*
 * ap_fixed.h  —  software-emulation stub for Xilinx ap_fixed<W,I,Q,O>
 * Used by g++ C-simulation.  HLS synthesis uses the real Xilinx headers.
 *
 * Semantics:
 *   W = total bits (including sign)
 *   I = integer bits (including sign)
 *   FRAC = W - I  (fractional bits)
 *   step = 2^(-FRAC)
 *   range: [-2^(I-1), 2^(I-1) - step]   (signed 2's complement)
 *   AP_RND  → round to nearest
 *   AP_SAT  → saturate on overflow
 */

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>

static const int AP_RND  = 0;
static const int AP_TRN  = 1;
static const int AP_SAT  = 0;
static const int AP_WRAP = 1;

template<int W, int I, int _Q = AP_RND, int _O = AP_SAT>
struct ap_fixed {
    double _v;  // internal storage; quantized on every assignment

    // Precomputed once per ap_fixed<W,I,...> specialization — avoids recomputing
    // step/inv_step/max_v/min_v on every quantization call (critical for sim speed).
    static __attribute__((always_inline)) double _quant(double v) {
        constexpr int FRAC = W - I;
        static const double step     = (FRAC >= 0) ? (1.0 / (int64_t(1) << FRAC))
                                                    :  double(int64_t(1) << (-FRAC));
        static const double inv_step = 1.0 / step;
        static const double max_v    = double( (int64_t(1) << (W-1)) - 1 ) * step;
        static const double min_v    = double( -(int64_t(1) << (W-1))    ) * step;
        v = std::round(v * inv_step) * step;
        return std::max(min_v, std::min(max_v, v));
    }

    // Constructors
    ap_fixed()           : _v(0) {}
    ap_fixed(double  v)  : _v(_quant(v)) {}
    ap_fixed(float   v)  : _v(_quant(double(v))) {}
    ap_fixed(int     v)  : _v(_quant(double(v))) {}
    ap_fixed(int64_t v)  : _v(_quant(double(v))) {}
    template<int W2,int I2,int Q2,int O2>
    ap_fixed(const ap_fixed<W2,I2,Q2,O2>& o) : _v(_quant(o._v)) {}

    // Implicit conversions
    operator double() const { return _v; }
    operator float()  const { return float(_v); }
    operator int()    const { return int(_v); }

    // Assignment
    ap_fixed& operator=(double  v) { _v = _quant(v);         return *this; }
    ap_fixed& operator=(float   v) { _v = _quant(double(v)); return *this; }
    ap_fixed& operator=(int     v) { _v = _quant(double(v)); return *this; }
    template<int W2,int I2,int Q2,int O2>
    ap_fixed& operator=(const ap_fixed<W2,I2,Q2,O2>& o) { _v = _quant(o._v); return *this; }

    // Compound assignment (accumulate in double, then quantize)
    ap_fixed& operator+=(double v) { _v = _quant(_v + v);         return *this; }
    ap_fixed& operator+=(float  v) { _v = _quant(_v + double(v)); return *this; }
    template<int W2,int I2,int Q2,int O2>
    ap_fixed& operator+=(const ap_fixed<W2,I2,Q2,O2>& o) { _v = _quant(_v + o._v); return *this; }
    template<int W2,int I2,int Q2,int O2>
    ap_fixed& operator-=(const ap_fixed<W2,I2,Q2,O2>& o) { _v = _quant(_v - o._v); return *this; }

    // Comparisons (against ap_fixed of any shape)
    template<int W2,int I2,int Q2,int O2>
    bool operator> (const ap_fixed<W2,I2,Q2,O2>& o) const { return _v >  o._v; }
    template<int W2,int I2,int Q2,int O2>
    bool operator< (const ap_fixed<W2,I2,Q2,O2>& o) const { return _v <  o._v; }
    template<int W2,int I2,int Q2,int O2>
    bool operator>=(const ap_fixed<W2,I2,Q2,O2>& o) const { return _v >= o._v; }
    template<int W2,int I2,int Q2,int O2>
    bool operator<=(const ap_fixed<W2,I2,Q2,O2>& o) const { return _v <= o._v; }
    template<int W2,int I2,int Q2,int O2>
    bool operator==(const ap_fixed<W2,I2,Q2,O2>& o) const { return _v == o._v; }
    bool operator> (double v) const { return _v >  v; }
    bool operator> (float  v) const { return _v >  double(v); }
    bool operator> (int    v) const { return _v >  double(v); }
    bool operator< (double v) const { return _v <  v; }
    bool operator>=(double v) const { return _v >= v; }
    bool operator<=(double v) const { return _v <= v; }

    // Binary arithmetic — return double so the caller's ap_fixed assignment quantizes
    template<int W2,int I2,int Q2,int O2>
    friend double operator+(const ap_fixed& a, const ap_fixed<W2,I2,Q2,O2>& b) { return a._v + b._v; }
    template<int W2,int I2,int Q2,int O2>
    friend double operator-(const ap_fixed& a, const ap_fixed<W2,I2,Q2,O2>& b) { return a._v - b._v; }
    template<int W2,int I2,int Q2,int O2>
    friend double operator*(const ap_fixed& a, const ap_fixed<W2,I2,Q2,O2>& b) { return a._v * b._v; }
    template<int W2,int I2,int Q2,int O2>
    friend double operator/(const ap_fixed& a, const ap_fixed<W2,I2,Q2,O2>& b) { return a._v / b._v; }

    friend double operator+(const ap_fixed& a, double b) { return a._v + b; }
    friend double operator+(double a, const ap_fixed& b) { return a + b._v; }
    friend double operator+(const ap_fixed& a, float  b) { return a._v + double(b); }
    friend double operator+(float a,  const ap_fixed& b) { return double(a) + b._v; }
    friend double operator-(const ap_fixed& a, double b) { return a._v - b; }
    friend double operator-(double a, const ap_fixed& b) { return a - b._v; }
    friend double operator*(const ap_fixed& a, double b) { return a._v * b; }
    friend double operator*(double a, const ap_fixed& b) { return a * b._v; }
    friend double operator*(const ap_fixed& a, float  b) { return a._v * double(b); }
    friend double operator*(float a,  const ap_fixed& b) { return double(a) * b._v; }
    friend double operator/(const ap_fixed& a, double b) { return a._v / b; }
    friend double operator/(double a, const ap_fixed& b) { return a / b._v; }
    friend double operator/(const ap_fixed& a, float  b) { return a._v / double(b); }
    friend double operator/(float a,  const ap_fixed& b) { return double(a) / b._v; }
    friend double operator/(const ap_fixed& a, int    b) { return a._v / double(b); }
    friend double operator/(int    a, const ap_fixed& b) { return double(a) / b._v; }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, const ap_fixed& a) {
        return os << a._v;
    }
};
