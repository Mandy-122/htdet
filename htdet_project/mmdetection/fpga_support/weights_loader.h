/*
 * weights_loader.h
 * Utility to load binary weight files exported by export_weights.py
 * into host-side memory, then copy to FPGA DDR via XRT buffer handles.
 *
 * Usage pattern (pseudo-code):
 *   WeightBuffers bufs;
 *   weights_load_all("weights/", &bufs);
 *   // ... create XRT buffers, copy bufs.backbone_w into them ...
 *   weights_free(&bufs);
 */

#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ============================================================
// Weight element sizes (must match sizes.txt from export_weights.py)
// These are conservative upper bounds; actual sizes are read from sizes.txt.
// ============================================================
#define BACKBONE_W_ELEMS_MAX    6000000
#define FPN_W_ELEMS_MAX         2600000
#define CLS_CONV_W_ELEMS_MAX    2360000
#define REG_CONV_W_ELEMS_MAX    2360000
#define CLS_PRED_W_ELEMS_MAX      82944
#define CLS_PRED_B_ELEMS_MAX         36
#define REG_PRED_W_ELEMS_MAX      82944
#define REG_PRED_B_ELEMS_MAX         36

// ============================================================
// Weight buffer struct
// Each buffer is a flat int16_t array (Q8.8 fixed-point).
// Sizes are populated from sizes.txt.
// ============================================================
typedef struct {
    int16_t* backbone_w;
    int16_t* fpn_w;
    int16_t* cls_conv_w;
    int16_t* reg_conv_w;
    int16_t* cls_pred_w;
    int16_t* cls_pred_b;
    int16_t* reg_pred_w;
    int16_t* reg_pred_b;

    size_t   backbone_n;
    size_t   fpn_n;
    size_t   cls_conv_n;
    size_t   reg_conv_n;
    size_t   cls_pred_w_n;
    size_t   cls_pred_b_n;
    size_t   reg_pred_w_n;
    size_t   reg_pred_b_n;
} WeightBuffers;

// ============================================================
// Load sizes.txt into the WeightBuffers size fields.
// Returns 0 on success, -1 on error.
// ============================================================
static inline int weights_load_sizes(const char* dir, WeightBuffers* b) {
    char path[512];
    snprintf(path, sizeof(path), "%s/sizes.txt", dir);
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[weights_loader] Cannot open %s\n", path);
        return -1;
    }
    char key[128];
    size_t val;
    while (fscanf(f, "%127s %zu", key, &val) == 2) {
        if      (strcmp(key, "backbone_w")  == 0) b->backbone_n    = val;
        else if (strcmp(key, "fpn_w")       == 0) b->fpn_n         = val;
        else if (strcmp(key, "cls_conv_w")  == 0) b->cls_conv_n    = val;
        else if (strcmp(key, "reg_conv_w")  == 0) b->reg_conv_n    = val;
        else if (strcmp(key, "cls_pred_w")  == 0) b->cls_pred_w_n  = val;
        else if (strcmp(key, "cls_pred_b")  == 0) b->cls_pred_b_n  = val;
        else if (strcmp(key, "reg_pred_w")  == 0) b->reg_pred_w_n  = val;
        else if (strcmp(key, "reg_pred_b")  == 0) b->reg_pred_b_n  = val;
    }
    fclose(f);
    return 0;
}

// ============================================================
// Load a single binary file (int16_t elements) into a newly
// malloc'd buffer.  Returns NULL on failure.
// ============================================================
static inline int16_t* load_bin(const char* path, size_t n_elems) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[weights_loader] Cannot open %s\n", path); return NULL; }
    int16_t* buf = (int16_t*)malloc(n_elems * sizeof(int16_t));
    if (!buf) { fprintf(stderr, "[weights_loader] malloc failed for %zu elems\n", n_elems); fclose(f); return NULL; }
    size_t rd = fread(buf, sizeof(int16_t), n_elems, f);
    fclose(f);
    if (rd != n_elems) {
        fprintf(stderr, "[weights_loader] %s: expected %zu got %zu elements\n", path, n_elems, rd);
        free(buf);
        return NULL;
    }
    return buf;
}

// ============================================================
// Load all weight files from a directory into WeightBuffers.
// Calls weights_load_sizes first, then malloc+read each file.
// Returns 0 on success, -1 on error.
// ============================================================
static inline int weights_load_all(const char* dir, WeightBuffers* b) {
    memset(b, 0, sizeof(*b));
    if (weights_load_sizes(dir, b) != 0) return -1;

    char path[512];

#define LOAD(field, fname, n_field) \
    snprintf(path, sizeof(path), "%s/" fname, dir); \
    b->field = load_bin(path, b->n_field); \
    if (!b->field) return -1;

    LOAD(backbone_w,  "backbone_w.bin",  backbone_n)
    LOAD(fpn_w,       "fpn_w.bin",       fpn_n)
    LOAD(cls_conv_w,  "cls_conv_w.bin",  cls_conv_n)
    LOAD(reg_conv_w,  "reg_conv_w.bin",  reg_conv_n)
    LOAD(cls_pred_w,  "cls_pred_w.bin",  cls_pred_w_n)
    LOAD(cls_pred_b,  "cls_pred_b.bin",  cls_pred_b_n)
    LOAD(reg_pred_w,  "reg_pred_w.bin",  reg_pred_w_n)
    LOAD(reg_pred_b,  "reg_pred_b.bin",  reg_pred_b_n)
#undef LOAD

    printf("[weights_loader] Loaded:\n");
    printf("  backbone_w  : %7zu elems  (%5.1f MB)\n", b->backbone_n,   b->backbone_n   * 2.0 / 1e6);
    printf("  fpn_w       : %7zu elems  (%5.1f MB)\n", b->fpn_n,        b->fpn_n        * 2.0 / 1e6);
    printf("  cls_conv_w  : %7zu elems  (%5.1f MB)\n", b->cls_conv_n,   b->cls_conv_n   * 2.0 / 1e6);
    printf("  reg_conv_w  : %7zu elems  (%5.1f MB)\n", b->reg_conv_n,   b->reg_conv_n   * 2.0 / 1e6);
    printf("  cls_pred_w/b: %7zu / %3zu elems\n",      b->cls_pred_w_n, b->cls_pred_b_n);
    printf("  reg_pred_w/b: %7zu / %3zu elems\n",      b->reg_pred_w_n, b->reg_pred_b_n);
    return 0;
}

// ============================================================
// Free all malloc'd buffers.
// ============================================================
static inline void weights_free(WeightBuffers* b) {
    free(b->backbone_w);  b->backbone_w  = NULL;
    free(b->fpn_w);       b->fpn_w       = NULL;
    free(b->cls_conv_w);  b->cls_conv_w  = NULL;
    free(b->reg_conv_w);  b->reg_conv_w  = NULL;
    free(b->cls_pred_w);  b->cls_pred_w  = NULL;
    free(b->cls_pred_b);  b->cls_pred_b  = NULL;
    free(b->reg_pred_w);  b->reg_pred_w  = NULL;
    free(b->reg_pred_b);  b->reg_pred_b  = NULL;
}

// ============================================================
// Load a float32 image (CHW, 3×H×W) from a binary file.
// Normalises to [-1,1] using ImageNet stats if normalize=1,
// otherwise expects the file to already be normalised.
// H and W must match INPUT_H and INPUT_W.
// Returns heap-allocated float array (caller frees).
// ============================================================
static inline float* load_image_bin(const char* path, int C, int H, int W) {
    size_t n = (size_t)C * H * W;
    float* buf = (float*)malloc(n * sizeof(float));
    if (!buf) { fprintf(stderr, "[weights_loader] malloc failed\n"); return NULL; }
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[weights_loader] Cannot open image %s; using zeros.\n", path);
        memset(buf, 0, n * sizeof(float));
        return buf;
    }
    size_t rd = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (rd != n) {
        fprintf(stderr, "[weights_loader] Image file truncated: got %zu / %zu\n", rd, n);
    }
    return buf;
}

// ============================================================
// Convert float32 CHW image to Q8.8 int16 (for FPGA input).
// dst must be pre-allocated with C*H*W elements.
// ============================================================
static inline void image_f32_to_q8_8(const float* src, int16_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        float v = src[i] * 256.0f;              // scale by 2^8 (8 fractional bits)
        if (v >  32767.0f) v =  32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        dst[i] = (int16_t)(v >= 0 ? v + 0.5f : v - 0.5f);
    }
}

#endif // WEIGHTS_LOADER_H
