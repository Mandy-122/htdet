/*
 * mobilevit_backbone.h  (PTQ INT8 — backbone proj 1x1 dequant scale fix applied)
 *
 * Changes vs mobilevit_backbone.h:
 *   - mobilevit_block_s2/s3/s4 now read proj_1x1 dequant scale[d] from w_meta
 *     (moff advances by d after each proj 1x1 call)
 *   - backbone_meta layout adds proj_1x1_scale[d] per MViT block (3 blocks total):
 *       Stage 2: +144 floats, Stage 3: +192 floats, Stage 4: +240 floats
 *       New BACKBONE_META_ELEMS = 2926768 + 576 = 2927344
 *   - Top-level function renamed mobilevit_backbone
 */

#ifndef MOBILEVIT_BACKBONE_V2_H
#define MOBILEVIT_BACKBONE_V2_H

#include "fpga_types.h"
#include "fpga_utils.h"

void mbconv_160(const act_t* in, act_t* out,
                const weight_t* w_conv, int& coff,
                const meta_t* w_meta, int& moff,
                int in_ch, int out_ch, int expand, int stride);

void mbconv_80(const act_t* in, act_t* out,
               const weight_t* w_conv, int& coff,
               const meta_t* w_meta, int& moff,
               int in_ch, int out_ch, int expand, int stride);

void mbconv_40(const act_t* in, act_t* out,
               const weight_t* w_conv, int& coff,
               const meta_t* w_meta, int& moff,
               int in_ch, int out_ch, int expand, int stride);

void mbconv_20(const act_t* in, act_t* out,
               const weight_t* w_conv, int& coff,
               const meta_t* w_meta, int& moff,
               int in_ch, int out_ch, int expand, int stride);

void mbconv_10(const act_t* in, act_t* out,
               const weight_t* w_conv, int& coff,
               const meta_t* w_meta, int& moff,
               int in_ch, int out_ch, int expand, int stride);

void transformer_block_s2(act_t* tokens, const meta_t* w);
void transformer_block_s3(act_t* tokens, const meta_t* w);
void transformer_block_s4(act_t* tokens, const meta_t* w);

void mobilevit_block_s2(const act_t* input, act_t* output,
                        const weight_t* w_conv, int& coff,
                        const meta_t* w_meta, int& moff);

void mobilevit_block_s3(const act_t* input, act_t* output,
                        const weight_t* w_conv, int& coff,
                        const meta_t* w_meta, int& moff);

void mobilevit_block_s4(const act_t* input, act_t* output,
                        const weight_t* w_conv, int& coff,
                        const meta_t* w_meta, int& moff);

void mobilevit_backbone(
    const input_t*  image,
    const weight_t* backbone_conv,
    const meta_t*   backbone_meta,
    act_t* c1,
    act_t* c2,
    act_t* c3,
    act_t* c4
);

#endif // MOBILEVIT_BACKBONE_V2_H
