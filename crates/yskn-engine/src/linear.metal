// Compressed linear operator, integer-only (docs/spec/02_COMPRESSED_LINEAR.md).
// One SIMD lane owns one output row; the SIMD groups of a threadgroup split
// that row's 256-weight sequences and are summed exactly in i64 at the end.
// The host prepends `#define DECODE_BLOCK<b>` (b = 4..8) before compiling: 32
// straight-line table lookups with constant shifts (`decode_block` in metal.rs).
#include <metal_stdlib>
using namespace metal;

constant uint BITS [[function_constant(0)]];    // bits per step, 4..8
constant uint MC [[function_constant(1)]];      // activation rows in this dispatch, 1..16
constant bool WARP [[function_constant(2)]];    // FORMAT.md 4.3 layout
constant bool ALIGNED [[function_constant(3)]]; // stream start is 4-byte aligned
constant bool FLOAT_MAC [[function_constant(4)]]; // binary32 products and sums (exact, see METAL.md)
constant uint WORDS = 4 * BITS;                 // 32-bit words per sequence

struct Params {
    uint rows;          // output rows computed (a prefix of the physical rows)
    uint physical_rows;
    uint sequences;     // cols / 256
    uint cols;
    uint shift;         // stream start relative to the bound offset, 0..3
};

// All indices were bounded on the CPU: word index < stream_bytes / 4, and the
// bound offset + shift + stream_bytes lies inside the mapped buffer.
static inline uint load_word(device const uchar *codes, uint shift, uint index) {
    if (ALIGNED) {
        return ((device const uint *)codes)[index];
    }
    device const uchar *q = codes + shift + index * 4;
    return uint(q[0]) | (uint(q[1]) << 8) | (uint(q[2]) << 16) | (uint(q[3]) << 24);
}

template <typename T>
static inline void decode(thread T *e, threadgroup const T *lut, thread const uint *wb) {
    if (BITS == 4) {
        DECODE_BLOCK4
    } else if (BITS == 5) {
        DECODE_BLOCK5
    } else if (BITS == 6) {
        DECODE_BLOCK6
    } else if (BITS == 7) {
        DECODE_BLOCK7
    } else {
        DECODE_BLOCK8
    }
}

kernel void weighted_sums(device const uchar *codes [[buffer(0)]],
                          device const ushort *table [[buffer(1)]],      // 4096 byte pairs
                          device const short2 *activation [[buffer(2)]], // MC x cols, widened
                          device const int *scales [[buffer(3)]],        // MC x cols/512
                          device long *output [[buffer(4)]],             // MC x rows
                          constant Params &p [[buffer(5)]],
                          uint2 tile [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint split [[simdgroup_index_in_threadgroup]],
                          uint splits [[simdgroups_per_threadgroup]],
                          uint flat [[thread_index_in_threadgroup]],
                          uint2 size [[threads_per_threadgroup]]) {
    // 16 KB: the table widened to short2 (low byte = first weight), later the reduction.
    threadgroup long shared_memory[2048];
    threadgroup short2 *lut = (threadgroup short2 *)shared_memory;
    threadgroup half2 *lut_f = (threadgroup half2 *)shared_memory;
    // The threadgroup is 32 x splits; a scalar threads_per_threadgroup would be only
    // its width, so each entry would be written `splits` times. flat = y * 32 + x.
    const uint threads = size.x * size.y;
    for (uint i = flat; i < 4096; i += threads) {
        const ushort t = table[i];
        const short2 w = short2(short(as_type<char>(uchar(t & 0xff))), short(as_type<char>(uchar(t >> 8))));
        if (FLOAT_MAC) {
            lut_f[i] = half2(w); // |w| <= 128 is exact in binary16
        } else {
            lut[i] = w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tile.x * 32 + lane;
    const bool live = row < p.rows;
    const uint half_cols = p.cols / 2;
    const uint groups = p.sequences / 2;
    long total[16];
    for (uint m = 0; m < MC; ++m) {
        total[m] = 0;
    }
    for (uint s = split; live && s < p.sequences; s += splits) {
        int dot[16];
        float dot_f[16];
        for (uint m = 0; m < MC; ++m) {
            dot[m] = 0;
            dot_f[m] = 0.0f;
        }
        // A 32-step block spans exactly BITS words, so every block has the same
        // shifts. wb[0] is the previous block's last word (zero before the sequence).
        uint wb[9];
        wb[BITS] = 0;
        for (uint block = 0; block < 4; ++block) {
            wb[0] = wb[BITS];
            for (uint i = 0; i < BITS; ++i) {
                const uint word = block * BITS + i;
                uint index;
                if (WARP) {
                    const uint q = s * p.physical_rows + row;
                    index = (((q / 32) * BITS + word / 4) * 32 + q % 32) * 4 + word % 4;
                } else {
                    index = (row * p.sequences + s) * WORDS + word;
                }
                wb[1 + i] = load_word(codes, p.shift, index);
            }
            const uint pair0 = s * 128 + block * 32;
            if (FLOAT_MAC) {
                half2 e[32];
                decode(e, lut_f, wb);
                device const half2 *activation_f = (device const half2 *)activation;
                for (uint m = 0; m < MC; ++m) {
                    device const half2 *a = activation_f + m * half_cols + pair0;
                    float part = 0.0f;
                    for (uint j = 0; j < 32; ++j) {
                        const half2 x = a[j];
                        // Integer operands and partial sums below 2^24: exact (METAL.md).
                        part = fma(float(e[j].x), float(x.x), part);
                        part = fma(float(e[j].y), float(x.y), part);
                    }
                    dot_f[m] += part;
                }
            } else {
                short2 e[32];
                decode(e, lut, wb);
                for (uint m = 0; m < MC; ++m) {
                    device const short2 *a = activation + m * half_cols + pair0;
                    int part = 0;
                    for (uint j = 0; j < 32; ++j) {
                        const short2 x = a[j];
                        // |w| <= 127 and |a| <= 128: |w0*a0 + w1*a1| <= 32512 is exact in 16 bits.
                        part += int(short(e[j].x * x.x + e[j].y * x.y));
                    }
                    dot[m] += part;
                }
            }
        }
        // Integer distributivity: scaling each sequence's dot separately is exact.
        for (uint m = 0; m < MC; ++m) {
            const int d = FLOAT_MAC ? int(dot_f[m]) : dot[m];
            total[m] += long(scales[m * groups + s / 2]) * long(d);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup); // the table is no longer read
    threadgroup long *partial = shared_memory;
    for (uint m = 0; m < MC; ++m) {
        partial[split * 32 + lane] = total[m];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0 && live) {
            long sum = 0;
            for (uint k = 0; k < splits; ++k) {
                sum += partial[k * 32 + lane];
            }
            output[m * p.rows + row] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void empty_kernel() {}
