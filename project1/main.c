
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #include <cpuid.h>
  #include <immintrin.h>
#elif defined(__aarch64__)
  #include <arm_acle.h>
  #include <sys/auxv.h>
  #include <arm_neon.h>
  #ifndef HWCAP_PMULL
    #define HWCAP_PMULL (1<<4)
  #endif
  #ifndef HWCAP2_SM4
    #define HWCAP2_SM4 (1<<2) 
  #endif
#endif


static inline uint32_t ROL32_c(uint32_t x, int r){ return (x<<r) | (x>>(32-r)); }
#if defined(SM4_ENABLE_AVX2) || defined(SM4_ENABLE_GFNI)
  #if defined(__AVX512VL__) && defined(__AVX512F__)
    #define HAVE_VPROLD 1
    static inline __m128i ROL32_v(__m128i x, int r){ return _mm_rol_epi32(x, r); }
  #elif defined(__AVX2__)
    static inline __m128i ROL32_v(__m128i x, int r){
        __m128i l = _mm_slli_epi32(x, r);
        __m128i rr= _mm_srli_epi32(x, 32-r);
        return _mm_or_si128(l, rr);
    }
  #endif
#endif


static const uint32_t FK[4] = {0xa3b1bac6U,0x56aa3350U,0x677d9197U,0xb27022dcU};
static const uint32_t CK[32]={
  0x00070e15U,0x1c232a31U,0x383f464dU,0x545b6269U,0x70777e85U,0x8c939aa1U,0xa8afb6bdU,0xc4cbd2d9U,
  0xe0e7eef5U,0xfc030a11U,0x181f262dU,0x343b4249U,0x50575e65U,0x6c737a81U,0x888f969dU,0xa4abb2b9U,
  0xc0c7ced5U,0xdce3eaf1U,0xf8ff060dU,0x141b2229U,0x30373e45U,0x4c535a61U,0x686f767dU,0x848b9299U,
  0xa0a7aeb5U,0xbcc3cad1U,0xd8dfe6edU,0xf4fb0209U,0x10171e25U,0x2c333a41U,0x484f565dU,0x646b7279U
};

static const uint8_t Sbox[256]={
0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05,
0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99,
0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62,
0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6,
0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8,
0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35,
0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87,
0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e,
0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1,
0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3,
0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f,
0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51,
0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8,
0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0,
0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84,
0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48
};

static inline uint32_t U8x4_to_u32(const uint8_t *p){
    return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|((uint32_t)p[3]);
}
static inline void u32_to_U8x4(uint32_t v, uint8_t *p){
    p[0]=v>>24; p[1]=(v>>16)&0xff; p[2]=(v>>8)&0xff; p[3]=v&0xff;
}



static inline uint32_t tau_sc(uint32_t x){
    return ((uint32_t)Sbox[x>>24] << 24)
         | ((uint32_t)Sbox[(x>>16)&0xff] << 16)
         | ((uint32_t)Sbox[(x>>8)&0xff]  << 8)
         |  (uint32_t)Sbox[x&0xff];
}
static inline uint32_t L_enc_sc(uint32_t b){
    return b ^ ROL32_c(b,2) ^ ROL32_c(b,10) ^ ROL32_c(b,18) ^ ROL32_c(b,24);
}
static inline uint32_t Lp_key_sc(uint32_t b){ /* 密钥扩展线性层 */
    return b ^ ROL32_c(b,13) ^ ROL32_c(b,23);
}
static inline uint32_t T_enc_sc(uint32_t x){ return L_enc_sc(tau_sc(x)); }

typedef struct { uint32_t rk[32]; } sm4_roundkeys;

/* key schedule */
static void sm4_key_schedule(const uint8_t key[16], sm4_roundkeys* rks){
    uint32_t K[4];
    for(int i=0;i<4;i++){ K[i]=U8x4_to_u32(key+4*i)^FK[i]; }
    for(int i=0;i<32;i++){
        uint32_t t = K[1]^K[2]^K[3]^CK[i];
        uint32_t b = tau_sc(t);
        uint32_t rk= K[0] ^ Lp_key_sc(b);
        rks->rk[i]=rk;
        K[0]=K[1]; K[1]=K[2]; K[2]=K[3]; K[3]=rk;
    }
}

static void sm4_encrypt_block_sc(const sm4_roundkeys* rks, const uint8_t in[16], uint8_t out[16]){
    uint32_t X[4];
    for(int i=0;i<4;i++) X[i]=U8x4_to_u32(in+4*i);
    for(int i=0;i<32;i++){
        uint32_t t = X[1]^X[2]^X[3]^rks->rk[i];
        uint32_t y = T_enc_sc(t);
        uint32_t n = X[0]^y;
        X[0]=X[1]; X[1]=X[2]; X[2]=X[3]; X[3]=n;
    }
    for(int i=0;i<4;i++) u32_to_U8x4(X[3-i], out+4*i);
}
static void sm4_decrypt_block_sc(const sm4_roundkeys* rks, const uint8_t in[16], uint8_t out[16]){
    uint32_t X[4];
    for(int i=0;i<4;i++) X[i]=U8x4_to_u32(in+4*i);
    for(int i=0;i<32;i++){
        uint32_t t = X[1]^X[2]^X[3]^rks->rk[31-i];
        uint32_t y = T_enc_sc(t);
        uint32_t n = X[0]^y;
        X[0]=X[1]; X[1]=X[2]; X[2]=X[3]; X[3]=n;
    }
    for(int i=0;i<4;i++) u32_to_U8x4(X[3-i], out+4*i);
}


#ifdef SM4_ENABLE_TTABLE
static uint32_t T0[256],T1[256],T2[256],T3[256];
static void sm4_build_T_tables(void){
    static int inited=0; if(inited) return; inited=1;
    for(int x=0;x<256;x++){
        uint32_t b=Sbox[x];
        uint32_t B0=b<<24, B1=b<<16, B2=b<<8, B3=b;
        uint32_t L0=L_enc_sc(B0),L1=L_enc_sc(B1),L2=L_enc_sc(B2),L3=L_enc_sc(B3);
        T0[x]=L0; T1[x]=L1; T2[x]=L2; T3[x]=L3;
    }
}
static inline uint32_t T_tab(uint32_t a){
    return T0[a>>24]^T1[(a>>16)&0xff]^T2[(a>>8)&0xff]^T3[a&0xff];
}
static void sm4_encrypt_block_ttab(const sm4_roundkeys* rks,const uint8_t in[16],uint8_t out[16]){
    sm4_build_T_tables();
    uint32_t X[4];
    for(int i=0;i<4;i++) X[i]=U8x4_to_u32(in+4*i);
    for(int i=0;i<32;i++){
        uint32_t t = X[1]^X[2]^X[3]^rks->rk[i];
        uint32_t y = T_tab(t);
        uint32_t n = X[0]^y;
        X[0]=X[1]; X[1]=X[2]; X[2]=X[3]; X[3]=n;
    }
    for(int i=0;i<4;i++) u32_to_U8x4(X[3-i], out+4*i);
}
#endif /* SM4_ENABLE_TTABLE */


#ifdef SM4_ENABLE_GFNI
static const uint64_t SM4_A1 = 0xA1D5D5BB3A1A58C7ULL; 
static const uint64_t SM4_A2 = 0xB66C7F0D3A6D5B97ULL;
static const uint8_t  SM4_C1 = 0xD3;
static const uint8_t  SM4_C2 = 0x3E;

static inline __m128i sm4_sbox_gfni_16(__m128i x){
    __m128i C1 = _mm_set1_epi8((char)SM4_C1);
    __m128i C2 = _mm_set1_epi8((char)SM4_C2);
    __m128i y  = _mm_xor_si128(x, C1);
    y = _mm_gf2p8affineinv_epi64_epi8(y, _mm_set1_epi64x((long long)SM4_A1), 0);
    y = _mm_gf2p8affine_epi64_epi8   (y, _mm_set1_epi64x((long long)SM4_A2), 0);
    y = _mm_xor_si128(y, C2);
    return y;
}
static inline __m128i sm4_L_16(__m128i b){
    __m128i r2  = ROL32_v(b,2);
    __m128i r10 = ROL32_v(b,10);
    __m128i r18 = ROL32_v(b,18);
    __m128i r24 = ROL32_v(b,24);
    return _mm_xor_si128(_mm_xor_si128(b, r2), _mm_xor_si128(_mm_xor_si128(r10, r18), r24));
}
#endif



typedef struct {
    void (*encrypt_block)(const sm4_roundkeys*, const uint8_t*, uint8_t*);
    void (*decrypt_block)(const sm4_roundkeys*, const uint8_t*, uint8_t*);
    void (*ctr_crypt)(const sm4_roundkeys*, uint8_t ctr[16], const uint8_t* in, uint8_t* out, size_t len);
} sm4_vtable;

static sm4_vtable g_vt; 

static inline void ctr_inc32_be(uint8_t ctr[16]){
   
    for(int i=15;i>=12;i--){ uint8_t x=ctr[i]+1; ctr[i]=x; if(x) break; }
}


#ifdef SM4_ENABLE_AVX2
static void sm4_encrypt_block_sc_batch4(const sm4_roundkeys* rks, const uint8_t* in, uint8_t* out){
 
    for(int i=0;i<4;i++) sm4_encrypt_block_sc(rks, in+16*i, out+16*i);
}
static void sm4_ctr_crypt_avx2(const sm4_roundkeys* rks, uint8_t ctr[16],
                               const uint8_t* in, uint8_t* out, size_t len){
    size_t i=0;
    uint8_t ctrblk[64], ks[64];
    while(len - i >= 64){
        uint8_t c[16];
        for(int j=0;j<4;j++){
            memcpy(c, ctr, 16);
            ctr_inc32_be(ctr);
            memcpy(ctrblk+16*j, c, 16);
        }
        sm4_encrypt_block_sc_batch4(rks, ctrblk, ks);
        for(int j=0;j<64;j++) out[i+j] = in[i+j]^ks[j];
        i+=64;
    }
    while(i < len){
        uint8_t s[16], c[16];
        memcpy(c, ctr, 16); ctr_inc32_be(ctr);
        sm4_encrypt_block_sc(rks, c, s);
        size_t n = (len-i<16)?(len-i):16;
        for(size_t k=0;k<n;k++) out[i+k] = in[i+k]^s[k];
        i+=n;
    }
}
#endif


static void sm4_ctr_crypt_sc(const sm4_roundkeys* rks, uint8_t ctr[16],
                             const uint8_t* in, uint8_t* out, size_t len){
    size_t i=0;
    while(i<len){
        uint8_t s[16], c[16];
        memcpy(c, ctr, 16); ctr_inc32_be(ctr);
        sm4_encrypt_block_sc(rks, c, s);
        size_t n = (len-i<16)?(len-i):16;
        for(size_t k=0;k<n;k++) out[i+k]=in[i+k]^s[k];
        i+=n;
    }
}



typedef struct { uint8_t b[16]; } block128;


static inline __m128i loadu_be128(const uint8_t *p){
    __m128i x = _mm_loadu_si128((const __m128i*)p);
 
    return x;
}
static inline void storeu_be128(uint8_t *p, __m128i x){ _mm_storeu_si128((__m128i*)p, x); }

#if defined(SM4_ENABLE_PCLMUL) && (defined(__x86_64__)||defined(__i386__))

static inline __m128i ghash_mul_pclmul(__m128i X, __m128i H){
 
    __m128i Xl = _mm_clmulepi64_si128(X, H, 0x00);
    __m128i Xh = _mm_clmulepi64_si128(X, H, 0x11);
    __m128i Xm = _mm_xor_si128(_mm_srli_si128(X,8), _mm_shuffle_epi32(X, _MM_SHUFFLE(1,0,3,2)));
    __m128i Hm = _mm_xor_si128(_mm_srli_si128(H,8), _mm_shuffle_epi32(H, _MM_SHUFFLE(1,0,3,2)));
    __m128i Xk = _mm_clmulepi64_si128(Xm, Hm, 0x00);
    __m128i mid= _mm_xor_si128(_mm_xor_si128(Xk, Xl), Xh);
    __m128i t0 = _mm_xor_si128(_mm_slli_si128(Xh, 8), _mm_slli_si128(mid, 4));
    __m128i t1 = _mm_xor_si128(_mm_srli_si128(mid, 12), _mm_srli_si128(Xh, 8));
    __m128i z0 = _mm_xor_si128(Xl, t0);
    __m128i z1 = t1;

 
    __m128i v = z1;
    __m128i r = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(v, _mm_srli_epi64(v, 1)),
                                             _mm_srli_epi64(v, 2)),
                               _mm_srli_epi64(v, 7));
    r = _mm_xor_si128(_mm_xor_si128(r, _mm_slli_si128(v,8)),
                      _mm_xor_si128(_mm_slli_si128(_mm_srli_epi64(v,1),8),
                                    _mm_xor_si128(_mm_slli_si128(_mm_srli_epi64(v,2),8),
                                                  _mm_slli_si128(_mm_srli_epi64(v,7),8))));
    return _mm_xor_si128(z0, r);
}
#elif defined(SM4_ENABLE_PMULL) && defined(__aarch64__)
/* ARM: PMULL GHASH */
static inline uint8x16_t ghash_mul_pmull(uint8x16_t X, uint8x16_t H){
 
    poly64x2_t x = vreinterpretq_p64_u8(X);
    poly64x2_t h = vreinterpretq_p64_u8(H);
    poly128_t xh = vmull_high_p64(vget_low_p64(x), vget_low_p64(h)); /* 仅示意，详实实现需完整 Karatsuba 合成与约减 */
    poly128_t xl = vmull_p64(vget_low_p64(x), vget_low_p64(h));
  
    uint8x16_t z = vreinterpretq_u8_p128(veorq_p128(xh, xl));
    return z;
}
#else

static void ghash_mul_c(uint8_t Y[16], const uint8_t H[16]){
    uint8_t Z[16]={0}, V[16]; memcpy(V,H,16);
    for(int i=0;i<16;i++){
        uint8_t x = Y[i];
        for(int b=0;b<8;b++){
            if(x & 0x80) for(int k=0;k<16;k++) Z[k]^=V[k];
            /* V = gf_mul_x(V) mod (x^128 + x^7 + x^2 + x + 1) */
            uint8_t lsb = V[15]&1;
            for(int k=15;k>0;k--) V[k]=(V[k]>>1)|((V[k-1]&1)<<7);
            V[0]>>=1;
            if(lsb){
                V[0]^=0xe1;
            }
            x <<= 1;
        }
    }
    memcpy(Y,Z,16);
}
#endif


typedef struct {
    sm4_roundkeys rks;
    uint8_t H[16];    
} sm4_gcm_ctx;

static void sm4_encrypt_block_dispatch(const sm4_roundkeys* rks, const uint8_t in[16], uint8_t out[16]){
#if defined(SM4_ENABLE_TTABLE)
    sm4_encrypt_block_ttab(rks, in, out);
#else
    sm4_encrypt_block_sc(rks, in, out);
#endif
}

static void sm4_init(sm4_gcm_ctx* ctx, const uint8_t key[16]){
    sm4_key_schedule(key, &ctx->rks);
    uint8_t zero[16]={0};
    sm4_encrypt_block_dispatch(&ctx->rks, zero, ctx->H);
}

static void ghash_update(const uint8_t H[16], uint8_t Y[16], const uint8_t* data, size_t len){
    size_t i=0;
    while(i+16<=len){
        for(int k=0;k<16;k++) Y[k]^=data[i+k];
#if defined(SM4_ENABLE_PCLMUL) && (defined(__x86_64__)||defined(__i386__))
        __m128i y = loadu_be128(Y), h = loadu_be128(H);
        y = ghash_mul_pclmul(y,h);
        storeu_be128(Y,y);
#elif defined(SM4_ENABLE_PMULL) && defined(__aarch64__)
        uint8x16_t y = vld1q_u8(Y), h = vld1q_u8(H);
        y = ghash_mul_pmull(y,h);
        vst1q_u8(Y,y);
#else
        ghash_mul_c(Y,H);
#endif
        i+=16;
    }
    if(i<len){
        uint8_t last[16]={0};
        size_t n=len-i;
        memcpy(last, data+i, n);
        for(int k=0;k<16;k++) Y[k]^=last[k];
#if defined(SM4_ENABLE_PCLMUL) && (defined(__x86_64__)||defined(__i386__))
        __m128i y = loadu_be128(Y), h = loadu_be128(H);
        y = ghash_mul_pclmul(y,h);
        storeu_be128(Y,y);
#elif defined(SM4_ENABLE_PMULL) && defined(__aarch64__)
        uint8x16_t y = vld1q_u8(Y), h = vld1q_u8(H);
        y = ghash_mul_pmull(y,h);
        vst1q_u8(Y,y);
#else
        ghash_mul_c(Y,H);
#endif
    }
}

static void sm4_gcm_encrypt(const uint8_t key[16], const uint8_t* iv, size_t ivlen,
                            const uint8_t* aad, size_t aadlen,
                            const uint8_t* pt, size_t ptlen,
                            uint8_t* ct, uint8_t tag[16])
{
    sm4_gcm_ctx ctx; sm4_init(&ctx, key);
    uint8_t J0[16]={0}, Y[16]={0};

    if(ivlen==12){
        memcpy(J0, iv, 12); J0[15]=1;
    }else{
      
        ghash_update(ctx.H, Y, iv, ivlen);
        uint8_t lenblk[16]={0};
        uint64_t ivbits = (uint64_t)ivlen * 8;
      
        lenblk[8]=(ivbits>>56)&0xFF; lenblk[9]=(ivbits>>48)&0xFF;
        lenblk[10]=(ivbits>>40)&0xFF;lenblk[11]=(ivbits>>32)&0xFF;
        lenblk[12]=(ivbits>>24)&0xFF;lenblk[13]=(ivbits>>16)&0xFF;
        lenblk[14]=(ivbits>>8)&0xFF; lenblk[15]=(ivbits)&0xFF;
        ghash_update(ctx.H, Y, lenblk, 16);
        memcpy(J0, Y, 16);
        memset(Y,0,16);
    }

 
#if defined(SM4_ENABLE_AVX2)
    sm4_ctr_crypt_avx2(&ctx.rks, J0, pt, ct, ptlen);
#else
    sm4_ctr_crypt_sc(&ctx.rks, J0, pt, ct, ptlen);
#endif

   
    memset(Y,0,16);
    if(aad && aadlen) ghash_update(ctx.H, Y, aad, aadlen);
    if(ct  && ptlen)  ghash_update(ctx.H, Y, ct , ptlen);

   
    uint8_t lenblk[16]={0};
    uint64_t a_bits=(uint64_t)aadlen*8, c_bits=(uint64_t)ptlen*8;
  
    lenblk[0]=(a_bits>>56)&0xFF; lenblk[1]=(a_bits>>48)&0xFF;
    lenblk[2]=(a_bits>>40)&0xFF; lenblk[3]=(a_bits>>32)&0xFF;
    lenblk[4]=(a_bits>>24)&0xFF; lenblk[5]=(a_bits>>16)&0xFF;
    lenblk[6]=(a_bits>>8)&0xFF;  lenblk[7]=(a_bits)&0xFF;
    lenblk[8]=(c_bits>>56)&0xFF; lenblk[9]=(c_bits>>48)&0xFF;
    lenblk[10]=(c_bits>>40)&0xFF;lenblk[11]=(c_bits>>32)&0xFF;
    lenblk[12]=(c_bits>>24)&0xFF;lenblk[13]=(c_bits>>16)&0xFF;
    lenblk[14]=(c_bits>>8)&0xFF; lenblk[15]=(c_bits)&0xFF;
    ghash_update(ctx.H, Y, lenblk, 16);

    
    uint8_t J0base[16];
    if(ivlen==12){ memcpy(J0base, iv, 12); J0base[15]=1; }
    else { memcpy(J0base, J0, 16); } 
    uint8_t EkJ0[16];
    sm4_encrypt_block_dispatch(&ctx.rks, J0base, EkJ0);
    for(int i=0;i<16;i++) tag[i]=EkJ0[i]^Y[i];
}


typedef struct {
    int avx2, pclmul, vpclmul, gfni, pmull, sm4ni;
} cpu_features;

static cpu_features detect_features(void){
    cpu_features f={0};
#if defined(__x86_64__)||defined(__i386__)
    unsigned eax, ebx, ecx, edx;
    if(__get_cpuid(1, &eax,&ebx,&ecx,&edx)){
        f.pclmul = (ecx & bit_PCLMUL) != 0;
    }
    if(__get_cpuid_count(7,0,&eax,&ebx,&ecx,&edx)){
        f.avx2    = (ebx & bit_AVX2)!=0;
        f.gfni    = (ecx & bit_GFNI)!=0;
        f.vpclmul = (ecx & bit_VPCLMULQDQ)!=0;
    }
#elif defined(__aarch64__)
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    f.pmull = (hwcap & HWCAP_PMULL)!=0;
    f.sm4ni = (hwcap2 & HWCAP2_SM4)!=0;
#endif
    return f;
}

static void init_vtable(void){
    cpu_features f = detect_features();
    g_vt.encrypt_block = sm4_encrypt_block_sc;
    g_vt.decrypt_block = sm4_decrypt_block_sc;
#if defined(SM4_ENABLE_AVX2)
    if(f.avx2){
        g_vt.ctr_crypt = sm4_ctr_crypt_avx2;
    }else
#endif
    {
        g_vt.ctr_crypt = sm4_ctr_crypt_sc;
    }
 
}



void sm4_key_expand(const uint8_t key[16], sm4_roundkeys* rks){ sm4_key_schedule(key, rks); }
void sm4_encrypt_block(const sm4_roundkeys* rks, const uint8_t in[16], uint8_t out[16]){
#if defined(SM4_ENABLE_TTABLE)
    sm4_encrypt_block_ttab(rks, in, out);
#else
    sm4_encrypt_block_sc(rks, in, out);
#endif
}
void sm4_decrypt_block(const sm4_roundkeys* rks, const uint8_t in[16], uint8_t out[16]){
    sm4_decrypt_block_sc(rks, in, out);
}
void sm4_ctr_crypt(const sm4_roundkeys* rks, uint8_t ctr[16], const uint8_t* in, uint8_t* out, size_t len){
    if(!g_vt.ctr_crypt) init_vtable();
    g_vt.ctr_crypt(rks, ctr, in, out, len);
}
void sm4_gcm_encrypt_all(const uint8_t key[16], const uint8_t* iv, size_t ivlen,
                         const uint8_t* aad, size_t aadlen,
                         const uint8_t* pt, size_t ptlen,
                         uint8_t* ct, uint8_t tag[16]){
    if(!g_vt.ctr_crypt) init_vtable();
    sm4_gcm_encrypt(key, iv, ivlen, aad, aadlen, pt, ptlen, ct, tag);
}


#ifdef SM4_TEST_MAIN
static void hex(const char* cap, const uint8_t* p, size_t n){
    printf("%s: ", cap);
    for(size_t i=0;i<n;i++) printf("%02x", p[i]);
    puts("");
}
int main(void){
    init_vtable();
    
    uint8_t key[16] = {
        0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10
    };
    uint8_t pt[16]  = {0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10};
    uint8_t exp[16] = {0x68,0x1e,0xdf,0x34,0xd2,0x06,0x96,0x5e,0x86,0xb3,0xe9,0x4f,0x53,0x6e,0x42,0x46};
    sm4_roundkeys rk; sm4_key_expand(key, &rk);
    uint8_t ct[16]; sm4_encrypt_block(&rk, pt, ct);
    hex("SM4-Enc", ct, 16);
    int ok = (memcmp(ct,exp,16)==0);
    printf("KAT: %s\n", ok?"OK":"FAIL");

   
    uint8_t iv[12]={0x12,0x34,0x56,0x78,0x90,0xab,0xcd,0xef,0x00,0x00,0x00,0x01};
    uint8_t aad[20]; for(int i=0;i<20;i++) aad[i]=(uint8_t)i;
    uint8_t msg[64]; for(int i=0;i<64;i++) msg[i]=(uint8_t)(i*3+1);
    uint8_t out[64], tag[16];
    sm4_gcm_encrypt_all(key, iv, sizeof(iv), aad, sizeof(aad), msg, sizeof(msg), out, tag);
    hex("GCM-CT", out, 64); hex("GCM-TAG", tag, 16);
    return ok?0:1;
}
#endif
