/*
 * debug/vae_parity.c — C `iris_vae` vs mflux teacher VAE parity harness (C-1).
 *
 * The flywheel precomputes VAE latents with the mflux Flux2VAE *teacher*, but
 * inference decodes/encodes with the separate C `iris_vae`. If they diverge, the
 * IP-Adapter trains in one latent space and runs in another. This harness pins
 * that invariant on real weights (no GPU; CPU encode).
 *
 * Inputs (raw float32, little-endian, produced by debug/gen_vae_parity_fixture.py):
 *   argv[1] = VAE safetensors path (e.g. flux-klein-model/vae/diffusion_pytorch_model.safetensors)
 *   argv[2] = image  .bin : [3, H, W] CHW in [-1,1]   (here 3x512x512)
 *   argv[3] = teacher.bin : [32, H/8, W/8] pre-patchify (here 32x64x64)
 *   argv[4] = H (default 512), argv[5] = W (default 512)
 *
 * C `iris_vae_encode` returns the PATCHIFIED [128, H/16, W/16] (batch-normed) — the
 * transformer "packed" space. The mflux teacher's encode() / the stored precompute
 * latent is the un-patchified VAE-latent space `(mean-shift)*scale` (no BN). These are
 * two legitimate representations bridged by the VAE's BatchNorm (mflux
 * decode_packed_latents: packed*bn_std + bn_mean -> unpatchify -> VAE-latent). So this
 * harness unpatchifies C's output and reports CORRELATION as the encoder-parity signal:
 * a high corr with a near-constant scale = the encoders agree (the scale is just the
 * BN convention); a low corr = a real encoder divergence.
 *
 * Findings (2026-06-05, image 000004_3398, flux-klein-model):
 *   - BLAS build:    corr 0.99906, scale ~1.76x (BN)  -> encoders AGREE.
 *   - generic build: corr ~ -0.04                     -> non-BLAS VAE encode is BROKEN.
 *
 * CPU-only build (no USE_METAL), mirrors debug/test_vae.c link set. Build with BLAS:
 *   cc -O2 -I. -DUSE_BLAS -DACCELERATE_NEW_LAPACK -o /tmp/vae_parity \
 *      debug/vae_parity.c iris_kernels.c iris_safetensors.c -framework Accelerate -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "iris.h"
#include "iris_vae_config.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include "iris_vae.c"

/* Decode path in iris_vae.c references iris_image_create/free; we only encode, so
 * provide minimal local stubs (as debug/test_vae.c does) to avoid linking the
 * PNG/JPEG-heavy iris_image.c. */
iris_image *iris_image_create(int width, int height, int channels) {
    iris_image *img = (iris_image *)malloc(sizeof(iris_image));
    if (!img) return NULL;
    img->width = width; img->height = height; img->channels = channels;
    img->data = (uint8_t *)calloc((size_t)width * height * channels, 1);
    if (!img->data) { free(img); return NULL; }
    return img;
}
void iris_image_free(iris_image *img) { if (img) { free(img->data); free(img); } }

static float *read_bin(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    float *p = malloc(n * sizeof(float));
    size_t got = fread(p, sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "short read %s: %zu/%zu\n", path, got, n); free(p); return NULL; }
    return p;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s vae.safetensors image.bin teacher.bin [H W]\n", argv[0]);
        return 2;
    }
    const char *vae_path = argv[1], *img_path = argv[2], *teacher_path = argv[3];
    int H = argc > 4 ? atoi(argv[4]) : 512;
    int W = argc > 5 ? atoi(argv[5]) : 512;
    int lh = H / 8, lw = W / 8;                 /* teacher spatial */
    int z = 32;

    /* Resolve scale/shift/z from the VAE's own config.json (next to weights). */
    int z_channels = IRIS_VAE_Z_CHANNELS; float scaling = 0.0f, shift = 0.0f;
    char cfgpath[1024]; snprintf(cfgpath, sizeof(cfgpath), "%.*s/config.json",
        (int)(strrchr(vae_path, '/') ? strrchr(vae_path, '/') - vae_path : 1), vae_path);
    FILE *cf = fopen(cfgpath, "r");
    if (cf) { char b[4096]; size_t n = fread(b,1,sizeof(b)-1,cf); b[n]='\0'; fclose(cf);
              iris_parse_vae_config(b, &z_channels, &scaling, &shift); }
    printf("VAE config: z_channels=%d scaling=%.6f shift=%.6f\n", z_channels, scaling, shift);

    safetensors_file_t *sf = safetensors_open(vae_path);
    if (!sf) { fprintf(stderr, "open VAE failed\n"); return 1; }
    iris_vae_t *vae = iris_vae_load_safetensors_ex(sf, z_channels, scaling, shift);
    safetensors_close(sf);
    if (!vae) { fprintf(stderr, "load VAE failed\n"); return 1; }

    float *img = read_bin(img_path, (size_t)3 * H * W);
    float *teacher = read_bin(teacher_path, (size_t)z * lh * lw);
    if (!img || !teacher) return 1;

    int oh = 0, ow = 0;
    float *lat_patch = iris_vae_encode(vae, img, 1, H, W, &oh, &ow);   /* [128, H/16, W/16] */
    if (!lat_patch) { fprintf(stderr, "encode failed\n"); return 1; }
    printf("C encode -> patchified [%d, %d, %d]\n", vae->latent_channels, oh, ow);

    /* Unpatchify [128, oh, ow] -> [32, oh*2, ow*2] = [32, H/8, W/8]. */
    float *lat_c = malloc((size_t)z * lh * lw * sizeof(float));
    iris_unpatchify(lat_c, lat_patch, 1, z, oh, ow, 2);

    /* Dump C tensors for offline convention analysis (no re-encode needed). */
    FILE *fp = fopen("/tmp/c_patch.bin", "wb");
    if (fp) { fwrite(lat_patch, sizeof(float), (size_t)vae->latent_channels*oh*ow, fp); fclose(fp); }
    FILE *fu = fopen("/tmp/c_unpatch.bin", "wb");
    if (fu) { fwrite(lat_c, sizeof(float), (size_t)z*lh*lw, fu); fclose(fu); }
    { double mn=0, vv=0; size_t nn=(size_t)z*lh*lw;
      for (size_t i=0;i<nn;i++) mn+=lat_c[i]; mn/=nn;
      for (size_t i=0;i<nn;i++){ double d=lat_c[i]-mn; vv+=d*d; }
      printf("C unpatch stats: mean=%.4f std=%.4f\n", mn, sqrt(vv/nn)); }

    /* Compare lat_c vs teacher in [32, lh, lw]. */
    size_t sp = (size_t)lh * lw, n = (size_t)z * sp;
    double max_abs = 0, sse = 0, dot = 0, na = 0, nb = 0;
    double worst_std_ratio_dev = 0; int worst_ch = -1;
    for (int c = 0; c < z; c++) {
        double ma=0, mb=0;
        for (size_t i=0;i<sp;i++){ ma+=lat_c[c*sp+i]; mb+=teacher[c*sp+i]; }
        ma/=sp; mb/=sp;
        double va=0, vb=0;
        for (size_t i=0;i<sp;i++){ double da=lat_c[c*sp+i]-ma, db=teacher[c*sp+i]-mb; va+=da*da; vb+=db*db; }
        double sa=sqrt(va/sp), sb=sqrt(vb/sp);
        double ratio = sb>1e-9 ? sa/sb : 0.0;
        if (fabs(ratio-1.0) > worst_std_ratio_dev) { worst_std_ratio_dev=fabs(ratio-1.0); worst_ch=c; }
    }
    for (size_t i=0;i<n;i++){
        double a=lat_c[i], b=teacher[i], d=fabs(a-b);
        if (d>max_abs) max_abs=d; sse+=d*d; dot+=a*b; na+=a*a; nb+=b*b;
    }
    double rmse = sqrt(sse/n);
    double corr = (na>0&&nb>0) ? dot/(sqrt(na)*sqrt(nb)) : 0.0;
    printf("\n=== C(unpatchified) vs teacher  [%d,%d,%d] ===\n", z, lh, lw);
    printf("  max_abs_err = %.5f\n", max_abs);
    printf("  rmse        = %.5f\n", rmse);
    printf("  cosine_corr = %.5f\n", corr);
    printf("  worst per-channel std ratio dev = %.4f (ch %d)\n", worst_std_ratio_dev, worst_ch);
    /* Encoder parity = high correlation. NOTE the representation gap: this compares
     * C's packed/BN output (unpatchified) against the teacher's *VAE-latent* space
     * (mflux encode() = (mean-shift)*scale, no BN). They legitimately differ by the
     * BatchNorm bridge (~1.76x global scale here), so correlation — not abs error — is
     * the encoder-parity signal. A high corr with a near-constant scale = encoders
     * agree; a low corr = a real encoder divergence. */
    int parity = (corr > 0.99);
    printf("\n  encoder parity (corr>0.99): %s\n", parity ? "PASS" : "FAIL");
    printf("  [scale gap %.3fx is the packed(BN) vs VAE-latent convention, not an "
           "encoder error]\n", corr > 0 ? (na>0? sqrt(nb/na):0) : 0);
    int pass = parity;
    printf("\n%s\n", pass ? "ENCODER PARITY PASS (convention-adjusted)"
                          : "ENCODER MISMATCH (real divergence — investigate impl)");

    free(img); free(teacher); free(lat_patch); free(lat_c);
    iris_vae_free(vae);
    return pass ? 0 : 1;
}
