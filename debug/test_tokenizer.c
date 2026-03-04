/*
 * test_tokenizer.c - Unit tests for the Qwen3 BPE tokenizer (TB-001)
 *
 * Tests qwen3_tokenizer_load / qwen3_tokenize / qwen3_tokenize_chat /
 * qwen3_pad_tokens / qwen3_get_id / qwen3_get_token.
 *
 * Requires the tokenizer JSON from any downloaded model.
 * Searches for it automatically; skips gracefully if not found.
 *
 * Reference token IDs verified against the Qwen3 tokenizer in the
 * flux-klein-model directory.
 *
 * Build:
 *   gcc -O2 -I. -o /tmp/iris_test_tokenizer \
 *       debug/test_tokenizer.c iris_qwen3_tokenizer.c iris_kernels.c -lm
 * Run:
 *   /tmp/iris_test_tokenizer                    # auto-finds tokenizer.json
 *   /tmp/iris_test_tokenizer /path/to/tokenizer.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iris_qwen3.h"

/* qwen3_tokenize is not in the public header but is a non-static symbol */
int *qwen3_tokenize(qwen3_tokenizer_t *tok, const char *text,
                    int *num_tokens, int max_len);

/* =========================================================================
 * Test harness
 * ========================================================================= */

static int failures = 0;
static int passes = 0;
static int skipped = 0;

static void check_true(const char *name, int cond) {
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", name);
        failures++;
    } else {
        printf("PASS %s\n", name);
        passes++;
    }
}

static void check_int(const char *name, int got, int expected) {
    if (got != expected) {
        fprintf(stderr, "FAIL %s: got %d, expected %d\n", name, got, expected);
        failures++;
    } else {
        printf("PASS %s\n", name);
        passes++;
    }
}

/* =========================================================================
 * Helper: find a tokenizer.json automatically
 * ========================================================================= */

static const char *CANDIDATE_PATHS[] = {
    "flux-klein-model/tokenizer/tokenizer.json",
    "flux-klein-4b/tokenizer/tokenizer.json",
    "flux-klein-4b-base/tokenizer/tokenizer.json",
    "flux-klein-9b/tokenizer/tokenizer.json",
    "zimage-turbo/tokenizer/tokenizer.json",
    NULL
};

static const char *find_tokenizer_json(void) {
    for (int i = 0; CANDIDATE_PATHS[i]; i++) {
        FILE *f = fopen(CANDIDATE_PATHS[i], "r");
        if (f) { fclose(f); return CANDIDATE_PATHS[i]; }
    }
    return NULL;
}

/* =========================================================================
 * Tests
 * ========================================================================= */

static void test_load_not_null(qwen3_tokenizer_t *tok) {
    check_true("tokenizer loaded (not null)", tok != NULL);
}

static void test_vocab_size_reasonable(qwen3_tokenizer_t *tok) {
    /* Qwen3 tokenizer has ~152k tokens.
     * Check via public API: token 100000 should exist, 200000 should not. */
    check_true("vocab has id 100000", qwen3_get_token(tok, 100000) != NULL);
    check_true("vocab has id 151644 (im_start)", qwen3_get_token(tok, 151644) != NULL);
    check_true("vocab lacks id 200000", qwen3_get_token(tok, 200000) == NULL);
}

/* -------------------------------------------------------------------------
 * Special token IDs — hard-wired in the C implementation
 * ------------------------------------------------------------------------- */

static void test_special_token_ids(qwen3_tokenizer_t *tok) {
    check_int("im_start id", qwen3_get_id(tok, "<|im_start|>"), 151644);
    check_int("im_end id",   qwen3_get_id(tok, "<|im_end|>"),   151645);
    check_int("eos id",      qwen3_get_id(tok, "<|endoftext|>"), 151643);
    check_int("think_s id",  qwen3_get_id(tok, "<think>"),      151667);
    check_int("think_e id",  qwen3_get_id(tok, "</think>"),     151668);
}

static void test_get_token_roundtrip(qwen3_tokenizer_t *tok) {
    /* id -> string -> id should be identity for known IDs */
    int id = 4616;  /* "cat" */
    const char *s = qwen3_get_token(tok, id);
    check_true("get_token(4616) not null", s != NULL);
    if (s) {
        int back = qwen3_get_id(tok, s);
        check_int("id->token->id roundtrip", back, id);
    }
}

static void test_get_id_unknown(qwen3_tokenizer_t *tok) {
    int id = qwen3_get_id(tok, "THIS_TOKEN_DOES_NOT_EXIST_XYZ");
    check_true("get_id unknown returns -1", id == -1);
}

/* -------------------------------------------------------------------------
 * Single-token words (no BPE merges needed)
 * Reference values from probe binary run against actual tokenizer.json.
 * ------------------------------------------------------------------------- */

static void test_single_token_words(qwen3_tokenizer_t *tok) {
    struct { const char *text; int expected_id; } cases[] = {
        { "cat",   4616  },
        { "dog",   18457 },
        { "a",     64    },
        { "the",   1782  },
        { "Hello", 9707  },
        { NULL, 0 }
    };
    for (int i = 0; cases[i].text; i++) {
        int n;
        int *ids = qwen3_tokenize(tok, cases[i].text, &n, 64);
        char name[64];
        snprintf(name, sizeof(name), "tokenize '%s' -> 1 token", cases[i].text);
        check_int(name, n, 1);
        if (n == 1) {
            snprintf(name, sizeof(name), "tokenize '%s' id=%d", cases[i].text, cases[i].expected_id);
            check_int(name, ids[0], cases[i].expected_id);
        }
        free(ids);
    }
}

/* -------------------------------------------------------------------------
 * Multi-token sequences
 * "cat and dog" -> [4616=cat, 323=and, 5562=Ġdog] (space before dog)
 * "a cat"       -> [64=a, 8251=Ġcat]
 * ------------------------------------------------------------------------- */

static void test_multi_token_sequence(qwen3_tokenizer_t *tok) {
    {
        int n;
        int *ids = qwen3_tokenize(tok, "cat and dog", &n, 64);
        check_int("'cat and dog' token count", n, 3);
        if (n == 3) {
            check_int("'cat and dog' tok[0]=cat",     ids[0], 4616);
            check_int("'cat and dog' tok[1]= and",    ids[1], 323);
            check_int("'cat and dog' tok[2]= dog",    ids[2], 5562);
        }
        free(ids);
    }
    {
        int n;
        int *ids = qwen3_tokenize(tok, "a cat", &n, 64);
        check_int("'a cat' token count", n, 2);
        if (n == 2) {
            check_int("'a cat' tok[0]=a",    ids[0], 64);
            check_int("'a cat' tok[1]= cat", ids[1], 8251);
        }
        free(ids);
    }
}

/* -------------------------------------------------------------------------
 * max_len truncation
 * ------------------------------------------------------------------------- */

static void test_max_len_truncation(qwen3_tokenizer_t *tok) {
    int n;
    int *ids = qwen3_tokenize(tok, "cat and dog", &n, 2);
    check_true("max_len=2 truncates to <=2", n <= 2);
    free(ids);
}

/* -------------------------------------------------------------------------
 * Empty string does not crash and returns 0 tokens
 * ------------------------------------------------------------------------- */

static void test_empty_string(qwen3_tokenizer_t *tok) {
    int n = -1;
    int *ids = qwen3_tokenize(tok, "", &n, 64);
    check_true("empty string: no crash", 1);
    check_true("empty string: n==0", n == 0);
    free(ids);
}

/* -------------------------------------------------------------------------
 * Chat template structure (Flux mode: skip_think_tags=0)
 *
 * Expected sequence for "a cat":
 *   151644  <|im_start|>
 *   872     "user"  (first token of "user\n")
 *   198     "\n"    (second token)
 *   64      "a"
 *   8251    " cat"
 *   151645  <|im_end|>
 *   198     "\n"
 *   151644  <|im_start|>
 *   77091   "assistant" (first token of "assistant\n")
 *   198     "\n"
 *   151667  <think>
 *   271     "\n\n"
 *   151668  </think>
 *   271     "\n\n"
 *   total:  14 tokens
 * ------------------------------------------------------------------------- */

static void test_chat_template_flux(qwen3_tokenizer_t *tok) {
    int n;
    int *ids = qwen3_tokenize_chat(tok, "a cat", &n, 512, 0);
    check_int("chat flux token count", n, 14);
    if (n >= 14) {
        check_int("chat flux tok[0]=im_start",    ids[0],  151644);
        check_int("chat flux tok[4]= cat",         ids[4],  8251);
        check_int("chat flux tok[5]=im_end",       ids[5],  151645);
        check_int("chat flux tok[7]=im_start",     ids[7],  151644);
        check_int("chat flux tok[10]=think_start", ids[10], 151667);
        check_int("chat flux tok[12]=think_end",   ids[12], 151668);
    }
    free(ids);
}

/* Chat template — Z-Image mode (skip_think_tags=1): no <think> blocks */
static void test_chat_template_zimage(qwen3_tokenizer_t *tok) {
    int n;
    int *ids = qwen3_tokenize_chat(tok, "a cat", &n, 512, 1);
    check_int("chat zimage token count", n, 10);
    if (n >= 10) {
        check_int("chat zimage tok[0]=im_start", ids[0], 151644);
        check_int("chat zimage tok[5]=im_end",   ids[5], 151645);
        check_int("chat zimage tok[7]=im_start", ids[7], 151644);
        /* No think tokens in Z-Image mode */
        int has_think = 0;
        for (int i = 0; i < n; i++)
            if (ids[i] == 151667 || ids[i] == 151668) has_think = 1;
        check_true("chat zimage no think tokens", !has_think);
    }
    free(ids);
}

/* The chat template must start with <|im_start|> regardless of prompt */
static void test_chat_template_always_starts_with_im_start(qwen3_tokenizer_t *tok) {
    const char *prompts[] = { "x", "a fluffy cat", "hello world", NULL };
    for (int i = 0; prompts[i]; i++) {
        int n;
        int *ids = qwen3_tokenize_chat(tok, prompts[i], &n, 512, 0);
        char name[64];
        snprintf(name, sizeof(name), "chat starts with im_start '%s'", prompts[i]);
        check_true(name, n > 0 && ids[0] == 151644);
        free(ids);
    }
}

/* -------------------------------------------------------------------------
 * qwen3_pad_tokens
 * "cat" -> 1 token [4616]; padded to 8 -> [4616, 151643×7]
 * mask: [1, 0, 0, 0, 0, 0, 0, 0]
 * PAD_ID = 151643 (<|endoftext|>)
 * ------------------------------------------------------------------------- */

static void test_pad_tokens(qwen3_tokenizer_t *tok) {
    int n;
    int *ids = qwen3_tokenize(tok, "cat", &n, 64);
    check_int("pad: 'cat' tokenizes to 1 token", n, 1);

    int mask[8] = {0};
    int *padded = qwen3_pad_tokens(ids, n, 8, mask);
    check_true("pad: result not null", padded != NULL);
    if (padded) {
        check_int("pad: tok[0]=cat",    padded[0], 4616);
        check_int("pad: tok[1]=pad_id", padded[1], 151643);
        check_int("pad: tok[7]=pad_id", padded[7], 151643);
        check_int("pad: mask[0]=1",     mask[0], 1);
        check_int("pad: mask[1]=0",     mask[1], 0);
        check_int("pad: mask[7]=0",     mask[7], 0);
        free(padded);
    }
    free(ids);
}

static void test_pad_tokens_exact_length(qwen3_tokenizer_t *tok) {
    /* Padding to exact length: no extra pads needed */
    int n;
    int *ids = qwen3_tokenize(tok, "cat", &n, 64);
    int mask[1];
    int *padded = qwen3_pad_tokens(ids, n, 1, mask);
    check_true("pad exact: not null", padded != NULL);
    if (padded) {
        check_int("pad exact: tok[0]=cat", padded[0], 4616);
        check_int("pad exact: mask[0]=1",  mask[0],   1);
        free(padded);
    }
    free(ids);
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(int argc, char **argv) {
    printf("=== Qwen3 tokenizer unit tests (TB-001) ===\n\n");

    /* Find the tokenizer JSON */
    const char *tok_path = (argc >= 2) ? argv[1] : find_tokenizer_json();
    if (!tok_path) {
        printf("SKIP: no tokenizer.json found (download a model first)\n");
        printf("      searched: flux-klein-model/, flux-klein-4b/, flux-klein-4b-base/, ...\n");
        printf("      pass path explicitly: %s /path/to/tokenizer.json\n",
               argc >= 1 ? argv[0] : "test_tokenizer");
        skipped = 1;
        return 0;
    }
    printf("Using tokenizer: %s\n\n", tok_path);

    qwen3_tokenizer_t *tok = qwen3_tokenizer_load(tok_path);

    printf("-- load --\n");
    test_load_not_null(tok);
    if (!tok) {
        fprintf(stderr, "Cannot continue without tokenizer\n");
        return 1;
    }
    test_vocab_size_reasonable(tok);

    printf("\n-- special tokens --\n");
    test_special_token_ids(tok);
    test_get_token_roundtrip(tok);
    test_get_id_unknown(tok);

    printf("\n-- single-token words --\n");
    test_single_token_words(tok);

    printf("\n-- multi-token sequences --\n");
    test_multi_token_sequence(tok);

    printf("\n-- edge cases --\n");
    test_max_len_truncation(tok);
    test_empty_string(tok);

    printf("\n-- chat template --\n");
    test_chat_template_flux(tok);
    test_chat_template_zimage(tok);
    test_chat_template_always_starts_with_im_start(tok);

    printf("\n-- pad_tokens --\n");
    test_pad_tokens(tok);
    test_pad_tokens_exact_length(tok);

    qwen3_tokenizer_free(tok);

    printf("\n");
    if (failures > 0) {
        fprintf(stderr, "%d/%d FAILED\n", failures, passes + failures);
        return 1;
    }
    printf("All %d tokenizer tests PASSED\n", passes);
    return 0;
}
