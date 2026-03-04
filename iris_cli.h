/*
 * FLUX Interactive CLI Mode
 */

#ifndef IRIS_CLI_H
#define IRIS_CLI_H

#include "iris.h"

/*
 * Run the interactive CLI. Called when flux is invoked without a prompt.
 * Returns 0 on success, non-zero on error.
 */
int iris_cli_run(iris_ctx *ctx, const char *model_dir);

#endif /* IRIS_CLI_H */
