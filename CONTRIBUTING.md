# Contributing

YASAKANI is a one-person project. Reports, test results and small fixes are welcome; this page says what helps most.
日本語の issue・プルリクエストも歓迎します。

## Bug reports and test results

Open an issue and include:

* the output of `yasakani info --model-dir ... --dll ...` (GPU, driver, file sizes). It does not print your user name, but check the command line you
  paste: a path may contain it. Replace it with `~`;
* what you ran, what you expected and what happened;
* for a crash or a wrong result: whether `yasakani selfcheck` passes.

A result from another GPU (RTX 4090, 3090, 5080, ...) is useful even when it says "does not work": give `yasakani info` and the error text.
Please link large logs and model files instead of pasting them.

For a security problem, do not open an issue: see [SECURITY.md](SECURITY.md).

## Pull requests

* Small fixes (documentation, error messages, tests, portability) are welcome.
* For anything larger (a new feature, another model, a change to the file format or the kernels), open an issue first and say what you want to do.
  One person cannot always take a large change, and it is better to learn that before you spend the time.
* The Python package has unit tests that need no GPU (the engine library is replaced by a small stand-in). Run `python -m unittest` in the package
  folder before you send a change.
* Keep the scope in mind: Windows 11, RTX 5090, Qwen3.8-27B, one request at a time. A change that adds another platform, GPU or model needs its own
  test results to be useful.
* By sending a contribution you agree that it is licensed under the Apache License 2.0 like the rest of the project (see [LICENSE](LICENSE)).
  Do not send code or data you do not have the right to license that way.

## What to expect

I try to read everything, but I do this alone, so a reply can take days. I may decline a change that does not fit the project's small scope; that
is not a judgment of the work.

## Behaviour

Be kind and be specific. Harassment is not tolerated; I may close issues or block users who ignore this.
