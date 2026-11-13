# Security policy

YASAKANI runs on your own computer, loads large binary model files, and can listen on a network port. This page says what to report, how to
report it, and what the project does and does not protect.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

* Use GitHub's **private vulnerability reporting**: the *Security* tab of [this repository](https://github.com/yasakani-dev/yasakani/security), then
  *Report a vulnerability*.
* Or e-mail <yasakani0823@gmail.com> with "Security" in the subject.
* This is a one-person project. Expect an acknowledgement within about 7 days, and an honest answer about what can and cannot be fixed.
  If you want credit in the release notes, say so.

Supported versions: the latest release only.

## In scope

* the engine library (`yskn.dll`) and its model-file loader,
* the Python package `yasakani` (command line and HTTP server),
* the release files and their published checksums.

## What the server does and does not protect

The server is for **one user on their own PC**.

What it does:

* Listens on `127.0.0.1` only, unless you pass `--host`. A non-local `--host` is refused unless you also pass `--api-key`; the key is compared in
  constant time.
* The `Host` header must name the address the server is bound to, and a request that carries an `Origin` from another site (or a
  `Sec-Fetch-Site` other than `same-origin` / `none`) is refused. A web page open in your browser therefore cannot use your GPU.
* Request bodies are limited to 32 MB; idle connections time out after 300 s; responses carry `X-Frame-Options: DENY`,
  `frame-ancestors 'none'` and `nosniff`.
* The Python package makes **no outgoing network connections** (it only listens; the tokenizer is read from the local model folder) and runs one external program: `nvidia-smi`, in
  `yasakani info` and `yasakani bench`. It does not execute code the model writes. It does not write prompts or answers to disk or to a log. It writes a file only
  when you run `yasakani selfcheck --write`.
* The engine library (`yskn.dll`) imports no networking functions and starts no other program, and it opens the model files read-only (memory-mapped).
  This was checked on the import table of the release build (2026-11-10, `yskn.dll` sha256 starting `0f5d973b`: it imports only the Windows
  kernel library and the Visual C++ / C runtime) and in its source. The one file it can write is a debugging log of the draft-and-check
  step, and only if the environment variable `YSKN_SPECX_DUMP` names a path (off by default).
* The library reads about a hundred `YSKN_*` environment variables (tuning and debugging switches). The Python package sets the ones it needs and
  removes every other `YSKN_*` variable from its own process before it loads the library (it prints a note when it does), so a stray variable in
  your environment cannot change how `yasakani` runs. A program that loads `yskn.dll` directly has no such protection.

What it does not do:

* **No TLS.** It speaks plain HTTP. If it leaves your LAN, put it behind a VPN or a TLS reverse proxy.
* **One key, no users, no rate limits, no quotas.** Anyone who has the key (or can reach a local-only server from this PC) can use the GPU.
* **One request at a time.** A long request makes the others wait, so any client that can reach the server can keep it busy.
* Do not expose it directly to the internet.

## Model files

`.yskn` files are binary files read by native code. A damaged or deliberately crafted file could crash the process or, in the worst case,
exploit a bug in the loader. Load only the files from the official release, and check them: `yasakani verify` compares every file with the
SHA-256 checksums in `yasakani.json`, and the release page lists the checksum of each download.

## Windows warnings

The release files are not code-signed (this is a one-person project). Windows SmartScreen may warn about the download, and some antivirus
products flag unsigned native code from an unknown publisher. Compare the SHA-256 of what you downloaded with the value on the release page
before you run it. The engine's source is in this repository; you can build `yskn.dll` yourself.
See [BUILDING.md](BUILDING.md).

## What the model itself can do

A language model can produce wrong, biased or unsafe text. YASAKANI does not add safety filters and does not claim to make the output safe.
