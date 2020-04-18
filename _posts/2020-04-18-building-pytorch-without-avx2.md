---
layout: post
title: "Building PyTorch without AVX2 on MacOS"
tags:
- Building
- Software Engineering
- PyTorch
- AVX2
thumbnail_path: blog/thumbs/pytorch.png
pytorch_url: "https://github.com/pytorch/pytorch"
pytorch_fork_url: "https://github.com/dmikushin/pytorch-no-avx2-macos"
---

In order to quickly explore [PyTorch]({{ page.pytorch_url }}) internals, I decided to compile and install a Debug build on my local machine. The first problem was that modern Clang surprisingly crashes on compiling Sobol RNG initial state setup, which is a very regular piece of code:

```
```

Turns out, the crash is likely connected to the fact that `initsobolstate` is a very large static array. A patch that worked (with some additional improvements) basically hides array nature behind a `memcpy` call:

```patch
--- a/aten/src/ATen/native/SobolEngineOps.cpp
+++ b/aten/src/ATen/native/SobolEngineOps.cpp
@@ -129,21 +129,19 @@ Tensor& _sobol_engine_initialize_state_(Tensor& sobolstate, int64_t dimension) {
 
   /// Use a tensor accessor for `sobolstate`
   auto ss_a = sobolstate.accessor<int64_t, 2>();
-  for (int64_t d = 0; d < dimension; ++d) {
+  for (int32_t d = 0; d < dimension; ++d) {
     int64_t p = poly[d];
-    int64_t m = bit_length(p) - 1;
+    int32_t m = bit_length(p) - 1;
 
-    for (int64_t i = 0; i < m; ++i) {
-      ss_a[d][i] = initsobolstate[d][i];
-    }
+    memcpy(&ss_a[d][0], &initsobolstate[d][0], m * sizeof(ss_a[d][0]));
 
-    for (int64_t j = m; j < MAXBIT; ++j) {
+    for (int32_t j = m; j < MAXBIT; ++j) {
       int64_t newv = ss_a[d][j - m];
-      int64_t pow2 = 1;
-      for (int64_t k = 0; k < m; ++k) {
+      int64_t pow2 = 1LL;
+      for (int32_t k = 0; k < m; ++k) {
         pow2 <<= 1;
-        if ((p >> (m - 1 - k)) & 1) {
-          newv = newv ^ (pow2 * ss_a[d][j - k - 1]);
+        if ((p >> (m - 1 - k)) & 1LL) {
+          newv ^= pow2 * ss_a[d][j - k - 1];
         }
       }
       ss_a[d][j] = newv;
```

For the reference, here is a [PyTorch fork]({{ page.pytorch_url }}) that could be used to reproduce my experience.

