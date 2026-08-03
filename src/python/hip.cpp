/*
    hip.cpp -- instantiates the drjit.hip.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "hip.h"
#include "random.h"
#include "texture.h"
#include "event.h"

#if defined(DRJIT_ENABLE_HIP)
void export_hip(nb::module_ &m) {
    using Guide = dr::HIPArray<float>;

    ArrayBinding b;
    dr::bind_all<Guide>(b);
    bind_rng<Guide>(m);

    // bind_texture_all<> instantiates dr::Texture, which for HIP resolves to
    // the software path -- HasGPUTexture excludes this backend deliberately
    // (see the comment in include/drjit/texture.h). Filtered lookups are
    // therefore emulated, which is correct and, at the sub-texel level, more
    // accurate than the hardware units; only the performance question is open.
    bind_texture_all<Guide>(m);

    m.attr("Float32") = m.attr("Float");
    m.attr("Int32") = m.attr("Int");
    m.attr("UInt32") = m.attr("UInt");

    bind_event<JitBackend::HIP>(m, "Event");
}
#endif
