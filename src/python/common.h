/*
    common.h -- Common definitions used by the Dr.Jit Python bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include <drjit/python.h>
#include <nanobind/stl/pair.h>
#include "docstr.h"

namespace nb = nanobind;
namespace dr = drjit;

using nb::literals::operator""_a;

using dr::ArrayMeta;
using dr::ArraySupplement;
using dr::ArrayBinding;
using dr::ArrayOp;
using dr::ArrayBase;
using dr::vector;

inline const ArraySupplement &supp(nb::handle h) {
    return nb::type_supplement<ArraySupplement>(h);
}

inline ArrayBase* inst_ptr(nb::handle h) {
    return nb::inst_ptr<ArrayBase>(h);
}

/// Helper function to perform a tuple-based function call directly using the
/// CPython API. nanobind lacks a nice abstraction for this.
inline nb::object tuple_call(nb::handle callable, nb::handle tuple) {
    nb::object result = nb::steal(PyObject_CallObject(callable.ptr(), tuple.ptr()));
    if (!result.is_valid())
        nb::raise_python_error();
    return result;
}

#define raise_if(expr, ...)                                                    \
    do {                                                                       \
        if (NB_UNLIKELY(expr))                                                 \
            nb::raise(__VA_ARGS__);                                    \
    } while (false)

/// Create interned string for a few very commonly used identifiers
#define DR_STR(x) s_##x
extern nb::handle DR_STR(DRJIT_STRUCT);
extern nb::handle DR_STR(__dataclass_fields__);
extern nb::handle DR_STR(name);
extern nb::handle DR_STR(type);
extern nb::handle DR_STR(_traverse_write);
extern nb::handle DR_STR(_traverse_read);
extern nb::handle DR_STR(_traverse_1_cb_rw);
extern nb::handle DR_STR(_traverse_1_cb_ro);

/// Dr.Jit can lazily import and then cache the following objects to avoid
/// costly nb::module_::import() calls.
enum class LazyImport {
    DataclassesFields,   // dataclasses.fields
    TypingGetTypeHints,  // typing.get_type_hints
    TypingGetArgs,       // typing.get_args
    Count
};

/// Fetch one of the object from \ref LazyImport
extern nb::handle lazy_import(LazyImport value);

/// Release every object cached by lazy_import().
extern void lazy_import_shutdown();

/// Extract the DRJIT_STRUCT element of a custom data structure type, if available
inline nb::dict get_drjit_struct(nb::handle tp) {
    nb::object result = nb::getattr(tp, DR_STR(DRJIT_STRUCT), nb::handle());
    if (result.is_valid() && !result.type().is(&PyDict_Type))
        result = nb::object();
    return nb::borrow<nb::dict>(result);
}

/// Extract the dataclass fields element of a custom data structure type, if available
inline nb::object get_dataclass_fields(nb::handle tp) {
    nb::object result = nb::getattr(tp, DR_STR(__dataclass_fields__), nb::handle());
    if (result.is_valid()) {
        result = lazy_import(LazyImport::DataclassesFields)(tp);

        // Resolve postponed (string) annotations only if present: the expensive
        // typing.get_type_hints() call is unnecessary for concrete field types.
        bool needs_hints = false;
        for (auto field : result) {
            if (nb::isinstance<nb::str>(field.attr(DR_STR(type)))) {
                needs_hints = true;
                break;
            }
        }

        if (needs_hints) {
            nb::object hints = lazy_import(LazyImport::TypingGetTypeHints)(tp);
            for (auto field : result) {
                if (field.attr(DR_STR(type)).type().is(&PyUnicode_Type))
                    field.attr(DR_STR(type)) = hints[field.attr(DR_STR(name))];
            }
        }
    }
    return result;
}
/// Return a pointer to the underlying C++ class if the Python object inherits
/// from TraversableBase or null otherwise
inline drjit::TraversableBase *get_traversable_base(nb::handle h) {
    drjit::TraversableBase *result = nullptr;
    nb::try_cast(h, result);
    return result;
}

/// Extract a read-only callback to traverse custom data structures
inline nb::object get_traverse_cb_ro(nb::handle tp) {
    return nb::getattr(tp, DR_STR(_traverse_1_cb_ro), nb::handle());
}

/// Extract a read-write callback to traverse custom data structures
inline nb::object get_traverse_cb_rw(nb::handle tp) {
    return nb::getattr(tp, DR_STR(_traverse_1_cb_rw), nb::handle());
}

/// Does this backend keep its data in device memory the host cannot read?
///
/// A predicate rather than an enumeration at each use site, because the sites
/// that need it were written as `backend == CUDA || backend == Metal` and a
/// backend added later is then silently treated as host-resident: no error,
/// just a pointer the consumer cannot dereference. This is the same shape of
/// omission found in drjit-core's scatter-reduce capability table and in
/// op.cpp's packet-op selection, so it is worth having one place to add to.
inline bool is_device_backend(JitBackend backend) {
    return backend == JitBackend::CUDA ||
           backend == JitBackend::Metal ||
           backend == JitBackend::HIP;
}

/// DLPack device type for a backend's memory, for __dlpack__ and friends.
inline int32_t dlpack_device_type(JitBackend backend) {
    switch (backend) {
        case JitBackend::CUDA:  return nb::device::cuda::value;
        // ROCm, not CUDA -- even under the development CUDA shim, where the
        // pointer really is CUDA memory. The backend's meaning is what belongs
        // in an interchange format; the shim is a scaffold, and encoding its
        // artifact here would be wrong on every real AMD device.
        case JitBackend::HIP:   return nb::device::rocm::value;
        case JitBackend::Metal: return nb::device::metal::value;
        default:                return nb::device::cpu::value;
    }
}
