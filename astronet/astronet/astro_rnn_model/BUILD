package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "configurations",
    srcs = ["configurations.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//astronet/astro_model:configurations",
    ],
)

py_library(
    name = "rnn_layer_builder",
    srcs = ["rnn_layer_builder.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "astro_rnn_model",
    srcs = ["astro_rnn_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//astronet/astro_model",
        "//astronet/astro_rnn_model:configurations",
        "//astronet/astro_rnn_model:rnn_layer_builder",
    ],
)

py_test(
    name = "astro_rnn_model_test",
    size = "small",
    srcs = ["astro_rnn_model_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":astro_rnn_model",
        ":configurations",
        "//astronet/ops:input_ops",
        "//astronet/ops:testing",
        "//astronet/util:configdict",
    ],
)
