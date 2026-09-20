# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only dispatcher checks; no AITER import or extension compilation."""
import importlib.util
import sys
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

source = Path(__file__).resolve().parents[1] / "atom/model_ops/v4_kernels/cpp_topk.py"
name = "atom.model_ops.v4_kernels.cpp_topk"
module = sys.modules.get(name)
if module is None:
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def arguments():
    with FakeTensorMode():
        return (
            torch.empty(2, 8192, device="cuda"),
            torch.empty(2, dtype=torch.int32, device="cuda"),
            torch.empty(1, 128, dtype=torch.int32, device="cuda"),
            torch.empty(2, dtype=torch.int64, device="cuda"),
            torch.empty(3, dtype=torch.int32, device="cuda"),
            torch.empty(2, dtype=torch.int32, device="cuda"),
            torch.empty(3072, dtype=torch.int32, device="cuda"),
        )


def test_fake_output_and_mutation_schema():
    args = arguments()
    with args[0].fake_mode:
        out = module.cpp_topk_csa(*args, 1024, 448, 64, 512)
    assert out.shape == (2, 1024) and out.dtype == torch.int32
    assert out is not args[-1]
    schema = torch.ops.atom_v4.cpp_topk_csa.default._schema
    packed = next(a for a in schema.arguments if a.name == "packed_indices")
    assert packed.alias_info is not None and packed.alias_info.is_write


def test_dead_code_elimination_keeps_packed_write():
    def write_only(*args):
        module.cpp_topk_csa(*args, 1024, 448, 64, 512)
        return args[-1]

    graph = make_fx(write_only, tracing_mode="fake")(*arguments())
    graph.graph.eliminate_dead_code()
    assert any(n.target == torch.ops.atom_v4.cpp_topk_csa.default for n in graph.graph.nodes)
