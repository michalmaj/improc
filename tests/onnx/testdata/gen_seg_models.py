"""
Generates two tiny ONNX models for OnnxSegmentor / OnnxInstanceSegmentor tests.
Run from the testdata/ directory: python gen_seg_models.py
Requires: pip install onnx numpy
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

rng = np.random.default_rng(42)

# ── tiny_segmentor_semantic.onnx ─────────────────────────────────────────────
# Input: images [1, 3, 8, 8]
# Conv 3→4 channels, kernel 1×1, no padding → output [1, 4, 8, 8]
# Used in OnnxSegmentor tests at input_size(8, 8); output has 4 classes.

W_conv = rng.standard_normal((4, 3, 1, 1)).astype(np.float32)
W_init = numpy_helper.from_array(W_conv, name="W_conv")

conv_node = helper.make_node("Conv", ["images", "W_conv"], ["output"],
                             kernel_shape=[1, 1])

seg_graph = helper.make_graph(
    [conv_node],
    "tiny_segmentor_semantic",
    [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
    initializer=[W_init],
)
seg_model = helper.make_model(seg_graph,
                              opset_imports=[helper.make_opsetid("", 17)])
seg_model.ir_version = 8
onnx.checker.check_model(seg_model)
onnx.save(seg_model, "tiny_segmentor_semantic.onnx")
print("Saved tiny_segmentor_semantic.onnx  (input [1,3,8,8] → output [1,4,8,8])")

# ── tiny_instance_segmentor.onnx ─────────────────────────────────────────────
# Input:   images  [1, 3, 8, 8]
# Output0: output0 [1, 38, 10]  — 4 (box) + 2 (classes) + 32 (mask coeffs), 10 anchors
# Output1: output1 [1, 32, 4, 4] — prototype masks
#
# Graph: Flatten → MatMul0 → Reshape0 (output0)
#                → MatMul1 → Reshape1 (output1)

W0 = rng.standard_normal((192, 380)).astype(np.float32)   # 3*8*8=192, 38*10=380
W1 = rng.standard_normal((192, 512)).astype(np.float32)   # 192 → 32*4*4=512
W0_init     = numpy_helper.from_array(W0, name="W0")
W1_init     = numpy_helper.from_array(W1, name="W1")
shape0_init = numpy_helper.from_array(np.array([1, 38, 10],   dtype=np.int64), name="shape0")
shape1_init = numpy_helper.from_array(np.array([1, 32, 4, 4], dtype=np.int64), name="shape1")

inst_nodes = [
    helper.make_node("Flatten", ["images"], ["flat"], axis=1),
    helper.make_node("MatMul",  ["flat", "W0"], ["out0_flat"]),
    helper.make_node("Reshape", ["out0_flat", "shape0"], ["output0"]),
    helper.make_node("MatMul",  ["flat", "W1"], ["out1_flat"]),
    helper.make_node("Reshape", ["out1_flat", "shape1"], ["output1"]),
]

inst_graph = helper.make_graph(
    inst_nodes,
    "tiny_instance_segmentor",
    [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
    [
        helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 38, 10]),
        helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, 32, 4, 4]),
    ],
    initializer=[W0_init, W1_init, shape0_init, shape1_init],
)
inst_model = helper.make_model(inst_graph,
                               opset_imports=[helper.make_opsetid("", 17)])
inst_model.ir_version = 8
onnx.checker.check_model(inst_model)
onnx.save(inst_model, "tiny_instance_segmentor.onnx")
print("Saved tiny_instance_segmentor.onnx  (input [1,3,8,8] → output0 [1,38,10], output1 [1,32,4,4])")
