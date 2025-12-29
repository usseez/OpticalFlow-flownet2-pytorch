import onnx
from onnxconverter_common import float16

m = onnx.load("./weights/251114/451_FlowNet2C.onnx")
m16 = float16.convert_float_to_float16(m, keep_io_types=True)  # I/O는 FP32 유지(호환성↑)
onnx.save(m16, "model_fp16.onnx")