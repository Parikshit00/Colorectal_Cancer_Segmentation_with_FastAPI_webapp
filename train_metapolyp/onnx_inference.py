import torch
import onnx
import onnxruntime

# Load the ONNX model
onnx_path = "caformer_s18.onnx"
onnx_model = onnx.load(onnx_path)

'''
# Print model details
print("Model inputs:")
for input in onnx_model.graph.input:
    print(f"Name: {input.name}, Shape: {input.type.tensor_type.shape.dim}, Type: {input.type.tensor_type.elem_type}")

print("\nModel outputs:")
for output in onnx_model.graph.output:
    print(f"Name: {output.name}, Shape: {output.type.tensor_type.shape.dim}, Type: {output.type.tensor_type.elem_type}")
'''
# Initialize ONNX Runtime session
ort_session = onnxruntime.InferenceSession(onnx_path)


# Run inference with ONNX Runtime
dummy_input = torch.randn(1, 3, 256, 256).cpu().numpy()
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
ort_outputs = ort_session.run(None, ort_inputs)

for o in ort_outputs:
    print(o.shape)
dummy_input = torch.randn(1, 3, 256, 256)
torch_model = torch.onnx.load(onnx_path)

# Run inference with the PyTorch model
output_tensor = torch_model(dummy_input)

print(output_tensor.shape)