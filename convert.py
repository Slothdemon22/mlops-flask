import tensorflow as tf
import tf2onnx

# Load your existing Keras model
model = tf.keras.models.load_model("best_model_v3.keras")

# Convert to ONNX
spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
output_path = "best_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save ONNX model
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print("✅ Converted to ONNX format:", output_path)
