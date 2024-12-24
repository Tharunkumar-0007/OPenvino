from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Load the Keras model
model = load_model('D:/openvino/Openvino_skin/image_classifier_model.h5')

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
concrete_function = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Convert ConcreteFunction to frozen graph
frozen_func = convert_variables_to_constants_v2(concrete_function)
frozen_func.graph.as_graph_def()

# Save the frozen graph (.pb file)
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="D:/openvino/Openvino_skin",
                  name="frozen_model.pb",
                  as_text=False)
