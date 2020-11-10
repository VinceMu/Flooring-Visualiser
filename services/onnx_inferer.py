import onnxruntime


class OnnxInferer:
    def __init__(self, inference_session: onnxruntime.InferenceSession,
                 num_inputs: int):
        self.onnx_runtime = inference_session
        outputs = self.onnx_runtime.get_outputs()
        self.output_names = list(map(lambda output: output.name, outputs))
        inputs = self.onnx_runtime.get_inputs()
        self.input_names = []
        for idx in range(num_inputs):
            self.input_names.append(inputs[idx].name)

    def infer_with_onnx(self, inputs: list):
        """
        params: inputs correlates to order of input_names.
        """
        input_dict = {}
        for idx, input_name in enumerate(self.input_names):
            input_dict[input_name] = inputs[idx]
        return self.onnx_runtime.run(self.output_names, input_dict)
