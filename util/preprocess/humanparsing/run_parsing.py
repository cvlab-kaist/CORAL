from pathlib import Path
import os
import sys
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference


class Parsing:
    def __init__(self, atr_path, lip_path):
        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = os.cpu_count() // 2
        session_options.intra_op_num_threads = os.cpu_count() // 2
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(atr_path,
                                            sess_options=session_options, providers=['CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(lip_path,
                                                sess_options=session_options, providers=['CPUExecutionProvider'])

    def __call__(self, input_image):
        parsed_image, face_mask = onnx_inference(
            self.session, self.lip_session, input_image)
        return parsed_image, face_mask

