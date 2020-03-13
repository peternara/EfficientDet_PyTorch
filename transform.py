from models.efficientdet import EfficientDetBiFPN
import torch

import onnxruntime

def split_model_dict(model):
    backbone_dict = {}
    neck_dict = {}
    bbox_head_dict = {}

    num_backbone, num_neck, num_bbox_head = 0, 0, 0
    num_model = 0
    for k, v in model.items():
        if("backbone" in k):
            backbone_dict[k] = v
            num_backbone = num_backbone + 1
        elif('neck' in k):
            neck_dict[k] = v
            num_neck = num_neck + 1
        elif('bbox_head' in k):
            bbox_head_dict[k] = v
            num_bbox_head = num_bbox_head + 1
        num_model = num_model + 1
    assert num_model == num_backbone + num_neck + num_bbox_head
    return backbone_dict, neck_dict, bbox_head_dict

def efficientdet_torch_to_onnx(input, onnx_path):
    det_bifpn = EfficientDetBiFPN(D_bifpn=2, W_bifpn=64)
    det_bifpn.backbone.set_swish(memory_efficient=False)

    efficient_data = torch.load('checkpoint_48.pth', map_location='cpu')
    model = efficient_data['state_dict']

    backbone_dict, neck_dict, bbox_head_dict = split_model_dict(model)

    bifpn_model = dict(backbone_dict, **neck_dict)
    det_bifpn.load_state_dict(bifpn_model)
    torch.onnx.export(det_bifpn, input, onnx_path, opset_version = 11, verbose=False)


def tensor_to_numpy(input_tensor):
    return input_tensor.cpu().numpy()

if __name__ == '__main__':
    dummy_input = torch.randn(4, 3, 512, 512)
    onnx_path = f"efficientdet-d0.onnx"
    efficientdet_torch_to_onnx(input= dummy_input, onnx_path = onnx_path)

    ort_session = onnxruntime.InferenceSession(path_or_bytes = onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: tensor_to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)