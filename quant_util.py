import torch
from torch import nn
import copy

from torch.export import export, ExportedProgram
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    QConfigMapping,
)

import torch.ao.quantization.quantize_fx as quantize_fx


def quantize_ptq_export(model: nn.Module, data_loader, per_channel=False, use_reference_representation=False) -> None:

    model.eval()
    _dummy_input_data = (next(iter(data_loader))[0],)
    model = capture_pre_autograd_graph(model, _dummy_input_data)

    quantizer = XNNPACKQuantizer()
    quantization_config = get_symmetric_quantization_config(is_per_channel=per_channel, is_qat=False)
    quantizer.set_global(quantization_config)
    # prepare_pt2e folds BatchNorm operators into preceding Conv2d operators, and inserts observers in appropriate places in the model.
    prepared_model = prepare_pt2e(model, quantizer)

    # model(*_dummy_input_data)
    # get 128 input data for calibration

    # TODO: Run 1 image just for now
    for i, (image, _) in enumerate(data_loader):
        if i >= 1:
            break

        prepared_model(image)

    quantized_model = convert_pt2e(prepared_model, use_reference_representation=use_reference_representation)

    return quantized_model


def quantize_ptq_fx(model: nn.Module, data_loader):

    model.eval()
    example_inputs = (next(iter(data_loader))[0], )
    # example_inputs = (torch.ra)

    qconfig_mapping = get_default_qconfig_mapping("qnnpack")

    # prepare
    model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)

    # TODO: Run 1 image just for now
    for i, (images, _) in enumerate(data_loader):
        if i >= 1:
            break
        
        model_prepared(images)

    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)

    return model_quantized


def quantize_ptq(model: nn.Module, data_loader):
    
    model.eval()

    model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    model_prepared = torch.ao.quantization.prepare(model)

    for i, (images, _) in enumerate(data_loader):
        if i >= 1:
            break

        model_prepared(images)

    model_quantized = torch.ao.quantization.convert(model_prepared)

    return model_quantized


def quantize_proportion(model):
    fp32_param = 0.0
    total = 0.0
    q8_param = 0.0

    # print(model.state_dict())

    for name, param in model.state_dict().items():

        if isinstance(param, torch.Tensor):
            total += torch.numel(param)

            if(param.dtype == torch.float32):
                fp32_param += torch.numel(param)

            if(param.dtype == torch.qint8):
                q8_param += torch.numel(param)

        elif isinstance(param, torch.dtype):
            # print(param)
            pass

        elif isinstance(param, tuple):
            for sub_param in param:
                if isinstance(sub_param, torch.Tensor):
                    total += torch.numel(sub_param)

                    if sub_param.dtype == torch.float32:
                        fp32_param += torch.numel(sub_param)

                    elif sub_param.dtype == torch.qint8:
                        q8_param += torch.numel(sub_param)

    # print(fp32_param, total, q8_param, q8_param / total)

    return q8_param / total