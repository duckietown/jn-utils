import sys

import numpy as np
import torch.onnx
import onnx
import onnxruntime as ort
from model_ignore.model_sol import Model

def convert(model, input, device, filename):
    # setup
    model = model.eval()
    model = model.to(device)
    input = input.to(device)

    print("First, a sanity check.")

    try:
        model(input)
        print("Sanity check passed.")
    except Exception as e:
        print(e)
        print("Sanity check did not work. Bailing out now...")
        exit(250)

    print()
    print("Second, let's do the export to onnx format.")
    torch.onnx.export(
        model,
        input,
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},  # todo this might be the wrong thing for you, but this generally works in most cases
                      'output': {0: 'batch_size'}}
    )
    print("Export went fine.")
    print()

    print("Third, let's run the onnx model_ignore checker on the new model_ignore.")
    onnx.checker.check_model(
        onnx.load(filename)
    )
    print("That went fine.")
    print()

    print("Finally, let's ensure that the onnx model_ignore is able to do inference")
    ort_session = ort.InferenceSession('model.onnx')
    outputs = ort_session.run(None, {"input": input.numpy()})
    print("The inference was alright.")
    print()
    print("Now, please verify that the output shape and type makes sense:")
    print(outputs)
    print("Keep in mind, the output shape of your original model_ignore was this one:")
    print(model(input))
    print("Also, keep in mind that it's normal if onnx replaces dicts by lists and does other normalization steps.")
    print(f""" 
    Be advised that this onnx model expects inputs of type {input.numpy().dtype} and of shape {input.numpy().shape}. 
    Anything else will cause an error.""", file=sys.stderr)
    print()
    print("Have a great day :)")

if __name__ == "__main__":
    # The only thing you have to do is change the functions "model" and "input"
    def model():
        """
        Returns your model.
        :return: your torch model
        """
        # todo change this to get your own model_ignore
        m = Model()
        m.load_state_dict(torch.load("./model_ignore/weights/model.pt"))
        m = m.model
        # end todo
        return m

    def input():
        """
        Returns a sample input for your model. Can be a static input instead of a random one.
        :return: a torch tensor of appropriate shape and type
        """
        # todo change this to get adequate sample input for your model_ignore
        i = np.random.randint(0, 255, (224, 224, 3))
        from torchvision.transforms.functional import to_tensor
        i = to_tensor(i)
        i = i.to(dtype=torch.float)
        i = i.unsqueeze(0)
        # end todo
        return i

    try:
        convert(model(), input(), torch.device("cpu"), "model.onnx")
    except:
        print("It seems like the conversion failed.")
        print("Consider changing the torch.device to cuda instead of cpu. Sometimes, the conversion only works on cpu, and vice-versa.")
        # Source for the cpu vs cuda device thing: https://github.com/pytorch/vision/issues/1706
        print("Make sure that the input function is adequate for your model.")