import importlib
import torch.nn as nn
def find_sovler_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "solver." + model_name + "Solver"

    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of torch.nn.Module,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'solver'
    for name, cls in modellib.__dict__.items():

        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_sovler_using_name(model_name)
    return model_class.modify_options
