from models.finetuning import Finetuning

def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetuning":
        return Finetuning(args)
    else:
        assert 0
