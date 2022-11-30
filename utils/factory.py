from models.finetuning import Finetuning
from models.replay import Replay

def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetuning":
        return Finetuning(args)
    elif name == "replay":
        return Replay(args)
    else:
        assert 0
