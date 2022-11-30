from models.finetuning import Finetuning
from models.replay import Replay
from models.joint import Joint

def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetuning":
        return Finetuning(args)
    elif name == "replay":
        return Replay(args)
    elif name == "joint":
        return Joint(args)
    else:
        assert 0
