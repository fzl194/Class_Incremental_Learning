from models.finetuning import Finetuning
from models.replay import Replay
from models.joint import Joint
from models.icarl import iCaRL

def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetuning":
        return Finetuning(args)
    elif name == "replay":
        return Replay(args)
    elif name == "joint":
        return Joint(args)
    elif name == "icarl":
        return iCaRL(args)
    else:
        assert 0
