import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import wandb


class Logger(ABC):

    @abstractmethod
    def log(self, data: Dict[str, Any]):
        ...

    @abstractmethod
    def destroy(self):
        ...


class FileLogger(Logger):

    def __init__(self, log_path):
        self.log_path = log_path
        self.logger = None

    def setup(self):
        if os.path.isfile(self.log_path):
            self.logger = open(self.log_path, 'a')
        else:
            self.logger = open(self.log_path, 'w')
            # logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
            self.logger.write("\n%s\t\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'mIOU (val)', 'lr'))
        self.logger.flush()

    def log(self, data: Dict[str, Any]):
        formatted_values = []

        for key, value in data.items():
            if isinstance(value, int):
                formatted_values.append("%d" % value)
            elif isinstance(value, float):
                formatted_values.append("%.7f" % value)
            else:
                formatted_values.append(str(value))

        formatted_string = "\t\t".join(formatted_values)

        self.logger.write("\n" + formatted_string)
        self.logger.flush()

    def destroy(self):
        self.logger.close()


class WandbLogger(Logger):

    def setup(self, args, model):
        wandb.init(config=args, project='EffSegNets')
        wandb.watch(model)

    def log(self, data: Dict[str, Any]):
        wandb.log(data)

    def destroy(self):
        pass
