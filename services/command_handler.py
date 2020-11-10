from argparse import ArgumentParser
import heapq
from commands.command_base import CommandBase


class CommandHandler(ArgumentParser):
    def __init__(self):
        self.command_list = []
        super(CommandHandler, self).__init__()

    def addCommand(self, cmd: CommandBase):
        self.add_argument(cmd.raw_name, **cmd.getArgumentOptions())
        heapq.heappush(self.command_list, cmd)

    def run_commands(self):
        args = vars(self.parse_args())
        output = {}
        for cmd in self.command_list:
            if cmd.name not in args:
                continue
            cmd_output = cmd.run(args, output)
            output[cmd.name] = cmd_output
        return output
