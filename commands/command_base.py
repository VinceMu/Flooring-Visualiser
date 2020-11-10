from abc import abstractmethod
import re

FORBIDDEN_CHARACTERS = ['-']

PREFIX_CHAR_REGEX = fr'^[{"".join(FORBIDDEN_CHARACTERS)}]{{0,2}}([^\d{"".join(FORBIDDEN_CHARACTERS)}]+)$'
"""
# TODO: add subcommand as a param.
A subcommand is a command contained in the current command. This command is executed
before the command is executed. Current command inherits prerequisites the subcommand has.
"""


class CommandBase():
    def __init__(self, name: str, prereq: 'CommandBase', **kwargs):
        self.raw_name = name
        self.prereq = prereq
        self.name = re.search(PREFIX_CHAR_REGEX, name).group(1)
        if self.name == "":
            raise InvalidCommandName("invalid characters in name!")
        self.argument_options = kwargs

    def run(self, args, outputs):
        cmd_val = args[self.name]
        # if a flag command is not present skip the command's execution
        if self.argument_options["action"] == "store_true" and cmd_val is False:
            return None
        previous_output = None
        if self.prereq is not None:
            previous_output = outputs[self.prereq.name]
        return self._run(args[self.name], previous_output)

    @abstractmethod
    def _run(self, arg, previous_output):
        pass

    def getArgumentOptions(self):
        return self.argument_options

    def getPrerequisite(self):
        return self.prereq

    def __eq__(self, other):
        # TODO: contentious, revise if necessary!
        return self.name == other.name

    def __lt__(self, other):
        return self._recursive_isin(self)

    def _recursive_isin(self, parent):
        if (parent.prereq is None):
            return False
        elif (parent == parent.prereq):
            return True
        else:
            return self._recursive_isin(parent.prereq)


class InvalidCommandName(Exception):
    pass


class PrerequisiteNotSatisfied(Exception):
    pass
