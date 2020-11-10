
"""
TODO: decide whether we need metaclasses after all?
      - Iterate through meta class in order to get order of instantiation
        and execution order.
      - Use CommandContainer to instantiate command classes in the right order.
      - CommandHandler uses order provided by CommandContainer.
      - merge command handler with 'Command Context' command handler instantiates the commands.
"""
prereq_attr = "prereq"

class CommandMeta(type):

    _order = {}

    def __init__(cls, name, bases, namespace):
        super(CommandMeta, cls).__init__(name, bases, namespace)
        if not hasattr(cls, prereq_attr):
            raise AttributeError(f"no attribute {prereq_attr}")

        
    def __iter__(cls):
        pass
