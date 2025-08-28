
class EvolutionProcessor:
    def __init__(self, memory):
        self.memory = memory
        self.field = None

    def receive_field(self, field):
        self.field = field

    def evolve(self):
        # Aplicar mutaciones, autoorganización
        return self.field

    def consult_memory(self):
        return self.memory.find_similar(self.field)
