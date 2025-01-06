
class MathExample:
    def __init__(self, problem, solution, id):
        self.problem = problem
        self.solution = solution
        self.id = id
        self.generated_solution = ""

    def is_concluded(self):
        raise NotImplementedError

    def is_in_progress(self):
        raise NotImplementedError

    def update_solution(self, solution_step):
        raise NotImplementedError

class Environment:
    def reset(self):
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

