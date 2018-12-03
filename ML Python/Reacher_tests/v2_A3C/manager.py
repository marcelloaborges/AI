import threading, logging

from v2_A3C.environment import Environment
from v2_A3C.agent import Agent

class Manager(threading.Thread):

    def __init__(self, environment, agent_id, brain, gamma, gamma_n, n_step_return, args=(), kwargs=None):
        threading.Thread.__init__(self)

        self.work_queue = args

        self.env = Environment(environment, show_info=False)
        self.agent = Agent(agent_id, brain, gamma, gamma_n, n_step_return)

    def run(self):
        while not self.work_queue.stop():
            logging.debug('Agent {} running episode'.format(self.agent.id))
            self.run_episode()
            self.work_queue.increase()

    def run_episode(self):
        state = self.env.reset()

        while True:
            action, _ = self.agent.act(state[self.agent.id])
            
            next_state, reward, done = self.env.step(self.agent.id, action)

            self.agent.step(state, action, reward, next_state, done)

            state = next_state

            if done:
                break
            