import threading, logging

class Optimizer(threading.Thread):

    def __init__(self, brain, args=(), kwargs=None):
        threading.Thread.__init__(self)
        
        self.work_queue = args

        self.brain = brain
        
    def run(self):
        while not self.work_queue.stop():
            logging.debug('Optimizer called')
            self.brain.learn()
