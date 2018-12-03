import logging
import threading

class WorkQueue:

    def __init__(self, amount_work=1000):        
        self.amount_work = amount_work
        self.work_done = 0

    def increase(self):
        self.work_done += 1
        logging.debug('Word done: {}'.format(self.work_done))

    def stop(self):
        return self.work_done >= self.amount_work