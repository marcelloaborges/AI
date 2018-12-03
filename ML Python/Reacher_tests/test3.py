import logging
import random
import threading
import time

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-10s) %(message)s',)


class Producer(threading.Thread):
    def __init__(self, args=(), kwargs=None):
        threading.Thread.__init__(self)
        
        self.production = args                
        
    def run(self):
        while not self.production.end():
            time.sleep(2)  

            self.production.add(random.randint(1, 10))
            # logging.debug('Producting: %d', self.production.state())
                    
        logging.debug('Done')

class Consumer(threading.Thread):
    def __init__(self, args=(), kwargs=None):
        threading.Thread.__init__(self)
            
        self.production = args

    def run(self):
        while not self.production.end():
            time.sleep(1)  

            self.production.consume(random.randint(1, 10))
            # logging.debug('Producting: %d', self.production.state())
                    
        logging.debug('Done')


class Production:
    def __init__(self, init=0, total_work=60):
        self.resources = init
        self.total_work = total_work

    def add(self, value):
        logging.debug('Adding %d to resources', value)
        self.resources += value
        self.total_work -= 1
        self.state()

    def consume(self, value):
        logging.debug('Consuming %d from resources', value)
        self.resources -= value
        self.total_work -= 1
        self.state()

    def state(self):
        logging.debug('State %d', self.resources)

    def end(self):
        return self.total_work <= 0

PRODUCTION = 0
STOP_SIGNAL = False
GOAL = 60
N_PROD = 2
N_CONS = 1

production = Production(init=0, total_work=GOAL)

producers = [ Producer(args=production) for i in range(N_PROD) ]
consumers = [ Consumer(args=production) for i in range(N_CONS) ]

for p in producers:
    p.start()

for c in consumers:
    c.start()

logging.debug('BEGINNING THE WORK')