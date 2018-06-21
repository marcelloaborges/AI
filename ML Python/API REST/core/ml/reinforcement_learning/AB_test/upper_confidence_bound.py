import random
import math

class UpperConfidenceBound:

    def Run(self, real_chance_1, real_chance_2, real_chance_3, events):
        
        bandit_1 = Bandit(real_chance_1)
        bandit_2 = Bandit(real_chance_2)
        bandit_3 = Bandit(real_chance_3)

        bandits = [bandit_1, bandit_2, bandit_3]

        wins = 0

        for i in range(0, events):
            upper_bandit = None
            max_upper_bound = 0

            for bandit in bandits:
                upper_bound = 0

                if bandit._pulls > 0:
                    avarage_reward = bandit._wins / bandit._pulls
                    deltaI = math.sqrt(1.5 * math.log(i + 1) / bandit._pulls)
                    upper_bound = avarage_reward + deltaI
                else:
                    upper_bound = 1e4
                
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    upper_bandit = bandit
            
            result = upper_bandit.Pull()
            if result:
                wins = wins + 1
        
        print('1=> Real:' + str(bandit_1._real_chance) + ' - Picks:' + str(bandit_1._pulls) + ' - Wins:' + str(bandit_1._wins))
        print('2=> Real:' + str(bandit_2._real_chance) + ' - Picks:' + str(bandit_2._pulls) + ' - Wins:' + str(bandit_2._wins))
        print('3=> Real:' + str(bandit_3._real_chance) + ' - Picks:' + str(bandit_3._pulls) + ' - Wins:' + str(bandit_3._wins))
        print("Wins: " + str(wins))

        result = {
                    'Item_1': 
                    {
                        "Real_Chance" : bandit_1._real_chance,
                        "Picks" : bandit_1._pulls,
                        "Wins" : bandit_1._wins
                    }, 
                    'Item_2': 
                    {
                        "Real_Chance" : bandit_2._real_chance,
                        "Picks" : bandit_2._pulls,
                        "Wins" : bandit_2._wins
                    },
                    'Item_3': 
                    {
                        "Real_Chance" : bandit_3._real_chance,
                        "Picks" : bandit_3._pulls,
                        "Wins" : bandit_3._wins
                    },
                    'Wins' : str(wins / events * 100) + '%'
                }

        return result


class Bandit:
    _real_chance = 0
    _pulls = 0
    _wins = 0

    def __init__(self, real_chance):
        self._real_chance = real_chance

    def Pull(self):
        chance = random.random()

        self._pulls = self._pulls + 1

        if chance < self._real_chance:
            self._wins = self._wins + 1
            return True
        
        return False