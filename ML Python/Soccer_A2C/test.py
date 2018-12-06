from collections import namedtuple

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

e = experience('state', 0, 1, 'next_state', False)

print(e.state)
# print(e['state'])