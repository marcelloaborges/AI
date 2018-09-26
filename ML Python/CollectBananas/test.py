from collections import namedtuple

memory = []
experience = namedtuple("Experience", field_names=["state", "loss"])        

e1 = experience(1, 0.7)
e2 = experience(2, 0.8)

memory.append(e1)
memory.append(e2)

memory = sorted(memory, key=lambda x: x[1], reverse = True)
print(memory)


e3 = experience(3, 0.9)
memory.append(e3)
memory = sorted(memory, key=lambda x: x[1], reverse = True)
memory.pop()
print(memory)