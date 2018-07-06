import matplotlib.pyplot as plt

def f1(n):
    return 1 / ( n ** 2 )

def f2(n):    
    return 1 / (n ** (2/3) )

#INTEGRAL F1
def f3(n):
    return -1 / n

#INTEGRAL F2
def f4(n):
    return (n ** 1/3) / (1/3)

x = []
dataf1 = [] 
dataf2 = []
dataf3 = []
dataf4 = []

for i in range(1, 100):
    x.append(i)
    dataf1.append(f1(i))
    dataf2.append(f2(i))
    dataf3.append(f3(i))    
    dataf4.append(f4(i))    

print(sum(dataf1))
plt.plot(x, dataf1)
plt.show()

print(sum(dataf2))
plt.plot(x, dataf2)
plt.show()

print(sum(dataf3))
plt.plot(x, dataf3)
plt.show()

print(sum(dataf4))
plt.plot(x, dataf4)
plt.show()