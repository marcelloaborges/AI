class Item:
    def __init__(self, a, b, q):
        self.a = a
        self.b = b
        self.q = q

i1 = Item(0, 1, 10)
i2 = Item(0, 1, 5)
i3 = Item(1, 1, 2)
i4 = Item(1, 2, 3)
i5 = Item(2, 2, 7)
i6 = Item(2, 2, 6)

items = [ i2, i1, i4, i6, i3, i5 ]
print(items)

items.sort( key = lambda x : [ x.a, x.b, x.q * -1 ] ) 
print(items)