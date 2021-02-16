class Sparse:
    def __init__(self, x, y):
        if x <= 0 or y <= 0: raise ValueError('Size must be > 0')
        self._x = x
        self._y = y
        self._data = {}
    def check_coordinates(self, x, y):
        if x >= self._x or y >= self._y: raise KeyError('Incorrect Coordinates')
    def __getitem__(self, key):
        self.check_coordinates(*key)
        return self._data[key] if key in self._data else 0
    def __setitem__(self, key, value):
        self.check_coordinates(*key)
        self._data[key] = value
    def get_max_for_x(self, x):
        y = [(k[1], v) for k, v in self._data.items() if k[0] == x]
        if not len(y): return (0, 0)
        y.sort(key=lambda x: x[0])
        for i, v in enumerate(y):
            if i != v[0]:
                y.append((i, 0))
                break
        return max(y, key=lambda x: x[1])

s = Sparse(50, 50)
s[1,5]=-10
print(s.get_max_for_x(1))
print(s[0, 0])
print(s[1, 5])