

def generate():
    a = 13
    for i in range(a):
        yield i, 1

def batch(itera):
    batch_size = 20
    for i in range(batch_size):
        a = next(itera, '1234')
        yield a

if __name__ == '__main__':
    b = 1
    ge = generate()
    ba = batch(ge)
    for i in range(30):
        result = next(ba)
        if result == '1234':
            ge = None
            ge = generate()
            ba = None
            ba = batch(ge)
            continue
        print(result)
        b += 1
    print(b)
