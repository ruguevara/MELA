# coding: utf-8

def eq_f(a, b, msg=None):
    a, b = float(a), float(b)
    if abs(a - b) > 0.0000001:
        raise AssertionError(msg or "%r != %r" % (a, b))
