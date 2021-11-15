from op import *
from ad import Executor


def test_ad():
    x1 = var(2.0, require_grad=True)
    x2 = var(5.0, require_grad=True)

    y = sub(add(ln(x1), mul(x1, x2)), sin(x2))

    exe = Executor(y)
    output_val = exe.forward(debug=True)

    exe.backward(gradient=1.0, debug=True)
    print('x1.grad =', x1.grad)
    print('x2.grad =', x2.grad)


if __name__ == '__main__':
    test_ad()



