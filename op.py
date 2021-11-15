import math
from ad import Node


class Op(object):
    def __call__(self):
        pass

    def fn(self, input_vals):
        raise NotImplementedError

    def grad_fn(self, input_vals, output_grad):
        raise NotImplementedError


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Node(self, [node_A, node_B])
        new_node.require_grad = node_A.require_grad or node_B.require_grad
        return new_node

    def fn(self, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def grad_fn(self, input_vals, output_grad):
        return [output_grad, output_grad]


class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Node(self, [node_A, node_B])
        new_node.require_grad = node_A.require_grad or node_B.require_grad
        return new_node

    def fn(self, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def grad_fn(self, input_vals, output_grad):
        return [output_grad, -output_grad]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Node(self, [node_A, node_B])
        new_node.require_grad = node_A.require_grad or node_B.require_grad
        return new_node

    def fn(self, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def grad_fn(self, input_vals, output_grad):
        return [input_vals[1] * output_grad, input_vals[0] * output_grad]


class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Node(self, [node_A, node_B])
        new_node.require_grad = node_A.require_grad or node_B.require_grad
        return new_node

    def fn(self, input_vals):
        assert len(input_vals) == 2
        assert input_vals[1] != 0
        return input_vals[0] / input_vals[1]

    def grad_fn(self, input_vals, output_grad):
        return [1.0/input_vals[1] * output_grad, -input_vals[0]/input_vals[1]**2 * output_grad]


class LnOp(Op):
    def __call__(self, node_A):
        new_node = Node(self, [node_A], require_grad=node_A.require_grad)
        return new_node

    def fn(self, input_vals):
        assert len(input_vals) == 1 and input_vals[0] > 0
        return math.log(input_vals[0])

    def grad_fn(self, input_vals, output_grad):
        return [1.0 / input_vals[0] * output_grad]


class SinOp(Op):
    def __call__(self, node_A):
        new_node = Node(self, [node_A], require_grad=node_A.require_grad)
        return new_node

    def fn(self, input_vals):
        assert len(input_vals) == 1
        return math.cos(input_vals[0])

    def grad_fn(self, input_vals, output_grad):
        return [math.cos(input_vals[0]) * output_grad]


class InputOp(Op):
    def __call__(self, input_val, require_grad):
        new_node = Node(self, [input_val], require_grad=require_grad)
        return new_node

    def fn(self, input_vals):
        assert len(input_vals) == 1
        return input_vals[0]

    def grad_fn(self, input_vals, output_grad):
        return [output_grad]


# more ops ...


# instaniate global ops
add = AddOp()
sub = SubOp()
mul = MulOp()
div = DivOp()
ln  = LnOp()
sin = SinOp()
var = InputOp()




