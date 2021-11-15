class Node(object):
    uid = 0

    def __init__(self, op, inputs, require_grad=False):
        self.op = op
        self.inputs = inputs    # Node or scalar
        self.require_grad = require_grad
        self.grad = None
        self.evaluate()
        self.id = Node.uid
        Node.uid += 1

    def input2values(self):
        input_vals = []
        for inp in self.inputs:
            if isinstance(inp, Node):
                inp = inp.value    
            input_vals.append(inp)
        return input_vals

    def evaluate(self):
        self.value = self.op.fn(self.input2values())

    def __repr__(self):
        return "<Node%2d: op=%s, inputs=%s, require_grad=%s>" % (
            self.id, repr(self.op), repr(self.input2values()), self.require_grad)


class Executor(object):
    def __init__(self, output_node):
        self.root = output_node
        self.topo_order = self._topo_sort(self.root)

    def forward(self, debug=False):
        """
        forward propagation to compute output
        """
        for node in self.topo_order:
            node.evaluate()
            if debug:
                print("forward prop: %s" % node)
        return self.root.value

    def backward(self, gradient=1.0, debug=False):
        """
        backward propagation to compute gradient
        """
        reverse_topo_order = list(reversed(self.topo_order))
        reverse_topo_order[0].grad = gradient
        for node in reverse_topo_order:
            grads = node.op.grad_fn(node.input2values(), node.grad)
            for inode, grad in zip(node.inputs, grads):
                if isinstance(inode, Node):
                    if inode.grad is None:
                        inode.grad = 0.0
                    inode.grad += grad
        if debug:
            for node in reverse_topo_order:
                print("after backard:", node)

    def _topo_sort(self, root):
        """topological sorting"""
        topo_order = []
        visited = set()
        self._dfs(root, visited, topo_order)
        return topo_order

    def _dfs(self, node, visited, topo_order):
        """postorder dfs"""
        if node is None or not isinstance(node, Node):
            return
        if node in visited:
            return
        visited.add(node)
        for inode in node.inputs:
            self._dfs(inode, visited, topo_order)
        topo_order.append(node)


