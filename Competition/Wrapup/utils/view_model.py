import torch
from torch.autograd import Variable
from graphviz import Digraph
import efficientnet_pytorch

__all__ = ['view_model']

def make_dot(var, params=None):
  if params is not None:
    param_map = {id(v): k for k, v in params.item()}
    
  node_attr = dict(style='filled',
            shape='box',
            align='left',
            fontsize='12',
            rankep='0.1',
            height='0.2'
  )
  dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
  seen=set()
  def size_to_str(size):
      return '('+(', ').join(['%d' % v for v in size])+')'

  def add_nodes(var):
      if var not in seen:
          if torch.is_tensor(var):
              dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
          elif hasattr(var, 'variable'):
              u = var.variable
              name = param_map[id(u)] if params is not None else ''
              node_name = '%s\n %s' % (name, size_to_str(u.size()))
              dot.node(str(id(var)), node_name, fillcolor='lightblue')
          else:
              dot.node(str(id(var)), str(type(var).__name__))
          seen.add(var)
          if hasattr(var, 'next_functions'):
              for u in var.next_functions:
                  if u[0] is not None:
                      dot.edge(str(id(u[0])), str(id(var)))
                      add_nodes(u[0])
          if hasattr(var, 'saved_tensors'):
              for t in var.saved_tensors:
                  dot.edge(str(id(t)), str(id(var)))
                  add_nodes(t)
  add_nodes(var.grad_fn)
  return dot


def view_model(net, input_shape):
    x = Variable(torch.randn(1, *input_shape))
    y = net(x)
    g = make_dot(y)
    g.view()

    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        print("layer parameters size:" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("layer parameters:" + str(l))
        k = k + l
    print("total parameters:" + str(k))


if __name__ == '__main__':
    net = CNN()
    view_model(net)
    # x = Variable(torch.randn(1, 1, 28, 28))
    # y = net(x)
    # g = make_dot(y)
    # g.view()
    #
    # params = list(net.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("layer parameters:" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("layer parameters:" + str(l))
    #     k = k + l
    # print("total parameters:" + str(k))
