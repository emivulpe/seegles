import mpmath as m
import sys
import numpy as np
import sympy as sp
import re

from anytree import Node, RenderTree, PreOrderIter

import matplotlib.pyplot as plt
from IPython.display import display
from cycler import cycler
import matplotlib.colors as colors
import matplotlib.cm as cmx

#### Basic defitions
Oo = sp.Matrix([0,0,0,1])
xo = sp.Matrix([1,0,0,1])
yo = sp.Matrix([0,1,0,1])
zo = sp.Matrix([0,0,1,1])

SYM_TABLE = {}

#### Basic functions 
def rad(a):
  return m.radians(a)

def identity():
  T = sp.Matrix(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
     )
  return T

def transl(vx,vy,vz):
  T = sp.Matrix(
    [[1, 0, 0, vx],
     [0, 1, 0, vy],
     [0, 0, 1, vz],
     [0, 0, 0, 1]]
     )
  return T

def rotz(q):
  Tz = sp.Matrix(
    [[sp.cos(rad(q)), -sp.sin(rad(q)), 0, 0],
     [sp.sin(rad(q)), sp.cos(rad(q)), 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
     )
  return Tz
  
def rotx(q):
  Tx = sp.Matrix(
    [[1, 0, 0, 0],
     [0, sp.cos(rad(q)), -sp.sin(rad(q)), 0],
     [0, sp.sin(rad(q)), sp.cos(rad(q)),  0],
     [0, 0, 0, 1]]
     )
  return Tx  

def roty(q):
  Ty = sp.Matrix(
    [[sp.cos(rad(q)), 0, sp.sin(rad(q)), 0],
     [0, 1, 0, 0],
     [-sp.sin(rad(q)), 0, sp.cos(rad(q)), 0],
     [0, 0, 0, 1]]
     )
  return Ty  
  
#### Frame manipulations

def toList(s):
  if "*" in s:
    return s.split("*")
  else:
    return [s]

def SYMB(s):
  if type(s) is sp.Symbol: return s
  if s not in SYM_TABLE:
    sym = sp.Symbol(s, real=True)
    SYM_TABLE[s] = sym
  return SYM_TABLE[s]

def _parseArgs(s):
  """ Transforms s:arg in SYMB('arg') """
  return  re.sub(r's:(\w+)', r"SYMB('\1')", s) 

class Frame(Node):
  def __init__(self, name, transf="identity()", **kwargs):
    super(Frame, self).__init__(str(name), **kwargs)
    self.transf = transf
    self._parseTransf()

  def _parseTransf(self):
    ts = toList(self.transf)
    H = identity()
    for t in ts:
      t = eval(_parseArgs(t))
      H *= t
    self.T = H
            
  def __repr__(self):
    s = self.name
    if self.transf: s += "("+self.transf+")"
    return s
  
#### Transformation tree

def round_expr(expr, num_digits=2):
  return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})

class TransformationTree(object):
  def __init__(self):
    self.Hs = []
    self.root = Frame("O")
      
  def __str__(self):
    return str(RenderTree(self.root))
  
  def plotInFrame(self, v, ax, frame_id, artist='o-'):
    O_a = sp.Matrix([0,0,0,1])
    O_b = self.Hs[frame_id]*O_a
    v = sp.Matrix(v)
    Nv = self.Hs[frame_id]*v  
    ax.plot( (O_b[0], Nv[0]), (O_b[1], Nv[1]), artist, lw=4, mew=5, alpha=0.7)
    return O_b, Nv

  def compose(self):
    H = identity()
    for node in PreOrderIter(self.root):
      H *= node.T
      self.Hs.append(H)

    return H
  
  def plotFrames(self, Q={}, xlim = 2.0, ylim = 2.0, zlim=2.0, verbose=False, use3D=True):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if use3D:
      ax = fig.add_subplot(111, projection='3d')

    # Set axes limits and labels
    if use3D:
      ax.set_xlim3d([-xlim, xlim]); ax.set_xlabel('X')
      ax.set_ylim3d([-ylim, ylim]); ax.set_ylabel('Y')
      ax.set_zlim3d([-zlim, zlim]); ax.set_zlabel('Z')
    else:
      ax.set_xlim([-xlim, xlim]); ax.set_xlabel('X')
      ax.set_ylim([-zlim, zlim]); ax.set_ylabel('Z')
      
    # For every frame stored
    for i, H in enumerate(self.Hs):  
      # Substitute bound parameters      
      H = H.copy().subs(Q)
      # Calculate new origin and new versors
      O = H * Oo; x = H * xo; y = H * yo; z = H * zo
      # Plot new axes
      if use3D:
        ax.plot([O[0], x[0]],[O[1], x[1]],[O[2], x[2]], color="r")
        ax.plot([O[0], y[0]],[O[1], y[1]],[O[2], y[2]], color="k")
        ax.plot([O[0], z[0]],[O[1], z[1]],[O[2], z[2]], color="b")
      else:
        ax.plot([O[0], x[0]],[O[2], x[2]], color="r")
        ax.plot([O[0], y[0]],[O[2], y[2]], color="k")
        ax.plot([O[0], z[0]],[O[2], z[2]], color="b")  

    return ax

if __name__ == "__main__":
  from sympy import init_printing, pprint
  init_printing(use_latex=True) 
    
  tt = TransformationTree()
  A = Frame("A", transf="rotz(20)*transl(1,0,0)", parent=tt.root)
  B = Frame("B", transf="rotz(80)", parent=A)

  print tt

  tt.plotFrames()
  
  plt.savefig("frames.png")

  tt = TransformationTree()
  A = Frame("A", transf="rotz(s:alpha)*transl(s:L1,0,0)", parent=tt.root)
  B = Frame("B", transf="rotz(s:beta)", parent=A)
  print tt
  tt_eff = tt.compose()*sp.Matrix([SYMB("L2"),0,0,1])
  print tt_eff
  ab = [SYMB("alpha"),SYMB("beta")]
  print tt_eff.jacobian(ab)
  
