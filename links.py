#!/usr/bin/python

import frames
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display


class Link(frames.Frame):
    def __init__(self, lid, d, theta, r, alpha):
        self.lid = lid
        self._dh = (d, theta, r, alpha)
        t = "transl(.0,.0,%s)*rotz(%s)*transl(%s,.0,.0)*rotx(%s)"%(
          str(d),str(theta),str(r),str(alpha))
        super(Link, self).__init__(lid, transf=t)
  
    def _post_attach(self, parent):
        if isinstance(parent, Link):
            self._parseTransf()


class MultiLink(object):
    def __init__(self):
        self.tt = frames.TransformationTree()
        self.links = []
        self.eff = sp.Matrix([0,0,0,1]) # End effector
        self.H = None
        self.J = None
        self.H_eff = None
        self._boundSym = {}

    def compose(self, *kargs):
        self.H = self.tt.compose()
        self.H_eff = self.H*self.eff
        symbols = [frames.SYMB(s) for s in kargs]
        if symbols:
          self.J = self.H_eff.jacobian(symbols)

    def addLink(self, dhpars):
        p = self.tt.root
        if self.links: p=self.links[-1]
        l = Link("l%d"%len(self.links), *dhpars)
        l.parent = p
        self.links.append(l)
        return l

    def plotLinks(self, verbose=True, use3D=True):
      # First draw all the reference systems:
      ax = self.tt.plotFrames(Q=self._boundSym, use3D=use3D)

      # Now draw all the links
      for i,link in enumerate(self.links):
        # For each link, recalculate the old and the new origin
        O_a = self.tt.Hs[i].subs(self._boundSym) * frames.Oo
        O_b = self.tt.Hs[i+1].subs(self._boundSym) * frames.Oo
        # Now connect them
        if use3D:
          ax.plot( [O_b[0], O_a[0]], [O_b[1], O_a[1]], [O_b[2], O_a[2]], 
            'o-', lw=4, mew=5, alpha=0.7)
        else:
          ax.plot( [O_b[0], O_a[0]], [O_b[2], O_a[2]], 
            'o-', lw=4, mew=5, alpha=0.7)

    def bindSymbols(self, syms):
      self._boundSym = syms
      
    def getJacobian(self):
      return self.J.subs(self._boundSym)

    def _IKoptimize(self, target_frame, starting_nodes_angles, chain, regularization_parameter=None, max_iter=None):
      """Computes the inverse kinematic on the specified target with an optimization method

      :param numpy.array target: The desired target.
      :param numpy.array starting_nodes_angles: The initial pose of your chain.
      :param float regularization_parameter: The coefficient of the regularization.
      :param int max_iter: Maximum number of iterations for the optimisation algorithm.
      """    
      import scipy.optimize

      # Only get the position
      target = target_frame[:3, 3]

      if starting_nodes_angles is None:
          raise ValueError("starting_nodes_angles must be specified")

      # Compute squared distance to target
      def optimize_target(x):
        # y = np.append(starting_nodes_angles[:chain.first_active_joint], x)
        y = chain.active_to_full(x, starting_nodes_angles)
        squared_distance = np.linalg.norm(chain.forward_kinematics(y)[:3, -1] - target)
        return squared_distance

      # If a regularization is selected
      if regularization_parameter is not None:
        def optimize_total(x):
          regularization = np.linalg.norm(x - starting_nodes_angles[chain.first_active_joint:])
          return optimize_target(x) + regularization_parameter * regularization
      else:
        def optimize_total(x):
          return optimize_target(x)

      # Compute bounds
      real_bounds = [link.bounds for link in chain.links]
      # real_bounds = real_bounds[chain.first_active_joint:]
      real_bounds = chain.active_from_full(real_bounds)

      options = {}
      # Manage iterations maximum
      if max_iter is not None:
        options["maxiter"] = max_iter

      # Use L-BFGS-B optimization
      res = scipy.optimize.minimize(optimize_total, chain.active_from_full(starting_nodes_angles), method='L-BFGS-B', bounds=real_bounds, options=options)

      print ("Inverse kinematic optimisation OK, done in {} iterations".format(res.nit))

      return(chain.active_to_full(res.x, starting_nodes_angles))    

    def positionToJoints(self, x,y,z, method="dummy"):
      """ This is the inverse kinematic """
      if method=="dummy":
        a,b,g = .0,.0,.0
      elif method  == "jinv":
        a,b,g = .0,.0,.0      
      elif method  == "optim":
        a,b,g = self._IKoptimize( (x,y,z) )      
      return a,b,g

    def jointsToPosition(self, *kargs, **kwargs):
      """ This is the direct kinematic """
      return self.H_eff.subs(self._boundSym)

    def isReachable(self, x,y,z):
      """ TODO: Tests if the point is inside the circle made by the arm"""
      return True
      
    def fromDH(self, pars):
      d = pars["d"]
      theta = pars["theta"]
      r = pars["r"]
      alpha = pars["alpha"]
      _p = zip(d,theta,r,alpha)
      for i, v in enumerate(_p):
        print self.addLink(v)
    
if __name__ == '__main__':
  from sympy import init_printing, pprint
  init_printing() 

  l1 = 1.0; a1 = 25.0
  l2 = 1.0; a2 = -10.0
  l3 = 1.0; a3 = -30.0
  arm = MultiLink()
  _dh1 = (.0,a1,l1,.0)
  _dh2 = (.0,a2,l2,.0)
  _dh3 = (.0,a3,l3,.0)    
  link1 = arm.addLink(_dh1)
  link2 = arm.addLink(_dh2)
  link3 = arm.addLink(_dh3)
  print arm.tt
  
  arm.compose()

  arm.plotLinks()
  
  plt.savefig("links.png")
  
  arm3 = MultiLink()
  _dh = {"d":[.0,.0,.0],
      "theta":[a1,a2,a3],
      "r":[l1,l2,l3],
      "alpha":[.0,.0,.0]}

  arm3.fromDH(_dh)
  print arm3.tt
  
  arm3.compose()

  arm3.plotLinks()
  
  plt.savefig("links2.png")

  arm2 = MultiLink()
  _dh1 = (.0,"s:alpha","s:L1",.0)
  _dh2 = (.0,"s:beta","s:L2",.0)
  a2l1 = arm2.addLink(_dh1)
  a2l2 = arm2.addLink(_dh2)
  print arm2.tt
  arm2.compose("alpha","beta")
  arm2.bindSymbols({frames.SYMB("alpha"):2.0,frames.SYMB("beta"):22.0,
                    frames.SYMB("L1"):1.0,frames.SYMB("L2"):1.0
                    })

  arm2.plotLinks(verbose=False)

  print arm2.jointsToPosition(alpha=45.0, 
      beta=-10, 
      L1=1.0, 
      L2=1.0)

  #print arm2.tt
  #print arm2.H_eff 
  Js = arm2.getJacobian()
  display(Js)
  display(arm2.J)
  #print arm2.J
  #print Js.evalf()






