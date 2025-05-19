import numpy as np
import matplotlib.pyplot as plt



class Node:
    def __init__(self, id, coord=[0,0], dofs=None, restrain=[0, 0], name=None):
        self.index = id
        self.coord = coord
        self.name = name if name is not None else f"Node_{id}"  # Asigna un nombre por defecto
        if dofs is None:
            self.dofs = np.array([(id * 2)-1, id * 2 ])
            

        else:
            self.dofs = np.array(dofs)
        self.restrain = np.array(restrain)


   