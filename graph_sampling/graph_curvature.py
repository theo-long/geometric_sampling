from GraphRicciCurvature.FormanRicci import FormanRicci as FR
from GraphRicciCurvature.OllivierRicci import OllivierRicci as OR
from abc import ABC, abstractmethod

from logging import getLogger
logger = getLogger()

class GraphCurvature(ABC):
    @abstractmethod
    def compute_ricci_curvature(self):
        ...

    @abstractmethod
    def compute_ricci_curvature_edges(self, edges=None, **kwargs):
        ...

class FormanRicci(FR, GraphCurvature):

    def compute_ricci_curvature_edges(self, edges=None, recompute=True):
        """Compute Ricci Curvature for a set of edges."""
        # Edge Forman curvature
        results = {}
        for (v1, v2) in edges:
            if self.G[v1][v2].get("formanCurvature", False) and not recompute:
                continue
            if self.G.is_directed():
                v1_nbr = set(list(self.G.predecessors(v1)) + list(self.G.successors(v1)))
                v2_nbr = set(list(self.G.predecessors(v2)) + list(self.G.successors(v2)))
            else:
                v1_nbr = set(self.G.neighbors(v1))
                v1_nbr.remove(v2)
                v2_nbr = set(self.G.neighbors(v2))
                v2_nbr.remove(v1)
            face = v1_nbr & v2_nbr
            # G[v1][v2]["face"]=face
            prl_nbr = (v1_nbr | v2_nbr) - face
            # G[v1][v2]["prl_nbr"]=prl_nbr

            curvature = len(face) + 2 - len(prl_nbr)
            self.G[v1][v2]["formanCurvature"] = curvature
            results[(v1, v2)] = (curvature)

            logger.debug("Source: %s, target: %d, Forman-Ricci curvature = %f  " % (
                v1, v2, self.G[v1][v2]["formanCurvature"])) 
        
        return results

    


class BalancedFormanRicci(GraphCurvature):
    pass

# All methods already inherited from OR class
class OllivierRicci(OR, GraphCurvature):
    pass
