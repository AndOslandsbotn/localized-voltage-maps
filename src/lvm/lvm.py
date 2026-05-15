from .cells import build_voronoi_cells
from .graph import build_graph, apply_ground_resistance
from .voltage import solve_voltage_maps
from .landmarks import select_landmark_indices

class LocalizedVoltageMaps:
    def __init__(self, n_centroids: int):
        pass

    def fit(self, stream):
        centroids =self.build_voroni_cells(stream)
        graph = self.build_graph(centroids)
        grounded_graph = self.apply_ground_resistance(graph)
        voltage_maps = self.solve_voltage_maps(grounded_graph)
        landmark_indices = self.select_landmark_indices(voltage_maps)
        return landmark_indices
      