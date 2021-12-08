import math
import itertools

from ltron.geometry.utils import metric_close_enough, immutable_vector

class GridBucket:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.clear()
    
    def clear(self):
        self.cell_to_value_positions = {}
        self.value_to_cell_positions = {}
    
    def position_to_cell(self, position):
        cell = tuple(math.floor(x / self.cell_size) for x in position)
        return cell
    
    def cells_in_radius(self, position, radius):
        min_cell = self.position_to_cell([x - radius for x in position])
        max_cell = self.position_to_cell([x + radius for x in position])
        grid_ranges = [range(min_x, max_x+1)
                for min_x, max_x in zip(min_cell, max_cell)]
        cells = itertools.product(*grid_ranges)
        return cells
        
    def insert(self, value, position):
        position = immutable_vector(position)
        cell = self.position_to_cell(position)
        
        if cell not in self.cell_to_value_positions:
            self.cell_to_value_positions[cell] = set()
        self.cell_to_value_positions[cell].add((value, position))
        
        if value not in self.value_to_cell_positions:
            self.value_to_cell_positions[value] = set()
        self.value_to_cell_positions[value].add((cell, position))
    
    def insert_many(self, values, positions):
        for value, position in zip(values, positions):
            self.insert(value, position)
    
    def remove(self, value):
        if value in self.value_to_cell_positions:
            cell_positions = self.value_to_cell_positions[value]
            for cell, position in cell_positions:
                self.cell_to_value_positions[cell].remove((value, position))
                if len(self.cell_to_value_positions[cell]) == 0:
                    del(self.cell_to_value_positions[cell])
            del(self.value_to_cell_positions[value])
    
    def lookup(self, position, radius):
        position = immutable_vector(position)
        cells = self.cells_in_radius(position, radius)
        cell_contents = set().union(*(
                self.cell_to_value_positions.get(cell, set())
                for cell in cells))
        values_in_radius = set(
                value for value, value_position in cell_contents
                if metric_close_enough(position, value_position, radius))
        return values_in_radius
    
    def lookup_many(self, positions, radius):
        return [self.lookup(position, radius) for position in positions]
