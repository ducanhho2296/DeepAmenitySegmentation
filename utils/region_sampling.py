import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from shapely.geometry import Polygon

class Gridgenerating():
    def __init__(self, city_df,
                 spacing_x=0.0015, spacing_y=0.0015) -> None:
        self.city_df = city_df
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y

    def grid_points_in_bbox(self, bbox):
        min_x, min_y, max_x, max_y = bbox
        x_coords = np.arange(min_x, max_x + self.spacing_x, self.spacing_x)
        y_coords = np.arange(min_y, max_y + self.spacing_y, self.spacing_y)
        
        self.grid_points = []
        for x in x_coords:
            for y in y_coords:
                self.grid_points.append(Point(x, y))
                
        return self.grid_points
    #                                        crs=self.city_df.crs)  
    #     joined_gdf = gpd.sjoin(grid_points_gdf, self.city_df, how="left", op="intersects")
    #     # Keep only grid points that intersect with a building
    #     points_with_buildings = joined_gdf.dropna(subset=["index_right"])

    #     # Reset the index
    #     points_with_buildings.reset_index(drop=True, inplace=True)

    #     # Remove the 'index_right' column
    #     points_with_buildings.drop(columns=["index_right"], inplace=True)
    #     points_list = [(point.x, point.y) for point in points_with_buildings.geometry]
    #     #longitudes, latitudes = zip(*points_list) #just for ploting
    #     return points_list
    
    def create_bbox_poly(self,square_bbox):
        min_x, min_y, max_x, max_y = square_bbox
        bbox_poly = Polygon([
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y)
        ])
        return bbox_poly
    
    def capture_square_bbox(self, point_position, grid_points=None):
        if grid_points is None:
            grid_points = self.grid_points

        if point_position == len(grid_points):
            point_position -= 1

        #the last point1 must be <= grid_points[len(grid_points)-1]
        if point_position < len(self.grid_points):
            point1 = self.grid_points[point_position]
            point2 = Point(point1.x, point1.y + self.spacing_y)
            point3 = Point(point1.x - self.spacing_x, point1.y)
            point4 = Point(point2.x - self.spacing_x, point2.y)

            min_x = min(point1.x, point3.x)
            min_y = min(point1.y, point2.y)
            max_x = max(point2.x, point4.x)
            max_y = max(point3.y, point4.y)

            square_bbox = (min_x, min_y, max_x, max_y)
            self.bbox_df = gpd.GeoDataFrame(geometry=[self.create_bbox_poly(square_bbox)]) #=> can be used to visualize location of bbox inside city
            
            self.bbox_df.crs = self.city_df.crs
            #create building dataframe storing buildings inside bbox
            buildings_within = gpd.sjoin(self.city_df, self.bbox_df, predicate="intersects") 
            return buildings_within, self.bbox_df


if __name__  == "__main__":
    Grid = Gridgenerating(city_df=None)
    b, bbox = Grid.capture_square_bbox(1)