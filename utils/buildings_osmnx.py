import osmnx as ox
import numpy as np
import time


class CityData:
    def __init__(self, city:str):
        """
        Initialize CityData object.

        city: Name of the city for creating the GeoDataFrame.
        """
        self.city = city
        self.buildings = None
        self.city_bbox = None

    def get_data(self):
        food_tags = ["restaurant",
                "fast_food",
                "food_court",
                "ice_cream",
                "bakery",
                "cafe",
                "bar",
                "pub",
                "biergarten"]

        school_tags = ["kindergarten",
                "driving_school",
                "language_school",
                "college",
                "music_school",
                "university",
                "library",
                "toy_library",
                "school"]
        
        healthcare_tags = ["clinic",
                    "dentist",
                    "doctors",
                    "pharmacy",
                    "baby_hatch",
                    "hospital",
                    "nursing_home",
                    "social_facility",
                    "veterinary"]

        entertainment_tags = ["arts_centre",
                        "brothel",
                        "casino",
                        "cinema",
                        "community_centre",
                        "conference_centre",
                        "events_venue",
                        "fountain",
                        "gambling",
                        "love_hotel",
                        "nightclub",
                        "planetarium",
                        "public_bookcase",
                        "social_centre",
                        "stripclub",
                        "studio",
                        "swingerclub",
                        "theatre"]

        public_tags = ["courthouse",
                "police",
                "post_office",
                "fire_station",
                "post_depot",
                "prison",
                "ranger_station",
                "townhall",
                "post_box"]

        roads = ["motorwoy",
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "unclassified",
                "residential",
                "motorway_link",
                "trunk_link",
                "primary_link",
                "secondary_link",
                "tertiary_link",
                "living_street",
                "service",
                "pedestrian",
                "track",
                "bus_guideway",
                "escape",
                "raceway",
                "road",
                "busway",
                "footway",
                "bridleway",
                "steps",
                "corridor",
                "path",
                "cycleway"]


        buildings = ox.geometries_from_place(query=self.city, tags={'building':True})

        food = ox.geometries_from_place(query=self.city, tags={"amenity": food_tags})
        school = ox.geometries_from_place(query=self.city, tags={"amenity": school_tags})
        healthcare = ox.geometries_from_place(query=self.city, tags={"amenity": healthcare_tags})
        entertainment = ox.geometries_from_place(query=self.city, tags={"amenity": entertainment_tags})
        public = ox.geometries_from_place(query=self.city, tags={"amenity": public_tags})

        retail = ox.geometries_from_place(query=self.city, tags={"shop": True})
        sport = ox.geometries_from_place(query=self.city, tags={"sport": True})
        leisure = ox.geometries_from_place(query=self.city, tags={'leisure':True})
        highway = ox.geometries_from_place(query=self.city, tags={'highway':roads})

        food["function"] = "food"
        school["function"] = "school"
        healthcare["function"] = "healthcare"
        entertainment["function"] = "entertainment"
        public["function"] = "public"

        retail["function"] = "retail"
        sport["function"] = "sport"
        leisure["function"] = "leisure"
        highway["function"] = "highway"


        food_buildings = buildings.sjoin(food, how="left")
        school_buildings = buildings.sjoin(school, how="left")
        healthcare_buildings = buildings.sjoin(healthcare, how="left")
        entertainment_buildings = buildings.sjoin(entertainment, how="left")
        public_buildings = buildings.sjoin(public, how="left")
        retail_buildings = buildings.sjoin(retail, how="left")
        
        sport_buildings = buildings.sjoin(sport, how="left")
        leisure_buildings = buildings.sjoin(leisure, how="left")
        highway_buildings = buildings.sjoin(highway, how="left")

        #---------------------------------------------------------

        buildings.reset_index(inplace=True)
        food_buildings.reset_index(inplace=True)
        school_buildings.reset_index(inplace=True)
        healthcare_buildings.reset_index(inplace=True)
        entertainment_buildings.reset_index(inplace=True)
        public_buildings.reset_index(inplace=True)
        retail_buildings.reset_index(inplace=True)

        sport_buildings.reset_index(inplace=True)
        leisure_buildings.reset_index(inplace=True)
        highway_buildings.reset_index(inplace=True)
        #-----------------------------------------------------------
        buildings["function"] = np.nan
        buildings.function.fillna(food_buildings.function,inplace=True)
        buildings.function.fillna(school_buildings.function,inplace=True)
        buildings.function.fillna(healthcare_buildings.function,inplace=True)
        buildings.function.fillna(entertainment_buildings.function,inplace=True)
        buildings.function.fillna(public_buildings.function,inplace=True)
        buildings.function.fillna(retail_buildings.function,inplace=True)

        buildings.function.fillna(sport_buildings.function,inplace=True)
        buildings.function.fillna(leisure_buildings.function,inplace=True)
        buildings.function.fillna(highway_buildings.function,inplace=True)

        print(buildings.function.value_counts())
        self.buildings = buildings
        self.city_bbox = buildings.total_bounds
        if buildings is not None:
                print("--------------------------------------------------------------")
                print("Building geodataframes extracted successfully.")
                print("Number of buildings: {}".format(buildings.shape[0]))
                print("Bounding box of the city: {}".format(self.city_bbox))
        return self.buildings, self.city_bbox


if __name__ == "__main__":
    city_data = CityData("Berlin, Germany")
    buildings, city_bbox = city_data.get_data()
    print(buildings.shape)
    print(city_bbox)
