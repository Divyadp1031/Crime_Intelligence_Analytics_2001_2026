
import geopandas as gpd

# Read India district boundary data
gdf = gpd.read_file(
    "data/india_districts/2011_Dist.shp"
)

# Display district information
print("District columns:")
print(gdf.columns)

print("\nFirst five districts:")
print(gdf.head())

# Convert into GeoJSON
gdf = gdf.to_crs(epsg=4326)

gdf.to_file(
    "data/india_districts.geojson",
    driver="GeoJSON"
)

print("\nGeoJSON created successfully!")
