'''

Download required modules or packages here:

'''

import arcpy
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time
import shutil
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic

''' 
Philip Yang
PhD Comprehensive Exam mini project

'''

''' THIS CODE CREATES THE PYTHON TOOLBOX.   
NOTE: This cannot be run inside ArcGIS unless the Python environment has been duplicated and run to install all of the 
packages and modules above. It will run fine in any Python environment that has these packages.

THE REQUIREMENTS FOR THIS SCRIPT INCLUDE: 
(1) A DATASET WITH LOCATION DATA IN LAT LON IN DECIMAL DEGREES AND 
(2) A DATASET IN ONE FOLDER OF NOAA STORM OR HURRICANE DATA THAT HAVE SUBFOLDERS WITH PT SHAPEFILES ALSO IN IN DECIMAL DEGREES

'''
class Toolbox(object):
    def __init__(self):
        """Calculating distances from some location data to the center of storms"""
        self.label = "Processing storm distance data to compare to time-series data"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [CSV_toShapefile_distanceCalculation]


class CSV_toShapefile_distanceCalculation(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Hurricane to location distance and plotting tool"
        self.description = "Takes csv with XY coordinate columns and converts to a shapefile of points"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []
        input_csv = arcpy.Parameter(name="Input csv",
                                     displayName="Input csv file",
                                     datatype= "DETable",
                                     parameterType="Required",  # Required|Optional|Derived
                                     direction="Input",  # Input|Output
                                     )

        '''THIS IS WHERE YOU CHANGE THE INPUT PATH OF LOCATION DATA'''
        input_csv.value = r"C:\Users\marecotec\Desktop\PFYang\ComprehensiveExams\DATA\locationData.csv"
        params.append(input_csv)

        Site_column = arcpy.Parameter(name="Input csv site column name",
                                    displayName="Input csv site column name",
                                    datatype="GPString",
                                    parameterType="Optional",  # Required|Optional|Derived
                                    direction="Input",  # Input|Output
                                    )

        '''THIS IS WHERE YOU CHANGE THE INPUT COLUMN NAME OF THE SITE NAME'''
        Site_column.value = "Site" # example value
        params.append(Site_column)

        input_folder = arcpy.Parameter(name="Input hurricane data",
                                       displayName="Input hurricane folder",
                                       datatype="DEFolder",
                                       parameterType="Required",  # Required|Optional|Derived
                                       direction="Input",  # Input|Output
                                       )

        '''THIS IS WHERE YOU CHANGE THE INPUT FOLDER PATH OF INPUT HURRICANE DATA'''
        input_folder.value = r"C:\Users\marecotec\Desktop\PFYang\ComprehensiveExams\DATA\Hurricanes_2019_2020"
        params.append(input_folder)

        output_folder = arcpy.Parameter(name="Output folder name and path",
                                        displayName="Output folder name and path",
                                        datatype="DEFolder",
                                        parameterType="Required",  # Required|Optional|Derived
                                        direction="Output",  # Input|Output
                                        )

        '''THIS IS WHERE YOU SELECT WHERE YOU WOULD LIKE THE OUTPUT FOLDER LOCATED AND NAMED'''
        output_folder.value = r"C:\Users\marecotec\Desktop\PFYang\ComprehensiveExams\DATA\Test"
        params.append(output_folder)

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        '''IMPORT THE INPUTS FROM ABOVE HERE AS TEXT VALUES TO USE IN THE CODE BELOW'''
        csvFilePath = parameters[0].valueAsText
        SiteColumn = parameters[1].valueAsText
        hurricaneFolderPath = parameters[2].valueAsText
        outputFolderPath = parameters[3].valueAsText

        arcpy.env.overwriteOutput = True

        # Create output folder
        if not os.path.exists(outputFolderPath):
            os.mkdir(outputFolderPath)
        else: print(f"Output folder '{outputFolderPath}' NOT created")

        # Process csv file
        csvTable = pd.read_csv(csvFilePath)

        # This function returns the names of the input df and the lat lon columns
        def find_lat_lon_columns(csvTable):
            # Read the CSV file
            df = pd.read_csv(csvFilePath)

            # Identify columns that start with 'lat' or 'lon' (case insensitive)
            lat = [col for col in df.columns if col.lower().startswith(('lat'))]
            lon = [col for col in df.columns if col.lower().startswith(('lon'))]

            return df, lat, lon

        # Run function here
        df, lat, lon = find_lat_lon_columns(csvTable)
        print(f"{df.head()} \nLatitude and Longitude Columns:", lat, lon)

        # This function makes a plot of the lat lon points in the table of interest - PLOT SAVED AS OUTPUT IN OUTPUT FOLDER
        def plot_map(csv_file, site_column, output_folder):
            df, lat_cols, lon_cols = find_lat_lon_columns(csv_file)

            if not lat_cols or not lon_cols:
                print("No latitude or longitude columns found.")
                return

            lat_col = lat_cols[0]
            lon_col = lon_cols[0]

            fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([df[lon_col].min() - 1, df[lon_col].max() + 1, df[lat_col].min() - 1, df[lat_col].max() + 1])

            ax.add_feature(cfeature.LAND, color='tan')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

            ax.scatter(df[lon_col], df[lat_col], c='red', marker='o', transform=ccrs.PlateCarree())

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Map of Sites")
            ax.legend()

            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, "map_output.png")
            plt.savefig(output_path)
            plt.show()
            print(f"Map saved to {output_path}")

        # Run Function here
        plot_map(df, SiteColumn, outputFolderPath)

        print("\nMaking Point Shapefile from Locations...")

        '''RUN THE XY TABLE TO POINT TOOL IN ARCPY TO MAKE A POINT SHAPEFILE FROM THE DATA'''
        # Define the tool variables
        in_table = csvFilePath
        out_feature_class = os.path.join(outputFolderPath, "location_points.shp")
        x_field = lat[0] if lat else None
        y_field = lon[0] if lon else None
        z_field = ""
        coordinate_system = arcpy.SpatialReference(4326)  # WGS 1984

        # Run tool
        arcpy.management.XYTableToPoint(in_table,
                                        out_feature_class,
                                        x_field,
                                        y_field,
                                        z_field,
                                        coordinate_system)

        if arcpy.Exists(out_feature_class):
            arcpy.AddMessage(f"Point Shapefile for {csvFilePath} Created in {out_feature_class}")
        else:
            arcpy.AddMessage("Point Shapefile Not Created :`(")

        # Now deal with hurricane data and get the folder paths with this function
        def get_folders(hurricaneFolderPath):
            """
            Returns a list of folder names within the specified directory.

            Parameters:
                hurricaneFolderPath (str): The path to the directory.

            Returns:
                list: A list of folder names within the directory.
            """
            if not os.path.isdir(hurricaneFolderPath):
                print(f"Error: {hurricaneFolderPath} is not a valid directory.")
                return []

            return [name for name in os.listdir(hurricaneFolderPath) if
                    os.path.isdir(os.path.join(hurricaneFolderPath, name))]

        # Run here
        hurricaneFolders = get_folders(hurricaneFolderPath)
        print("\nNames of the hurricane folders within", hurricaneFolders)

        '''FIND THE SHAPEFILES IN THE HURRICANE OR STORM FOLDERS'''
        def find_shapefiles(hurricaneFolderPath):
            """
            Iterates through each folder in hurricaneFolderPath and finds files matching *_pts.shp.

            Parameters:
                hurricaneFolderPath (str): The path to the main directory containing hurricane folders.

            Returns:
                list: A list of full file paths for all matching shapefiles.
            """
            shapefile_paths = []

            # Get all folders within hurricaneFolderPath
            folders = get_folders(hurricaneFolderPath)

            for folder in folders:
                folder_path = os.path.join(hurricaneFolderPath, folder)

                # Use glob to find all files matching *_pts.shp in the current folder
                matching_files = glob.glob(os.path.join(folder_path, "*_pts.shp"))

                # Add found file paths to the list
                shapefile_paths.extend(matching_files)

            return shapefile_paths

        print("\nSaved shapefile paths for *_pts.shp search because that is the file to calculate distance")


        '''MAKE A PLOT OF THE HURRICANE TRACKS AND LABEL BY THE HURRICANE CODE - SAVE THE PLOT IN THE OUTPUT FOLDER'''
        def plot_hurricane_tracks(hurricaneFolderPath, output_folder):
            """Plots all hurricane shapefiles on a map with extent set dynamically."""
            shapefiles = find_shapefiles(hurricaneFolderPath)

            if not shapefiles:
                print("No shapefiles found.")
                return

            all_lat = []
            all_lon = []

            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

            # Add geographic features
            ax.add_feature(cfeature.LAND, color='tan')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=":")

            for shp in shapefiles:
                gdf = gpd.read_file(shp)

                shapefile_label = os.path.basename(shp).split("_")[0]

                # Ensure CRS is converted to WGS 1984
                if gdf.crs is not None and gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs(epsg=4326)

                # Identify LAT and LON columns dynamically
                lat_col = next((col for col in gdf.columns if col.lower().startswith("lat")), None)
                lon_col = next((col for col in gdf.columns if col.lower().startswith("lon")), None)

                if lat_col and lon_col and "STORMNAME" in gdf.columns and "YEAR" in gdf.columns:
                    all_lat.extend(gdf[lat_col].dropna().tolist())
                    all_lon.extend(gdf[lon_col].dropna().tolist())

                    ax.scatter(gdf[lon_col], gdf[lat_col], marker='o', s=10, transform=ccrs.PlateCarree(),
                               label=f"{shapefile_label}")

                    # Add labels at the first point of each storm
                    first_row = gdf.iloc[0]
                    ax.text(first_row[lon_col], first_row[lat_col], f"{shapefile_label} {first_row['YEAR']}",
                            fontsize=8, ha='right', transform=ccrs.PlateCarree())

            if all_lat and all_lon:
                # Set extent based on min/max lat/lon values
                ax.set_extent([min(all_lon), max(all_lon), min(all_lat), max(all_lat)], crs=ccrs.PlateCarree())

            ax.set_title("Hurricane Tracks")
            ax.legend(loc="upper right")

            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, "hurricane_map.png")
            plt.savefig(output_path)
            plt.show()
            print(f"Map saved to {output_path}")

        # Example usage
        plot_hurricane_tracks(hurricaneFolderPath, outputFolderPath)


        '''THIS FUNCTION TAKES THE INPUT LOCATION SHAPEFILE AND THE INPUT HURRICANE SHAPEFILES AND ITERATIVELY 
        CALCULATES THE DISTANCE FROM ONE SITE TO ALL OF THE HURRICANE PATH LOCATIONS THROUGH TIME (DOING THIS FOR EACH SITE)
        
        THE FINAL OUTPUT IS SAVED AS A CSV FILE TO THE OUTPUT FOLDER COMBINING BOTH OF THE FILES INTO ONE WITH A NEW COLUMN DISTANCE IN KM
        '''
        def calculate_nearest_distances(location_shp, hurricaneFolderPath, output_folder):
            """Calculates distances between location points and hurricane tracks, merging all attributes into a CSV."""

            # Load location points shapefile
            location_gdf = gpd.read_file(location_shp)

            # Convert location points to WGS 1984 if necessary
            if location_gdf.crs is not None and location_gdf.crs != "EPSG:4326":
                location_gdf = location_gdf.to_crs(epsg=32615)

            # Find hurricane shapefiles
            hurricane_shapefiles = find_shapefiles(hurricaneFolderPath)

            if not hurricane_shapefiles:
                print("No hurricane shapefiles found.")
                return

            # Store results
            results = []

            for hurricane_shp in hurricane_shapefiles:
                print(f"\nProcessing {hurricane_shp}")

                hurricane_gdf = gpd.read_file(hurricane_shp)
                filename = os.path.basename(hurricane_shp)
                hurricane_name = filename.split("_pts.shp")[0]
                print(hurricane_name)

                # Convert hurricane points to WGS 1984 if necessary
                if hurricane_gdf.crs is not None and hurricane_gdf.crs != "EPSG:4326":
                    hurricane_gdf = hurricane_gdf.to_crs(epsg=32615)

                # Identify LAT and LON columns dynamically
                lat_col = next((col for col in hurricane_gdf.columns if col.lower().startswith("lat")), None)
                lon_col = next((col for col in hurricane_gdf.columns if col.lower().startswith("lon")), None)

                if lat_col and lon_col and "STORMNAME" in hurricane_gdf.columns and "YEAR" in hurricane_gdf.columns:
                    # Convert hurricane points to shapely Point geometries
                    hurricane_gdf["geometry"] = hurricane_gdf.apply(lambda row: Point(row[lon_col], row[lat_col]),
                                                                    axis=1)

                    # Loop through each location point
                    for _, loc_row in location_gdf.iterrows():
                        loc_point = loc_row.geometry  # Site location geometry

                        print("\nLoc Point:", loc_point.x, loc_point.y)

                        # Compute distances from this location to all hurricane points
                        for _, hurricane_row in hurricane_gdf.iterrows():
                            hurricane_point = hurricane_row.geometry

                            lat, lon = loc_point.x, loc_point.y  # Ensure lat/lon order for geodesic distance
                            hurr_lat, hurr_lon = hurricane_point.y, hurricane_point.x

                            print("Lat and Lon:", lat, lon, "\nStorm Lat and Lon:", hurr_lat, hurr_lon)

                            # Compute geodesic distance
                            distance_km = geodesic((lat, lon), (hurr_lat, hurr_lon)).km

                            # Merge location attributes with hurricane attributes and distance
                            merged_data = {
                                "Site_Lat": loc_point.x,
                                "Site_Lon": loc_point.y,
                                **loc_row.to_dict(),
                                "Hurricane_Lat": hurr_lat,
                                "Hurricane_Lon": hurr_lon,
                                **hurricane_row.to_dict(),
                                "Distance_km": distance_km,
                                "StormCode": hurricane_name
                            }
                            results.append(merged_data)

            # Convert results to a DataFrame
            results_df = pd.DataFrame(results)

            # Save to CSV
            os.makedirs(output_folder, exist_ok=True)
            output_csv_path = os.path.join(output_folder, "hurricane_distance_results.csv")
            results_df.to_csv(output_csv_path, index=False)

            print(f"\n\nDistance calculations saved to {output_csv_path}")

        # Example usage
        calculate_nearest_distances(out_feature_class, hurricaneFolderPath, outputFolderPath)

        ''' THIS FUNCTION TAKES THE PRODUCED CSV FILE OF CALCULATED DISTANCES, CONVERTS DATETIME COL AND SAVES NEW DF 
        
        MAKES PLOTS FOR SPECIFIC SITES OR STORMS OF INTEREST AND SAVES THEM IN THE OUTPUT FOLDER
        '''
        def process_hurricane_data(file_path, site_name, storm_name, output_folder):
            # Load the dataset
            df = pd.read_csv(file_path)

            # Create the DateTime column from YEAR, MONTH, DAY, and HHMM (dropping minutes)
            df['DateTime'] = pd.to_datetime(
                df[['YEAR', 'MONTH', 'DAY']].astype(int).astype(str).agg('-'.join, axis=1) +
                ' ' + df['HHMM'].astype(str).str.zfill(4).str[:2] + ':00'
            )

            # Save the filtered dataframe to a CSV file
            output_csv_path = os.path.join(output_folder, f"Location_Distances_DateTimeCol.csv")
            df.to_csv(output_csv_path, index=False)
            print(f"\nFiltered data saved to: {output_csv_path}")

            # Filter for the specified Site and Storm Name
            df_filtered = df[(df['Site'] == site_name) & (df['StormCode'] == storm_name)].copy()

            # Convert Distance_km to numeric
            df_filtered['Distance_km'] = pd.to_numeric(df_filtered['Distance_km'], errors='coerce')

            # Print head of the new dataframe and check for NA values
            print(df_filtered.head())
            print("\nMissing values:\n", df_filtered.isna().sum())

            # Ensure output folder exists
            os.makedirs(output_folder, exist_ok=True)

            # Plot boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='Site', y='Distance_km')
            plt.xticks(rotation=30)
            plt.xlabel('Site')
            plt.ylabel('Distance (km)')
            plt.title(f'Hurricane {storm_name} Distance by Site')

            # Save plot
            plot_path = os.path.join(output_folder, f"{storm_name}_distance_boxplot.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"\nPlot saved to: {plot_path}")

            # Plot Distance_km over DateTime and save it
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df_filtered, x='DateTime', y='Distance_km', marker='o', color='b')
            plt.xlabel("DateTime")
            plt.ylabel("Distance (km)")
            plt.title(f"Distance of Hurricane '{storm_name}' from Site '{site_name}'")
            plt.xticks(rotation=20)
            plt.grid()

            # Save the plot
            plot_path = os.path.join(output_folder, f"{site_name}_{storm_name}_distance_plot.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {plot_path}")
            plt.show()
            plt.close()

            # Map plotting and saving
            if not df_filtered.empty:
                site_lat, site_lon = df_filtered.iloc[0][['Site_Lat', 'Site_Lon']]
                hurricane_lats = df_filtered['Hurricane_Lat']
                hurricane_lons = df_filtered['Hurricane_Lon']

                fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent([min(hurricane_lons) - 2, max(hurricane_lons) + 2,
                               min(hurricane_lats) - 2, max(hurricane_lats) + 2])

                # Add land and coastline features
                ax.add_feature(cfeature.LAND, color='lightgray')
                ax.add_feature(cfeature.COASTLINE)
                ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)

                # Plot hurricane track
                ax.plot(hurricane_lons, hurricane_lats, marker='o', color='red', linestyle='-', label="Hurricane Path")

                # Plot site location
                ax.scatter(site_lon, site_lat, color='blue', s=100, edgecolor='black', marker='*', label="Site Location")

                ax.set_title(f"Hurricane '{storm_name}' Track Near '{site_name}'")
                ax.legend()

                # Save the map
                map_path = os.path.join(output_folder, f"{site_name}_{storm_name}_hurricane_map.png")
                plt.savefig(map_path, dpi=300, bbox_inches='tight')
                print(f"\nMap saved to: {map_path}")
                plt.show()
                plt.close()
            else:
                print(f"\nNo data available for site '{site_name}' and storm '{storm_name}'.")

        # Example usage with naming the output csv
        input_csv_path = os.path.join(outputFolderPath, "hurricane_distance_results.csv")
        process_hurricane_data(input_csv_path, "STE", "AL132020", outputFolderPath)

        return



# This code block allows you to run your code in a test-mode within PyCharm, i.e. you do not have to open the tool in
# ArcMap. This works best for a "single tool" within the Toolbox.
def main():
    tool = CSV_toShapefile_distanceCalculation()
    tool.execute(tool.getParameterInfo(), None)

if __name__ == '__main__':
    main()
