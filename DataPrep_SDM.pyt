
import arcpy
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time
import shutil

base_dir = r"C:\Users\Philip Yang\OneDrive - University of Rhode Island\NRS_528\ArcGIS_Python_Class\Final_Toolbox"
arcpy.env.overwriteOutput = True

''' 
Philip Yang
Final toolbox project
NRS 528

This toolbox takes one csv file of habitat occurrence data (Marissa Nuttall, NOAA FGB NMS) and intersects it with a 
5 by 5 m bathymetry mosaic tif (FGB NMS, USGS, mosaicked by Philip) and creates a new raster that possesses cells with 
the most common habitat value from the point shapefile. Then, a new raster is built from that in-between raster where any 
habitat values of "Coralline algae" are assigned a new field value of 1 and the other two values (Deep reef and soft bottom)
are assigned a value of 0. This is to create a final output raster that contains the occurrence data of coralline algae 
from this dataset in a raster format that overlaps with the 5 by 5 bathymetry mosaic. The data needed for this toolbox 
are in the zipped folders GoM_occurrence_data and fgb_bathy_mosaic. 

RUN THE SCRIPTS AT THE BOTTOM OF EACH TOOL BY UNCOMMENTING 'def main():....main()'
To avoid issues, run the tools individually.
'''

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file). Python Toolbox XYTable to Point to Intersect with ArcGIS Shapefile to New GRID Shapefile"""
        self.label = "Processing Data for Habitat Distribution Modelling"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [ProcessCSV_ToShapefile, ExtractMultiRasterValues, Habitat_DataCleaning, DeleteFolder]


class ProcessCSV_ToShapefile(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "XY to Point Tool"
        self.description = "Takes csv with XY coordinate columns and converts to a shapefile of points"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []
        input_folder = arcpy.Parameter(name="input_folder",
                                     displayName="Input folder with csv files",
                                     datatype="DEFolder",
                                     parameterType="Required",  # Required|Optional|Derived
                                     direction="Input",  # Input|Output
                                     )
        input_folder.value = os.path.join(base_dir, "GoM_occurrence_data")  # This is a default value that can be over-ridden in the toolbox
        params.append(input_folder)

        output_folder = arcpy.Parameter(name="output_folder",
                                        displayName="Output folder",
                                        datatype="DEFolder",
                                        parameterType="Required",  # Required|Optional|Derived
                                        direction="Output",  # Input|Output
                                        )
        output_folder.value = os.path.join(base_dir, "temporary_files")  # This is a default value that can be over-ridden in the toolbox
        params.append(output_folder)

        output_points = arcpy.Parameter(name="output_points",
                                        displayName="Output shapefile",
                                        datatype="DEFeatureClass",
                                        parameterType="Required",  # Required|Optional|Derived
                                        direction="Output",  # Input|Output
                                        )
        output_points.value = os.path.join(base_dir, "temporary_files", "habitat_occurrence_raw_XYTableToPoint.shp")  # This is a default value that can be over-ridden in the toolbox
        params.append(output_points)

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

        in_folder = parameters[0].valueAsText
        temp_folder = parameters[1].valueAsText
        out_feature_class = parameters[2].valueAsText

        # Create output folder
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

        print("\nProcessing CSV files in folder:", in_folder)
        print("Files should show up in:", temp_folder)

        # Initialize an empty list to store DataFrame objects
        dfs = []

        # Iterate over CSV files in the folder
        for csv_file in glob.glob(os.path.join(in_folder, "*.csv")):
            print("\n",os.path.basename(csv_file))

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file, low_memory=False)

            # Check if "Major Habitat" column exists - only a problem with the Nuttall dataset
            if "Major Habitat" and "Level 6: Macrohabitat" and "Level 7: Habitat" in df.columns:
                # Rename "Major Habitat" column to "Habitat"
                df.rename(columns={"Major Habitat": "Habitat"}, inplace=True)
                df.drop(columns={"Level 6: Macrohabitat"}, inplace=True) # Find the other columns named 'habitat' that we don't want and drop them
                df.drop(columns={"Level 7: Habitat"}, inplace=True)
            else: print("\nNo column called 'Major Habitat' in the CSV file:", csv_file)
            print(df)

            # Create a new DataFrame with selected columns
            selected_df = pd.DataFrame()

            # Rename selected columns and add to the new DataFrame
            for col in df.columns:
                if "latitude" in col.lower():
                    selected_df["Latitude"] = df[col]
                elif "longitude" in col.lower():
                    selected_df["Longitude"] = df[col]
                elif "depth" in col.lower():
                    selected_df["DepthInMeters"] = df[col]
                elif "habitat" in col.lower():
                    selected_df["Habitat"] = df[col]

            # # Keep only the first four selected columns
            selected_df = selected_df.iloc[:, :4]
            print(selected_df)

            # Add the source filename as a new column filled with the filename
            selected_df['source_file'] = os.path.splitext(os.path.basename(csv_file))[0]

            # Add the DataFrame to the list
            dfs.append(selected_df)

        # Concatenate all DataFrames in the list along the rows (axis=0)
        concatenated_df = pd.concat(dfs, ignore_index=True)
        print(concatenated_df)

        # Write the concatenated DataFrame to a new CSV file
        occurrence_all_csv = os.path.join(temp_folder, "occurrence_all.csv")
        concatenated_df.to_csv(occurrence_all_csv, index=False)
        print("CSV files have been processed and concatenated successfully!")

        # See the unique values in habitat:
        df = pd.read_csv(occurrence_all_csv)
        column_name = "Habitat"
        value_counts = df[column_name].value_counts()
        print("\nThe unique values in the dataset are:","\n",value_counts, "\nWe want to standardize them")

        # Recode Habitat column to standardize the data
        recoding_codes = {
            "Coralline Algae": ["Coralline Algae",
                                 "Algal Nodules",
                                 "Coralline algae",
                                 "Coralline Algal Reef",
                                 "Coralline Algae Reef",
                                 "Corraline Algae",
                                 "Coralline Aglae",
                                 "Algal nodules",
                                 "Algal Nodules "],
            "Soft Bottom": ["Soft Bottom",
                             "Soft Substrate",
                             "Soft bottom",
                             "Soft Bottom ",
                             "soft bottom"],
            "Deep Reef": ["Deep Reef",
                          "Deep Reef ",
                           "Deep reef",
                           "Deep Coral",
                           "Deep Reefs",],
            "Reef": ["Reef",
                     "Coral Reef",
                     "Coral Community",
                     "Coral Reef "]
        }
        values_to_drop = ["Brine Seep", "Mud Volcano"]

        # Load the DataFrame from the "occurrence_all.csv" file
        occurrence_all_csv = os.path.join(temp_folder, "occurrence_all.csv")
        df = pd.read_csv(occurrence_all_csv)

        # Drop rows with values we are not interested in
        mask = df['Habitat'].isin(values_to_drop)
        df = df[~mask]
        # Drop rows where either latitude or longitude has no value
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

        # Recode the "Habitat" column based on the provided codes
        for new_value, old_values in recoding_codes.items():
            for old_value in old_values:
                df.loc[df['Habitat'].str.lower() == old_value.lower(), 'Habitat'] = new_value
        print(df)
        column_names = ["Habitat", "Latitude", "Longitude", "DepthInMeters"]
        value_counts = df[column_names[0]].value_counts()
        print(value_counts)
        df = df.dropna(subset=['Habitat'])

        # Deal with DepthInMeters not being a float
        # Convert non-numeric values to NaN
        df[column_names[3]] = pd.to_numeric(df[column_names[3]], errors='coerce')
        print(df.info())
        df[column_names[3]] = df[column_names[3]].astype(float)
        # print(df.info())

        # Check if any value in the column is not negative
        if (df['DepthInMeters'] >= 0).any():
            # Print a message indicating that not all values are negative
            print("\nWarning: Some values in 'DepthInMeters' column are not negative. Processing...")
            # Modify values to make them negative
            df.loc[df['DepthInMeters'] >= 0, 'DepthInMeters'] *= -1

        # Test to check that they were all converted
        if (df['DepthInMeters'] < 0).any():
            print("...All values in 'DepthInMeters' column are now negative.")

        print("\nThe order of the integers for x axis of habitats should follow this left to right: ",
              df['Habitat'].unique(), "Could not figure out for hours why PyCharm and PLT don't like categories here, "
                                      "but if the order is correct it seems Reef and Deep Reef depth ranges are not different")

        # Convert 'Habitat' column to categorical data type
        df['Habitat'] = df['Habitat'].astype('category')
        # Replace 'DepthInMeters' and 'Habitat' with actual column names from your DataFrame
        sns.boxplot(x=df['Habitat'], y='DepthInMeters', data=df)
        plt.title("LEFT TO RIGHT SHOULD BE:" "'Soft Bottom' 'Deep Reef 'Coralline Algae' 'Reef'")
        plt.xlabel('Habitat')
        plt.ylabel('Depth in Meters')
        # Add figure caption
        fig_caption = (
            "Figure 1: Boxplot of Depth in Meters by Habitat. The order of categories from 0 to 3 corresponds to "
            "the unique values: ['Soft Bottom' 'Deep Reef' 'Coralline Algae' 'Reef']")
        plt.text(0.5, -0.1, fig_caption, ha='center', fontsize=10, transform=plt.gcf().transFigure)
        plt.show()
        print("\n")

        # Check if any column contains NaN values
        for colname in column_names:
            if df[colname].isna().any():
                print(f"The column '{colname}' contains NaN values.")
            else:
                print(f"The column '{colname}' does not contain NaN values.")

        # Save the cleaned DataFrame to a new CSV file
        occurrence_all_clean_csv = os.path.join(temp_folder, "occurrence_all_clean.csv")
        df.to_csv(occurrence_all_clean_csv, index=False)

        print(f"\nCleaned DataFrame saved to {occurrence_all_clean_csv}")

        # Now we are ready to create a XY Point Shapefile!
        print("\nMaking Point Shapefile...")
        # Define the tool variables
        in_table = os.path.join(temp_folder, "occurrence_all_clean.csv")
        x_field = "Longitude"
        y_field = "Latitude"
        z_field = ""
        coordinate_system = arcpy.SpatialReference(4326) # WGS 1984

        # Run tool
        arcpy.management.XYTableToPoint(in_table,
                                        out_feature_class,
                                        x_field,
                                        y_field,
                                        z_field,
                                        coordinate_system)

        if arcpy.Exists(out_feature_class):
            arcpy.AddMessage(" Point Shapefile Created!")
        else: arcpy.AddMessage("Point Shapefile Not Created :`(")

        return


# This code block allows you to run your code in a test-mode within PyCharm, i.e. you do not have to open the tool in
# ArcMap. This works best for a "single tool" within the Toolbox.
# def main():
#     tool = ProcessCSV_ToShapefile()
#     tool.execute(tool.getParameterInfo(), None)
#
# if __name__ == '__main__':
#     main()


class ExtractMultiRasterValues(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Extract raster values using a point shapefile"
        self.description = "Inputting a bathymetry grid and shapefile containing multiple points, run Extract Multi Value to Point to get the depth values for each habitat occurrence point"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []
        input_points = arcpy.Parameter(name="input_points",
                                     displayName="Input points",
                                     datatype="DEShapeFile",
                                     parameterType="Required",  # Required|Optional|Derived
                                     direction="Input",  # Input|Output
                                     )
        input_points.value = os.path.join(base_dir, "temporary_files", "habitat_occurrence_raw_XYTableToPoint.shp")  # This is a default value that can be over-ridden in the toolbox
        params.append(input_points)

        input_raster = arcpy.Parameter(name="input_raster",
                                        displayName="Input raster",
                                        datatype="GPRasterLayer",
                                        parameterType="Required",  # Required|Optional|Derived
                                        direction="Input",  # Input|Output
                                        )
        input_raster.value = os.path.join(base_dir, "fgb_bathy_mosaic", "fgb_mosaic_bathy_mercator_bilinear_resampled50m.tif")  # This is a default value that can be over-ridden in the toolbox
        params.append(input_raster)

        output_folder = arcpy.Parameter(name="output_folder",
                                        displayName="Output folder",
                                        datatype="DEFolder",
                                        parameterType="Required",  # Required|Optional|Derived
                                        direction="Input",  # Input|Output
                                        )
        output_folder.value = os.path.join(base_dir, "output_files") # This is a default value that can be over-ridden in the toolbox
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

        point_shapefile = parameters[0].valueAsText
        raster_file = parameters[1].valueAsText
        output_folder = parameters[2].valueAsText

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Describe the raster file for extracting multi values to point
        desc = arcpy.Describe(raster_file)
        extent = desc.extent
        xmin, ymin, xmax, ymax = extent.XMin, extent.YMin, extent.XMax, extent.YMax
        cell_width, cell_height = desc.meanCellWidth, desc.meanCellHeight
        print(f"Cell width: {cell_width}, \nCell height: {cell_height}, \nExtent: {extent}, \nxmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")

        raster = arcpy.Raster(raster_file)
        print("\nRaster formatted")

        # Delete the field: used to keep the same columns from populating everytime I ran the code
        arcpy.management.DeleteField(point_shapefile, "DepthInM_1")

        # Use Extract Multi Value to Point to obtain depth values from a bathymetry grid for habitat presence/absence locations
        print("\nExtracting multi values...")

        in_point_features = point_shapefile
        in_rasters = [[raster, "DepthInMeters"]]
        bilinear_interpolate_values = "NONE"

        arcpy.gp.ExtractMultiValuesToPoints(in_point_features, in_rasters, bilinear_interpolate_values)

        print("Raster multi values extracted to points!")

        # View the new attribute table of the point file to check the tool worked
        # Describe the shapefile to get its fields
        fields = arcpy.ListFields(point_shapefile)
        print("\nAttribute Table Header:")
        field_names = [field.name for field in fields]
        print("\t".join(field_names))
        # Open a search cursor to iterate through records
        with arcpy.da.SearchCursor(point_shapefile, field_names) as cursor:
            # Iterate through the first 10 rows
            for i, row in enumerate(cursor):
                if i >= 10:
                    break
                # Print the values for each field in the row
                print("\t".join(str(value) for value in row))

        # Count invalid values
        invalid_count = 0
        field_name = "DepthInM_1"
        # Open a search cursor to iterate through records
        with arcpy.da.SearchCursor(point_shapefile, field_name) as cursor:
            for row in cursor:
                depth_value = row[0]  # Assuming the field is single-valued
                if depth_value < -300 or depth_value > 0:
                    invalid_count += 1

        print(f"\nNumber of invalid depth values (-9999) found: {invalid_count}")

        return

# This code block allows you to run your code in a test-mode within PyCharm, i.e. you do not have to open the tool in
# ArcMap. This works best for a "single tool" within the Toolbox.
def main():
    tool = ExtractMultiRasterValues()
    tool.execute(tool.getParameterInfo(), None)

if __name__ == '__main__':
    main()


class Habitat_DataCleaning(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Edit the point shapefile"
        self.description = "Input a point shapefile for habitat occurrence and clean the shapefile, query the relevant points and make a new shapefile of the query. Then delete temporary file folder."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []
        input_points = arcpy.Parameter(name="input_points",
                                     displayName="Input points",
                                     datatype="DEShapeFile",
                                     parameterType="Required",  # Required|Optional|Derived
                                     direction="Input",  # Input|Output
                                     )
        input_points.value = os.path.join(base_dir, "temporary_files", "NWGOM18_CC_final_XYTableToPoint.shp")  # This is a default value that can be over-ridden in the toolbox
        params.append(input_points)

        output_points = arcpy.Parameter(name="output_points",
                                        displayName="Output point layer",
                                        datatype="GPString",
                                        parameterType="Required",  # Required|Optional|Derived
                                        direction="Output",  # Input|Output
                                        )
        output_points.value = os.path.join(base_dir, "output_files", "habitat_occurrence_clean.shp")  # This is a default value that can be over-ridden in the toolbox
        params.append(output_points)

        output_folder = arcpy.Parameter(name="output_folder",
                                        displayName="Output folder",
                                        datatype="DEFolder",
                                        parameterType="Required",  # Required|Optional|Derived
                                        direction="Input",  # Input|Output
                                        )
        output_folder.value = os.path.join(base_dir, "output_files")  # This is a default value that can be over-ridden in the toolbox
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

        point_shapefile = parameters[0].valueAsText
        output_shapefile = parameters[1].valueAsText
        output_folder = parameters[2].valueAsText

        field_name = "DepthInM_1"
        # Create a query to select features where the values are not between 0 and -300
        query = f"{field_name} < 0 OR {field_name} > -300"
        # Make a layer from the input shapefile
        arcpy.MakeFeatureLayer_management(point_shapefile, "temp_layer")
        # Select features using the query
        arcpy.SelectLayerByAttribute_management("temp_layer", "NEW_SELECTION", query)
        # Delete selected features
        arcpy.DeleteFeatures_management("temp_layer")
        # Save the changes to a new shapefile
        arcpy.CopyFeatures_management("temp_layer", output_shapefile)

        if os.path.exists(output_shapefile):
            print("Final shapefile of habitat occurrence locations created!")

        # Copy and output clean csv
        # Construct the full path for the output CSV file
        input_csv = os.path.join(base_dir, "temporary_files", "occurrence_all_clean.csv")
        output_csv = os.path.join(output_folder, "occurrence_all_clean.csv")
        shutil.copyfile(input_csv, output_csv)
        if os.path.exists(output_csv):
            print("Final csv created!")

        return

# This code block allows you to run your code in a test-mode within PyCharm, i.e. you do not have to open the tool in
# ArcMap. This works best for a "single tool" within the Toolbox.
# def main():
#     tool = Habitat_DataCleaning()
#     tool.execute(tool.getParameterInfo(), None)
#
# if __name__ == '__main__':
#     main()


class DeleteFolder(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Delete unwanted folders"
        self.description = "Delete temporary files folder from this tree"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []

        temp_folder = arcpy.Parameter(name="temp_folder",
                                        displayName="Temporary folder",
                                        datatype="DEFolder",
                                        parameterType="Required",  # Required|Optional|Derived
                                        direction="Input",  # Input|Output
                                        )
        temp_folder.value = os.path.join(base_dir, "temporary_files")  # This is a default value that can be over-ridden in the toolbox
        params.append(temp_folder)

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

        folder = parameters[0].valueAsText

        # # Delete temporary file directory
        if os.path.exists(folder):
            arcpy.Delete_management(folder)
            print("\nTemporary folder emptied...")
        else: print("Temporary directory not emptied...")

        arcpy.Delete_management(folder)

        print('Temporary folder fully deleted')

        return

# This code block allows you to run your code in a test-mode within PyCharm, i.e. you do not have to open the tool in
# ArcMap. This works best for a "single tool" within the Toolbox.
# def main():
#     tool = DeleteFolder()
#     tool.execute(tool.getParameterInfo(), None)
#
# if __name__ == '__main__':
#     main()