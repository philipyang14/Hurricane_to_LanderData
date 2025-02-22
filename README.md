# Hurricane_to_LanderData
 
Philip Yang
2/22/25
Comprehensive Exam - PhD Biological Oceanography

Taking hurricane or tropical storm paths and calculating the distance from the center of the storm to some time-series data collection location. Provided is a python toolbox in the 'CODE' folder that can take any storm location information and find the distance to some number of location objects over the time measurements of the storm. Here we use NOAA storm tracking data (https://www.nhc.noaa.gov/data/tcr/) and locations of sensors placed on the seafloor for two year time periods in the 'DATA' folder as example for how the toolbox should run. 

The toolbox can be imported into ArcGIS Pro and used like any other toolbox for other similar datasets. The prerequisite is to have a csv file with at least two columns named some variation of LAT and LON in abbreviation or full in lower and upper case (latitude, longitude) and a folder with the NOAA storm shapefiles that have been unzipped.