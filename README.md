# Bike-Symbol-Capture
Using a CNN (Yolov8) to capture bike symbols from aerial imagery. Relevant for finding cycling corridors.

This project used Roboflow for labeling data and used unlicenced CIP imagery from the Vicmap Basemap, meaning that it is <b><ins><i>not suitable for commercial use.</b></ins></i>

---

To use the model to find bicycle symbols, run the 'detect_bikes' function, inputting:

-The Top left and Bottom right coordinates of the area you would like to detect bicycles in (VicGrid2020, epsg:7899), 

-A path to the output csv and shapefile,

-And the path to a folder to store the scraped images.

---

By default the function is run as:

<i>detect_bikes([2499035.6,2409288.1],[2501254.1,2407376.6],'Predicted_Bikes.csv','Predicted_Bikes.shp','Test_images')</i>

The outputs of this are included in the dataset. 'Target_Area' is the area defined by the input coordinates, and the 'Predicted_Bikes'.shp contains the location of predicted bikes, with the confidence as an attribute.