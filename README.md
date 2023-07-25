# Bike-Symbol-Capture
Using a Convolutional Neural Network (Yolov8n) to capture bike symbols from aerial imagery. Relevant for finding cycling corridors.

<img src="https://github.com/jcw12/Bike-Symbol-Capture/assets/36462497/d444e6b5-55e3-42df-be1e-42bb188fb147)">

This project used Roboflow for labeling data and used unlicensed CIP imagery from the Vicmap Basemap, meaning that it is <b><ins><i>not suitable for commercial use.</b></ins></i>

---

To use the model to find bicycle symbols, run the 'detect_bikes' function, inputting:

-The Top left and Bottom right coordinates of the area you would like to detect bicycles in (VicGrid2020, epsg:7899), 

-A path to the output csv and shapefile,

-And the path to a folder to store the scraped images.

---

By default the function is run as:

<i>detect_bikes([2499035.6,2409288.1],[2501254.1,2407376.6],'Predicted_Bikes.csv','Predicted_Bikes.shp','Test_images')</i>

The outputs of this are included in the dataset. 'Target_Area' is the area defined by the input coordinates, and the 'Predicted_Bikes'.shp contains the location of predicted bikes, with the confidence as an attribute.

---
<i>This model was trained on magnitudes less than the recommended training data, and is extremely inaccurate, capturing less than half of the admittedly small validation / test data. This is a proof of concept, not a finished product. </i>

This project was created by James Currie, reach out to me at jcurrie987@gmail.com with any queries or comments
