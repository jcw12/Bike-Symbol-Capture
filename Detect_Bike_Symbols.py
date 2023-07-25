# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 17:00:16 2023

@author: James Currie, s3901655, jcurrie987@gmail.com
"""

import pandas as pd
import os
import fnmatch
import geopandas
import geopandas as gpd
from shapely.geometry import box

from ultralytics import YOLO
import requests

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(r"E:\Bicycle Capture Github\Bike-Symbol-Capture/")
path = r"Test_images/"
pics = os.listdir(path)



"""
THIS SECTION DEFINS FUNCTIONS THAT TRANSLATE BETWEEN THE VICGRID AND IMAGE TILE COORDINATES

"""

#FIND CORNER - finds the vicgrid coordinates at the bottom left corner of an image tile.
def fc(Row,Col):
    Row0 = 3081000.000000
    Col0 = 1786000.000000
    
    imgX = Col0 + (Col*108.374) 
    imgY = Row0 - (Row*108.374) - 108.374 
    return imgX, imgY


#Findrowcol
def frc(imgX, imgY):
    Row0 = 3081000.000000
    Col0 = 1786000.000000

    Col = (imgX - Col0) / 108.374
    Row = (Row0 - imgY - 108.374) / 108.374

    return Row, Col


"""
THIS SECTION SCRAPES THE IMAGE TILES IN THE DEFINED AREA FROM THE VICMAP BASEMAP

"""

#Bounding box to scrape from vicmap basemap
def Scrape(Topleft, Lowright, outFolder = None):
    
    tlx,tly = Topleft[0], Topleft[1]
    lrx, lry = Lowright[0], Lowright[1]
        
    tlr, tlc = frc(tlx,tly)
    lrr, lrc = frc(lrx,lry)
    tlr = int(tlr)
    tlc = int(tlc)
    lrr = int(lrr)
    lrc = int(lrc)
    headers={'User-Agent': 'Mozilla/5.0',
             'Referer':'https://delwp.maps.arcgis.com/home/webmap/viewer.html?webmap=10bb5f1bb12945af9730c6794e3d1430',
             'Sec-Ch-Ua-Platform':"Windows"}
    print(range(tlr,lrr))
    for i in range(tlr,lrr):
        for j in range(tlc,lrc):
            print('Scraping Image:',i,j)
            URL = "https://base.maps.vic.gov.au/service?SERVICE=WMTS&VERSION=1.0.0&REQUEST=GetTile&LAYER=AERIAL_VG2020&STYLE=default&FORMAT=image/png&TILEMATRIXSET=EPSG:7899&TILEMATRIX=13&TILEROW="+str(i)+"&TILECOL="+str(j)
            response = requests.get(URL, headers = headers, stream = True)
            with open(outFolder +'/'+ str(i)+str(j)+'VICCIP.png', 'wb') as f:
                 f.write(response.content)
            print('image saved to',outFolder +'/'+ str(i)+str(j)+'VICCIP.png')
    return tlr, tlc, lrr, lrc

#Scrape([2499670.8,2397044.5],[2501883.6,2394108.2],r'E:\Bicycle Capture Github\Bike-Symbol-Capture\Test_images')

"""
THIS SECTION RUNS THROUGH THE SCRAPED IMAGES USING THE MODEL WEIGHTS, THEN OUTPUTS THE RESULTS TO A CSV AND POINTS SHP WITH CONFIDENCE AS AN ATTRIBUTE

"""
def predict_and_export(image_path, output_csv, output_shapefile, base_coordinates, yolo_model_weights):
    # Import ultralytics YOLO model
    model = YOLO(yolo_model_weights)

    Row0, Col0 = base_coordinates[1], base_coordinates[0]

    df = pd.DataFrame(columns=['X', 'Y', 'Confidence'])

    for i, pic in enumerate(pics):
        print('image', i)
        results = model(path + pic)

        for bbox in results[0].cpu().boxes.data.numpy():
            x_min, y_min, x_max, y_max = bbox[0:4]

            centroid = [((x_min + x_max) / 2), ((y_min + y_max) / 2)]

            row = float(pic[:4])
            col = float(pic[4:8])

            imgX = Col0 + (col * 108.374)
            imgY = Row0 - (row * 108.374) - 108.374

            normalised_bx = 108.374 * (centroid[0] / 512)
            normalised_by = 108.374 * (centroid[1] / 512)

            X = imgX + normalised_bx - 3
            Y = imgY + 108.374 - normalised_by + 3

            df.loc[len(df)] = [X, Y, float(bbox[4])]

    df.to_csv(output_csv)

    gdf = geopandas.GeoDataFrame(df)
    gdf.set_geometry(geopandas.points_from_xy(gdf['X'], gdf['Y']), inplace=True, crs='EPSG:7899')
    gdf.to_file(output_shapefile)


"""
THIS SECTION CREATES CONTEXTUAL INFORMATION - THE TARGET AREA WHERE THE BIKE LANES ARE DETECTED AND THE LOCATIONS OF THE LABELLED DATASETS AS SHAPEFILES
"""

#This outputs a shapefile of the area which inwhich bikes have been detected
def target_area(top_left, bottom_right, output_filename):
    
    # Create a bounding box polygon using the coordinates
    bbox = box(*top_left, *bottom_right)
    
    # Convert the bounding box to a GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:7899")
    # Save the GeoDataFrame as a shapefile
    gdf.to_file(output_filename)

def find_jpeg(folder_path):
    jpg_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in fnmatch.filter(files, "*.jpg"):
            namepath = os.path.join(root, filename)
            if 'test' not in namepath:
                jpg_files.append(filename)
                
    return jpg_files

def labelled_imgs(folder_path):
    labelled = find_jpeg(folder_path)
    i = 0
    for img in labelled:
        print(img)
        coord = fc(int(img[:4]), int(img[4:8]))
        tl = coord[0], (coord[1] + 108.374)
        lr = (coord[0] + 108.374), coord[1]
        bbox = box(*tl, *lr)
        if i != 1:
            gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:7899")
        else:
            gdf1 = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:7899")
            gdf = pd.concat([gdf, gdf1])
        i = 1
    gdf = gdf.dissolve()
    gdf = gdf.dissolve()
    gdf.to_file('Labelled.shp')

"""
THIS SECTION DOES IT ALL
"""

#This is the complete package - input the coordinates of th bbox in which you want to detect bikes, and the imagery is scraped and the model run
def detect_bikes(Topleft, Lowright, output_csv, output_shapefile, downFolder, base_coordinates=[1786000.000000, 3081000.000000],yolo_model_weights= r'model\weights\best.pt'):
    target_area(Topleft,Lowright,'Target_Area.shp')
    Scrape(Topleft,Lowright,downFolder)
    predict_and_export(downFolder,output_csv, output_shapefile, base_coordinates, yolo_model_weights)
    print('detect_bikes complete')
    
detect_bikes([2499035.6,2409288.1],[2501254.1,2407376.6],'Predicted_Bikes.csv','Predicted_Bikes.shp','Test_images')
