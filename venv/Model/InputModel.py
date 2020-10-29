import numpy as np
from osgeo import gdal, ogr, gdal_array# I/O image data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QMessageBox



class InputModule():

    Training_File_Path = ""
    Validation_File_Path = ""
    Trg_Attribute_Selected = ""
    Val_Attribute_Selected = ""
    RS_Image_Path = ""

    """ setter functions for class  """

    def set_training_file_path(self,path):
        self.Training_File_Path = path

    def set_validation_file_path(self,path):
        self.Validation_File_Path = path

    def set_trg_attribute_selected(self,attr):
        self.Trg_Attribute_Selected = attr

    def set_val_attribute_selected(self,attr):
        self.Val_Attribute_Selected = attr

    def set_rs_image_path(self,path):
        self.RS_Image_Path = path

    """ getter functions for the class """

    def get_training_path(self):
        return self.Training_File_Path

    def get_validation_path(self):
        return self.Validation_File_Path

    def get_trg_attribute_selected(self):
        return self.Trg_Attribute_Selected

    def get_val_attribute_selected(self):
        return self.Val_Attribute_Selected

    def get_rs_image_path(self):
        return self.RS_Image_Path

# loading function

    def loadimagedata(self,img_path):
        try:
            img_ds = gdal.Open(img_path, gdal.GA_ReadOnly)

            img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                       gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
            for b in range(img.shape[2]):
                img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
            return [img_ds,img]

        except ValueError as e:
            print("Error in loading Image file.")
            print(e)

# loading training/validation data

    def load_training_data(self,trg_path,Trg_Attribute_Selected,img_ds,type):
        try:
            driver = ogr.GetDriverByName('ESRI Shapefile')
            shape_dataset = driver.Open(trg_path)
            shape_layer = shape_dataset.GetLayer()
            mem_drv = gdal.GetDriverByName('MEM')
            if(type == "Classification"):
                mem_raster = mem_drv.Create('', img_ds.RasterXSize, img_ds.RasterYSize, 1, gdal.GDT_UInt16)
            if(type == "Regression"):
                mem_raster = mem_drv.Create('', img_ds.RasterXSize, img_ds.RasterYSize, 1, gdal.GDT_Float32)
            mem_raster.SetProjection(img_ds.GetProjection())
            mem_raster.SetGeoTransform(img_ds.GetGeoTransform())
            mem_band = mem_raster.GetRasterBand(1)
            mem_band.Fill(0)
            mem_band.SetNoDataValue(0)
            att_ = 'ATTRIBUTE=' + Trg_Attribute_Selected
            err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1], [att_, "ALL_TOUCHED=TRUE"])
            assert err == gdal.CE_None
            roi = mem_raster.ReadAsArray()
            return roi
        except ValueError as e:
            print("Could not load Training Data")
            print(e)

    def load_validation_data(self, Validation_File_Path, Val_Attribute_Selected, img_ds):
        try:
            print(Val_Attribute_Selected)
            shape_dataset_v = ogr.Open(Validation_File_Path)
            shape_layer_v = shape_dataset_v.GetLayer()
            mem_drv_v = gdal.GetDriverByName('MEM')
            mem_raster_v = mem_drv_v.Create('', img_ds.RasterXSize, img_ds.RasterYSize, 1, gdal.GDT_UInt16)
            mem_raster_v.SetProjection(img_ds.GetProjection())
            mem_raster_v.SetGeoTransform(img_ds.GetGeoTransform())
            mem_band_v = mem_raster_v.GetRasterBand(1)
            mem_band_v.Fill(0)
            mem_band_v.SetNoDataValue(0)

            att_ = 'ATTRIBUTE=' + Val_Attribute_Selected

            # # http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
            # # http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
            err_v = gdal.RasterizeLayer(mem_raster_v, [1], shape_layer_v, None, None, [1], [att_, "ALL_TOUCHED=TRUE"])
            assert err_v == gdal.CE_None
            roi_v = mem_raster_v.ReadAsArray()
            return roi_v

        except ValueError as e:
            print("Could not load Validation Data")
            print(e)


#loading attributes
    def FindAttributes(self, filepath):
        try:
            driver = ogr.GetDriverByName('ESRI Shapefile')
            shape_dataset = driver.Open(filepath)
            shape_layer = shape_dataset.GetLayer()
            field_names = [field.name for field in shape_layer.schema]
            return field_names
        except ValueError as e:
            print(e)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("NO ATTRIBUTES FOUND")
            msg.setText("NO ATTRIBUTES FOUND IN .SHP FILE. Atrribute Error ")
            msg.exec_()

    """ Creates 2 subplots of Training Data and Image  """

    def create_training_subplots(self,data1,data2):
        try:
            fig = plt.figure(figsize=(7, 6))
            fig.suptitle('Training data', fontsize=14)

            plt.subplot(121)
            plt.imshow(data1, cmap=plt.cm.Greys_r)   # data  = img[:, :, 0] &&& cmap = plt.cm.Greys_r
            plt.title('RS image - first band')

            plt.subplot(122)
            plt.imshow(data2, cmap=plt.cm.Spectral) # data = roi && cmap = plt.cm.Spectral
            plt.title('Training Image')
            plt.show()
        except ValueError as e:
            print(e)
            print("Could not plot the data")

    def create_validation_subplots(self,img,class_prediction,roi,roi_v):
        try:
            fig = plt.figure(figsize=(6, 6))
            plt.subplot(221)
            plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
            plt.title('RS_Image - first band')

            plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
            plt.title('RS_Image - first band')

            plt.subplot(222)
            plt.imshow(class_prediction, cmap=plt.cm.Spectral)
            plt.title('Classification result')

            plt.subplot(223)
            plt.imshow(roi, cmap=plt.cm.Spectral)
            plt.title('Training Data')

            plt.subplot(224)
            plt.imshow(roi_v, cmap=plt.cm.Spectral)
            plt.title('Validation Data')
            plt.show()
        except ValueError as e:
            print(e)
            print("Could not create plots for Training/Validation")







