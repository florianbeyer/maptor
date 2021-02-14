import sys
# sys.path.append(r"F:\Work\Maptor\venv\Model")
from InputModel import InputModule


class InputController():
    input_C = InputModule()

    def set_training_path(self,path):
        self.input_C.Training_File_Path = path

    def getTrainingPath(self):
        return self.input_C.Training_File_Path


    def set_validation_path(self,path):
        self.input_C.Validation_File_Path = path

    def getValidationPath(self):
        return self.input_C.Validation_File_Path

    def set_img_path(self,path):
        self.input_C.RS_Image_Path = path

    def getImagePath(self):
        return str(self.input_C.RS_Image_Path)


    def set_training_attr(self,attr):
        self.input_C.Trg_Attribute_Selected = attr
        print(self.input_C.Trg_Attribute_Selected)

    def getTrainingAttr(self):
        return self.input_C.Trg_Attribute_Selected

    def set_validation_attr(self,attr):
        self.input_C.Val_Attribute_Selected = attr
        print(self.input_C.Val_Attribute_Selected)

    def getValidationAttr(self):
        return self.input_C.Val_Attribute_Selected

    def FindAttributes(self, filepath):
        return self.input_C.FindAttributes(filepath)
        # driver = ogr.GetDriverByName('ESRI Shapefile')
        # shape_dataset = driver.Open(filepath)
        # shape_layer = shape_dataset.GetLayer()
        # field_names = [field.name for field in shape_layer.schema]
        # return field_names

    def load_image_data(self):
        path = self.input_C.RS_Image_Path
        return self.input_C.loadimagedata(path)

    def load_train_data(self,img_ds,type):
        path = self.input_C.Training_File_Path
        attr = self.input_C.Trg_Attribute_Selected
        return self.input_C.load_training_data(path,attr,img_ds,type)

    def load_validation_data(self,img_ds):
        path = self.input_C.Validation_File_Path
        attr = self.input_C.Val_Attribute_Selected
        return self.input_C.load_validation_data(path,attr,img_ds)

    def create_training_subplots(self, data1, data2):
        self.input_C.create_training_subplots(data1,data2)

    def create_validation_subplots(self,img,class_prediction,roi,roi_v):
        return self.input_C.create_validation_subplots(img,class_prediction,roi,roi_v);


