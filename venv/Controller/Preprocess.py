import sys
sys.path.append(r'F:\Work\MaptorReivsed\venv\Model')
from InputModel import InputModule
from ClassificationModel import RandomForrest
import matplotlib.pyplot as plt


class Prep():
    test = InputModule()
    rf_test = RandomForrest()
    LoadingImages=""
    img_ds=""
    img=""
    TrainingData=""
    ValidatiionData=""
    class_prediction=""
    rf2=""

    def set_img_ds(self):
        self.img_ds = self.LoadingImages[0]

    def set_img(self):
        self.img = self.LoadingImages[1]


    def LoadImages(self,test):
        self.LoadingImages = test.loadimagedata(test.RS_Image_Path)
        self.set_img()
        self.set_img_ds()

    def get_training_data(self,test,img_ds):
        self.test.load_training_data("F:\Work\Data\shape\holl_cal.shp","Code",img_ds)

    def get_validation_data(self,test):
        ValidationData = self.test.load_validation_data("F:\Work\Data\shape\holl_val.shp", "Code", self.img_ds)

    def get_subplots(self,test):
        test.create_subplots("Training Data", self.img[:, :, 0], 'RS image - first band', plt.cm.Greys_r, self.TrainingData,
                             "Training Image", plt.cm.Spectral)

    def RF_classification(self):
        self.rf_test.rf_classifier(self.img, self.img_ds, self.TrainingData)

    def RF_prediction(self):
        self.rf_test.rf_prediction(self.img, self.rf2)

    def RF_plot_class_Prediction(self,rf_test):
        rf_test.plot_class_prediction(self.img, self.class_prediction)

    def RF_save_result_img(self,rf_test,img_ds, img, class_prediction):
        rf_test.save_result_image(img_ds, img, class_prediction, r"F:\Work\Data\test1.tif")

    def RF_validation_subplots(self,test,img, class_prediction, TrainingData, ValidationData):
        test.create_validation_subplots(img, class_prediction, TrainingData, ValidationData)

    def RF_validation_process(self,rf_test,ValidationData, class_prediction, TrainingData):
        rf_test.validation_processing(ValidationData, class_prediction, TrainingData)



