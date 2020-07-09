import sys
sys.path.append("rF:/Work/MaptorReivsed/venv/Model")
from RFModel import RandomForrest

class RandomForrestController():

    rf = RandomForrest()

    def set_RF_trees(self,trees):
        self.rf.Trees = trees

    def rf_classifier(self,img,img_ds,roi,trees):
        return self.rf.rf_classifier(img,img_ds,roi,trees)

    def rf_prediction(self,img,rf2):
        return self.rf.rf_prediction(img,rf2)

    def plot_class_prediction(self,img,class_prediction):
        return self.rf.plot_class_prediction(img,class_prediction)

    def save_result_image(self,img_ds,img,class_prediction,classification_image):
        return self.rf.save_result_image(img_ds,img,class_prediction,classification_image)

    def validation_processing(self,ValidationData,class_prediction,TrainingData,doc,model_path,Image_savePath):
        return self.rf.validation_processing(ValidationData,class_prediction,TrainingData,doc,model_path,Image_savePath)

    def save_model(self, model, file_name):
        return self.rf.save_model(model, file_name)

