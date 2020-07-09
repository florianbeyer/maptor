import sys
sys.path.append(r"F:/Work/MaptorReivsed/venv/Model")
from RegressionModel import RegressionModel

class RegressionController():
    RegMdl = RegressionModel()

    def RF_regressor(self,roi,img,attribute):
        return self.RegMdl.RF_regressor(roi,img,attribute)

    def RF_prediction(self,RFR,img):
        return self.RegMdl.RF_prediction(RFR,img)

    def RF_save_PredictionImage(self,prediction,prediction_map,img,img_ds):
        return self.RegMdl.save_predictionImage(prediction,prediction_map,img,img_ds)

    def save_model(self, model, file_name):
        return self.RegMdl.save_model(model, file_name)
