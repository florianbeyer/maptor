import sys
# sys.path.append(r"F:\Work\Maptor\venv\Model")
from RFR_SSS_Model import RFR_SSS_Model

class RFR_SSS_Controller():
    RegMdl = RFR_SSS_Model()

    def RFR_Regressor(self,X,y,est,features,labels):
        return self.RegMdl.RFR_Regressor(X,y,est,features,labels)

    def RFR_prediction(self,RFR,img_as_array):
         return self.RegMdl.RFR_prediction(RFR,img_as_array)
    #
    def RF_save_PredictionImage(self,prediction_,prediction_map,img,img_ds):
        return self.RegMdl.RF_save_PredictionImage(prediction_,prediction_map,img,img_ds)

    def save_model(self, model, file_name):
        return self.RegMdl.save_model(model, file_name)
