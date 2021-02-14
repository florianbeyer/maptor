import sys
# sys.path.append(r"F:\Work\Maptor\venv\Model")
from RF_NSS_Model import RF_NSS_Model

class RF_NSS_Controller():
    RegMdl = RF_NSS_Model()

    def RF_regressor(self,roi,img,attribute,test_size,est):
        return self.RegMdl.RF_regressor(roi,img,attribute,test_size,est)

    def RF_prediction(self,RFR,img):
        return self.RegMdl.RF_prediction(RFR,img)

    def RF_save_predictionImage(self,prediction,prediction_map,img,img_ds):
        return self.RegMdl.RF_save_predictionImage(prediction,prediction_map,img,img_ds)

    def save_model(self, model, file_name):
        return self.RegMdl.save_model(model, file_name)
