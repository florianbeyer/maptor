import sys,logging,traceback
# sys.path.append(r"F:\Work\Maptor\venv\Model")
from PLSR_LDS_Model import PLSR_LDS_Model


class PLSR_LDS_Controller():
    mdl = PLSR_LDS_Model()

    def ComponentRegressor(self,features,X,y):
        return self.mdl.ComponentRegressor(features,X,y)

    def Regressor(self,suggested_comp,train_features,train_labels):
        return self.mdl.Regressor(suggested_comp,train_features,train_labels)

    def vip(self,plsr):
        return self.mdl.vip(plsr)

    def testpredict(self,plsr,test_features):
        return self.mdl.testpredict(plsr,test_features)

    def finaltraining(self,plsr,features,labels):
        return self.mdl.finaltraining(plsr,features,labels)

    def finalprediction(self,plsr,img_as_array):
        return self.mdl.finalprediction(plsr,img_as_array)

    def saveimage(self, img, prediction_, prediction_map, img_ds):
        return self.mdl.saveimage(img, prediction_, prediction_map, img_ds)

    def savemodel(self, model, file_name):
        return self.mdl.savemodel(model,file_name)