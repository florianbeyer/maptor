# try:
import sys,logging,traceback
sys.path.append(r"F:\Work\Maptor\venv\Model")
from PLSR_SSS_Model import PLSR_SSS_Model
# except Exception as e:
# logging.error("Exception occurred", exc_info=True)
# print('Can not import files:' + str(e))
# input("Press Enter to exit!")
# sys.exit(0)

class PLSR_SSS_Controller():
    PLSR = PLSR_SSS_Model()

    log_format = "%(asctime)s::%(levelname)s::%(name)s::" \
                 "%(filename)s::%(lineno)d::%(message)s"

    logging.basicConfig(filename='app2.log', filemode='w', level=logging.DEBUG, format=log_format)

    def PLSR_Regressor(self,features,X,y):
        try:
            return self.PLSR.PLSR_Regressor(features,X,y)
        except Exception as e:
            logging.debug("Exception occurred", exc_info=True)
            logging.error(traceback.print_exc())
            logging.info((traceback.print_exc()))


    def PLSR_vip(self,model):
        return self.PLSR.PLSR_vip(model)

    def PLSR_Predict(self,doc, score_cv, importance, img, y_c, y, y_cv,labels, pls_opt):
        return self.PLSR.PLSR_Predict(doc, score_cv, importance, img, y_c, y, y_cv,labels, pls_opt)

    def PLSR_SaveImage(self, prediction_map,img, img_ds, prediction_):
        return self.PLSR.PLSR_SaveImage(prediction_map,img,img_ds,prediction_)

    def PLSR_SaveModel(self, model, file_name):
        return self.PLSR.PLSR_SaveModel(model, file_name)