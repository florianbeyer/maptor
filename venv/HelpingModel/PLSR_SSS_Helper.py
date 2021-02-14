class PLSR_SSS_Helper(object):
    def __init__(self,train_data,X,mse,msemin,component,y,y_c,y_cv,attribute,importance,
                 prediction,dir_path,reportpath,prediction_map,modelsavepath,img_path,tran_path):

        self.train_data = train_data
        self.X = X
        self.mse = mse
        self.msemin = msemin
        self.component = component
        self.y = y
        self.y_c = y_c
        self.y_cv = y_cv
        self.attribute = attribute
        self.importance = importance
        self.prediction = prediction
        self.dir_path = dir_path
        self.reportpath = reportpath
        self.prediction_map = prediction_map
        self.modelsavepath = modelsavepath
        self.img_path = img_path
        self.tran_path = tran_path
    # #
    #
    # def __str__(self):
    #     return self.dir_path



