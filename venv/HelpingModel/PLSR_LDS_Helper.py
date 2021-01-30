class PLSR_LDS_Helper(object):
    def __init__(self,img,train_data,features,train_features,test_features,train_labels,test_labels,mse,component,predictions_test_ds,labels,
                 prediction,importance,X,y,attribute,reportpath,prediction_map,modelsavepath,img_path,tran_path):
        # self.dir_path = dir_path,
        self.img = img
        self.train_data = train_data
        self.features = features
        self.train_features = train_features
        self.test_features = test_features
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.mse = mse
        self.component = component
        self.predictions_test_ds = predictions_test_ds
        self.labels = labels
        self.prediction = prediction
        self.importance = importance
        self.X = X
        self.y = y
        self.attribute = attribute
        self.reportpath = reportpath
        self.prediction_map = prediction_map
        self.modelsavepath = modelsavepath
        self.img_path = img_path
        self.tran_path = tran_path


    # #
    #
    # def __str__(self):
    #     return self.dir_path
