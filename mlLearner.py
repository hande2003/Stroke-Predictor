class my_machine_learner:
    def __init__(self, model,train_x, test_x, train_y, test_y, en_scale, scale):
        self.model = model
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.en_scale = en_scale
        self.scale = scale
        
    def my_model_selection (self):
        from sklearn.pipeline import make_pipeline
        
        self.pipe = make_pipeline(self.en_scale, self.scale, self.model)
        self.pipe.fit(self.train_x, self.train_y)
        self.prediction = self.pipe.predict(self.test_x)
        return self.prediction, self.pipe
            
    def confusion_matrix (self, my_model_selection):
        from sklearn.metrics import confusion_matrix

        self.prediction, self.pipe = my_model_selection
        self.confusion_matrix = confusion_matrix(self.test_y, self.prediction)

        return self.confusion_matrix
    
    def accuracy_score (self, my_model_selection):
        from sklearn.metrics import accuracy_score

        self.prediction, self.pipe = my_model_selection
        self.accuracy_score = accuracy_score(self.test_y, self.prediction)

        return self.accuracy_score
    
    def recall_score (self, my_model_selection):
        from sklearn.metrics import recall_score

        self.prediction, self.pipe = my_model_selection
        self.recall_score = recall_score(self.test_y, self.prediction)

        return self.recall_score
    
    def log_loss (self, my_model_selection):
        from sklearn.metrics import log_loss

        self.prediction, self.pipe = my_model_selection
        self.log_loss = log_loss(self.test_y, self.prediction)

        return self.log_loss
    
    def classification_report (self, my_model_selection):
        from sklearn.metrics import classification_report

        self.prediction, self.pipe = my_model_selection
        self.classification_report = classification_report(self.test_y, self.prediction)

        return self.classification_report
    
    def precision_score (self, my_model_selection):
        from sklearn.metrics import precision_score

        self.prediction, self.pipe = my_model_selection
        self.precision_score = precision_score(self.test_y, self.prediction)

        return self.precision_score
    
    def f1_score (self, my_model_selection):
        from sklearn.metrics import f1_score

        self.prediction, self.pipe = my_model_selection
        self.f1_score = f1_score(self.test_y, self.prediction)

        return self.f1_score