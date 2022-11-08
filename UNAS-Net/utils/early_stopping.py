class EarlyStopping:


    """
    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_loss = None
        self.best_valid_score = None
        self.best_tr_score = None
        self.best_epoch = None
        self.worst_counter = 0
    
    def stopTraining(self, epoch, valid_loss, valid_score, tr_score):

        if epoch == 0:
            self.best_loss = valid_loss
            self.best_valid_score = valid_score
            self.best_tr_score = tr_score
            self.best_epoch = epoch

        if self.best_loss < valid_loss:
            self.worst_counter = self.worst_counter + 1
        else:
            self.worst_counter = 0
            self.best_loss = valid_loss
        
        if valid_score > self.best_valid_score:
            self.best_valid_score = valid_score
            self.best_epoch = epoch

        if tr_score > self.best_tr_score:
            self.best_tr_score = tr_score

        if self.worst_counter >= self.patience:
            self.stopped_epoch = epoch
            return True
        else:
            return False