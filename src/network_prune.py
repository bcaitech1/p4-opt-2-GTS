import optuna


class Pruner():
    temp_best = 0
    best_score = 0
    temp_train_info = []
    best_train_info = []
    def __init__(self, prune_type, endure_rate, train_epoch, ):
        self.prune_type = prune_type
        self.temp_best = 0
        self.best_score = 0
        self.trial = None
        self.endure_rate = endure_rate
        self.train_epoch = train_epoch
        self.endure_count = 0
        self.pruned_architecture = set()

    def init_train(self):
        self.temp_best = 0
        self.temp_train_info = []
        self.endure_count = 0

    def add_train_info(self, trial, epoch, test_score):
        if self.prune_type == 0:
            return
        elif self.prune_type == 1:
            self.trial = trial
            self.trial.report(test_score, epoch)
        elif self.prune_type == 2:
            self.temp_train_info.append(test_score)
            self.temp_best = max(self.temp_best, test_score)

            if (epoch + 1) == self.train_epoch:
                if self.temp_best > self.best_score:
                    self.best_score = self.temp_best
                    self.best_train_info = self.temp_train_info
    
    def train_prune(self):
        if self.prune_type == 0:
            return False
        elif self.prune_type == 1:
            return self.trial.should_prune()
        elif self.prune_type == 2:
            if len(self.best_train_info) >= len(self.temp_train_info):
                index = len(self.temp_train_info) - 1
                if self.temp_train_info[index] < self.best_train_info[index]:
                    self.endure_count += 1
                    if self.endure_count >= self.endure_rate:
                        return True
                else:
                    self.endure_count = 0
        
        return False

    def get_architecture_list(self, backbone):
        arc_list = []
        for arc in backbone:
            arc_list.append(arc[1])
        arc_list = tuple(arc_list)
        return arc_list         

    def add_pruned_backbone(self, backbone):
        arc_list = self.get_architecture_list(backbone)

        self.pruned_architecture.add(arc_list)

    def architect_prune(self, backbone):
        arc_list = self.get_architecture_list(backbone)

        return arc_list in self.pruned_architecture