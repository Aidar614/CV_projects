class AccuracyTracker:
    def __init__(self, vocab):
        self.vocab = vocab
        self.last_accuracies = []  # the last 10 values
        self.epoch_accuracies = []  #accuracy for the entire epoch
        self.reset_batch_stats()
    
    def reset_batch_stats(self):
        self.batch_correct = 0
        self.batch_total = 0
    
    def calculate_accuracy(self, output, targets):
        preds = output.argmax(dim=-1)  # [seq_len, batch_size]
        
        pad_token = self.vocab.stoi["<PAD>"]
        sos_token = self.vocab.stoi["<SOS>"]
        eos_token = self.vocab.stoi["<EOS>"]
        mask = (targets != pad_token) & (targets != sos_token) & (targets != eos_token)
        
        correct = (preds[mask] == targets[mask]).sum().item()
        total = mask.sum().item()
        
        if total > 0:
            accuracy = correct / total
            self._update_accuracies(accuracy)
            self.batch_correct += correct
            self.batch_total += total
            return accuracy
        return 0.0
    
    def _update_accuracies(self, accuracy):
        if len(self.last_accuracies) >= 10:
            self.last_accuracies.pop(0)
        self.last_accuracies.append(accuracy)
    
    def get_moving_avg(self):
        return sum(self.last_accuracies) / len(self.last_accuracies) if self.last_accuracies else 0.0
    
    def epoch_end(self):
        epoch_accuracy = self.batch_correct / self.batch_total if self.batch_total > 0 else 0.0
        self.epoch_accuracies.append(epoch_accuracy)
        self.reset_batch_stats()
        return epoch_accuracy