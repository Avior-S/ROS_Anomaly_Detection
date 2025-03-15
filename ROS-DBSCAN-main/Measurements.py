import numpy as np
import sklearn.metrics
from Style import Configure as Conf

class Measurements:

  @staticmethod
  def calculate_TP(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    TP = np.sum(np.logical_and(pred_labels == Conf.POSITIVE_LABEL, true_labels == Conf.POSITIVE_LABEL))
    return TP

  @staticmethod
  def calculate_TN(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    TN = np.sum(np.logical_and(pred_labels == Conf.NEGATIVE_LABEL, true_labels == Conf.NEGATIVE_LABEL))
    return TN

  @staticmethod
  def calculate_FP(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    FP = np.sum(np.logical_and(pred_labels == Conf.POSITIVE_LABEL, true_labels == Conf.NEGATIVE_LABEL))
    return FP

  @staticmethod
  def calculate_FN(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    FN = np.sum(np.logical_and(pred_labels == Conf.NEGATIVE_LABEL, true_labels == Conf.POSITIVE_LABEL))
    return FN

  @staticmethod
  def calculate_FPR(true_labels, pred_labels):
    fp = Measurements.calculate_FP(true_labels, pred_labels)
    tn = Measurements.calculate_TN(true_labels, pred_labels)
    return float(fp) / (fp + tn)

  @staticmethod
  def calculate_TPR(true_labels, pred_labels): # sensitivity
    tp = Measurements.calculate_TP(true_labels, pred_labels)
    fn = Measurements.calculate_FN(true_labels, pred_labels)
    return float(tp) / (tp + fn)

  @staticmethod
  def calculate_TNR(true_labels, pred_labels): # specificity
    tn = Measurements.calculate_TN(true_labels, pred_labels)
    fp = Measurements.calculate_FP(true_labels, pred_labels)
    return float(tn) / (tn + fp)

  @staticmethod
  def calculate_PPV(true_labels, pred_labels): # positive predictive value
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    from sklearn.metrics import precision_score
    tpr = precision_score(true_labels, pred_labels)
    return tpr #tp / (tp + fp)

  @staticmethod
  def accuracy_score(y_true, y_pred):
    return sklearn.metrics.accuracy_score(y_true, y_pred)

  @staticmethod
  def longest_sequence(y_pred, value):
    ret = 0
    for i in range(len(y_pred)):
      count = 0
      if y_pred[i] != value:
        continue
      count += 1
      for j in range(i+1, len(y_pred)):
        if y_pred[j] != value:
          break
        count += 1
      ret = max(ret, count)
    return ret

  @staticmethod
  def longest_sequence_old(y_pred, value):
      #y_pred = [(x+1)%2 for x in y_pred]
      #value = (value+1)%2
      seq = []
      ret = 0
      for i in range(len(y_pred)):
          count = 0
          if y_pred[i] != value:
              continue
          count += 1
          for j in range(i + 1, len(y_pred)):
              if y_pred[j] != value:
                  seq += [count]
                  break
              count += 1
          ret = max(ret, count)
      seq = sorted(seq)
      seq.reverse()
      if len(seq) == 0:
          return 0
      elif len(seq) == 1 :
          return seq[0]
      ret = seq[0]*9 + seq[1]
      print("####0")
      print(seq)
      print(ret)
      print("####")
      return ret

  @staticmethod
  def count_warnings(y_pred, value, threshold):
    warnings = 0.0
    for i in range(len(y_pred) - threshold + 1):
      tmp = y_pred[i:i+threshold]
      if len([x for x in tmp if x==value]) == threshold:
        warnings+=1
    return warnings

  @staticmethod
  def draw_chart(title, path, y_pred, label):
    def is_equal(array, label):
      for i in array:
        if i != label:
          return False
      return True
    def  count_len(array, label):
      count = 0
      for j in array:
        if j != label:
          break
        count+=1
      return count
    def len_count(array, label):
      count = 0
      j = len(array) - 1
      while j >= 0:
        if array[j] != label:
          break
        count+=1
        j= j-1
      return count
    # import matplotlib.pyplot as plt
    # mapping = {}
    # # longest = Measurements.longest_sequence(y_pred,label)
    # for i in range(len(y_pred)):
    #   mapping[i] = len_count(y_pred[:i+1], label)
    #     # mapping[i] = count_len(y_pred[i:], label)
    # plt.title(title)
    # # plt.axis([0, 6, 0, 20])
    # # plt.
    # axes = plt.gca()
    # axes.set_ylim([0,40])
    # plt.xlabel('time')
    # plt.ylabel('negative sequence')
    # plt.plot(mapping.keys(), mapping.values(), 'ro')
    # plt.savefig(path)
    # plt.clf()
    # plt.cla()


#if __name__ == '__main__':
  #pos_label, neg_label = 1, 0
  #measures = Measurements(pos_label, neg_label)
  #y_true = [1,1,1,1]
  #y_pred = [0,1,0,1]
  #measures.accuracy_score(y_true, y_pred)

