import torch
import numpy as np


def calculate_metrics(output, target, agg=True):
    """
    output: list of tensor from fc, [(B, H)]
    target: list of label [B]
    """
    from sklearn.metrics import accuracy_score

    predict = [torch.argmax(i, dim=1).cpu() for i in output]
    target = [i.cpu() for i in target]

    accs = [accuracy_score(y_true=t.numpy(), y_pred=p.numpy()) for t, p in zip(target, predict)]
    if agg:
      avg_acc = sum(accs) / len(accs)
      min_acc = min(accs)
      max_acc = max(accs)

      return avg_acc, min_acc, max_acc
    else:
      return np.array(accs)



def calculate_tag_metrics(output, target, threshold=0.5, agg=True):
    """
    output: [B, n_classes]
    target: [B, n_classes]
    """
    from sklearn.metrics import accuracy_score
    predict = np.array(output.cpu() > threshold, dtype=float)
    predict = np.array_split(predict, 40, 1)

    target = torch.unbind(target, 1)
    target = [i.cpu().numpy() for i in target]

    accs = [accuracy_score(y_true=t, y_pred=np.squeeze(p)) for t, p in zip(target, predict)]
    if agg:
      avg_acc = sum(accs) / len(accs)
      min_acc = min(accs)
      max_acc = max(accs)

      return avg_acc, min_acc, max_acc
    else:
      return np.array(accs)

