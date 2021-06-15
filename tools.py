import torch

def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    batch_size = target.size(0)

    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
      curr_ind = pred[:,curr_k]
      num_eq = torch.eq(curr_ind, target).sum()
      acc = num_eq/len(output)
      res_total += acc
    return res_total*100

def fooling_rate(output_original, output_attacked):
    class_original = torch.argmax(output_original, dim=1)
    acc = accuracy_topk(output_attacked.data, class_original.data, k=1)
    fool = 100 - acc
    return fool

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def kl_avg(logits1, logits2):
  sf = torch.nn.Softmax(dim=1)
  pmf1 = sf(logits1)
  pmf2 = torch.log(sf(logits2)) # log for kl div function
  kl_mean = torch.nn.functional.kl_div(pmf2, pmf1, reduction='batchmean')
  return kl_mean
