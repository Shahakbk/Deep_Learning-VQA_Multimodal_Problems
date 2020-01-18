import os
import torchvision.transforms as transforms
import torch.cuda
from config import paths

qa_dir = paths["qa_dir"]


def get_path(train=False, val=False, test=False, question=False, answer=False):
    """
        Return the path to the specified data.
    """
    # Make sure to receive only one type of set (train/val/test) and type (question/answer)
    assert train + val + test == 1
    assert question + answer == 1

    # Get the correct directory
    if train:
        _set = 'train2014'
    elif val:
        _set = 'val2014'
    else:
        _set = 'test2015'

    # Get the correct type (Q/A) format
    json = None
    if question:
        json = 'v2_OpenEnded_mscoco_' + _set + '_questions.json'
    else:
        if test:
            # Just load validation data in the test=answer=True case, will be ignored anyway
            _set = 'val2014'
        json = 'v2_mscoco_' + _set + '_annotations.json'

    return os.path.join(qa_dir, json)


def get_transform(target_size, central_fraction=1.0):
    """
        Create a transformation based on the scaling, crop size & target size and normalization
    """
    return transforms.Compose([transforms.Scale(int(target_size / central_fraction)),
                               transforms.CenterCrop(target_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                               ])


def check_cuda():
    """
        Checking for CUDA availability
    """

    if torch.cuda.is_available():
        print('CUDA is available, working on GPU.')
        device = torch.device('cuda:0')
    else:
        print('CUDA is NOT available, working on CPU.')
        device = torch.device('cpu')
    return device


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    '''
    Acc needs to be averaged over all 10 choose 9 subsets of human answers.
    While we could just use a loop, surely this can be done more efficiently (and indeed, it can).
    There are two cases for the 1 chosen answer to be discarded:
    (1) the discarded answer is not the predicted answer => acc stays the same
    (2) the discarded answer is the predicted answer => we have to subtract 1 from the number of agreeing answers

    There are (10 - num_agreeing_answers) of case 1 and num_agreeing_answers of case 2, thus
    acc = ((10 - agreeing) * min( agreeing      / 3, 1)
           +     agreeing  * min((agreeing - 1) / 3, 1)) / 10

    Let's do some more simplification:
    if num_agreeing_answers == 0:
        acc = 0  since the case 1 min term becomes 0 and case 2 weighting term is 0
    if num_agreeing_answers >= 4:
        acc = 1  since the min term in both cases is always 1
    The only cases left are for 1, 2, and 3 agreeing answers.
    In all of those cases, (agreeing - 1) / 3  <  agreeing / 3  <=  1, so we can get rid of all the mins.
    By moving num_agreeing_answers from both cases outside the sum we get:
        acc = agreeing * ((10 - agreeing) + (agreeing - 1)) / 3 / 10
    which we can simplify to:
        acc = agreeing * 0.3
    Finally, we can combine all cases together with:
        min(agreeing * 0.3, 1)
    '''
    return (agreeing * 0.3).clamp(max=1)


class Tracker:
    """ Keep track of results over time, while having access to monitors to display information about them. """
    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}

    class ListStorage:
        """ Storage of data points that updates the given monitors """
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
