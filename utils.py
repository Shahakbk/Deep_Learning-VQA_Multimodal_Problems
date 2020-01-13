import os
from config import qa_dir


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
