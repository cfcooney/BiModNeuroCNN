import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array,check_consistent_length

def weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()

def cross_entropy(y_true, y_pred, eps=1e-15, labels=None):
    """
    A metric that compares the predicted utterance likelihoods and
    the actual utterance identities across all trials for a subject.
    Given utterance log-likelihoods predicted by a model, cross entropy
    measures the average number of bits required to correctly classify
    those utterances. Cross entropy consideres predicted probabilities, not
    simply the most likely class for each trial.
    -- Lower cross entropy indicates better performance.
    :return: loss: float
    """

    y_pred = check_array(y_pred, ensure_2d=False)

    lb = LabelBinarizer()
    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)


    if len(lb.classes_) <= 1:
        raise ValueError("Only 1 or 0 labels have been provided. Please provide correct labels.")

    transformed_labels = lb.transform(y_true)
    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels,
                                       transformed_labels, axis=1)


    y_pred = np.clip(y_pred, eps, 1 - eps) #clipping required to protect against 1 and 0 probabilities

    transformed_labels = check_array(transformed_labels)

    if len(lb.classes_) != y_pred.shape[1]:
        raise ValueError("Ground truth and predictions contain a different number of values!")

    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]

    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

    return  weighted_sum(loss, None,normalize=True)

if __name__ == '__main__':
   
    labels = ['pig','cow','car','bus']
    y_true = [1,2,0,3]
    y_pred = [[.1,.5,.2,.2], [.3,.05,.55,.1], [.5,.0,.0,.5], [.0,.35,0,.65]]

    # labels = ['pig']
    # y_true = [0,1]
    # y_pred = [7.0722e-01, 2.3728e-05, 1.1968e-04, 2.9264e-01]

    print(cross_entropy(y_true, y_pred, eps=1e-15, labels=None))

