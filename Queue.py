import numpy as np
import torch
from tqdm import tqdm


class Queue(object):
    # The queue model for streaming learning.

    def __init__(self, input_shape, num_classes, capacity, cuda_device=None):
        self.input_shape = input_shape
        self.cuda_device = cuda_device

        self.X = np.zeros((num_classes * capacity, input_shape))
        self.y = np.ones(num_classes * capacity) * np.inf
        self.t = np.zeros(num_classes * capacity)
        self.counter = np.zeros((num_classes))
        self.capacity = capacity
        self.idx_count = 0

    def fit(self, X_train, y_train, i):
        """
        This function will be used to fit the model to training data.
        :param X_train: the training data
        :param y_train: the training labels
        :param i: location of point in stream
        :return:
        """
        X_train = np.reshape(X_train, (1, self.input_shape))

        # check if counter for class is equal to capacity
        if self.counter[y_train] == self.capacity:
            idxs = np.where(self.y[0:self.idx_count] == y_train)[0]  # find indices of current class
            idx = np.argmin(self.t[idxs])  # find the index of the oldest point
            idx = idxs[idx]
            self.X[idx, :] = X_train
            self.y[idx] = y_train
            self.t[idx] = i
        else:
            self.X[self.idx_count, :] = X_train
            self.y[self.idx_count] = y_train
            self.t[self.idx_count] = i
            self.counter[y_train] += 1
            self.idx_count += 1

    def find_dists(self, A, B):
        """
        Given a matrix of points A, return the indices of the closest points in A to B using L2 distance.
        :param A: N x d matrix of points
        :param B: M x d matrix of points for predictions
        :return: indices of closest points in A
        """
        M, d = B.shape
        with torch.no_grad():
            B = torch.reshape(B, (M, 1, d))  # reshaping for broadcasting
            square_sub = torch.mul(A - B, A - B)  # square all elements
            dist = torch.sum(square_sub, dim=2)
            dists, inds = torch.min(dist, dim=1)
        return inds, dists

    def predict(self, X, mb=1000):
        """
        Make predictions on X.
        :param X: data to make predictions on
        :param mb: mini-batch size to speed up predictions
        :return: predicted labels
        """
        num_predict_pts = X.shape[0]
        num_stored_pts = self.idx_count

        # grab necessary data
        X_s = self.X[0:num_stored_pts]
        y_s = self.y[0:num_stored_pts]

        if self.cuda_device is not None:
            X = torch.Tensor(X).cuda()
            X_s = torch.Tensor(X_s).cuda()
            y_s = torch.Tensor(y_s).cuda()
            dists_preds = torch.ones(num_predict_pts).cuda() * float("Inf")
            inds_preds = torch.zeros(num_predict_pts).cuda()
        else:
            X = torch.Tensor(X)
            X_s = torch.Tensor(X_s)
            y_s = torch.Tensor(y_s)
            dists_preds = torch.ones(num_predict_pts) * float("Inf")
            inds_preds = torch.zeros(num_predict_pts)

        # loop over all of the points we want to predict on
        for i in tqdm(range(0, num_predict_pts, mb)):
            start = i
            end = min(i + mb, num_predict_pts)
            X_predict_tmp = X[start:end]
            dp = dists_preds[start:end]
            ip = inds_preds[start:end]

            # loop over all of the stored points
            for j in range(0, num_stored_pts, mb):
                start2 = j
                end2 = min(j + mb, num_stored_pts)
                X_stored_tmp = X_s[start2:end2]
                inds, dists = self.find_dists(X_stored_tmp, X_predict_tmp)

                # overwrite smallest distances and indices
                update_inds = dists < dp
                if self.cuda_device is not None:
                    update_inds = update_inds.type(torch.cuda.ByteTensor)
                else:
                    update_inds = update_inds.type(torch.ByteTensor)
                dp[update_inds] = dists[update_inds]
                if self.cuda_device is not None:
                    idx = inds.type(torch.cuda.FloatTensor)
                else:
                    idx = inds.type(torch.FloatTensor)
                ip[update_inds] = idx[update_inds] + start2

        if self.cuda_device is not None:
            y = y_s[inds_preds.type(torch.cuda.LongTensor)]
        else:
            y = y_s[inds_preds.type(torch.LongTensor)]
        return y.cpu().data.numpy()
