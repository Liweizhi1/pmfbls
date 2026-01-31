import numpy as np
import scipy.linalg
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder

class DRBLS(BaseEstimator, ClassifierMixin):
    """
    Double-Relaxation Broad Learning System (DRBLS) for Few-Shot Learning.
    
    Objective Function:
    min J(P, W, Q, S) = ||Y - P||^2 + lambda1 * ||AW - P||^2 
                      + lambda2 * ||WS - Q||^2 + lambda3 * (||W||^2 + ||Q||^2 + ||S||^2)
    
    Parameters:
    -----------
    lambda1 : float
        Coefficient for prediction error (AW - P).
    lambda2 : float
        Coefficient for structure relaxation (WS - Q).
    lambda3 : float
        Regularization coefficient.
    max_iter : int
        Maximum number of iterations for alternating optimization.
    tol : float
        Tolerance for convergence check.
    """
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.1, max_iter=20, tol=1e-4, verbose=False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.W = None
        self.one_hot_encoder = None

    def fit(self, X, y):
        """
        Fit the DRBLS model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data (features extracted from backbone).
        y : array-like of shape (n_samples,)
            Target labels.
        """
        # Convert to numpy if input is torch tensor
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
            
        n_samples, n_features = X.shape
        
        # One-hot encode labels
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
        Y = self.one_hot_encoder.fit_transform(y.reshape(-1, 1))
        n_classes = Y.shape[1]
        
        # Initialize variables
        # A is just X in this context (assuming features are already enhanced or just using raw features)
        # In standard BLS, A would be [Z | H]. Here we treat input X as the feature matrix A.
        A = X
        
        # Initialize P, W, Q, S
        # P initialized to Y
        P = Y.copy()
        
        # W initialized randomly or using ridge regression solution
        # W = (A^T A + lambda I)^-1 A^T Y
        I_d = np.eye(n_features)
        W = np.linalg.solve(A.T @ A + self.lambda3 * I_d, A.T @ Y)
        
        # S and Q initialized randomly
        # Dimension of Q and S: 
        # W is (D, C). Q is (D, K). S is (C, K).
        # Usually K (latent dim) can be set to C or smaller/larger. 
        # Based on the formula ||WS - Q||, if W is (D,C), S must be (C, K), then Q is (D, K).
        # Let's assume K = C for simplicity unless specified otherwise.
        K = n_classes 
        S = np.eye(n_classes) # Identity initialization for structure
        Q = W @ S
        
        # Precompute constant terms for efficiency
        ATA = A.T @ A
        I_d = np.eye(n_features)
        I_k = np.eye(K)
        
        # Iterative Optimization
        for it in range(self.max_iter):
            W_old = W.copy()
            
            # 1. Update P (Label Relaxation)
            # P = (Y + lambda1 * AW) / (1 + lambda1)
            AW = A @ W
            P = (Y + self.lambda1 * AW) / (1 + self.lambda1)
            
            # 2. Update Q (Auxiliary Matrix)
            # Q = (lambda2 * WS) / (lambda2 + lambda3)
            WS = W @ S
            Q = (self.lambda2 * WS) / (self.lambda2 + self.lambda3)
            
            # 3. Update S (Structure Matrix)
            # S = (lambda2 * W^T W + lambda3 * I)^-1 * (lambda2 * W^T Q)
            WTW = W.T @ W
            # Regularization for inversion stability
            S_lhs = self.lambda2 * WTW + self.lambda3 * I_k
            S_rhs = self.lambda2 * W.T @ Q
            S = np.linalg.solve(S_lhs, S_rhs)
            
            # 4. Update W (Weight Matrix) via Sylvester Equation
            # (lambda1 * A^T A + lambda3 * I) W + W (lambda2 * S S^T) = lambda1 * A^T P + lambda2 * Q S^T
            # Form: AX + XB = C
            
            Syl_A = self.lambda1 * ATA + self.lambda3 * I_d
            Syl_B = self.lambda2 * (S @ S.T)
            Syl_C = self.lambda1 * (A.T @ P) + self.lambda2 * (Q @ S.T)
            
            try:
                W = scipy.linalg.solve_sylvester(Syl_A, Syl_B, Syl_C)
            except Exception as e:
                if self.verbose:
                    print(f"Sylvester solver failed at iter {it}: {e}")
                break
                
            # Check convergence
            diff = np.linalg.norm(W - W_old) / (np.linalg.norm(W_old) + 1e-8)
            if self.verbose:
                print(f"Iter {it+1}/{self.max_iter}, diff: {diff:.6f}")
                
            if diff < self.tol:
                if self.verbose:
                    print("Converged.")
                break
                
        self.W = W
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
            
        if self.W is None:
            raise ValueError("Model not fitted yet.")
            
        # Y_pred = XW
        scores = X @ self.W
        
        # Convert scores to labels
        # If we used one-hot, we take argmax
        pred_indices = np.argmax(scores, axis=1)
        
        # Map back to original labels
        return self.one_hot_encoder.categories_[0][pred_indices]

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
            
        scores = X @ self.W
        # Apply softmax to get probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
