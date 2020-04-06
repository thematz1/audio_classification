"""Build an sklearn SVC model for classification."""
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import plot_learning_curve as plot


def main(save='model.pkl'):
    """Build the classifier model"""
    data = pd.read_pickle('../../full_dataframe.pkl')
    X = data[['time', 'chromagram', 'spectral_centroids', 'tempo',
              'contrast', 'rms', 'tonnetz', 'mfcc_val']]
    y = data['target']
    X = X.reset_index()
    X = X.drop(columns='file')
    y = y.reset_index()
    y = y.drop(columns='file')
    # print(X.head())  # Create assertion unit test
    # print(y.head())  # Create assertion unit test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1)
    model = DecisionTreeClassifier(max_depth=25, min_samples_split=80)
    scores = cross_val_score(model, X_test, y_test, cv=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Model accuracy is {:.2f}'.format(acc))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    l_curve = plot.plot_learning_curve(model,
                                       "Tree Model Learning Curve",
                                       X_train, y_train)
    l_curve.show()
    with open(save, 'wb') as f:
        pickle.dump(model, f, protocol=-1)
    return model

if __name__ == "__main__":
    main()
