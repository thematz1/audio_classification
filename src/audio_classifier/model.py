"""Build an sklearn SVC model for classification."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import src.audio_classifier.plot_learning_curve as plot


def main():
    data = pd.read_pickle('full_dataframe.pkl')
    X = data[['time', 'chromagram', 'spectral_centroids', 'tempo',
                'contrast', 'rms', 'tonnetz', 'mfcc_val']]
    y = data['target']
    X = X.reset_index()
    X = X.drop(columns='file')
    y = y.reset_index()
    y = y.drop(columns='file')
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1)
    model = DecisionTreeClassifier()
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
    


if __name__ == "__main__":
    main()
