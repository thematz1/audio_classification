"""Build an sklearn SVC model for classification."""
from src.audio_classifier.create_dataframe_v1 import create_dataframe_context
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    """Quick svm.SVC model."""
    with create_dataframe_context('/Users/mathewzaharopoulos/\
                                  dev/audio_classification/samples_data',
                                  save='example.pkl') as data:
        df = data.reset_index()
    X = df[['chromagram', 'spectral_centroids', 'tempo', 'contrast',
            'rms', 'tonnetz', 'mfcc_val']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Model accuracy is {:.2f}'.format(acc))


if __name__ == "__main__":
    main()
