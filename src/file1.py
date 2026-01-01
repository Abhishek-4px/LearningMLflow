import mlflow 
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns

ds=load_wine()
X= ds.data
y= ds.target

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.10,random_state=42)


n_estimators=10
max_depth=5


print("TRACKING URI : ", mlflow.get_tracking_uri())
with mlflow.start_run():
    rf= RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)
    # Collecting Preds
    y_pred=rf.predict(X_test)
    accuracy= accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_params({
        "n_estimators":n_estimators,
        "max_depth": max_depth
    })


    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True, fmt ="d", cmap="Blues", xticklabels=ds.target_names,yticklabels=ds.target_names )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.ylabel("Confusion Matrix")

    plt.savefig("Confusion-matrix.png")

    mlflow.log_artifacts("Confusion-matrix.png")
    mlflow.log_artifacts(__file__)

    print(accuracy)


