import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath: str = "train.csv"
    test_size: float = 0.25
    random_state: int = 6
    n_estimators: int = 150
    learning_rate: float = 0.2382548454734279
    max_depth: int = 1

hp = Hyperparameters()

def create_dataframe(filepath):
    return pd.read_csv(filepath)

def preprocess_dataset(df):
    cols_to_drop = ["id", "s56", "s57","s59"]
    # cols_to_drop = ["id", "s56", "s57","s59", "s53", "s54", "s55"]
    df.drop(columns=cols_to_drop, axis=1, inplace=True)
    df.replace({
        "gender":{"M":1,"F":0},
        "s11":{"Y":1,"N":0},
        "s12":{"Y":1,"N":0},
        "s52":{"l":1,"o":0},
        "s53":{"  ":"a"," ":"b"},
        "s69":{"x":1,"0":0, "~1":1,"C`":0},
        "s70":{"op: D":"D","op: B":"B", "op: C":"C","op: A":"A"},
        }, inplace=True)
    df.s55.fillna("Unknown", inplace=True)
    df.s54.fillna("Unknown", inplace=True)
    cat_cols = ["s16","s17","s18", "s53", "s58", "s70", "s71","s54", "s55"]
    df_ohe = pd.get_dummies(data=df, columns=cat_cols)
    return df_ohe

def split_dataset(df, test_size, random_state):
    X = df.drop(["label"], axis=1)
    y = df["label"]
    return train_test_split(X, y, test_size = test_size, random_state = random_state)


def train_model(X_train, y_train, n_estimators, learning_rate, max_depth, random_state):
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    return gbc.fit(X_train, y_train)

def run_wf(hp: Hyperparameters) -> GradientBoostingClassifier:
    df = create_dataframe(hp.filepath)
    df = preprocess_dataset(df=df)
    X_train, X_test, y_train, y_test = split_dataset(df=df, test_size=hp.test_size, random_state=hp.random_state)
    return train_model(X_train=X_train, y_train=y_train, n_estimators=hp.n_estimators, learning_rate=hp.learning_rate, max_depth=hp.max_depth, random_state=hp.random_state)

if __name__=="__main__":
    run_wf(hp=hp)