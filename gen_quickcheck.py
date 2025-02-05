import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import scipy.sparse
import joblib  # モデル保存用

# 定数定義
MODEL_PATH = "model.xgb"
VECTORIZER_PATH = "vectorizer.pkl"
TEMPLATES = {
    1: "{A}の場合、{B}を実行し、{C}を確認する",
    2: "{A}を確認し、{B}を実施して{C}を達成する"
}

def load_data(file_path):
    """Excel/CSVファイルからデータを読み込み"""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")

def extract_features(df, vectorizer=None, mode='train'):
    """特徴量の抽出とベクトル化"""
    # A+B+Cを結合して1つのテキスト特徴量に
    combined_texts = df['A'] + " " + df['B'] + " " + df['C']
    
    if mode == 'train':
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(combined_texts)
        return features, vectorizer
    else:
        features = vectorizer.transform(combined_texts)
        return features

def train_model(X, y, save_model=True):
    """モデルの訓練と保存"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    # 精度検証
    y_pred = model.predict(X_test)
    print(f"Validation Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    if save_model:
        joblib.dump(model, MODEL_PATH)
    return model

def generate_instructions(model, vectorizer, new_data):
    """新しいデータから手順を生成"""
    # 特徴量抽出
    features = extract_features(new_data, vectorizer, mode='predict')
    
    # テンプレート予測
    template_ids = model.predict(features)
    
    # テンプレート適用
    results = []
    for i, row in new_data.iterrows():
        template = TEMPLATES.get(template_ids[i], "{A}に基づき{B}を実施")
        results.append(template.format(
            A=row['A'], 
            B=row['B'], 
            C=row['C']
        ))
    return results

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='自動化手順生成ツール')
    parser.add_argument('input_file', help='入力ファイルパス (.xlsx or .csv)')
    parser.add_argument('--save-model', action='store_true', help='モデルを保存する')
    args = parser.parse_args()

    try:
        # データ読み込み
        df = load_data(args.input_file)
        
        # 特徴量抽出（訓練時）
        X, vectorizer = extract_features(df, mode='train')
        y = df['D'].map(lambda x: 1 if "確認" in x else 2)  # 簡易ラベル変換
        
        # モデル訓練
        model = train_model(X, y, args.save_model)
        
        # 新しいデータで予測（デモ用に同じデータを使用）
        print("\n生成された手順:")
        instructions = generate_instructions(model, vectorizer, df)
        for instr in instructions:
            print(f"- {instr}")
            
        # モデルとベクトルライザを保存
        if args.save_model:
            joblib.dump(vectorizer, VECTORIZER_PATH)
            print(f"\nモデルとベクトルライザを保存しました: {MODEL_PATH}, {VECTORIZER_PATH}")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()