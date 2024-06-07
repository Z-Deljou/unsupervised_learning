import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering, Birch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import joblib
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def file_information(df):
    print(df.columns);
    print(df.shape)

    print('\n********************* df.describe ******************** ')
    df.describe()

    # Convert 'Timestamp' column to datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Plot the distribution of transaction time
    sns.distplot(df['Timestamp'], color='b')
    plt.title('Distribution of Transaction Time', fontsize=14)
    plt.show()

    print(df.shape)
    df = df.drop_duplicates(keep='first')
    df.shape


def load_and_preprocess_data(data):
    data = data.drop(['Timestamp'], axis=1)

    # Handle missing values
    df = data.dropna(axis=1, thresh=len(data) * 0.5)
    df = df.fillna(df.mode().iloc[0])

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Scale numerical features
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def preprocess(data, samples_size):
    df = load_and_preprocess_data(data)
    sample_size = int(samples_size * len(df))
    df_sampled = pd.DataFrame(df).sample(n=sample_size, random_state=42)

    return df_sampled


def test_model(model, X_train, X_test):
    train_predictions = model.fit_predict(X_train)
    test_predictions = model.predict(X_test)

    silhouette_score_train = silhouette_score(X_train, train_predictions)
    silhouette_score_test = silhouette_score(X_test, test_predictions)
    db_train = davies_bouldin_score(X_train, train_predictions)
    db_test = davies_bouldin_score(X_test, test_predictions)
    ch_train = calinski_harabasz_score(X_train, train_predictions)
    ch_test = calinski_harabasz_score(X_test, test_predictions)

    return silhouette_score_train, silhouette_score_test, db_train, db_test, ch_train, ch_test


def train_model(model, X_train):
    model.fit(X_train)


# def agglomerative(X_train, X_test):
#     agglomerative_model = AgglomerativeClustering(n_clusters=2)
#     train_model(agglomerative_model, X_train)
#     return evaluate_model(agglomerative_model, X_train, X_test)

def visualize(results):
    methods = list(results.keys())
    evaluation_metrics = list(results[methods[0]].keys())
    n_methods = len(methods)
    n_metrics = len(evaluation_metrics)

    fig, axs = plt.subplots(n_metrics, n_methods, figsize=(15, 10), sharex=True)
    for i, method in enumerate(methods):
        for j, metric in enumerate(evaluation_metrics):
            values = np.array(results[method][metric])
            axs[j, i].bar(["Train", "Test"], np.mean(values, axis=0), yerr=np.std(values, axis=0))
            axs[j, i].set_title(f"{method}\n{metric}")
            axs[j, i].set_ylabel("Score")
    plt.tight_layout()
    plt.show()


# def main():
#     samples_size = float(input("please enter the sample size: "))
#     file_path = '/content/drive/MyDrive/Final_HI-Small_Trans.csv'
#     X = preprocess(file_path, samples_size)
#     # Perform PCA
#     pca = PCA(n_components=0.95)
#     X_pca = pca.fit_transform(X)
#
#     # Split data into train and test sets
#     X_train, X_test = train_test_split(X_pca, test_size=0.2, random_state=42)
#
#     # Initialize dictionary to store evaluation results
#     methods = ["MiniBatchKMeans", "KMeans++", "BIRCH", "Isolation Forest"]
#     evaluation_metrics = ["Silhouette", "Davies Bouldin", "Calinski Harabasz"]
#     results = {method: {metric: [] for metric in evaluation_metrics} for method in methods}
#
#     # Instantiate clustering models
#     models = {
#         "MiniBatchKMeans": MiniBatchKMeans(n_clusters=2, random_state=42),
#         "KMeans++": KMeans(n_clusters=2, init='k-means++', random_state=42),
#         "BIRCH": Birch(n_clusters=2),
#         "Isolation Forest": IsolationForest(random_state=42),
#     }
#
#     # Perform k-fold cross-validation
#     kfold = KFold(n_splits=3, shuffle=True, random_state=42)
#     for method in methods:
#         fold = 1
#         for train_idx, test_idx in kfold.split(X_train):
#             X_train_fold, X_val_fold = X_train[train_idx], X_train[test_idx]
#
#             # Train the model
#             model = models[method]
#             train_model(model, X_train_fold)
#
#             # Evaluate the model
#             silhouette_score_train, silhouette_score_test, \
#                 db_train, db_test, ch_train, ch_test = test_model(model, X_train_fold, X_val_fold)
#
#             results[method]["Silhouette"].append(silhouette_score_test)
#             results[method]["Davies Bouldin"].append(db_test)
#             results[method]["Calinski Harabasz"].append(ch_test)
#
#             # Print evaluation metrics for test set
#             print(f"Method: {method}, Fold: {fold}, Test Set:")
#             print("Silhouette Test:", silhouette_score_test)
#             print("Davies Bouldin Test:", db_test)
#             print("Calinski Harabasz Test:", ch_test)
#
#             # Calculate and print evaluation metrics for train set
#             print("Silhouette Train:", silhouette_score_train)
#             print("Davies Bouldin Train:", db_train)
#             print("Calinski Harabasz Train:", ch_train)
#
#             print("-----------------------")
#             fold += 1
#     # Visualize the evaluation results
#     visualize(results)


def main(file_):
    X_train = None
    X_test = None

    while True:
        print("\nMenu:")
        print("1. Train")
        print("2. Test")
        print("3. Use")
        print("4. All")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            print("\n*************** Train phase **********************")
            # Train the model
            sample_size = float(input("Please enter the sample size: "))
            # file_path = '/content/drive/MyDrive/Final_HI-Small_Trans.csv'
            X = preprocess(file_, sample_size)
            pca = PCA(n_components=0.95)
            X_pca = pca.fit_transform(X)
            X_train, X_test = train_test_split(X_pca, test_size=0.2, random_state=42)
            models = {
                "MiniBatchKMeans": MiniBatchKMeans(n_clusters=2, random_state=42),
                "KMeans++": KMeans(n_clusters=2, init='k-means++', random_state=42),
                # "BIRCH": Birch(n_clusters=2),
                "Isolation Forest": IsolationForest(random_state=42),
            }
            method = "MiniBatchKMeans"
            model = models[method]
            train_model(model, X_pca)
            joblib.dump(pca, 'pca.pkl')  # Save the pca
            joblib.dump(model, 'model.pkl')  # Save the trained model
            print(f'Model is {method}')
            print("\n*************** End train phase **********************")

        elif choice == "2":
            print("\n*************** Test phase **********************")
            # Test the model
            if X_train is None or X_test is None:
                print("Error: Model not trained yet. Please train the model first.")
            else:
                model = joblib.load('model.pkl')
                silhouette_score_train, silhouette_score_test, \
                    db_train, db_test, ch_train, ch_test = test_model(model, X_train, X_test)
                print("Model Evaluation:")
                print("Silhouette Score Test:", silhouette_score_test)
                print("Davies Bouldin Test:", db_test)
                print("Calinski Harabasz Test:", ch_test)

            print("\n*************** End test phase **********************")

        elif choice == "3":
            print("\n*************** Use the model **********************")

            # Use the model

            le_from_account = LabelEncoder()
            le_to_account = LabelEncoder()
            le_payment_format = LabelEncoder()
            le_receiving_currency = LabelEncoder()

            print("\nplease enter inputs:")
            # timestamp = input("Enter Timestamp (Format: 'M/D/YYYY HH:MM:SS AM/PM'): ")
            from_bank = float(input("Enter From Bank (Ex:70): "))
            from_account = input("Enter From Account (Ex:100428660): ")
            to_bank = float(input("Enter To Bank (Ex:1124): "))
            to_account = input("Enter To Account (Ex:800825340): ")
            amount_received = float(input("Enter Amount Received (Ex:389769.39): "))

            currency_dict = {
                "US Dollar": 1,
                "Euro": 2,
                "Yuan": 3,
                "Bitcoin": 4,
                "Rupee": 5,
                "Yen": 6,
                "Australian Dollar": 1,
                "Mexican Peso": 1,
                "UK Pound": 1,
                "Ruble": 1,
            }
            receiving_currency = input("Enter Receiving Currency (Ex: US Dollar): ")
            receiving_currency_encoded = currency_dict.get(receiving_currency, 0)

            amount_paid = float(input("Enter Amount Paid (Ex: 389769.39): "))
            payment_currency = input("Enter Payment Currency (Ex: US Dollar): ")
            payment_currency_encoded = currency_dict.get(payment_currency, 0)

            payment_format = input("Enter Payment Format (Ex:Cheque): ")
            is_laundering = float(input("Enter Is Laundering: "))
            limited_amount = float(input("Enter limited_amount: "))
            exceed_trn_num = float(input("Enter exceed_trn_num: "))
            quick_withdrawal = float(input("Enter Quick Withdrawal: "))
            variz = float(input("Enter variz: "))
            gardeshemali = float(input("Enter Gardeshemali: "))
            Max_Transactions_3Days = float(input("Enter Max_Transactions_3Days: "))

            # Encode categorical features
            from_account_encoded = le_from_account.fit_transform([from_account])[0]
            to_account_encoded = le_to_account.fit_transform([to_account])[0]
            payment_format_encoded = le_payment_format.fit_transform([payment_format])[0]
            payment_format_map = {
                'Cash': 1,
                'Cheque': 2,
                'ACH': 3,
                'Credit Card': 4,
                'Wire': 5,
                'Bitcoin': 6,
                'Reinvestment': 7
            }

            payment_format_encoded = payment_format_map.get(payment_format, 0)
            sample = np.array([[
                # datetime.strptime(timestamp, '%m/%d/%Y %I:%M:%S %p').timestamp(),
                from_bank,
                from_account_encoded,
                to_bank,
                to_account_encoded,
                amount_received,
                receiving_currency_encoded,
                amount_paid,
                payment_currency_encoded,
                payment_format_encoded,
                is_laundering,
                limited_amount,
                exceed_trn_num,
                quick_withdrawal,
                variz,
                gardeshemali,
                Max_Transactions_3Days
            ]])
            pca = joblib.load('pca.pkl')
            model = joblib.load('model.pkl')

            sample_transformed = pca.transform(sample)
            prediction = model.predict(sample_transformed)

            if prediction == 0:
                print("This transaction is not fraudulent.")
            else:
                print("This transaction is fraudulent.")

            print("\n*************** end use the model **********************")

        elif choice == "4":
            print("\n*************** All **********************")
            sample_size = float(input("Please enter the sample size: "))
            X = preprocess(file_, sample_size)
            pca = PCA(n_components=0.95)
            X_pca = pca.fit_transform(X)
            X_train, X_test = train_test_split(X_pca, test_size=0.2, random_state=42)
            models = {
                "MiniBatchKMeans": MiniBatchKMeans(n_clusters=2, random_state=42),
                "KMeans++": KMeans(n_clusters=2, init='k-means++', random_state=42),
                "BIRCH": Birch(n_clusters=2),
                "Isolation Forest": IsolationForest(random_state=42),
            }
            method = "MiniBatchKMeans"
            model = models[method]
            train_model(model, X_pca)
            joblib.dump(model, 'clustering_model.pkl')

            # Test the model
            model = joblib.load('clustering_model.pkl')
            silhouette_score_train, silhouette_score_test, \
                db_train, db_test, ch_train, ch_test = test_model(model, X_train, X_test)
            print("Model Evaluation:")
            print("Silhouette Score Test:", silhouette_score_test)
            print("Davies Bouldin Test:", db_test)
            print("Calinski Harabasz Test:", ch_test)
            model = joblib.load('clustering_model.pkl')
            le_from_account = LabelEncoder()
            le_to_account = LabelEncoder()
            le_payment_format = LabelEncoder()
            le_receiving_currency = LabelEncoder()

            # timestamp = input("Enter Timestamp (Format: 'M/D/YYYY HH:MM:SS AM/PM'): ")
            from_bank = float(input("Enter From Bank: "))
            from_account = input("Enter From Account: ")
            to_bank = float(input("Enter To Bank: "))
            to_account = input("Enter To Account: ")
            amount_received = float(input("Enter Amount Received: "))

            currency_dict = {
                "US Dollar": 1,
                "Euro": 2,
                "Yuan": 3,
                "Bitcoin": 4,
                "Rupee": 5,
                "Yen": 6,
                "Australian Dollar": 1,
                "Mexican Peso": 1,
                "UK Pound": 1,
                "Ruble": 1,
            }
            receiving_currency = input("Enter Receiving Currency: ")
            receiving_currency_encoded = currency_dict.get(receiving_currency, 0)

            amount_paid = float(input("Enter Amount Paid: "))
            payment_currency = input("Enter Payment Currency: ")
            payment_currency_encoded = currency_dict.get(payment_currency, 0)

            payment_format = input("Enter Payment Format: ")
            is_laundering = float(input("Enter Is Laundering: "))
            limited_amount = float(input("Enter limited_amount: "))
            exceed_trn_num = float(input("Enter exceed_trn_num: "))
            quick_withdrawal = float(input("Enter Quick Withdrawal: "))
            variz = float(input("Enter variz: "))
            gardeshemali = float(input("Enter Gardeshemali: "))
            Max_Transactions_3Days = float(input("Enter Max_Transactions_3Days: "))
            from_account_encoded = le_from_account.fit_transform([from_account])[0]
            to_account_encoded = le_to_account.fit_transform([to_account])[0]
            payment_format_encoded = le_payment_format.fit_transform([payment_format])[0]

            payment_format_map = {
                'Cash': 1,
                'Cheque': 2,
                'ACH': 3,
                'Credit Card': 4,
                'Wire': 5,
                'Bitcoin': 6,
                'Reinvestment': 7
            }
            payment_format_encoded = payment_format_map.get(payment_format, 0)
            sample = np.array([[
                # datetime.strptime(timestamp, '%m/%d/%Y %I:%M:%S %p').timestamp(),
                from_bank,
                from_account_encoded,
                to_bank,
                to_account_encoded,
                amount_received,
                receiving_currency_encoded,
                amount_paid,
                payment_currency_encoded,
                payment_format_encoded,
                is_laundering,
                limited_amount,
                exceed_trn_num,
                quick_withdrawal,
                variz,
                gardeshemali,
                Max_Transactions_3Days
            ]])

            sample_transformed = pca.transform(sample)
            prediction = model.predict(sample_transformed)
            if prediction == 0:
                print("This transaction is not fraudulent.")
            else:
                print("This transaction is fraudulent.")

            print("\n*************** End pahse **********************")

        elif choice == "5":
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please enter a valid option.")


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    df = pd.read_csv("Final_HI-Small_Trans.csv")
    file_information(df)

    main(df)
