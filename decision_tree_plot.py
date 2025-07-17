# Passo 1: Importar Bibliotecas Necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from graphviz import Source

# Passo 2: Carregar e Preparar o Dataset
# 2.1 - Carregar o Dataset
file_path = "fetal_health.csv"  # Substitua pelo caminho do seu arquivo CSV
data = pd.read_csv(file_path)
print("Dados originais:")
print(data.head())
# 2.1 - Preparar o Dataset

# ----- Uma necessidade é Converter os valores categóricos para numéricos
# Identifica colunas categóricas
categorical_columns = data.select_dtypes(include=['object']).columns
# Dicionário para armazenar os LabelEncoders de cada coluna categórica
label_encoders = {}
# Aplica LabelEncoder apenas nas colunas categóricas
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

print("\nDados após a preparação:")
print(data.head())

# ----- Outra necessidade é definir as colunas que serão usadas na modelagem, recursos (X), e o alvo, alvo (y).
X = data.drop(columns=["Saude Fetal"])  # Exclui a coluna alvo e outras irrelevantes
y = data["Saude Fetal"]  # Coluna alvo

# Passo 3: Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Treinar o Modelo de Árvore de Decisão
clf = DecisionTreeClassifier(criterion="gini",max_depth=6, min_samples_split=12, random_state=42)
clf.fit(X_train, y_train)

# Passo 5: Fazer Previsões e Avaliar o Modelo
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {accuracy:.2f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Passo 6: Exportar e Visualizar a Árvore de Decisão usando Graphviz
dot_data = export_graphviz(
    clf,
    feature_names=X.columns,
    class_names=["Normal", "Suspeito", "Patologico"],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = Source(dot_data)
graph.render("Fetal Health Arvore de decisão", format="png")  # Salva o arquivo como PNG
graph.view()  # Abre o arquivo gerado no visualizador padrão
