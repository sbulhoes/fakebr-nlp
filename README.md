# Classificação de Fake News com NLP (corpus Fake.br)

Atividade Prática de NLP – Uninter  
**RU: 4957311**

## Objetivo
Criar um modelo que classifica notícias em português como verdadeiras (**REAL**) ou falsas (**FAKE**) usando o corpus [Fake.br](https://github.com/roneysco/Fake.br-Corpus) (arquivo `pre-processed.csv`, 7.200 textos: 3.600 REAL e 3.600 FAKE), com acurácia mínima de 85%.

## Estrutura do repositório
| Arquivo | Descrição |
|---|---|
| `fakebr_nlp_4957311.ipynb` | Notebook com todo o pipeline |
| `pre-processed.csv` | Corpus Fake.br pré-processado |
| `requirements.txt` | Dependências Python |

## Etapas do pipeline
1. **Aquisição e estruturação:** leitura do CSV em UTF-8 com Pandas e padronização dos rótulos (`fake` → FAKE, `true` → REAL).
2. **Pré-processamento (NLTK):** minúsculas, remoção de acentos, tokenização (`RegexpTokenizer`), remoção de stopwords em português e de tokens de uma letra.
3. **Balanceamento:** os textos REAL são muito mais longos que os FAKE. Eles são truncados em um limite fixo de tokens, escolhido para que o total de palavras dos dois rótulos fique praticamente igual. Isso evita que o modelo classifique pelo tamanho do texto e não pelo conteúdo.
4. **Divisão:** 75% treino e 25% teste, estratificada por rótulo.
5. **Mineração:** TF-IDF (Scikit-Learn) com unigramas, bigramas e trigramas (`min_df=3`, `max_df=0.9`, `sublinear_tf=True`).
6. **Modelagem:** SVM linear (`LinearSVC`), avaliado com acurácia, relatório de classificação e matriz de confusão.
7. **Representação:** nuvens de palavras (WordCloud) para REAL e FAKE, com o tamanho de cada termo proporcional ao seu peso TF-IDF médio no rótulo.

## Como executar
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Abra a pasta no VS Code, mantenha `pre-processed.csv` na mesma pasta do notebook, selecione o kernel `.venv` e execute todas as células (*Run All*). O notebook baixa automaticamente as stopwords do NLTK.

## Saídas geradas
- Acurácia no conjunto de teste, relatório de classificação e matriz de confusão.
- Contagem de palavras, bigramas e trigramas usados em cada rótulo.
- `nuvem_real_4957311.png` e `nuvem_fake_4957311.png`.

## Observações
- Acurácias muito acima de 93% neste corpus costumam indicar que o tamanho dos textos não foi normalizado; por isso o balanceamento é uma etapa obrigatória.
- Os resultados podem variar levemente conforme a semente aleatória e a versão das bibliotecas (semente usada: 42).

