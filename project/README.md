# Итоговый проект. Классификация отзывов IMDB

Все эксперименты проводятся на [датасете IMDB](http://ai.stanford.edu/~amaas/data/sentiment/index.html).

+ В папке CNN лежит исходный код реализации свёрточной нейросети из статьи [Rie Johnson, Tong Zhang. Effective Use of Word Order for Text Categorization with Convolutional Neural Networks](https://paperswithcode.com/paper/effective-use-of-word-order-for-text-1).
Рассмотренная CNN модель состоит из следующих слоёв: Embedding + 2x(Conv1D + GlobalMaxPooling1D) + Concat + Dense.

+ В biLSTM.ipynb реализован похожий код из статьи [Yue Zhang, Qi Liu and Linfeng Song. Sentence-State LSTM for Text Representation](https://arxiv.org/pdf/1805.02474v1.pdf). 
Наша LSTM состоит из 1 слоя эмбеддинга + 2 слоев BiLSTM (размерности 64) + выходной слой. 

+ В cnn_lstm.py находится похожий скрипт из статьи [Jose Camacho-Collados, Mohammad Taher Pilehvar. On the Role of Text Preprocessing in Neural Network Architectures: An Evaluation Study on Text Categorization and Sentiment Analysis](https://arxiv.org/pdf/1707.01780v3.pdf). 
CNN + LSTM состоит из следующих слоёв: Embedding + Dropout(0.25) + Conv1D + MaxPooling + LSTM + Dense. Рассмотренная LSTM модель состоит из следующих слоёв: Embedding + 2xBiLSTM + Dense.


Выполнили: Екатерина Махлина, Алина Смирнова, Анна Щербакова.
