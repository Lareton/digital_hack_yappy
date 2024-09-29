import pickle
from sklearn.neighbors import KDTree
import numpy as np

#from qdrant_client import QdrantClient
#from qdrant_client.http import models
import numpy as np


class VectorDatabaseQdrant:
    """
    Класс для управления векторной базой данных и поиска ближайших соседей с использованием Qdrant.

    Этот класс взаимодействует с векторной базой данных Qdrant для хранения и поиска ближайших соседей
    по евклидовой или косинусной метрике.
    """

    def __init__(self, host: str, port: int, collection_name: str, metric: str = 'cosine'):
        """
        Инициализирует объект класса `VectorDatabaseQdrant` и подключается к базе данных Qdrant.

        Параметры:
        ----------
        host : str
            Адрес сервера Qdrant (например, "localhost").
        port : int
            Порт, на котором работает Qdrant (например, 6333).
        collection_name : str
            Имя коллекции в Qdrant, которая будет использоваться для поиска.
        metric : str, по умолчанию "cosine"
            Метрика для поиска (может быть "cosine" или "euclidean").

        """
        # Инициализация клиента Qdrant
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.metric = metric

        # Проверка наличия коллекции
        if not self.client.get_collection(self.collection_name):
            raise ValueError(f"Коллекция {self.collection_name} не найдена в Qdrant.")

    def query(self, emb: np.ndarray, k: int = 5) -> list:
        """
        Выполняет поиск ближайших соседей для заданного вектора признаков.

        Параметры:
        ----------
        emb : np.ndarray
            Вектор признаков, по которому выполняется поиск ближайших соседей.
        k : int, по умолчанию 5
            Количество ближайших соседей, которые нужно найти.

        Возвращает:
        ----------
        list : List[dict]
            Список ближайших соседей с информацией о точках (id и расстояние).

        Пример использования:
        ---------------------
        >>> qdrant_db = VectorDatabaseQdrant("localhost", 6333, "video_embeddings")
        >>> query_emb = np.array([0.1, 0.2, 0.3, 0.4])
        >>> qdrant_db.query(query_emb, k=3)
        [{'id': 1, 'distance': 0.12}, {'id': 2, 'distance': 0.15}, ...]
        """
        # Преобразование вектора признаков в список
        emb = emb.tolist()

        # Выполнение поиска ближайших соседей в Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=emb,
            limit=k,
            with_payload=False
        )

        # Формирование списка ближайших соседей
        return [{'id': result.id, 'distance': result.score} for result in search_result]


class VectorDatabase:
    """
    Класс для управления векторной базой данных и поиска ближайших соседей с использованием KDTree.

    Описание:
    --------
    Этот класс предназначен для хранения и обработки векторов признаков видео и предоставляет методы
    для поиска ближайших соседей по евклидовой метрике с помощью структуры данных KDTree.

    Атрибуты:
    ---------
    full_index : list
        Полный индекс, загруженный из файла, представляющий собой список кортежей (идентификатор, вектор признаков).
    indexes : list
        Список идентификаторов видео из индекса.
    embs : list
        Список векторов признаков, преобразованных в одномерный массив.
    tree : KDTree
        Структура данных KDTree для быстрого поиска ближайших соседей по евклидовой метрике.

    """

    def __init__(self, path_to_index: str):
        """
        Инициализирует объект класса `VectorDatabase` и загружает векторный индекс из файла.

        Параметры:
        ----------
        path_to_index : str
            Путь к файлу с сериализованным индексом (формат .pkl), содержащим список кортежей (идентификатор, вектор).

        Описание:
        --------
        1. Загружает векторный индекс из файла и сохраняет его в атрибут `full_index`.
        2. Извлекает список идентификаторов (`indexes`) и векторов признаков (`embs`) из полного индекса.
        3. Преобразует все векторы в одномерные массивы и строит дерево KDTree для поиска ближайших соседей.
        """
        self.full_index = None
        with open(path_to_index, "rb") as f:
            self.full_index = pickle.load(f)

        # Извлечение идентификаторов и признаков из индекса
        self.indexes = [i[0] for i in self.full_index]
        self.embs = [i[1].reshape(-1) for i in self.full_index]

        # Построение дерева KDTree для поиска ближайших соседей
        self.tree = KDTree(np.array(self.embs), metric="euclidean")

    def query(self, emb: np.ndarray, k: int = 5) -> list:
        """
        Выполняет поиск ближайших соседей для заданного вектора признаков.

        Параметры:
        ----------
        emb : np.ndarray
            Вектор признаков, по которому выполняется поиск ближайших соседей.
        k : int, по умолчанию 5
            Количество ближайших соседей, которые нужно найти.

        Возвращает:
        ----------
        list : List[tuple]
            Список кортежей (индекс, вектор), представляющий ближайшие соседние векторы.
            - индекс: идентификатор ближайшего соседа в базе данных.
            - вектор: соответствующий вектор признаков из базы данных.

        Пример использования:
        ---------------------
        >>> db = VectorDatabase("full_index2.pkl")
        >>> query_emb = np.array([0.1, 0.2, 0.3, 0.4])
        >>> db.query(query_emb, k=3)
        [(0, array([0.1, 0.2, 0.3, 0.4])), (1, array([0.2, 0.3, 0.4, 0.5])), ...]
        """
        # Преобразование вектора признаков в одномерный массив
        emb = emb.reshape(-1)

        # Поиск ближайших соседей
        dist, inds = self.tree.query([emb], k=k)

        # Извлечение ближайших соседей по индексам
        inds = list(inds[0])
        best_embs = [self.embs[i] for i in list(inds)]

        # Возвращение списка кортежей (индекс, вектор)
        return list(zip(inds, best_embs))

