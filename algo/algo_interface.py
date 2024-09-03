from abc import abstractmethod
import pandas as pd


class IAlgo:

    @abstractmethod
    def run(self, query:str, k:int) -> pd.DataFrame:
        """
        Method to run to search.
        :param query: String to be used in the search
        :param k: Number of records to return
        :return:
        """
        raise NotImplementedError
