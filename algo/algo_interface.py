from abc import abstractmethod
from typing import Dict, Union, Literal

import pandas as pd

import sys
sys.path.append("..")

from utils.evaultation.algo_types import AlgoType


class IAlgo:

    @abstractmethod
    def run(self, query:str, k:int) -> pd.DataFrame:
        """
        Method to run to search. This method will be executed by a runner class, where the time and results
        will be recorded.
        :param query: String to be used in the search
        :param k: Number of records to return
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> AlgoType:
        """
        Method to return the name of the algorithm class
        :return: str, name of the algo class
        """
        raise NotImplementedError

    @abstractmethod
    def data_source(self) -> Literal["starbucks"]:
        """
        Method to return the name of the data source (e.g. starbucks)
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def details(self) -> Dict[str, Union[str, int]]:
        """
        Method to return a dictionary that details the hyperparameters of the class.
        This will perform similarly to __repr__, but as a purposefully set method.
        :return: dictionary containing kwargs of hyperparameters
        """
        raise NotImplementedError
