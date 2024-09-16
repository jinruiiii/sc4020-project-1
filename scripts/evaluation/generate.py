"""
File to be run to generate the search results, and time taken for each query.
"""
import sys
from typing import List, Callable
import numpy as np


sys.path.append("../..")
from utils.evaultation.generator import Generator
from algo.algo_interface import IAlgo
from algo.LSHAlgo import LSH
from algo.HNSWAlgo import HNSW
from algo.VSMAlgo import VSM


def main():
    # Lazily instantiated list of runners
    runners: List[Callable[[], IAlgo]] = []

    print("Adding runners to list")

    runners.append(
        lambda: VSM("starbucks", "bge")
    )

    m_vals = [2**i for i in range(4, 10)]
    con_vals = [2**i for i in range(10)]
    search_vals = [2**i for i in range(10)]

    for m in m_vals:
        for c in con_vals:
            for s in search_vals:
                runners.append(
                    lambda m_=m, c_=c, s_=s: HNSW("starbucks", "bge", m_, c_, s_)
                )

    for n_pwr in range(3, 10):
        runners.append(lambda pwr=n_pwr: LSH("starbucks", "bge", 2**pwr))

    runners.append(lambda pwr=n_pwr: LSH("starbucks", "bge", 768))

    print(f"Starting generator with {len(runners)} runners!")

    # Execute each runner
    Generator([
        "Have been frequenting Starbucks for 30 years. Have liked most of their drinks. Their new iced apple macchiato is something I would not order again. Normally I love apple anything but disliked the flavor of this and didn’t finish… very disappointing.",
        "Second time in a week I've gone to Starbucks and ordered pistachio grande and they didn't put the pistachio pumps in. We get it and leave with a terrible coffee for a very high price. And they expect us to tip as well! I'm not going back if I can't treat myself to what I've ordered. BTW this was on Frederica St..",
        "I visited the Starbucks, Greensboro, NC  Sept 27, 2008, about 2:40 pm. and ordered a Grande Chai Latte with Skim. I can't find the price online, but it was something around $3.60.  I gave the cashier a $20 bill.  Being stupid, I did not count the change, or looked at the bills, except for the $10.  I  dropped the coins in the tip jar, and put the bills in my pocket.",
        "This time this girl named KP was taking my order and when I ask her about discount she was so rude about it and she is like \"There is no button on screen that I can give you discount.\" I told her \"We always get discount\" but she was like \"There is no discount. I never saw you before.\" and she was very rude when she was talking to me.",
        "DEMANDED TIPS FROM ME, THEN MADE ME WAIT UNTIL MY COFFEES WERE COLD AND MELTED ( I HAD 20 OF THEM). They have yet to address the problem, the store still demands tips!"
    ], top_k=10+1).run(runners)


if __name__ == "__main__":
    main()
