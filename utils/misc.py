import pandas as pd


class Colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    def make_bold(self, text: str):
        return self.__apply_color(text, self.BOLD)

    def underline(self, text: str):
        return self.__apply_color(text, self.UNDERLINE)

    def __apply_color(self, text: str, color: str):
        return f"{color}{text}{self.END}"


class AlwaysIn:
    def __contains__(self, item):
        return True


def nested_dict_to_pandas(d):
    new_d = {}
    for outer, inner in d.items():
        for i_key, vals in inner.items():
            new_d[(outer, i_key)] = vals
    return pd.DataFrame(new_d)
