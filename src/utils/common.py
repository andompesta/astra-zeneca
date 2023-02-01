from typing import Any

SELECT = "SELECT"
AND = "AND"
WHERE = "WHERE"

PAD = "PAD"
SOS = "SOS"
EOS = "EOS"
OOV = "OOV"


def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)


def detokenize(tokens):
    ret = ''
    for g, a in zip(tokens['gloss'], tokens['after']):
        ret += g + a
    return ret.strip()


class EntityToId(object):
    # only works for bijective mapping
    # note: older pythn version dict are not ordered
    def __init__(self, *args, **kwargs):
        super(EntityToId, self).__init__(*args, **kwargs)
        # helper to froze the mapping, usefull for processing dev and test dataset
        self._frozen = False
        self.data = dict()
        self.inverse = dict()

    @property
    def is_frozen(self):
        return self._frozen

    def froze(self):
        self._frozen = True
        return self

    def unfroze(self):
        self._frozen = False
        return self

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default):
        return self.data.get(key, default)

    def __setitem__(self, key, value):
        self.data[key] = value
        self.inverse[value] = key

    def __delitem__(self, key):
        self.inverse.__delitem__[self[key]]
        self.data.__delitem__(key)

    def add_entity(
        self,
        key: Any,
    ):
        if (key not in self.data) and (not self._frozen):
            value = len(self.data)
            self.data[key] = value
            self.inverse[value] = key
