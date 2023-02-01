from typing import Any


def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)


def detokenize(tokens):
    ret = ''
    for g, a in zip(tokens['gloss'], tokens['after']):
        ret += g + a
    return ret.strip()


class EntityToId(dict):
    # only works for bijective mapping
    # note: older pythn version dict are not ordered
    def __init__(self, *args, **kwargs):
        super(EntityToId, self).__init__(*args, **kwargs)
        # helper to froze the mapping, usefull for processing dev and test dataset
        self._frozen = False
        self.inverse = {}
        for key, value in self.items():
            self.inverse[value] = key

    @property
    def is_frozen(self):
        return self._frozen

    def froze(self):
        self._frozen = True
        return self

    def unfroze(self):
        self._frozen = False
        return self

    def __setitem__(self, key, value):
        super(EntityToId, self).__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        if self[key] in self.inverse:
            del self.inverse[self[key]]
        super(EntityToId, self).__delitem__(key)

    def add_entity(
        self,
        key: Any,
    ):
        if (key not in self) and (not self._frozen):
            value = len(self)
            super(EntityToId, self).__setitem__(key, value)
            self.inverse[value] = key
