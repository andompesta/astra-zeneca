from collections import namedtuple
import re
from src.utils.table import Table

Condition = namedtuple("Condition", [
    "column",
    "operator",
    "condition",
])

re_whitespace = re.compile(r'\s+', flags=re.UNICODE)


class Query:

    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = [
        'SELECT',
        'WHERE',
        'AND',
        'COL',
        'TABLE',
        'CAPTION',
        'PAGE',
        'SECTION',
        'OP',
        'COND',
        'QUESTION',
        'AGG',
        'AGGOPS',
        'CONDOPS',
    ]

    def __init__(
            self,
            table_id: str,
            sel_op: str,
            agg_op: str,
            conditions=list[Condition],
            ordered=False,
    ):
        self.sel_op = sel_op
        self.agg_op = agg_op
        self.conditions = conditions
        self.ordered = ordered
        self.table_id = table_id

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            indices = self.sel_op == other.sel_op and self.agg_op == other.agg_op
            if other.ordered:
                conds = self.conditions == other.conditions
            else:
                conds = set(self.conditions) == set(other.conditions)

            return indices and conds
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self):
        rep = 'SELECT {agg} {sel} FROM {table_id}'.format(
            sel=self.sel_op,
            agg=self.agg_op,
            table_id=self.table_id,
        )
        if self.conditions:
            rep += ' WHERE ' + ' AND '.join([
                '{} {} {}'.format(col, op, val)
                for col, op, val in self.conditions
            ])
        return re_whitespace.sub(" ", rep)

    @classmethod
    def from_dict(cls, d, table: Table, ordered=False):
        assert table.table_id == d["table_id"], "wrong table used to parse the query\n{}\n{}".format(
            d,
            table,
        )

        conditions = [
            Condition(
                # fails if column_index is overflow
                table.header[condition[0]],
                # fails if condition is unknown
                Query.cond_ops[condition[1]],
                # parse all conditions to string as need to be tokenized
                str(condition[2]).lower(),
            # return empy list if no conditions are provided
            ) for condition in d.get('conds', [])
        ]

        return cls(
            table_id=table.table_id,
            sel_op=table.header[d['sel']],
            agg_op=Query.agg_ops[d['agg']],
            conditions=conditions,
            ordered=ordered,
        )
