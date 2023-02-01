from tabulate import tabulate
from typing import Any

class Table:

    def __init__(
        self,
        table_id: str,
        header: list[str],
        types: list[str],
        rows: list[list[Any]],
        *args,
        **kwargs,
    ):
        self.table_id = table_id
        self.header = header
        self.types = types
        self.rows = rows
        self.caption = kwargs.get("caption", None)

    def __repr__(self):
        return 'Table: {id}\nCaption: {caption}\n{tabulate}'.format(
            id=self.table_id,
            caption=self.caption,
            tabulate=tabulate(self.rows, headers=self.header)
        )
