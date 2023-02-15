class ORNode:
    def __init__(self,
                 scope,
                 id=None,
                 children=None,
                 weights=None,
                 ess=None,
                 row_indices=None,
                 col_indices=None,
                 depth=0,
                 clt=None,
                 clt_score=0.0,
                 flag=None):
        self.scope = scope
        self.id = id
        self.ess = ess
        self.children = children
        self.weights = weights
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.depth = depth
        self.clt = clt
        self.clt_score = clt_score
        self.flag = flag

    def is_leaf(self):
        return True if self.clt else False
    